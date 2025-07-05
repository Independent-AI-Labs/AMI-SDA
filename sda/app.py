# app.py

"""
The main application entry point and the central CodeAnalysisFramework class.

This module implements the Facade design pattern by providing a single,
simplified interface to the complex subsystems like ingestion, analysis, and Git
integration. The Gradio UI and other external clients will interact with an
instance of the CodeAnalysisFramework class.
"""

import logging
import json # For parsing Dgraph metrics response
import queue
import requests # Added for Dgraph metrics
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Generator

from PIL import Image
from llama_index.core.llms import ChatMessage
from sqlalchemy import func
from sqlalchemy.orm import joinedload

from sda.config import DB_URL, WORKSPACE_DIR, AIConfig, GOOGLE_API_KEY, DGRAPH_HOST # Added DGRAPH_HOST
from sda.core.db_management import DatabaseManager
from sda.core.models import Repository, File as DBFile, DBCodeChunk, Task, BillingUsage
from sda.services.agent import AgentManager
from sda.services.analysis import EnhancedAnalysisEngine
from sda.services.chunking import TokenAwareChunker
from sda.services.editing import SafeFileEditingSystem
from sda.services.git_integration import GitService
from sda.services.ingestion import IntelligentIngestionService
from sda.services.navigation import AdvancedCodeNavigationTools
from sda.services.partitioning import SmartPartitioningService
from sda.services.search import FullTextSearchService
from sda.utils.limiter import RateLimiter
from sda.utils.task_executor import TaskExecutor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class CodeAnalysisFramework:
    """A unified facade for the entire code analysis system."""

    def __init__(self, db_url: str = DB_URL, workspace_dir: str = WORKSPACE_DIR):
        """Initializes all underlying services and managers."""
        logging.info("Initializing the Code Analysis Framework...")

        google_api_keys = [key.strip() for key in GOOGLE_API_KEY.split(',')] if GOOGLE_API_KEY else []

        # Initialize the centralized TaskExecutor
        self.task_executor = TaskExecutor()
        
        # Initialize RateLimiter with the structured Pydantic models.
        self.rate_limiter = RateLimiter(
            model_configs=AIConfig.get_all_llm_configs(),
            api_keys=google_api_keys
        )
        self.db_manager = DatabaseManager(db_url=db_url, is_worker=False)
        self.chunker = TokenAwareChunker()
        self.git_service = GitService(workspace_dir=workspace_dir)
        self.full_text_search_service = FullTextSearchService()
        self.analysis_engine = EnhancedAnalysisEngine(db_manager=self.db_manager, rate_limiter=self.rate_limiter)
        
        # Instantiate and pass the new partitioning service
        self.partitioning_service = SmartPartitioningService()
        self.ingestion_service = IntelligentIngestionService(
            db_manager=self.db_manager,
            git_service=self.git_service,
            task_executor=self.task_executor,
            partitioning_service=self.partitioning_service
        )
        self.navigation_tools = AdvancedCodeNavigationTools(db_manager=self.db_manager)
        self.editing_system = SafeFileEditingSystem(db_manager=self.db_manager, chunker=self.chunker)
        self.agent_manager = AgentManager(framework=self, rate_limiter=self.rate_limiter)
        logging.info("Code Analysis Framework initialized successfully.")

    # --- Task Management ---

    def _start_task(self, repo_id: int, name: str, created_by: str, parent_id: Optional[int] = None) -> Task:
        """Creates a new Task record in the database, representing a running job."""
        with self.db_manager.get_session("public") as session:
            now = datetime.utcnow()
            task = Task(repository_id=repo_id, name=name, created_by=created_by,
                        status='running', message='Task started...', parent_id=parent_id,
                        details={},
                        log_history=f"{now.isoformat()}: Task '{name}' initiated by {created_by}.\n")
            session.add(task)
            session.flush()
            session.refresh(task)
            session.expunge(task)
            return task

    def _update_task(self, task_id: int, message: str, progress: float,
                     log_message: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        """Updates the status of an existing Task, merging new details into the existing JSON."""
        with self.db_manager.get_session("public") as session:
            task = session.get(Task, task_id)
            if task:
                task.message = message
                task.progress = progress
                if log_message:
                    now = datetime.utcnow()
                    task.log_history = (task.log_history or "") + f"{now.isoformat()}: {log_message}\n"
                if details:
                    # Merge new details into existing details JSON
                    # This requires SQLAlchemy to be configured to detect in-place mutations.
                    if task.details:
                        task.details.update(details)
                    else:
                        task.details = details
                    # Manually flag the JSON column as modified to ensure the change is persisted.
                    from sqlalchemy.orm.attributes import flag_modified
                    flag_modified(task, "details")
            else:
                logging.warning(f"Attempted to update a non-existent task with ID: {task_id}")


    def _complete_task(self, task_id: int, result: Any = None, error: Optional[str] = None):
        """Marks a task as completed or failed."""
        with self.db_manager.get_session("public") as session:
            task = session.get(Task, task_id)
            if task:
                now = datetime.utcnow()
                task.completed_at = now
                task.progress = 100.0
                if error:
                    task.status = 'failed'
                    task.error_message = error
                    task.message = 'Task failed.'
                    task.log_history = (task.log_history or "") + f"{now.isoformat()}: Task failed. See error details.\n"
                else:
                    task.status = 'completed'
                    task.result = result
                    task.message = 'Task completed successfully.'
                    task.log_history = (task.log_history or "") + f"{now.isoformat()}: Task completed successfully.\n"
            else:
                logging.warning(f"Attempted to complete a non-existent task with ID: {task_id}")

    def get_latest_task(self, repo_id: int) -> Optional[Task]:
        """Retrieves the most recent parent task for a repository, including its children."""
        if not repo_id: return None
        with self.db_manager.get_session("public") as session:
            task = session.query(Task).options(
                joinedload(Task.children)
            ).filter(
                Task.repository_id == repo_id, Task.parent_id.is_(None)
            ).order_by(
                Task.started_at.desc()
            ).first()

            if task:
                session.expunge_all()
            return task

    # --- Agent and Chat ---

    def run_agent_chat_stream(self, query: str, repo_id: int, branch: str, chat_history: List[ChatMessage]) -> Generator[str, None, None]:
        """Runs a chat query through the agent and streams back responses."""
        message_queue = queue.Queue()

        def stream_callback(message: str):
            message_queue.put(message)

        def agent_thread_target():
            try:
                final_answer = ""
                for chunk in self.agent_manager.run_chat_stream(repo_id, branch, query, chat_history, stream_callback):
                    final_answer = chunk
                message_queue.put(None)
            except Exception as e:
                logging.error(f"Agent thread failed: {e}", exc_info=True)
                error_message = f"An error occurred in the agent: {e}"
                message_queue.put(error_message)
                message_queue.put(None)

        thread = threading.Thread(target=agent_thread_target)
        thread.start()

        while True:
            message = message_queue.get()
            if message is None:
                break
            yield message
            
        thread.join()

    # --- Repository and File Operations ---

    def add_repository(self, repo_identifier: str) -> Optional[Repository]:
        """Adds a new repository by cloning or registering a local path."""
        logging.info(f"Adding repository from identifier: {repo_identifier}")
        local_path_str = self.git_service.clone_or_pull(repo_identifier)
        if not local_path_str:
            logging.error(f"Failed to clone or find repository at {repo_identifier}")
            return None

        with self.db_manager.get_session("public") as session:
            repo = session.query(Repository).filter_by(path=local_path_str).first()
            if not repo:
                path_obj = Path(local_path_str)
                name = path_obj.name
                git_remote = repo_identifier if repo_identifier.startswith(('http', 'git@')) else None
                current_branch = self.git_service.get_current_branch(local_path_str)
                repo = Repository(path=local_path_str, name=name, git_remote=git_remote, active_branch=current_branch, db_schemas=[])
                session.add(repo)
                session.flush()
                session.refresh(repo)

            session.expunge(repo)
            return repo

    def get_repository_by_id(self, repo_id: int) -> Optional[Repository]:
        """Retrieves a repository by its primary key."""
        with self.db_manager.get_session("public") as session:
            repo = session.get(Repository, repo_id)
            if repo:
                session.expunge(repo)
            return repo

    def get_all_repositories(self) -> List[Repository]:
        """Returns a list of all configured repositories."""
        with self.db_manager.get_session("public") as session:
            repos = session.query(Repository).order_by(Repository.name).all()
            session.expunge_all()
            return repos

    def get_repository_branches(self, repo_id: int) -> List[str]:
        """Lists all local branches for a given repository."""
        repo = self.get_repository_by_id(repo_id)
        if not repo: return []
        return self.git_service.list_branches(repo.path)

    def list_files_in_repo(self, repo_id: int, branch: str, path_prefix: Optional[str] = None) -> List[str]:
        """
        Lists files and directories for a specific repo, branch, and optional path prefix.
        If no prefix is given, lists top-level contents. Directories are indicated by a trailing '/'.
        To explore the project structure, start with no prefix and then use the returned
        directory paths in subsequent calls.
        """
        repo = self.get_repository_by_id(repo_id)
        if not repo or not repo.db_schemas:
            return ["Error: Repository not found or not analyzed."]

        all_paths_from_db = set()
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self._list_files_in_schema_unfiltered, schema, repo_id, branch) for schema in repo.db_schemas]
            for future in as_completed(futures):
                try:
                    all_paths_from_db.update(future.result())
                except Exception as e:
                    logging.error(f"Failed to list files from a schema: {e}")

        if not all_paths_from_db:
            return ["No files found in the database for this branch."]

        # Sanitize and normalize the path prefix
        prefix = ''
        if path_prefix:
            prefix = Path(path_prefix).as_posix().strip()
            if prefix and prefix != '.' and not prefix.endswith('/'):
                prefix += '/'
            if prefix == './':
                prefix = ''

        if prefix and prefix.rstrip('/') in all_paths_from_db:
            return [f"Error: '{prefix.rstrip('/')}' is a file, not a directory."]

        entries = set()
        for p_str in all_paths_from_db:
            if not p_str.startswith(prefix):
                continue
            suffix = p_str[len(prefix):]
            if not suffix: continue
            first_sep_idx = suffix.find('/')
            if first_sep_idx != -1:
                child_name = suffix[:first_sep_idx + 1]
                entries.add(prefix + child_name)
            else:
                entries.add(prefix + suffix)

        if not entries and prefix:
            if any(p.startswith(prefix) for p in all_paths_from_db):
                return ["(Directory is empty)"]
            else:
                return [f"Error: Path '{prefix.rstrip('/')}' not found."]
        return sorted(list(entries))

    def _list_files_in_schema_unfiltered(self, schema: str, repo_id: int, branch: str) -> List[str]:
        """Helper to list all files from a single repository schema."""
        with self.db_manager.get_session(schema) as session:
            files = session.query(DBFile.relative_path).filter_by(repository_id=repo_id, branch=branch).all()
            return [f[0] for f in files]

    def get_file_content_by_path(self, repo_id: int, branch: str, relative_path: str) -> Optional[str]:
        """Retrieves the content of a file from disk, verifying it's tracked in the DB."""
        repo = self.get_repository_by_id(repo_id)
        if not repo or not repo.db_schemas:
            return f"Error: Repository {repo_id} not found or has no analyzed data."

        file_record = None
        for schema in repo.db_schemas:
            with self.db_manager.get_session(schema) as session:
                record = session.query(DBFile).filter_by(repository_id=repo_id, branch=branch, relative_path=relative_path).first()
                if record:
                    file_record = record
                    break
        if file_record:
            file_path = Path(file_record.file_path)
            if file_path.exists():
                try:
                    return file_path.read_text(encoding='utf-8')
                except Exception as e:
                    return f"Error reading file: {e}"
            else:
                return f"Error: File path not found on disk at {file_path}."
        return f"Error: File '{relative_path}' not found in the database for the current branch."

    def search_files_for_text(self, repo_id: int, queries: List[str]) -> List[str]:
        """Performs a fast, case-insensitive, multi-keyword search across all text files."""
        repo = self.get_repository_by_id(repo_id)
        if not repo: return ["Error: Repository not found."]
        return self.full_text_search_service.search(repo.path, [q.strip() for q in queries])

    # --- Analysis & Ingestion Triggers ---

    def _run_ingestion_in_background(self, task_id: int, repo_id: int, repo_path: str, repo_uuid: str, branch: str):
        """Worker target for running the full ingestion pipeline in a separate thread."""
        try:
            self.ingestion_service.ingest_repository(
                repo_path_str=repo_path, repo_uuid=repo_uuid, branch=branch, repo_id=repo_id,
                parent_task_id=task_id, _framework_start_task=self._start_task,
                _framework_update_task=self._update_task, _framework_complete_task=self._complete_task
            )
        except Exception as e:
            log_msg = f"Ingestion task runner failed: {e}\n{traceback.format_exc()}"
            logging.error(log_msg)
            self._update_task(task_id, "Ingestion failed unexpectedly.", 100.0, log_msg)
            self._complete_task(task_id, error=traceback.format_exc())

    def analyze_branch(self, repo_id: int, branch: str, created_by: str = 'user') -> Task:
        """Starts a background task to run the full ingestion and analysis for a branch."""
        repo = self.get_repository_by_id(repo_id)
        if not repo:
            raise ValueError(f"Repository with ID {repo_id} not found.")

        task = self._start_task(repo_id, f"Ingest Branch: {branch}", created_by)
        thread = threading.Thread(
            target=self._run_ingestion_in_background,
            args=(task.id, repo.id, repo.path, repo.uuid, branch),
            name=f"IngestionWorker-Repo{repo.id}-{branch}"
        )
        thread.start()
        return task

    def get_repository_stats(self, repo_id: int, branch: str) -> Dict[str, Any]:
        """Aggregates and returns statistics for a repository branch across all its schemas."""
        repo = self.get_repository_by_id(repo_id)
        if not repo or not repo.db_schemas:
            return {}

        def _get_stats_from_schema(schema: str) -> Dict[str, Any]:
            stats = {"file_count": 0, "total_lines": 0, "total_tokens": 0, "language_breakdown": {}}
            with self.db_manager.get_session(schema) as session:
                file_stats = session.query(func.count(DBFile.id), func.sum(DBFile.line_count)).filter(
                    DBFile.repository_id == repo_id, DBFile.branch == branch
                ).first()
                if file_stats:
                    stats["file_count"] = file_stats[0] or 0
                    stats["total_lines"] = file_stats[1] or 0
                
                token_sum = session.query(func.sum(DBCodeChunk.token_count)).filter(
                    DBCodeChunk.repository_id == repo_id, DBCodeChunk.branch == branch
                ).scalar()
                stats["total_tokens"] = token_sum or 0
                
                lang_breakdown = session.query(DBFile.language, func.count(DBFile.id)).filter(
                    DBFile.repository_id == repo_id, DBFile.branch == branch, DBFile.language.isnot(None)
                ).group_by(DBFile.language).all()
                stats["language_breakdown"] = {lang: count for lang, count in lang_breakdown}
            return stats

        total_stats = {"file_count": 0, "total_lines": 0, "total_tokens": 0, "language_breakdown": {}}
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(_get_stats_from_schema, schema): schema for schema in repo.db_schemas}
            for future in as_completed(futures):
                try:
                    schema_stats = future.result()
                    total_stats["file_count"] += schema_stats["file_count"]
                    total_stats["total_lines"] += schema_stats["total_lines"]
                    total_stats["total_tokens"] += schema_stats["total_tokens"]
                    for lang, count in schema_stats["language_breakdown"].items():
                        total_stats["language_breakdown"][lang] = total_stats["language_breakdown"].get(lang, 0) + count
                except Exception as e:
                    logging.error(f"Failed to get stats from schema {futures[future]}: {e}")

        total_stats["schema_count"] = len(repo.db_schemas) if repo.db_schemas else 0
        return total_stats

    def get_cpg_analysis(self, repo_id: int, branch: str) -> Dict[str, Any]:
        """Retrieves an analysis of the Code Property Graph from Dgraph."""
        query = """
        query CentralNodes($repo_id: string, $branch: string) {
          var(func: type(ASTNode)) @filter(eq(repo_id, $repo_id) and eq(branch, $branch)) {
            incoming_call_count as count(~calls)
          }
          central_nodes(func: uid(incoming_call_count), orderdesc: val(incoming_call_count), first: 5) {
            name
            file_path
            incoming_calls: val(incoming_call_count)
          }
        }
        """
        variables = {"$repo_id": str(repo_id), "$branch": branch}
        response = self.db_manager.query_dgraph(query, variables)

        analysis = {"node_count": 0, "edge_count": "N/A", "central_nodes": []}
        if not response or 'central_nodes' not in response:
            analysis["error"] = "Could not retrieve graph analysis from Dgraph."
            return analysis

        analysis['central_nodes'] = [{"name": n.get('name'), "file_path": n.get('file_path'), "incoming_calls": n.get('incoming_calls')} for n in response['central_nodes']]

        count_query = """
            query NodeCount($repo_id: string, $branch: string) {
                var(func: type(ASTNode)) @filter(eq(repo_id, $repo_id) and eq(branch, $branch)) { C as count(uid) }
                q() { count: sum(val(C)) }
            }
        """
        count_response = self.db_manager.query_dgraph(count_query, variables)
        if count_response and 'q' in count_response and count_response['q']:
            analysis['node_count'] = count_response['q'][0].get('count', 0)

        return analysis

    def _run_find_dead_code_in_background(self, task_id: int, repo_id: int, branch: str):
        """Worker target for finding dead code."""
        try:
            self._update_task(task_id, "Searching Dgraph for unused code...", 10.0)
            result = self.navigation_tools.find_dead_code(repo_id, branch)
            self._complete_task(task_id, result={"dead_code": [res.to_dict() for res in result]})
        except Exception as e:
            self._complete_task(task_id, error=traceback.format_exc())

    def find_dead_code_for_repo(self, repo_id: int, branch: str, created_by: str = 'user') -> Task:
        """Starts a background task to find potentially unused code."""
        task = self._start_task(repo_id, "find_dead_code", created_by)
        thread = threading.Thread(target=self._run_find_dead_code_in_background, args=(task.id, repo_id, branch))
        thread.start()
        return task

    def _run_find_duplicate_code_in_background(self, task_id: int, repo_id: int, branch: str):
        """Worker target for finding duplicate code."""
        try:
            self._update_task(task_id, "Finding duplicate code...", 10.0)
            result = self.analysis_engine.find_duplicate_code(repo_id, branch)
            self._complete_task(task_id, result={"duplicate_code": [res.__dict__ for res in result]})
        except Exception as e:
            self._complete_task(task_id, error=traceback.format_exc())

    def find_duplicate_code_for_repo(self, repo_id: int, branch: str, created_by: str = 'user') -> Task:
        """Starts a background task to find semantically duplicate code."""
        task = self._start_task(repo_id, "find_duplicate_code", created_by)
        thread = threading.Thread(target=self._run_find_duplicate_code_in_background, args=(task.id, repo_id, branch))
        thread.start()
        return task

    # --- Git & Version Control ---

    def get_repository_status(self, repo_id: int) -> Optional[Dict[str, List[str]]]:
        """Gets the git status for the repository's working directory."""
        repo = self.get_repository_by_id(repo_id)
        if not repo: return None
        return self.git_service.get_status(repo.path)

    def get_file_diff_or_content(self, repo_id: int, file_path: str, is_new_file_from_explorer: bool = False) -> tuple[Optional[str], Optional[Image.Image]]:
        """
        Gets the git diff for a modified file, or raw content for an image or a new file from explorer.
        If is_new_file_from_explorer is True, it will try to read the file content directly
        instead of getting a git diff, unless it's an image.
        """
        repo = self.get_repository_by_id(repo_id)
        if not repo: return None, None

        full_path = Path(repo.path) / file_path
        if full_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp']:
            try:
                img = Image.open(full_path); img.load()
                return None, img
            except Exception as e:
                return f"Error opening image: {e}", None
        else:
            if is_new_file_from_explorer:
                # If called from file explorer for a non-image, get raw content
                try:
                    return full_path.read_text(encoding='utf-8'), None
                except Exception as e:
                    return f"Error reading file: {e}", None
            else:
                # Original behavior: get git diff for (presumably) modified files
                return self.git_service.get_diff(repo.path, file_path), None

    def get_file_tree(self, repo_id: int, branch: str) -> List[Tuple[str, str]]:
        """
        Returns a list of (value, label) tuples for files in the repository,
        suitable for gr.Tree. Value and label are typically the relative file path.
        This is a placeholder. A real implementation would scan the repo directory
        or query a database of indexed files for the given branch.
        """
        # Placeholder implementation - mirrors the dummy data in UI for now
        # In a real scenario, this would involve:
        # 1. Checking out the specified branch (if not already active).
        # 2. Walking the file system of the repo path.
        # 3. Filtering out .git directory, and potentially other ignored files (.gitignore).
        # 4. Formatting into List[Tuple[str, str]] where paths are relative to repo root.
        # Example: [("README.md", "README.md"), ("src/app.py", "src/app.py")]
        repo = self.get_repository_by_id(repo_id)
        if not repo:
            return []

        # This is a very basic placeholder.
        # A more robust version would use self.git_service.list_files_in_tree(repo.path, branch)
        # or walk the filesystem after ensuring the correct branch is checked out.
        # For now, returning a fixed list similar to UI's dummy data.
        # This method *must* be implemented properly for the feature to work.
        logging.warning(f"Placeholder get_file_tree called for repo {repo_id}, branch {branch}. Returning dummy data.")
        return [
            ("README.md", "README.md"),
            ("src/app.py", "src/app.py"),
            ("src/utils.py", "src/utils.py"),
            ("tests/test_app.py", "tests/test_app.py"),
            ("assets/image.png", "assets/image.png"),
            (".gitignore", ".gitignore")
        ]


    def revert_file_changes(self, repo_id: int, file_path: str) -> bool:
        """Reverts uncommitted changes to a specific file."""
        repo = self.get_repository_by_id(repo_id)
        if not repo: return False
        return self.git_service.reset_file_changes(repo.path, file_path)

    # --- System Information ---
    def get_postgres_db_size(self) -> Optional[int]:
        """Retrieves the total size of the PostgreSQL database."""
        return self.db_manager.get_database_size()

    def get_dgraph_disk_usage(self) -> Optional[str]:
        """
        Retrieves Dgraph disk usage.
        Currently, this is a placeholder as direct Dgraph disk usage query is complex.
        Future: Implement querying Dgraph's /state endpoint or metrics.
        """
        # Placeholder implementation.
        # For a real implementation, you might:
        # 1. Use `requests` to hit Dgraph's /state endpoint (if available and provides disk info).
        #    Example: requests.get(f"http://{DGRAPH_HOST}:8080/state") and parse.
        # 2. Or, if Dgraph is running in a known environment (e.g., Docker),
        #    exec into the container to check disk usage of Dgraph's data directories (`p`, `w` folders).
        #    This is more complex and less portable.
        # 3. Check Dgraph documentation for recommended ways to monitor disk usage.

        # Dgraph Alpha's default HTTP port is 8080. DGRAPH_PORT from config is the gRPC port (9080).
        # We need to target the HTTP port for /debug/vars.
        dgraph_http_port = 8080 # Common default, might need to be configurable if user changes it.
        metrics_url = f"http://{DGRAPH_HOST}:{dgraph_http_port}/debug/vars"

        try:
            response = requests.get(metrics_url, timeout=5) # 5 second timeout
            response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)
            data = response.json()

            # Look for Badger specific metrics for data size (p directory)
            # Common metrics are 'badger_lsm_size_bytes' and 'badger_vlog_size_bytes'
            # Summing them gives an estimate of the 'p' (postings) directory.
            # WAL ('w' directory) size is usually separate and might not be easily available here.

            lsm_size = data.get("badger_lsm_size_bytes", 0)
            vlog_size = data.get("badger_vlog_size_bytes", 0)

            if isinstance(lsm_size, (int, float)) and isinstance(vlog_size, (int, float)):
                total_badger_size_bytes = int(lsm_size + vlog_size)
                if total_badger_size_bytes > 0:
                    # Format the size
                    if total_badger_size_bytes < 1024:
                        return f"{total_badger_size_bytes} Bytes (Badger data)"
                    elif total_badger_size_bytes < 1024**2:
                        return f"{total_badger_size_bytes/1024:.2f} KB (Badger data)"
                    elif total_badger_size_bytes < 1024**3:
                        return f"{total_badger_size_bytes/1024**2:.2f} MB (Badger data)"
                    else:
                        return f"{total_badger_size_bytes/1024**3:.2f} GB (Badger data)"
                else: # Metrics found but are zero
                    return "Approx. 0 MB (Badger data)"

            # Fallback if specific badger metrics aren't found directly,
            # or if the structure is different. Some versions might have a total.
            # This is highly speculative.
            disk_used_metrics = [
                "dgraph_disk_used_bytes", "badger_disk_usage", "disk_usage_bytes" # Add other potential keys
            ]
            for key in disk_used_metrics:
                if key in data and isinstance(data[key], (int, float)) and data[key] > 0:
                    size_bytes = int(data[key])
                    # Format size (similar to above)
                    if size_bytes < 1024: return f"{size_bytes} Bytes"
                    # ... (add KB, MB, GB formatting)
                    return f"{size_bytes/1024**3:.2f} GB (metric: {key})"

            logging.warning(f"Dgraph metrics endpoint {metrics_url} accessible, but known disk usage keys (e.g., badger_lsm_size_bytes, badger_vlog_size_bytes) not found or zero. Full response keys: {list(data.keys())}")
            return "Dgraph usage: N/A (Metrics found, specific keys missing)"

        except requests.exceptions.RequestException as e:
            logging.warning(f"Could not connect to Dgraph metrics endpoint at {metrics_url}: {e}")
            return "Dgraph usage: N/A (Connection error)"
        except json.JSONDecodeError:
            logging.warning(f"Failed to decode JSON from Dgraph metrics endpoint {metrics_url}")
            return "Dgraph usage: N/A (Invalid JSON response)"
        except Exception as e:
            logging.error(f"An unexpected error occurred while fetching Dgraph disk usage: {e}", exc_info=True)
            return "Dgraph usage: N/A (Error)"

    def get_usage_statistics(self) -> Dict[str, Any]:
        """Retrieves general and AI usage statistics."""
        stats = {
            "general": {
                "num_repositories": 0,
                "total_files_analyzed": 0, # This might be complex to get accurately across all branches/schemas
                "total_lines_analyzed": 0, # Same complexity as files
            },
            "ai": {
                "total_llm_calls": 0, # Placeholder, assuming BillingUsage doesn't distinguish calls vs tokens for a single "call" record
                "total_tokens_processed": 0,
                "estimated_cost": 0.0,
                "models_used": {} # To store breakdown by model
            }
        }

        with self.db_manager.get_session("public") as session:
            # General Stats
            stats["general"]["num_repositories"] = session.query(func.count(Repository.id)).scalar() or 0

            # AI Stats from BillingUsage
            # Assuming each row in BillingUsage is one "call" or "transaction"
            # If a single agent interaction results in multiple DB rows, this count might be high.
            # For now, count rows where provider is not 'local' (assuming 'local' is for embeddings)
            ai_usage_query = session.query(
                func.count(BillingUsage.id),
                func.sum(BillingUsage.total_tokens),
                func.sum(BillingUsage.cost)
            ).filter(BillingUsage.provider != 'local') # Exclude local embedding "costs" if any

            ai_results = ai_usage_query.first()
            if ai_results:
                stats["ai"]["total_llm_calls"] = ai_results[0] or 0
                stats["ai"]["total_tokens_processed"] = ai_results[1] or 0
                stats["ai"]["estimated_cost"] = ai_results[2] or 0.0

            # AI Model specific breakdown
            model_usage_query = session.query(
                BillingUsage.model_name,
                func.count(BillingUsage.id), # Calls per model
                func.sum(BillingUsage.total_tokens),
                func.sum(BillingUsage.cost)
            ).filter(BillingUsage.provider != 'local').group_by(BillingUsage.model_name)

            for row in model_usage_query.all():
                stats["ai"]["models_used"][row.model_name] = {
                    "calls": row[1] or 0,
                    "tokens": row[2] or 0,
                    "cost": row[3] or 0.0
                }

        # For total_files_analyzed and total_lines_analyzed, it's more complex.
        # We could iterate through all repositories and their active branches,
        # then call self.get_repository_stats(repo.id, repo.active_branch)
        # and sum them up. This could be slow if there are many repos.
        # For now, these will remain 0 or be simplified.
        # Let's try a simplified sum from all DBFile entries if feasible,
        # but this doesn't respect branches or schemas properly.
        # A more accurate way for "active" stats would be to sum from latest task results.
        # Keeping it simple for now.

        return stats

    def get_task_history(self, repo_id: Optional[int], offset: int = 0, limit: int = 20) -> List[Task]:
        """
        Retrieves a paginated list of all parent tasks for a repository (or all tasks if repo_id is None),
        ordered by most recent first. Includes children tasks.
        """
        with self.db_manager.get_session("public") as session:
            query = session.query(Task).options(joinedload(Task.children)).filter(Task.parent_id.is_(None))
            if repo_id is not None:
                query = query.filter(Task.repository_id == repo_id)

            tasks = query.order_by(Task.started_at.desc()).offset(offset).limit(limit).all()
            session.expunge_all()
            return tasks