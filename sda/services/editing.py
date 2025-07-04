# sda/services/editing.py

"""
Provides a robust API for safely modifying files on disk.

The SafeFileEditingSystem ensures that every file modification is safe by
including features like automated backups, syntax validation (using tree-sitter),
and detailed diff generation. After any modification, it marks the file for
reprocessing by the Ingestion Service.
"""

import difflib
import logging
import re
import shutil
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Tuple

from tree_sitter import Parser, Node

from sda.config import BACKUP_DIR, IngestionConfig
from sda.core.db_management import DatabaseManager
from sda.core.models import File, Repository
from sda.services.chunking import TokenAwareChunker

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class EditResult(Enum):
    """Enumeration for the result of an editing operation."""
    SUCCESS = "success"
    ERROR = "error"
    VALIDATION_FAILED = "validation_failed"
    FILE_NOT_FOUND = "file_not_found"
    FILE_EXISTS = "file_exists"


class EditResponse:
    """A structured response for file editing operations."""

    def __init__(self, result: EditResult, message: str, diff: Optional[str] = None):
        self.result = result
        self.message = message
        self.diff = diff

    def to_dict(self):
        return {
            "result": self.result.value,
            "message": self.message,
            "diff": self.diff
        }


class SafeFileEditingSystem:
    """Manages safe and validated modifications of source files."""

    def __init__(self, db_manager: DatabaseManager, chunker: TokenAwareChunker, backup_dir: str = BACKUP_DIR):
        """
        Initializes the file editing system.

        Args:
            db_manager: An instance of DatabaseManager for DB operations.
            chunker: An instance of TokenAwareChunker to access its parsers for validation.
            backup_dir: The directory to store file backups.
        """
        self.db_manager = db_manager
        self.parsers: Dict[str, Parser] = chunker.parsers
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(exist_ok=True)
        logging.info(f"SafeFileEditingSystem initialized. Backups will be saved to {self.backup_dir}")

    def replace_lines(self, file_path_str: str, start_line: int, end_line: int, new_content: str) -> EditResponse:
        """
        Replaces a range of lines in a file with new content. Lines are 1-indexed.
        """
        file_path = Path(file_path_str)
        if not file_path.exists():
            return EditResponse(EditResult.FILE_NOT_FOUND, f"File not found: {file_path}")

        try:
            original_content = file_path.read_text(encoding='utf-8')
            original_lines = original_content.splitlines(keepends=True)
        except IOError as e:
            return EditResponse(EditResult.ERROR, f"Could not read file: {e}")

        if not new_content.endswith('\n'):
            new_content += '\n'
        new_lines = new_content.splitlines(keepends=True)

        # Build the new content from parts to preserve line endings.
        modified_lines = original_lines[:start_line - 1] + new_lines + original_lines[end_line:]
        modified_content = "".join(modified_lines)

        is_valid, validation_msg = self._validate_syntax(file_path, modified_content)
        if not is_valid:
            return EditResponse(EditResult.VALIDATION_FAILED, validation_msg)

        self._create_backup(file_path)

        try:
            file_path.write_text(modified_content, encoding='utf-8')
        except IOError as e:
            return EditResponse(EditResult.ERROR, f"Could not write to file: {e}")

        self._mark_file_for_reprocessing(file_path_str)
        diff = self._generate_diff(original_content, modified_content, file_path.name)
        return EditResponse(EditResult.SUCCESS, f"Successfully modified {file_path.name}", diff)

    def find_and_replace(self, file_path_str: str, find_regex: str, replace_str: str) -> EditResponse:
        """
        Finds and replaces content in a file using a regular expression.
        """
        file_path = Path(file_path_str)
        if not file_path.exists():
            return EditResponse(EditResult.FILE_NOT_FOUND, f"File not found: {file_path}")

        try:
            original_content = file_path.read_text(encoding='utf-8')
        except IOError as e:
            return EditResponse(EditResult.ERROR, f"Could not read file: {e}")

        modified_content = re.sub(find_regex, replace_str, original_content)

        if original_content == modified_content:
            return EditResponse(EditResult.SUCCESS, f"No changes made to {file_path.name}.")

        is_valid, validation_msg = self._validate_syntax(file_path, modified_content)
        if not is_valid:
            return EditResponse(EditResult.VALIDATION_FAILED, validation_msg)

        self._create_backup(file_path)

        try:
            file_path.write_text(modified_content, encoding='utf-8')
        except IOError as e:
            return EditResponse(EditResult.ERROR, f"Could not write to file: {e}")

        self._mark_file_for_reprocessing(file_path_str)
        diff = self._generate_diff(original_content, modified_content, file_path.name)
        return EditResponse(EditResult.SUCCESS, f"Successfully modified {file_path.name}", diff)

    def create_file(self, file_path_str: str, repo_path_str: str, content: str = "") -> EditResponse:
        """
        Creates a new file on disk and registers it in the database.
        """
        file_path = Path(file_path_str)
        if file_path.exists():
            return EditResponse(EditResult.FILE_EXISTS, f"File already exists: {file_path}")

        is_valid, validation_msg = self._validate_syntax(file_path, content)
        if not is_valid:
            return EditResponse(EditResult.VALIDATION_FAILED, validation_msg)

        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content, encoding='utf-8')
        except IOError as e:
            return EditResponse(EditResult.ERROR, f"Could not create file: {e}")

        self._register_new_file(file_path_str, repo_path_str)
        return EditResponse(EditResult.SUCCESS, f"Successfully created new file: {file_path.name}")

    def _validate_syntax(self, file_path: Path, content: str) -> Tuple[bool, str]:
        """Checks the syntax of the content using a language-specific tree-sitter parser."""
        lang_name = IngestionConfig.LANGUAGE_MAPPING.get(file_path.suffix)
        if not lang_name:
            logging.warning(f"No language mapping for '{file_path.suffix}', skipping syntax validation.")
            return True, "No language mapping available for this file type."

        parser = self.parsers.get(lang_name)
        if not parser:
            logging.warning(f"No parser configured for language '{lang_name}', skipping syntax validation.")
            return True, "No parser available for this file type."

        tree = parser.parse(bytes(content, "utf8"))
        if tree.root_node.has_error:
            error_node = self._find_error_node(tree.root_node)
            if error_node:
                line, col = error_node.start_point
                return False, f"Syntax error near line {line + 1}, column {col + 1}."
            return False, "Syntax validation failed."
        return True, "Syntax is valid."

    def _find_error_node(self, node: Node) -> Optional[Node]:
        """Recursively finds the first node of type ERROR in the AST."""
        if node.type == 'ERROR' or node.is_missing:
            return node
        for child in node.children:
            error_node = self._find_error_node(child)
            if error_node:
                return error_node
        return None

    def _create_backup(self, file_path: Path):
        """Creates a timestamped backup of a file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file_path = self.backup_dir / f"{file_path.name}.{timestamp}.bak"
        try:
            shutil.copy2(file_path, backup_file_path)
            logging.info(f"Created backup: {backup_file_path}")
        except Exception as e:
            logging.error(f"Failed to create backup for {file_path}: {e}")

    def _mark_file_for_reprocessing(self, file_path_str: str):
        """Updates all branch records for a file in the DB to trigger re-ingestion."""
        with self.db_manager.get_session("public") as session:
            repo = session.query(Repository).filter(Repository.path == str(Path(file_path_str).parent.resolve())).first()
            if not repo or not repo.db_schemas:
                logging.warning(f"Could not find repo or schemas for file {file_path_str}")
                return

            marked_count = 0
            for schema in repo.db_schemas:
                with self.db_manager.get_session(schema) as schema_session:
                    # Find file records by absolute path across all branches
                    file_records = schema_session.query(File).filter_by(file_path=file_path_str).all()
                    if file_records:
                        for record in file_records:
                            record.content_hash = None  # Setting hash to None forces reprocessing
                        marked_count += len(file_records)

            if marked_count > 0:
                logging.info(f"Marked {marked_count} branch record(s) for re-ingestion: {file_path_str}")
            else:
                logging.warning(f"Could not mark file for reprocessing, record not found for: {file_path_str}")

    def _register_new_file(self, file_path_str: str, repo_path_str: str):
        """Creates a placeholder File record to ensure it's picked up by the next ingestion run."""
        with self.db_manager.get_session("public") as session:
            repo = session.query(Repository).filter_by(path=repo_path_str).first()
            if not repo or not repo.active_branch or not repo.db_schemas:
                logging.error(f"Cannot register new file, repository at {repo_path_str} not found or is not properly initialized.")
                return

            # Determine correct schema for the new file
            file_path = Path(file_path_str)
            relative_path = file_path.relative_to(Path(repo_path_str))
            subdir = relative_path.parts[0] if len(relative_path.parts) > 1 else "_root"
            schema_name = f"repo_{repo.uuid[:8]}_{re.sub(r'[^a-z0-9]+', '_', subdir.lower()).strip('_')}"[:63]

            if schema_name not in repo.db_schemas:
                logging.warning(f"Schema '{schema_name}' for new file does not exist. It will be created on next full ingestion.")
                # We can't add to a schema that doesn't exist yet, so we abort.
                # The file will be picked up on the next full 'analyze_branch' run.
                return

            with self.db_manager.get_session(schema_name) as schema_session:
                file_record = schema_session.query(File).filter_by(file_path=file_path_str, branch=repo.active_branch).first()
                if not file_record:
                    new_file = File(
                        repository_id=repo.id,
                        branch=repo.active_branch,  # Assume it belongs to the current active branch
                        file_path=file_path_str,
                        relative_path=str(relative_path),
                        language=IngestionConfig.LANGUAGE_MAPPING.get(file_path.suffix),
                        content_hash=None,  # Will be processed on next ingestion run
                    )
                    schema_session.add(new_file)
                    logging.info(f"Registered new file in schema '{schema_name}' for branch '{repo.active_branch}': {file_path_str}")

    def _generate_diff(self, old_content: str, new_content: str, filename: str) -> str:
        """Generates a unified diff string."""
        diff_lines = difflib.unified_diff(
            old_content.splitlines(keepends=True),
            new_content.splitlines(keepends=True),
            fromfile=f"a/{filename}",
            tofile=f"b/{filename}",
        )
        return "".join(diff_lines)
