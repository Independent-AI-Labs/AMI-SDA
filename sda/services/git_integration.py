# sda/services/git_integration.py

"""
Provides a service for interacting with Git repositories via the command line.

This service manages cloning, pulling, checking out branches, and executing
other Git commands within a designated workspace directory, acting as the
foundation for the framework's version control capabilities.
"""

import logging
import re
import subprocess
import threading
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Dict

from sda.config import WORKSPACE_DIR

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class GitService:
    """
    Manages local clones of Git repositories. This class is thread-safe.
    """

    def __init__(self, workspace_dir: str = WORKSPACE_DIR):
        """
        Initializes the Git service.

        Args:
            workspace_dir: The directory to store cloned repositories.
        """
        self.workspace_dir = Path(workspace_dir)
        self.workspace_dir.mkdir(exist_ok=True, parents=True)
        # A lock to ensure that only one git command runs at a time per service instance,
        # preventing race conditions on a single repository.
        self.lock = threading.Lock()
        logging.info(f"GitService initialized. Workspace: {self.workspace_dir.resolve()}")

    def _execute_command(self, repo_path: Path, command: List[str]) -> Optional[str]:
        """Executes a Git command in a specific repository directory, capturing stdout."""
        try:
            # For cloning, the CWD should be the workspace, not the (non-existent) repo_path.
            cwd = self.workspace_dir if command[0] == 'clone' else repo_path
            git_command = ['git'] + command
            logging.debug(f"Executing Git command: `{' '.join(git_command)}` in `{cwd}`")
            process = subprocess.run(
                git_command,
                cwd=cwd,
                check=True,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='ignore'  # Ignore decoding errors from git output
            )
            return process.stdout.strip()
        except FileNotFoundError:
            logging.error("Git command not found. Is Git installed and in your system's PATH?")
            return None
        except subprocess.CalledProcessError as e:
            logging.error(f"Git command failed: `git {' '.join(command)}` with code {e.returncode}")
            logging.error(f"Stderr: {e.stderr.strip()}")
            return None

    def _has_remote(self, repo_path: Path, remote_name: str = 'origin') -> bool:
        """Checks if a given remote exists in the repository."""
        remotes = self._execute_command(repo_path, ['remote'])
        return remotes is not None and remote_name in remotes.split()

    def get_repo_path(self, repo_url: str) -> Path:
        """Derives a sanitized local directory name from a Git URL or path."""
        # Use the last part of the URL/path
        repo_name = repo_url.split('/')[-1]
        # Remove .git suffix if it exists
        if repo_name.endswith('.git'):
            repo_name = repo_name[:-4]
        # Sanitize the name to be a valid directory name
        repo_name = re.sub(r'[^a-zA-Z0-9_.-]', '_', repo_name)
        return self.workspace_dir / repo_name

    def clone_or_pull(self, repo_identifier: str) -> Optional[str]:
        """
        Ensures a repository is available locally. Clones if new, fetches if existing.
        Also handles registering a pre-existing local repository.

        Args:
            repo_identifier: A remote Git URL or an absolute local file path.

        Returns:
            The absolute path to the local repository, or None on failure.
        """
        with self.lock:
            # Check if the identifier is a local path that is a valid git repo.
            local_path = Path(repo_identifier).resolve()
            if local_path.is_dir() and (local_path / '.git').is_dir():
                logging.info(f"Registering existing local repository at {local_path}")
                if self._has_remote(local_path):
                    # Fetch without prune to be more robust against broken local refs
                    self._execute_command(local_path, ['fetch', '--all', '--quiet'])
                return str(local_path)

            # If not a local path, assume it's a URL and proceed with clone/pull logic.
            repo_path = self.get_repo_path(repo_identifier)
            if repo_path.is_dir() and (repo_path / '.git').is_dir():
                logging.info(f"Repository exists at {repo_path}. Fetching updates...")
                if self._has_remote(repo_path):
                    self._execute_command(repo_path, ['fetch', '--all', '--quiet'])
            else:
                logging.info(f"Cloning repository {repo_identifier} into {repo_path}...")
                result = self._execute_command(self.workspace_dir, ['clone', '--quiet', repo_identifier, str(repo_path)])
                if result is None:
                    logging.error(f"Failed to clone repository: {repo_identifier}")
                    return None
            return str(repo_path.resolve())

    def checkout(self, repo_path_str: str, branch: str) -> bool:
        """
        Checks out a specific branch. Assumes it has been fetched.

        This method no longer automatically pulls changes to avoid potential merge
        conflicts that cannot be resolved by an automated system. It is expected
        that the repository is kept up-to-date by other means or that analysis
        on a slightly older commit is acceptable.
        """
        with self.lock:
            repo_path = Path(repo_path_str)
            logging.info(f"Checking out branch '{branch}' in {repo_path}...")

            # Checkout the branch. It must exist locally from a previous fetch.
            if self._execute_command(repo_path, ['checkout', branch, '--quiet']) is None:
                logging.error(f"Failed to checkout local branch '{branch}'. It may not exist.")
                return False

            return True

    def get_current_branch(self, repo_path_str: str) -> Optional[str]:
        """Gets the name of the current active branch."""
        with self.lock:
            repo_path = Path(repo_path_str)
            return self._execute_command(repo_path, ['rev-parse', '--abbrev-ref', 'HEAD'])

    def list_branches(self, repo_path_str: str) -> List[str]:
        """Lists all local branches, which are the ones a user can directly work with."""
        with self.lock:
            repo_path = Path(repo_path_str)
            # Fetch ensures we have the latest refs, but we list local branches
            # as these are what can be checked out.
            if self._has_remote(repo_path):
                self._execute_command(repo_path, ['fetch', '--all', '--quiet'])

            output = self._execute_command(repo_path, ['branch', '--list', '--no-color'])
            if output is None:
                return []

            branches = [line.strip().lstrip('* ').strip() for line in output.split('\n') if line]
            # Filter out HEAD pointers like 'HEAD -> origin/main'
            return sorted([b for b in branches if '->' not in b])

    def stage_all(self, repo_path_str: str) -> bool:
        """Stages all modified, new, and deleted files."""
        with self.lock:
            repo_path = Path(repo_path_str)
            logging.info(f"Staging all changes in {repo_path}...")
            return self._execute_command(repo_path, ['add', '-A']) is not None

    def commit(self, repo_path_str: str, message: str) -> bool:
        """Commits all staged changes."""
        with self.lock:
            repo_path = Path(repo_path_str)
            logging.info(f"Committing changes in {repo_path} with message: '{message}'")
            return self._execute_command(repo_path, ['commit', '-m', message]) is not None

    def get_status(self, repo_path_str: str) -> Optional[Dict[str, List[str]]]:
        """Gets git status, parsing the porcelain format for machine readability."""
        with self.lock:
            repo_path = Path(repo_path_str)
            output = self._execute_command(repo_path, ['status', '--porcelain', '-u'])
            if output is None: return None

            changes = defaultdict(list)
            for line in output.split('\n'):
                if not line: continue
                status, file_info = line[:2], line[3:]
                # Handle renamed files which appear as "R  old_path -> new_path"
                if status.startswith('R'):
                    changes["renamed"].append(file_info.split(' -> ')[1])
                elif status.startswith(' '):
                    changes["modified"].append(file_info)
                elif status.startswith('M'):
                    changes["modified"].append(file_info)
                elif status.startswith('A'):
                    changes["new"].append(file_info)
                elif status.startswith('D'):
                    changes["deleted"].append(file_info)
                elif status.startswith('??'):
                    changes["untracked"].append(file_info)
            return dict(changes)

    def get_diff(self, repo_path_str: str, file_path: Optional[str] = None) -> Optional[str]:
        """Gets the git diff for un-staged changes, for the repo or a specific file."""
        with self.lock:
            repo_path = Path(repo_path_str)
            command = ['diff', '--no-color']
            if file_path:
                command.extend(['--', file_path])
            return self._execute_command(repo_path, command)

    def reset_file_changes(self, repo_path_str: str, file_path: str) -> bool:
        """Discards un-staged changes for a specific file."""
        with self.lock:
            repo_path = Path(repo_path_str)
            logging.info(f"Resetting changes for file '{file_path}' in {repo_path}")
            # First, unstage the file if it was staged.
            self._execute_command(repo_path, ['restore', '--staged', '--', file_path])
            # Then, discard changes in the working directory.
            return self._execute_command(repo_path, ['restore', '--', file_path]) is not None

    def get_all_files_in_branch(self, repo_path_str: str, branch_name: Optional[str] = None) -> List[str]:
        """
        Lists all files in the specified branch (or current HEAD if no branch specified).
        Returns a list of relative file paths.
        """
        with self.lock:
            repo_path = Path(repo_path_str)
            target_tree_ish = branch_name if branch_name else 'HEAD'

            # Ensure the branch is available locally if specified
            if branch_name:
                # This is a simplification. A robust check would involve `git branch --list`
                # and potentially `git fetch origin branch_name` if not found.
                # For now, assume `git ls-tree` can handle valid branch names that exist.
                pass

            # Use `git ls-tree` to list all files in the given tree-ish (branch or commit)
            # -r: recurse into subtrees
            # --name-only: show only file names
            # HEAD: can be replaced with a branch name or commit SHA
            command = ['ls-tree', '-r', '--name-only', target_tree_ish]
            output = self._execute_command(repo_path, command)

            if output is None:
                logging.error(f"Failed to list files for tree-ish '{target_tree_ish}' in {repo_path}")
                return [] # Return empty list on error

            files = [line.strip() for line in output.split('\n') if line.strip()]
            return files