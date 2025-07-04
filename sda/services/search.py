# sda/services/search.py

import logging
from pathlib import Path
from typing import List

import ahocorasick

from sda.config import IngestionConfig


class FullTextSearchService:
    """Performs fast multi-keyword searches across all text files in a repository."""

    # A comprehensive set of extensions for files that are likely to be text-based.
    ADDITIONAL_TEXT_EXTENSIONS = {
        '.md', '.rst', '.txt', '.json', '.xml', '.yaml', '.yml', '.toml', '.ini',
        '.cfg', '.conf', '.log', '.csv', '.tsv', '.html', '.htm', '.css', '.sh',
        '.bat', '.ps1', '.sql', '.properties', '.tex', '.svg'
    }
    # Common text files that often lack an extension.
    TEXT_FILES_NO_EXT = {'README', 'LICENSE', 'Dockerfile', '.gitignore', '.gitattributes'}

    def __init__(self):
        """Initializes the FullTextSearchService."""
        # Pre-compile the set of all text extensions for faster lookups.
        self.all_text_extensions = self.ADDITIONAL_TEXT_EXTENSIONS.union(
            set(IngestionConfig.LANGUAGE_MAPPING.keys())
        )
        logging.info("FullTextSearchService initialized.")

    def search(self, repo_path_str: str, queries: List[str]) -> List[str]:
        """
        Finds files containing any of the given query strings.

        This method uses the Aho-Corasick algorithm for efficient multi-pattern
        string searching, making it much faster than iterating and checking for
        each query individually.

        Args:
            repo_path_str: The absolute path to the repository's root directory.
            queries: A list of strings to search for.

        Returns:
            A sorted list of relative file paths that match one or more queries.
        """
        if not queries:
            return []

        repo_path = Path(repo_path_str)
        # Build the Aho-Corasick automaton for all query keywords.
        automaton = ahocorasick.Automaton()
        for index, keyword in enumerate(queries):
            automaton.add_word(keyword.casefold(), (index, keyword))
        automaton.make_automaton()

        matching_files = set()

        for file_path in repo_path.rglob('*'):
            # Skip ignored directories and non-files.
            if any(part in IngestionConfig.DEFAULT_IGNORE_DIRS for part in file_path.parts):
                continue
            if not file_path.is_file():
                continue

            is_text_file = (
                file_path.suffix.lower() in self.all_text_extensions
                or file_path.name in self.TEXT_FILES_NO_EXT
            )
            if is_text_file:
                try:
                    # Using read_text for simplicity and robustness with encodings.
                    content = file_path.read_text(encoding='utf-8', errors='ignore')
                    # The `exist()` method is a highly optimized check.
                    if automaton.exist(content.casefold()):
                        matching_files.add(str(file_path.relative_to(repo_path)))
                except Exception as e:
                    # Log and continue if a file cannot be read.
                    logging.debug(f"Could not read or process {file_path} during full-text search: {e}")
                    continue

        return sorted(list(matching_files))