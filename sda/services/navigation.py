# sda/services/navigation.py

"""
Provides a suite of tools for deterministic, branch-aware code navigation.

The AdvancedCodeNavigationTools class queries the structured data layer
(Postgres for node data, Dgraph for relationships) to answer questions about
code structure, such as finding definitions, references, and file outlines,
all within the context of a specific version control branch.
"""

import logging
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import joinedload

from sda.core.db_management import DatabaseManager
from sda.core.models import ASTNode, File, Repository

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class SymbolReference:
    """A structured data class for returning symbol information."""

    def __init__(self, file_path: str, start_line: int, end_line: int, code_snippet: str,
                 name: Optional[str] = None, node_type: Optional[str] = None,
                 node_id: Optional[str] = None, # Dgraph/internal ID
                 analysis_reason: Optional[str] = None): # Reason for inclusion in a specific analysis
        self.file_path = file_path
        self.start_line = start_line
        self.end_line = end_line
        self.code_snippet = code_snippet
        self.name = name
        self.node_type = node_type
        self.node_id = node_id
        self.analysis_reason = analysis_reason

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__


class AdvancedCodeNavigationTools:
    """A collection of functions for branch-aware code navigation and analysis."""

    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        logging.info("AdvancedCodeNavigationTools initialized.")

    def _get_repo_details(self, repo_id: int) -> Optional[Repository]:
        """Helper to get full repository details from the public schema."""
        with self.db_manager.get_session("public") as session:
            repo = session.get(Repository, repo_id)
            if repo:
                session.expunge(repo)
            return repo

    def _read_file_snippet(self, file_path_str: str, start_line: int, end_line: int) -> str:
        """Reads a specific line range from a file, with robust error handling."""
        if not all([file_path_str, start_line, end_line]):
            return "// Content not available (incomplete data)"
        try:
            file_path = Path(file_path_str)
            # Read all lines once and slice. More efficient for many small snippets.
            lines = file_path.read_text(encoding='utf-8', errors='ignore').splitlines(keepends=True)
            # Line numbers are 1-based, list indices are 0-based
            snippet_lines = lines[start_line - 1: end_line]
            return "".join(snippet_lines)
        except Exception as e:
            logging.warning(f"Could not read snippet from {file_path_str}: {e}")
            return f"// Could not read content from {file_path_str}"

    def _get_nodes_from_db(self, node_ids: List[str], repo_id: int) -> List[SymbolReference]:
        """Fetches full ASTNode details from Postgres across all repository schemas for given Dgraph node IDs."""
        if not node_ids: return []

        repo = self._get_repo_details(repo_id)
        if not repo or not repo.db_schemas:
            logging.error(f"Repository details or schemas not found for repo_id {repo_id}")
            return []

        all_results: List[SymbolReference] = []

        def find_in_schema(schema: str, ids_to_find: List[str]):
            with self.db_manager.get_session(schema) as s:
                # Fetch nodes ensuring the original node_id (from Dgraph query) is preserved and passed to SymbolReference
                nodes_data = s.query(ASTNode).filter(ASTNode.node_id.in_(ids_to_find)).options(joinedload(ASTNode.file)).all()
                # Create a map of postgres_node.node_id to the original dgraph_node_id if they could differ
                # (they should be the same, so n.node_id is the one we want)
                return [SymbolReference(
                    file_path=n.file.relative_path, start_line=n.start_line,
                    end_line=n.end_line, code_snippet=self._read_file_snippet(n.file.file_path, n.start_line, n.end_line),
                    name=n.name, node_type=n.node_type,
                    node_id=n.node_id # Populate the node_id field
                ) for n in nodes_data]

        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(find_in_schema, schema, node_ids): schema for schema in repo.db_schemas}
            for future in as_completed(futures):
                try:
                    all_results.extend(future.result())
                except Exception as e:
                    logging.error(f"Error fetching nodes from schema {futures[future]}: {e}")

        # De-duplicate results, as a node might be found in multiple schemas if logic overlaps
        return list({ref.file_path + str(ref.start_line): ref for ref in all_results}.values())

    def find_symbol_definition(self, symbol_name: str, repo_id: int, branch: str) -> List[SymbolReference]:
        """Finds the definition(s) of a symbol within a repository and branch."""
        repo = self._get_repo_details(repo_id)
        if not repo or not repo.db_schemas: return []

        all_results = []

        def find_in_schema(schema):
            with self.db_manager.get_session(schema) as s:
                nodes = s.query(ASTNode).join(File).filter(
                    File.repository_id == repo_id, File.branch == branch, ASTNode.name == symbol_name,
                    ASTNode.node_type.in_(['function_definition', 'class_definition', 'method_declaration', 'constructor_declaration'])
                ).options(joinedload(ASTNode.file)).all()
                return [SymbolReference(
                    file_path=n.file.relative_path, start_line=n.start_line,
                    end_line=n.end_line, code_snippet=self._read_file_snippet(n.file.file_path, n.start_line, n.end_line),
                    name=n.name, node_type=n.node_type
                ) for n in nodes]

        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(find_in_schema, s) for s in repo.db_schemas}
            for future in as_completed(futures):
                try:
                    all_results.extend(future.result())
                except Exception as e:
                    logging.error(f"Error finding symbol in schema: {e}")
        return all_results

    def find_symbol_references(self, symbol_name: str, repo_id: int, branch: str) -> List[SymbolReference]:
        """Finds all usages of a symbol by querying the 'calls' relationship in Dgraph."""
        logging.info(f"Searching Dgraph for references to '{symbol_name}' in repo {repo_id}, branch '{branch}'")

        query = """
            query References($repoId: string, $branch: string, $symbol: string) {
                # Find the definition nodes for the symbol
                definitions(func: type(ASTNode)) @filter(
                    eq(repo_id, $repoId) AND eq(branch, $branch) AND eq(name, $symbol)
                    AND (eq(node_type, "function_definition") OR eq(node_type, "class_definition") OR eq(node_type, "method_declaration"))
                ) {
                    # Find nodes that call any of the definitions
                    ~calls @filter(eq(repo_id, $repoId) AND eq(branch, $branch)) {
                        node_id
                    }
                }
            }
        """
        variables = {"$repoId": str(repo_id), "$branch": branch, "$symbol": symbol_name}
        response = self.db_manager.query_dgraph(query, variables)

        if not response or not response.get('definitions'):
            return []

        referencing_node_ids = {
            caller['node_id']
            for def_block in response['definitions'] if '~calls' in def_block
            for caller in def_block['~calls'] if 'node_id' in caller
        }

        return self._get_nodes_from_db(list(referencing_node_ids), repo_id) if referencing_node_ids else []

    def get_call_hierarchy(self, symbol_name: str, repo_id: int, branch: str) -> Dict[str, Any]:
        """Traces call hierarchy (callers and callees) using Dgraph."""
        # This is a placeholder as a full recursive implementation can be complex and slow.
        # It requires multiple Dgraph queries or a more complex single query.
        return {'symbol': symbol_name, 'callers': [], 'callees': [], 'message': 'This feature is a work in progress.'}

    def get_file_outline(self, file_path_str: str, repo_id: int, branch: str) -> Dict[str, Any]:
        """Provides a structured outline (classes, functions) of a single file."""
        repo = self._get_repo_details(repo_id)
        if not repo or not repo.db_schemas: return {"file_path": file_path_str, "symbols": []}

        # The schema can't be derived, so we must search.
        for schema in repo.db_schemas:
            with self.db_manager.get_session(schema) as session:
                file_record = session.query(File).filter(
                    File.repository_id == repo_id, File.branch == branch, File.relative_path == file_path_str
                ).options(joinedload(File.ast_nodes)).first()

                if file_record:
                    top_level_nodes = sorted(
                        [n for n in file_record.ast_nodes if n.node_type in ('function_definition', 'class_definition', 'method_declaration')],
                        key=lambda n: n.start_line
                    )
                    return {"file_path": file_path_str, "symbols": [{
                        "name": node.name, "type": node.node_type,
                        "start_line": node.start_line, "end_line": node.end_line
                    } for node in top_level_nodes]}

        return {"file_path": file_path_str, "symbols": []}

    def find_dead_code(self, repo_id: int, branch: str) -> List[SymbolReference]:
        """Finds potentially unused code by checking for nodes with no incoming 'calls' edges."""
        logging.info(f"Searching Dgraph for dead code in repo {repo_id}, branch '{branch}'")

        query = """
            query DeadCode($repoId: string, $branch: string) {
              # Find all functions and classes for the repo/branch that have no incoming calls.
              dead(func: type(ASTNode)) @filter(
                eq(repo_id, $repoId) AND eq(branch, $branch) 
                AND (eq(node_type, "function_definition") OR eq(node_type, "class_definition") OR eq(node_type, "method_declaration"))
                AND not has(~calls)
              ) {
                node_id
              }
            }
        """
        variables = {"$repoId": str(repo_id), "$branch": branch}
        response = self.db_manager.query_dgraph(query, variables)

        if not response or not response.get('dead'):
            return []

        dead_node_ids = {node['node_id'] for node in response['dead'] if 'node_id' in node}

        results = []
        if dead_node_ids:
            symbol_references = self._get_nodes_from_db(list(dead_node_ids), repo_id)
            for ref in symbol_references:
                ref.analysis_reason = "Node is a definition with no incoming calls found in Dgraph."
                results.append(ref)
        return results

    def analyze_dependencies(self, file_path_str: str, repo_id: int, branch: str) -> Dict[str, Any]:
        """Analyzes a file's dependencies (placeholder)."""
        return {"file_path": file_path_str, "outgoing_calls": {}, "incoming_calls": {}, 'message': 'This feature is a work in progress.'}