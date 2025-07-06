# sda/services/chunking.py

import logging
import os
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple, Callable

import tiktoken
from llama_index.core.text_splitter import TokenTextSplitter
from tree_sitter import Parser, Node, Language

from sda.config import IngestionConfig
from sda.core.data_models import TransientNode, TransientChunk # Added import

# It's assumed that the necessary tree-sitter language packages are installed,
# e.g., 'pip install tree-sitter-python tree-sitter-java'

# To add a new language, ensure its library is installed and add it to this import block.
# Example: from tree_sitter_go import language as go_language
try:
    from tree_sitter_python import language as python_language
    from tree_sitter_java import language as java_language
    from tree_sitter_javascript import language as javascript_language
    from tree_sitter_typescript import language_typescript as typescript_language
    # New language imports
    from tree_sitter_bash import language as bash_language
    from tree_sitter_markdown import language as markdown_language # Assuming tree_sitter_markdown provides 'language'
    from tree_sitter_html import language as html_language
    from tree_sitter_css import language as css_language
except ImportError as e:
    logging.error(f"Failed to import one or more tree-sitter languages. Ensure all required parsers (Python, Java, JS, TS, Bash, Markdown, HTML, CSS) are installed. Error: {e}")
    # Depending on policy, we might want to raise e or log and continue with available parsers.
    # For now, let's make it critical if any are missing, as config expects them.
    raise

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class Lazy:
    """A wrapper class for lazy initialization of objects in worker processes."""

    def __init__(self, constructor: Callable[[], Any]):
        self._constructor = constructor  # Stores the function to create the object (e.g., lambda: TokenAwareChunker())
        self._instance = None  # The future object starts as None

    def __call__(self) -> Any:
        if self._instance is None:
            # If the object hasn't been created yet...
            logging.info(f"Lazily initializing object in process {os.getpid()}...")
            self._instance = self._constructor()  # ...create it now.
        return self._instance  # Return the created (or pre-existing) object.


class TokenAwareChunker:
    """
    A chunker that uses tree-sitter parsers to create syntactically aware chunks
    for multiple programming languages, ensuring full file coverage.
    """

    def __init__(self,
                 max_tokens: int = IngestionConfig.MAX_CHUNK_TOKENS,
                 model_name: str = IngestionConfig.TOKENIZER_MODEL):
        """Initializes the chunker and loads parsers for all configured languages."""
        self.max_tokens = max_tokens
        self.weights = IngestionConfig.IMPORTANCE_WEIGHTS
        try:
            self.tokenizer = tiktoken.get_encoding(model_name)
        except ValueError:
            logging.warning(f"Tokenizer model '{model_name}' not found. Using 'cl100k_base'.")
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

        self.parsers: Dict[str, Parser] = {}
        # This map allows for easy extension with new languages.
        # Just add the language name and its imported library here.
        language_map = {
            'python': python_language(),
            'java': java_language(),
            'javascript': javascript_language(),
            'typescript': typescript_language(),
            'bash': bash_language(),
            'markdown': markdown_language(),
            'html': html_language(),
            'css': css_language(),
        }

        for lang_name, lang_lib in language_map.items():
            if lang_name in IngestionConfig.LANGUAGE_MAPPING.values() and lang_lib:
                language = Language(lang_lib)
                parser = Parser(language)
                self.parsers[lang_name] = parser
        logging.info(f"TokenAwareChunker initialized for languages: {list(self.parsers.keys())}")

    def count_tokens(self, text: str) -> int:
        """Counts the number of tokens in a string using the configured tokenizer."""
        return len(self.tokenizer.encode(text, allowed_special="all"))

    def _calculate_complexity(self, node: Node, lang_name: str) -> int:
        """Calculates cyclomatic complexity for a given node based on language-specific rules."""
        complexity_nodes = IngestionConfig.LANGUAGE_SETTINGS[lang_name].get('COMPLEXITY_NODES', set())
        complexity = 1
        queue = [node]
        while queue:
            current = queue.pop(0)
            if current.type in complexity_nodes:
                # Special handling for Python boolean operators which can inflate complexity
                if lang_name == 'python' and current.type in ('boolean_operator', 'binary_operator'):
                    if current.text.decode('utf8', errors='ignore') in ('and', 'or'):
                        complexity += 1
                else:
                    complexity += 1
            queue.extend(current.children)
        return complexity

    def process_file(self, file_path: Path, content: str, file_identifier: str) -> Tuple[List[TransientNode], List[TransientChunk]]:
        """
        Processes a file's content to extract AST nodes and create semantic chunks.

        Args:
            file_path: The path to the file.
            content: The string content of the file.
            file_identifier: A unique string for the file (e.g., relative path).

        Returns:
            A tuple containing a list of TransientNodes and a list of TransientChunks.
        """
        lang_name = IngestionConfig.LANGUAGE_MAPPING.get(file_path.suffix)
        if not lang_name or lang_name not in self.parsers:
            logging.debug(f"Unsupported language or no parser for '{file_path.suffix}'. Skipping AST processing.")
            return [], []

        parser = self.parsers[lang_name]
        tree = parser.parse(bytes(content, "utf8"))

        if not tree.root_node or tree.root_node.has_error:
            logging.warning(f"Could not parse or found errors in file: {file_path}. Skipping AST extraction.")
            return [], []

        ast_nodes = self._extract_ast_nodes(tree.root_node, file_identifier, lang_name)
        chunks = self._group_nodes_into_chunks(content, ast_nodes, file_identifier, lang_name)

        return ast_nodes, chunks


    def process_file_for_hierarchical_storage(
        self,
        file_path: Path,
        file_content: str,
        file_identifier: str, # Typically relative_path_posix
        repo_metadata: Dict[str, Any] # Contains repo_id, branch, source_blob_hash
    ) -> Tuple[List[TransientNode], List[Dict[str, Any]]]:
        """
        Processes a file's content to extract AST nodes with detailed offsets
        and define DBCodeChunk data (as dicts) with offsets relative to the file content.
        """
        lang_name = IngestionConfig.LANGUAGE_MAPPING.get(file_path.suffix)
        if not lang_name or lang_name not in self.parsers:
            logging.debug(f"Unsupported language or no parser for '{file_path.suffix}'. Skipping AST processing for hierarchical storage.")
            return [], []

        parser = self.parsers[lang_name]
        tree = parser.parse(bytes(file_content, "utf8"))

        if not tree.root_node or tree.root_node.has_error:
            logging.warning(f"Could not parse or found errors in file: {file_path}. Skipping AST extraction for hierarchical storage.")
            return [], []

        # Pass file_path as string for TransientNode, and repo_metadata
        ast_nodes_with_offsets = self._extract_ast_nodes_with_offsets(
            tree.root_node,
            file_identifier, # Used for node_id generation prefix
            lang_name,
            relative_file_path_str=file_identifier, # Explicitly pass relative file path for TransientNode.file_path
            repo_metadata=repo_metadata
        )

        db_chunk_definitions = self._create_db_chunk_definitions(
            ast_nodes_with_offsets,
            file_content,
            file_identifier, # relative_path_posix
            lang_name,
            repo_metadata
        )

        return ast_nodes_with_offsets, db_chunk_definitions

    def _extract_docstring(self, node: Node, lang_name: str) -> Optional[str]:
        """Extracts a docstring/JSDoc/JavaDoc comment for a given AST node."""
        if lang_name == 'python':
            body_node = node.child_by_field_name('body')
            if body_node and body_node.named_child_count > 0:
                first_child = body_node.named_children[0]
                if first_child.type == 'expression_statement' and first_child.child_count > 0 and first_child.children[0].type == 'string':
                    return first_child.children[0].text.decode('utf8', errors='ignore')
        elif lang_name in ['java', 'javascript', 'typescript']:
            # For Java/JS/TS, the doc comment is typically the preceding sibling comment.
            prev_sibling = node.prev_named_sibling
            if prev_sibling and prev_sibling.type in ('block_comment', 'comment'):
                # Check for Javadoc/JSDoc style
                if prev_sibling.text.decode('utf8', errors='ignore').startswith(('/**', '/*')):
                    return prev_sibling.text.decode('utf8', errors='ignore')
        return None

    def _get_node_name(self, node: Node, lang_name: str) -> Optional[str]:
        """
        Extracts the name of an identifier node with language-specific logic.
        This is a critical fix for symbol lookups.
        """
        name_node = node.child_by_field_name('name')
        if name_node:
            return name_node.text.decode('utf8', errors='ignore')

        # Language-specific fallbacks for when 'name' field isn't present.
        if lang_name == 'java':
            if node.type in ('class_declaration', 'method_declaration', 'constructor_declaration'):
                # Search for an 'identifier' or 'type_identifier' child if 'name' is missing.
                for child in node.children:
                    if child.type in ('identifier', 'type_identifier'):
                        return child.text.decode('utf8', errors='ignore')
            elif node.type == 'method_invocation':
                name_node = node.child_by_field_name('name')
                if name_node:
                    return name_node.text.decode('utf8', errors='ignore')

        elif lang_name in ('javascript', 'typescript'):
            # Handle `const myFunc = () => {}` or `let myVar = ...`
            if node.type == 'lexical_declaration':
                # Navigate: lexical_declaration -> variable_declarator -> identifier
                for var_declarator in node.named_children:
                    if var_declarator.type == 'variable_declarator':
                        name_node = var_declarator.child_by_field_name('name')
                        if name_node:
                            return name_node.text.decode('utf8', errors='ignore')
            elif node.type == 'call_expression':
                # Navigate: call_expression -> identifier
                callee = node.child_by_field_name('function')
                if callee:
                    return callee.text.decode('utf8', errors='ignore')

        # Fallback to searching common identifier node types if still not found
        for child in node.children:
            if child.type in ('identifier', 'type_identifier', 'property_identifier'):
                return child.text.decode('utf8', errors='ignore')

        return None

    def _extract_ast_nodes_with_offsets(
        self,
        root_node: Node,
        file_identifier_prefix: str, # Used for node_id generation
        lang_name: str,
        relative_file_path_str: str, # For TransientNode.file_path
        repo_metadata: Dict[str, Any]
    ) -> List[TransientNode]:
        """
        Recursively traverses the AST to extract relevant nodes, populating them
        with detailed offset information and repo_metadata.
        Does NOT store text_content in TransientNode.
        """
        nodes: List[TransientNode] = []
        # These sets define which node types are considered significant for extraction
        identifier_nodes = IngestionConfig.LANGUAGE_SETTINGS[lang_name].get('IDENTIFIER_NODES', set())
        chunkable_nodes = IngestionConfig.LANGUAGE_SETTINGS[lang_name].get('CHUNKABLE_NODES', set()) # For complexity etc.

        def traverse(node: Node, parent_node_id_str: Optional[str] = None, depth: int = 0):
            # node_id generation: file_identifier_prefix is relative_path_posix
            # Using start_byte for uniqueness within the file
            current_node_id = f"{file_identifier_prefix}:{node.type}:{node.start_byte}-{node.end_byte}"
            node_type = node.type

            # We process nodes that are considered "identifier_nodes" by config
            # These are typically definitions, calls, or other semantically important elements.
            if node_type in identifier_nodes:
                name = self._get_node_name(node, lang_name)

                # If it's a chunkable_node (like a function/class def) but has no name, it might be less useful.
                # However, we still want to capture its structure.
                # For now, let's include it even if name is None, as its structure might be chunked.

                signature_text = name # Default signature to name
                complexity_score_val = None
                if node_type in chunkable_nodes: # e.g. function, class, method definitions
                    complexity_score_val = self._calculate_complexity(node, lang_name)
                    # Try to get a more detailed signature for functions/methods
                    params_node = node.child_by_field_name('parameters')
                    if name and params_node:
                        try:
                            signature_text = f"{name}{params_node.text.decode('utf8', errors='ignore')}"
                        except AttributeError: # If params_node is None or no text
                            pass

                if name and len(name) > 512: # Ensure name is not overly long
                    name = name[:512]

                docstring_text = self._extract_docstring(node, lang_name)

                # Create TransientNode without text_content
                # Line numbers from tree-sitter are 0-indexed, convert to 1-indexed for start_line/end_line
                transient_node_obj = TransientNode(
                    node_id=current_node_id,
                    file_path=relative_file_path_str, # Store relative file path
                    node_type=node_type,
                    name=name,
                    start_line=node.start_point[0] + 1,
                    start_column=node.start_point[1],
                    end_line=node.end_point[0] + 1,
                    end_column=node.end_point[1],
                    start_char_offset=node.start_byte, # tree-sitter provides byte offsets
                    end_char_offset=node.end_byte,     # tree-sitter provides byte offsets (exclusive end)
                    signature=signature_text,
                    docstring=docstring_text,
                    parent_id=parent_node_id_str,
                    depth=depth,
                    complexity_score=complexity_score_val,
                    repo_metadata=repo_metadata.copy() # Pass along repo context (includes source_blob_hash)
                )
                nodes.append(transient_node_obj)

            # Recurse for children
            # The parent_id for children should be the current_node_id if it was significant enough to be created,
            # otherwise, propagate the parent_node_id_str from the level above.
            effective_parent_id_for_children = current_node_id if node_type in identifier_nodes else parent_node_id_str
            for child_node in node.children:
                traverse(child_node, parent_node_id_str=effective_parent_id_for_children, depth=depth + 1)

        traverse(root_node)
        return nodes

    def _create_db_chunk_definitions(
        self,
        ast_nodes: List[TransientNode],
        full_file_content: str,
        file_identifier: str, # relative_path_posix
        lang_name: str,
        repo_metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]: # Returns list of dicts for DBCodeChunk
        """
        Groups AST nodes and file content into DBCodeChunk definitions (dictionaries).
        These definitions include offsets and metadata, but not the direct content.
        Ensures 100% file content coverage through structural and remainder chunks.
        """
        if not full_file_content:
            return []

        db_chunk_definitions: List[Dict[str, Any]] = []

        # Map node_id to TransientNode for quick lookup if needed for parent_chunk_id logic
        # node_id_to_transient_node_map = {node.node_id: node for node in ast_nodes}

        chunkable_nodes_set = IngestionConfig.LANGUAGE_SETTINGS[lang_name].get('CHUNKABLE_NODES', set())
        content_lines = full_file_content.splitlines(keepends=True) # For line-based coverage tracking
        covered_lines = [False] * len(content_lines) # To track line coverage

        text_splitter = TokenTextSplitter(
            chunk_size=self.max_tokens,
            chunk_overlap=int(self.max_tokens * IngestionConfig.CHUNK_OVERLAP_RATIO),
            tokenizer=self.tokenizer.encode # Pass the tokenizer's encode method
        )

        # 1. Create chunks from major structural nodes (identified by TransientNode.node_type)
        # Sort by start_char_offset to process in order
        structural_transient_nodes = sorted(
            [tn for tn in ast_nodes if tn.node_type in chunkable_nodes_set and tn.name], # Ensure it has a name
            key=lambda tn: tn.start_char_offset
        )

        for i, s_node in enumerate(structural_transient_nodes):
            # Content for this structural node is defined by its char offsets
            # No need to concatenate docstring here, as offsets define the full node text.
            # The docstring is part of the node's span if it's conventionally placed.

            # We need the actual text slice to determine if it needs further splitting by TokenTextSplitter
            node_text_slice = full_file_content[s_node.start_char_offset : s_node.end_char_offset]
            tokens_for_node_slice = self.count_tokens(node_text_slice)

            # Base for chunk_id, using file_identifier and node's own ID (which includes offsets)
            chunk_id_base = f"{file_identifier}:STRUCT:{s_node.node_id}"

            parent_dgraph_node_uid = s_node.parent_id # Dgraph UID of parent AST Node

            if tokens_for_node_slice > self.max_tokens:
                # If the structural node's content itself is too large, split it.
                # The sub-chunks will share the same dgraph_node_uid and ast_node_ids (pointing to the main structural node)
                # Their offsets will be relative to the start of the structural node, then adjusted to be file-relative.

                sub_chunk_texts = text_splitter.split_text(node_text_slice)
                current_sub_offset = 0
                parent_logical_chunk_id_for_split = f"{chunk_id_base}_0" # First sub-chunk is its own parent in this context

                for sub_idx, sub_text_content in enumerate(sub_chunk_texts):
                    sub_chunk_start_char = s_node.start_char_offset + current_sub_offset
                    sub_chunk_end_char = sub_chunk_start_char + len(sub_text_content.encode('utf-8')) # Use byte length for offset calc if text is unicode

                    # A more robust way for sub_chunk_end_char if sub_text_content is unicode:
                    # Find where sub_text_content starts in node_text_slice and add its length.
                    # This is tricky if sub_text_content is not unique.
                    # Assuming TokenTextSplitter gives clean splits, we can track cumulative length.
                    # For simplicity, let's assume byte length is a decent proxy or that split_text handles this.
                    # A better way: map sub-chunk start/end back to original content string indices.
                    # This part is complex if sub_text_content can have overlaps or is not directly from original.
                    # For now, let's assume simple concatenation of lengths for offsets. This needs robustification.
                    # This is a placeholder for proper sub-offset calculation
                    actual_sub_text_start_in_node_slice = node_text_slice.find(sub_text_content, current_sub_offset if sub_idx > 0 else 0)
                    if actual_sub_text_start_in_node_slice != -1:
                         sub_chunk_start_char = s_node.start_char_offset + actual_sub_text_start_in_node_slice
                         sub_chunk_end_char = sub_chunk_start_char + len(sub_text_content) # char length
                    else: # Fallback if find fails (e.g. due to normalization by splitter)
                         sub_chunk_start_char = s_node.start_char_offset + current_sub_offset
                         sub_chunk_end_char = sub_chunk_start_char + len(sub_text_content)


                    current_sub_offset += len(sub_text_content)


                    # Sub-chunk ID needs to be unique
                    sub_chunk_id = f"{chunk_id_base}_{sub_idx}"

                    db_chunk_def = {
                        "chunk_id": sub_chunk_id,
                        "repository_id": repo_metadata['repository_id'],
                        "branch": repo_metadata['branch'],
                        "language": lang_name,
                        "source_blob_hash": repo_metadata['source_blob_hash'],
                        "start_char_offset": sub_chunk_start_char,
                        "end_char_offset": sub_chunk_end_char,
                        "relative_file_path": file_identifier,
                        # Line numbers for sub-chunks are hard to get accurately without re-parsing or detailed mapping.
                        # For now, use parent node's lines, or mark as approximate.
                        "start_line": s_node.start_line,
                        "end_line": s_node.end_line,
                        "dgraph_node_uid": s_node.node_id, # All sub-chunks point to the same main Dgraph node
                        "ast_node_ids": [s_node.node_id],
                        "chunk_metadata": {"type": f"split_{s_node.node_type}", "original_node_id": s_node.node_id, "split_index": sub_idx},
                        "importance_score": self.weights.get("score_split_structural", 0.4) * self._calculate_importance(s_node, self.count_tokens(sub_text_content)), # Adjust score
                        "parent_chunk_id": parent_logical_chunk_id_for_split if sub_idx > 0 else None # Link to first sub-chunk
                    }
                    db_chunk_definitions.append(db_chunk_def)
            else:
                # Structural node fits in one chunk
                db_chunk_def = {
                    "chunk_id": chunk_id_base,
                    "repository_id": repo_metadata['repository_id'],
                    "branch": repo_metadata['branch'],
                    "language": lang_name,
                    "source_blob_hash": repo_metadata['source_blob_hash'],
                    "start_char_offset": s_node.start_char_offset,
                    "end_char_offset": s_node.end_char_offset,
                    "relative_file_path": file_identifier,
                    "start_line": s_node.start_line,
                    "end_line": s_node.end_line,
                    "dgraph_node_uid": s_node.node_id,
                    "ast_node_ids": [s_node.node_id], # Could include children IDs if desired for context
                    "chunk_metadata": {"type": s_node.node_type, "symbol_name": s_node.name, "start_line": s_node.start_line, "end_line": s_node.end_line},
                    "importance_score": self._calculate_importance(s_node, tokens_for_node_slice),
                    "parent_chunk_id": None # Top-level structural chunks might find their parent later if logic is added
                }
                db_chunk_definitions.append(db_chunk_def)

            # Mark lines covered by this structural node (original extent, not sub-chunks)
            for line_num in range(s_node.start_line - 1, s_node.end_line): # start_line is 1-indexed
                if 0 <= line_num < len(covered_lines):
                    covered_lines[line_num] = True

        # 2. Create fallback "remainder" chunks for code between structural chunks or uncovered code
        current_remainder_start_char = -1
        current_remainder_line_start = -1

        for line_idx, line_content_with_ending in enumerate(content_lines):
            char_offset_at_line_start = sum(len(l.encode('utf-8')) for l in content_lines[:line_idx]) # Approximate byte offset

            if not covered_lines[line_idx]:
                if current_remainder_start_char == -1: # Start of a new remainder block
                    current_remainder_start_char = char_offset_at_line_start
                    current_remainder_line_start = line_idx + 1
            elif current_remainder_start_char != -1: # End of a remainder block
                # Process the collected remainder block
                remainder_block_text = full_file_content[current_remainder_start_char:char_offset_at_line_start] # Slice up to start of current covered line

                if remainder_block_text.strip(): # Only process if there's non-whitespace content
                    # Split this remainder block if it's too large
                    sub_remainder_texts = text_splitter.split_text(remainder_block_text)
                    temp_sub_offset = 0
                    for rem_sub_idx, rem_sub_text in enumerate(sub_remainder_texts):
                        rem_chunk_id = f"{file_identifier}:REM:{current_remainder_start_char}_{rem_sub_idx}"

                        # Sub-offset calculation for remainder chunks
                        actual_rem_sub_start_in_block = remainder_block_text.find(rem_sub_text, temp_sub_offset if rem_sub_idx > 0 else 0)
                        rem_sub_start_char_abs = current_remainder_start_char + (actual_rem_sub_start_in_block if actual_rem_sub_start_in_block != -1 else temp_sub_offset)
                        rem_sub_end_char_abs = rem_sub_start_char_abs + len(rem_sub_text)
                        temp_sub_offset += len(rem_sub_text)

                        db_chunk_def = {
                            "chunk_id": rem_chunk_id,
                            "repository_id": repo_metadata['repository_id'],
                            "branch": repo_metadata['branch'],
                            "language": lang_name, # Could be None if it's mixed content or just text
                            "source_blob_hash": repo_metadata['source_blob_hash'],
                            "start_char_offset": rem_sub_start_char_abs,
                            "end_char_offset": rem_sub_end_char_abs,
                            "relative_file_path": file_identifier,
                            "start_line": current_remainder_line_start, # Approximate line numbers for remainder
                            "end_line": line_idx, # Approximate end line
                            "dgraph_node_uid": None, # Remainder chunks usually don't map to a single Dgraph node
                            "ast_node_ids": [], # No specific AST nodes
                            "chunk_metadata": {"type": "remainder", "start_line": current_remainder_line_start, "end_line": line_idx},
                            "importance_score": self.weights.get("score_remainder", 0.2),
                            "parent_chunk_id": None
                        }
                        db_chunk_definitions.append(db_chunk_def)
                current_remainder_start_char = -1 # Reset for next block
                current_remainder_line_start = -1

        # Check for any remaining uncovered block at the end of the file
        if current_remainder_start_char != -1:
            remainder_block_text = full_file_content[current_remainder_start_char:]
            if remainder_block_text.strip():
                sub_remainder_texts = text_splitter.split_text(remainder_block_text)
                temp_sub_offset = 0
                for rem_sub_idx, rem_sub_text in enumerate(sub_remainder_texts):
                    rem_chunk_id = f"{file_identifier}:REM:{current_remainder_start_char}_{rem_sub_idx}"

                    actual_rem_sub_start_in_block = remainder_block_text.find(rem_sub_text, temp_sub_offset if rem_sub_idx > 0 else 0)
                    rem_sub_start_char_abs = current_remainder_start_char + (actual_rem_sub_start_in_block if actual_rem_sub_start_in_block != -1 else temp_sub_offset)
                    rem_sub_end_char_abs = rem_sub_start_char_abs + len(rem_sub_text)
                    temp_sub_offset += len(rem_sub_text)

                    db_chunk_def = {
                        "chunk_id": rem_chunk_id,
                        "repository_id": repo_metadata['repository_id'], "branch": repo_metadata['branch'], "language": lang_name,
                        "source_blob_hash": repo_metadata['source_blob_hash'],
                        "start_char_offset": rem_sub_start_char_abs, "end_char_offset": rem_sub_end_char_abs,
                        "relative_file_path": file_identifier,
                        "start_line": current_remainder_line_start, "end_line": len(content_lines),
                        "dgraph_node_uid": None, "ast_node_ids": [],
                        "chunk_metadata": {"type": "remainder", "start_line": current_remainder_line_start, "end_line": len(content_lines)},
                        "importance_score": self.weights.get("score_remainder", 0.2), "parent_chunk_id": None
                    }
                    db_chunk_definitions.append(db_chunk_def)

        # Optional: Establish parent_chunk_id for structural chunks based on AST hierarchy
        # This logic would be similar to the original _group_nodes_into_chunks parent linking
        # but using TransientNode.parent_id and matching with dgraph_node_uid of created chunk defs.
        # This is complex and might be better handled by directly using Dgraph hierarchy for navigation.
        # For now, parent_chunk_id for structural chunks is mostly None unless split.

        return db_chunk_definitions


    def _group_nodes_into_chunks(self, content: str, ast_nodes: List[TransientNode], file_identifier: str,
                                 lang_name: str) -> List[TransientChunk]:
        """
        Groups AST nodes into chunks, ensuring 100% file content coverage.
        It first creates high-quality chunks from major structures (classes, functions)
        and then creates fallback chunks for any remaining code.
        """
        if not content: return []

        node_id_map = {node.node_id: node for node in ast_nodes}
        chunk_map: Dict[str, TransientChunk] = {}
        structural_chunks: List[TransientChunk] = []
        chunkable_nodes_set = IngestionConfig.LANGUAGE_SETTINGS[lang_name].get('CHUNKABLE_NODES', set())
        content_lines = content.splitlines(keepends=True)
        covered_lines = [False] * len(content_lines)
        text_splitter = TokenTextSplitter(
            chunk_size=self.max_tokens,
            chunk_overlap=int(self.max_tokens * IngestionConfig.CHUNK_OVERLAP_RATIO),
            tokenizer=self.tokenizer.encode
        )

        # 1. Create high-quality chunks from major structural nodes
        top_level_nodes = sorted(
            [node for node in ast_nodes if node.node_type in chunkable_nodes_set and node.name],
            key=lambda n: n.start_line
        )

        for i, node in enumerate(top_level_nodes):
            node_content = node.text_content
            if node.docstring and node.docstring not in node_content:
                node_content = f"{node.docstring}\n{node_content}"

            tokens = self.count_tokens(node_content)
            chunk_id_base = f"{file_identifier}:struct:{i}"

            if tokens > self.max_tokens:
                sub_chunks_content = text_splitter.split_text(node_content)
                parent_chunk_id = f"{chunk_id_base}:0"
                for sub_idx, sub_text in enumerate(sub_chunks_content):
                    chunk_id = f"{chunk_id_base}:{sub_idx}"
                    chunk = TransientChunk(
                        chunk_id=chunk_id, content=sub_text, ast_node_ids=[node.node_id],
                        importance_score=self.weights.get("score_split_structural", 0.4),
                        metadata={"type": "split_structural"},
                        parent_chunk_id=parent_chunk_id if sub_idx > 0 else None
                    )
                    structural_chunks.append(chunk)
            else:
                chunk = TransientChunk(
                    chunk_id=chunk_id_base, content=node_content, ast_node_ids=[node.node_id],
                    importance_score=self._calculate_importance(node, tokens),
                    metadata={"type": node.node_type, "symbol_name": node.name, "start_line": node.start_line, "end_line": node.end_line}
                )
                structural_chunks.append(chunk)
                chunk_map[node.node_id] = chunk

            for line_num in range(node.start_line - 1, node.end_line):
                if line_num < len(covered_lines): covered_lines[line_num] = True

        # 2. Create fallback "remainder" chunks for code between structural chunks.
        remainder_chunks: List[TransientChunk] = []
        current_remainder_block = ""
        for i, line in enumerate(content_lines):
            if not covered_lines[i]:
                current_remainder_block += line
            elif current_remainder_block:
                for sub_text in text_splitter.split_text(current_remainder_block):
                    chunk_id = f"{file_identifier}:rem:{len(remainder_chunks)}"
                    remainder_chunks.append(TransientChunk(
                        chunk_id=chunk_id, content=sub_text,
                        importance_score=self.weights.get("score_remainder", 0.2),
                        metadata={"type": "remainder"}
                    ))
                current_remainder_block = ""
        if current_remainder_block:
            for sub_text in text_splitter.split_text(current_remainder_block):
                chunk_id = f"{file_identifier}:rem:{len(remainder_chunks)}"
                remainder_chunks.append(TransientChunk(
                    chunk_id=chunk_id, content=sub_text,
                    importance_score=self.weights.get("score_remainder", 0.2),
                    metadata={"type": "remainder"}
                ))

        # 3. Combine chunks and establish parent-child relationships.
        all_chunks = structural_chunks + remainder_chunks
        for chunk in structural_chunks:
            if not chunk.ast_node_ids: continue
            node = node_id_map.get(chunk.ast_node_ids[0])
            if not node or not node.parent_id: continue

            parent_node = node_id_map.get(node.parent_id)
            while parent_node:
                if parent_node.node_id in chunk_map:
                    chunk.parent_chunk_id = chunk_map[parent_node.node_id].chunk_id
                    break
                parent_node = node_id_map.get(parent_node.parent_id) if parent_node.parent_id else None

        return all_chunks

    def _calculate_importance(self, node: TransientNode, token_count: int) -> float:
        """Calculates an importance score for a chunk based on heuristics."""
        score = self.weights.get("base_score", 0.5)
        if node.node_type.endswith(('class_definition', 'class_declaration')):
            score += self.weights.get("type_class", 0.2)
        elif 'function' in node.node_type or 'method' in node.node_type:
            score += self.weights.get("type_function", 0.1)

        if node.docstring:
            score += self.weights.get("has_docstring", 0.3)
        if node.complexity_score and node.complexity_score > self.weights.get("complexity_threshold", 5):
            score += self.weights.get("high_complexity", 0.15)
        if token_count < self.weights.get("short_chunk_threshold", 20):
            score += self.weights.get("is_short", -0.1)
        elif token_count > self.weights.get("long_chunk_threshold", 200):
            score += self.weights.get("is_long", 0.1)

        return max(0.0, min(1.0, score))