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
except ImportError as e:
    logging.error(f"Failed to import tree-sitter languages. Please ensure they are installed. Error: {e}")
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

    def _extract_ast_nodes(self, root_node: Node, file_identifier: str, lang_name: str) -> List[TransientNode]:
        """Recursively traverses the AST to extract relevant nodes based on language settings."""
        nodes: List[TransientNode] = []
        identifier_nodes = IngestionConfig.LANGUAGE_SETTINGS[lang_name].get('IDENTIFIER_NODES', set())
        chunkable_nodes = IngestionConfig.LANGUAGE_SETTINGS[lang_name].get('CHUNKABLE_NODES', set())

        def traverse(node: Node, parent_id: Optional[str] = None, depth: int = 0):
            node_id = f"{file_identifier}:{node.start_byte}-{node.end_byte}"
            node_type = node.type

            if node_type in identifier_nodes:
                name = self._get_node_name(node, lang_name)
                # If we couldn't find a name for a definition, it's not useful for lookups.
                if not name and node_type in chunkable_nodes:
                    pass # Continue traversal but don't add this node
                else:
                    signature = name
                    complexity = None
                    if node_type in chunkable_nodes:
                        complexity = self._calculate_complexity(node, lang_name)
                        params_node = node.child_by_field_name('parameters')
                        if params_node:
                            signature = f"{name}{params_node.text.decode('utf8', errors='ignore')}"

                    if name and len(name) > 512:
                        name = name[:512]

                    docstring = self._extract_docstring(node, lang_name)

                    nodes.append(TransientNode(
                        node_id=node_id, node_type=node_type, name=name,
                        start_line=node.start_point[0] + 1, start_column=node.start_point[1],
                        end_line=node.end_point[0] + 1, end_column=node.end_point[1],
                        text_content=node.text.decode('utf8', errors='ignore'),
                        signature=signature, docstring=docstring,
                        parent_id=parent_id, depth=depth,
                        complexity_score=complexity
                    ))

            # Recurse
            for child in node.children:
                current_node_id = node_id if node_type in identifier_nodes else parent_id
                traverse(child, parent_id=current_node_id, depth=depth + 1)

        traverse(root_node)
        return nodes

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