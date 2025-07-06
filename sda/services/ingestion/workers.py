# sda/services/ingestion/workers.py

import json
import logging
import os
import multiprocessing as mp
import queue
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple # Added Tuple

# Imports that were originally in ingestion.py and are needed by these workers
import hashlib
from sda.services.chunking import TokenAwareChunker
from sda.core.data_models import TransientNode # TransientChunk might not be needed if chunker changes output
# from sda.core.data_models import TransientNode, TransientChunk # Assuming these are the Pydantic models

# Global chunker instance for parsing worker, similar to original setup
_chunker_instance: Optional[TokenAwareChunker] = None

def _initialize_parsing_worker():
    global _chunker_instance
    pid = os.getpid()
    if _chunker_instance is None:
        logging.info(f"[ParserWorker PID:{pid}] Initializing TokenAwareChunker...")
        _chunker_instance = TokenAwareChunker()

def _persistent_embedding_worker(device: str, model_name: str, cache_folder: str, work_queue: mp.Queue, result_queue: mp.Queue, shutdown_event: mp.Event):
    import torch, gc
    from sentence_transformers import SentenceTransformer
    # Ensure multiprocessing is imported if not already (it was imported in original ingestion.py)
    # import multiprocessing as mp # mp is already an alias from line 4

    pid = os.getpid()
    log_prefix = f"[EmbedWorker PID:{pid} Device:{device}]"
    worker_model = None

    try:
        logging.info(f"{log_prefix} Initializing model...")
        worker_model = SentenceTransformer(model_name_or_path=model_name, device=device, cache_folder=cache_folder)
        worker_model.eval()
        worker_model.half() # Use half precision if supported and beneficial
        logging.info(f"{log_prefix} Model initialization complete.")
    except Exception as e:
        logging.error(f"{log_prefix} Failed to initialize model: {e}", exc_info=True)
        result_queue.put(("error", str(e)))
        return

    result_queue.put(("ready", pid))

    while not shutdown_event.is_set():
        try:
            chunk_batch = work_queue.get(timeout=1.0) # Expects list of (schema, id, content, token_count)
            if chunk_batch is None: break # Shutdown signal

            texts_to_embed = [content for _, _, content, _ in chunk_batch]
            if not texts_to_embed: continue

            embeddings = worker_model.encode(texts_to_embed, show_progress_bar=False, convert_to_tensor=False)
            # Ensure results match the structure expected by _persist_vector_batch
            results = [(d[0], d[1], emb.tolist(), d[3]) for d, emb in zip(chunk_batch, embeddings)]
            result_queue.put(("result", (pid, results)))
        except queue.Empty:
            continue
        except Exception as e:
            logging.error(f"{log_prefix} Error processing batch: {e}", exc_info=True)
            result_queue.put(("error", str(e))) # Communicate error back
        finally:
            # Clean up to free GPU memory
            if 'embeddings' in locals(): del embeddings
            if 'texts_to_embed' in locals(): del texts_to_embed
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()

    logging.info(f"{log_prefix} Shutting down.")

def _create_postgres_payload_format(): # Renamed to avoid conflict if imported directly
    """Helper to define the structure of the postgres payload for partition-specific tables and DBCodeChunks."""
    return {'files': {}, 'nodes': [], 'chunks': []} # 'nodes' for ASTNode, 'chunks' for DBCodeChunk definitions

def _pass1_parse_files_worker(
    file_batch: List[str],
    repo_root_str: str,
    schema_name: str,
    cache_path: Path,
    repo_id: int, # ADDED for CodeBlobs context
    branch: str   # ADDED for CodeBlobs context
) -> Dict[str, Any]: # Return type changed slightly to include blob data
    """
    Worker that processes files, prepares data for CodeBlobs, ASTNodes (Dgraph),
    and DBCodeChunks (Postgres). Saves intermediate results to cache files.
    """
    global _chunker_instance
    if _chunker_instance is None: _initialize_parsing_worker()
    chunker = _chunker_instance
    assert chunker is not None

    # This payload is for partition-specific File and ASTNode tables, and DBCodeChunk definitions
    postgres_payload = _create_postgres_payload_format()

    # Data for Dgraph ASTNodes
    dgraph_node_mutations: List[Dict[str, Any]] = []

    # Data for graph edges
    definitions: List[Tuple[str, str]] = [] # Tuple of (name, node_id)
    unresolved_calls: List[Tuple[str, str]] = [] # Tuple of (caller_node_id, called_name)

    # Data for the new CodeBlobs table (to be aggregated and persisted once per file)
    code_blob_data_list: List[Dict[str, Any]] = []

    for file_path_str in file_batch:
        abs_path = Path(file_path_str)
        # Ensure relative paths are stored in POSIX format (forward slashes)
        relative_path_posix = abs_path.relative_to(repo_root_str).as_posix()

        try:
            content = abs_path.read_text(encoding='utf-8', errors='ignore')
            content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()

            # Prepare data for CodeBlobs table
            code_blob_data_list.append({
                "blob_hash": content_hash,
                "content": content,
                "repository_id": repo_id,
                "branch": branch,
                "file_path": relative_path_posix
            })

            current_repo_metadata = {
                "repository_id": repo_id,
                "branch": branch,
                "source_blob_hash": content_hash # For DBCodeChunk definitions
            }

            # This is a CRITICAL change: TokenAwareChunker needs to be updated
            # to return List[TransientNode] and List[Dict for DBCodeChunk]
            nodes, chunk_definitions_for_db = chunker.process_file_for_hierarchical_storage(
                file_path=abs_path,
                file_content=content,
                file_identifier=relative_path_posix, # Used for node_id generation in TransientNode
                repo_metadata=current_repo_metadata
            )

            # Data for partition-specific File table
            postgres_payload['files'][relative_path_posix] = {
                'line_count': len(content.splitlines()),
                'hash': content_hash,
                'chunk_count': len(chunk_definitions_for_db),
                'blob_hash': content_hash # Link to CodeBlobs
            }

            # TransientNode instances (already have new offset fields, no text_content)
            postgres_payload['nodes'].extend([n.model_dump() for n in nodes])

            # DBCodeChunk definitions (already dicts with source_blob_hash, offsets etc.)
            # Ensure 'relative_file_path' is added here if needed by _persist_chunks_for_schema
            for chunk_def in chunk_definitions_for_db:
                chunk_def['relative_file_path'] = relative_path_posix # Ensure this is available
            postgres_payload['chunks'].extend(chunk_definitions_for_db)


            # Prepare Dgraph mutations from TransientNode instances
            for node in nodes: # node is TransientNode
                dgraph_node_mutations.append({
                    "uid": f"_:{node.node_id}", "dgraph.type": "ASTNode",
                    "node_id": node.node_id,
                    "node_type": node.node_type,
                    "name": node.name,
                    "file_path": node.file_path,
                    "repo_id": str(repo_id),
                    "branch": branch,
                    "startCharOffset": node.start_char_offset,
                    "endCharOffset": node.end_char_offset,
                    "startLine": node.start_line,
                    "endLine": node.end_line,
                    "startCol": node.start_column,
                    "endCol": node.end_column,
                    "parent_id": node.parent_id,
                    "depth": node.depth,
                    "complexity_score": node.complexity_score
                    # Any other scalar fields from TransientNode that should go to Dgraph
                })
                # Logic for definitions and calls (ensure node.name is appropriate)
                if node.node_type in ('function_definition', 'class_definition', 'method_declaration', 'function_declaration') and node.name:
                    definitions.append((node.name, node.node_id))
                if node.node_type in ('call', 'call_expression', 'method_invocation') and node.name:
                    unresolved_calls.append((node.node_id, node.name.split('.')[-1]))

        except Exception as e:
            logging.error(f"Error processing file {file_path_str} in worker {os.getpid()}: {e}", exc_info=True)

    pid = os.getpid()
    pg_file = cache_path / f"pg_{schema_name}_{pid}.json"
    with pg_file.open("w", encoding='utf-8') as f:
        json.dump(postgres_payload, f)

    dgnodes_file = cache_path / f"dgnodes_{schema_name}_{pid}.json"
    dgnodes_file.write_text(json.dumps(dgraph_node_mutations), encoding='utf-8')

    defs_file = cache_path / f"defs_{schema_name}_{pid}.json"
    defs_file.write_text(json.dumps(definitions), encoding='utf-8')

    calls_file = cache_path / f"calls_{schema_name}_{pid}.json"
    calls_file.write_text(json.dumps(unresolved_calls), encoding='utf-8')

    return {
        'schema_name': schema_name,
        'postgres_file': str(pg_file),
        'dgraph_nodes_file': str(dgnodes_file),
        'definitions_file': str(defs_file),
        'calls_file': str(calls_file),
        'code_blob_data_list': code_blob_data_list
    }
