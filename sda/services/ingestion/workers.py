# sda/services/ingestion/workers.py

import json
import logging
import os
import queue
from pathlib import Path
from typing import List, Dict, Any, Optional

# Imports that were originally in ingestion.py and are needed by these workers
import hashlib
from sda.services.chunking import TokenAwareChunker
from sda.core.data_models import TransientNode, TransientChunk # Assuming these are the Pydantic models

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
    import multiprocessing as mp

    pid = os.getpid()
    log_prefix = f"[EmbedWorker PID:{pid} Device:{device}]"
    worker_model = None

    try:
        logging.info(f"{log_prefix} Initializing model...")
        worker_model = SentenceTransformer(model_name_or_path=model_name, device=device, cache_folder=cache_folder)
        worker_model.eval()
        worker_model.half()
        logging.info(f"{log_prefix} Model initialization complete.")
    except Exception as e:
        logging.error(f"{log_prefix} Failed to initialize model: {e}", exc_info=True)
        result_queue.put(("error", str(e)))
        return

    result_queue.put(("ready", pid))

    while not shutdown_event.is_set():
        try:
            chunk_batch = work_queue.get(timeout=1.0)
            if chunk_batch is None: break

            texts_to_embed = [content for _, _, content, _ in chunk_batch]
            embeddings = worker_model.encode(texts_to_embed, show_progress_bar=False, convert_to_tensor=False)
            results = [(d[0], d[1], emb.tolist(), d[3]) for d, emb in zip(chunk_batch, embeddings)]
            result_queue.put(("result", (pid, results)))
        except queue.Empty:
            continue
        except Exception as e:
            logging.error(f"{log_prefix} Error processing batch: {e}", exc_info=True)
            result_queue.put(("error", str(e)))
        finally:
            if 'embeddings' in locals(): del embeddings
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()

    logging.info(f"{log_prefix} Shutting down.")

def _create_postgres_payload_format(): # Renamed to avoid conflict if imported directly
    """Helper to define the structure of the postgres payload."""
    return {'files': {}, 'nodes': [], 'chunks': []}

def _pass1_parse_files_worker(file_batch: List[str], repo_root_str: str, schema_name: str, cache_path: Path) -> Dict[str, str]:
    """Worker that processes files and saves results to cache files to avoid memory overflow."""
    global _chunker_instance
    if _chunker_instance is None: _initialize_parsing_worker()
    chunker = _chunker_instance
    assert chunker is not None

    postgres_payload = _create_postgres_payload_format()
    dgraph_node_mutations, definitions, unresolved_calls = [], [], []

    for file_path_str in file_batch:
        abs_path = Path(file_path_str)
        relative_path_str = str(abs_path.relative_to(repo_root_str))
        try:
            content = abs_path.read_text(encoding='utf-8', errors='ignore')
            # Assuming TransientNode and TransientChunk are Pydantic models now
            nodes, chunks = chunker.process_file(abs_path, content, file_identifier=relative_path_str)

            postgres_payload['files'][relative_path_str] = {
                'line_count': len(content.splitlines()),
                'hash': hashlib.sha256(content.encode('utf-8')).hexdigest(),
                'chunk_count': len(chunks)
            }
            # Use .model_dump() for Pydantic models
            postgres_payload['nodes'].extend([n.model_dump() for n in nodes])
            postgres_payload['chunks'].extend([c.model_dump() for c in chunks])

            for node in nodes: # node is TransientNode
                dgraph_node_mutations.append({
                    "uid": f"_:{node.node_id}", "dgraph.type": "ASTNode",
                    "node_id": node.node_id, "node_type": node.node_type, "name": node.name,
                    "file_path": node.node_id.split(':', 1)[0]
                })
                if node.node_type in ('function_definition', 'class_definition', 'method_declaration') and node.name:
                    definitions.append((node.name, node.node_id))
                if node.node_type in ('call', 'call_expression', 'method_invocation') and node.name:
                    unresolved_calls.append((node.node_id, node.name.split('.')[-1]))
        except Exception as e:
            logging.error(f"Error processing file {file_path_str} in worker: {e}", exc_info=True)

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
        'calls_file': str(calls_file)
    }
