# sda/services/ingestion/workers.py

import json
import logging
import tempfile # Added missing import
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
    import numpy as np # Import numpy for NaN checking
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

            embeddings_np = worker_model.encode(texts_to_embed, show_progress_bar=False, convert_to_tensor=False)

            # Handle potential NaN values in embeddings
            sanitized_embeddings = []
            for emb_array in embeddings_np:
                if np.isnan(emb_array).any():
                    logging.warning(f"{log_prefix} Found NaN in embedding, replacing with zeros. Original embedding (first 5 vals): {emb_array[:5]}")
                    emb_array[np.isnan(emb_array)] = 0.0
                sanitized_embeddings.append(emb_array.tolist())

            # Ensure results match the structure expected by _persist_vector_batch
            results = [(d[0], d[1], emb_list, d[3]) for d, emb_list in zip(chunk_batch, sanitized_embeddings)]
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


def _batch_process_pdfs_worker(
    pdf_batch_paths: List[str],  # List of absolute paths to PDF files in this batch
    repo_id: int,
    branch_name: str,
    repo_root_str: str, # For calculating relative paths
    db_url: str,
    mineru_path: str,
    cache_path: Path    # Base cache path for this ingestion run (currently unused by this worker but good for future)
) -> Dict[str, Any]:
    """
    Processes a batch of PDF files using MinerU.
    Saves parsed data directly to the database.
    Returns a dictionary containing results for each PDF (success/failure, UUIDs, errors).
    """
    pid = os.getpid()
    batch_results: List[Dict[str, Any]] = []

    # These imports are needed within the worker process.
    from sda.services.pdf_parser import PDFParsingService
    from sda.core.db_management import DatabaseManager
    import asyncio
    import uuid # For batch worker id if needed later

    # Initialize services once per batch worker invocation.
    db_manager_instance = DatabaseManager(db_url=db_url, is_worker=True)
    pdf_parser_instance = PDFParsingService(mineru_path=mineru_path)

    batch_id = str(uuid.uuid4())[:8] # Unique ID for this batch execution
    log_prefix_batch = f"[PDFBatchWorker PID:{pid} BatchID:{batch_id}]"

    logging.info(f"{log_prefix_batch} Starting processing for batch of {len(pdf_batch_paths)} PDFs.")

    # Convert string paths to Path objects for easier manipulation
    original_pdf_paths_as_path_obj = [Path(p) for p in pdf_batch_paths]

    with tempfile.TemporaryDirectory(prefix=f"mineru_batch_{batch_id}_") as temp_dir_str:
        batch_input_temp_dir = Path(temp_dir_str)
        logging.info(f"{log_prefix_batch} Created temp directory for symlinks: {batch_input_temp_dir}")

        # Create symlinks in the temporary directory
        valid_original_paths_for_mineru: List[Path] = []
        for original_pdf_path_obj in original_pdf_paths_as_path_obj:
            try:
                symlink_path = batch_input_temp_dir / original_pdf_path_obj.name
                symlink_path.symlink_to(original_pdf_path_obj)
                valid_original_paths_for_mineru.append(original_pdf_path_obj)
            except Exception as e_symlink:
                logging.error(f"{log_prefix_batch} Failed to create symlink for {original_pdf_path_obj.name}: {e_symlink}", exc_info=True)
                # Add error result for this specific file immediately
                # Need to calculate relative_path here if possible, or use absolute path
                try:
                    failed_relative_path = original_pdf_path_obj.relative_to(repo_root_str).as_posix()
                except ValueError: # If original path is not under repo_root_str (should not happen with absolute paths)
                    failed_relative_path = str(original_pdf_path_obj)
                batch_results.append({
                    "file": failed_relative_path, "status": "error_symlinking",
                    "path": str(original_pdf_path_obj), "error_message": str(e_symlink)
                })

        if not valid_original_paths_for_mineru:
            logging.warning(f"{log_prefix_batch} No valid PDFs to process after symlinking failures.")
            return {"worker_id": f"pdf_batch_worker_{pid}_{batch_id}", "results": batch_results}

        logging.info(f"{log_prefix_batch} Calling MinerU for directory: {batch_input_temp_dir} containing {len(valid_original_paths_for_mineru)} symlinks.")

        try:
            # Call the new batch parsing method in PDFParsingService
            # It expects list of original Path objects (absolute) and the Path to temp dir of symlinks
            parsed_results_list, all_image_blobs = asyncio.run(
                pdf_parser_instance.parse_pdfs_from_directory_input(
                    original_pdf_paths=valid_original_paths_for_mineru, # Pass list of original Path objects
                    mineru_input_dir=batch_input_temp_dir
                )
            )

            logging.info(f"{log_prefix_batch} MinerU call finished. Processing {len(parsed_results_list)} results. Found {len(all_image_blobs)} unique images.")

            # Consolidate image blobs: db_manager_instance.save_pdf_document expects a list of PDFImageBlob objects.
            # The new parser service method already returns a consolidated list of unique image blobs.

            for file_result in parsed_results_list: # file_result is a Dict
                original_pdf_path_str = file_result["original_path"]
                parsed_document_data = file_result["document"]
                status = file_result["status"]
                error_msg = file_result["error_message"]

                current_pdf_path_obj = Path(original_pdf_path_str)
                pdf_filename_for_log = current_pdf_path_obj.name
                log_prefix_file = f"[PDFBatchWorker PID:{pid} BatchID:{batch_id} File:{pdf_filename_for_log}]"

                try:
                    relative_path = current_pdf_path_obj.relative_to(repo_root_str).as_posix()
                except ValueError as ve_relpath:
                    logging.error(f"{log_prefix_file} Error calculating relative path for '{original_pdf_path_str}' against root '{repo_root_str}': {ve_relpath}. Skipping save.")
                    batch_results.append({
                        "file": original_pdf_path_str, # Use original_path if relative_path fails
                        "status": "error_path_calculation",
                        "path": original_pdf_path_str,
                        "error_message": str(ve_relpath)
                    })
                    continue

                if status == "success" and parsed_document_data:
                    logging.info(f"{log_prefix_file} Saving parsed data to database.")
                    doc_uuid = db_manager_instance.save_pdf_document(
                        parsed_document_data=parsed_document_data,
                        image_blobs_data=all_image_blobs,
                        repository_id=repo_id,
                        branch_name=branch_name,
                        relative_path=relative_path
                    )
                    if doc_uuid:
                        logging.info(f"{log_prefix_file} Successfully processed and saved. PDF Document UUID: {doc_uuid}")
                        batch_results.append({"file": relative_path, "status": "success", "uuid": doc_uuid, "path": original_pdf_path_str})
                    else:
                        logging.error(f"{log_prefix_file} Failed to save PDF document to database (doc_uuid is None).")
                        batch_results.append({"file": relative_path, "status": "db_save_failed_no_uuid", "path": original_pdf_path_str, "error_message": "DB save returned no UUID."})
                else:
                    # Handle various failure statuses from parsing service
                    logging.warning(f"{log_prefix_file} Processing failed for this file. Status: {status}. Error: {error_msg}")
                    batch_results.append({
                        "file": relative_path,
                        "status": status, # e.g., "json_not_found", "json_parse_error"
                        "path": original_pdf_path_str,
                        "error_message": error_msg
                    })

        except Exception as e_batch_proc:
            logging.error(f"{log_prefix_batch} Critical error during batch PDF processing: {e_batch_proc}", exc_info=True)
            # This is an error for the whole batch operation. We might not have individual file results.
            # Add a generic error for all files that were attempted in this batch if not already individually handled.
            for p_path_obj in valid_original_paths_for_mineru:
                # Avoid adding duplicate errors if some were already logged (e.g. symlink errors)
                # A simple check:
                is_already_errored = any(res.get("path") == str(p_path_obj) and res.get("status") != "success" for res in batch_results)
                if not is_already_errored:
                    try:
                        err_relative_path = p_path_obj.relative_to(repo_root_str).as_posix()
                    except ValueError:
                        err_relative_path = str(p_path_obj)
                    batch_results.append({"file": err_relative_path, "status": "error_batch_execution", "path": str(p_path_obj), "error_message": str(e_batch_proc)})

    # TemporaryDirectory is cleaned up automatically here.
    logging.info(f"{log_prefix_batch} Finished processing batch. Results count: {len(batch_results)}.")
    return {"worker_id": f"pdf_batch_worker_{pid}_{batch_id}", "results": batch_results}
