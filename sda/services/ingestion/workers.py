# sda/services/ingestion/workers.py

import json
import logging
import tempfile
import os
import multiprocessing as mp
import queue
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import hashlib
from sda.services.chunking import TokenAwareChunker
from sda.core.data_models import TransientNode

_chunker_instance: Optional[TokenAwareChunker] = None

def _initialize_parsing_worker():
    global _chunker_instance
    pid = os.getpid()
    if _chunker_instance is None:
        logging.info(f"[ParserWorker PID:{pid}] Initializing TokenAwareChunker...")
        _chunker_instance = TokenAwareChunker()

def _persistent_embedding_worker(device: str, model_name: str, cache_folder: str, work_queue: mp.Queue, result_queue: mp.Queue, shutdown_event: mp.Event):
    import torch, gc
    import numpy as np
    from sentence_transformers import SentenceTransformer

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
        embeddings_np = None
        texts_to_embed = None
        try:
            chunk_batch = work_queue.get(timeout=1.0)
            if chunk_batch is None: break

            texts_to_embed = [content for _, _, content, _ in chunk_batch]
            if not texts_to_embed: continue

            embeddings_np = worker_model.encode(texts_to_embed, show_progress_bar=False, convert_to_tensor=False)

            sanitized_embeddings = []
            for emb_array in embeddings_np:
                if np.isnan(emb_array).any():
                    logging.warning(f"{log_prefix} Found NaN in embedding, replacing with zeros. Original embedding (first 5 vals): {emb_array[:5]}")
                    emb_array[np.isnan(emb_array)] = 0.0
                sanitized_embeddings.append(emb_array.tolist())

            results = [(d[0], d[1], emb_list, d[3]) for d, emb_list in zip(chunk_batch, sanitized_embeddings)]
            result_queue.put(("result", (pid, results)))
        except queue.Empty:
            continue
        except Exception as e:
            logging.error(f"{log_prefix} Error processing batch: {e}", exc_info=True)
            result_queue.put(("error", str(e)))
        finally:
            if embeddings_np is not None and 'embeddings_np' in locals(): del embeddings_np
            if texts_to_embed is not None and 'texts_to_embed' in locals(): del texts_to_embed
            gc.collect()
            if device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif device.startswith("xpu") and hasattr(torch, 'xpu') and torch.xpu.is_available():
                torch.xpu.empty_cache()
    logging.info(f"{log_prefix} Shutting down.")

def _create_postgres_payload_format():
    return {'files': {}, 'nodes': [], 'chunks': []}

def _pass1_parse_files_worker(
    file_batch: List[str], repo_root_str: str, schema_name: str, cache_path: Path,
    repo_id: int, branch: str
) -> Dict[str, Any]:
    global _chunker_instance
    if _chunker_instance is None: _initialize_parsing_worker()
    chunker = _chunker_instance
    assert chunker is not None

    postgres_payload = _create_postgres_payload_format()
    dgraph_node_mutations: List[Dict[str, Any]] = []
    definitions: List[Tuple[str, str]] = []
    unresolved_calls: List[Tuple[str, str]] = []
    code_blob_data_list: List[Dict[str, Any]] = []

    for file_path_str in file_batch:
        abs_path = Path(file_path_str)
        relative_path_posix = abs_path.relative_to(repo_root_str).as_posix()
        try:
            content = abs_path.read_text(encoding='utf-8', errors='ignore')
            content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
            code_blob_data_list.append({
                "blob_hash": content_hash, "content": content,
                "repository_id": repo_id, "branch": branch, "file_path": relative_path_posix
            })
            current_repo_metadata = {
                "repository_id": repo_id, "branch": branch, "source_blob_hash": content_hash
            }
            nodes, chunk_definitions_for_db = chunker.process_file_for_hierarchical_storage(
                file_path=abs_path, file_content=content,
                file_identifier=relative_path_posix, repo_metadata=current_repo_metadata
            )
            postgres_payload['files'][relative_path_posix] = {
                'line_count': len(content.splitlines()), 'hash': content_hash,
                'chunk_count': len(chunk_definitions_for_db), 'blob_hash': content_hash
            }
            postgres_payload['nodes'].extend([n.model_dump() for n in nodes])
            for chunk_def in chunk_definitions_for_db:
                chunk_def['relative_file_path'] = relative_path_posix
            postgres_payload['chunks'].extend(chunk_definitions_for_db)
            for node in nodes:
                dgraph_node_mutations.append({
                    "uid": f"_:{node.node_id}", "dgraph.type": "ASTNode", "node_id": node.node_id,
                    "node_type": node.node_type, "name": node.name, "file_path": node.file_path,
                    "repo_id": str(repo_id), "branch": branch,
                    "startCharOffset": node.start_char_offset, "endCharOffset": node.end_char_offset,
                    "startLine": node.start_line, "endLine": node.end_line,
                    "startCol": node.start_column, "endCol": node.end_column,
                    "parent_id": node.parent_id, "depth": node.depth,
                    "complexity_score": node.complexity_score
                })
                if node.node_type in ('function_definition', 'class_definition', 'method_declaration', 'function_declaration') and node.name:
                    definitions.append((node.name, node.node_id))
                if node.node_type in ('call', 'call_expression', 'method_invocation') and node.name:
                    unresolved_calls.append((node.node_id, node.name.split('.')[-1]))
        except Exception as e:
            logging.error(f"Error processing file {file_path_str} in worker {os.getpid()}: {e}", exc_info=True)

    pid = os.getpid()
    pg_file = cache_path / f"pg_{schema_name}_{pid}.json"
    with pg_file.open("w", encoding='utf-8') as f: json.dump(postgres_payload, f)
    dgnodes_file = cache_path / f"dgnodes_{schema_name}_{pid}.json"
    dgnodes_file.write_text(json.dumps(dgraph_node_mutations), encoding='utf-8')
    defs_file = cache_path / f"defs_{schema_name}_{pid}.json"
    defs_file.write_text(json.dumps(definitions), encoding='utf-8')
    calls_file = cache_path / f"calls_{schema_name}_{pid}.json"
    calls_file.write_text(json.dumps(unresolved_calls), encoding='utf-8')
    return {
        'schema_name': schema_name, 'postgres_file': str(pg_file),
        'dgraph_nodes_file': str(dgnodes_file), 'definitions_file': str(defs_file),
        'calls_file': str(calls_file), 'code_blob_data_list': code_blob_data_list
    }

def _batch_process_pdfs_worker(
    pdf_batch_paths: List[str], repo_id: int, branch_name: str, repo_root_str: str,
    db_url: str, mineru_path: str, cache_path: Path, assigned_xpu_id_str: Optional[str] = None
) -> Dict[str, Any]:
    pid = os.getpid()
    batch_results: List[Dict[str, Any]] = []
    from sda.services.pdf_parser import PDFParsingService # Import here due to patch dependency
    from sda.core.db_management import DatabaseManager
    import asyncio
    import uuid

    # This import applies the patch if PYTHONSTARTUP points to it or if it's the first ultralytics import
    # from sda.utils import ultralytics_xpu_patch
    # No, pdf_parser.py imports ultralytics_xpu_patch, which should be enough if env is inherited.

    db_manager_instance = DatabaseManager(db_url=db_url, is_worker=True)
    pdf_parser_instance = PDFParsingService(mineru_path=mineru_path) # Patch should be active when this is init'd
    batch_id = str(uuid.uuid4())[:8]
    log_prefix_batch = f"[PDFBatchWorker PID:{pid} BatchID:{batch_id} XPU_ID:{assigned_xpu_id_str or 'CPU'}]"
    logging.info(f"{log_prefix_batch} Starting processing for batch of {len(pdf_batch_paths)} PDFs.")

    env_for_mineru: Dict[str, str] = {} # Initialize as dict
    mineru_device_cli_arg: Optional[str] = None

    # Set ULTRALYTICS_XPU_PATCH_ENABLED - this is critical for the patch script to activate
    # This environment variable will be inherited by the mineru.EXE subprocess.
    env_for_mineru["SDA_ULTRALYTICS_XPU_PATCH_ENABLED"] = "1"
    logging.info(f"{log_prefix_batch} Setting SDA_ULTRALYTICS_XPU_PATCH_ENABLED=1 for MinerU subprocess.")

    if assigned_xpu_id_str:
        # For ONEAPI_DEVICE_SELECTOR (might be used by underlying oneAPI tools if MinerU calls them)
        env_for_mineru["ONEAPI_DEVICE_SELECTOR"] = assigned_xpu_id_str
        logging.info(f"{log_prefix_batch} Environment for MinerU will include: ONEAPI_DEVICE_SELECTOR={assigned_xpu_id_str}")

        mineru_device_cli_arg = "xpu"
        logging.info(f"{log_prefix_batch} MinerU CLI --device argument will be: {mineru_device_cli_arg}")
    else:
        mineru_device_cli_arg = "cpu"
        logging.info(f"{log_prefix_batch} MinerU CLI --device argument will be: {mineru_device_cli_arg}")
        # No ONEAPI_DEVICE_SELECTOR for CPU

    original_pdf_paths_as_path_obj = [Path(p) for p in pdf_batch_paths]
    try:
        with tempfile.TemporaryDirectory(prefix=f"mineru_batch_{batch_id}_") as temp_dir_str:
            batch_input_temp_dir = Path(temp_dir_str)
            logging.info(f"{log_prefix_batch} Created temp directory for symlinks: {batch_input_temp_dir}")
            valid_original_paths_for_mineru: List[Path] = []
            for original_pdf_path_obj in original_pdf_paths_as_path_obj:
                try:
                    symlink_path = batch_input_temp_dir / original_pdf_path_obj.name
                    symlink_path.symlink_to(original_pdf_path_obj)
                    valid_original_paths_for_mineru.append(original_pdf_path_obj)
                except Exception as e_symlink:
                    logging.error(f"{log_prefix_batch} Failed to create symlink for {original_pdf_path_obj.name}: {e_symlink}", exc_info=True)
                    try:
                        failed_relative_path = original_pdf_path_obj.relative_to(repo_root_str).as_posix()
                    except ValueError:
                        failed_relative_path = str(original_pdf_path_obj)
                    batch_results.append({
                        "file": failed_relative_path, "status": "error_symlinking",
                        "path": str(original_pdf_path_obj), "error_message": str(e_symlink)
                    })
            if not valid_original_paths_for_mineru:
                logging.warning(f"{log_prefix_batch} No valid PDFs to process after symlinking failures.")
            else:
                logging.info(f"{log_prefix_batch} Calling MinerU for directory: {batch_input_temp_dir} containing {len(valid_original_paths_for_mineru)} symlinks.")
                try:
                    parsed_results_list, all_image_blobs = asyncio.run(
                        pdf_parser_instance.parse_pdfs_from_directory_input(
                            original_pdf_paths=valid_original_paths_for_mineru,
                            mineru_input_dir=batch_input_temp_dir,
                            env_override=env_for_mineru, # This now includes SDA_ULTRALYTICS_XPU_PATCH_ENABLED and ONEAPI_DEVICE_SELECTOR
                            mineru_device_arg=mineru_device_cli_arg
                        )
                    )
                    logging.info(f"{log_prefix_batch} MinerU call finished. Processing {len(parsed_results_list)} results. Found {len(all_image_blobs)} unique images.")
                    for file_result in parsed_results_list:
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
                                "file": original_pdf_path_str, "status": "error_path_calculation",
                                "path": original_pdf_path_str, "error_message": str(ve_relpath)
                            })
                            continue
                        if status == "success" and parsed_document_data:
                            logging.info(f"{log_prefix_file} Saving parsed data to database.")
                            doc_uuid = db_manager_instance.save_pdf_document(
                                parsed_document_data=parsed_document_data, image_blobs_data=all_image_blobs,
                                repository_id=repo_id, branch_name=branch_name, relative_path=relative_path
                            )
                            if doc_uuid:
                                logging.info(f"{log_prefix_file} Successfully processed and saved. PDF Document UUID: {doc_uuid}")
                                batch_results.append({"file": relative_path, "status": "success", "uuid": doc_uuid, "path": original_pdf_path_str})
                            else:
                                logging.error(f"{log_prefix_file} Failed to save PDF document to database (doc_uuid is None).")
                                batch_results.append({"file": relative_path, "status": "db_save_failed_no_uuid", "path": original_pdf_path_str, "error_message": "DB save returned no UUID."})
                        else:
                            logging.warning(f"{log_prefix_file} Processing failed for this file. Status: {status}. Error: {error_msg}")
                            batch_results.append({
                                "file": relative_path, "status": status,
                                "path": original_pdf_path_str, "error_message": error_msg
                            })
                except Exception as e_batch_proc:
                    logging.error(f"{log_prefix_batch} Critical error during batch PDF processing (inside MinerU/DB operations): {e_batch_proc}", exc_info=True)
                    for p_path_obj in valid_original_paths_for_mineru:
                        is_already_errored = any(res.get("path") == str(p_path_obj) and res.get("status") != "success" for res in batch_results)
                        if not is_already_errored:
                            try:
                                err_relative_path = p_path_obj.relative_to(repo_root_str).as_posix()
                            except ValueError:
                                err_relative_path = str(p_path_obj)
                            batch_results.append({"file": err_relative_path, "status": "error_batch_execution", "path": str(p_path_obj), "error_message": str(e_batch_proc)})
    except Exception as e_outer:
        logging.error(f"{log_prefix_batch} Unexpected error in outer try block of PDF batch processing: {e_outer}", exc_info=True)
        for pdf_path_str_original_input in pdf_batch_paths:
            is_already_errored = any(res.get("path") == pdf_path_str_original_input and res.get("status") != "success" for res in batch_results)
            if not is_already_errored:
                try:
                    err_relative_path = Path(pdf_path_str_original_input).relative_to(repo_root_str).as_posix()
                except ValueError:
                    err_relative_path = pdf_path_str_original_input
                batch_results.append({
                    "file": err_relative_path, "status": "error_outer_scope",
                    "path": pdf_path_str_original_input, "error_message": str(e_outer)
                })
    logging.info(f"{log_prefix_batch} Finished processing batch. Results count: {len(batch_results)}.")
    return {"worker_id": f"pdf_batch_worker_{pid}_{batch_id}", "results": batch_results}
