# sda/services/ingestion.py

import concurrent.futures
import hashlib
import json
import logging
import multiprocessing as mp
import os
import queue
import shutil
import traceback
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import List, Callable, Optional, Tuple, Dict, Any, Generator

import tiktoken
# Removed: from sqlalchemy.dialects.postgresql import insert # Not used directly here anymore

from sda.config import IngestionConfig, AIConfig, DATA_DIR, INGESTION_CACHE_DIR
from sda.core.db_management import DatabaseManager
from sda.core.models import Repository, File, ASTNode, DBCodeChunk, BillingUsage, Task, CodeBlob # Added CodeBlob
from sda.services.analysis import resolve_embedding_devices
# from sda.services.chunking import TokenAwareChunker # No longer directly needed here
from sda.services.git_integration import GitService
from sda.services.partitioning import SmartPartitioningService
from sda.utils.task_executor import TaskExecutor

# Imports from the new ingestion submodules
from .workers import _initialize_parsing_worker, _persistent_embedding_worker, _pass1_parse_files_worker, _process_single_pdf_worker # Added _process_single_pdf_worker
# _chunker_instance is defined and used within workers.py

from sda.services.pdf_parser import PDFParsingService # For type hinting

from .persistence import (
    _aggregate_postgres_payloads, _persist_files_for_schema, _persist_nodes_for_schema,
    _persist_chunks_for_schema, _persist_dgraph_nodes, _persist_vector_batch,
    _process_graph_edges_streaming, _load_and_merge_json_files,
    _create_postgres_payload, _create_batches,
    _persist_code_blobs # NEW IMPORT
)
from .utils import _stream_chunks_from_files, _stream_batcher


StatusUpdater = Callable[[int, str, float, Optional[str], Optional[Dict[str, Any]]], None]


class IntelligentIngestionService:
    def __init__(self, db_manager: DatabaseManager, git_service: GitService,
                 task_executor: TaskExecutor, partitioning_service: SmartPartitioningService,
                 pdf_parsing_service: PDFParsingService): # Added pdf_parsing_service
        self.db_manager = db_manager
        self.git_service = git_service
        self.task_executor = task_executor
        self.partitioning_service = partitioning_service
        self.pdf_parsing_service = pdf_parsing_service # Store it
        self.tokenizer = tiktoken.get_encoding(IngestionConfig.TOKENIZER_MODEL)
        self.embedding_config = AIConfig.get_active_embedding_config()
        logging.info("IntelligentIngestionService initialized with TaskExecutor, SmartPartitioningService, and PDFParsingService.")

    def _setup_embedding_workers(self) -> Tuple[List[mp.Process], List[mp.Queue], mp.Queue, mp.Event]:
        devices = resolve_embedding_devices()
        num_workers = min(len(devices), AIConfig.MAX_EMBEDDING_WORKERS)
        devices_to_use = devices[:num_workers] or ["cpu"] # Default to CPU if no GPUs
        work_queues, workers = [], []
        result_queue = mp.Queue()
        shutdown_event = mp.Event()
        for i, device in enumerate(devices_to_use):
            work_queue = mp.Queue()
            # Corrected arguments for _persistent_embedding_worker based on its definition
            worker = mp.Process(target=_persistent_embedding_worker,
                                args=(device, self.embedding_config.model_name, str(DATA_DIR / "embedding_models"),
                                      work_queue, result_queue, shutdown_event),
                                name=f"EmbeddingWorker-{i}-{device}")
            workers.append(worker)
            work_queues.append(work_queue)
            worker.start()

        ready_count = 0
        # Wait for all workers to signal readiness or error
        for _ in range(len(workers)): # Expect one message per worker
            try:
                msg_type, data = result_queue.get(timeout=120.0) # Increased timeout
                if msg_type == "ready":
                    ready_count += 1
                    logging.info(f"Embedding worker {data} is ready.")
                elif msg_type == "error":
                    # If one worker fails, it might be better to raise immediately
                    raise RuntimeError(f"Embedding worker failed to initialize: {data}")
            except queue.Empty:
                raise TimeoutError("Timed out waiting for embedding workers to initialize.")

        if ready_count != len(workers):
             raise RuntimeError(f"Not all embedding workers initialized successfully. Expected {len(workers)}, got {ready_count}")
        logging.info(f"All {len(workers)} embedding workers are ready.")
        return workers, work_queues, result_queue, shutdown_event

    def _shutdown_embedding_workers(self, workers, work_queues, shutdown_event):
        logging.info("Signaling embedding workers to shut down...")
        shutdown_event.set()
        for wq in work_queues:
            try:
                wq.put(None, block=False) # Non-blocking put
            except queue.Full:
                logging.warning("Work queue full during shutdown signal, worker might miss None.")

        for i, w in enumerate(workers):
            w.join(timeout=15) # Increased join timeout
            if w.is_alive():
                logging.warning(f"Embedding worker {i} did not shut down cleanly, terminating.")
                w.terminate()
        logging.info("Embedding workers shut down process complete.")


    def ingest_repository(self, repo_path_str: str, repo_uuid: str, branch: str, repo_id: int, parent_task_id: int, _framework_start_task: Callable,
                          _framework_update_task: StatusUpdater, _framework_complete_task: Callable):
        cache_path = Path(INGESTION_CACHE_DIR) / f"{repo_uuid}_{branch.replace('/', '_')}"
        if cache_path.exists(): shutil.rmtree(cache_path)
        cache_path.mkdir(parents=True, exist_ok=True)

        workers, work_queues, result_queue, shutdown_event = None, None, None, None
        all_code_blob_data_from_workers: List[Dict[str, Any]] = [] # Define here for finally block

        try:
            # --- PREP ---
            _framework_update_task(parent_task_id, "Preparing environment...", 2.0, None, None)
            if not self.git_service.checkout(repo_path_str, branch): raise RuntimeError(f"Could not checkout branch '{branch}'.")

            _framework_update_task(parent_task_id, "Clearing old data...", 3.0, None, None)
            with self.db_manager.get_session("public") as public_session:
                logging.info(f"Clearing old CodeBlob data from public schema for repo_id: {repo_id}, branch: {branch}")
                public_session.query(CodeBlob).filter(
                    CodeBlob.repository_id == repo_id, # Assuming CodeBlob has these fields
                    CodeBlob.branch == branch
                ).delete(synchronize_session=False)

                logging.info(f"Clearing old DBCodeChunk data from public schema for repo_id: {repo_id}, branch: {branch}")
                public_session.query(DBCodeChunk).filter(
                    DBCodeChunk.repository_id == repo_id,
                    DBCodeChunk.branch == branch
                ).delete(synchronize_session=False)
                public_session.commit()

            self.db_manager.clear_dgraph_data_for_branch(repo_id, branch)

            with self.db_manager.get_session("public") as s:
                repo = s.get(Repository, repo_id)
                if repo and repo.uuid: # repo.uuid should be same as repo_uuid argument
                    logging.info(f"Wiping individual partition schemas for repo_uuid: {repo.uuid}")
                    self.db_manager._wipe_repo_schemas(repo.uuid) # This drops 'repo_xyz_%' schemas

            _framework_update_task(parent_task_id, "Scanning files...", 4.0, None, None)

            all_files_in_repo = [p for p in Path(repo_path_str).rglob('*') if p.is_file() and not any(
                d in p.parts for d in IngestionConfig.DEFAULT_IGNORE_DIRS)]

            pdf_files_to_process: List[Path] = []
            code_files_for_partitioning: List[Path] = []

            for p in all_files_in_repo:
                if p.suffix.lower() == '.pdf':
                    pdf_files_to_process.append(p)
                elif p.suffix in IngestionConfig.LANGUAGE_MAPPING:
                    code_files_for_partitioning.append(p)

            logging.info(f"Discovered {len(code_files_for_partitioning)} code files and {len(pdf_files_to_process)} PDF files for ingestion.")
            _framework_update_task(parent_task_id, "Scanning files complete.", 5.0, None, {'Code Files Found': len(code_files_for_partitioning), 'PDF Files Found': len(pdf_files_to_process)})

            if not code_files_for_partitioning and not pdf_files_to_process:
                _framework_complete_task(parent_task_id, result={"message": "No relevant files (code or PDF) to ingest."}); return

            pdf_processing_futures = []
            if pdf_files_to_process:
                _framework_update_task(parent_task_id, "Submitting PDF processing tasks...", 6.0, None, None)
                for pdf_path_obj in pdf_files_to_process:
                    relative_pdf_path = pdf_path_obj.relative_to(repo_path_str).as_posix()
                    pdf_processing_futures.append(
                        self.task_executor.submit(
                            'postgres', # Using a generic pool name, actual execution is local.
                            _process_single_pdf_worker,
                            str(pdf_path_obj.resolve()),
                            repo_id,
                            branch,
                            relative_pdf_path,
                            self.db_manager.db_url,
                            self.pdf_parsing_service.mineru_path
                        )
                    )
                _framework_update_task(parent_task_id, f"{len(pdf_files_to_process)} PDF tasks submitted.", 7.0, None, None)

            _framework_update_task(parent_task_id, "Analyzing repository structure (for code files)...", 8.0, None, {'Code Files to Analyze': len(code_files_for_partitioning)})

            if not code_files_for_partitioning: # Only PDFs were found and submitted
                logging.info("No code files found for partitioning. Waiting for PDF tasks to complete.")
                if pdf_processing_futures:
                    _framework_update_task(parent_task_id, "Waiting for PDF processing tasks...", 8.5, None, None)
                    # Wait for PDF processing to finish and check for errors
                    successful_pdfs = 0
                    for future in concurrent.futures.as_completed(pdf_processing_futures):
                        try:
                            future.result() # Raise exception if PDF processing failed
                            successful_pdfs += 1
                        except Exception as e:
                            logging.error(f"A PDF processing task failed: {e}", exc_info=True)
                            # Optionally, update parent task to reflect partial failure or log specific PDF
                    _framework_update_task(parent_task_id, f"PDF processing finished ({successful_pdfs}/{len(pdf_files_to_process)} successful).", 90.0, None, None)

                # Finalize task for PDF-only ingestion
                with self.db_manager.get_session("public") as session:
                    repo_obj_final = session.get(Repository, repo_id)
                    if repo_obj_final:
                        repo_obj_final.active_branch = branch # Ensure active branch is set
                        repo_obj_final.last_scanned = datetime.utcnow()
                        # repo_obj_final.db_schemas might be empty or set if we decide to create a schema for PDFs later
                _framework_complete_task(parent_task_id, result={"message": f"PDF processing complete. Processed {len(pdf_files_to_process)} PDFs. No code files."})
                return # Exit ingestion if only PDFs were processed.

            # Proceed with code file partitioning if code_files_for_partitioning is not empty
            file_to_schema_map = self.partitioning_service.generate_schema_map(repo_path_str, code_files_for_partitioning, repo_uuid)
            schema_to_files_map = defaultdict(list)
            for file_path_obj, schema_name_val in file_to_schema_map.items(): # file_path_obj is Path
                schema_to_files_map[schema_name_val].append(str(file_path_obj))

            _framework_update_task(parent_task_id, "Initializing workers...", 10.0, None, {'Partitions Created': len(schema_to_files_map)})
            workers, work_queues, result_queue, shutdown_event = self._setup_embedding_workers()

            # --- STAGE 0: PARALLEL CPU-BOUND FILE PARSING ---
            with concurrent.futures.ProcessPoolExecutor(max_workers=IngestionConfig.MAX_CPU_WORKERS, initializer=_initialize_parsing_worker) as cpu_executor:
                _framework_update_task(parent_task_id, "Parsing files...", 12.0,
                                       f"Processing {len(files_on_disk)} files across {len(schema_to_files_map)} schemas.", None)
                pass1_futures = []
                sorted_schemas = sorted(schema_to_files_map.items(), key=lambda item: len(item[1]), reverse=True)

                for schema_name, files_in_schema in sorted_schemas:
                    files_in_schema.sort(key=lambda f_path: Path(f_path).stat().st_size, reverse=True) # f_path is str
                    for batch in _create_batches(files_in_schema, IngestionConfig.FILE_PROCESSING_BATCH_SIZE):
                        pass1_futures.append(cpu_executor.submit(_pass1_parse_files_worker, batch, repo_path_str, schema_name, cache_path, repo_id, branch))

                schema_cache_files = defaultdict(list) # Maps schema_name to list of its postgres_file cache paths
                dgraph_files, definitions_files, calls_files = [], [], []
                # all_code_blob_data_from_workers defined outside try for finally block access

                for i, future_obj in enumerate(concurrent.futures.as_completed(pass1_futures)): # Renamed f to future_obj
                    try:
                        res = future_obj.result()
                        s_name = res['schema_name']
                        schema_cache_files[s_name].append(res['postgres_file'])
                        dgraph_files.append(res['dgraph_nodes_file'])
                        definitions_files.append(res['definitions_file'])
                        calls_files.append(res['calls_file'])
                        if 'code_blob_data_list' in res:
                            all_code_blob_data_from_workers.extend(res['code_blob_data_list'])
                    except Exception as e:
                        logging.error(f"Failed to get result from parsing future: {e}", exc_info=True)
                    progress = 12.0 + (i / len(pass1_futures) * 23.0)
                    _framework_update_task(parent_task_id, f"Parsing file batches...", progress, None, {'Parsed Batches': f"{i + 1}/{len(pass1_futures)}"})

            # --- NEW STAGE: PERSIST CODE BLOBS ---
            _framework_update_task(parent_task_id, "Persisting unique file contents (CodeBlobs)...", 35.0, None, {'Blobs Found': len(all_code_blob_data_from_workers)})
            if all_code_blob_data_from_workers:
                _persist_code_blobs(self.db_manager, all_code_blob_data_from_workers, repo_id, branch) # Pass repo_id, branch for context
            _framework_update_task(parent_task_id, "CodeBlobs persistence complete.", 40.0, None, None)


            # --- STAGES 1-4: PARALLEL PERSISTENCE OF METADATA ---
            _framework_update_task(parent_task_id, "Aggregating & Persisting metadata...", 40.0, "Creating database schemas and aggregating data.", None)
            all_schema_payloads = {} # schema_name -> aggregated_payload_dict
            for schema_name_key, pg_cache_files in schema_cache_files.items():
                all_schema_payloads[schema_name_key] = _aggregate_postgres_payloads(pg_cache_files)

            schema_creation_futures = [self.task_executor.submit('postgres', self.db_manager.create_schema_and_tables, s_name) for s_name in
                                       all_schema_payloads.keys()]
            concurrent.futures.wait(schema_creation_futures)

            _framework_update_task(parent_task_id, "Persisting file records...", 45.0, f"Submitting {len(all_schema_payloads)} schemas for file persistence.", None)
            file_id_maps: Dict[str, Dict[str, int]] = {} # schema_name -> {relative_path: file_id}
            file_futures_map = { # future -> schema_name
                self.task_executor.submit('postgres', _persist_files_for_schema, self.db_manager, sch, payld, repo_id, branch, repo_path_str): sch
                for sch, payld in all_schema_payloads.items()}
            for i, f_comp in enumerate(concurrent.futures.as_completed(file_futures_map)):
                s_name_done = file_futures_map[f_comp]
                file_id_maps[s_name_done] = f_comp.result() # result is {relative_path: file_id}
                _framework_update_task(parent_task_id, f"Persisting file records for schema {s_name_done}...", 45.0 + (i / len(file_futures_map) * 5.0), None, None)

            _framework_update_task(parent_task_id, "Persisting AST nodes and code chunks...", 50.0, "Submitting node and chunk persistence tasks.", None)
            node_persistence_futures = [self.task_executor.submit('postgres', _persist_nodes_for_schema, self.db_manager, sch, payld, file_id_maps.get(sch, {}), branch)
                                        for sch, payld in all_schema_payloads.items()]

            chunk_persistence_futures_map = { # future -> schema_name
                self.task_executor.submit('postgres', _persist_chunks_for_schema, self.db_manager, sch, payld, file_id_maps.get(sch, {}), repo_id, branch,
                                          cache_path, self.tokenizer): sch for sch, payld in all_schema_payloads.items()}

            embed_files_to_process, total_db_chunks = [], 0
            for i, f_comp_chunk in enumerate(concurrent.futures.as_completed(chunk_persistence_futures_map)):
                s_name_done_chunk = chunk_persistence_futures_map[f_comp_chunk]
                if res_chunk := f_comp_chunk.result(): # result is (path_to_embed_file, count_of_chunks)
                    path_val, count_val = res_chunk
                    embed_files_to_process.append(path_val)
                    total_db_chunks += count_val
                _framework_update_task(parent_task_id, f"Persisting chunks for schema {s_name_done_chunk}...", 50.0 + (i / len(chunk_persistence_futures_map) * 10.0), None, None)

            concurrent.futures.wait(node_persistence_futures)

            _framework_update_task(parent_task_id, "Metadata persistence complete.", 60.0, "AST, Chunks, and File records saved.",
                                   {'Chunks for Embedding': total_db_chunks})

            # --- STAGES 5 & 6: PARALLEL GRAPH AND VECTOR PROCESSING ---
            _framework_update_task(parent_task_id, "Processing graph and vector embeddings...", 60.0, "Starting parallel data processing.", None)
            graph_task = _framework_start_task(repo_id, "Graph Processing", "system", parent_task_id)
            vector_task = _framework_start_task(repo_id, "Vector Embedding", "system", parent_task_id)

            # _run_graph_processing_task and _run_vector_embedding_task definitions (largely unchanged from original, ensure parameters passed are correct)
            # ... (definitions for _run_graph_processing_task and _run_vector_embedding_task) ...
            # Ensure they use the correct variables like dgraph_files, definitions_files, calls_files, embed_files_to_process, total_db_chunks

            def _run_graph_processing_task_local(current_task: Task): # Renamed to avoid outer scope issues if any
                try:
                    _framework_update_task(current_task.id, "Loading graph data from cache...", 5.0, None, None)
                    dgraph_nodes_data_local = _load_and_merge_json_files(dgraph_files)
                    logging.info(f"Loaded {len(dgraph_nodes_data_local)} dgraph nodes from {len(dgraph_files)} cache files for task {current_task.id}")

                    if dgraph_nodes_data_local:
                        dgraph_nodes_data_local.sort(key=lambda x: x.get('node_id', '')) # Robust sort
                        nodes_persisted_val = _persist_dgraph_nodes(self.db_manager, repo_id, branch, dgraph_nodes_data_local, current_task.id, _framework_update_task)
                        del dgraph_nodes_data_local; import gc; gc.collect()
                    else:
                        nodes_persisted_val = 0
                        _framework_update_task(current_task.id, "No graph nodes found to persist.", 50.0, None, {'Nodes Persisted': '0'})

                    _framework_update_task(current_task.id, "Processing graph edges...", 50.0, f"Persisted {nodes_persisted_val} graph nodes.", {'Nodes Persisted': nodes_persisted_val})
                    edges_persisted_val = _process_graph_edges_streaming(self.db_manager, definitions_files, calls_files, current_task.id, _framework_update_task)

                    final_msg = "Graph processing complete."
                    _framework_update_task(current_task.id, final_msg, 100.0, final_msg, {'Nodes Persisted': nodes_persisted_val, 'Edges Persisted': edges_persisted_val})
                    _framework_complete_task(current_task.id, result={'nodes_persisted': nodes_persisted_val, 'edges_persisted': edges_persisted_val})
                except Exception as e_graph:
                    logging.error(f"Graph processing task {current_task.id} failed: {e_graph}", exc_info=True)
                    _framework_complete_task(current_task.id, error=traceback.format_exc())

            def _run_vector_embedding_task_local(current_task: Task): # Renamed
                try:
                    if not embed_files_to_process or total_db_chunks == 0:
                        _framework_update_task(current_task.id, "No content to embed.", 100.0, None, {'Total Chunks': 0})
                        _framework_complete_task(current_task.id, result={"message": "No content to embed."}); return

                    _framework_update_task(current_task.id, f"Preparing to process {total_db_chunks} chunks...", 5.0, None,
                                           {'Total Chunks': total_db_chunks, 'Embed Files': len(embed_files_to_process)})

                    processed_db_chunks, total_calc_tokens = 0, 0
                    vector_persistence_futures = []
                    MAX_PENDING_FUTURES_VEC = AIConfig.MAX_EMBEDDING_WORKERS * 2 # Limit pending persistence
                    batches_sent_to_embed_workers = 0

                    _framework_update_task(current_task.id, "Starting streaming embedding process...", 10.0, None, {'Max Pending Persistence': MAX_PENDING_FUTURES_VEC})

                    # Corrected: _stream_chunks_from_files yields dicts, _stream_batcher batches them
                    # _persistent_embedding_worker expects list of (schema, id, content, token_count)
                    # The .jsonl files from _persist_chunks_for_schema contain this format.

                    chunk_data_stream = _stream_chunks_from_files(embed_files_to_process) # This yields dicts from jsonl

                    for chunk_batch_for_embedding_worker in _stream_batcher(chunk_data_stream, IngestionConfig.EMBEDDING_BATCH_SIZE):
                        # Convert dicts to tuples for worker
                        worker_input_batch = [(item['schema'], item['id'], item['content'], item['token_count']) for item in chunk_batch_for_embedding_worker]
                        if not worker_input_batch: continue

                        work_queues[batches_sent_to_embed_workers % len(work_queues)].put(worker_input_batch)
                        batches_sent_to_embed_workers += 1

                        try:
                            while True: # Process all available results before sending more work or if queue is filling
                                msg_type, msg_data = result_queue.get(block=False) # Non-blocking check
                                if msg_type == "result":
                                    _, batch_embedding_results = msg_data # batch_embedding_results is list of (schema, id, embedding, token_count)
                                    # batch_embedding_results.sort(key=lambda x: x[1]) # Sort by chunk_id (db id)

                                    if len(vector_persistence_futures) >= MAX_PENDING_FUTURES_VEC:
                                        num_to_complete = len(vector_persistence_futures) - MAX_PENDING_FUTURES_VEC + 1
                                        for f_done in concurrent.futures.as_completed(vector_persistence_futures, timeout=300): # Wait for some to complete
                                            try: total_calc_tokens += f_done.result()
                                            except Exception as e_pers: logging.error(f"Vector persistence task failed: {e_pers}")
                                            num_to_complete -=1
                                            if num_to_complete <= 0: break
                                        vector_persistence_futures = [f_p for f_p in vector_persistence_futures if not f_p.done()]

                                    future_p = self.task_executor.submit('postgres', _persist_vector_batch, self.db_manager, batch_embedding_results)
                                    vector_persistence_futures.append(future_p)
                                    processed_db_chunks += len(batch_embedding_results)

                                    prog_pct = 15.0 + ((processed_db_chunks / total_db_chunks) * 70.0) if total_db_chunks > 0 else 85.0
                                    _framework_update_task(current_task.id, "Streaming embedding process...", prog_pct, None,
                                                           {'Chunks Processed': f"{processed_db_chunks}/{total_db_chunks}", 'Batches Sent': batches_sent_to_embed_workers,
                                                            'Pending Persistence': len(vector_persistence_futures), 'Tokens Processed (est)': total_calc_tokens })
                                elif msg_type == "error":
                                    logging.error(f"Embedding worker error: {msg_data}")
                                    # Decide if to continue or halt
                                    break # from inner while true
                                if result_queue.empty(): break # No more results for now
                        except queue.Empty: pass # No results ready yet

                        if batches_sent_to_embed_workers % 100 == 0: # Less frequent update
                             _framework_update_task(current_task.id, f"Embedding batches sent: {batches_sent_to_embed_workers}", 15.0, None, {'Batches Sent': batches_sent_to_embed_workers})

                    logging.info(f"Finished sending {batches_sent_to_embed_workers} batches to embedding workers for task {current_task.id}")
                    _framework_update_task(current_task.id, "Processing remaining embeddings...", 85.0, None, {'Final Batches': batches_sent_to_embed_workers})

                    # Signal workers to finish processing their current queues then stop by sending None after loop
                    # This is handled by _shutdown_embedding_workers

                    # Collect remaining results
                    while processed_db_chunks < total_db_chunks or any(not wq.empty() for wq in work_queues) or not result_queue.empty():
                        try:
                            msg_type, msg_data = result_queue.get(timeout=30.0) # Longer timeout for final results
                            if msg_type == "result":
                                _, batch_embedding_results = msg_data
                                if not batch_embedding_results: continue
                                # batch_embedding_results.sort(key=lambda x: x[1])
                                future_p_final = self.task_executor.submit('postgres', _persist_vector_batch, self.db_manager, batch_embedding_results)
                                vector_persistence_futures.append(future_p_final)
                                processed_db_chunks += len(batch_embedding_results)
                                prog_pct_final = 85.0 + ((processed_db_chunks / total_db_chunks) * 10.0) if total_db_chunks > 0 else 95.0
                                _framework_update_task(current_task.id, "Finalizing embeddings...", prog_pct_final, None,
                                                       {'Chunks Processed': f"{processed_db_chunks}/{total_db_chunks}", 'Pending Persistence': len(vector_persistence_futures)})
                            elif msg_type == "error":
                                logging.error(f"Embedding worker error during final collection: {msg_data}")
                                # Potentially break or log and continue to try to get other results
                        except queue.Empty:
                            logging.warning(f"Timeout waiting for final embedding results for task {current_task.id}. Processed {processed_db_chunks}/{total_db_chunks}.")
                            # Check if all workers are done if queues are empty
                            if all(wq.empty() for wq in work_queues) and result_queue.empty():
                                break # Exit if everything seems processed or stuck

                    _framework_update_task(current_task.id, "Waiting for final persistence completion...", 95.0, None, {'Remaining Tasks': len(vector_persistence_futures)})
                    for f_done_final in concurrent.futures.as_completed(vector_persistence_futures):
                        try: total_calc_tokens += f_done_final.result()
                        except Exception as e_pers_final: logging.error(f"Final vector persistence task failed: {e_pers_final}")

                    cost_val = (total_calc_tokens / 1_000_000) * self.embedding_config.price_per_million_tokens

                    # For local embedding models, there's no direct API key.
                    # We'll use the provider name for the hash, or a placeholder if needed.
                    # The EmbeddingConfig model doesn't have an 'api_key' attribute.
                    api_key_source_for_hash = self.embedding_config.provider # e.g., "local"
                    if not api_key_source_for_hash: # Should always have a provider
                        api_key_source_for_hash = "unknown_embedding_provider"

                    usage = BillingUsage(
                        model_name=self.embedding_config.model_name,
                        provider=self.embedding_config.provider,
                        api_key_used_hash=hashlib.sha256(api_key_source_for_hash.encode()).hexdigest(),
                        total_tokens=total_calc_tokens,
                        cost=cost_val
                    )
                    with self.db_manager.get_session("public") as s: s.add(usage)

                    final_dets = {'Total Cost': f"${cost_val:.4f}", 'Tokens Processed': total_calc_tokens, 'Chunks Processed': f"{processed_db_chunks}/{total_db_chunks}"}
                    _framework_update_task(current_task.id, "Vector embedding complete.", 100.0, None, final_dets)
                    _framework_complete_task(current_task.id, result={"message": f"Embedding complete. Total cost: ${cost_val:.4f}", 'tokens': total_calc_tokens, 'cost': cost_val})
                except Exception as e_vec:
                    logging.error(f"Vector embedding task {current_task.id} failed: {e_vec}", exc_info=True)
                    _framework_complete_task(current_task.id, error=traceback.format_exc())


            logging.info(f"Submitting graph processing task {graph_task.id} and vector embedding task {vector_task.id} to TaskExecutor")
            graph_future = self.task_executor.submit('dgraph', _run_graph_processing_task_local, graph_task)
            vector_future = self.task_executor.submit('postgres', _run_vector_embedding_task_local, vector_task)

            logging.info("Waiting for graph and vector processing tasks to complete...")
            # Wait for both tasks with timeout and better error handling
            all_sub_tasks = [graph_future, vector_future]
            done_tasks, not_done_tasks = concurrent.futures.wait(all_sub_tasks, timeout=IngestionConfig.SUB_TASK_TIMEOUT, return_when=concurrent.futures.ALL_COMPLETED)

            for future_item in done_tasks: # Check for exceptions in completed tasks
                try: future_item.result()
                except Exception as e_sub: logging.error(f"Sub-task completed with error: {e_sub}", exc_info=True) # Error already logged by task itself

            if not_done_tasks:
                for future_item in not_done_tasks:
                    logging.warning(f"Sub-task did not complete within timeout: {future_item}. Attempting to cancel.")
                    future_item.cancel()
                    # Potentially mark associated DB Task as timed out / error

            # --- FINALIZE ---
            _framework_update_task(parent_task_id, "Finalizing...", 95.0, None, None)
            with self.db_manager.get_session("public") as session:
                repo_obj = session.get(Repository, repo_id) # Renamed repo to repo_obj
                if repo_obj:
                    repo_obj.db_schemas = list(all_schema_payloads.keys()) # Schemas used for this repo
                    repo_obj.active_branch = branch
                    repo_obj.last_scanned = datetime.utcnow()
                    # session.commit() # Done by context manager
                else:
                    logging.error(f"Finalize Ingestion: Repository with ID {repo_id} not found. Cannot update.")
            _framework_complete_task(parent_task_id, result={"status": "completed"})
            logging.info(f"Repository ingestion completed successfully for repo_id {repo_id}, branch {branch}")

        except Exception as e:
            logging.error(f"Ingestion failed for repo_id {repo_id}, branch {branch}: {e}", exc_info=True)
            _framework_complete_task(parent_task_id, error=traceback.format_exc())
        finally:
            if workers: # Ensure workers are shut down
                logging.info("Ensuring embedding workers are shut down...")
                self._shutdown_embedding_workers(workers, work_queues, shutdown_event)

            # Clean up worker outputs from memory
            if 'all_code_blob_data_from_workers' in locals() and all_code_blob_data_from_workers:
                del all_code_blob_data_from_workers
            if 'dgraph_files' in locals(): del dgraph_files
            if 'definitions_files' in locals(): del definitions_files
            if 'calls_files' in locals(): del calls_files
            if 'schema_cache_files' in locals(): del schema_cache_files
            if 'all_schema_payloads' in locals(): del all_schema_payloads
            import gc; gc.collect()

            if cache_path.exists():
                logging.info(f"Cleaning up cache directory: {cache_path}")
                shutil.rmtree(cache_path)
