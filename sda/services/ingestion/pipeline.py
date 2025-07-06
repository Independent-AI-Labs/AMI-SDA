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
from sqlalchemy.dialects.postgresql import insert

from sda.config import IngestionConfig, AIConfig, DATA_DIR, INGESTION_CACHE_DIR
from sda.core.db_management import DatabaseManager
from sda.core.models import Repository, File, ASTNode, DBCodeChunk, BillingUsage, Task
from sda.services.analysis import resolve_embedding_devices
# from sda.services.chunking import TokenAwareChunker # No longer directly needed here
from sda.services.git_integration import GitService
from sda.services.partitioning import SmartPartitioningService
from sda.utils.task_executor import TaskExecutor

# Imports from the new ingestion submodules
# Assuming this file will be sda/services/ingestion/pipeline.py, use relative imports
from .workers import _initialize_parsing_worker, _persistent_embedding_worker, _pass1_parse_files_worker
# _chunker_instance is defined and used within workers.py

from .persistence import (
    _aggregate_postgres_payloads, _persist_files_for_schema, _persist_nodes_for_schema,
    _persist_chunks_for_schema, _persist_dgraph_nodes, _persist_vector_batch,
    _process_graph_edges_streaming, _load_and_merge_json_files,
    _create_postgres_payload, _create_batches # Also moved to persistence
)
from .utils import _stream_chunks_from_files, _stream_batcher


StatusUpdater = Callable[[int, str, float, Optional[str], Optional[Dict[str, Any]]], None]

# Commented out stubs for moved functions are now removed by this change

class IntelligentIngestionService:
    def __init__(self, db_manager: DatabaseManager, git_service: GitService,
                 task_executor: TaskExecutor, partitioning_service: SmartPartitioningService):
        self.db_manager = db_manager
        self.git_service = git_service
        self.task_executor = task_executor
        self.partitioning_service = partitioning_service
        self.tokenizer = tiktoken.get_encoding(IngestionConfig.TOKENIZER_MODEL)
        self.embedding_config = AIConfig.get_active_embedding_config()
        logging.info("IntelligentIngestionService initialized with TaskExecutor and SmartPartitioningService.")

    # Persistence methods that were previously here are now in sda.services.ingestion.persistence.py
    # and will be called as functions, passing self.db_manager and other necessary params.

    # _process_graph_edges_from_data was here but seems unused, _process_graph_edges_streaming is used and moved.

    def _setup_embedding_workers(self) -> Tuple[List[mp.Process], List[mp.Queue], mp.Queue, mp.Event]:
        devices = resolve_embedding_devices()
        num_workers = min(len(devices), AIConfig.MAX_EMBEDDING_WORKERS)
        devices_to_use = devices[:num_workers] or ["cpu"]
        work_queues, workers = [], []
        result_queue = mp.Queue()
        shutdown_event = mp.Event()
        for i, device in enumerate(devices_to_use):
            work_queue = mp.Queue()
            worker = mp.Process(target=_persistent_embedding_worker,
                                args=(device, self.embedding_config.model_name, str(DATA_DIR / "embedding_models"), work_queue, result_queue, shutdown_event),
                                name=f"EmbeddingWorker-{i}-{device}")
            workers.append(worker);
            work_queues.append(work_queue);
            worker.start()
        ready_count = 0
        while ready_count < len(workers):
            try:
                msg_type, _ = result_queue.get(timeout=60.0)
                if msg_type == "ready":
                    ready_count += 1
                elif msg_type == "error":
                    raise RuntimeError("Embedding worker failed to initialize.")
            except queue.Empty:
                raise TimeoutError("Timed out waiting for embedding workers to initialize.")
        return workers, work_queues, result_queue, shutdown_event

    def _shutdown_embedding_workers(self, workers, work_queues, shutdown_event):
        shutdown_event.set()
        for wq in work_queues: wq.put(None)
        for w in workers: w.join(timeout=10)

    # _persist_vector_batch and _process_graph_edges_streaming have been moved to sda.services.ingestion.persistence.py

    def ingest_repository(self, repo_path_str: str, repo_uuid: str, branch: str, repo_id: int, parent_task_id: int, _framework_start_task: Callable,
                          _framework_update_task: StatusUpdater, _framework_complete_task: Callable):
        cache_path = Path(INGESTION_CACHE_DIR) / f"{repo_uuid}_{branch.replace('/', '_')}"
        if cache_path.exists(): shutil.rmtree(cache_path)
        cache_path.mkdir(parents=True, exist_ok=True)
        workers, work_queues, result_queue, shutdown_event = None, None, None, None
        try:
            # --- PREP ---
            _framework_update_task(parent_task_id, "Preparing environment...", 2.0, None, None)
            if not self.git_service.checkout(repo_path_str, branch): raise RuntimeError(f"Could not checkout branch '{branch}'.")
            _framework_update_task(parent_task_id, "Clearing old data...", 4.0, None, None)
            self.db_manager.clear_dgraph_data_for_branch(repo_id, branch) # Dgraph data for branch

            # Clear DBCodeChunks for this repo and branch from the public schema once
            with self.db_manager.get_session("public") as public_session:
                logging.info(f"Clearing old DBCodeChunk data from public schema for repo_id: {repo_id}, branch: {branch}")
                public_session.query(DBCodeChunk).filter(
                    DBCodeChunk.repository_id == repo_id,
                    DBCodeChunk.branch == branch
                ).delete(synchronize_session=False)
                public_session.commit() # Commit this deletion

            # Wipe individual repo schemas (containing File, ASTNode tables for partitions)
            with self.db_manager.get_session("public") as s: # New session for safety, though commit above helps
                repo = s.get(Repository, repo_id)
                if repo and repo.uuid:
                    logging.info(f"Wiping individual partition schemas for repo_uuid: {repo.uuid}")
                    self.db_manager._wipe_repo_schemas(repo.uuid) # This drops 'repo_xyz_%' schemas

            files_on_disk = [p for p in Path(repo_path_str).rglob('*') if not any(
                d in p.parts for d in IngestionConfig.DEFAULT_IGNORE_DIRS) and p.is_file() and p.suffix in IngestionConfig.LANGUAGE_MAPPING]
            if not files_on_disk:
                _framework_complete_task(parent_task_id, result={"message": "No files to ingest."});
                return

            # --- PARTITIONING ---
            _framework_update_task(parent_task_id, "Analyzing repository structure...", 8.0, None, {'Files Found': len(files_on_disk)})
            file_to_schema_map = self.partitioning_service.generate_schema_map(repo_path_str, files_on_disk, repo_uuid)
            schema_to_files_map = defaultdict(list)
            for file_path, schema_name in file_to_schema_map.items():
                schema_to_files_map[schema_name].append(str(file_path))
            _framework_update_task(parent_task_id, "Initializing workers...", 10.0, None, {'Partitions Created': len(schema_to_files_map)})
            workers, work_queues, result_queue, shutdown_event = self._setup_embedding_workers()

            # --- STAGE 0: PARALLEL CPU-BOUND FILE PARSING ---
            with concurrent.futures.ProcessPoolExecutor(max_workers=IngestionConfig.MAX_CPU_WORKERS, initializer=_initialize_parsing_worker) as cpu_executor:
                _framework_update_task(parent_task_id, "Parsing files...", 12.0,
                                       f"Processing {len(files_on_disk)} files across {len(schema_to_files_map)} schemas.", None)
                pass1_futures = []
                sorted_schemas = sorted(schema_to_files_map.items(), key=lambda item: len(item[1]), reverse=True)

                for schema_name, files_in_schema in sorted_schemas:
                    files_in_schema.sort(key=lambda f: Path(f).stat().st_size, reverse=True)
                    for batch in _create_batches(files_in_schema, IngestionConfig.FILE_PROCESSING_BATCH_SIZE):
                        pass1_futures.append(cpu_executor.submit(_pass1_parse_files_worker, batch, repo_path_str, schema_name, cache_path))

                # Collect cache files by category
                schema_cache_files = defaultdict(list)
                dgraph_files, definitions_files, calls_files = [], [], []

                for i, f in enumerate(concurrent.futures.as_completed(pass1_futures)):
                    try:
                        res = f.result()
                        schema_name = res['schema_name']
                        schema_cache_files[schema_name].append(res['postgres_file'])
                        dgraph_files.append(res['dgraph_nodes_file'])
                        definitions_files.append(res['definitions_file'])
                        calls_files.append(res['calls_file'])
                    except Exception as e:
                        logging.error(f"Failed to get result from parsing future: {e}")
                    progress = 12.0 + (i / len(pass1_futures) * 28.0)  # Parsing is 12-40%
                    _framework_update_task(parent_task_id, f"Parsing file batches...", progress, None, {'Parsed Batches': f"{i + 1}/{len(pass1_futures)}"})

            # --- STAGES 1-4: PARALLEL PERSISTENCE OF METADATA ---
            _framework_update_task(parent_task_id, "Persisting metadata...", 40.0, "Creating database schemas and aggregating data.", None)

            # Aggregate payloads per schema from cache files
            all_schema_payloads = {}
            for schema_name, cache_files in schema_cache_files.items():
                # Call imported function
                all_schema_payloads[schema_name] = _aggregate_postgres_payloads(cache_files)

            schema_creation_futures = [self.task_executor.submit('postgres', self.db_manager.create_schema_and_tables, schema_name) for schema_name in
                                       all_schema_payloads.keys()]
            concurrent.futures.wait(schema_creation_futures)

            _framework_update_task(parent_task_id, "Persisting file records...", 45.0, f"Submitting {len(all_schema_payloads)} schemas for file persistence.",
                                   None)
            file_id_maps, file_futures = {}, {
                self.task_executor.submit('postgres', _persist_files_for_schema, self.db_manager, schema, payload, repo_id, branch, repo_path_str): schema for # Call imported function
                schema, payload in all_schema_payloads.items()}
            for i, f in enumerate(concurrent.futures.as_completed(file_futures)):
                schema_name = file_futures[f];
                file_id_maps[schema_name] = f.result()
                _framework_update_task(parent_task_id, f"Persisting file records...", 45.0 + (i / len(file_futures) * 5.0), None, None)

            _framework_update_task(parent_task_id, "Persisting AST nodes and code chunks...", 50.0, "Submitting node and chunk persistence tasks.", None)
            node_futures = [self.task_executor.submit('postgres', _persist_nodes_for_schema, self.db_manager, schema, payload, file_id_maps.get(schema, {}), branch) for # Call imported function
                            schema, payload in all_schema_payloads.items()]
            chunk_futures = {
                self.task_executor.submit('postgres', _persist_chunks_for_schema, self.db_manager, schema, payload, file_id_maps.get(schema, {}), repo_id, branch, # Call imported function
                                          cache_path, self.tokenizer): schema for schema, payload in all_schema_payloads.items()}

            embed_files, total_chunks = [], 0
            for i, f in enumerate(concurrent.futures.as_completed(chunk_futures)):
                if res := f.result(): path, count = res; embed_files.append(path); total_chunks += count
            concurrent.futures.wait(node_futures)
            _framework_update_task(parent_task_id, "Metadata persistence complete.", 60.0, "AST, Chunks, and File records saved.",
                                   {'Chunks for Embedding': total_chunks})

            # --- STAGES 5 & 6: PARALLEL GRAPH AND VECTOR PROCESSING ---
            _framework_update_task(parent_task_id, "Processing graph and vector embeddings...", 60.0, "Starting parallel data processing.", None)
            graph_task = _framework_start_task(repo_id, "Graph Processing", "system", parent_task_id)
            vector_task = _framework_start_task(repo_id, "Vector Embedding", "system", parent_task_id)

            def _run_graph_processing_task(task: Task):
                try:
                    _framework_update_task(task.id, "Loading graph data from cache...", 5.0, None, None)

                    # Load and merge dgraph data from cache files
                    dgraph_nodes_data = _load_and_merge_json_files(dgraph_files) # Call imported function
                    logging.info(f"Loaded {len(dgraph_nodes_data)} dgraph nodes from {len(dgraph_files)} cache files")

                    if dgraph_nodes_data:
                        dgraph_nodes_data.sort(key=lambda x: x['node_id'])
                        # Call imported function, pass self.db_manager
                        nodes_persisted = _persist_dgraph_nodes(self.db_manager, repo_id, branch, dgraph_nodes_data, task.id, _framework_update_task)

                        # Free memory immediately after persisting nodes
                        del dgraph_nodes_data
                        import gc;
                        gc.collect()
                    else:
                        nodes_persisted = 0
                        _framework_update_task(task.id, "No graph nodes found to persist.", 50.0, None, {'Nodes Persisted': '0'})

                    _framework_update_task(task.id, "Processing graph edges...", 50.0, f"Persisted {nodes_persisted} graph nodes.",
                                           {'Nodes Persisted': nodes_persisted})

                    # Process definitions and calls in streaming fashion to avoid memory buildup
                    # Call imported function, pass self.db_manager
                    edges_persisted = _process_graph_edges_streaming(self.db_manager, definitions_files, calls_files, task.id, _framework_update_task)

                    final_message = "Graph processing complete."
                    _framework_update_task(task.id, final_message, 100.0, final_message,
                                           {'Nodes Persisted': nodes_persisted, 'Edges Persisted': edges_persisted})
                    _framework_complete_task(task.id, result={'nodes_persisted': nodes_persisted, 'edges_persisted': edges_persisted})
                except Exception as e:
                    logging.error(f"Graph processing task failed: {e}", exc_info=True)
                    _framework_complete_task(task.id, error=traceback.format_exc())

            def _run_vector_embedding_task(task: Task):
                try:
                    if not embed_files or total_chunks == 0:
                        _framework_update_task(task.id, "No content to embed.", 100.0, None, {'Total Chunks': 0})
                        _framework_complete_task(task.id, result={"message": "No content to embed."})
                        return

                    _framework_update_task(task.id, f"Preparing to process {total_chunks} chunks...", 5.0, None,
                                           {'Total Chunks': total_chunks, 'Embed Files': len(embed_files)})

                    # Don't load all chunks at once - process them in streaming fashion
                    processed_chunks, total_tokens = 0, 0
                    persistence_futures = []

                    # Much smaller limit to prevent memory buildup
                    MAX_PENDING_FUTURES = 25
                    batches_sent = 0

                    _framework_update_task(task.id, "Starting streaming embedding process...", 10.0, None, {'Max Pending': MAX_PENDING_FUTURES})

                    # Process chunks in streaming fashion
                    chunk_stream = _stream_chunks_from_files(embed_files)

                    for batch in _stream_batcher(chunk_stream, 256):
                        # Distribute batch to workers
                        work_queues[batches_sent % len(work_queues)].put(batch)
                        batches_sent += 1

                        # Check for completed embeddings
                        try:
                            msg_type, msg_data = result_queue.get(timeout=0.1)  # Non-blocking check
                            if msg_type == "result":
                                _, batch_results = msg_data
                                batch_results.sort(key=lambda x: x[1])  # Sort by chunk_id

                                # Manage pending futures with sliding window
                                if len(persistence_futures) >= MAX_PENDING_FUTURES:
                                    # Wait for oldest futures to complete
                                    completed_count = 0
                                    target_completions = MAX_PENDING_FUTURES // 2

                                    for completed_future in concurrent.futures.as_completed(persistence_futures):
                                        try:
                                            total_tokens += completed_future.result()
                                            completed_count += 1
                                            if completed_count >= target_completions:
                                                break
                                        except Exception as e:
                                            logging.error(f"Persistence task failed: {e}")
                                            completed_count += 1

                                    # Remove completed futures
                                    persistence_futures = [f for f in persistence_futures if not f.done()]

                                # Call imported function, pass self.db_manager
                                future = self.task_executor.submit('postgres', _persist_vector_batch, self.db_manager, batch_results)
                                persistence_futures.append(future)
                                processed_chunks += len(batch_results)

                                progress_pct = 15.0 + ((processed_chunks / total_chunks) * 70.0) if total_chunks > 0 else 85.0
                                _framework_update_task(task.id, "Streaming embedding process...", progress_pct, None, {
                                    'Chunks Processed': f"{processed_chunks}/{total_chunks}",
                                    'Batches Sent': batches_sent,
                                    'Pending Persistence': len(persistence_futures),
                                    'Tokens Processed': total_tokens
                                })

                            elif msg_type == "error":
                                logging.error(f"Embedding worker error: {msg_data}")
                        except queue.Empty:
                            pass  # No results ready yet, continue

                        # Periodic progress update for batches sent
                        if batches_sent % 1000 == 0:
                            _framework_update_task(task.id, f"Sent {batches_sent} batches...", 15.0, None, {
                                'Batches Sent': batches_sent,
                                'Pending Persistence': len(persistence_futures)
                            })

                    logging.info(f"Finished sending {batches_sent} batches to workers")

                    # Continue processing remaining results
                    _framework_update_task(task.id, "Processing remaining embeddings...", 85.0, None, {'Final Batches': batches_sent})

                    while processed_chunks < total_chunks:
                        try:
                            msg_type, msg_data = result_queue.get(timeout=60.0)
                            if msg_type == "result":
                                _, batch_results = msg_data
                                batch_results.sort(key=lambda x: x[1])

                                # Call imported function, pass self.db_manager
                                future = self.task_executor.submit('postgres', _persist_vector_batch, self.db_manager, batch_results)
                                persistence_futures.append(future)
                                processed_chunks += len(batch_results)

                                progress_pct = 85.0 + ((processed_chunks / total_chunks) * 10.0) if total_chunks > 0 else 95.0
                                _framework_update_task(task.id, "Finalizing embeddings...", progress_pct, None, {
                                    'Chunks Processed': f"{processed_chunks}/{total_chunks}",
                                    'Pending Persistence': len(persistence_futures)
                                })
                            elif msg_type == "error":
                                logging.error(f"Embedding worker error: {msg_data}")
                                break
                        except queue.Empty:
                            logging.warning("Timeout waiting for final embedding results")
                            break

                    # Wait for all persistence to complete
                    _framework_update_task(task.id, "Waiting for persistence completion...", 95.0, None, {'Remaining Tasks': len(persistence_futures)})

                    for f in concurrent.futures.as_completed(persistence_futures):
                        try:
                            total_tokens += f.result()
                        except Exception as e:
                            logging.error(f"Persistence task failed: {e}")

                    cost = (total_tokens / 1_000_000) * self.embedding_config.price_per_million_tokens
                    usage_record = BillingUsage(
                        model_name=self.embedding_config.model_name,
                        provider=self.embedding_config.provider,
                        api_key_used_hash=hashlib.sha256(self.embedding_config.provider.encode()).hexdigest(),
                        total_tokens=total_tokens,
                        cost=cost
                    )
                    with self.db_manager.get_session("public") as s:
                        s.add(usage_record)

                    final_details = {
                        'Total Cost': f"${cost:.4f}",
                        'Tokens Processed': total_tokens, # Add final token count here
                        'Chunks Processed': f"{processed_chunks}/{total_chunks}" # Add final chunk count
                    }
                    _framework_update_task(task.id, "Vector embedding complete.", 100.0, None, final_details)
                    _framework_complete_task(task.id, result={"message": f"Embedding complete. Total cost: ${cost:.4f}", 'tokens': total_tokens, 'cost': cost})
                except Exception as e:
                    logging.error(f"Vector embedding task failed: {e}", exc_info=True)
                    _framework_complete_task(task.id, error=traceback.format_exc())

            # Submit both tasks and wait for completion
            logging.info(f"Submitting graph processing task {graph_task.id} to TaskExecutor")
            logging.info(f"Submitting vector embedding task {vector_task.id} to TaskExecutor")

            try:
                graph_future = self.task_executor.submit('dgraph', _run_graph_processing_task, graph_task)
                vector_future = self.task_executor.submit('postgres', _run_vector_embedding_task, vector_task)

                logging.info("Waiting for graph and vector processing tasks to complete...")

                # Wait for both tasks with timeout and better error handling
                done, not_done = concurrent.futures.wait([graph_future, vector_future], timeout=3600)  # 1 hour timeout

                if not_done:
                    logging.warning(f"Tasks did not complete within timeout: {len(not_done)} tasks still running")
                    for future in not_done:
                        future.cancel()

                # Check for exceptions
                for future in done:
                    try:
                        result = future.result()
                        logging.info(f"Task completed with result: {result}")
                    except Exception as e:
                        logging.error(f"Task failed with exception: {e}", exc_info=True)

            except Exception as e:
                logging.error(f"Error submitting or waiting for tasks: {e}", exc_info=True)
                raise

            # --- FINALIZE ---
            _framework_update_task(parent_task_id, "Finalizing...", 95.0, None, None)
            with self.db_manager.get_session("public") as session:
                repo = session.get(Repository, repo_id)
                if repo:
                    repo.db_schemas = list(all_schema_payloads.keys())
                    repo.active_branch = branch
                    repo.last_scanned = datetime.utcnow()
                    session.commit()
                else:
                    logging.error(f"Finalize Ingestion: Repository with ID {repo_id} not found in public schema. Cannot update db_schemas.")
            _framework_complete_task(parent_task_id, result={"status": "completed"})
            logging.info("Repository ingestion completed successfully")

        except Exception as e:
            logging.error(f"Ingestion failed: {e}", exc_info=True)
            _framework_complete_task(parent_task_id, error=traceback.format_exc())
        finally:
            if workers:
                logging.info("Shutting down embedding workers...")
                self._shutdown_embedding_workers(workers, work_queues, shutdown_event)
            if cache_path.exists():
                logging.info(f"Cleaning up cache directory: {cache_path}")
                shutil.rmtree(cache_path)
