# sda/services/ingestion.py

import concurrent.futures
import dataclasses
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
from sda.services.chunking import TokenAwareChunker
from sda.services.git_integration import GitService
from sda.services.partitioning import SmartPartitioningService
from sda.utils.task_executor import TaskExecutor

StatusUpdater = Callable[[int, str, float, Optional[str], Optional[Dict[str, Any]]], None]

_chunker_instance: Optional[TokenAwareChunker] = None


def _initialize_parsing_worker():
    global _chunker_instance
    pid = os.getpid()
    if _chunker_instance is None:
        logging.info(f"[ParserWorker PID:{pid}] Initializing TokenAwareChunker...")
        _chunker_instance = TokenAwareChunker()


def _create_postgres_payload():
    return {'files': {}, 'nodes': [], 'chunks': []}


def _create_batches(items: List[Any], batch_size: int) -> Generator[List[Any], None, None]:
    if batch_size <= 0: batch_size = 1
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def _stream_chunks_from_files(file_paths: List[str]) -> Generator[Tuple[str, int, str, int], None, None]:
    for file_path in file_paths:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        yield (data['schema'], data['id'], data['content'], data['token_count'])
        except FileNotFoundError:
            logging.warning(f"Chunk cache file not found, skipping: {file_path}")
        except Exception as e:
            logging.error(f"Error reading chunk cache file {file_path}: {e}")


def _stream_batcher(stream: Generator, batch_size: int) -> Generator[List[Any], None, None]:
    batch = []
    for item in stream:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _persistent_embedding_worker(device: str, model_name: str, cache_folder: str, work_queue: mp.Queue, result_queue: mp.Queue, shutdown_event: mp.Event):
    import torch, gc
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


def _pass1_parse_files_worker(file_batch: List[str], repo_root_str: str, schema_name: str, cache_path: Path) -> Dict[str, str]:
    """Worker that processes files and saves results to cache files to avoid memory overflow."""
    global _chunker_instance
    if _chunker_instance is None: _initialize_parsing_worker()
    chunker = _chunker_instance
    assert chunker is not None

    postgres_payload = _create_postgres_payload()
    dgraph_node_mutations, definitions, unresolved_calls = [], [], []

    for file_path_str in file_batch:
        abs_path = Path(file_path_str)
        relative_path_str = str(abs_path.relative_to(repo_root_str))
        try:
            content = abs_path.read_text(encoding='utf-8', errors='ignore')
            nodes, chunks = chunker.process_file(abs_path, content, file_identifier=relative_path_str)

            postgres_payload['files'][relative_path_str] = {
                'line_count': len(content.splitlines()),
                'hash': hashlib.sha256(content.encode('utf-8')).hexdigest(),
                'chunk_count': len(chunks)
            }
            postgres_payload['nodes'].extend([dataclasses.asdict(n) for n in nodes])
            postgres_payload['chunks'].extend([dataclasses.asdict(c) for c in chunks])

            for node in nodes:
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

    # Write results to cache files to avoid memory overflow
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

    def _aggregate_postgres_payloads(self, cache_files: List[str]) -> Dict[str, Any]:
        """Aggregates postgres payloads from cache files with deduplication."""
        aggregated = _create_postgres_payload()
        seen_nodes, seen_chunks = set(), set()

        for file_path in cache_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    payload = json.load(f)

                # Merge files
                aggregated['files'].update(payload.get('files', {}))

                # Deduplicate nodes
                for node in payload.get('nodes', []):
                    if (node_id := node.get('node_id')) and node_id not in seen_nodes:
                        aggregated['nodes'].append(node)
                        seen_nodes.add(node_id)

                # Deduplicate chunks
                for chunk in payload.get('chunks', []):
                    if (chunk_id := chunk.get('chunk_id')) and chunk_id not in seen_chunks:
                        aggregated['chunks'].append(chunk)
                        seen_chunks.add(chunk_id)

            except Exception as e:
                logging.error(f"Error reading cache file {file_path}: {e}")

        return aggregated

    def _load_and_merge_json_files(self, file_paths: List[str]) -> List[Any]:
        """Loads and merges JSON files containing lists."""
        merged_data = []
        for file_path in file_paths:
            try:
                if Path(file_path).exists():
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            merged_data.extend(data)
                else:
                    logging.warning(f"Cache file not found: {file_path}")
            except Exception as e:
                logging.error(f"Error reading file {file_path}: {e}")
        return merged_data

    def _persist_files_for_schema(self, schema_name: str, payload: Dict, repo_id: int, branch: str, repo_path_str: str) -> Dict[str, int]:
        """Persists only the File records for a given schema and returns their DB IDs."""
        if not payload.get('files'): return {}
        with self.db_manager.get_session(schema_name) as session:
            repo_path = Path(repo_path_str)
            file_mappings = [
                {"repository_id": repo_id, "branch": branch, "file_path": str(repo_path / rel_path), "relative_path": rel_path,
                 "language": IngestionConfig.LANGUAGE_MAPPING.get(Path(rel_path).suffix), "content_hash": data['hash'],
                 "last_modified": datetime.fromtimestamp((repo_path / rel_path).stat().st_mtime), "processed_at": datetime.utcnow(),
                 "line_count": data['line_count'], "chunk_count": data['chunk_count']}
                for rel_path, data in payload['files'].items() if (repo_path / rel_path).exists()
            ]
            if not file_mappings: return {}
            stmt = insert(File).values(file_mappings)
            stmt = stmt.on_conflict_do_update(
                index_elements=['repository_id', 'branch', 'relative_path'],
                set_=dict(content_hash=stmt.excluded.content_hash, last_modified=stmt.excluded.last_modified,
                          processed_at=stmt.excluded.processed_at, line_count=stmt.excluded.line_count,
                          chunk_count=stmt.excluded.chunk_count))
            session.execute(stmt)
            return {path: fid for fid, path in
                    session.query(File.id, File.relative_path).filter(File.repository_id == repo_id, File.branch == branch).all()}

    def _persist_nodes_for_schema(self, schema_name: str, payload: Dict, file_id_map: Dict[str, int], branch: str):
        """Persists only the ASTNode records for a given schema."""
        if not payload.get('nodes') or not file_id_map: return

        node_mappings = []
        for d in payload['nodes']:
            relative_path = d['node_id'].split(':', 1)[0]
            if file_id := file_id_map.get(relative_path):
                node_mappings.append({
                    'file_id': file_id, 'branch': branch, 'node_id': d.get('node_id'),
                    'node_type': d.get('node_type'), 'name': d.get('name'),
                    'start_line': d.get('start_line'), 'start_column': d.get('start_column'),
                    'end_line': d.get('end_line'), 'end_column': d.get('end_column'),
                    'parent_id': d.get('parent_id'), 'depth': d.get('depth'),
                    'complexity_score': d.get('complexity_score'),
                })
        if not node_mappings: return
        with self.db_manager.get_session(schema_name) as session:
            session.bulk_insert_mappings(ASTNode, node_mappings)

    def _persist_chunks_for_schema(self, schema_name: str, payload: Dict, file_id_map: Dict[str, int], repo_id: int, branch: str, cache_path: Path) -> Optional[
        Tuple[str, int]]:
        """Persists DBCodeChunk records and writes chunks to a cache file for embedding."""
        if not payload.get('chunks') or not file_id_map: return None
        chunk_mappings = []
        for c in payload['chunks']:
            if file_id := file_id_map.get(c['chunk_id'].split(':', 1)[0]):
                token_count = len(self.tokenizer.encode(c['content']))
                chunk_mappings.append({
                    'repository_id': repo_id, 'file_id': file_id, 'branch': branch,
                    'language': IngestionConfig.LANGUAGE_MAPPING.get(Path(c['chunk_id'].split(':', 1)[0]).suffix),
                    'token_count': token_count, 'start_line': c['metadata'].get('start_line'),
                    'end_line': c['metadata'].get('end_line'), 'ast_node_ids': c['ast_node_ids'],
                    'chunk_metadata': c['metadata'], 'content': c['content'], 'chunk_id': c['chunk_id'],
                    'importance_score': c.get('importance_score', 0.5)})
        if not chunk_mappings: return None
        with self.db_manager.get_session(schema_name) as session:
            session.bulk_insert_mappings(DBCodeChunk, chunk_mappings)
            all_persisted_chunks = session.query(DBCodeChunk.id, DBCodeChunk.content, DBCodeChunk.token_count).filter(
                DBCodeChunk.repository_id == repo_id, DBCodeChunk.branch == branch).all()
            if not all_persisted_chunks: return None
            chunks_for_embedding_path = cache_path / f"embed_{schema_name}.jsonl"
            with open(chunks_for_embedding_path, 'w', encoding='utf-8') as f:
                for chunk_id, content, token_count in all_persisted_chunks:
                    f.write(json.dumps({"schema": schema_name, "id": chunk_id, "content": content, "token_count": token_count}) + '\n')
            return str(chunks_for_embedding_path), len(all_persisted_chunks)

    def _persist_dgraph_nodes(self, repo_id: int, branch: str, dgraph_nodes_data: List[Dict], task_id: int, updater: StatusUpdater):
        """Executes Dgraph mutations to persist AST nodes with progress tracking."""
        if not dgraph_nodes_data:
            updater(task_id, "No graph nodes to persist.", 50.0, None, {'Nodes Persisted': '0/0'})
            return 0

        seen_node_ids = set()
        total_persisted = 0
        total_nodes = len(dgraph_nodes_data)
        progress_share = 50.0

        updater(task_id, f"Starting to persist {total_nodes} graph nodes...", 0.0, None, {'Total Nodes': total_nodes})

        for i, batch in enumerate(_create_batches(dgraph_nodes_data, IngestionConfig.DGRAPH_BATCH_SIZE)):
            unique_batch = []
            for m in batch:
                if m['node_id'] not in seen_node_ids:
                    m.update({'repo_id': str(repo_id), 'branch': branch})
                    unique_batch.append(m)
                    seen_node_ids.add(m['node_id'])
            if unique_batch:
                self.db_manager.execute_dgraph_mutations(unique_batch)
                total_persisted += len(unique_batch)

            processed_nodes = i * IngestionConfig.DGRAPH_BATCH_SIZE + len(batch)
            current_progress = (processed_nodes / total_nodes) * progress_share if total_nodes > 0 else 0
            updater(task_id, f"Persisting graph nodes...", current_progress, None, {'Nodes Persisted': f"{total_persisted}/{total_nodes}"})
        return total_persisted

    def _process_graph_edges_from_data(self, definitions: List[Tuple[str, str]], calls: List[Tuple[str, str]]):
        """Executes Dgraph mutations to create `calls` edges between nodes from in-memory data."""
        if not definitions or not calls:
            return 0

        all_definitions = defaultdict(list)
        for name, node_id in definitions:
            all_definitions[name].append(node_id)
        if not all_definitions: return 0

        mutations = []
        for caller_id, name in calls:
            if name in all_definitions:
                for callee_id in all_definitions[name]:
                    mutations.append({'uid': f"_:{caller_id}", "calls": {"uid": f"_:{callee_id}"}})

        if mutations:
            self.db_manager.execute_dgraph_mutations(mutations)
        return len(mutations)

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

    def _persist_vector_batch(self, batch_results: List[tuple]) -> int:
        """Persists a single batch of vectors with deadlock avoidance through consistent ordering."""
        vector_updates = defaultdict(list)
        total_tokens = 0
        for schema, chunk_id, embedding, token_count in batch_results:
            if embedding:
                vector_updates[schema].append({'id': chunk_id, 'embedding': embedding})
            total_tokens += token_count

        for schema, updates in vector_updates.items():
            if updates:
                # Sort updates by ID to ensure consistent lock ordering across all concurrent operations
                updates.sort(key=lambda x: x['id'])

                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        with self.db_manager.get_session(schema) as s:
                            s.bulk_update_mappings(DBCodeChunk, updates)
                            break
                    except Exception as e:
                        if "deadlock detected" in str(e).lower() and attempt < max_retries - 1:
                            import time
                            import random
                            # Exponential backoff with jitter
                            wait_time = (0.1 * (2 ** attempt)) + random.uniform(0, 0.1)
                            logging.warning(f"Deadlock detected for schema {schema}, retrying in {wait_time:.3f}s")
                            time.sleep(wait_time)
                        else:
                            raise
        return total_tokens

    def _process_graph_edges_streaming(self, definitions_files: List[str], calls_files: List[str], task_id: int, updater: StatusUpdater):
        """Processes graph edges in a streaming fashion to avoid memory buildup."""
        if not definitions_files or not calls_files:
            return 0

        updater(task_id, "Building definitions index...", 55.0, None, {'Definitions Files': len(definitions_files)})

        # Build definitions index in chunks to manage memory
        all_definitions = defaultdict(list)
        total_definitions = 0

        for i, def_file in enumerate(definitions_files):
            try:
                if Path(def_file).exists():
                    with open(def_file, 'r', encoding='utf-8') as f:
                        definitions = json.load(f)
                        if isinstance(definitions, list):
                            for name, node_id in definitions:
                                all_definitions[name].append(node_id)
                            total_definitions += len(definitions)
            except Exception as e:
                logging.error(f"Error reading definitions file {def_file}: {e}")

            if i % 50 == 0:  # Update progress every 50 files
                progress = 55.0 + ((i / len(definitions_files)) * 15.0)
                updater(task_id, f"Processing definitions files {i + 1}/{len(definitions_files)}...", progress, None,
                        {'Definitions Loaded': total_definitions})

        if not all_definitions:
            updater(task_id, "No definitions found for edge processing.", 95.0, None, {'Total Definitions': 0})
            return 0

        logging.info(f"Built definitions index with {len(all_definitions)} unique symbols, {total_definitions} total definitions")
        updater(task_id, "Processing calls and creating edges...", 70.0, None, {'Unique Symbols': len(all_definitions), 'Total Definitions': total_definitions})

        # Process calls in batches to manage memory and create edges incrementally
        total_edges = 0
        processed_call_files = 0
        edge_batch = []
        EDGE_BATCH_SIZE = 1000

        for call_file in calls_files:
            try:
                if Path(call_file).exists():
                    with open(call_file, 'r', encoding='utf-8') as f:
                        calls = json.load(f)
                        if isinstance(calls, list):
                            for caller_id, name in calls:
                                if name in all_definitions:
                                    for callee_id in all_definitions[name]:
                                        edge_batch.append({'uid': f"_:{caller_id}", "calls": {"uid": f"_:{callee_id}"}})

                                        # Persist edges in batches to avoid memory buildup
                                        if len(edge_batch) >= EDGE_BATCH_SIZE:
                                            self.db_manager.execute_dgraph_mutations(edge_batch)
                                            total_edges += len(edge_batch)
                                            edge_batch = []  # Clear the batch
            except Exception as e:
                logging.error(f"Error reading calls file {call_file}: {e}")

            processed_call_files += 1
            if processed_call_files % 50 == 0:  # Update progress every 50 files
                progress = 70.0 + ((processed_call_files / len(calls_files)) * 25.0)
                updater(task_id, f"Processing calls files {processed_call_files}/{len(calls_files)}...", progress, None,
                        {'Edges Created': total_edges, 'Files Processed': processed_call_files})

        # Persist any remaining edges
        if edge_batch:
            self.db_manager.execute_dgraph_mutations(edge_batch)
            total_edges += len(edge_batch)

        logging.info(f"Created {total_edges} graph edges from {processed_call_files} call files")
        return total_edges

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
            self.db_manager.clear_dgraph_data_for_branch(repo_id, branch)
            with self.db_manager.get_session("public") as s:
                repo = s.get(Repository, repo_id)
                if repo and repo.uuid:
                    self.db_manager._wipe_repo_schemas(repo.uuid)

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
                all_schema_payloads[schema_name] = self._aggregate_postgres_payloads(cache_files)

            schema_creation_futures = [self.task_executor.submit('postgres', self.db_manager.create_schema_and_tables, schema_name) for schema_name in
                                       all_schema_payloads.keys()]
            concurrent.futures.wait(schema_creation_futures)

            _framework_update_task(parent_task_id, "Persisting file records...", 45.0, f"Submitting {len(all_schema_payloads)} schemas for file persistence.",
                                   None)
            file_id_maps, file_futures = {}, {
                self.task_executor.submit('postgres', self._persist_files_for_schema, schema, payload, repo_id, branch, repo_path_str): schema for
                schema, payload in all_schema_payloads.items()}
            for i, f in enumerate(concurrent.futures.as_completed(file_futures)):
                schema_name = file_futures[f];
                file_id_maps[schema_name] = f.result()
                _framework_update_task(parent_task_id, f"Persisting file records...", 45.0 + (i / len(file_futures) * 5.0), None, None)

            _framework_update_task(parent_task_id, "Persisting AST nodes and code chunks...", 50.0, "Submitting node and chunk persistence tasks.", None)
            node_futures = [self.task_executor.submit('postgres', self._persist_nodes_for_schema, schema, payload, file_id_maps.get(schema, {}), branch) for
                            schema, payload in all_schema_payloads.items()]
            chunk_futures = {
                self.task_executor.submit('postgres', self._persist_chunks_for_schema, schema, payload, file_id_maps.get(schema, {}), repo_id, branch,
                                          cache_path): schema for schema, payload in all_schema_payloads.items()}

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
                    dgraph_nodes_data = self._load_and_merge_json_files(dgraph_files)
                    logging.info(f"Loaded {len(dgraph_nodes_data)} dgraph nodes from {len(dgraph_files)} cache files")

                    if dgraph_nodes_data:
                        dgraph_nodes_data.sort(key=lambda x: x['node_id'])
                        nodes_persisted = self._persist_dgraph_nodes(repo_id, branch, dgraph_nodes_data, task.id, _framework_update_task)

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
                    edges_persisted = self._process_graph_edges_streaming(definitions_files, calls_files, task.id, _framework_update_task)

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

                                future = self.task_executor.submit('postgres', self._persist_vector_batch, batch_results)
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

                                future = self.task_executor.submit('postgres', self._persist_vector_batch, batch_results)
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

                    _framework_update_task(task.id, "Vector embedding complete.", 100.0, None, {'Total Cost': f"${cost:.4f}"})
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
                repo.db_schemas = list(all_schema_payloads.keys())
                repo.active_branch, repo.last_scanned = branch, datetime.utcnow()
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
