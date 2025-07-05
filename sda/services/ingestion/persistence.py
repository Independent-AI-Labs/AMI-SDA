# sda/services/ingestion/persistence.py

import json
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Generator

from sqlalchemy.dialects.postgresql import insert
import tiktoken # For _persist_chunks_for_schema tokenizer

# Imports from original ingestion.py / project structure
from sda.config import IngestionConfig, AIConfig # AIConfig for tokenizer model, IngestionConfig for DGRAPH_BATCH_SIZE
from sda.core.db_management import DatabaseManager
from sda.core.models import File, ASTNode, DBCodeChunk # SQLAlchemy models

# Type alias from ingestion.py, if needed by functions here (e.g. _persist_dgraph_nodes)
StatusUpdater = Callable[[int, str, float, Optional[str], Optional[Dict[str, Any]]], None]


def _create_postgres_payload():
    """Helper to define the structure of the postgres payload."""
    return {'files': {}, 'nodes': [], 'chunks': []}

def _create_batches(items: List[Any], batch_size: int) -> Generator[List[Any], None, None]:
    if batch_size <= 0: batch_size = 1
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]

def _aggregate_postgres_payloads(cache_files: List[str]) -> Dict[str, Any]:
    """Aggregates postgres payloads from cache files with deduplication."""
    aggregated = _create_postgres_payload()
    seen_nodes, seen_chunks = set(), set()

    for file_path in cache_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                payload = json.load(f)

            aggregated['files'].update(payload.get('files', {}))
            for node in payload.get('nodes', []):
                if (node_id := node.get('node_id')) and node_id not in seen_nodes:
                    aggregated['nodes'].append(node)
                    seen_nodes.add(node_id)
            for chunk in payload.get('chunks', []):
                if (chunk_id := chunk.get('chunk_id')) and chunk_id not in seen_chunks:
                    aggregated['chunks'].append(chunk)
                    seen_chunks.add(chunk_id)
        except Exception as e:
            logging.error(f"Error reading cache file {file_path}: {e}")
    return aggregated

def _load_and_merge_json_files(file_paths: List[str]) -> List[Any]:
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

def _persist_files_for_schema(db_manager: DatabaseManager, schema_name: str, payload: Dict, repo_id: int, branch: str, repo_path_str: str) -> Dict[str, int]:
    if not payload.get('files'): return {}
    with db_manager.get_session(schema_name) as session:
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

def _persist_nodes_for_schema(db_manager: DatabaseManager, schema_name: str, payload: Dict, file_id_map: Dict[str, int], branch: str):
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
    with db_manager.get_session(schema_name) as session:
        session.bulk_insert_mappings(ASTNode, node_mappings)

def _persist_chunks_for_schema(db_manager: DatabaseManager, schema_name: str, payload: Dict, file_id_map: Dict[str, int], repo_id: int, branch: str, cache_path: Path, tokenizer: tiktoken.Encoding) -> Optional[Tuple[str, int]]:
    if not payload.get('chunks') or not file_id_map: return None
    chunk_mappings = []
    for c in payload['chunks']:
        if file_id := file_id_map.get(c['chunk_id'].split(':', 1)[0]):
            token_count = len(tokenizer.encode(c['content']))
            chunk_mappings.append({
                'repository_id': repo_id, 'file_id': file_id, 'branch': branch,
                'language': IngestionConfig.LANGUAGE_MAPPING.get(Path(c['chunk_id'].split(':', 1)[0]).suffix),
                'token_count': token_count, 'start_line': c['metadata'].get('start_line'),
                'end_line': c['metadata'].get('end_line'), 'ast_node_ids': c['ast_node_ids'],
                'chunk_metadata': c['metadata'], 'content': c['content'], 'chunk_id': c['chunk_id'],
                'importance_score': c.get('importance_score', 0.5)})
    if not chunk_mappings: return None
    with db_manager.get_session(schema_name) as session:
        session.bulk_insert_mappings(DBCodeChunk, chunk_mappings)
        all_persisted_chunks = session.query(DBCodeChunk.id, DBCodeChunk.content, DBCodeChunk.token_count).filter(
            DBCodeChunk.repository_id == repo_id, DBCodeChunk.branch == branch).all()
        if not all_persisted_chunks: return None
        chunks_for_embedding_path = cache_path / f"embed_{schema_name}.jsonl"
        with open(chunks_for_embedding_path, 'w', encoding='utf-8') as f:
            for chunk_id, content, token_count in all_persisted_chunks:
                f.write(json.dumps({"schema": schema_name, "id": chunk_id, "content": content, "token_count": token_count}) + '\n')
        return str(chunks_for_embedding_path), len(all_persisted_chunks)

def _persist_dgraph_nodes(db_manager: DatabaseManager, repo_id: int, branch: str, dgraph_nodes_data: List[Dict], task_id: int, updater: StatusUpdater):
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
            db_manager.execute_dgraph_mutations(unique_batch)
            total_persisted += len(unique_batch)
        processed_nodes = i * IngestionConfig.DGRAPH_BATCH_SIZE + len(batch)
        current_progress = (processed_nodes / total_nodes) * progress_share if total_nodes > 0 else 0
        updater(task_id, f"Persisting graph nodes...", current_progress, None, {'Nodes Persisted': f"{total_persisted}/{total_nodes}"})
    return total_persisted

def _persist_vector_batch(db_manager: DatabaseManager, batch_results: List[tuple]) -> int:
    vector_updates = defaultdict(list)
    total_tokens = 0
    for schema, chunk_id, embedding, token_count in batch_results:
        if embedding:
            vector_updates[schema].append({'id': chunk_id, 'embedding': embedding})
        total_tokens += token_count
    for schema, updates in vector_updates.items():
        if updates:
            updates.sort(key=lambda x: x['id'])
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    with db_manager.get_session(schema) as s:
                        s.bulk_update_mappings(DBCodeChunk, updates)
                        break
                except Exception as e:
                    if "deadlock detected" in str(e).lower() and attempt < max_retries - 1:
                        import time, random
                        wait_time = (0.1 * (2 ** attempt)) + random.uniform(0, 0.1)
                        logging.warning(f"Deadlock detected for schema {schema}, retrying in {wait_time:.3f}s")
                        time.sleep(wait_time)
                    else:
                        raise
    return total_tokens

def _process_graph_edges_streaming(db_manager: DatabaseManager, definitions_files: List[str], calls_files: List[str], task_id: int, updater: StatusUpdater):
    if not definitions_files or not calls_files: return 0
    updater(task_id, "Building definitions index...", 55.0, None, {'Definitions Files': len(definitions_files)})
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
        if i % 50 == 0:
            progress = 55.0 + ((i / len(definitions_files)) * 15.0)
            updater(task_id, f"Processing definitions files {i + 1}/{len(definitions_files)}...", progress, None, {'Definitions Loaded': total_definitions})
    if not all_definitions:
        updater(task_id, "No definitions found for edge processing.", 95.0, None, {'Total Definitions': 0})
        return 0
    logging.info(f"Built definitions index with {len(all_definitions)} unique symbols, {total_definitions} total definitions")
    updater(task_id, "Processing calls and creating edges...", 70.0, None, {'Unique Symbols': len(all_definitions), 'Total Definitions': total_definitions})
    total_edges, processed_call_files, edge_batch = 0, 0, []
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
                                    if len(edge_batch) >= EDGE_BATCH_SIZE:
                                        db_manager.execute_dgraph_mutations(edge_batch)
                                        total_edges += len(edge_batch)
                                        edge_batch = []
        except Exception as e:
            logging.error(f"Error reading calls file {call_file}: {e}")
        processed_call_files += 1
        if processed_call_files % 50 == 0:
            progress = 70.0 + ((processed_call_files / len(calls_files)) * 25.0)
            updater(task_id, f"Processing calls files {processed_call_files}/{len(calls_files)}...", progress, None, {'Edges Created': total_edges, 'Files Processed': processed_call_files})
    if edge_batch:
        db_manager.execute_dgraph_mutations(edge_batch)
        total_edges += len(edge_batch)
    logging.info(f"Created {total_edges} graph edges from {processed_call_files} call files")
    return total_edges

# Placeholder for StatusUpdater type if not defined elsewhere or to avoid circular dependency
from typing import Callable
StatusUpdater = Callable[[int, str, float, Optional[str], Optional[Dict[str, Any]]], None]
