# sda/services/ingestion/persistence.py

import json
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Generator, Callable

from sqlalchemy.dialects.postgresql import insert
import tiktoken # For _persist_chunks_for_schema tokenizer

# Imports from original ingestion.py / project structure
from sda.config import IngestionConfig, AIConfig # AIConfig for tokenizer model, IngestionConfig for DGRAPH_BATCH_SIZE
from sda.core.db_management import DatabaseManager
from sda.core.models import File, ASTNode, DBCodeChunk, CodeBlob # ADD CodeBlob
# SQLAlchemy models

# Type alias from ingestion.py, if needed by functions here (e.g. _persist_dgraph_nodes)
StatusUpdater = Callable[[int, str, float, Optional[str], Optional[Dict[str, Any]]], None]


def _create_postgres_payload():
    """Helper to define the structure of the postgres payload."""
    return {'files': {}, 'nodes': [], 'chunks': []}

def _create_batches(items: List[Any], batch_size: int) -> Generator[List[Any], None, None]:
    if batch_size <= 0: batch_size = 1 # Ensure batch_size is positive
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]

def _aggregate_postgres_payloads(cache_files: List[str]) -> Dict[str, Any]:
    """Aggregates postgres payloads from cache files with deduplication."""
    aggregated = _create_postgres_payload()
    # For nodes and chunks, use their respective unique IDs for deduplication
    seen_node_ids = set()
    seen_chunk_ids = set() # Assuming chunk_id is the unique textual ID

    for file_path_str in cache_files: # file_path_str is path to cache file
        try:
            with open(file_path_str, 'r', encoding='utf-8') as f:
                payload = json.load(f)

            # Files data is a dict, update merges it. Assumes unique file paths across payloads if merging.
            aggregated['files'].update(payload.get('files', {}))

            for node_data in payload.get('nodes', []): # node_data is a dict from TransientNode.model_dump()
                if (node_id_val := node_data.get('node_id')) and node_id_val not in seen_node_ids:
                    aggregated['nodes'].append(node_data)
                    seen_node_ids.add(node_id_val)

            for chunk_data in payload.get('chunks', []): # chunk_data is dict for DBCodeChunk
                if (chunk_id_val := chunk_data.get('chunk_id')) and chunk_id_val not in seen_chunk_ids:
                    aggregated['chunks'].append(chunk_data)
                    seen_chunk_ids.add(chunk_id_val)
        except Exception as e:
            logging.error(f"Error reading or processing cache file {file_path_str}: {e}", exc_info=True)
    return aggregated

def _load_and_merge_json_files(file_paths: List[str]) -> List[Any]:
    """Loads and merges JSON files containing lists of objects (e.g., Dgraph mutations)."""
    merged_data = []
    for file_path_str in file_paths: # file_path_str is path to cache file
        try:
            p = Path(file_path_str)
            if p.exists() and p.stat().st_size > 0: # Check if file exists and is not empty
                with open(file_path_str, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        merged_data.extend(data)
                    else:
                        logging.warning(f"Expected a list in {file_path_str}, got {type(data)}. Skipping.")
            else:
                logging.warning(f"Cache file not found or empty: {file_path_str}")
        except json.JSONDecodeError as e:
            logging.error(f"JSON decode error in file {file_path_str}: {e}", exc_info=True)
        except Exception as e:
            logging.error(f"Error reading file {file_path_str}: {e}", exc_info=True)
    return merged_data


# NEW function to persist CodeBlobs
def _persist_code_blobs(db_manager: DatabaseManager, all_code_blob_data: List[Dict[str, Any]], current_repo_id: int, current_branch: str):
    if not all_code_blob_data:
        logging.info("No CodeBlob data to persist.")
        return

    # Deduplicate by blob_hash, as blob_hash is PK and content-derived
    final_blobs_for_stmt: List[Dict[str, Any]] = []
    seen_blob_hashes = set()
    for blob_dict in all_code_blob_data:
        if blob_dict['blob_hash'] not in seen_blob_hashes:
            # Ensure all required fields for CodeBlob model are present
            final_blobs_for_stmt.append({
                "blob_hash": blob_dict['blob_hash'],
                "content": blob_dict['content'],
                "repository_id": blob_dict['repository_id'], # Should match current_repo_id
                "branch": blob_dict['branch'],               # Should match current_branch
                "file_path": blob_dict['file_path']
            })
            seen_blob_hashes.add(blob_dict['blob_hash'])

    if not final_blobs_for_stmt:
        logging.info("No unique CodeBlobs to persist after deduplication.")
        return

    with db_manager.get_session("public") as session:
        # CodeBlob PK is blob_hash. If content is identical, hash is same, so ON CONFLICT DO NOTHING.
        stmt = insert(CodeBlob).values(final_blobs_for_stmt)
        stmt = stmt.on_conflict_do_nothing(index_elements=['blob_hash'])
        try:
            session.execute(stmt)
            logging.info(f"Persisted/Ensured {len(final_blobs_for_stmt)} CodeBlobs into public.CodeBlobs.")
        except Exception as e:
            logging.error(f"Error persisting CodeBlobs: {e}", exc_info=True)
            raise # Re-raise to signal failure in pipeline
        finally:
            # Free memory, especially for 'content' field
            del final_blobs_for_stmt
            del all_code_blob_data


def _persist_files_for_schema(db_manager: DatabaseManager, schema_name: str, payload: Dict, repo_id: int, branch: str, repo_path_str: str) -> Dict[str, int]:
    if not payload.get('files'): return {}

    file_mappings = []
    repo_path_obj = Path(repo_path_str) # repo_path_obj for clarity

    for rel_path, data in payload['files'].items():
        # Ensure file still exists on disk before trying to stat it.
        # Worker already did this, but a check here might be useful if pipeline is long.
        abs_file_path = repo_path_obj / rel_path
        if abs_file_path.exists():
            file_map_data = {
                "repository_id": repo_id,
                "branch": branch,
                "file_path": str(abs_file_path), # Full path
                "relative_path": rel_path,       # Relative path (POSIX from worker)
                "language": IngestionConfig.LANGUAGE_MAPPING.get(Path(rel_path).suffix),
                "content_hash": data['hash'],    # This is the blob_hash
                "blob_hash": data['blob_hash'],  # Explicitly add blob_hash for File table
                "last_modified": datetime.fromtimestamp(abs_file_path.stat().st_mtime),
                "processed_at": datetime.utcnow(),
                "line_count": data['line_count'],
                "chunk_count": data['chunk_count']
            }
            file_mappings.append(file_map_data)
        else:
            logging.warning(f"File {abs_file_path} not found during _persist_files_for_schema, skipping.")

    if not file_mappings: return {}

    with db_manager.get_session(schema_name) as session:
        stmt = insert(File).values(file_mappings)
        set_data = dict(
            content_hash=stmt.excluded.content_hash,
            blob_hash=stmt.excluded.blob_hash,
            last_modified=stmt.excluded.last_modified,
            processed_at=stmt.excluded.processed_at,
            line_count=stmt.excluded.line_count,
            chunk_count=stmt.excluded.chunk_count
            # Ensure all updatable fields in File model are covered
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=['repository_id', 'branch', 'relative_path'], # UK constraint
            set_=set_data
        )
        session.execute(stmt)

        # Efficiently get IDs for the inserted/updated files
        # Assuming relative_path is unique per repo_id and branch
        persisted_files = session.query(File.id, File.relative_path).filter(
            File.repository_id == repo_id,
            File.branch == branch,
            File.relative_path.in_([fm["relative_path"] for fm in file_mappings])
        ).all()
        return {rel_p: fid for fid, rel_p in persisted_files}


def _persist_nodes_for_schema(db_manager: DatabaseManager, schema_name: str, payload: Dict, file_id_map: Dict[str, int], branch: str):
    if not payload.get('nodes') or not file_id_map: return

    node_mappings = []
    for d_node in payload['nodes']: # d_node is a dict from TransientNode.model_dump()
        relative_path = d_node['file_path']
        if file_id := file_id_map.get(relative_path): # file_id_map maps relative_path to File.id
            node_map = {
                'file_id': file_id,
                'branch': branch,
                'node_id': d_node.get('node_id'),
                'node_type': d_node.get('node_type'),
                'name': d_node.get('name'),
                'start_line': d_node.get('start_line'),
                'start_column': d_node.get('start_column'),
                'end_line': d_node.get('end_line'),
                'end_column': d_node.get('end_column'),
                'start_char_offset': d_node.get('start_char_offset'),
                'end_char_offset': d_node.get('end_char_offset'),
                'parent_id': d_node.get('parent_id'),
                'depth': d_node.get('depth'),
                'complexity_score': d_node.get('complexity_score'),
            }
            # Ensure all fields match the ASTNode SQLAlchemy model
            node_mappings.append(node_map)

    if not node_mappings: return
    with db_manager.get_session(schema_name) as session:
        logging.info(f"[{schema_name}] Clearing old ASTNode data for branch: {branch}")
        session.query(ASTNode).filter(ASTNode.branch == branch).delete(synchronize_session=False)
        session.bulk_insert_mappings(ASTNode, node_mappings)
        logging.info(f"[{schema_name}] Persisted {len(node_mappings)} AST nodes for branch: {branch}")


def _persist_chunks_for_schema(db_manager: DatabaseManager, schema_name: str, payload: Dict, file_id_map: Dict[str, int], repo_id: int, branch: str, cache_path: Path, tokenizer: tiktoken.Encoding) -> Optional[Tuple[str, int]]:
    if not payload.get('chunks') or not file_id_map: return None

    chunk_mappings_for_db: List[Dict[str, Any]] = []
    # Chunks by blob for efficient content fetching for .jsonl
    chunks_grouped_by_blob_hash: Dict[str, List[Dict]] = defaultdict(list)

    for c_def in payload['chunks']: # c_def is a dictionary for DBCodeChunk from worker
        relative_file_path = c_def.get("relative_file_path")
        if not relative_file_path:
            logging.warning(f"Chunk definition missing 'relative_file_path': {c_def.get('chunk_id')}. Skipping.")
            continue

        file_id = file_id_map.get(relative_file_path)
        if not file_id:
            logging.warning(f"No file_id for relative_path '{relative_file_path}' from chunk {c_def.get('chunk_id')}. Skipping.")
            continue

        db_chunk_map = {
            'chunk_id': c_def['chunk_id'], # Unique textual ID
            'repository_id': repo_id,
            'file_id': file_id,
            'branch': branch,
            'language': c_def.get('language', IngestionConfig.LANGUAGE_MAPPING.get(Path(relative_file_path).suffix)),
            'source_blob_hash': c_def['source_blob_hash'],
            'start_char_offset': c_def['start_char_offset'],
            'end_char_offset': c_def['end_char_offset'],
            'start_line': c_def.get('start_line'),
            'end_line': c_def.get('end_line'),
            'dgraph_node_uid': c_def.get('dgraph_node_uid'),
            'ast_node_ids': c_def.get('ast_node_ids', []),
            'chunk_metadata': c_def.get('chunk_metadata', {}),
            'importance_score': c_def.get('importance_score', 0.5),
            # token_count and embedding are initially null or handled by embedding step
        }
        chunk_mappings_for_db.append(db_chunk_map)
        # Store the full definition for later slicing (includes offsets)
        chunks_grouped_by_blob_hash[c_def['source_blob_hash']].append(c_def)

    if not chunk_mappings_for_db: return None

    chunks_for_embedding_jsonl: List[Dict[str, Any]] = []
    with db_manager.get_session("public") as session: # DBCodeChunk is in public schema
        # Clear old chunks for this repo/branch was done in pipeline.py

        # Perform bulk insert of chunk definitions
        # Assuming DBCodeChunk SQLAlchemy model matches db_chunk_map structure
        session.bulk_insert_mappings(DBCodeChunk, chunk_mappings_for_db)
        logging.info(f"[public] Inserted {len(chunk_mappings_for_db)} DBCodeChunk definitions for repo_id: {repo_id}, branch: {branch}")

        # Fetch content for .jsonl file generation
        blob_contents_cache: Dict[str, str] = {}
        required_blob_hashes = list(chunks_grouped_by_blob_hash.keys())
        if required_blob_hashes:
            blobs_from_db = session.query(CodeBlob.blob_hash, CodeBlob.content).filter(CodeBlob.blob_hash.in_(required_blob_hashes)).all()
            blob_contents_cache.update(dict(blobs_from_db))

        # Query back the DBCodeChunk.id (PK) for the .jsonl file
        # Match based on unique chunk_id (textual) for this repo and branch
        inserted_chunk_text_ids = [cm['chunk_id'] for cm in chunk_mappings_for_db]
        persisted_chunks_with_db_id = session.query(
            DBCodeChunk.id, DBCodeChunk.chunk_id, DBCodeChunk.source_blob_hash,
            DBCodeChunk.start_char_offset, DBCodeChunk.end_char_offset
        ).filter(
            DBCodeChunk.repository_id == repo_id,
            DBCodeChunk.branch == branch,
            DBCodeChunk.chunk_id.in_(inserted_chunk_text_ids)
        ).all()

        for db_pk_id, text_chunk_id, blob_hash, start_offset, end_offset in persisted_chunks_with_db_id:
            blob_content = blob_contents_cache.get(blob_hash)
            if blob_content is None:
                logging.warning(f"Content for blob_hash {blob_hash} not found for chunk {text_chunk_id}. Skipping for embedding.")
                continue

            sliced_content = blob_content[start_offset:end_offset]
            token_count = len(tokenizer.encode(sliced_content))

            chunks_for_embedding_jsonl.append({
                "schema": "public", # DBCodeChunk is in public, so vector update targets public.DBCodeChunk
                "id": db_pk_id,     # This is the DBCodeChunk.id (PK)
                "content": sliced_content,
                "token_count": token_count
            })

            # Update the token_count in the DBCodeChunk record itself
            # This assumes 'session' is still active and the objects are part of it,
            # or we need to fetch and update. Since we just queried them, let's try to update.
            # A more robust way is to collect IDs and token_counts and do a bulk update.
            # For simplicity here, let's do individual updates, but this could be slow.
            # A better approach is to add token_count to the initial bulk_insert_mappings if possible,
            # or collect (db_pk_id, token_count) pairs and do a bulk update after the loop.

            # Collect (id, token_count) for bulk update
            # This part will be moved outside the loop for a bulk update.
            # For now, let's prepare data for a bulk update.
            pass # Placeholder - actual update logic will be a bulk update after this loop.

    # After iterating through all chunks and preparing jsonl:
    updates_for_token_counts = []
    for item in chunks_for_embedding_jsonl: # chunks_for_embedding_jsonl now contains schema, id, content, token_count
        updates_for_token_counts.append({'id': item['id'], 'token_count': item['token_count']})

    if updates_for_token_counts:
        logging.debug(f"[public] Preparing to bulk update token_count for {len(updates_for_token_counts)} DBCodeChunks. Repo_id: {repo_id}, branch: {branch}. Sample data: {updates_for_token_counts[:5]}")
        try:
            # The session used for querying 'persisted_chunks_with_db_id' is still in scope.
            # This session is for the 'public' schema.
            session.bulk_update_mappings(DBCodeChunk, updates_for_token_counts)
            logging.info(f"[public] DBCodeChunk.token_count bulk_update_mappings call executed for {len(updates_for_token_counts)} items for repo_id: {repo_id}, branch: {branch}. Session will commit on block exit.")
            # Note: Actual commit happens when the `with db_manager.get_session("public") as session:` block exits successfully.
        except Exception as e_token_update:
            logging.error(f"[public] CRITICAL: Error during bulk_update_mappings for DBCodeChunk.token_count. Repo_id: {repo_id}, branch: {branch}. Error: {e_token_update}", exc_info=True)
            # This error could lead to token_counts not being persisted.
    else:
        logging.info(f"[public] No token count updates to perform for DBCodeChunks for repo_id: {repo_id}, branch: {branch}.")

    if not chunks_for_embedding_jsonl:
        logging.warning(f"[{schema_name if schema_name else 'unknown_schema'}] No chunks prepared for embedding jsonl file for repo_id: {repo_id}, branch: {branch} (this means updates_for_token_counts was also empty)")
        return None

    jsonl_file_path = cache_path / f"embed_{schema_name}_{repo_id}_{branch.replace('/', '_')}.jsonl" # More specific name
    with open(jsonl_file_path, 'w', encoding='utf-8') as f:
        for entry in chunks_for_embedding_jsonl:
            f.write(json.dumps(entry) + '\n')

    logging.info(f"[{schema_name}] Prepared {len(chunks_for_embedding_jsonl)} chunks for embedding in {jsonl_file_path}")
    return str(jsonl_file_path), len(chunks_for_embedding_jsonl)


def _persist_dgraph_nodes(db_manager: DatabaseManager, repo_id: int, branch: str, dgraph_nodes_data: List[Dict[str, Any]], task_id: int, updater: StatusUpdater):
    if not dgraph_nodes_data: # dgraph_nodes_data is list of mutation dicts
        updater(task_id, "No graph nodes to persist.", 50.0, None, {'Nodes Persisted': '0/0'})
        return 0

    # Worker already added repo_id, branch, and all offset fields to each dict in dgraph_nodes_data.
    # Dgraph upsert logic (based on 'node_id' predicate usually) handles uniqueness.

    total_persisted_count = 0
    total_nodes_to_submit = len(dgraph_nodes_data)
    progress_interval = 50.0 # This task's share of progress for the sub-task
    updater(task_id, f"Starting to persist {total_nodes_to_submit} graph nodes to Dgraph...", 0.0, None, {'Total Dgraph Nodes': total_nodes_to_submit})

    for i, batch_of_mutations in enumerate(_create_batches(dgraph_nodes_data, IngestionConfig.DGRAPH_BATCH_SIZE)):
        if not batch_of_mutations: continue

        # db_manager.execute_dgraph_mutations should handle the set_obj=batch_of_mutations
        # It's assumed this function correctly forms and executes the Dgraph transaction.
        db_manager.execute_dgraph_mutations(batch_of_mutations)
        total_persisted_count += len(batch_of_mutations)

        processed_count = (i + 1) * IngestionConfig.DGRAPH_BATCH_SIZE
        current_progress = (min(processed_count, total_nodes_to_submit) / total_nodes_to_submit) * progress_interval if total_nodes_to_submit > 0 else 0
        updater(task_id, f"Persisting Dgraph nodes batch {i+1}...", current_progress, None, {'Nodes Persisted': f"{total_persisted_count}/{total_nodes_to_submit}"})

    logging.info(f"Attempted to persist {total_persisted_count} Dgraph nodes for repo {repo_id}, branch {branch}.")
    # updater(task_id, "Dgraph node persistence complete.", progress_interval, None, {'Nodes Persisted': f"{total_persisted_count}/{total_nodes_to_submit}"}) # Final update for this part
    return total_persisted_count


def _persist_vector_batch(db_manager: DatabaseManager, batch_results: List[Tuple[str, int, Any, int]]) -> int:
    # batch_results: List of (schema_name, db_chunk_id PK, embedding_vector, token_count)
    # schema_name is now always "public" because DBCodeChunk is in public schema.

    vector_updates_public_schema: List[Dict[str, Any]] = []
    total_tokens_processed = 0

    for schema_name_recv, chunk_db_pk_id, embedding_data, token_c in batch_results:
        if schema_name_recv != "public":
            logging.warning(f"Vector batch received for unexpected schema: {schema_name_recv}. Expected 'public'. Skipping update for DBCodeChunk.id {chunk_db_pk_id}")
            continue
        if embedding_data: # Ensure embedding is not None or empty
            vector_updates_public_schema.append({'id': chunk_db_pk_id, 'embedding': embedding_data}) # embedding_data should be list from .tolist()
        total_tokens_processed += token_c # Summing tokens from all processed chunks in batch

    if vector_updates_public_schema:
        vector_updates_public_schema.sort(key=lambda x: x['id']) # Good for DB performance
        max_retries = 3
        for attempt in range(max_retries):
            try:
                with db_manager.get_session("public") as s: # Explicitly use "public" schema session
                    # Assuming DBCodeChunk model has 'embedding' field of correct pgvector type
                    s.bulk_update_mappings(DBCodeChunk, vector_updates_public_schema)
                    logging.info(f"Updated {len(vector_updates_public_schema)} vector embeddings in public.DBCodeChunk.")
                    break # Success
            except Exception as e:
                if "deadlock detected" in str(e).lower() and attempt < max_retries - 1:
                    import time, random
                    wait_time = (0.1 * (2 ** attempt)) + random.uniform(0, 0.1)
                    logging.warning(f"Deadlock on public.DBCodeChunk vector update, retrying in {wait_time:.3f}s (Attempt {attempt+1})")
                    time.sleep(wait_time)
                else:
                    logging.error(f"Failed to update vector batch in public.DBCodeChunk: {e}", exc_info=True)
                    # Depending on policy, might raise or just log and skip batch
                    raise # Re-raise if not a deadlock or retries exhausted
    return total_tokens_processed


def _process_graph_edges_streaming(db_manager: DatabaseManager, definitions_files: List[str], calls_files: List[str], task_id: int, updater: StatusUpdater):
    if not definitions_files or not calls_files:
        logging.info("No definition or call files for graph edge processing.")
        updater(task_id, "No definition/call files for edges.", 95.0, None, {'Edges Created': 0})
        return 0

    updater(task_id, "Building definitions index for graph edges...", 55.0, None, {'Definitions Files': len(definitions_files)})
    all_definitions: Dict[str, List[str]] = defaultdict(list) # name -> list of node_ids
    total_defs_loaded = 0
    for i, def_file_path in enumerate(definitions_files):
        try:
            p = Path(def_file_path)
            if p.exists() and p.stat().st_size > 0:
                with open(def_file_path, 'r', encoding='utf-8') as f:
                    defs_list = json.load(f) # List of [name, node_id]
                    if isinstance(defs_list, list):
                        for name, node_id_val in defs_list:
                            all_definitions[name].append(node_id_val)
                        total_defs_loaded += len(defs_list)
        except Exception as e:
            logging.error(f"Error reading definitions file {def_file_path}: {e}", exc_info=True)
        if i % 50 == 0 or i == len(definitions_files) - 1: # Update progress periodically
            progress = 55.0 + (( (i+1) / len(definitions_files)) * 15.0) # Definitions loading: 55-70%
            updater(task_id, f"Processing definitions files {i+1}/{len(definitions_files)}...", progress, None, {'Definitions Loaded': total_defs_loaded})

    if not all_definitions:
        updater(task_id, "No definitions found for edge processing.", 95.0, None, {'Total Definitions': 0, 'Edges Created': 0})
        return 0
    logging.info(f"Built definitions index with {len(all_definitions)} unique symbols, {total_defs_loaded} total definitions.")

    updater(task_id, "Processing calls and creating Dgraph edges...", 70.0, None, {'Unique Symbols': len(all_definitions), 'Total Definitions': total_defs_loaded})
    total_edges_created, processed_c_files, edge_mutations_batch = 0, 0, []
    DGRAPH_EDGE_BATCH_SIZE = IngestionConfig.DGRAPH_BATCH_SIZE # Use configured Dgraph batch size

    for call_file_path in calls_files:
        try:
            p = Path(call_file_path)
            if p.exists() and p.stat().st_size > 0:
                with open(call_file_path, 'r', encoding='utf-8') as f:
                    calls_list = json.load(f) # List of [caller_node_id, called_name]
                    if isinstance(calls_list, list):
                        for caller_node_id, called_name_val in calls_list:
                            if called_name_val in all_definitions:
                                for callee_node_id_val in all_definitions[called_name_val]:
                                    # Dgraph mutation for edge: caller --calls--> callee
                                    edge_mutations_batch.append({'uid': f"_:{caller_node_id}", "calls": {"uid": f"_:{callee_node_id_val}"}})
                                    if len(edge_mutations_batch) >= DGRAPH_EDGE_BATCH_SIZE:
                                        db_manager.execute_dgraph_mutations(edge_mutations_batch)
                                        total_edges_created += len(edge_mutations_batch)
                                        edge_mutations_batch = []
        except Exception as e:
            logging.error(f"Error reading calls file {call_file_path}: {e}", exc_info=True)

        processed_c_files += 1
        if processed_c_files % 50 == 0 or processed_c_files == len(calls_files): # Update progress
            progress = 70.0 + (((processed_c_files) / len(calls_files)) * 25.0) # Calls processing: 70-95%
            updater(task_id, f"Processing calls files {processed_c_files}/{len(calls_files)}...", progress, None, {'Edges Created': total_edges_created, 'Call Files Processed': processed_c_files})

    if edge_mutations_batch: # Process any remaining mutations
        db_manager.execute_dgraph_mutations(edge_mutations_batch)
        total_edges_created += len(edge_mutations_batch)

    logging.info(f"Created {total_edges_created} graph edges from {processed_c_files} call files.")
    # Final update for this part is handled by the calling graph task function
    return total_edges_created
