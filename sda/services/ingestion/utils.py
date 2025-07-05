# sda/services/ingestion/utils.py

import json
import logging
from typing import List, Tuple, Generator, Any

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
