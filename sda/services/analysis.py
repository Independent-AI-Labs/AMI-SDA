# sda/services/analysis.py

import hashlib
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager, asynccontextmanager
from typing import Dict, List, Optional, Any, AsyncGenerator, Generator

import google.generativeai as genai
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.base.llms.types import ChatResponse, CompletionResponse, LLMMetadata
from llama_index.core.vector_stores import ExactMatchFilter, MetadataFilters
from llama_index.llms.gemini import Gemini
from llama_index.vector_stores.postgres import PGVectorStore
from pydantic import PrivateAttr
from sqlalchemy.orm import joinedload

from sda.config import AIConfig, GOOGLE_API_KEY, DATA_DIR, IngestionConfig
from sda.core.config_models import LLMConfig
from sda.core.data_models import DuplicatePair, SemanticSearchResult # Added import
from sda.core.db_management import DatabaseManager
from sda.core.models import DBCodeChunk, Repository, BillingUsage
from sda.utils.limiter import RateLimiter

# Module-level lock for genai.configure() is now in llm_clients.py with RateLimitedGemini
# _GEMINI_CONFIG_LOCK = threading.Lock() # Removed

from sda.services.llm_clients import RateLimitedGemini # Added import

def resolve_embedding_devices() -> List[str]:
    """Detects and resolves available hardware for embedding, supporting XPU, CUDA, and ROCm."""
    raw_devices_config = AIConfig.EMBEDDING_DEVICES
    detected_devices = []
    use_auto = "auto" in raw_devices_config

    try:
        import torch
    except ImportError:
        logging.warning("PyTorch is not installed. Falling back to CPU for all operations.")
        return ["cpu"]

    # --- Intel XPU Detection ---
    if use_auto or "xpu" in raw_devices_config:
        try:
            import intel_extension_for_pytorch as ipex
            if torch.xpu.is_available():
                xpu_count = torch.xpu.device_count()
                if xpu_count > 0:
                    logging.info(f"Found {xpu_count} Intel XPU devices.")
                    detected_devices.extend([f"xpu:{i}" for i in range(xpu_count)])
        except ImportError:
            pass

    if use_auto and detected_devices: return detected_devices

    # --- NVIDIA CUDA Detection ---
    if use_auto or "cuda" in raw_devices_config:
        if torch.cuda.is_available() and not (hasattr(torch.version, 'hip') and torch.version.hip):
            cuda_count = torch.cuda.device_count()
            if cuda_count > 0:
                logging.info(f"Found {cuda_count} NVIDIA CUDA devices.")
                detected_devices.extend([f"cuda:{i}" for i in range(cuda_count)])

    if use_auto and detected_devices: return detected_devices

    # --- AMD ROCm Detection ---
    if use_auto or "rocm" in raw_devices_config:
        if torch.cuda.is_available() and hasattr(torch.version, 'hip') and torch.version.hip:
            rocm_count = torch.cuda.device_count()
            if rocm_count > 0:
                logging.info(f"Found {rocm_count} AMD ROCm devices.")
                detected_devices.extend([f"cuda:{i}" for i in range(rocm_count)])

    if detected_devices: return detected_devices

    logging.info("No supported GPU devices (XPU/CUDA/ROCm) found or configured. Falling back to CPU.")
    return ["cpu"]


class EnhancedAnalysisEngine:
    """Orchestrates RAG, semantic search, and other code analysis tasks."""

    def __init__(self, db_manager: DatabaseManager, rate_limiter: RateLimiter):
        self.db_manager = db_manager
        self.rate_limiter = rate_limiter
        self._embedding_model = None
        self._resolved_devices = None
        self.embedding_config = AIConfig.get_active_embedding_config()

        if not GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY not found in environment variables.")

        active_llm_config = AIConfig.get_active_llm_config()
        Settings.llm = RateLimitedGemini(
            db_manager=self.db_manager,
            rate_limiter=self.rate_limiter,
            model=active_llm_config.model_name
        )
        self.vector_store_cache: Dict[str, PGVectorStore] = {}
        logging.info("EnhancedAnalysisEngine initialized (models will be lazy-loaded).")

    def _get_resolved_devices(self) -> List[str]:
        if self._resolved_devices is None:
            self._resolved_devices = resolve_embedding_devices()
        return self._resolved_devices

    def _get_embedding_model(self):
        if self._embedding_model is None:
            from sentence_transformers import SentenceTransformer
            primary_device = self._get_resolved_devices()[0]
            self._embedding_model = SentenceTransformer(
                model_name_or_path=self.embedding_config.model_name,
                device=primary_device,
                cache_folder=str(DATA_DIR / "embedding_models")
            )
        return self._embedding_model

    def _get_vector_store_for_schema(self, schema_name: str) -> PGVectorStore:
        if schema_name in self.vector_store_cache:
            return self.vector_store_cache[schema_name]

        vector_store = PGVectorStore(
            engine=self.db_manager.engine, schema_name=schema_name,
            table_name=AIConfig.VECTOR_COLLECTION_NAME,
            embed_dim=self.embedding_config.dimension,
        )
        self.vector_store_cache[schema_name] = vector_store
        return vector_store

    def _search_in_schema(self, schema: str, branch: str, top_k: int, query_embedding: List[float]) -> List[SemanticSearchResult]:
        vector_store = self._get_vector_store_for_schema(schema)
        index = VectorStoreIndex.from_vector_store(vector_store)
        filters = MetadataFilters(filters=[ExactMatchFilter(key="branch", value=branch)])
        retriever = index.as_retriever(similarity_top_k=top_k, vector_store_query_mode="default", filters=filters)
        nodes_with_scores = retriever.retrieve_from_embedding(query_embedding)
        chunk_ids = [node.metadata['chunk_id'] for node in nodes_with_scores if 'chunk_id' in node.metadata]
        if not chunk_ids: return []

        scores = {node.metadata['chunk_id']: node.score for node in nodes_with_scores if 'chunk_id' in node.metadata}
        with self.db_manager.get_session(schema) as session:
            chunks = session.query(DBCodeChunk).filter(DBCodeChunk.chunk_id.in_(chunk_ids)).options(joinedload(DBCodeChunk.file)).all()
            return [SemanticSearchResult(
                file_path=c.file.relative_path, content=c.content, start_line=c.start_line,
                end_line=c.end_line, score=scores.get(c.chunk_id, 0.0)
            ) for c in chunks]

    def _execute_parallel_search(self, repo_id: int, branch: str, top_k: int, query_embedding: List[float]) -> List[SemanticSearchResult]:
        with self.db_manager.get_session("public") as public_session:
            repo = public_session.get(Repository, repo_id)
            if not repo or not repo.db_schemas: return []
            schemas = repo.db_schemas

        all_results = []
        with ThreadPoolExecutor(max_workers=IngestionConfig.MAX_DB_WORKERS) as executor:
            futures = {executor.submit(self._search_in_schema, schema, branch, top_k, query_embedding): schema for schema in schemas}
            for future in as_completed(futures):
                try:
                    all_results.extend(future.result())
                except Exception as e:
                    logging.error(f"Search failed for schema {futures[future]}: {e}")
        return sorted(all_results, key=lambda r: r.score, reverse=True)[:top_k]

    def search_chunks_by_symbol(self, repo_id: int, branch: str, symbol_name: str, top_k: int = 5) -> List[SemanticSearchResult]:
        model = self._get_embedding_model()
        embedding = model.encode(symbol_name).tolist()
        return self._execute_parallel_search(repo_id, branch, top_k, embedding)

    def find_similar_chunks_by_snippet(self, repo_id: int, branch: str, code_snippet: str, top_k: int = 5) -> List[SemanticSearchResult]:
        model = self._get_embedding_model()
        embedding = model.encode(code_snippet).tolist()
        return self._execute_parallel_search(repo_id, branch, top_k, embedding)

    def find_duplicate_code(self, repo_id: int, branch: str, similarity_threshold: float = 0.98, top_k: int = 5) -> List[DuplicatePair]:
        import torch
        from sentence_transformers import util

        with self.db_manager.get_session("public") as public_session:
            repo = public_session.get(Repository, repo_id)
            if not repo or not repo.db_schemas: return []
            schemas = repo.db_schemas

        all_chunks = []
        for schema in schemas:
            with self.db_manager.get_session(schema) as session:
                chunks_in_schema = session.query(DBCodeChunk).options(joinedload(DBCodeChunk.file)).filter(
                    DBCodeChunk.repository_id == repo_id, DBCodeChunk.branch == branch, DBCodeChunk.embedding.isnot(None)
                ).all()
                all_chunks.extend(chunks_in_schema)

        if len(all_chunks) < 2: return []

        corpus_embeddings = torch.tensor([c.embedding for c in all_chunks], dtype=torch.float32)
        hits = util.semantic_search(corpus_embeddings, corpus_embeddings, top_k=top_k + 1, score_function=util.cos_sim)
        duplicate_pairs: List[DuplicatePair] = []
        processed_pairs = set()

        for i, hit_list in enumerate(hits):
            chunk_a = all_chunks[i]
            for hit in hit_list[1:]:
                if hit['score'] >= similarity_threshold:
                    chunk_b = all_chunks[hit['corpus_id']]
                    pair_key = tuple(sorted((chunk_a.id, chunk_b.id)))
                    if pair_key not in processed_pairs:
                        processed_pairs.add(pair_key)
                        duplicate_pairs.append(DuplicatePair(
                            chunk_a_id=chunk_a.chunk_id, chunk_b_id=chunk_b.chunk_id,
                            file_a=chunk_a.file.relative_path, file_b=chunk_b.file.relative_path,
                            lines_a=f"{chunk_a.start_line}-{chunk_a.end_line}", lines_b=f"{chunk_b.start_line}-{chunk_b.end_line}",
                            similarity=hit['score'], content_a=chunk_a.content, content_b=chunk_b.content
                        ))
        return sorted(duplicate_pairs, key=lambda p: p.similarity, reverse=True)