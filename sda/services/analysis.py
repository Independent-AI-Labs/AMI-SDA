# sda/services/analysis.py

import hashlib
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager, asynccontextmanager
from dataclasses import dataclass
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
from sda.core.db_management import DatabaseManager
from sda.core.models import DBCodeChunk, Repository, BillingUsage
from sda.utils.limiter import RateLimiter

# Module-level lock to protect the global genai.configure() call.
_GEMINI_CONFIG_LOCK = threading.Lock()


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


@dataclass
class DuplicatePair:
    """Represents a pair of semantically similar code chunks."""
    chunk_a_id: str
    chunk_b_id: str
    file_a: str
    file_b: str
    lines_a: str
    lines_b: str
    similarity: float
    content_a: str
    content_b: str


@dataclass
class SemanticSearchResult:
    """Represents a single result from a semantic search."""
    file_path: str
    content: str
    start_line: Optional[int]
    end_line: Optional[int]
    score: float


class RateLimitedGemini(Gemini):
    """
    A wrapper for the Gemini LLM to enforce rate limiting, API key rotation,
    and automatic billing/usage tracking. It uses PrivateAttr for internal state
    to avoid conflicts with the parent Pydantic model.
    """
    # Declare private attributes that are not part of the Pydantic model schema.
    _db_manager: DatabaseManager = PrivateAttr()
    _rate_limiter: RateLimiter = PrivateAttr()
    _model_config: LLMConfig = PrivateAttr()

    def __init__(self, db_manager: DatabaseManager, rate_limiter: RateLimiter, **kwargs: Any):
        # Call the parent Pydantic model's initializer first.
        super().__init__(**kwargs)

        # Now, safely set the internal attributes.
        self._db_manager = db_manager
        self._rate_limiter = rate_limiter

        model_name = kwargs.get("model")
        if not model_name:
            raise ValueError("RateLimitedGemini requires 'model' to be provided.")

        model_config = AIConfig.get_all_llm_configs().get(model_name)
        if not model_config:
            raise ValueError(f"No configuration found for model '{model_name}'.")
        self._model_config = model_config

        # Configure the global genai client inside a lock.
        with _GEMINI_CONFIG_LOCK:
            initial_api_key = self._rate_limiter.acquire(model_name=model_name)
            genai.configure(api_key=initial_api_key)

    def _record_billing_usage(self, api_key: str, response: Optional[ChatResponse] = None):
        """Records token usage and calculated cost into the database."""
        if not response or self._model_config.provider != 'google':
            return

        # Try to get usage metadata from different possible locations
        usage_metadata = None

        # Check for usage_metadata attribute directly
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            usage_metadata = response.usage_metadata
        # Check for metadata attribute with usage_metadata
        elif hasattr(response, 'metadata') and response.metadata and hasattr(response.metadata, 'usage_metadata'):
            usage_metadata = response.metadata.usage_metadata
        # Check for raw attribute that might contain usage info
        elif hasattr(response, 'raw') and response.raw:
            if hasattr(response.raw, 'usage_metadata'):
                usage_metadata = response.raw.usage_metadata

        if not usage_metadata:
            # Log that we couldn't find usage metadata but don't fail
            logging.debug("No usage metadata found in response, skipping billing record")
            return

        try:
            prompt_tokens = getattr(usage_metadata, 'prompt_token_count', 0)
            completion_tokens = getattr(usage_metadata, 'completion_token_count', 0) or getattr(usage_metadata, 'candidates_token_count', 0)
            total_tokens = prompt_tokens + completion_tokens

            cost = (
                    (prompt_tokens / 1_000_000) * self._model_config.input_price_per_million_tokens +
                    (completion_tokens / 1_000_000) * self._model_config.output_price_per_million_tokens
            )

            usage_record = BillingUsage(
                model_name=self.model,
                provider=self._model_config.provider,
                api_key_used_hash=hashlib.sha256(api_key.encode()).hexdigest(),
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                cost=cost,
            )
            with self._db_manager.get_session("public") as session:
                session.add(usage_record)
        except Exception as e:
            logging.error(f"Failed to record billing usage: {e}", exc_info=True)

    @contextmanager
    def _rate_limited_context(self) -> Generator[str, None, None]:
        with _GEMINI_CONFIG_LOCK:
            api_key = self._rate_limiter.acquire(model_name=self.model)
            try:
                genai.configure(api_key=api_key)
                yield api_key
            finally:
                pass

    @asynccontextmanager
    async def _arate_limited_context(self) -> AsyncGenerator[str, None]:
        async with self._rate_limiter.async_lock:
            with _GEMINI_CONFIG_LOCK:
                api_key = await self._rate_limiter.aacquire(model_name=self.model)
                try:
                    genai.configure(api_key=api_key)
                    yield api_key
                finally:
                    pass

    def chat(self, *args: Any, **kwargs: Any) -> ChatResponse:
        with self._rate_limited_context() as api_key:
            response = super().chat(*args, **kwargs)
            self._record_billing_usage(api_key, response)
            return response

    def stream_chat(self, *args: Any, **kwargs: Any) -> Generator[ChatResponse, None, None]:
        with self._rate_limited_context() as api_key:
            stream = super().stream_chat(*args, **kwargs)
            final_response = None
            for chunk in stream:
                final_response = chunk
                yield chunk
            if final_response:
                self._record_billing_usage(api_key, final_response)

    async def achat(self, *args: Any, **kwargs: Any) -> ChatResponse:
        async with self._arate_limited_context() as api_key:
            response = await super().achat(*args, **kwargs)
            self._record_billing_usage(api_key, response)
            return response

    async def astream_chat(self, *args: Any, **kwargs: Any) -> AsyncGenerator[ChatResponse, None]:
        async with self._arate_limited_context() as api_key:
            stream = await super().astream_chat(*args, **kwargs)
            final_response = None
            async for chunk in stream:
                final_response = chunk
                yield chunk
            if final_response:
                self._record_billing_usage(api_key, final_response)

    def complete(self, *args: Any, **kwargs: Any) -> CompletionResponse:
        with self._rate_limited_context() as api_key:
            response = super().complete(*args, **kwargs)
            # For completion responses, we need to handle them differently
            self._record_completion_billing_usage(api_key, response)
            return response

    async def acomplete(self, *args: Any, **kwargs: Any) -> CompletionResponse:
        async with self._arate_limited_context() as api_key:
            response = await super().acomplete(*args, **kwargs)
            self._record_completion_billing_usage(api_key, response)
            return response

    def _record_completion_billing_usage(self, api_key: str, response: Optional[CompletionResponse] = None):
        """Records billing usage for completion responses."""
        if not response or self._model_config.provider != 'google':
            return

        try:
            # For completion responses, we'll estimate token usage based on text length
            # This is not as accurate as getting actual usage data, but it's better than nothing
            text = response.text or ""
            estimated_tokens = len(text.split()) * 1.3  # Rough estimation

            cost = (estimated_tokens / 1_000_000) * self._model_config.output_price_per_million_tokens

            usage_record = BillingUsage(
                model_name=self.model,
                provider=self._model_config.provider,
                api_key_used_hash=hashlib.sha256(api_key.encode()).hexdigest(),
                prompt_tokens=0,  # We don't have prompt token count for completions
                completion_tokens=int(estimated_tokens),
                total_tokens=int(estimated_tokens),
                cost=cost,
            )
            with self._db_manager.get_session("public") as session:
                session.add(usage_record)
        except Exception as e:
            logging.error(f"Failed to record completion billing usage: {e}", exc_info=True)


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