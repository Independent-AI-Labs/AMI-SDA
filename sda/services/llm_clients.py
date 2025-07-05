# sda/services/llm_clients.py

import hashlib
import logging
import threading
from contextlib import contextmanager, asynccontextmanager
from typing import Optional, Any, AsyncGenerator, Generator

import google.generativeai as genai
from llama_index.core.base.llms.types import ChatResponse, CompletionResponse
from llama_index.llms.gemini import Gemini
from pydantic import PrivateAttr

from sda.config import AIConfig # For LLMConfig and get_all_llm_configs
from sda.core.config_models import LLMConfig
from sda.core.db_management import DatabaseManager
from sda.core.models import BillingUsage
from sda.utils.limiter import RateLimiter

# Module-level lock to protect the global genai.configure() call.
# This lock is specific to Gemini client configuration.
_GEMINI_CONFIG_LOCK = threading.Lock()


class RateLimitedGemini(Gemini):
    """
    A wrapper for the Gemini LLM to enforce rate limiting, API key rotation,
    and automatic billing/usage tracking. It uses PrivateAttr for internal state
    to avoid conflicts with the parent Pydantic model.
    """
    _db_manager: DatabaseManager = PrivateAttr()
    _rate_limiter: RateLimiter = PrivateAttr()
    _model_config: LLMConfig = PrivateAttr()

    def __init__(self, db_manager: DatabaseManager, rate_limiter: RateLimiter, **kwargs: Any):
        super().__init__(**kwargs)
        self._db_manager = db_manager
        self._rate_limiter = rate_limiter

        model_name = kwargs.get("model")
        if not model_name:
            raise ValueError("RateLimitedGemini requires 'model' to be provided.")

        # TODO: Review if AIConfig.get_all_llm_configs() is the best way, or pass LLMConfig directly
        model_config = AIConfig.get_all_llm_configs().get(model_name)
        if not model_config:
            raise ValueError(f"No configuration found for model '{model_name}'.")
        self._model_config = model_config

        with _GEMINI_CONFIG_LOCK:
            initial_api_key = self._rate_limiter.acquire(model_name=model_name)
            # It's important that genai.configure is thread-safe if called from multiple instances
            # or that instances are created in a controlled manner.
            genai.configure(api_key=initial_api_key)

    def _record_billing_usage(self, api_key: str, response: Optional[ChatResponse] = None):
        if not response or self._model_config.provider != 'google':
            return
        usage_metadata = None
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            usage_metadata = response.usage_metadata
        elif hasattr(response, 'metadata') and response.metadata and hasattr(response.metadata, 'usage_metadata'):
            usage_metadata = response.metadata.usage_metadata
        elif hasattr(response, 'raw') and response.raw and hasattr(response.raw, 'usage_metadata'):
            usage_metadata = response.raw.usage_metadata

        if not usage_metadata:
            logging.debug("No usage metadata found in response, skipping billing record")
            return
        try:
            prompt_tokens = getattr(usage_metadata, 'prompt_token_count', 0)
            completion_tokens = getattr(usage_metadata, 'completion_token_count', 0) or \
                                getattr(usage_metadata, 'candidates_token_count', 0)
            total_tokens = prompt_tokens + completion_tokens
            cost = (
                (prompt_tokens / 1_000_000) * self._model_config.input_price_per_million_tokens +
                (completion_tokens / 1_000_000) * self._model_config.output_price_per_million_tokens
            )
            usage_record = BillingUsage(
                model_name=self.model, provider=self._model_config.provider,
                api_key_used_hash=hashlib.sha256(api_key.encode()).hexdigest(),
                prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
                total_tokens=total_tokens, cost=cost,
            )
            with self._db_manager.get_session("public") as session:
                session.add(usage_record)
        except Exception as e:
            logging.error(f"Failed to record billing usage: {e}", exc_info=True)

    @contextmanager
    def _rate_limited_context(self) -> Generator[str, None, None]:
        # This lock ensures that genai.configure is not called concurrently by different threads
        # if multiple RateLimitedGemini instances exist and are used in threads.
        with _GEMINI_CONFIG_LOCK:
            api_key = self._rate_limiter.acquire(model_name=self.model)
            try:
                genai.configure(api_key=api_key)
                yield api_key
            finally:
                # Consider if genai.configure needs to be reset or if it's okay for it to
                # hold the last used key globally until the next call.
                # If multiple Gemini models with different keys are used, this needs careful handling.
                pass

    @asynccontextmanager
    async def _arate_limited_context(self) -> AsyncGenerator[str, None]:
        async with self._rate_limiter.async_lock: # Assuming RateLimiter has an async_lock
            with _GEMINI_CONFIG_LOCK: # Still need this for the synchronous genai.configure
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
            for chunk_response in stream: # Renamed 'chunk' to 'chunk_response' to avoid var name collision
                final_response = chunk_response
                yield chunk_response
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
            async for chunk_response in stream: # Renamed 'chunk' to 'chunk_response'
                final_response = chunk_response
                yield chunk_response
            if final_response:
                self._record_billing_usage(api_key, final_response)

    def complete(self, *args: Any, **kwargs: Any) -> CompletionResponse:
        with self._rate_limited_context() as api_key:
            response = super().complete(*args, **kwargs)
            self._record_completion_billing_usage(api_key, response)
            return response

    async def acomplete(self, *args: Any, **kwargs: Any) -> CompletionResponse:
        async with self._arate_limited_context() as api_key:
            response = await super().acomplete(*args, **kwargs)
            self._record_completion_billing_usage(api_key, response)
            return response

    def _record_completion_billing_usage(self, api_key: str, response: Optional[CompletionResponse] = None):
        if not response or self._model_config.provider != 'google':
            return
        try:
            text = response.text or ""
            # This estimation is very rough. Consider if more accurate token counting is needed/possible.
            estimated_tokens = len(text.split()) * 1.3
            cost = (estimated_tokens / 1_000_000) * self._model_config.output_price_per_million_tokens
            usage_record = BillingUsage(
                model_name=self.model, provider=self._model_config.provider,
                api_key_used_hash=hashlib.sha256(api_key.encode()).hexdigest(),
                prompt_tokens=0,  # Prompt tokens are not available for standard completion responses here
                completion_tokens=int(estimated_tokens),
                total_tokens=int(estimated_tokens), cost=cost,
            )
            with self._db_manager.get_session("public") as session:
                session.add(usage_record)
        except Exception as e:
            logging.error(f"Failed to record completion billing usage: {e}", exc_info=True)
