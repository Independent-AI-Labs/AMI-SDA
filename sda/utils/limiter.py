# sda/utils/limiter.py

import asyncio
import logging
import time
from asyncio import Lock as AsyncLock
from collections import deque
from threading import Lock
from typing import List, Optional, Dict

# CORRECTED: Import LLMConfig from the new config_models.py file.
from sda.core.config_models import LLMConfig


class RateLimiter:
    """
    A thread-safe class to enforce multiple, model-specific rate limits (e.g., per
    second, minute, hour) and rotate through a list of API keys to maximize throughput.
    """

    def __init__(self, model_configs: Dict[str, LLMConfig], api_keys: Optional[List[str]] = None):
        """
        Initializes the rate limiter using structured configuration.

        Args:
            model_configs: A dictionary mapping model names to their LLMConfig Pydantic model.
            api_keys: An optional list of API keys to rotate through.
        """
        self.model_configs = model_configs
        self.api_keys = api_keys if (api_keys and api_keys[0]) else ["__default__"]

        # Structure: {api_key: {model_name: [deque_for_limit1, deque_for_limit2, ...]}}
        self.timestamps: Dict[str, Dict[str, List[deque]]] = {
            key: {
                model_name: [deque() for _ in config.rate_limits]
                for model_name, config in self.model_configs.items()
            }
            for key in self.api_keys
        }

        self.current_key_index = 0
        self.lock = Lock()
        self.async_lock = AsyncLock()

        logging.info(f"RateLimiter initialized for {len(self.api_keys)} API key(s) with model-specific configs.")

    def _get_config_for_model(self, model_name: str) -> Optional[LLMConfig]:
        """Returns the configuration object for a model."""
        return self.model_configs.get(model_name)

    def _check_and_get_wait_time(self, key: str, model_name: str, now: float) -> float:
        """Checks all limits for a key and model, and returns the required wait time."""
        config = self._get_config_for_model(model_name)
        if not config:
            return 0.0

        max_wait_time = 0.0
        # Ensure the timestamp structure exists for the given key and model
        if model_name not in self.timestamps[key]:
             self.timestamps[key][model_name] = [deque() for _ in config.rate_limits]

        model_timestamps = self.timestamps[key][model_name]

        for i, limit in enumerate(config.rate_limits):
            key_timestamps = model_timestamps[i]
            
            # Clean up old timestamps
            while key_timestamps and key_timestamps[0] <= now - limit.period_seconds:
                key_timestamps.popleft()

            # Check if the limit is exceeded
            if len(key_timestamps) >= limit.requests:
                # Calculate how long to wait until the oldest request expires
                wait_time = (key_timestamps[0] + limit.period_seconds) - now
                if wait_time > max_wait_time:
                    max_wait_time = wait_time
        return max_wait_time

    def _record_request(self, key: str, model_name: str, now: float):
        """Records a new request timestamp for the given key and model."""
        config = self._get_config_for_model(model_name)
        if not config:
            return
            
        model_timestamps = self.timestamps[key][model_name]
        for i in range(len(config.rate_limits)):
            model_timestamps[i].append(now)

    def acquire(self, model_name: str) -> str:
        """Blocks until an API key is available for the given model, then returns it."""
        config = self._get_config_for_model(model_name)
        if not config:
            return self.api_keys[0]

        with self.lock:
            while True:
                # Try to find a ready key by rotating through them
                for i in range(len(self.api_keys)):
                    key_index = (self.current_key_index + i) % len(self.api_keys)
                    key = self.api_keys[key_index]

                    wait_time = self._check_and_get_wait_time(key, model_name, time.monotonic())

                    if wait_time <= 0:
                        self._record_request(key, model_name, time.monotonic())
                        self.current_key_index = key_index + 1  # Start next search from the next key
                        return key

                # All keys are busy, calculate the minimum wait time and sleep
                now = time.monotonic()
                min_wait_time = float('inf')
                for key in self.api_keys:
                    wait_time = self._check_and_get_wait_time(key, model_name, now)
                    if wait_time < min_wait_time:
                        min_wait_time = wait_time

                sleep_time = max(min_wait_time, 0) + 0.01
                logging.warning(f"All API keys are rate-limited for model '{model_name}'. Waiting for {sleep_time:.2f} seconds.")
                time.sleep(sleep_time)

    async def aacquire(self, model_name: str) -> str:
        """Asynchronously waits until an API key is available for a model, then returns it."""
        config = self._get_config_for_model(model_name)
        if not config:
            return self.api_keys[0]

        async with self.async_lock:
            while True:
                for i in range(len(self.api_keys)):
                    key_index = (self.current_key_index + i) % len(self.api_keys)
                    key = self.api_keys[key_index]

                    wait_time = self._check_and_get_wait_time(key, model_name, time.monotonic())

                    if wait_time <= 0:
                        self._record_request(key, model_name, time.monotonic())
                        self.current_key_index = key_index + 1
                        return key

                now = time.monotonic()
                min_wait_time = float('inf')
                for key in self.api_keys:
                    wait_time = self._check_and_get_wait_time(key, model_name, now)
                    if wait_time < min_wait_time:
                        min_wait_time = wait_time

                sleep_time = max(min_wait_time, 0) + 0.01
                logging.warning(f"All API keys are rate-limited for model '{model_name}'. Waiting for {sleep_time:.2f} seconds.")
                await asyncio.sleep(sleep_time)