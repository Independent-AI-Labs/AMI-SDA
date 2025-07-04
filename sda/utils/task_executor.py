# sda/utils/task_executor.py

"""
Provides a centralized, robust manager for concurrent task execution.

This module implements the Bulkhead pattern by creating separate, isolated
ThreadPoolExecutors for different types of I/O-bound workloads (e.g.,
Postgres, Dgraph). This prevents a slowdown in one external service from
impacting the performance of others.
"""

import logging
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Dict, Callable, Any

from sda.config import IngestionConfig

class TaskExecutor:
    """Manages isolated thread pools for different workload types."""

    def __init__(self):
        """
        Initializes the TaskExecutor and creates the dedicated thread pools
        based on the MAX_DB_WORKERS_PER_TARGET configuration.
        """
        self._executors: Dict[str, ThreadPoolExecutor] = {}
        worker_configs = IngestionConfig.MAX_DB_WORKERS_PER_TARGET

        for target, num_workers in worker_configs.items():
            pool_size = max(1, num_workers)
            self._executors[target] = ThreadPoolExecutor(
                max_workers=pool_size,
                thread_name_prefix=f'{target}_worker'
            )
            logging.info(
                f"TaskExecutor initialized '{target}' pool with {pool_size} workers."
            )

    def submit(self, workload_type: str, fn: Callable, *args: Any, **kwargs: Any) -> Future:
        """
        Submits a function to the appropriate workload-specific thread pool.

        Args:
            workload_type: The type of workload (e.g., 'postgres' or 'dgraph').
                           Must match a key in IngestionConfig.MAX_DB_WORKERS_PER_TARGET.
            fn: The function to execute.
            *args: Positional arguments for the function.
            **kwargs: Keyword arguments for the function.

        Returns:
            A Future object representing the execution of the task.
        """
        if workload_type not in self._executors:
            raise ValueError(f"Unknown workload_type: '{workload_type}'. Must be one of {list(self._executors.keys())}")
        
        executor = self._executors[workload_type]
        return executor.submit(fn, *args, **kwargs)

    def shutdown(self, wait: bool = True):
        """Shuts down all managed thread pools."""
        logging.info("Shutting down all task executor pools...")
        for name, executor in self._executors.items():
            executor.shutdown(wait=wait)
            logging.info(f"'{name}' pool shut down.")