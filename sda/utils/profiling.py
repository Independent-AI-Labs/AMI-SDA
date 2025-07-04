# sda/utils/profiling.py

import logging
import time
from collections import defaultdict


class ThroughputLogger:
    """A helper class to log performance metrics periodically."""

    def __init__(self, name: str, log_interval_sec: float = 10.0):
        """
        Initializes the logger.

        Args:
            name: The name of the process being logged (e.g., "Parsing").
            log_interval_sec: How often to log, in seconds.
        """
        self.name = name
        self.log_interval_sec = log_interval_sec
        self.start_time = time.perf_counter()
        self.last_log_time = self.start_time
        # Counters for the entire run
        self.total_items = defaultdict(int)
        # Counters for the last logged interval
        self.last_logged_items = defaultdict(int)

    def update(self, **kwargs):
        """
        Updates the counters with new items processed.
        Example: logger.update(files=1, chunks=10, tokens=512)
        """
        for key, value in kwargs.items():
            self.total_items[key] += value
        self._maybe_log()

    def _maybe_log(self):
        """Logs metrics if the interval has passed."""
        now = time.perf_counter()
        if (now - self.last_log_time) >= self.log_interval_sec:
            self._log_metrics(now)
            self.last_log_time = now
            for key, value in self.total_items.items():
                self.last_logged_items[key] = value

    def _log_metrics(self, now: float):
        """Formats and logs the throughput metrics."""
        elapsed_total = now - self.start_time
        elapsed_interval = now - self.last_log_time
        if elapsed_interval < 1e-6:
            return  # Avoid division by zero

        metrics_str = [f"[{self.name} Progress]"]
        for key, total_count in self.total_items.items():
            interval_count = total_count - self.last_logged_items.get(key, 0)
            mean_rate = total_count / elapsed_total if elapsed_total > 0 else 0
            inst_rate = interval_count / elapsed_interval if elapsed_interval > 0 else 0
            metrics_str.append(
                f"{key.capitalize()}: {total_count} ({inst_rate:.1f}↗, "
                f"mean {mean_rate:.1f} {key}/s)"
            )
        logging.info(" | ".join(metrics_str))

    def final_log(self):
        """Logs the final total and mean metrics for the entire run."""
        self._log_metrics(time.perf_counter())