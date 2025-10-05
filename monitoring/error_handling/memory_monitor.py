"""
Memory monitoring system for tracking memory usage and handling memory issues.

This module provides utilities for monitoring memory consumption and
triggering actions when memory thresholds are exceeded.
"""

import gc
import time
import logging
import threading
import psutil
from typing import Dict, Optional, Callable


class MemoryMonitor:
    """Memory monitoring system that tracks memory usage and handles memory-related issues.

    Args:
        threshold_mb: Memory usage threshold in megabytes
        check_interval: Time interval between memory checks in seconds
        on_threshold_exceeded: Optional callback when memory threshold is exceeded
    """

    def __init__(
        self,
        threshold_mb: float = 1024.0,
        check_interval: float = 60.0,
        on_threshold_exceeded: Optional[Callable[[], None]] = None
    ):
        self.threshold_mb = threshold_mb
        self.check_interval = check_interval
        self.on_threshold_exceeded = on_threshold_exceeded
        self._monitoring_thread: Optional[threading.Thread] = None
        self._is_monitoring = False
        self._process = psutil.Process()
        self._lock = threading.Lock()

    def get_memory_info(self) -> Dict[str, float]:
        """Get current memory usage information.

        Returns:
            Dict containing memory usage stats in MB:
            - rss_mb: Resident Set Size
            - vms_mb: Virtual Memory Size
            - shared_mb: Shared Memory Size
            - data_mb: Data Segment Size
        """
        mem = self._process.memory_info()
        return {
            'rss_mb': mem.rss / (1024 * 1024),
            'vms_mb': mem.vms / (1024 * 1024),
            'shared_mb': getattr(mem, 'shared', 0) / (1024 * 1024),
            'data_mb': getattr(mem, 'data', 0) / (1024 * 1024)
        }

    def trigger_garbage_collection(self) -> Dict[str, int]:
        """Force garbage collection and return collection statistics.

        Returns:
            Dict containing collection statistics:
            - objects_collected: Total number of objects collected
            - collections: Number of collection runs
        """
        collected = 0
        for i in range(3):  # Run collection for all generations
            collected += gc.collect(i)

        return {
            'objects_collected': collected,
            'collections': 3
        }

    def check_memory_usage(self) -> bool:
        """Check if memory usage exceeds threshold.

        Returns:
            True if memory usage is below threshold, False otherwise.
        """
        mem_info = self.get_memory_info()
        if mem_info['rss_mb'] > self.threshold_mb:
            if self.on_threshold_exceeded:
                self.on_threshold_exceeded()
            return False
        return True

    def start_monitoring(self) -> None:
        """Start memory monitoring in a background thread."""
        with self._lock:
            if self._is_monitoring:
                return

            self._is_monitoring = True
            self._monitoring_thread = threading.Thread(
                target=self._monitoring_loop,
                daemon=True
            )
            self._monitoring_thread.start()

    def stop_monitoring(self) -> None:
        """Stop memory monitoring thread."""
        with self._lock:
            self._is_monitoring = False
            if self._monitoring_thread:
                self._monitoring_thread.join(timeout=5.0)
                self._monitoring_thread = None

    def _monitoring_loop(self) -> None:
        """Main monitoring loop that periodically checks memory usage."""
        while self._is_monitoring:
            try:
                self.check_memory_usage()
            except Exception as e:
                # Log the error but keep monitoring
                logging.error(f"Memory monitoring error: {e}")
            time.sleep(self.check_interval)
