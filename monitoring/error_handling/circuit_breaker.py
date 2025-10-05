"""
Circuit breaker for error rate monitoring.

This module provides circuit breaker functionality to prevent cascading failures
by temporarily disabling components that are experiencing high error rates.
"""

import time
import threading
from typing import Dict, List
from collections import defaultdict


class CircuitBreaker:
    """Circuit breaker for error rate monitoring."""

    def __init__(self, error_threshold: int, time_window: float, reset_timeout: float):
        """Initialize circuit breaker.

        Args:
            error_threshold: Number of errors before circuit breaks
            time_window: Time window in seconds for counting errors
            reset_timeout: Time in seconds before attempting to reset circuit
        """
        self.error_threshold = error_threshold
        self.time_window = time_window
        self.reset_timeout = reset_timeout
        self.error_counts: Dict[str, List[float]] = defaultdict(list)
        self.broken_circuits: Dict[str, float] = {}
        self._lock = threading.Lock()

    def record_error(self, component: str) -> bool:
        """Record an error and check if circuit should break.

        Args:
            component: Name of component experiencing error

        Returns:
            True if circuit broke due to this error, False otherwise
        """
        with self._lock:
            current_time = time.time()

            # Clean old errors
            self.error_counts[component] = [
                t for t in self.error_counts[component]
                if current_time - t <= self.time_window
            ]

            # Add new error
            self.error_counts[component].append(current_time)

            # Check if circuit should break
            if len(self.error_counts[component]) >= self.error_threshold:
                self.broken_circuits[component] = current_time
                return True

            return False

    def is_broken(self, component: str) -> bool:
        """Check if circuit is broken for component.

        Args:
            component: Name of component to check

        Returns:
            True if circuit is currently broken, False otherwise
        """
        with self._lock:
            if component not in self.broken_circuits:
                return False

            break_time = self.broken_circuits[component]
            if time.time() - break_time >= self.reset_timeout:
                del self.broken_circuits[component]
                self.error_counts[component].clear()
                return False

            return True
