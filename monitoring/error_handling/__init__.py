"""
Error Handling Package

Modular error handling system with centralized error management, circuit breaking,
memory monitoring, and recovery strategies.

Package Structure:
- types.py: Error enums and data structures
- decorators.py: Error boundary decorator and SafeOperation context manager
- circuit_breaker.py: Circuit breaker for error rate monitoring
- memory_monitor.py: Memory monitoring and garbage collection
- handler.py: Main ErrorHandler singleton

All classes re-exported here for convenient imports.
"""

from .types import (
    ErrorSeverity,
    ErrorCategory,
    RecoveryStrategy,
    ErrorContext,
    ErrorEvent,
)

from .decorators import (
    error_boundary,
    SafeOperation,
)

from .circuit_breaker import CircuitBreaker
from .memory_monitor import MemoryMonitor
from .handler import ErrorHandler

__all__ = [
    # Types
    'ErrorSeverity',
    'ErrorCategory',
    'RecoveryStrategy',
    'ErrorContext',
    'ErrorEvent',
    # Decorators
    'error_boundary',
    'SafeOperation',
    # Components
    'CircuitBreaker',
    'MemoryMonitor',
    'ErrorHandler',
]
