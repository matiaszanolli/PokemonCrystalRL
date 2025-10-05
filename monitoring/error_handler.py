"""
Error handler module for Pokemon Crystal RL.

This module has been refactored into the monitoring/error_handling/ package for better
maintainability and organization. All classes are re-exported here for backward compatibility.

Original file: 1029 lines → Now modular:
- monitoring/error_handling/types.py (71 lines) - Error enums and data structures
- monitoring/error_handling/decorators.py (104 lines) - Error boundary and SafeOperation
- monitoring/error_handling/circuit_breaker.py (79 lines) - Circuit breaker logic
- monitoring/error_handling/memory_monitor.py (116 lines) - Memory monitoring
- monitoring/error_handling/handler.py (731 lines) - Main ErrorHandler class

All existing imports will continue to work:
    from monitoring.error_handler import ErrorHandler, ErrorSeverity
    from monitoring.error_handler import error_boundary, SafeOperation
    from monitoring.error_handler import CircuitBreaker, MemoryMonitor

New preferred import style:
    from monitoring.error_handling import ErrorHandler, ErrorSeverity
    from monitoring.error_handling import error_boundary, SafeOperation
    from monitoring.error_handling import CircuitBreaker, MemoryMonitor
"""

from .error_handling import (
    # Types
    ErrorSeverity,
    ErrorCategory,
    RecoveryStrategy,
    ErrorContext,
    ErrorEvent,
    # Decorators
    error_boundary,
    SafeOperation,
    # Components
    CircuitBreaker,
    MemoryMonitor,
    ErrorHandler,
)

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
