"""
Error handling decorators and context managers.

This module provides decorators and context managers for safe error handling.
"""

import time
from functools import wraps
from typing import TYPE_CHECKING

from .types import ErrorSeverity, ErrorCategory

if TYPE_CHECKING:
    from .handler import ErrorHandler


def error_boundary(
    max_retries: int = 3,
    category: ErrorCategory = ErrorCategory.UNKNOWN
):
    """Decorator that creates an error boundary around a function.

    Args:
        component: Name of the component being protected
        severity: Error severity level for failures
        max_retries: Maximum number of retry attempts
        category: Category of errors to expect

    Example:
        @error_boundary("my_component", severity=ErrorSeverity.HIGH)
        def my_function():
            # Function code
            pass
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            from .handler import ErrorHandler
            handler = ErrorHandler.get_instance()
            retries = 0

            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    handler.handle_error(
                        e,
                        severity=severity,
                        category=category,
                        component=component
                    )
                    retries += 1
                    if retries >= max_retries:
                        raise
                    time.sleep(0.1 * retries)  # Exponential backoff
        return wrapper
    return decorator


class SafeOperation:
    """Context manager for safe operation execution with error handling.

    Args:
        component: Component name for error tracking
        operation: Operation name for error context
        severity: Error severity level
        category: Error category

    Example:
        with SafeOperation("my_component", "data_processing"):
            # Protected code
            process_data()
    """

    def __init__(
        self,
        component: str,
        operation: str,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        category: ErrorCategory = ErrorCategory.UNKNOWN
    ):
        self.component = component
        self.operation = operation
        self.severity = severity
        self.category = category
        self.handler = None

    def __enter__(self):
        from .handler import ErrorHandler
        self.handler = ErrorHandler.get_instance()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            # Handle the error but re-raise
            self.handler.handle_error(
                exc_value,
                message=f"Error in {self.operation}",
                severity=self.severity,
                category=self.category,
                component=self.component
            )
            return False  # Re-raise the exception
        return True
