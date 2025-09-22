"""
Error handler module for Pokemon Crystal RL.

This module provides centralized error handling, tracking, and reporting with features like:
- Error categorization and severity levels
- Error aggregation and deduplication
- Stack trace analysis and formatting
- Error recovery strategies
- Real-time error notifications
- Error rate monitoring and circuit breaking
"""

import sys
import traceback
from typing import Dict, List, Optional, Any, Callable, Type, Union
import threading
from datetime import datetime, timedelta
import logging
import json
from dataclasses import dataclass, asdict
import gc
import psutil
from collections import defaultdict
import time
from enum import Enum
from pathlib import Path
import signal
import queue
from contextlib import contextmanager
from functools import wraps

from .data_bus import DataType, get_data_bus


class ErrorSeverity(Enum):
    """Error severity levels."""
    CRITICAL = "critical"      # System-breaking errors
    HIGH = "high"             # High priority errors
    ERROR = "error"           # Serious errors that need immediate attention
    MEDIUM = "medium"         # Medium priority errors
    WARNING = "warning"       # Issues that need monitoring
    INFO = "info"            # Informational errors


class ErrorCategory(Enum):
    """Categories of errors for better organization."""
    SYSTEM = "system"         # System-level errors
    NETWORK = "network"       # Network-related issues
    DATABASE = "database"     # Database errors
    GAME = "game"            # Game-specific errors
    TRAINING = "training"     # ML training errors
    MEMORY = "memory"        # Memory-related issues
    PERFORMANCE = "performance"  # Performance problems
    UNKNOWN = "unknown"       # Uncategorized errors


class RecoveryStrategy(Enum):
    """Available recovery strategies"""
    NONE = "none"  # No recovery attempted
    RETRY = "retry"  # Retry the operation
    RESTART = "restart"  # Restart the component
    RESTART_COMPONENT = "restart_component"  # Restart specific component
    RESET = "reset"  # Full system reset
    GRACEFUL_SHUTDOWN = "graceful_shutdown"  # Graceful shutdown
    FALLBACK = "fallback"  # Use fallback mechanism


def error_boundary(
                 max_retries: int = 3, category: ErrorCategory = ErrorCategory.UNKNOWN):
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
    
    def __init__(self, component: str, operation: str,
                 severity: ErrorSeverity = ErrorSeverity.ERROR,
                 category: ErrorCategory = ErrorCategory.UNKNOWN):
        self.component = component
        self.operation = operation
        self.severity = severity
        self.category = category
        self.handler = ErrorHandler.get_instance()
    
    def __enter__(self):
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



@dataclass
class ErrorContext:
    """Context information for an error."""
    timestamp: float
    error_type: str
    error_message: str
    traceback: str
    severity: ErrorSeverity
    category: ErrorCategory
    component: str
    additional_data: Optional[Dict[str, Any]] = None
    error_id: Optional[str] = None
    handled: bool = False
    recovery_attempted: bool = False
    recovery_successful: bool = False


@dataclass
class ErrorEvent:
    """Simplified error event for testing and basic error handling."""
    timestamp: float
    component: str
    error_type: str
    message: str
    severity: ErrorSeverity
    traceback: str = ""
    recovery_strategy: RecoveryStrategy = RecoveryStrategy.NONE


class CircuitBreaker:
    """Circuit breaker for error rate monitoring."""
    
    def __init__(self, error_threshold: int, time_window: float, reset_timeout: float):
        self.error_threshold = error_threshold
        self.time_window = time_window
        self.reset_timeout = reset_timeout
        self.error_counts: Dict[str, List[float]] = defaultdict(list)
        self.broken_circuits: Dict[str, float] = {}
        self._lock = threading.Lock()
    
    def record_error(self, component: str) -> bool:
        """Record an error and check if circuit should break."""
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
        """Check if circuit is broken for component."""
        with self._lock:
            if component not in self.broken_circuits:
                return False
            
            break_time = self.broken_circuits[component]
            if time.time() - break_time >= self.reset_timeout:
                del self.broken_circuits[component]
                self.error_counts[component].clear()
                return False
            
            return True


class ErrorHandler:
    """Centralized error handler for the system."""

    _instance = None
    
    @classmethod
    def get_instance(cls) -> 'ErrorHandler':
        """Get the singleton instance of ErrorHandler.
        
        Returns:
            The singleton ErrorHandler instance
        """
        if cls._instance is None:
            cls._instance = cls.initialize()
        return cls._instance

    @classmethod
    def initialize(cls,
                  log_dir: str = "logs/errors",
                  memory_threshold_mb: float = 1024.0,
                  max_error_history: int = 100,
                  notification_batch_size: int = 10,
                  notification_interval: float = 5.0,
                  db_manager=None) -> "ErrorHandler":
        """Initialize and return the singleton error handler."""
        if cls._instance is None:
            cls._instance = cls(
                log_dir=log_dir,
                max_stored_errors=max_error_history,
                notification_batch_size=notification_batch_size,
                notification_interval=notification_interval,
                db_manager=db_manager
            )
        return cls._instance

    def __init__(self,
                 log_dir: str = "logs/errors",
                 max_stored_errors: int = 1000,
                 notification_batch_size: int = 10,
                 notification_interval: float = 5.0,
                 db_manager=None):
        # Component tracking
        self._components = {}
        self._component_callbacks = {}
        self._error_history = []

        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Error storage
        self.max_stored_errors = max_stored_errors
        self.recent_errors: List[ErrorContext] = []
        self.error_counts: Dict[str, int] = defaultdict(int)
        self._error_lock = threading.Lock()

        # Error deduplication
        self.error_signatures: Dict[str, datetime] = {}
        self.duplicate_counts: Dict[str, int] = defaultdict(int)

        # Circuit breaker
        self.circuit_breaker = CircuitBreaker(
            error_threshold=5,    # 5 errors
            time_window=60.0,     # in 1 minute
            reset_timeout=300.0   # 5 minutes cool-down
        )

        # Notification queue
        self.notification_queue = queue.Queue()
        self.notification_batch_size = notification_batch_size
        self.notification_interval = notification_interval
        self.is_notifying = False
        self._notification_thread: Optional[threading.Thread] = None

        # Recovery strategies
        self.recovery_strategies: Dict[
            Union[str, Type[Exception]], 
            Callable[[ErrorContext], bool]
        ] = {}

        # Logging
        self.logger = logging.getLogger("error_handler")
        self._setup_logging()

        # Data bus
        self.data_bus = get_data_bus()

        # Database manager for recording errors
        self.db_manager = db_manager

        # Register signal handlers
        self._setup_signal_handlers()

        # Register error handler as a component
        self.register_component('error_handler')
        
        self.logger.info("🚨 Error handler initialized")

        # Start notification thread
        self.start_notification_thread()

    def _setup_logging(self) -> None:
        """Setup error logging configuration."""
        handler = logging.FileHandler(self.log_dir / "error.log")
        formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.ERROR)

    def _setup_signal_handlers(self) -> None:
        """Setup handlers for system signals."""
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        signal.signal(signal.SIGINT, self._handle_shutdown)

    def _handle_shutdown(self, signum: int, frame: Any) -> None:
        """Handle system shutdown signals."""
        self.logger.info("Shutting down error handler...")
        self.stop_notification_thread()
        # Ensure all pending errors are saved
        self.save_errors(self.log_dir / "errors_shutdown.json")
        sys.exit(0)
    
    def register_component(self, component_name: str) -> None:
        """Register a component for health tracking."""
        if component_name not in self._components:
            self._components[component_name] = {
                'status': 'healthy',
                'error_count': 0,
                'last_error': None,
                'registered_at': time.time()
            }
    
    def get_component_health(self, component_name: str) -> Dict[str, Any]:
        """Get health status of a registered component."""
        if component_name not in self._components:
            return {'status': 'unknown', 'error_count': 0}
        return self._components[component_name]
    
    def add_recovery_callback(self, component_name: str, callback: Callable[[], None]) -> None:
        """Register a recovery callback for a component."""
        self.register_component(component_name)
        self._component_callbacks[component_name] = callback
    
    def force_component_recovery(self, component_name: str) -> bool:
        """Force recovery of a component using its registered callback."""
        if component_name in self._component_callbacks:
            try:
                self._component_callbacks[component_name]()
                self._components[component_name]['status'] = 'healthy'
                return True
            except Exception as e:
                self.logger.error(f"Recovery failed for {component_name}: {e}")
        return False
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics."""
        current_time = time.time()
        one_hour_ago = current_time - 3600
        
        # Count recent errors
        recent_errors = sum(1 for err in self._error_history 
                          if err['timestamp'] > one_hour_ago)
        
        # Count by severity and component
        errors_by_severity = defaultdict(int)
        errors_by_component = defaultdict(int)
        for err in self._error_history:
            if 'severity' in err:
                errors_by_severity[err['severity']] += 1
            if 'component' in err:
                errors_by_component[err['component']] += 1
        
        # Calculate component health rate
        healthy_components = sum(1 for comp in self._components.values() 
                               if comp['status'] == 'healthy')
        total_components = len(self._components) or 1  # Avoid division by zero
        health_rate = healthy_components / total_components
        
        memory_info = None
        try:
            memory_info = MemoryMonitor().get_memory_info()
        except Exception as e:
            self.logger.warning(f"Could not get memory info: {e}")
        
        return {
            'total_errors': len(self._error_history),
            'recent_errors_1h': recent_errors,
            'errors_by_severity': dict(errors_by_severity),
            'errors_by_component': dict(errors_by_component),
            'component_health': {
                'health_rate': health_rate,
                'healthy_count': healthy_components,
                'total_components': total_components
            },
            'memory_info': memory_info
        }
    
    def shutdown(self) -> None:
        """Clean shutdown of the error handler system."""
        try:
            # Stop all monitoring
            self.stop_notification_thread()
            
            # Save final error state
            self.save_errors(str(self.log_dir / "final_error_state.json"))
            
            # Reset components
            self._components.clear()
            self._component_callbacks.clear()
            self._error_history.clear()
            
            # Clear singleton instance
            type(self)._instance = None
            
            self.logger.info("Error handler shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")

    def handle_error(self,
                  error: Exception,
                  message: Optional[str] = None,
                  severity: ErrorSeverity = ErrorSeverity.ERROR,
                  category: Union[ErrorCategory, str] = ErrorCategory.UNKNOWN,
                  component: str = "unknown",
                  recovery_strategy: RecoveryStrategy = RecoveryStrategy.NONE,
                  strategy: RecoveryStrategy = None,  # Alias for recovery_strategy for backward compatibility
                  additional_data: Optional[Dict[str, Any]] = None) -> ErrorContext:
        """Handle an error and create error context.

        Args:
            error: The exception to handle
            message: Optional custom error message
            severity: Error severity level
            category: Error category
            component: Component where error occurred
            recovery_strategy: Recovery strategy to use
            strategy: Alias for recovery_strategy (for backward compatibility)
            additional_data: Additional error context data
        """
        # Handle backward compatibility: if strategy is provided, use it
        if strategy is not None:
            recovery_strategy = strategy

        try:
            # Convert string category to ErrorCategory enum if needed
            if isinstance(category, str):
                try:
                    category = ErrorCategory(category.lower())
                except ValueError:
                    category = ErrorCategory.UNKNOWN

            # Create error context
            error_context = ErrorContext(
                timestamp=time.time(),
                error_type=error.__class__.__name__,
                error_message=message or str(error),
                traceback=traceback.format_exc(),
                severity=severity,
                category=category,
                component=component,
                additional_data=additional_data or {},
                error_id=f"{int(time.time() * 1000)}_{id(error)}",
                handled=False
            )

            # Check circuit breaker
            if self.circuit_breaker.record_error(component):
                self.logger.critical(
                    f"Circuit breaker triggered for component: {component}"
                )
                self._notify_circuit_breaker(component)

            # Try recovery if strategy exists
            error_context.recovery_attempted = self._attempt_recovery(error_context)

            # Store error
            self._store_error(error_context)

            # Log error
            self._log_error(error_context)

            # Record error in database if available
            self._record_error_in_db(error_context, recovery_strategy)

            # Handle critical errors by changing training state
            if severity == ErrorSeverity.CRITICAL and hasattr(self, '_monitor'):
                from .web_monitor import TrainingState
                if hasattr(self._monitor, 'training_state'):
                    self._monitor.training_state = TrainingState.ERROR

            # Queue for notification
            self.notification_queue.put(error_context)

            return error_context

        except Exception as e:
            # Fallback logging if error handling fails
            self.logger.critical(f"Error handler failed: {e}")
            return ErrorContext(
                timestamp=time.time(),
                error_type="ErrorHandlerFailure",
                error_message=str(e),
                traceback=traceback.format_exc(),
                severity=ErrorSeverity.CRITICAL,
                category=ErrorCategory.SYSTEM,
                component="error_handler",
                handled=False
            )

    def handle_error_event(self, error_event: ErrorEvent) -> ErrorContext:
        """Handle an ErrorEvent object and convert it to ErrorContext."""
        try:
            # Convert ErrorEvent to ErrorContext
            error_context = ErrorContext(
                timestamp=error_event.timestamp,
                error_type=error_event.error_type,
                error_message=error_event.message,
                traceback=error_event.traceback,
                severity=error_event.severity,
                category=ErrorCategory.UNKNOWN,  # Default category
                component=error_event.component,
                error_id=f"{int(error_event.timestamp * 1000)}_{hash(error_event.message)}",
                handled=False
            )

            # Check circuit breaker
            if self.circuit_breaker.record_error(error_event.component):
                self.logger.critical(
                    f"Circuit breaker triggered for component: {error_event.component}"
                )
                self._notify_circuit_breaker(error_event.component)

            # Try recovery if strategy exists
            error_context.recovery_attempted = self._attempt_recovery(error_context)

            # Store error
            self._store_error(error_context)

            # Log error
            self._log_error(error_context)

            # Queue for notification
            self.notification_queue.put(error_context)

            return error_context

        except Exception as e:
            # Fallback logging if error handling fails
            self.logger.critical(f"Error event handler failed: {e}")
            return ErrorContext(
                timestamp=time.time(),
                error_type="ErrorEventHandlerFailure",
                error_message=str(e),
                traceback=traceback.format_exc(),
                severity=ErrorSeverity.CRITICAL,
                category=ErrorCategory.SYSTEM,
                component="error_handler",
                handled=False
            )

    def _store_error(self, error_context: ErrorContext) -> None:
        """Store error in recent errors list with deduplication."""
        with self._error_lock:
            # Generate error signature
            signature = f"{error_context.error_type}:{error_context.error_message}"

            # Check for duplicates
            if signature in self.error_signatures:
                last_seen = self.error_signatures[signature]
                if datetime.now() - last_seen < timedelta(minutes=5):
                    self.duplicate_counts[signature] += 1
                    return

            # Update signature timestamp
            self.error_signatures[signature] = datetime.now()
            self.duplicate_counts[signature] = 1

            # Create error history entry
            error_entry = {
                'error_id': error_context.error_id,
                'timestamp': error_context.timestamp,
                'error_type': error_context.error_type,
                'message': error_context.error_message,
                'severity': error_context.severity.value,
                'category': error_context.category.value,
                'component': error_context.component,
                'traceback': error_context.traceback
            }

            # Add to error history
            self._error_history.append(error_entry)
            if len(self._error_history) > self.max_stored_errors:
                self._error_history.pop(0)

            # Add to recent errors for backwards compatibility
            self.recent_errors.append(error_context)
            if len(self.recent_errors) > self.max_stored_errors:
                self.recent_errors.pop(0)

            # Update error counts
            self.error_counts[error_context.category.value] += 1
            
            # Update component status if applicable
            if error_context.component in self._components:
                component = self._components[error_context.component]
                component['error_count'] += 1
                component['last_error'] = error_entry
                if error_context.severity in (ErrorSeverity.CRITICAL, ErrorSeverity.HIGH):
                    component['status'] = 'degraded'

    def _log_error(self, error_context: ErrorContext) -> None:
        """Log error details."""
        log_message = (
            f"Error in {error_context.component}: {error_context.error_message}\n"
            f"Type: {error_context.error_type}\n"
            f"Severity: {error_context.severity.value}\n"
            f"Category: {error_context.category.value}\n"
            f"Traceback:\n{error_context.traceback}"
        )

        if error_context.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(log_message)
        else:
            self.logger.error(log_message)

    def register_recovery_strategy(self,
                               error_type: Union[str, Type[Exception]],
                               strategy: Callable[[ErrorContext], bool]) -> None:
        """Register a recovery strategy for an error type."""
        self.recovery_strategies[error_type] = strategy
        self.logger.info(f"Registered recovery strategy for {error_type}")

    def _attempt_recovery(self, error_context: ErrorContext) -> bool:
        """Attempt to recover from an error using registered strategies."""
        try:
            # Check for matching strategy
            strategy = self.recovery_strategies.get(
                error_context.error_type,
                self.recovery_strategies.get(type(error_context.error_type))
            )

            if strategy:
                error_context.recovery_successful = strategy(error_context)
                return True

            return False

        except Exception as e:
            self.logger.error(f"Recovery attempt failed: {e}")
            return False

    def start_notification_thread(self) -> None:
        """Start the notification processing thread."""
        if self.is_notifying:
            return

        self.is_notifying = True
        self._notification_thread = threading.Thread(
            target=self._notification_loop,
            daemon=True
        )
        self._notification_thread.start()

    def stop_notification_thread(self) -> None:
        """Stop the notification processing thread."""
        self.is_notifying = False
        if self._notification_thread and self._notification_thread.is_alive():
            self._notification_thread.join(timeout=5.0)

    def _notification_loop(self) -> None:
        """Process and send error notifications."""
        while self.is_notifying:
            try:
                # Collect errors for batch processing
                errors = []
                try:
                    while len(errors) < self.notification_batch_size:
                        error = self.notification_queue.get(
                            timeout=self.notification_interval
                        )
                        errors.append(error)
                except queue.Empty:
                    pass

                if not errors:
                    continue

                # Process batch
                self._send_notifications(errors)

            except Exception as e:
                self.logger.error(f"Notification processing failed: {e}")
                time.sleep(1)

    def _send_notifications(self, errors: List[ErrorContext]) -> None:
        """Send error notifications via data bus."""
        if not self.data_bus:
            return

        try:
            # Prepare notification data
            notification_data = {
                'timestamp': time.time(),
                'errors': [
                    {
                        'error_id': e.error_id,
                        'error_type': e.error_type,
                        'message': e.error_message,
                        'severity': e.severity.value,
                        'category': e.category.value,
                        'component': e.component,
                        'timestamp': e.timestamp
                    }
                    for e in errors
                ]
            }

            # Publish notification
            self.data_bus.publish(
                DataType.ERROR_NOTIFICATION,
                notification_data,
                'error_handler'
            )

        except Exception as e:
            self.logger.error(f"Failed to send notifications: {e}")

    def _notify_circuit_breaker(self, component: str) -> None:
        """Send circuit breaker notification."""
        if not self.data_bus:
            return

        try:
            self.data_bus.publish(
                DataType.SYSTEM_ALERT,
                {
                    'timestamp': time.time(),
                    'alert_type': 'circuit_breaker',
                    'component': component,
                    'message': f"Circuit breaker triggered for {component}"
                },
                'error_handler'
            )
        except Exception as e:
            self.logger.error(f"Failed to send circuit breaker notification: {e}")

    def get_recent_errors(self) -> List[Dict[str, Any]]:
        """Get list of recent errors."""
        with self._error_lock:
            return [{
                'error_id': e.error_id,
                'error_type': e.error_type,
                'message': e.error_message,
                'severity': e.severity.value,
                'category': e.category.value,
                'component': e.component,
                'timestamp': e.timestamp,
                'traceback': e.traceback
            } for e in self.recent_errors]

    def get_error_stats(self) -> Dict[str, Any]:
        """Get error statistics."""
        with self._error_lock:
            # Count warnings specifically
            warning_count = sum(
                1 for e in self.recent_errors 
                if e.severity == ErrorSeverity.WARNING
            )
            
            return {
                'total_errors': len(self.recent_errors),
                'error_counts': dict(self.error_counts),
                'duplicate_counts': dict(self.duplicate_counts),
                'warning_count': warning_count,
                'broken_circuits': [
                    comp for comp, _ in self.circuit_breaker.broken_circuits.items()
                ]
            }

    def save_errors(self, filepath: str) -> None:
        """Save error history to a JSON file."""
        try:
            with self._error_lock:
                error_data = {
                    'timestamp': datetime.now().isoformat(),
                    'errors': [
                        {
                            'error_id': e.error_id,
                            'error_type': e.error_type,
                            'message': e.error_message,
                            'traceback': e.traceback,
                            'severity': e.severity.value,
                            'category': e.category.value,
                            'component': e.component,
                            'timestamp': e.timestamp,
                            'additional_data': e.additional_data,
                            'handled': e.handled,
                            'recovery_attempted': e.recovery_attempted,
                            'recovery_successful': e.recovery_successful
                        }
                        for e in self.recent_errors
                    ],
                    'stats': self.get_error_stats()
                }

                # Try to publish error data to data bus first
                if self.data_bus:
                    try:
                        for error in error_data['errors']:
                            error_event = ErrorEvent(
                                timestamp=error['timestamp'],
                                component=error['component'],
                                error_type=error['error_type'],
                                message=error['message'],
                                severity=ErrorSeverity(error['severity']),
                                traceback=error['traceback'],
                                recovery_strategy=RecoveryStrategy.RETRY
                            )
                            self.data_bus.publish(
                                DataType.ERROR_EVENT,
                                error_event,
                                'error_handler'
                            )
                    except Exception as e:
                        self.logger.error(f"Failed to publish errors: {e}")

                # Save to file
                with open(filepath, 'w') as f:
                    json.dump(error_data, f, indent=2)

                self.logger.info(f"💾 Error history saved to {filepath}")

        except Exception as e:
            self.logger.error(f"Failed to save error history: {e}")

    @contextmanager
    def error_context(self,
                    severity: ErrorSeverity = ErrorSeverity.ERROR,
                    category: ErrorCategory = ErrorCategory.UNKNOWN,
                    component: str = "unknown",
                    additional_data: Optional[Dict[str, Any]] = None):
        """Context manager for handling errors in a block of code."""
        try:
            yield
        except Exception as e:
            self.handle_error(
                e,
                severity=severity,
                category=category,
                component=component,
                additional_data=additional_data
            )
            raise  # Re-raise the exception after handling


class MemoryMonitor:
    """Memory monitoring system that tracks memory usage and handles memory-related issues.

    Args:
        threshold_mb: Memory usage threshold in megabytes
        check_interval: Time interval between memory checks in seconds
        on_threshold_exceeded: Optional callback when memory threshold is exceeded
    """
    
    def __init__(self, 
                 threshold_mb: float = 1024.0,
                 check_interval: float = 60.0,
                 on_threshold_exceeded: Optional[Callable[[], None]] = None):
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

    def _record_error_in_db(self, error_context: ErrorContext, recovery_strategy: RecoveryStrategy) -> None:
        """Record error in database if database manager is available."""
        if not self.db_manager or not hasattr(self.db_manager, 'record_event'):
            return

        try:
            # Get current run_id from database manager or use a default
            run_id = getattr(self.db_manager, 'current_run_id', None)
            if not run_id:
                # Try to get from a parent monitor if available
                if hasattr(self, '_monitor') and hasattr(self._monitor, 'current_run_id'):
                    run_id = self._monitor.current_run_id
                else:
                    run_id = "unknown_run"

            # Record error as an event in the database
            event_data = {
                'error_id': error_context.error_id,
                'error_type': error_context.error_type,
                'message': error_context.error_message,
                'severity': error_context.severity.value,
                'category': error_context.category.value,
                'component': error_context.component,
                'recovery_strategy': recovery_strategy.value,
                'timestamp': error_context.timestamp,
                'traceback': error_context.traceback,
                'additional_data': error_context.additional_data
            }

            self.db_manager.record_event(
                run_id=run_id,
                event_type="error",
                event_data=event_data
            )

        except Exception as e:
            self.logger.error(f"Failed to record error in database: {e}")

    def set_monitor(self, monitor):
        """Set the parent monitor for accessing run_id."""
        self._monitor = monitor

    def __del__(self):
        """Cleanup on deletion."""
        self.stop_notification_thread()
