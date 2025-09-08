"""
Error Recovery System - Error handling and recovery mechanisms

Extracted from PokemonTrainer to handle error tracking, recovery strategies,
and system resilience with proper monitoring integration.
"""

import time
import logging
import threading
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass
from collections import defaultdict, deque
from enum import Enum


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium" 
    HIGH = "high"
    CRITICAL = "critical"


class RecoveryStrategy(Enum):
    """Recovery strategy types."""
    IGNORE = "ignore"
    RETRY = "retry" 
    RESTART = "restart"
    RESET = "reset"
    MANUAL = "manual"


@dataclass
class ErrorEvent:
    """Error event data structure."""
    timestamp: float
    error_type: str
    operation: str
    message: str
    severity: ErrorSeverity
    recovery_strategy: RecoveryStrategy
    traceback: Optional[str] = None
    component: str = "unknown"
    metadata: Dict[str, Any] = None


@dataclass
class RecoveryConfig:
    """Configuration for error recovery system."""
    max_recovery_attempts: int = 3
    recovery_cooldown: float = 5.0
    error_threshold: int = 10
    auto_recovery_enabled: bool = True
    publish_to_data_bus: bool = True
    track_error_patterns: bool = True


class ErrorRecoverySystem:
    """Handles error tracking, recovery strategies, and system resilience."""
    
    def __init__(self, config: RecoveryConfig = None, data_bus=None):
        self.config = config or RecoveryConfig()
        self.data_bus = data_bus
        self.logger = logging.getLogger("ErrorRecoverySystem")
        
        # Error tracking
        self.error_counts = defaultdict(int)
        self.error_history = deque(maxlen=1000)
        self.last_error_time = None
        self.recovery_attempts = 0
        
        # Recovery state
        self.recovery_handlers: Dict[str, Callable] = {}
        self.recovery_in_progress = False
        self._lock = threading.RLock()
        
        # Pattern tracking
        self.error_patterns = defaultdict(list)
        self.consecutive_errors = 0
        self.error_burst_threshold = 5
        
        # Recovery strategies per error type
        self.recovery_strategies = {
            'pyboy_crashes': RecoveryStrategy.RESTART,
            'llm_failures': RecoveryStrategy.RETRY,
            'capture_errors': RecoveryStrategy.IGNORE,
            'memory_errors': RecoveryStrategy.RESET,
            'connection_errors': RecoveryStrategy.RETRY,
            'timeout_errors': RecoveryStrategy.RETRY,
            'default': RecoveryStrategy.IGNORE
        }
        
        self.logger.info("Error recovery system initialized")
    
    def register_recovery_handler(self, error_type: str, handler: Callable) -> None:
        """Register a recovery handler for specific error type.
        
        Args:
            error_type: Type of error to handle
            handler: Recovery function to call
        """
        self.recovery_handlers[error_type] = handler
        self.logger.info(f"Recovery handler registered for: {error_type}")
    
    def handle_error(self, 
                    error: Exception,
                    operation: str = "unknown",
                    error_type: str = None,
                    component: str = "system",
                    metadata: Dict[str, Any] = None) -> bool:
        """Handle an error with appropriate recovery strategy.
        
        Args:
            error: Exception that occurred
            operation: Operation being performed when error occurred
            error_type: Custom error type classification
            component: Component where error occurred
            metadata: Additional error context
            
        Returns:
            bool: True if error was handled and recovery attempted
        """
        with self._lock:
            # Classify error
            error_type = error_type or self._classify_error(error)
            severity = self._determine_severity(error, error_type)
            strategy = self._get_recovery_strategy(error_type)
            
            # Create error event
            error_event = ErrorEvent(
                timestamp=time.time(),
                error_type=error_type,
                operation=operation,
                message=str(error),
                severity=severity,
                recovery_strategy=strategy,
                traceback=self._get_traceback(error),
                component=component,
                metadata=metadata or {}
            )
            
            # Record error
            self._record_error(error_event)
            
            # Publish to monitoring if enabled
            if self.config.publish_to_data_bus and self.data_bus:
                self._publish_error_event(error_event)
            
            # Attempt recovery
            return self._attempt_recovery(error_event)
    
    def _classify_error(self, error: Exception) -> str:
        """Classify error type based on exception."""
        error_name = type(error).__name__
        error_message = str(error).lower()
        
        # Classification based on exception type and message
        if 'pyboy' in error_message or 'emulation' in error_message:
            return 'pyboy_crashes'
        elif 'llm' in error_message or 'model' in error_message:
            return 'llm_failures'
        elif 'capture' in error_message or 'screen' in error_message:
            return 'capture_errors'
        elif 'memory' in error_message or isinstance(error, MemoryError):
            return 'memory_errors'
        elif 'connection' in error_message or 'network' in error_message:
            return 'connection_errors'
        elif 'timeout' in error_message or isinstance(error, TimeoutError):
            return 'timeout_errors'
        else:
            return error_name.lower()
    
    def _determine_severity(self, error: Exception, error_type: str) -> ErrorSeverity:
        """Determine error severity."""
        # Critical errors
        if error_type in ['memory_errors', 'pyboy_crashes'] or isinstance(error, (MemoryError, SystemError)):
            return ErrorSeverity.CRITICAL
        
        # High severity errors
        if error_type in ['connection_errors'] or self.error_counts[error_type] > self.config.error_threshold:
            return ErrorSeverity.HIGH
        
        # Medium severity errors  
        if error_type in ['llm_failures', 'timeout_errors']:
            return ErrorSeverity.MEDIUM
        
        # Low severity by default
        return ErrorSeverity.LOW
    
    def _get_recovery_strategy(self, error_type: str) -> RecoveryStrategy:
        """Get recovery strategy for error type."""
        return self.recovery_strategies.get(error_type, self.recovery_strategies['default'])
    
    def _get_traceback(self, error: Exception) -> Optional[str]:
        """Extract traceback from exception."""
        try:
            import traceback
            return traceback.format_exc()
        except Exception:
            return None
    
    def _record_error(self, error_event: ErrorEvent) -> None:
        """Record error event in tracking systems."""
        # Update counters
        self.error_counts[error_event.error_type] += 1
        self.error_counts['total_errors'] += 1
        
        # Add to history
        self.error_history.append(error_event)
        
        # Update timing
        self.last_error_time = error_event.timestamp
        
        # Track patterns if enabled
        if self.config.track_error_patterns:
            self._track_error_patterns(error_event)
        
        # Log error
        log_method = getattr(self.logger, error_event.severity.value, self.logger.error)
        log_method(f"{error_event.error_type} in {error_event.operation}: {error_event.message}")
    
    def _track_error_patterns(self, error_event: ErrorEvent) -> None:
        """Track error patterns for analysis."""
        # Track consecutive errors
        if len(self.error_history) > 1:
            last_error = self.error_history[-2]
            if (error_event.timestamp - last_error.timestamp < 10 and 
                error_event.error_type == last_error.error_type):
                self.consecutive_errors += 1
            else:
                self.consecutive_errors = 0
        
        # Track error bursts
        recent_errors = [e for e in self.error_history 
                        if error_event.timestamp - e.timestamp < 60]  # Last minute
        
        if len(recent_errors) > self.error_burst_threshold:
            self.logger.warning(f"Error burst detected: {len(recent_errors)} errors in last minute")
        
        # Track pattern by error type
        self.error_patterns[error_event.error_type].append(error_event.timestamp)
        # Keep only last 50 occurrences per type
        if len(self.error_patterns[error_event.error_type]) > 50:
            self.error_patterns[error_event.error_type] = self.error_patterns[error_event.error_type][-50:]
    
    def _publish_error_event(self, error_event: ErrorEvent) -> None:
        """Publish error event to data bus."""
        try:
            from monitoring.data_bus import DataType
            self.data_bus.publish(
                DataType.ERROR_EVENT,
                error_event,
                "error_recovery_system"
            )
        except Exception as e:
            self.logger.debug(f"Failed to publish error event: {e}")
    
    def _attempt_recovery(self, error_event: ErrorEvent) -> bool:
        """Attempt recovery based on error event.
        
        Args:
            error_event: Error event to recover from
            
        Returns:
            bool: True if recovery was attempted
        """
        if not self.config.auto_recovery_enabled:
            return False
        
        if self.recovery_in_progress:
            self.logger.debug("Recovery already in progress, skipping")
            return False
        
        # Check recovery attempt limits
        if self.recovery_attempts >= self.config.max_recovery_attempts:
            self.logger.warning(f"Max recovery attempts ({self.config.max_recovery_attempts}) exceeded")
            return False
        
        # Check cooldown period
        if (self.last_error_time and 
            time.time() - self.last_error_time < self.config.recovery_cooldown):
            self.logger.debug("Recovery cooldown active, skipping")
            return False
        
        try:
            self.recovery_in_progress = True
            self.recovery_attempts += 1
            
            success = self._execute_recovery_strategy(error_event)
            
            if success:
                self.logger.info(f"Recovery successful for {error_event.error_type}")
                self.recovery_attempts = 0  # Reset on success
            else:
                self.logger.warning(f"Recovery failed for {error_event.error_type}")
            
            return success
            
        except Exception as recovery_error:
            self.logger.error(f"Recovery attempt failed: {recovery_error}")
            return False
        finally:
            self.recovery_in_progress = False
    
    def _execute_recovery_strategy(self, error_event: ErrorEvent) -> bool:
        """Execute recovery strategy for error.
        
        Args:
            error_event: Error event requiring recovery
            
        Returns:
            bool: True if recovery was successful
        """
        strategy = error_event.recovery_strategy
        error_type = error_event.error_type
        
        self.logger.info(f"Executing {strategy.value} recovery for {error_type}")
        
        # Check for custom handler first
        if error_type in self.recovery_handlers:
            try:
                return self.recovery_handlers[error_type](error_event)
            except Exception as e:
                self.logger.error(f"Custom recovery handler failed: {e}")
                return False
        
        # Execute built-in recovery strategies
        if strategy == RecoveryStrategy.IGNORE:
            return True
        
        elif strategy == RecoveryStrategy.RETRY:
            return self._retry_recovery(error_event)
        
        elif strategy == RecoveryStrategy.RESTART:
            return self._restart_recovery(error_event)
        
        elif strategy == RecoveryStrategy.RESET:
            return self._reset_recovery(error_event)
        
        elif strategy == RecoveryStrategy.MANUAL:
            self.logger.warning(f"Manual recovery required for {error_type}")
            return False
        
        else:
            self.logger.warning(f"Unknown recovery strategy: {strategy}")
            return False
    
    def _retry_recovery(self, error_event: ErrorEvent) -> bool:
        """Implement retry recovery strategy."""
        # Simple retry - just wait a bit and return success
        # Actual retry logic would depend on the specific operation
        time.sleep(1.0)
        self.logger.info(f"Retry recovery completed for {error_event.error_type}")
        return True
    
    def _restart_recovery(self, error_event: ErrorEvent) -> bool:
        """Implement restart recovery strategy."""
        # This would typically restart a component
        # Implementation depends on what needs restarting
        self.logger.info(f"Restart recovery initiated for {error_event.error_type}")
        
        # Reset error counts for this type
        if error_event.error_type in self.error_counts:
            self.error_counts[error_event.error_type] = 0
        
        return True
    
    def _reset_recovery(self, error_event: ErrorEvent) -> bool:
        """Implement reset recovery strategy."""
        # This would typically reset system state
        self.logger.info(f"Reset recovery initiated for {error_event.error_type}")
        
        # Clear error history for clean slate
        self.error_history.clear()
        self.consecutive_errors = 0
        
        return True
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics.
        
        Returns:
            Dict: Error statistics and patterns
        """
        with self._lock:
            recent_errors = [e for e in self.error_history 
                           if time.time() - e.timestamp < 300]  # Last 5 minutes
            
            return {
                'total_errors': self.error_counts['total_errors'],
                'error_counts': dict(self.error_counts),
                'recent_errors': len(recent_errors),
                'consecutive_errors': self.consecutive_errors,
                'recovery_attempts': self.recovery_attempts,
                'recovery_in_progress': self.recovery_in_progress,
                'last_error_time': self.last_error_time,
                'error_rate_5min': len(recent_errors) / 5.0,
                'top_error_types': self._get_top_error_types(),
                'recovery_success_rate': self._calculate_recovery_success_rate()
            }
    
    def _get_top_error_types(self, limit: int = 5) -> List[tuple]:
        """Get top error types by frequency."""
        sorted_errors = sorted(
            [(k, v) for k, v in self.error_counts.items() if k != 'total_errors'],
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_errors[:limit]
    
    def _calculate_recovery_success_rate(self) -> float:
        """Calculate recovery success rate."""
        # This would track recovery success/failure in a real implementation
        # For now, return estimated rate based on error patterns
        if self.recovery_attempts == 0:
            return 1.0
        
        # Simple estimation - fewer consecutive errors = higher success rate
        if self.consecutive_errors > 5:
            return 0.3
        elif self.consecutive_errors > 2:
            return 0.6
        else:
            return 0.8
    
    def reset_error_counts(self, error_type: str = None) -> None:
        """Reset error counts for specific type or all types.
        
        Args:
            error_type: Specific error type to reset, or None for all
        """
        with self._lock:
            if error_type:
                self.error_counts[error_type] = 0
                self.logger.info(f"Reset error count for {error_type}")
            else:
                self.error_counts.clear()
                self.error_history.clear()
                self.consecutive_errors = 0
                self.recovery_attempts = 0
                self.logger.info("Reset all error counts and history")
    
    def create_error_context_manager(self, 
                                   operation: str, 
                                   error_type: str = None,
                                   component: str = "system"):
        """Create context manager for error handling.
        
        Args:
            operation: Operation being performed
            error_type: Expected error type
            component: Component performing operation
            
        Returns:
            Context manager for error handling
        """
        return ErrorContextManager(self, operation, error_type, component)


class ErrorContextManager:
    """Context manager for error handling with automatic recovery."""
    
    def __init__(self, 
                 recovery_system: ErrorRecoverySystem,
                 operation: str,
                 error_type: str = None,
                 component: str = "system"):
        self.recovery_system = recovery_system
        self.operation = operation
        self.error_type = error_type
        self.component = component
        self.needs_recovery = False
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None:
            return True
        
        # Don't handle KeyboardInterrupt
        if exc_type == KeyboardInterrupt:
            return False
        
        # Handle the error
        handled = self.recovery_system.handle_error(
            error=exc_value,
            operation=self.operation,
            error_type=self.error_type,
            component=self.component
        )
        
        # Return None to re-raise, True to suppress
        return None