"""
error_handler.py - Comprehensive error handling and recovery system.

This module provides a robust error handling system with severity levels,
recovery strategies, component health tracking, and memory monitoring.
"""

import threading
import time
import traceback
import gc
import psutil
from typing import Dict, List, Any, Optional, Callable
from enum import Enum
from dataclasses import dataclass
from datetime import datetime, timedelta


class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RecoveryStrategy(Enum):
    """Available recovery strategies"""
    NONE = "none"  # No recovery attempted
    RETRY = "retry"  # Retry the operation
    RESTART = "restart"  # Restart the component
    RESET = "reset"  # Full system reset
    FALLBACK = "fallback"  # Use fallback mechanism


@dataclass
class ErrorEvent:
    """Represents an error event with context"""
    timestamp: float
    component: str
    error_type: str
    message: str
    severity: ErrorSeverity
    traceback: Optional[str]
    recovery_strategy: RecoveryStrategy


class MemoryMonitor:
    """Monitor system memory usage and trigger cleanup when needed"""
    
    def __init__(self, threshold_mb: float = 512.0, check_interval: float = 1.0):
        """Initialize memory monitor"""
        self.threshold_mb = threshold_mb
        self.check_interval = check_interval
        self.monitoring_thread = None
        self.monitoring_active = False
    
    def get_memory_info(self) -> Dict[str, float]:
        """Get current memory usage info"""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,
            'vms_mb': memory_info.vms / 1024 / 1024,
            'percent': process.memory_percent()
        }
    
    def trigger_garbage_collection(self) -> Dict[str, int]:
        """Force garbage collection and return stats"""
        # Get count before collection
        objects_before = len(gc.get_objects())
        
        # Collect garbage
        gc.collect()
        
        # Get count after collection
        objects_after = len(gc.get_objects())
        objects_collected = objects_before - objects_after
        
        return {
            'objects_before': objects_before,
            'objects_after': objects_after,
            'objects_collected': objects_collected
        }
    
    def _monitor_memory(self):
        """Background memory monitoring thread"""
        while self.monitoring_active:
            memory_info = self.get_memory_info()
            
            if memory_info['rss_mb'] > self.threshold_mb:
                stats = self.trigger_garbage_collection()
                print(f"Memory threshold exceeded ({memory_info['rss_mb']:.1f}MB). "
                      f"Collected {stats['objects_collected']} objects.")
            
            time.sleep(self.check_interval)
    
    def start_monitoring(self):
        """Start memory monitoring"""
        if not self.monitoring_thread or not self.monitoring_thread.is_alive():
            self.monitoring_active = True
            self.monitoring_thread = threading.Thread(target=self._monitor_memory, daemon=True)
            self.monitoring_thread.start()
    
    def stop_monitoring(self):
        """Stop memory monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=1.0)


class SafeOperation:
    """Context manager for safe operation execution with error handling"""
    
    def __init__(self, component: str, operation: str):
        self.component = component
        self.operation = operation
        self.error_handler = ErrorHandler.get_instance()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            # Create error event
            error_event = ErrorEvent(
                timestamp=time.time(),
                component=self.component,
                error_type=exc_type.__name__,
                message=str(exc_val),
                severity=ErrorSeverity.MEDIUM,  # Default to medium
                traceback=traceback.format_exc(),
                recovery_strategy=RecoveryStrategy.RETRY
            )
            
            # Handle error
            self.error_handler.handle_error(error_event)
            
            # Don't suppress the exception
            return False
        return True


def error_boundary(component: str, severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                  max_retries: int = 1):
    """Decorator for error handling and recovery"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            error_handler = ErrorHandler.get_instance()
            retries = 0
            
            while retries <= max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    error_event = ErrorEvent(
                        timestamp=time.time(),
                        component=component,
                        error_type=type(e).__name__,
                        message=str(e),
                        severity=severity,
                        traceback=traceback.format_exc(),
                        recovery_strategy=RecoveryStrategy.RETRY
                    )
                    
                    error_handler.handle_error(error_event)
                    
                    if retries >= max_retries:
                        raise  # Re-raise if max retries exceeded
                    
                    retries += 1
                    time.sleep(0.1 * retries)  # Exponential backoff
            
            return None  # Should never reach here
        return wrapper
    return decorator


class ErrorHandler:
    """Singleton error handler with comprehensive error management"""
    
    _instance = None
    
    @classmethod
    def initialize(cls, max_error_history: int = 100,
                  memory_threshold_mb: float = 512.0) -> 'ErrorHandler':
        """Initialize the error handler singleton"""
        if cls._instance is None:
            cls._instance = cls(max_error_history, memory_threshold_mb)
        return cls._instance
    
    @classmethod
    def get_instance(cls) -> 'ErrorHandler':
        """Get the error handler singleton instance"""
        if cls._instance is None:
            raise RuntimeError("ErrorHandler not initialized. Call initialize() first.")
        return cls._instance
    
    def __init__(self, max_error_history: int, memory_threshold_mb: float):
        """Initialize error handler"""
        self._error_history: List[ErrorEvent] = []
        self._max_error_history = max_error_history
        self._components: Dict[str, Dict[str, Any]] = {}
        self._recovery_callbacks: Dict[str, List[Callable]] = {}
        self._last_error_time = None
        
        # Initialize memory monitor
        self.memory_monitor = MemoryMonitor(
            threshold_mb=memory_threshold_mb,
            check_interval=1.0
        )
        self.memory_monitor.start_monitoring()
    
    def register_component(self, component_name: str) -> None:
        """Register a component for error tracking"""
        if component_name not in self._components:
            self._components[component_name] = {
                'error_count': 0,
                'last_error': None,
                'status': 'healthy',
                'recovery_count': 0
            }
    
    def add_recovery_callback(self, component: str, callback: Callable) -> None:
        """Add recovery callback for a component"""
        if component not in self._recovery_callbacks:
            self._recovery_callbacks[component] = []
        self._recovery_callbacks[component].append(callback)
    
    def handle_error(self, error_event: ErrorEvent) -> None:
        """Handle an error event"""
        # Update error history
        self._error_history.append(error_event)
        if len(self._error_history) > self._max_error_history:
            self._error_history.pop(0)
        
        self._last_error_time = error_event.timestamp
        
        # Update component status
        if error_event.component in self._components:
            component = self._components[error_event.component]
            component['error_count'] += 1
            component['last_error'] = error_event
            
            # Update status based on error count and severity
            if error_event.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
                component['status'] = 'failing'
            elif component['error_count'] > 5:
                component['status'] = 'degraded'
        
        # Try recovery if strategy specified
        if error_event.recovery_strategy != RecoveryStrategy.NONE:
            self.attempt_recovery(error_event)
    
    def attempt_recovery(self, error_event: ErrorEvent) -> bool:
        """Attempt to recover from an error"""
        component = error_event.component
        if component not in self._components:
            return False
        
        # Execute recovery callbacks
        if component in self._recovery_callbacks:
            for callback in self._recovery_callbacks[component]:
                try:
                    callback()
                except Exception as e:
                    print(f"Recovery callback failed: {e}")
        
        # Update recovery stats
        self._components[component]['recovery_count'] += 1
        return True
    
    def force_component_recovery(self, component: str) -> bool:
        """Force recovery of a component"""
        if component not in self._components:
            return False
        
        if component in self._recovery_callbacks:
            for callback in self._recovery_callbacks[component]:
                try:
                    callback()
                except Exception as e:
                    print(f"Forced recovery failed: {e}")
                    return False
            return True
        return False
    
    def get_component_health(self, component: str) -> Dict[str, Any]:
        """Get health status of a component"""
        if component not in self._components:
            return {'status': 'unknown', 'error_count': 0}
        
        return {
            'status': self._components[component]['status'],
            'error_count': self._components[component]['error_count'],
            'last_error': self._components[component]['last_error'],
            'recovery_count': self._components[component]['recovery_count']
        }
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics"""
        now = datetime.now()
        one_hour_ago = now - timedelta(hours=1)
        
        recent_errors = [e for e in self._error_history 
                        if e.timestamp > one_hour_ago.timestamp()]
        
        # Count errors by component
        errors_by_component = {}
        for component in self._components:
            errors_by_component[component] = len([e for e in self._error_history 
                                               if e.component == component])
        
        # Count errors by severity
        errors_by_severity = {}
        for severity in ErrorSeverity:
            errors_by_severity[severity.value] = len([e for e in self._error_history 
                                                    if e.severity == severity])
        
        # Calculate component health rate
        healthy_components = len([c for c in self._components.values() 
                                if c['status'] == 'healthy'])
        health_rate = healthy_components / len(self._components) if self._components else 1.0
        
        return {
            'total_errors': len(self._error_history),
            'recent_errors_1h': len(recent_errors),
            'errors_by_component': errors_by_component,
            'errors_by_severity': errors_by_severity,
            'component_health': {
                'health_rate': health_rate,
                'healthy_count': healthy_components,
                'total_components': len(self._components)
            },
            'memory_info': self.memory_monitor.get_memory_info()
        }
    
    def shutdown(self) -> None:
        """Clean shutdown of the error handler"""
        if self.memory_monitor:
            self.memory_monitor.stop_monitoring()
