#!/usr/bin/env python3
"""
Comprehensive Error Handling and Recovery System

This module provides robust error boundaries, memory leak prevention,
graceful degradation, and automatic recovery mechanisms for the training system.
"""

import time
import threading
import logging
import traceback
import gc
import psutil
import os
import signal
import weakref
from typing import Dict, Any, List, Optional, Callable, Type
from dataclasses import dataclass, asdict
from enum import Enum, auto

class ErrorCategory(Enum):
    """Categories of errors that can occur."""
    SYSTEM = 'system'
    DATABASE = 'database'
    NETWORK = 'network'
    TRAINING = 'training'
    MONITORING = 'monitoring'
    UI = 'ui'
    OTHER = 'other'
from collections import defaultdict, deque
import functools

try:
    from .data_bus import get_data_bus, DataType
except ImportError:
    from data_bus import get_data_bus, DataType


class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RecoveryStrategy(Enum):
    """Recovery strategies for different types of errors"""
    IGNORE = "ignore"
    RETRY = "retry"
    RESTART_COMPONENT = "restart_component"
    RESTART_SYSTEM = "restart_system"
    GRACEFUL_SHUTDOWN = "graceful_shutdown"


@dataclass
class ErrorEvent:
    """Error event information"""
    timestamp: float
    component: str
    error_type: str
    message: str
    severity: ErrorSeverity
    traceback: Optional[str]
    recovery_strategy: RecoveryStrategy
    attempt_count: int = 0
    resolved: bool = False


@dataclass
class ComponentHealth:
    """Component health status"""
    name: str
    status: str  # "healthy", "degraded", "failed"
    last_error: Optional[ErrorEvent]
    error_count: int
    restart_count: int
    last_restart: Optional[float]
    memory_usage_mb: float
    uptime_seconds: float
    is_responsive: bool


class ErrorBoundary:
    """Error boundary decorator for component methods"""
    
    def __init__(self, component_name: str, severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                 recovery_strategy: RecoveryStrategy = RecoveryStrategy.RETRY,
                 max_retries: int = 3):
        self.component_name = component_name
        self.severity = severity
        self.recovery_strategy = recovery_strategy
        self.max_retries = max_retries
        
    def __call__(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            error_handler = ErrorHandler.get_instance()
            
            for attempt in range(self.max_retries + 1):
                try:
                    return func(*args, **kwargs)
                    
                except Exception as e:
                    error_event = ErrorEvent(
                        timestamp=time.time(),
                        component=self.component_name,
                        error_type=type(e).__name__,
                        message=str(e),
                        severity=self.severity,
                        traceback=traceback.format_exc(),
                        recovery_strategy=self.recovery_strategy,
                        attempt_count=attempt + 1
                    )
                    
                    # Handle the error through the error handler
                    should_retry = error_handler.handle_error(error_event)
                    
                    if not should_retry or attempt >= self.max_retries:
                        # Final attempt failed or no retry needed
                        raise
                    
                    # Wait before retry with exponential backoff
                    wait_time = min(2 ** attempt, 30)  # Cap at 30 seconds
                    time.sleep(wait_time)
                    
        return wrapper


class MemoryMonitor:
    """Monitor and prevent memory leaks"""
    
    def __init__(self, threshold_mb: float = 1024.0, check_interval: float = 30.0):
        self.threshold_mb = threshold_mb
        self.check_interval = check_interval
        self._monitoring_active = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._memory_history: deque = deque(maxlen=100)
        
        # Setup logging
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
    def start_monitoring(self) -> None:
        """Start memory monitoring"""
        if self._monitoring_active:
            return
            
        self._monitoring_active = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name="MemoryMonitor"
        )
        self._monitor_thread.start()
        self.logger.info("📊 Memory monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop memory monitoring"""
        if not self._monitoring_active:
            return
            
        self._monitoring_active = False
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5.0)
        
        self.logger.info("📊 Memory monitoring stopped")
    
    def get_memory_info(self) -> Dict[str, Any]:
        """Get current memory usage information"""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            
            return {
                "rss_mb": memory_info.rss / (1024 * 1024),
                "vms_mb": memory_info.vms / (1024 * 1024),
                "percent": process.memory_percent(),
                "available_mb": psutil.virtual_memory().available / (1024 * 1024),
                "threshold_mb": self.threshold_mb,
                "history": list(self._memory_history)
            }
        except Exception as e:
            self.logger.error(f"Failed to get memory info: {e}")
            return {}
    
    def trigger_garbage_collection(self) -> Dict[str, Any]:
        """Force garbage collection and return statistics"""
        before_memory = self.get_memory_info().get("rss_mb", 0)
        
        # Force garbage collection
        collected = gc.collect()
        
        after_memory = self.get_memory_info().get("rss_mb", 0)
        freed_mb = before_memory - after_memory
        
        gc_stats = {
            "objects_collected": collected,
            "memory_before_mb": before_memory,
            "memory_after_mb": after_memory,
            "memory_freed_mb": freed_mb,
            "timestamp": time.time()
        }
        
        self.logger.info(f"🧹 Garbage collection: {collected} objects, {freed_mb:.1f}MB freed")
        return gc_stats
    
    def _monitor_loop(self) -> None:
        """Memory monitoring loop"""
        while self._monitoring_active:
            try:
                memory_info = self.get_memory_info()
                current_memory = memory_info.get("rss_mb", 0)
                
                # Store in history
                self._memory_history.append({
                    "timestamp": time.time(),
                    "memory_mb": current_memory
                })
                
                # Check for memory threshold violation
                if current_memory > self.threshold_mb:
                    self.logger.warning(f"📊 Memory usage high: {current_memory:.1f}MB > {self.threshold_mb}MB")
                    
                    # Trigger garbage collection
                    gc_stats = self.trigger_garbage_collection()
                    
                    # Notify error handler if still high after GC
                    if gc_stats["memory_after_mb"] > self.threshold_mb:
                        error_handler = ErrorHandler.get_instance()
                        if error_handler:
                            error_handler._handle_memory_issue(memory_info, gc_stats)
                
                time.sleep(self.check_interval)
                
            except Exception as e:
                self.logger.error(f"Memory monitor error: {e}")
                time.sleep(self.check_interval)


class ErrorHandler:
    """
    Comprehensive error handling and recovery system
    
    Features:
    - Error boundaries and isolation
    - Component health monitoring
    - Automatic recovery strategies
    - Memory leak prevention
    - Graceful degradation
    """
    
    _instance: Optional['ErrorHandler'] = None
    _instance_lock = threading.Lock()
    
    def __init__(self, 
                 max_error_history: int = 1000,
                 component_restart_threshold: int = 5,
                 memory_threshold_mb: float = 1024.0):
        
        self.max_error_history = max_error_history
        self.component_restart_threshold = component_restart_threshold
        self.memory_threshold_mb = memory_threshold_mb
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Error tracking
        self._error_history: deque = deque(maxlen=max_error_history)
        self._error_counts: Dict[str, int] = defaultdict(int)
        self._component_health: Dict[str, ComponentHealth] = {}
        
        # Recovery state
        self._recovery_callbacks: Dict[str, List[Callable]] = defaultdict(list)
        self._component_instances: Dict[str, weakref.ref] = {}
        
        # Memory monitoring
        self.memory_monitor = MemoryMonitor(memory_threshold_mb)
        
        # Setup logging
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Data bus integration
        self.data_bus = get_data_bus()
        if self.data_bus:
            self.data_bus.register_component("error_handler", {
                "type": "error_handling",
                "max_history": max_error_history,
                "memory_threshold": memory_threshold_mb
            })
        
        # Start memory monitoring
        self.memory_monitor.start_monitoring()
        
        self.logger.info("🛡️ ErrorHandler initialized")
    
    @classmethod
    def get_instance(cls) -> Optional['ErrorHandler']:
        """Get singleton instance of ErrorHandler"""
        return cls._instance
    
    @classmethod
    def initialize(cls, **kwargs) -> 'ErrorHandler':
        """Initialize singleton instance of ErrorHandler"""
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls(**kwargs)
        return cls._instance
    
    def register_component(self, name: str, instance: Any = None) -> None:
        """Register a component for monitoring and recovery"""
        with self._lock:
            self._component_health[name] = ComponentHealth(
                name=name,
                status="healthy",
                last_error=None,
                error_count=0,
                restart_count=0,
                last_restart=None,
                memory_usage_mb=0.0,
                uptime_seconds=0.0,
                is_responsive=True
            )
            
            # Store weak reference to component instance
            if instance is not None:
                self._component_instances[name] = weakref.ref(instance)
        
        self.logger.info(f"📋 Registered component: {name}")
    
    def add_recovery_callback(self, component_name: str, callback: Callable) -> None:
        """Add a recovery callback for a component"""
        with self._lock:
            self._recovery_callbacks[component_name].append(callback)
        
        self.logger.debug(f"Added recovery callback for {component_name}")
    
    def handle_error(self, error_event: ErrorEvent) -> bool:
        """
        Handle an error event and determine if operation should be retried
        
        Returns:
            True if operation should be retried, False otherwise
        """
        with self._lock:
            # Record error
            self._error_history.append(error_event)
            self._error_counts[error_event.component] += 1
            
            # Update component health
            if error_event.component in self._component_health:
                health = self._component_health[error_event.component]
                health.last_error = error_event
                health.error_count += 1
                
                # Update status based on error severity and frequency
                if error_event.severity == ErrorSeverity.CRITICAL:
                    health.status = "failed"
                elif health.error_count >= 3:
                    health.status = "degraded"
            
            # Log error
            log_level = {
                ErrorSeverity.LOW: logging.INFO,
                ErrorSeverity.MEDIUM: logging.WARNING,
                ErrorSeverity.HIGH: logging.ERROR,
                ErrorSeverity.CRITICAL: logging.CRITICAL
            }[error_event.severity]
            
            self.logger.log(log_level, 
                f"🚨 Error in {error_event.component}: {error_event.message} "
                f"(attempt {error_event.attempt_count}, severity: {error_event.severity.value})")
            
            # Publish to data bus
            if self.data_bus:
                self.data_bus.publish(DataType.ERROR_EVENT, {
                    "component": error_event.component,
                    "error_type": error_event.error_type,
                    "message": error_event.message,
                    "severity": error_event.severity.value,
                    "attempt_count": error_event.attempt_count,
                    "recovery_strategy": error_event.recovery_strategy.value
                }, "error_handler")
            
            # Execute recovery strategy
            return self._execute_recovery_strategy(error_event)
    
    def get_component_health(self, component_name: Optional[str] = None) -> Dict[str, Any]:
        """Get health status of components"""
        with self._lock:
            if component_name:
                health = self._component_health.get(component_name)
                return asdict(health) if health else {}
            else:
                return {name: asdict(health) for name, health in self._component_health.items()}
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics and health overview"""
        with self._lock:
            current_time = time.time()
            
            # Recent errors (last hour)
            recent_errors = [
                error for error in self._error_history 
                if (current_time - error.timestamp) < 3600
            ]
            
            # Error counts by component
            error_by_component = defaultdict(int)
            error_by_severity = defaultdict(int)
            
            for error in recent_errors:
                error_by_component[error.component] += 1
                error_by_severity[error.severity.value] += 1
            
            # Component health summary
            healthy_components = sum(1 for h in self._component_health.values() if h.status == "healthy")
            total_components = len(self._component_health)
            
            return {
                "total_errors": len(self._error_history),
                "recent_errors_1h": len(recent_errors),
                "errors_by_component": dict(error_by_component),
                "errors_by_severity": dict(error_by_severity),
                "component_health": {
                    "healthy": healthy_components,
                    "total": total_components,
                    "health_rate": healthy_components / max(total_components, 1)
                },
                "memory_info": self.memory_monitor.get_memory_info()
            }
    
    def force_component_recovery(self, component_name: str) -> bool:
        """Force recovery of a specific component"""
        with self._lock:
            if component_name not in self._component_health:
                self.logger.error(f"Unknown component: {component_name}")
                return False
            
            self.logger.info(f"🔄 Forcing recovery of component: {component_name}")
            
            # Execute recovery callbacks
            callbacks = self._recovery_callbacks.get(component_name, [])
            for callback in callbacks:
                try:
                    callback()
                except Exception as e:
                    self.logger.error(f"Recovery callback failed for {component_name}: {e}")
            
            # Reset component health
            health = self._component_health[component_name]
            health.status = "healthy"
            health.restart_count += 1
            health.last_restart = time.time()
            
            return True
    
    def shutdown(self) -> None:
        """Clean shutdown of error handler"""
        self.logger.info("🛑 Shutting down ErrorHandler")
        
        # Stop memory monitoring
        self.memory_monitor.stop_monitoring()
        
        # Notify data bus
        if self.data_bus:
            self.data_bus.publish(
                DataType.COMPONENT_STATUS,
                {"component": "error_handler", "status": "shutdown"},
                "error_handler"
            )
        
        self.logger.info("✅ ErrorHandler shutdown complete")
    
    # Internal methods
    
    def _execute_recovery_strategy(self, error_event: ErrorEvent) -> bool:
        """Execute the appropriate recovery strategy"""
        strategy = error_event.recovery_strategy
        
        if strategy == RecoveryStrategy.IGNORE:
            return False
        
        elif strategy == RecoveryStrategy.RETRY:
            # Allow retry if under max attempts
            return error_event.attempt_count < 3
        
        elif strategy == RecoveryStrategy.RESTART_COMPONENT:
            self.logger.info(f"🔄 Attempting component restart: {error_event.component}")
            return self._restart_component(error_event.component)
        
        elif strategy == RecoveryStrategy.RESTART_SYSTEM:
            self.logger.critical("🔄 System restart required")
            self._initiate_system_restart()
            return False
        
        elif strategy == RecoveryStrategy.GRACEFUL_SHUTDOWN:
            self.logger.critical("🛑 Initiating graceful shutdown")
            self._initiate_graceful_shutdown()
            return False
        
        return False
    
    def _restart_component(self, component_name: str) -> bool:
        """Attempt to restart a specific component"""
        try:
            # Check if we have restart callbacks
            callbacks = self._recovery_callbacks.get(component_name, [])
            if not callbacks:
                self.logger.warning(f"No recovery callbacks for {component_name}")
                return False
            
            # Execute restart callbacks
            for callback in callbacks:
                callback()
            
            # Update component health
            if component_name in self._component_health:
                health = self._component_health[component_name]
                health.restart_count += 1
                health.last_restart = time.time()
                health.status = "healthy"  # Optimistically assume restart worked
            
            self.logger.info(f"✅ Component restarted successfully: {component_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to restart component {component_name}: {e}")
            return False
    
    def _handle_memory_issue(self, memory_info: Dict[str, Any], gc_stats: Dict[str, Any]) -> None:
        """Handle high memory usage"""
        self.logger.warning(f"💾 High memory usage detected: {memory_info.get('rss_mb', 0):.1f}MB")
        
        # Create memory error event
        error_event = ErrorEvent(
            timestamp=time.time(),
            component="system",
            error_type="MemoryUsageHigh",
            message=f"Memory usage {memory_info.get('rss_mb', 0):.1f}MB exceeds threshold {self.memory_threshold_mb}MB",
            severity=ErrorSeverity.HIGH,
            traceback=None,
            recovery_strategy=RecoveryStrategy.RETRY  # Try GC and monitoring
        )
        
        # Handle through normal error handling
        self.handle_error(error_event)
    
    def _initiate_system_restart(self) -> None:
        """Initiate system restart (placeholder - implement based on deployment)"""
        self.logger.critical("🔄 System restart not implemented - manual intervention required")
        
        # In a real deployment, this might:
        # - Send alert to monitoring system
        # - Trigger container restart
        # - Exit process with specific code
        # - Send signal to supervisor process
    
    def _initiate_graceful_shutdown(self) -> None:
        """Initiate graceful shutdown"""
        self.logger.critical("🛑 Initiating graceful shutdown")
        
        # Send shutdown signal to main process
        try:
            os.kill(os.getpid(), signal.SIGTERM)
        except Exception as e:
            self.logger.error(f"Failed to send shutdown signal: {e}")


# Convenience decorators
def error_boundary(component_name: str, severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                   recovery_strategy: RecoveryStrategy = RecoveryStrategy.RETRY,
                   max_retries: int = 3):
    """Decorator for adding error boundaries to functions"""
    return ErrorBoundary(component_name, severity, recovery_strategy, max_retries)


# Context managers
class SafeOperation:
    """Context manager for safe operations with automatic error handling"""
    
    def __init__(self, component_name: str, operation_name: str):
        self.component_name = component_name
        self.operation_name = operation_name
        self.error_handler = ErrorHandler.get_instance()
        
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None and self.error_handler:
            error_event = ErrorEvent(
                timestamp=time.time(),
                component=self.component_name,
                error_type=exc_type.__name__,
                message=str(exc_val),
                severity=ErrorSeverity.MEDIUM,
                traceback=traceback.format_exc(),
                recovery_strategy=RecoveryStrategy.RETRY
            )
            self.error_handler.handle_error(error_event)
        
        # Don't suppress the exception
        return False
