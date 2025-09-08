"""
Circuit Breaker - Fault tolerance and system protection

Extracted from ErrorHandler to implement the circuit breaker pattern
for preventing cascade failures and providing system resilience.
"""

import time
import logging
import threading
from typing import Dict, Optional, Callable, Any
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict, deque


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"         # Circuit is open (failures detected)
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    error_threshold: int = 5           # Errors before opening circuit
    time_window: float = 60.0          # Time window for error counting (seconds)
    reset_timeout: float = 300.0       # Time before attempting reset (seconds)
    success_threshold: int = 3         # Successes needed to close circuit in half-open
    max_failures_per_component: int = 10  # Max failures before permanent open


class CircuitBreakerMetrics:
    """Metrics tracking for circuit breaker."""
    
    def __init__(self, window_size: int = 100):
        self.error_count = 0
        self.success_count = 0
        self.total_requests = 0
        self.state_changes = 0
        
        # Sliding window for recent events
        self.recent_events = deque(maxlen=window_size)
        self.last_failure_time: Optional[float] = None
        self.last_success_time: Optional[float] = None
        self.circuit_opened_time: Optional[float] = None
        self.circuit_closed_time: Optional[float] = None
    
    def record_success(self):
        """Record successful operation."""
        self.success_count += 1
        self.total_requests += 1
        self.last_success_time = time.time()
        self.recent_events.append(('success', time.time()))
    
    def record_failure(self):
        """Record failed operation."""
        self.error_count += 1
        self.total_requests += 1
        self.last_failure_time = time.time()
        self.recent_events.append(('failure', time.time()))
    
    def record_state_change(self, new_state: CircuitBreakerState):
        """Record state change."""
        self.state_changes += 1
        current_time = time.time()
        
        if new_state == CircuitBreakerState.OPEN:
            self.circuit_opened_time = current_time
        elif new_state == CircuitBreakerState.CLOSED:
            self.circuit_closed_time = current_time
    
    def get_error_rate(self, time_window: float = 300.0) -> float:
        """Get error rate within time window."""
        if not self.recent_events:
            return 0.0
        
        current_time = time.time()
        recent_failures = sum(
            1 for event_type, timestamp in self.recent_events
            if event_type == 'failure' and current_time - timestamp <= time_window
        )
        recent_total = sum(
            1 for event_type, timestamp in self.recent_events
            if current_time - timestamp <= time_window
        )
        
        return recent_failures / max(recent_total, 1)


class CircuitBreaker:
    """Circuit breaker implementation for fault tolerance."""
    
    def __init__(self, config: CircuitBreakerConfig = None):
        self.config = config or CircuitBreakerConfig()
        self.logger = logging.getLogger("CircuitBreaker")
        
        # Component tracking
        self.components: Dict[str, CircuitBreakerState] = {}
        self.component_metrics: Dict[str, CircuitBreakerMetrics] = {}
        self.component_failures: Dict[str, deque] = defaultdict(lambda: deque())
        self.component_last_failure: Dict[str, float] = {}
        self.component_success_count: Dict[str, int] = defaultdict(int)
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Callbacks
        self.on_circuit_open: Optional[Callable[[str], None]] = None
        self.on_circuit_close: Optional[Callable[[str], None]] = None
        self.on_circuit_half_open: Optional[Callable[[str], None]] = None
        
        self.logger.info("Circuit breaker initialized")
    
    def register_component(self, component: str) -> None:
        """Register a component for circuit breaking.
        
        Args:
            component: Name of component to protect
        """
        with self._lock:
            if component not in self.components:
                self.components[component] = CircuitBreakerState.CLOSED
                self.component_metrics[component] = CircuitBreakerMetrics()
                self.logger.info(f"Registered component for circuit breaking: {component}")
    
    def record_success(self, component: str) -> None:
        """Record successful operation for component.
        
        Args:
            component: Component name
        """
        with self._lock:
            self.register_component(component)
            
            metrics = self.component_metrics[component]
            metrics.record_success()
            
            current_state = self.components[component]
            
            # Handle success in different states
            if current_state == CircuitBreakerState.HALF_OPEN:
                self.component_success_count[component] += 1
                
                # Close circuit if enough successes
                if self.component_success_count[component] >= self.config.success_threshold:
                    self._change_state(component, CircuitBreakerState.CLOSED)
                    self.component_success_count[component] = 0
                    
            elif current_state == CircuitBreakerState.OPEN:
                # Reset success count
                self.component_success_count[component] = 0
    
    def record_error(self, component: str) -> bool:
        """Record error for component and check if circuit should open.
        
        Args:
            component: Component name
            
        Returns:
            bool: True if circuit opened due to this error
        """
        with self._lock:
            self.register_component(component)
            
            current_time = time.time()
            metrics = self.component_metrics[component]
            metrics.record_failure()
            
            # Add error to sliding window
            failures = self.component_failures[component]
            failures.append(current_time)
            
            # Clean old failures outside time window
            cutoff_time = current_time - self.config.time_window
            while failures and failures[0] < cutoff_time:
                failures.popleft()
            
            # Update last failure time
            self.component_last_failure[component] = current_time
            
            current_state = self.components[component]
            
            # Check if circuit should open
            if current_state == CircuitBreakerState.CLOSED:
                if len(failures) >= self.config.error_threshold:
                    self._change_state(component, CircuitBreakerState.OPEN)
                    return True
                    
            elif current_state == CircuitBreakerState.HALF_OPEN:
                # Any failure in half-open state opens circuit
                self._change_state(component, CircuitBreakerState.OPEN)
                self.component_success_count[component] = 0
                return True
            
            return False
    
    def is_open(self, component: str) -> bool:
        """Check if circuit is open for component.
        
        Args:
            component: Component name
            
        Returns:
            bool: True if circuit is open
        """
        with self._lock:
            if component not in self.components:
                return False
            
            current_state = self.components[component]
            
            # If open, check if reset timeout has passed
            if current_state == CircuitBreakerState.OPEN:
                last_failure = self.component_last_failure.get(component, 0)
                if time.time() - last_failure >= self.config.reset_timeout:
                    self._change_state(component, CircuitBreakerState.HALF_OPEN)
                    return False
                return True
            
            return False
    
    def is_half_open(self, component: str) -> bool:
        """Check if circuit is in half-open state.
        
        Args:
            component: Component name
            
        Returns:
            bool: True if circuit is half-open
        """
        with self._lock:
            return self.components.get(component) == CircuitBreakerState.HALF_OPEN
    
    def force_open(self, component: str, reason: str = "manual") -> None:
        """Manually open circuit for component.
        
        Args:
            component: Component name
            reason: Reason for opening circuit
        """
        with self._lock:
            self.register_component(component)
            self._change_state(component, CircuitBreakerState.OPEN)
            self.logger.warning(f"Circuit manually opened for {component}: {reason}")
    
    def force_close(self, component: str, reason: str = "manual") -> None:
        """Manually close circuit for component.
        
        Args:
            component: Component name
            reason: Reason for closing circuit
        """
        with self._lock:
            self.register_component(component)
            self._change_state(component, CircuitBreakerState.CLOSED)
            
            # Reset counters
            self.component_success_count[component] = 0
            self.component_failures[component].clear()
            
            self.logger.info(f"Circuit manually closed for {component}: {reason}")
    
    def _change_state(self, component: str, new_state: CircuitBreakerState) -> None:
        """Change circuit state and trigger callbacks.
        
        Args:
            component: Component name
            new_state: New circuit state
        """
        old_state = self.components[component]
        if old_state == new_state:
            return
        
        self.components[component] = new_state
        self.component_metrics[component].record_state_change(new_state)
        
        self.logger.info(f"Circuit state changed for {component}: {old_state.value} -> {new_state.value}")
        
        # Trigger callbacks
        if new_state == CircuitBreakerState.OPEN and self.on_circuit_open:
            try:
                self.on_circuit_open(component)
            except Exception as e:
                self.logger.error(f"Error in circuit open callback: {e}")
                
        elif new_state == CircuitBreakerState.CLOSED and self.on_circuit_close:
            try:
                self.on_circuit_close(component)
            except Exception as e:
                self.logger.error(f"Error in circuit close callback: {e}")
                
        elif new_state == CircuitBreakerState.HALF_OPEN and self.on_circuit_half_open:
            try:
                self.on_circuit_half_open(component)
            except Exception as e:
                self.logger.error(f"Error in circuit half-open callback: {e}")
    
    def get_component_status(self, component: str) -> Dict[str, Any]:
        """Get status information for component.
        
        Args:
            component: Component name
            
        Returns:
            Dict: Status information
        """
        with self._lock:
            if component not in self.components:
                return {
                    'registered': False,
                    'state': 'unknown'
                }
            
            metrics = self.component_metrics[component]
            failures = self.component_failures[component]
            
            status = {
                'registered': True,
                'state': self.components[component].value,
                'error_count': metrics.error_count,
                'success_count': metrics.success_count,
                'total_requests': metrics.total_requests,
                'recent_failures': len(failures),
                'error_rate': metrics.get_error_rate(),
                'last_failure_time': self.component_last_failure.get(component),
                'last_success_time': metrics.last_success_time,
                'success_streak': self.component_success_count.get(component, 0),
                'state_changes': metrics.state_changes
            }
            
            # Add state-specific information
            current_state = self.components[component]
            if current_state == CircuitBreakerState.OPEN:
                last_failure = self.component_last_failure.get(component, 0)
                time_until_half_open = max(0, self.config.reset_timeout - (time.time() - last_failure))
                status['time_until_half_open'] = time_until_half_open
                
            elif current_state == CircuitBreakerState.HALF_OPEN:
                status['successes_needed'] = max(0, self.config.success_threshold - self.component_success_count.get(component, 0))
            
            return status
    
    def get_all_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status for all registered components.
        
        Returns:
            Dict: Status for each component
        """
        with self._lock:
            return {
                component: self.get_component_status(component)
                for component in self.components
            }
    
    def reset_component(self, component: str) -> None:
        """Reset circuit breaker state for component.
        
        Args:
            component: Component name
        """
        with self._lock:
            if component in self.components:
                self.components[component] = CircuitBreakerState.CLOSED
                self.component_failures[component].clear()
                self.component_success_count[component] = 0
                
                # Reset metrics
                self.component_metrics[component] = CircuitBreakerMetrics()
                
                self.logger.info(f"Circuit breaker reset for component: {component}")
    
    def get_global_status(self) -> Dict[str, Any]:
        """Get global circuit breaker statistics.
        
        Returns:
            Dict: Global statistics
        """
        with self._lock:
            total_components = len(self.components)
            open_circuits = sum(1 for state in self.components.values() if state == CircuitBreakerState.OPEN)
            half_open_circuits = sum(1 for state in self.components.values() if state == CircuitBreakerState.HALF_OPEN)
            closed_circuits = total_components - open_circuits - half_open_circuits
            
            total_errors = sum(metrics.error_count for metrics in self.component_metrics.values())
            total_successes = sum(metrics.success_count for metrics in self.component_metrics.values())
            total_requests = sum(metrics.total_requests for metrics in self.component_metrics.values())
            
            return {
                'total_components': total_components,
                'closed_circuits': closed_circuits,
                'open_circuits': open_circuits,
                'half_open_circuits': half_open_circuits,
                'total_errors': total_errors,
                'total_successes': total_successes,
                'total_requests': total_requests,
                'global_error_rate': total_errors / max(total_requests, 1),
                'config': {
                    'error_threshold': self.config.error_threshold,
                    'time_window': self.config.time_window,
                    'reset_timeout': self.config.reset_timeout,
                    'success_threshold': self.config.success_threshold
                }
            }
    
    def create_context_manager(self, component: str, operation: str = "operation"):
        """Create context manager for circuit breaker protection.
        
        Args:
            component: Component name
            operation: Operation description
            
        Returns:
            Context manager for circuit breaker
        """
        return CircuitBreakerContext(self, component, operation)


class CircuitBreakerContext:
    """Context manager for circuit breaker operations."""
    
    def __init__(self, circuit_breaker: CircuitBreaker, component: str, operation: str):
        self.circuit_breaker = circuit_breaker
        self.component = component
        self.operation = operation
        self.start_time = None
    
    def __enter__(self):
        # Check if circuit is open
        if self.circuit_breaker.is_open(self.component):
            raise CircuitBreakerOpenError(f"Circuit is open for component: {self.component}")
        
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None:
            # Success
            self.circuit_breaker.record_success(self.component)
        else:
            # Failure
            self.circuit_breaker.record_error(self.component)
        
        return False  # Don't suppress exceptions


class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open."""
    pass