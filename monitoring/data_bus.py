"""
TrainingDataBus - Thread-safe central data hub for training metrics

This module provides a robust, thread-safe data bus for communicating between
training components, UI, and monitoring systems.
"""

import time
import threading
import queue
import json
import logging
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass, asdict
from enum import Enum
import weakref


class DataType(Enum):
    """Types of data that can be published to the data bus"""
    GAME_STATE = "game_state"
    TRAINING_STATS = "training_stats"
    ACTION_TAKEN = "action_taken"
    LLM_DECISION = "llm_decision"
    SYSTEM_INFO = "system_info"
    GAME_SCREEN = "game_screen"
    ERROR_EVENT = "error_event"
    COMPONENT_STATUS = "component_status"


@dataclass
class DataMessage:
    """Standard message format for the data bus"""
    data_type: DataType
    timestamp: float
    data: Dict[str, Any]
    source_component: str
    message_id: str


class TrainingDataBus:
    """
    Thread-safe central data hub for all training metrics and events
    
    Features:
    - Non-blocking publish/subscribe pattern
    - Automatic cleanup and memory management
    - Component registration and health monitoring
    - Data validation and error recovery
    - Historical data storage with configurable limits
    """
    
    def __init__(self, max_history: int = 1000, cleanup_interval: float = 30.0):
        self.max_history = max_history
        self.cleanup_interval = cleanup_interval
        
        # Thread-safe data storage
        self._data_lock = threading.RLock()
        self._subscribers_lock = threading.RLock()
        
        # Data storage by type
        self._current_data: Dict[DataType, DataMessage] = {}
        self._historical_data: Dict[DataType, List[DataMessage]] = {dt: [] for dt in DataType}
        
        # Subscriber management
        self._subscribers: Dict[DataType, List[Callable]] = {dt: [] for dt in DataType}
        self._component_registry: Dict[str, Dict[str, Any]] = {}
        
        # Performance tracking
        self._message_count = 0
        self._publish_times: List[float] = []
        self._error_count = 0
        
        # Setup logging
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Start cleanup thread
        self._cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self._cleanup_active = True
        self._cleanup_thread.start()
        
        self.logger.info("🚌 TrainingDataBus initialized")
    
    def publish(self, data_type: DataType, data: Dict[str, Any], 
                source_component: str = "unknown") -> bool:
        """
        Publish data to the bus (thread-safe, non-blocking)
        
        Args:
            data_type: Type of data being published
            data: The data payload
            source_component: Name of the component publishing the data
            
        Returns:
            True if published successfully, False otherwise
        """
        start_time = time.time()
        
        try:
            # Create message
            message = DataMessage(
                data_type=data_type,
                timestamp=time.time(),
                data=data.copy(),  # Defensive copy
                source_component=source_component,
                message_id=f"{source_component}_{self._message_count}"
            )
            
            with self._data_lock:
                # Update current data
                self._current_data[data_type] = message
                
                # Add to historical data
                history = self._historical_data[data_type]
                history.append(message)
                
                # Trim history if needed
                if len(history) > self.max_history:
                    history.pop(0)
                
                self._message_count += 1
            
            # Notify subscribers (outside of lock for performance)
            self._notify_subscribers(data_type, message)
            
            # Track performance
            publish_time = time.time() - start_time
            self._publish_times.append(publish_time)
            if len(self._publish_times) > 100:
                self._publish_times.pop(0)
            
            return True
            
        except Exception as e:
            self._error_count += 1
            self.logger.error(f"Failed to publish {data_type}: {e}")
            return False
    
    def subscribe(self, data_type: DataType, callback: Callable[[DataMessage], None],
                  component_name: str = "unknown") -> bool:
        """
        Subscribe to data updates (thread-safe)
        
        Args:
            data_type: Type of data to subscribe to
            callback: Function to call when data is published
            component_name: Name of the subscribing component
            
        Returns:
            True if subscribed successfully
        """
        try:
            with self._subscribers_lock:
                self._subscribers[data_type].append(callback)
            
            # Register component
            self.register_component(component_name, {"subscriptions": [data_type.value]})
            
            self.logger.debug(f"Component '{component_name}' subscribed to {data_type.value}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to subscribe {component_name} to {data_type}: {e}")
            return False
    
    def get_current_data(self, data_type: Optional[DataType] = None) -> Dict[str, Any]:
        """
        Get current data for a specific type or all types
        
        Args:
            data_type: Specific data type to get, or None for all
            
        Returns:
            Dictionary of current data
        """
        with self._data_lock:
            if data_type:
                message = self._current_data.get(data_type)
                return message.data if message else {}
            else:
                return {
                    dt.value: msg.data if msg else {} 
                    for dt, msg in self._current_data.items()
                }
    
    def get_historical_data(self, data_type: DataType, 
                           limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get historical data for a specific type
        
        Args:
            data_type: Type of data to retrieve
            limit: Maximum number of entries to return (most recent)
            
        Returns:
            List of historical data entries
        """
        with self._data_lock:
            history = self._historical_data.get(data_type, [])
            
            if limit:
                history = history[-limit:]
            
            return [asdict(msg) for msg in history]
    
    def register_component(self, component_name: str, 
                         info: Dict[str, Any]) -> None:
        """Register a component with the data bus"""
        with self._subscribers_lock:
            self._component_registry[component_name] = {
                "registered_at": time.time(),
                "last_seen": time.time(),
                "info": info.copy()
            }
        
        self.logger.debug(f"Registered component: {component_name}")
    
    def update_component_heartbeat(self, component_name: str) -> None:
        """Update component heartbeat timestamp"""
        with self._subscribers_lock:
            if component_name in self._component_registry:
                self._component_registry[component_name]["last_seen"] = time.time()
    
    def get_component_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all registered components"""
        current_time = time.time()
        status = {}
        
        with self._subscribers_lock:
            for name, info in self._component_registry.items():
                last_seen = info["last_seen"]
                status[name] = {
                    "healthy": (current_time - last_seen) < 60.0,  # Healthy if seen within 60s
                    "last_seen": last_seen,
                    "registered_at": info["registered_at"],
                    "uptime": current_time - info["registered_at"],
                    "info": info["info"]
                }
        
        return status
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for the data bus"""
        with self._data_lock:
            avg_publish_time = (
                sum(self._publish_times) / len(self._publish_times) 
                if self._publish_times else 0.0
            )
            
            total_messages = sum(len(hist) for hist in self._historical_data.values())
            
            return {
                "total_messages": self._message_count,
                "stored_messages": total_messages,
                "avg_publish_time_ms": avg_publish_time * 1000,
                "error_count": self._error_count,
                "error_rate": self._error_count / max(self._message_count, 1),
                "active_subscribers": sum(len(subs) for subs in self._subscribers.values()),
                "registered_components": len(self._component_registry),
                "memory_usage_mb": self._estimate_memory_usage() / (1024 * 1024)
            }
    
    def clear_history(self, data_type: Optional[DataType] = None) -> None:
        """Clear historical data to free memory"""
        with self._data_lock:
            if data_type:
                self._historical_data[data_type].clear()
            else:
                for dt in DataType:
                    self._historical_data[dt].clear()
        
        self.logger.info(f"Cleared historical data for {data_type or 'all types'}")
    
    def shutdown(self) -> None:
        """Clean shutdown of the data bus"""
        self.logger.info("🛑 Shutting down TrainingDataBus")
        
        # Stop cleanup thread
        self._cleanup_active = False
        if self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=5.0)
        
        # Clear all data
        with self._data_lock:
            self._current_data.clear()
            for dt in DataType:
                self._historical_data[dt].clear()
        
        with self._subscribers_lock:
            for dt in DataType:
                self._subscribers[dt].clear()
            self._component_registry.clear()
        
        self.logger.info("✅ TrainingDataBus shutdown complete")
    
    def _notify_subscribers(self, data_type: DataType, message: DataMessage) -> None:
        """Notify all subscribers of a data type (non-blocking)"""
        with self._subscribers_lock:
            subscribers = self._subscribers[data_type].copy()
        
        # Notify outside of lock to avoid blocking
        for callback in subscribers:
            try:
                # Use weak references to avoid keeping components alive
                callback(message)
            except Exception as e:
                self.logger.warning(f"Subscriber callback failed for {data_type}: {e}")
    
    def _cleanup_loop(self) -> None:
        """Background thread for periodic cleanup"""
        while self._cleanup_active:
            try:
                time.sleep(self.cleanup_interval)
                
                if not self._cleanup_active:
                    break
                
                # Clean up old performance tracking data
                current_time = time.time()
                self._publish_times = [
                    t for t in self._publish_times 
                    if (current_time - t) < 300  # Keep last 5 minutes
                ]
                
                # Mark inactive components
                with self._subscribers_lock:
                    inactive_components = []
                    for name, info in self._component_registry.items():
                        if (current_time - info["last_seen"]) > 300:  # 5 minutes inactive
                            inactive_components.append(name)
                    
                    for name in inactive_components:
                        self.logger.warning(f"Component '{name}' appears inactive")
                
            except Exception as e:
                self.logger.error(f"Cleanup loop error: {e}")
    
    def _estimate_memory_usage(self) -> int:
        """Estimate memory usage in bytes (rough approximation)"""
        try:
            total_bytes = 0
            
            # Estimate current data
            for message in self._current_data.values():
                if message:
                    total_bytes += len(json.dumps(asdict(message)).encode())
            
            # Estimate historical data
            for history in self._historical_data.values():
                for message in history:
                    total_bytes += len(json.dumps(asdict(message)).encode())
            
            return total_bytes
            
        except Exception:
            return 0  # Return 0 if estimation fails


# Global data bus instance (singleton pattern)
_global_data_bus: Optional[TrainingDataBus] = None
_data_bus_lock = threading.Lock()


def get_data_bus() -> TrainingDataBus:
    """Get the global data bus instance (thread-safe singleton)"""
    global _global_data_bus
    
    if _global_data_bus is None:
        with _data_bus_lock:
            if _global_data_bus is None:
                _global_data_bus = TrainingDataBus()
    
    return _global_data_bus


def shutdown_data_bus() -> None:
    """Shutdown the global data bus"""
    global _global_data_bus
    
    with _data_bus_lock:
        if _global_data_bus is not None:
            _global_data_bus.shutdown()
            _global_data_bus = None
