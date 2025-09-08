"""
Data bus module for component communication
"""

from enum import Enum
from typing import Dict, Any, Optional, List
import threading
import time
import queue
import logging
from dataclasses import dataclass, asdict



class DataType(Enum):
    """Types of data that can be published to the data bus"""
    GAME_STATE = "game_state"
    TRAINING_STATS = "training_stats"
    TRAINING_STATE = "training_state"
    TRAINING_CONTROL = "training_control"
    TRAINING_METRICS = "training_metrics"
    ACTION_TAKEN = "action_taken"
    LLM_DECISION = "llm_decision"
    SYSTEM_INFO = "system_info"
    GAME_SCREEN = "game_screen"
    ERROR_EVENT = "error_event"
    ERROR_NOTIFICATION = "error_notification"
    COMPONENT_STATUS = "component_status"


@dataclass
class DataMessage:
    """Standard message format for the data bus"""
    data_type: DataType
    timestamp: float
    data: Dict[str, Any]
    source_component: str
    message_id: str


class DataBus:
    """
    Central data bus for component communication.
    
    Implements a publish-subscribe pattern for asynchronous component communication.
    """
    
    def __init__(self):
        """Initialize the data bus."""
        self._subscribers = {}
        self._components = {}
        self._lock = threading.Lock()
        self._queues = {}
        self._active = True
        self._running = True  # Alias for test compatibility
        # Set up logging
        self._logger = logging.getLogger("data_bus")
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s [%(name)s] %(levelname)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        if not self._logger.handlers:
            self._logger.addHandler(handler)
            self._logger.setLevel(logging.ERROR)  # Only show errors by default
        self._logger.debug("DataBus initialized")
        
    def register_component(self, component_id: str, metadata: Dict[str, Any]) -> None:
        """Register a component with the data bus.
        
        Args:
            component_id: Unique identifier for the component
            metadata: Component metadata dictionary
        """
        self._logger.debug(f"Registering component {component_id} with metadata {metadata}")
        with self._lock:
            self._components[component_id] = {
                "metadata": metadata,
                "last_seen": time.time()
            }
        self._logger.debug(f"Component {component_id} registered")
            
    def unregister_component(self, component_id: str) -> None:
        """Remove a component from the data bus.
        
        Args:
            component_id: ID of component to remove
        """
        self._logger.debug(f"Unregistering component {component_id}")
        with self._lock:
            if component_id in self._components:
                self._logger.debug(f"Removing {component_id} from components")
                del self._components[component_id]
                
            # Remove any subscriptions
            for data_type in self._subscribers:
                before_len = len(self._subscribers[data_type])
                self._subscribers[data_type] = [
                    s for s in self._subscribers[data_type] 
                    if s['component_id'] != component_id
                ]
                after_len = len(self._subscribers[data_type])
                if before_len != after_len:
                    self._logger.debug(f"Removed {before_len - after_len} subscriptions for {component_id}")
        self._logger.debug(f"Component {component_id} unregistered")
                
    def subscribe(self, data_type: DataType, component_id: str, 
                 callback: Optional[callable] = None,
                 queue_size: int = 100) -> Optional[queue.Queue]:
        """Subscribe to a data type.
        
        Args:
            data_type: Type of data to subscribe to
            component_id: ID of subscribing component
            callback: Optional callback function
            queue_size: Size of queue if using queue-based subscription
            
        Returns:
            Queue object if no callback provided, None otherwise
        """
        if data_type not in self._subscribers:
            self._subscribers[data_type] = []
            
        # Create subscription entry
        subscription = {
            'component_id': component_id,
            'callback': callback,
            'queue': None
        }
        
        # If no callback, create queue
        if not callback:
            subscription['queue'] = queue.Queue(maxsize=queue_size)
            
        self._subscribers[data_type].append(subscription)
        return subscription['queue']
        
    def publish(self, data_type: DataType, data: Dict[str, Any], 
                publisher_id: str) -> None:
        """Publish data to subscribers.
        
        Args:
            data_type: Type of data being published
            data: The data payload
            publisher_id: ID of publishing component
        """
        if not self._active:
            return
            
        if data_type not in self._subscribers:
            return
            
        # Update component last seen time
        with self._lock:
            if publisher_id in self._components:
                self._components[publisher_id]['last_seen'] = time.time()
                
        # Notify subscribers
        for subscription in self._subscribers[data_type]:
            try:
                if subscription['callback']:
                    try:
                        # Always pass data as first argument
                        if 'data' in data:
                            callback_data = data['data']
                        else:
                            callback_data = data
                        subscription['callback'](callback_data)
                    except Exception as e:
                        self._logger.error(f"Callback error: {e}")
                elif subscription['queue']:
                    try:
                        subscription['queue'].put_nowait({
                            'data': data,
                            'publisher': publisher_id,
                            'timestamp': time.time()
                        })
                    except queue.Full:
                        # Queue full - remove oldest item and try again
                        try:
                            subscription['queue'].get_nowait()
                            subscription['queue'].put_nowait({
                                'data': data,
                                'publisher': publisher_id,
                                'timestamp': time.time()
                            })
                        except (queue.Empty, queue.Full):
                            pass
            except Exception as e:
                print(f"Error delivering to subscriber: {e}")
                
    def get_component_status(self) -> Dict[str, Any]:
        """Get status of registered components.
        
        Returns:
            Dict with component status information
        """
        with self._lock:
            current_time = time.time()
            return {
                component_id: {
                    **info['metadata'],
                    'last_seen': current_time - info['last_seen']
                }
                for component_id, info in self._components.items()
            }
            
    def update_component_heartbeat(self, component_id: str) -> None:
        """Update the last seen time for a component."""
        with self._lock:
            if component_id in self._components:
                self._components[component_id]['last_seen'] = time.time()
    
    def shutdown(self) -> None:
        """Shutdown the data bus."""
        self._logger.debug("Starting data bus instance shutdown...")
        
        # First mark as inactive to prevent new messages
        with self._lock:
            self._active = False
            self._running = False
            remaining = list(self._components.keys())
            self._logger.debug(f"Data bus marked as inactive. Remaining components: {remaining}")
        
        # Give components a chance to unregister
        time.sleep(0.1)

        # Unregister remaining components outside the lock to avoid deadlocks
        if remaining:
            self._logger.debug(f"Cleaning up {len(remaining)} remaining components: {remaining}")
            for component_id in remaining:
                try:
                    self.unregister_component(component_id)
                except Exception as e:
                    self._logger.error(f"Error unregistering {component_id}: {e}")
        
        # Clear remaining data structures under the lock
        with self._lock:
            self._subscribers = {}
            self._components = {}
            
        self._logger.debug("Data bus shutdown complete")


# Global data bus instance (singleton pattern)
_global_data_bus: Optional[DataBus] = None
_data_bus_lock = threading.Lock()


def get_data_bus() -> DataBus:
    """Get the global data bus instance (thread-safe singleton)"""
    global _global_data_bus
    
    if _global_data_bus is None:
        with _data_bus_lock:
            if _global_data_bus is None:
                _global_data_bus = DataBus()
    
    return _global_data_bus


def shutdown_data_bus() -> None:
    """Shutdown the global data bus"""
    global _global_data_bus
    
    logger = logging.getLogger("data_bus")
    logger.setLevel(logging.DEBUG)  # Ensure debug is enabled
    
    logger.debug("Starting data bus shutdown sequence...")
    
    # Setup a watchdog timer for safety
    def watchdog():
        logger.warning("Data bus shutdown watchdog timeout - forcing cleanup")
        global _global_data_bus
        with _data_bus_lock:
            if _global_data_bus:
                logger.debug("Watchdog forcing data bus cleanup...")
                _global_data_bus._active = False
                _global_data_bus._running = False
                _global_data_bus._components.clear()
                _global_data_bus._subscribers.clear()
                logger.debug("Watchdog cleanup complete")
            _global_data_bus = None
            logger.debug("Watchdog cleared global reference")
    
    watchdog_timer = threading.Timer(3.0, watchdog)
    watchdog_timer.daemon = True
    watchdog_timer.start()
    logger.debug("Started 3s watchdog timer")
    
    try:
        # First get reference to data bus if it exists
        data_bus = None
        with _data_bus_lock:
            data_bus = _global_data_bus
            if data_bus is None:
                logger.debug("No data bus instance found")
                watchdog_timer.cancel()
                return
        
        # Get list of components to help with debugging
        try:
            components = data_bus.get_component_status()
            logger.debug(f"Components before shutdown: {components}")
        except Exception as e:
            logger.debug(f"Could not get component status: {e}")
        
        # If data bus exists, shut it down outside the lock
        # to prevent deadlocks with component unregistration
        try:
            data_bus.shutdown()
            logger.debug("Data bus shutdown completed")
        except Exception as e:
            logger.error(f"Error during data bus shutdown: {e}")
        
        # Clear global reference
        # Success - cancel watchdog and clear reference
        watchdog_timer.cancel()
        with _data_bus_lock:
            _global_data_bus = None
            logger.debug("Global data bus reference cleared")
            
    except Exception as e:
        logger.error(f"Unexpected error during data bus shutdown: {e}")
        # Let watchdog handle cleanup in case of failure
