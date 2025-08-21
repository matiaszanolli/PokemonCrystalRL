"""
Data bus module for component communication
"""

from enum import Enum
from typing import Dict, Any, Optional, List
import threading
import time
import queue
from dataclasses import dataclass, asdict



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
        
    def register_component(self, component_id: str, metadata: Dict[str, Any]) -> None:
        """Register a component with the data bus.
        
        Args:
            component_id: Unique identifier for the component
            metadata: Component metadata dictionary
        """
        with self._lock:
            self._components[component_id] = {
                "metadata": metadata,
                "last_seen": time.time()
            }
            
    def unregister_component(self, component_id: str) -> None:
        """Remove a component from the data bus.
        
        Args:
            component_id: ID of component to remove
        """
        with self._lock:
            if component_id in self._components:
                del self._components[component_id]
                
            # Remove any subscriptions
            for data_type in self._subscribers:
                self._subscribers[data_type] = [
                    s for s in self._subscribers[data_type] 
                    if s['component_id'] != component_id
                ]
                
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
                    subscription['callback'](data, publisher_id)
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
            
    def shutdown(self) -> None:
        """Shutdown the data bus."""
        self._active = False
        # Clear subscribers
        self._subscribers = {}
        # Clear components
        self._components = {}
        

# Global data bus instance
_DATA_BUS = None

def get_data_bus() -> Optional[DataBus]:
    """Get the global data bus instance.
    
    Returns:
        Global DataBus instance or None if not initialized
    """
    return _DATA_BUS

def init_data_bus() -> DataBus:
    """Initialize the global data bus.
    
    Returns:
        Initialized DataBus instance
    """
    global _DATA_BUS
    if _DATA_BUS is None:
        _DATA_BUS = DataBus()
    return _DATA_BUS
