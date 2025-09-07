"""
Core Monitoring Interfaces and Protocols

This module defines the base interfaces and protocols for the monitoring system.
These protocols provide a clean contract for monitoring components while enabling
modular architecture and easy testing.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Protocol, Dict, List, Set, Any, Optional, Callable, TypeVar

# Type variables for generic protocols
T = TypeVar('T')
K = TypeVar('K')

class MonitorComponent(Protocol):
    """Base protocol for all monitoring components.
    
    Any component in the monitoring system must implement this protocol,
    providing standardized lifecycle management and status reporting.
    """
    def start(self) -> bool:
        """Start the component operation.
        
        Returns:
            bool: True if started successfully, False otherwise
        """
        ...
    
    def stop(self) -> bool:
        """Stop the component operation.
        
        Returns:
            bool: True if stopped successfully, False otherwise
        """
        ...
    
    def get_status(self) -> Dict[str, Any]:
        """Get component status information.
        
        Returns:
            Dict[str, Any]: Component status details
        """
        ...

class DataPublisher(Protocol):
    """Protocol for components that publish monitoring data.
    
    Components implementing this protocol can publish data to topics,
    enabling decoupled communication between monitoring components.
    """
    def publish(self, topic: str, data: T) -> bool:
        """Publish data to a topic.
        
        Args:
            topic: The topic to publish to
            data: The data to publish
            
        Returns:
            bool: True if published successfully, False otherwise
        """
        ...

class DataSubscriber(Protocol):
    """Protocol for components that consume monitoring data.
    
    Components implementing this protocol can subscribe to topics and
    receive data through callbacks.
    """
    def subscribe(self, topic: str) -> bool:
        """Subscribe to a topic.
        
        Args:
            topic: The topic to subscribe to
            
        Returns:
            bool: True if subscribed successfully, False otherwise
        """
        ...
    
    def handle_data(self, topic: str, data: T) -> None:
        """Handle data received from a topic.
        
        Args:
            topic: The topic the data was published to
            data: The received data
        """
        ...

@dataclass
class MetricDefinition:
    """Definition of a metric type."""
    name: str
    type: str  # 'counter', 'gauge', 'histogram', etc.
    description: str
    unit: Optional[str] = None
    aggregation: str = 'last'  # 'sum', 'avg', 'min', 'max', 'last'

@dataclass
class ScreenCaptureConfig:
    """Configuration for screen capture component."""
    buffer_size: int = 10
    frame_rate: float = 30.0
    compression_quality: int = 85
    output_format: str = 'png'
    upscale_factor: int = 3  # For GameBoy's 160x144 -> 480x432

@dataclass
class MonitoringConfig:
    """Configuration for the monitoring system."""
    enable_capture: bool = True
    enable_metrics: bool = True
    enable_web: bool = True
    enable_data_bus: bool = True
    capture: ScreenCaptureConfig = field(default_factory=ScreenCaptureConfig)
    web: Dict[str, Any] = field(default_factory=lambda: {
        'host': 'localhost',
        'port': 5000,
        'debug': False
    })
    metrics: Dict[str, Any] = field(default_factory=lambda: {
        'retention_hours': 24,
        'storage_path': 'data/metrics'
    })

class BaseMonitor(ABC):
    """Abstract base class for monitors.
    
    This class defines the core interface that any monitor must implement,
    providing consistency across different monitoring implementations.
    """
    
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the monitor.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        pass
    
    @abstractmethod
    def start_monitoring(self) -> bool:
        """Start monitoring.
        
        Returns:
            bool: True if started successfully, False otherwise
        """
        pass
    
    @abstractmethod
    def stop_monitoring(self) -> bool:
        """Stop monitoring.
        
        Returns:
            bool: True if stopped successfully, False otherwise
        """
        pass
    
    @abstractmethod
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics.
        
        Returns:
            Dict[str, Any]: Current metric values
        """
        pass

class DataBusProtocol(Protocol):
    """Protocol for data bus implementations."""
    
    def publish(self, topic: str, data: Any) -> bool:
        """Publish data to a topic.
        
        Args:
            topic: Topic to publish to
            data: Data to publish
            
        Returns:
            bool: True if published successfully
        """
        ...
    
    def subscribe(self, topic: str, callback: Callable[[str, Any], None]) -> bool:
        """Subscribe to a topic with a callback.
        
        Args:
            topic: Topic to subscribe to
            callback: Function to call with received data
            
        Returns:
            bool: True if subscribed successfully
        """
        ...
    
    def unsubscribe(self, topic: str, callback: Callable[[str, Any], None]) -> bool:
        """Unsubscribe from a topic.
        
        Args:
            topic: Topic to unsubscribe from
            callback: Callback to remove
            
        Returns:
            bool: True if unsubscribed successfully
        """
        ...
    
    def list_topics(self) -> Set[str]:
        """Get list of active topics.
        
        Returns:
            Set[str]: Set of active topic names
        """
        ...

class MonitoringError(Exception):
    """Base class for monitoring errors."""
    pass

class ComponentError(MonitoringError):
    """Error from a specific component."""
    pass

class DataBusError(MonitoringError):
    """Error related to data bus operations."""
    pass
