"""Base interfaces for monitoring components."""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass
import numpy as np


@dataclass
class MonitoringStats:
    """Statistics data structure for monitoring."""
    episode: int = 0
    total_steps: int = 0
    total_reward: float = 0.0
    recent_actions: list = None
    training_state: Dict[str, Any] = None
    performance_metrics: Dict[str, Any] = None
    system_stats: Dict[str, Any] = None
    
    def __post_init__(self):
        self.recent_actions = self.recent_actions or []
        self.training_state = self.training_state or {}
        self.performance_metrics = self.performance_metrics or {}
        self.system_stats = self.system_stats or {}


class MonitoringComponent(ABC):
    """Base class for all monitoring components."""

    @abstractmethod
    def start(self) -> bool:
        """Start the monitoring component."""
        pass
        
    @abstractmethod
    def stop(self) -> bool:
        """Stop the monitoring component."""
        pass
    
    @abstractmethod
    def update_stats(self, stats: MonitoringStats) -> None:
        """Update component statistics.
        
        Args:
            stats: New statistics to update
        """
        pass


class ScreenCaptureComponent(MonitoringComponent):
    """Interface for screen capture functionality."""
    
    @abstractmethod
    def capture_screen(self) -> Optional[np.ndarray]:
        """Capture current screen state.
        
        Returns:
            Screen image as numpy array or None if capture failed
        """
        pass
    
    @abstractmethod
    def get_latest_screen(self) -> Optional[np.ndarray]:
        """Get most recent captured screen.
        
        Returns:
            Latest screen image or None if no captures available
        """
        pass


class WebMonitorInterface(MonitoringComponent):
    """Interface for web monitoring functionality."""
    
    @abstractmethod
    def get_port(self) -> int:
        """Get the port number the web monitor is running on.
        
        Returns:
            Port number
        """
        pass
        
    @abstractmethod
    def broadcast_update(self, update_type: str, data: Dict[str, Any]) -> None:
        """Broadcast an update to connected clients.
        
        Args:
            update_type: Type of update (e.g. 'stats', 'screen', etc)
            data: Update data to broadcast
        """
        pass
        
    @abstractmethod
    def add_endpoint(self, endpoint: str, handler_func: callable) -> None:
        """Add a new HTTP endpoint.
        
        Args:
            endpoint: URL path for endpoint
            handler_func: Function to handle endpoint requests
        """
        pass


class StatsCollectorInterface(MonitoringComponent):
    """Interface for statistics collection."""
    
    @abstractmethod
    def record_metric(self, name: str, value: Union[int, float], timestamp: Optional[float] = None) -> None:
        """Record a numeric metric.
        
        Args:
            name: Metric name
            value: Metric value
            timestamp: Optional timestamp for metric
        """
        pass
        
    @abstractmethod
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics.
        
        Returns:
            Dictionary of metrics
        """
        pass


class TrainingMonitorInterface(MonitoringComponent):
    """Interface for training monitoring functionality."""
    
    @abstractmethod
    def update_episode(self, episode: int, total_reward: float, steps: int) -> None:
        """Update episode information.
        
        Args:
            episode: Episode number
            total_reward: Total reward for episode
            steps: Number of steps in episode
        """
        pass
        
    @abstractmethod
    def update_step(self, step: int, reward: float, action: Optional[str] = None) -> None:
        """Update step information.
        
        Args:
            step: Step number
            reward: Reward for step
            action: Optional action taken
        """
        pass


class ErrorHandlerInterface(MonitoringComponent):
    """Interface for error handling functionality."""
    
    @abstractmethod
    def handle_error(self, error: Exception, error_type: str, context: Dict[str, Any]) -> None:
        """Handle an error.
        
        Args:
            error: Exception that occurred
            error_type: Type of error
            context: Additional context about error
        """
        pass
        
    @abstractmethod
    def get_error_stats(self) -> Dict[str, int]:
        """Get error statistics.
        
        Returns:
            Dictionary of error counts by type
        """
        pass
