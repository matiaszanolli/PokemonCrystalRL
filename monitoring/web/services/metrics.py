"""Metrics processing service.

This service handles collection, aggregation, and management of all metrics:
- Training metrics (episodes, steps, rewards)
- Game metrics (Pokemon, badges, money)
- System metrics (CPU, memory, network)
- Historical data and charts
"""

import threading
import time
import logging
import json
from typing import Dict, Any, Optional, List, Set
from dataclasses import dataclass, asdict, field
from collections import deque
from datetime import datetime

from monitoring.components.metrics import MetricsCollector


@dataclass
class MetricHistory:
    """Metric history tracking."""
    values: deque = field(default_factory=lambda: deque(maxlen=1000))
    timestamps: deque = field(default_factory=lambda: deque(maxlen=1000))
    
    def add(self, value: float, timestamp: Optional[float] = None) -> None:
        """Add value to history."""
        self.values.append(value)
        self.timestamps.append(timestamp or time.time())
    
    def get_since(self, since_time: float) -> List[Dict[str, Any]]:
        """Get values since given timestamp."""
        if not self.values:
            return []
            
        result = []
        for value, ts in zip(self.values, self.timestamps):
            if ts >= since_time:
                result.append({
                    'value': value,
                    'timestamp': ts
                })
        return result
    
    def clear(self) -> None:
        """Clear history."""
        self.values.clear()
        self.timestamps.clear()


@dataclass
class MetricsConfig:
    """Metrics service configuration."""
    history_size: int = 1000  # Maximum history entries per metric
    update_interval: float = 1.0  # Metrics update interval
    retention_hours: float = 24.0  # Data retention period


class MetricsService:
    """Handles metrics processing and aggregation.
    
    Features:
    - Training metrics collection and history
    - Game state metrics tracking
    - System resource monitoring
    - Historical data management
    - Chart data preparation
    """
    
    def __init__(self, config: Optional[MetricsConfig] = None):
        """Initialize metrics service.
        
        Args:
            config: Metrics configuration
        """
        self.config = config or MetricsConfig()
        
        # Component references
        self._collector: Optional[MetricsCollector] = None
        
        # Metric storage
        self._metrics: Dict[str, Any] = {}
        self._history: Dict[str, MetricHistory] = {}
        
        # Active metrics tracking
        self._active_metrics: Set[str] = set()
        self._total_recorded = 0
        
        # State
        self._running = False
        self._lock = threading.RLock()
        self._last_update = 0.0
        
        # Initialize metric histories
        self._init_histories()
        
        # Logging
        self.logger = logging.getLogger(__name__)
    
    def _init_histories(self) -> None:
        """Initialize metric history trackers."""
        # Training metrics
        self._history.update({
            'reward': MetricHistory(),
            'steps': MetricHistory(),
            'episode_length': MetricHistory(),
            'total_reward': MetricHistory()
        })
        
        # Game metrics
        self._history.update({
            'pokemon_count': MetricHistory(),
            'badge_count': MetricHistory(),
            'money': MetricHistory()
        })
        
        # Progress metrics
        self._history.update({
            'experience': MetricHistory(),
            'exploration': MetricHistory()
        })
        
        # Resource metrics
        self._history.update({
            'cpu_percent': MetricHistory(),
            'memory_usage': MetricHistory(),
            'network_bytes_sec': MetricHistory()
        })
    
    def set_metrics_collector(self, collector: MetricsCollector) -> None:
        """Set metrics collector component.
        
        Args:
            collector: Metrics collector component
        """
        self._collector = collector
    
    def update_training_metrics(self, metrics: Dict[str, Any]) -> None:
        """Update training-related metrics.
        
        Args:
            metrics: Training metrics dictionary
        """
        with self._lock:
            # Update current values
            self._metrics.update(metrics)
            
            # Track active metrics
            self._active_metrics.update(metrics.keys())
            self._total_recorded += len(metrics)
            
            # Update histories
            timestamp = time.time()
            
            # Episode metrics
            if 'episode' in metrics:
                if 'total_reward' in metrics:
                    self._history['reward'].add(
                        metrics['total_reward'],
                        timestamp
                    )
                if 'total_steps' in metrics:
                    self._history['steps'].add(
                        metrics['total_steps'],
                        timestamp
                    )
            
            # Game progress
            if 'experience' in metrics:
                self._history['experience'].add(
                    metrics['experience'],
                    timestamp
                )
            if 'exploration' in metrics:
                self._history['exploration'].add(
                    metrics['exploration'],
                    timestamp
                )
    
    def update_game_metrics(self, metrics: Dict[str, Any]) -> None:
        """Update game state metrics.
        
        Args:
            metrics: Game metrics dictionary
        """
        with self._lock:
            # Update current values
            self._metrics.update(metrics)
            
            # Track active metrics
            self._active_metrics.update(metrics.keys())
            self._total_recorded += len(metrics)
            
            # Update histories
            timestamp = time.time()
            
            if 'party_count' in metrics:
                self._history['pokemon_count'].add(
                    metrics['party_count'],
                    timestamp
                )
            if 'badges_total' in metrics:
                self._history['badge_count'].add(
                    metrics['badges_total'],
                    timestamp
                )
            if 'money' in metrics:
                self._history['money'].add(
                    metrics['money'],
                    timestamp
                )
    
    def update_resource_metrics(self, metrics: Dict[str, Any]) -> None:
        """Update system resource metrics.
        
        Args:
            metrics: Resource metrics dictionary
        """
        with self._lock:
            # Update current values
            self._metrics.update(metrics)
            
            # Track active metrics
            self._active_metrics.update(metrics.keys())
            self._total_recorded += len(metrics)
            
            # Update histories
            timestamp = time.time()
            
            if 'cpu_percent' in metrics:
                self._history['cpu_percent'].add(
                    metrics['cpu_percent'],
                    timestamp
                )
            if 'memory_usage_mb' in metrics:
                self._history['memory_usage'].add(
                    metrics['memory_usage_mb'],
                    timestamp
                )
            if 'network_bytes_sec' in metrics:
                self._history['network_bytes_sec'].add(
                    metrics['network_bytes_sec'],
                    timestamp
                )
    
    def get_metrics(self, names: Optional[List[str]] = None,
                   since: Optional[float] = None) -> Dict[str, Any]:
        """Get current metrics.
        
        Args:
            names: Optional list of metric names to retrieve
            since: Optional timestamp to get history since
            
        Returns:
            Dictionary of metrics and their values
        """
        with self._lock:
            # Get all current metrics
            metrics = self._metrics.copy()
            
            # Always include history data
            for name, history in self._history.items():
                if names is None or name in names:
                    metrics[f"{name}_history"] = [
                        {
                            'timestamp': ts,
                            'value': val
                        }
                        for ts, val in zip(history.timestamps, history.values)
                    ]
                    if since is not None:
                        metrics[f"{name}_history"] = [
                            item for item in metrics[f"{name}_history"]
                            if item['timestamp'] >= since
                        ]
            
            # Filter by names if specified
            if names:
                return {k: v for k, v in metrics.items() if k in names}
            return metrics
    
    def get_chart_data(self) -> Dict[str, Any]:
        """Get formatted data for charts.
        
        Returns:
            Dictionary with chart datasets
        """
        with self._lock:
            chart_data = {}
            
            # Reward history with timestamps
            chart_data['reward_history'] = [
                {
                    'timestamp': ts,
                    'value': val
                }
                for ts, val in zip(
                    self._history['reward'].timestamps,
                    self._history['reward'].values
                )
            ]
            
            # Progress metrics
            chart_data['progress'] = [
                {
                    'timestamp': ts,
                    'experience': exp,
                    'exploration': expl
                }
                for ts, exp, expl in zip(
                    self._history['experience'].timestamps,
                    self._history['experience'].values,
                    self._history['exploration'].values
                )
            ]
            
            # Resource metrics
            chart_data['resources'] = [
                {
                    'timestamp': ts,
                    'cpu': cpu,
                    'memory': mem,
                    'network': net
                }
                for ts, cpu, mem, net in zip(
                    self._history['cpu_percent'].timestamps,
                    self._history['cpu_percent'].values,
                    self._history['memory_usage'].values,
                    self._history['network_bytes_sec'].values
                )
            ]
            
            return chart_data
    
    def get_status(self) -> Dict[str, Any]:
        """Get service status.
        
        Returns:
            Dictionary with status information
        """
        return {
            'running': self._running,
            'active_metrics': len(self._active_metrics),
            'total_recorded': self._total_recorded,
            'last_update': self._last_update
        }
    
    def clear(self) -> None:
        """Clear all metrics and histories."""
        with self._lock:
            self._metrics.clear()
            self._active_metrics.clear()
            self._total_recorded = 0
            for history in self._history.values():
                history.clear()
            self._last_update = 0.0
