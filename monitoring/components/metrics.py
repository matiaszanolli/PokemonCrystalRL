"""
Metrics Collection Component

This module provides the metrics collection and analysis functionality for
monitoring training progress, system performance, and game state.
"""

import time
import collections
from typing import Dict, List, Any, Optional, Tuple, Deque
from dataclasses import dataclass
import threading
import json
import os
import numpy as np

from ..base import MonitorComponent, MetricDefinition, ComponentError
from ..base import DataSubscriber

@dataclass
class MetricValue:
    """Single metric value with metadata."""
    value: float
    timestamp: float
    metadata: Optional[Dict[str, Any]] = None

class MetricsCollector(MonitorComponent, DataSubscriber):
    """Training metrics collection and analysis.
    
    This component collects various metrics about training progress,
    system performance, and game state, with efficient storage and
    aggregation capabilities.
    """
    
    def __init__(self, retention_hours: float = 24.0,
                 storage_path: str = "data/metrics"):
        """Initialize metrics collector.
        
        Args:
            retention_hours: How long to retain metrics
            storage_path: Where to store metric data
        """
        self.retention_hours = retention_hours
        self.storage_path = storage_path
        
        # Thread management
        self._lock = threading.RLock()
        self._running = False
        self._cleanup_thread: Optional[threading.Thread] = None
        
        # Metrics storage
        self.metrics: Dict[str, Deque[MetricValue]] = {}
        self.definitions: Dict[str, MetricDefinition] = {}
        self._latest_values: Dict[str, float] = {}
        
        # Performance tracking
        self.metrics_recorded = 0
        self.cleanup_runs = 0
        self._last_cleanup = 0.0
        self._cleanup_times: List[float] = []
        
        # Ensure storage directory exists
        os.makedirs(storage_path, exist_ok=True)
    
    def register_metric(self, definition: MetricDefinition) -> bool:
        """Register a new metric type.
        
        Args:
            definition: Metric definition
            
        Returns:
            bool: True if registered successfully
        """
        with self._lock:
            if definition.name in self.definitions:
                return False
                
            self.definitions[definition.name] = definition
            self.metrics[definition.name] = collections.deque(maxlen=10000)
            return True
    
    def record_metric(self, name: str, value: float,
                     metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Record a metric value.
        
        Args:
            name: Metric name
            value: Metric value
            metadata: Optional metadata
            
        Returns:
            bool: True if recorded successfully
        """
        with self._lock:
            if name not in self.definitions:
                return False
                
            metric = MetricValue(value=value,
                               timestamp=time.time(),
                               metadata=metadata)
            
            self.metrics[name].append(metric)
            self._latest_values[name] = value
            self.metrics_recorded += 1
            
            return True
    
    def get_metrics(self, names: Optional[List[str]] = None,
                   since: Optional[float] = None) -> Dict[str, List[Tuple[float, float]]]:
        """Get metric values.
        
        Args:
            names: Which metrics to get (all if None)
            since: Only get metrics after this timestamp
            
        Returns:
            Dict mapping metric names to [(timestamp, value), ...]
        """
        with self._lock:
            # Default to all metrics
            if names is None:
                names = list(self.metrics.keys())
                
            result = {}
            now = time.time()
            
            for name in names:
                if name not in self.metrics:
                    continue
                    
                # Filter by timestamp if needed
                if since is not None:
                    values = [
                        (m.timestamp, m.value)
                        for m in self.metrics[name]
                        if m.timestamp >= since
                    ]
                else:
                    values = [
                        (m.timestamp, m.value)
                        for m in self.metrics[name]
                    ]
                    
                result[name] = values
                
            return result
    
    def get_latest(self, name: str) -> Optional[float]:
        """Get latest value for a metric.
        
        Args:
            name: Metric name
            
        Returns:
            Latest value or None if not found
        """
        return self._latest_values.get(name)
    
    def get_statistics(self, name: str) -> Optional[Dict[str, float]]:
        """Get statistics for a metric.
        
        Args:
            name: Metric name
            
        Returns:
            Dict with statistics or None if metric not found
        """
        with self._lock:
            if name not in self.metrics or not self.metrics[name]:
                return None
                
            values = [m.value for m in self.metrics[name]]
            
            return {
                "count": len(values),
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "last": values[-1]
            }
    
    def start(self) -> bool:
        """Start metrics collection.
        
        Returns:
            bool: True if started successfully
        """
        with self._lock:
            if self._running:
                return True
                
            self._running = True
            
            # Start cleanup thread
            self._cleanup_thread = threading.Thread(
                target=self._cleanup_loop,
                daemon=True,
                name="MetricsCleanup"
            )
            self._cleanup_thread.start()
            
            return True
    
    def stop(self) -> bool:
        """Stop metrics collection.
        
        Returns:
            bool: True if stopped successfully
        """
        with self._lock:
            self._running = False
            
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=5.0)
            return not self._cleanup_thread.is_alive()
            
        return True
    
    def get_status(self) -> Dict[str, Any]:
        """Get component status.
        
        Returns:
            Dict with status information
        """
        with self._lock:
            return {
                "running": self._running,
                "metrics_recorded": self.metrics_recorded,
                "cleanup_runs": self.cleanup_runs,
                "metric_counts": {
                    name: len(values)
                    for name, values in self.metrics.items()
                },
                "latest_values": self._latest_values.copy(),
                "avg_cleanup_time": (
                    sum(self._cleanup_times) / len(self._cleanup_times)
                    if self._cleanup_times else 0
                )
            }
    
    def save_metrics(self) -> bool:
        """Save metrics to storage.
        
        Returns:
            bool: True if saved successfully
        """
        try:
            filepath = os.path.join(self.storage_path, "metrics.json")
            
            with self._lock:
                # Convert metrics to serializable format
                data = {
                    "metrics": {
                        name: [
                            {
                                "value": m.value,
                                "timestamp": m.timestamp,
                                "metadata": m.metadata
                            }
                            for m in values
                        ]
                        for name, values in self.metrics.items()
                    },
                    "definitions": {
                        name: {
                            "name": d.name,
                            "type": d.type,
                            "description": d.description,
                            "unit": d.unit,
                            "aggregation": d.aggregation
                        }
                        for name, d in self.definitions.items()
                    }
                }
                
                # Save atomic by writing to temp file first
                temp_path = filepath + ".tmp"
                with open(temp_path, "w") as f:
                    json.dump(data, f)
                os.replace(temp_path, filepath)
                
                return True
                
        except Exception as e:
            raise ComponentError(f"Failed to save metrics: {e}")
    
    def load_metrics(self) -> bool:
        """Load metrics from storage.
        
        Returns:
            bool: True if loaded successfully
        """
        try:
            filepath = os.path.join(self.storage_path, "metrics.json")
            
            if not os.path.exists(filepath):
                return False
                
            with open(filepath) as f:
                data = json.load(f)
                
            with self._lock:
                # Load definitions first
                for name, d in data["definitions"].items():
                    self.definitions[name] = MetricDefinition(**d)
                    
                # Load metrics
                for name, values in data["metrics"].items():
                    self.metrics[name] = collections.deque(
                        [MetricValue(**v) for v in values],
                        maxlen=10000
                    )
                    if values:
                        self._latest_values[name] = values[-1]["value"]
                        
                return True
                
        except Exception as e:
            raise ComponentError(f"Failed to load metrics: {e}")
    
    def _cleanup_loop(self) -> None:
        """Background cleanup thread."""
        while self._running:
            try:
                cleanup_start = time.time()
                
                # Only run cleanup every 5 minutes
                if cleanup_start - self._last_cleanup < 300:
                    time.sleep(10)
                    continue
                    
                self._cleanup_metrics()
                
                # Track cleanup performance
                cleanup_time = time.time() - cleanup_start
                self._cleanup_times.append(cleanup_time)
                if len(self._cleanup_times) > 100:
                    self._cleanup_times.pop(0)
                    
                self._last_cleanup = cleanup_start
                self.cleanup_runs += 1
                
                # Save metrics periodically
                if self.cleanup_runs % 12 == 0:  # Every hour
                    self.save_metrics()
                    
            except Exception:
                time.sleep(60)  # Back off on errors
    
    def _cleanup_metrics(self) -> None:
        """Remove old metrics based on retention policy."""
        cutoff = time.time() - (self.retention_hours * 3600)
        
        with self._lock:
            for name in self.metrics:
                # Keep only recent metrics
                self.metrics[name] = collections.deque(
                    [m for m in self.metrics[name] if m.timestamp >= cutoff],
                    maxlen=10000
                )
    
    def subscribe(self, topic: str) -> bool:
        """Subscribe to a metric topic.
        
        Args:
            topic: Metric topic to subscribe to
            
        Returns:
            bool: True if subscribed successfully
        """
        # This is a minimal implementation
        # The DataBus handles actual subscriptions
        return True
    
    def handle_data(self, topic: str, data: Any) -> None:
        """Handle received metric data.
        
        Args:
            topic: Metric topic
            data: Metric data
        """
        if isinstance(data, tuple) and len(data) >= 2:
            self.record_metric(topic, data[0], data[1] if len(data) > 2 else None)
