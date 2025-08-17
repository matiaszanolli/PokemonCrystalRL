#!/usr/bin/env python3
"""
StatsCollector - Thread-safe real-time statistics aggregation

This module provides a robust stats collection system that captures training
metrics, game state changes, LLM decisions, and performance data in real-time.
"""

import time
import threading
import queue
import logging
import json
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, asdict
from collections import deque, defaultdict
import statistics
from enum import Enum

try:
    from .data_bus import get_data_bus, DataType, DataMessage
except ImportError:
    from data_bus import get_data_bus, DataType, DataMessage


class MetricType(Enum):
    """Types of metrics that can be collected"""
    COUNTER = "counter"          # Incrementing values (actions taken, errors)
    GAUGE = "gauge"              # Current values (speed, memory usage)
    HISTOGRAM = "histogram"      # Distribution of values (response times)
    TIMER = "timer"              # Duration measurements
    RATE = "rate"                # Rate of change over time


@dataclass
class MetricPoint:
    """A single metric data point"""
    timestamp: float
    value: float
    tags: Dict[str, str]


@dataclass
class AggregatedMetric:
    """Aggregated metric with statistics"""
    name: str
    metric_type: MetricType
    current_value: float
    count: int
    sum_value: float
    min_value: float
    max_value: float
    avg_value: float
    last_updated: float
    tags: Dict[str, str]


class StatsCollector:
    """
    Thread-safe real-time statistics aggregation system
    
    Features:
    - Non-blocking stats collection
    - Multiple metric types (counters, gauges, histograms, timers, rates)
    - Configurable aggregation windows
    - Historical data storage
    - Performance metrics calculation
    - Thread-safe operations
    """
    
    def __init__(self, 
                 max_history: int = 1000,
                 aggregation_window: float = 10.0,
                 cleanup_interval: float = 60.0,
                 enable_data_bus: bool = True):
        
        self.max_history = max_history
        self.aggregation_window = aggregation_window
        self.cleanup_interval = cleanup_interval
        self.enable_data_bus = enable_data_bus
        
        # Thread safety
        self._lock = threading.RLock()
        self._collection_thread: Optional[threading.Thread] = None
        self._collection_active = False
        
        # Metrics storage
        self._raw_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history))
        self._aggregated_metrics: Dict[str, AggregatedMetric] = {}
        self._metric_types: Dict[str, MetricType] = {}
        
        # Performance tracking
        self._collection_count = 0
        self._collection_errors = 0
        self._last_aggregation_time = 0.0
        
        # Rate calculation tracking
        self._rate_tracking: Dict[str, List[tuple]] = defaultdict(list)  # (timestamp, value)
        
        # Setup logging
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Data bus integration
        self.data_bus = get_data_bus() if enable_data_bus else None
        if self.data_bus:
            self.data_bus.register_component("stats_collector", {
                "type": "statistics",
                "max_history": max_history,
                "aggregation_window": aggregation_window
            })
            
            # Subscribe to relevant data types
            self.data_bus.subscribe(DataType.TRAINING_STATS, self._handle_training_stats, "stats_collector")
            self.data_bus.subscribe(DataType.ACTION_TAKEN, self._handle_action_taken, "stats_collector")
            self.data_bus.subscribe(DataType.LLM_DECISION, self._handle_llm_decision, "stats_collector")
            self.data_bus.subscribe(DataType.GAME_STATE, self._handle_game_state, "stats_collector")
            self.data_bus.subscribe(DataType.ERROR_EVENT, self._handle_error_event, "stats_collector")
        
        self.logger.info("📊 StatsCollector initialized")
    
    def start_collection(self) -> bool:
        """Start the stats collection process"""
        if self._collection_active:
            self.logger.warning("Stats collection already active")
            return True
        
        with self._lock:
            self._collection_active = True
            self._collection_thread = threading.Thread(
                target=self._collection_loop,
                daemon=True,
                name="StatsCollector"
            )
            self._collection_thread.start()
        
        self.logger.info("🚀 Stats collection started")
        return True
    
    def stop_collection(self) -> None:
        """Stop the stats collection process"""
        if not self._collection_active:
            return
        
        with self._lock:
            self._collection_active = False
        
        if self._collection_thread and self._collection_thread.is_alive():
            self._collection_thread.join(timeout=5.0)
            if self._collection_thread.is_alive():
                self.logger.warning("Stats collection thread did not shut down cleanly")
        
        self.logger.info("🛑 Stats collection stopped")
    
    def record_counter(self, name: str, value: float = 1.0, tags: Optional[Dict[str, str]] = None) -> None:
        """Record a counter metric (incrementing value)"""
        self._record_metric(name, MetricType.COUNTER, value, tags or {})
    
    def record_gauge(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record a gauge metric (current value)"""
        self._record_metric(name, MetricType.GAUGE, value, tags or {})
    
    def record_histogram(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record a histogram metric (distribution of values)"""
        self._record_metric(name, MetricType.HISTOGRAM, value, tags or {})
    
    def record_timer(self, name: str, duration: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record a timer metric (duration measurement)"""
        self._record_metric(name, MetricType.TIMER, duration, tags or {})
    
    def record_rate(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record a rate metric (rate of change over time)"""
        self._record_metric(name, MetricType.RATE, value, tags or {})
    
    def get_metric(self, name: str) -> Optional[AggregatedMetric]:
        """Get current aggregated metric"""
        with self._lock:
            return self._aggregated_metrics.get(name)
    
    def get_all_metrics(self) -> Dict[str, AggregatedMetric]:
        """Get all current aggregated metrics"""
        with self._lock:
            return self._aggregated_metrics.copy()
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics"""
        with self._lock:
            summary = {}
            
            for name, metric in self._aggregated_metrics.items():
                summary[name] = {
                    "type": metric.metric_type.value,
                    "current": metric.current_value,
                    "count": metric.count,
                    "avg": metric.avg_value,
                    "min": metric.min_value,
                    "max": metric.max_value,
                    "last_updated": metric.last_updated,
                    "tags": metric.tags
                }
            
            return summary
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for the collector itself"""
        with self._lock:
            return {
                "collection_count": self._collection_count,
                "collection_errors": self._collection_errors,
                "error_rate": self._collection_errors / max(self._collection_count, 1),
                "active_metrics": len(self._aggregated_metrics),
                "total_data_points": sum(len(deque_data) for deque_data in self._raw_metrics.values()),
                "last_aggregation": self._last_aggregation_time,
                "is_collecting": self._collection_active
            }
    
    def reset_metrics(self, pattern: Optional[str] = None) -> None:
        """Reset metrics (optionally matching a pattern)"""
        with self._lock:
            if pattern is None:
                self._raw_metrics.clear()
                self._aggregated_metrics.clear()
                self._metric_types.clear()
                self._rate_tracking.clear()
                self.logger.info("All metrics reset")
            else:
                # Reset metrics matching pattern
                to_remove = [name for name in self._aggregated_metrics.keys() if pattern in name]
                for name in to_remove:
                    del self._aggregated_metrics[name]
                    del self._raw_metrics[name]
                    del self._metric_types[name]
                    if name in self._rate_tracking:
                        del self._rate_tracking[name]
                self.logger.info(f"Reset {len(to_remove)} metrics matching '{pattern}'")
    
    def export_metrics(self, format: str = "json") -> str:
        """Export metrics in specified format"""
        with self._lock:
            if format == "json":
                data = {
                    "timestamp": time.time(),
                    "metrics": {name: asdict(metric) for name, metric in self._aggregated_metrics.items()},
                    "performance": self.get_performance_stats()
                }
                return json.dumps(data, indent=2)
            else:
                raise ValueError(f"Unsupported export format: {format}")
    
    def shutdown(self) -> None:
        """Clean shutdown of the stats collector"""
        self.logger.info("🛑 Shutting down StatsCollector")
        
        # Stop collection
        self.stop_collection()
        
        # Clear data
        with self._lock:
            self._raw_metrics.clear()
            self._aggregated_metrics.clear()
            self._metric_types.clear()
            self._rate_tracking.clear()
        
        # Notify data bus
        if self.data_bus:
            self.data_bus.publish(
                DataType.COMPONENT_STATUS,
                {"component": "stats_collector", "status": "shutdown"},
                "stats_collector"
            )
        
        self.logger.info("✅ StatsCollector shutdown complete")
    
    # Internal methods
    
    def _record_metric(self, name: str, metric_type: MetricType, value: float, tags: Dict[str, str]) -> None:
        """Internal method to record a metric"""
        current_time = time.time()
        
        try:
            with self._lock:
                # Store raw data point
                point = MetricPoint(current_time, value, tags)
                self._raw_metrics[name].append(point)
                
                # Track metric type
                self._metric_types[name] = metric_type
                
                # Handle rate metrics specially
                if metric_type == MetricType.RATE:
                    self._rate_tracking[name].append((current_time, value))
                    # Keep only recent data points for rate calculation
                    cutoff = current_time - self.aggregation_window
                    self._rate_tracking[name] = [(t, v) for t, v in self._rate_tracking[name] if t > cutoff]
                
                self._collection_count += 1
                
        except Exception as e:
            self._collection_errors += 1
            self.logger.error(f"Failed to record metric {name}: {e}")
    
    def _collection_loop(self) -> None:
        """Main collection loop running in a separate thread"""
        self.logger.info("🔄 Stats collection loop started")
        
        last_aggregation = time.time()
        
        while self._collection_active:
            try:
                current_time = time.time()
                
                # Aggregate metrics at regular intervals
                if (current_time - last_aggregation) >= self.aggregation_window:
                    self._aggregate_metrics()
                    last_aggregation = current_time
                    self._last_aggregation_time = current_time
                
                # Update heartbeat
                if self.data_bus:
                    self.data_bus.update_component_heartbeat("stats_collector")
                
                # Sleep briefly
                time.sleep(1.0)
                
            except Exception as e:
                self._collection_errors += 1
                self.logger.error(f"Error in collection loop: {e}")
                time.sleep(5.0)  # Back off on error
        
        self.logger.info("🔄 Stats collection loop ended")
    
    def _aggregate_metrics(self) -> None:
        """Aggregate raw metrics into summary statistics"""
        current_time = time.time()
        cutoff_time = current_time - self.aggregation_window
        
        with self._lock:
            for name, raw_data in self._raw_metrics.items():
                if not raw_data:
                    continue
                
                metric_type = self._metric_types.get(name, MetricType.GAUGE)
                
                # Filter recent data points
                recent_points = [p for p in raw_data if p.timestamp > cutoff_time]
                if not recent_points:
                    continue
                
                # Calculate statistics based on metric type
                values = [p.value for p in recent_points]
                
                if metric_type == MetricType.COUNTER:
                    # For counters, sum the increments
                    current_value = sum(values)
                elif metric_type == MetricType.RATE:
                    # For rates, calculate rate of change
                    if name in self._rate_tracking and len(self._rate_tracking[name]) >= 2:
                        rate_data = self._rate_tracking[name]
                        if len(rate_data) >= 2:
                            time_diff = rate_data[-1][0] - rate_data[0][0]
                            value_diff = rate_data[-1][1] - rate_data[0][1]
                            current_value = value_diff / time_diff if time_diff > 0 else 0.0
                        else:
                            current_value = 0.0
                    else:
                        current_value = 0.0
                else:
                    # For gauges, histograms, timers - use most recent value
                    current_value = values[-1]
                
                # Create aggregated metric
                aggregated = AggregatedMetric(
                    name=name,
                    metric_type=metric_type,
                    current_value=current_value,
                    count=len(values),
                    sum_value=sum(values),
                    min_value=min(values),
                    max_value=max(values),
                    avg_value=statistics.mean(values),
                    last_updated=current_time,
                    tags=recent_points[-1].tags  # Use tags from most recent point
                )
                
                self._aggregated_metrics[name] = aggregated
        
        # Publish aggregated metrics to data bus
        if self.data_bus:
            try:
                self.data_bus.publish(
                    DataType.TRAINING_STATS,
                    {
                        "metrics_count": len(self._aggregated_metrics),
                        "aggregation_time": current_time,
                        "collection_performance": self.get_performance_stats()
                    },
                    "stats_collector"
                )
            except Exception as e:
                self.logger.warning(f"Failed to publish stats to data bus: {e}")
    
    # Data bus event handlers
    
    def _handle_training_stats(self, message: DataMessage) -> None:
        """Handle training statistics from data bus"""
        data = message.data
        
        # Record various training metrics
        if "total_actions" in data:
            self.record_gauge("training.total_actions", float(data["total_actions"]))
        
        if "actions_per_second" in data:
            self.record_gauge("training.speed", float(data["actions_per_second"]))
        
        if "llm_calls" in data:
            self.record_gauge("training.llm_calls", float(data["llm_calls"]))
    
    def _handle_action_taken(self, message: DataMessage) -> None:
        """Handle action events from data bus"""
        data = message.data
        
        # Count actions
        self.record_counter("actions.total")
        
        if "action_id" in data:
            action_name = data.get("action_name", f"action_{data['action_id']}")
            self.record_counter(f"actions.{action_name}")
        
        if "execution_time" in data:
            self.record_timer("actions.execution_time", float(data["execution_time"]))
    
    def _handle_llm_decision(self, message: DataMessage) -> None:
        """Handle LLM decision events from data bus"""
        data = message.data
        
        # Count LLM decisions
        self.record_counter("llm.decisions")
        
        if "response_time" in data:
            self.record_timer("llm.response_time", float(data["response_time"]))
        
        if "success" in data:
            if data["success"]:
                self.record_counter("llm.success")
            else:
                self.record_counter("llm.errors")
    
    def _handle_game_state(self, message: DataMessage) -> None:
        """Handle game state changes from data bus"""
        data = message.data
        
        if "state" in data:
            state = data["state"]
            self.record_counter(f"game_state.{state}")
            
            # Track state transitions
            if "previous_state" in data and data["previous_state"]:
                transition = f"{data['previous_state']}_to_{state}"
                self.record_counter(f"transitions.{transition}")
    
    def _handle_error_event(self, message: DataMessage) -> None:
        """Handle error events from data bus"""
        data = message.data
        
        # Count errors
        self.record_counter("errors.total")
        
        if "component" in data:
            self.record_counter(f"errors.{data['component']}")
        
        if "severity" in data:
            self.record_counter(f"errors.{data['severity']}")


# Context manager for timing operations
class Timer:
    """Context manager for timing operations and recording to stats collector"""
    
    def __init__(self, stats_collector: StatsCollector, metric_name: str, tags: Optional[Dict[str, str]] = None):
        self.stats_collector = stats_collector
        self.metric_name = metric_name
        self.tags = tags or {}
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = time.time() - self.start_time
            self.stats_collector.record_timer(self.metric_name, duration, self.tags)
