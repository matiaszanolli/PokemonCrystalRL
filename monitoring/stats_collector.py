"""
Stats collector module for Pokemon Crystal RL.

This module handles collection, aggregation, and tracking of various metrics including:
- Training statistics (rewards, episodes, actions)
- System performance (memory usage, CPU load)
- Model metrics (loss values, gradients)
- Game state statistics (in-game progress, status)
"""

import time
import psutil
import numpy as np
from typing import Dict, List, Any, Optional, Union
from collections import deque
import threading
import logging
from datetime import datetime
import json

from .data_bus import DataType, get_data_bus


class MetricBuffer:
    """Thread-safe circular buffer for storing metric values."""
    
    def __init__(self, max_size: int = 1000):
        self.buffer = deque(maxlen=max_size)
        self._lock = threading.Lock()
        
    def add(self, value: Union[float, int]) -> None:
        """Add a value to the buffer."""
        with self._lock:
            self.buffer.append(value)
    
    def get_stats(self) -> Dict[str, float]:
        """Calculate statistics from buffered values."""
        with self._lock:
            if not self.buffer:
                return {
                    'mean': 0.0,
                    'std': 0.0,
                    'min': 0.0,
                    'max': 0.0,
                    'count': 0
                }
            
            values = np.array(self.buffer)
            return {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'count': len(values)
            }
    
    def clear(self) -> None:
        """Clear all values from the buffer."""
        with self._lock:
            self.buffer.clear()


class StatsCollector:
    """Collects and aggregates various training and system metrics."""
    
    def __init__(self, metrics_history_size: int = 1000,
                 update_interval: float = 1.0):
        self.metrics_history_size = metrics_history_size
        self.update_interval = update_interval
        
        # Initialize metric buffers
        self.metric_buffers: Dict[str, MetricBuffer] = {
            # Training metrics
            'episode_reward': MetricBuffer(metrics_history_size),
            'episode_length': MetricBuffer(metrics_history_size),
            'step_reward': MetricBuffer(metrics_history_size),
            
            # Model metrics
            'loss': MetricBuffer(metrics_history_size),
            'policy_loss': MetricBuffer(metrics_history_size),
            'value_loss': MetricBuffer(metrics_history_size),
            'entropy': MetricBuffer(metrics_history_size),
            'grad_norm': MetricBuffer(metrics_history_size),
            
            # Performance metrics
            'fps': MetricBuffer(metrics_history_size),
            'step_time': MetricBuffer(metrics_history_size),
            'inference_time': MetricBuffer(metrics_history_size),
        }
        
        # Episode tracking
        self.current_episode = 0
        self.total_steps = 0
        self.episode_start_time = time.time()
        
        # System metrics
        self.process = psutil.Process()
        
        # Collection state
        self.is_collecting = False
        self.collector_thread: Optional[threading.Thread] = None
        self._collection_lock = threading.Lock()
        
        # Data bus connection
        self.data_bus = get_data_bus()
        
        # Logging setup
        self.logger = logging.getLogger(__name__)
        
        self.logger.info("📊 Stats collector initialized")
    
    def start_collecting(self) -> None:
        """Start the metrics collection process."""
        with self._collection_lock:
            if self.is_collecting:
                return
            
            self.is_collecting = True
            self.collector_thread = threading.Thread(
                target=self._collection_loop,
                daemon=True
            )
            self.collector_thread.start()
            
            self.logger.info("📈 Stats collection started")
    
    def stop_collecting(self) -> None:
        """Stop the metrics collection process."""
        with self._collection_lock:
            self.is_collecting = False
            if self.collector_thread and self.collector_thread.is_alive():
                self.collector_thread.join(timeout=5.0)
            
            self.logger.info("⏹️ Stats collection stopped")
    
    def record_metric(self, name: str, value: Union[float, int]) -> None:
        """Record a single metric value."""
        if name in self.metric_buffers:
            self.metric_buffers[name].add(value)
            
            # Publish individual metrics for real-time monitoring
            if self.data_bus:
                self.data_bus.publish(
                    DataType.TRAINING_METRICS,
                    {
                        'metric': name,
                        'value': value,
                        'timestamp': time.time()
                    },
                    'stats_collector'
                )
    
    def record_episode_end(self, total_reward: float, episode_steps: int) -> None:
        """Record metrics for a completed episode."""
        self.current_episode += 1
        self.total_steps += episode_steps
        
        # Record episode metrics
        self.record_metric('episode_reward', total_reward)
        self.record_metric('episode_length', episode_steps)
        
        # Calculate episode timing
        duration = time.time() - self.episode_start_time
        fps = episode_steps / duration if duration > 0 else 0
        self.record_metric('fps', fps)
        
        # Reset episode timer
        self.episode_start_time = time.time()
        
        # Publish episode summary
        if self.data_bus:
            self.data_bus.publish(
                DataType.EPISODE_SUMMARY,
                {
                    'episode': self.current_episode,
                    'total_reward': total_reward,
                    'steps': episode_steps,
                    'fps': fps,
                    'total_steps': self.total_steps,
                    'duration': duration,
                    'timestamp': time.time()
                },
                'stats_collector'
            )
    
    def record_model_metrics(self, metrics: Dict[str, float]) -> None:
        """Record model-related metrics (loss, gradients, etc.)."""
        for name, value in metrics.items():
            if name in self.metric_buffers:
                self.record_metric(name, value)
    
    def get_all_metrics(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all tracked metrics."""
        return {
            name: buffer.get_stats()
            for name, buffer in self.metric_buffers.items()
        }
    
    def get_system_metrics(self) -> Dict[str, float]:
        """Get current system performance metrics."""
        try:
            cpu_percent = self.process.cpu_percent()
            memory_info = self.process.memory_info()
            
            return {
                'cpu_percent': cpu_percent,
                'memory_rss': memory_info.rss / (1024 * 1024),  # MB
                'memory_vms': memory_info.vms / (1024 * 1024),  # MB
                'threads': self.process.num_threads(),
                'system_cpu': psutil.cpu_percent(),
                'system_memory': psutil.virtual_memory().percent
            }
        except Exception as e:
            self.logger.error(f"⚠️ Error getting system metrics: {e}")
            return {}
    
    def clear_metrics(self) -> None:
        """Clear all collected metrics."""
        for buffer in self.metric_buffers.values():
            buffer.clear()
        
        self.current_episode = 0
        self.total_steps = 0
        self.episode_start_time = time.time()
        
        self.logger.info("🧹 Metrics cleared")
    
    def _collection_loop(self) -> None:
        """Main metrics collection loop."""
        last_update = time.time()
        
        while self.is_collecting:
            try:
                current_time = time.time()
                if current_time - last_update >= self.update_interval:
                    # Get current metrics
                    system_metrics = self.get_system_metrics()
                    all_metrics = self.get_all_metrics()
                    
                    # Prepare complete metrics package
                    metrics_package = {
                        'timestamp': datetime.now().isoformat(),
                        'system': system_metrics,
                        'training': all_metrics,
                        'metadata': {
                            'total_episodes': self.current_episode,
                            'total_steps': self.total_steps
                        }
                    }
                    
                    # Publish to data bus
                    if self.data_bus:
                        self.data_bus.publish(
                            DataType.SYSTEM_INFO,
                            metrics_package,
                            'stats_collector'
                        )
                    
                    last_update = current_time
                
                time.sleep(0.1)  # Prevent tight loop
                
            except Exception as e:
                self.logger.error(f"❌ Metrics collection error: {e}")
                time.sleep(1)  # Avoid rapid retries on error
    
    def save_metrics(self, filepath: str) -> None:
        """Save current metrics to a JSON file."""
        try:
            metrics_data = {
                'timestamp': datetime.now().isoformat(),
                'metrics': self.get_all_metrics(),
                'system': self.get_system_metrics(),
                'metadata': {
                    'total_episodes': self.current_episode,
                    'total_steps': self.total_steps
                }
            }
            
            with open(filepath, 'w') as f:
                json.dump(metrics_data, f, indent=2)
            
            self.logger.info(f"💾 Metrics saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"⚠️ Error saving metrics: {e}")
    
    def __del__(self):
        """Cleanup on deletion."""
        self.stop_collecting()
