"""
Trainer monitor bridge module for Pokemon Crystal RL.

This module serves as a bridge between the training system and monitoring components,
handling:
- Training metric collection and streaming
- Model state monitoring
- Training event processing
- Performance tracking
- Resource utilization monitoring
- Training progress visualization
"""

import time
import threading
from typing import Dict, Any, Optional, List, Callable, Union
from dataclasses import dataclass
import logging
from datetime import datetime
import json
from queue import Queue
from pathlib import Path
import numpy as np
from collections import deque

from .data_bus import DataType, get_data_bus
from .error_handler import ErrorHandler, ErrorCategory, ErrorSeverity
from .stats_collector import StatsCollector
from .text_logger import TextLogger


@dataclass
class TrainingState:
    """Current state of the training process."""
    episode: int
    total_steps: int
    current_reward: float
    episode_rewards: List[float]
    latest_loss: Optional[float] = None
    latest_policy_loss: Optional[float] = None
    latest_value_loss: Optional[float] = None
    latest_entropy: Optional[float] = None
    checkpoint_path: Optional[str] = None
    learning_rate: Optional[float] = None
    epsilon: Optional[float] = None
    last_save_time: Optional[float] = None
    is_training: bool = True


class TrainingMetrics:
    """Handler for training-specific metrics."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.metrics: Dict[str, deque] = {
            'episode_rewards': deque(maxlen=window_size),
            'episode_lengths': deque(maxlen=window_size),
            'losses': deque(maxlen=window_size),
            'policy_losses': deque(maxlen=window_size),
            'value_losses': deque(maxlen=window_size),
            'entropies': deque(maxlen=window_size),
            'grad_norms': deque(maxlen=window_size)
        }
    
    def update(self, metric: str, value: float) -> None:
        """Update a metric with a new value."""
        if metric in self.metrics:
            self.metrics[metric].append(value)
    
    def get_stats(self, metric: str) -> Dict[str, float]:
        """Get statistics for a specific metric."""
        if metric not in self.metrics or not self.metrics[metric]:
            return {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0}
        
        values = np.array(self.metrics[metric])
        return {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values))
        }
    
    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all metrics."""
        return {
            metric: self.get_stats(metric)
            for metric in self.metrics
        }


class TrainerMonitorBridge:
    """Bridge between training system and monitoring components."""
    
    def __init__(self,
                 save_dir: str = "training_data",
                 metrics_update_interval: float = 1.0,
                 state_update_interval: float = 5.0):
        
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.data_bus = get_data_bus()
        self.error_handler = ErrorHandler()
        self.stats_collector = StatsCollector()
        self.logger = TextLogger()
        
        # Training state
        self.training_state = TrainingState(
            episode=0,
            total_steps=0,
            current_reward=0.0,
            episode_rewards=[]
        )
        self._state_lock = threading.Lock()
        
        # Metrics tracking
        self.metrics = TrainingMetrics()
        self.metrics_queue = Queue()
        self.state_queue = Queue()
        
        # Update intervals
        self.metrics_update_interval = metrics_update_interval
        self.state_update_interval = state_update_interval
        
        # Processing state
        self.is_processing = False
        self._metrics_thread: Optional[threading.Thread] = None
        self._state_thread: Optional[threading.Thread] = None
        
        # Callbacks
        self.on_episode_end: Optional[Callable[[Dict[str, Any]], None]] = None
        self.on_checkpoint_save: Optional[Callable[[str, Dict[str, Any]], None]] = None
        
        # Setup logging
        self.logger.info("🌉 Trainer monitor bridge initialized")
        
        # Start processing threads
        self.start_processing()
    
    def start_processing(self) -> None:
        """Start the metric and state processing threads."""
        if self.is_processing:
            return
        
        self.is_processing = True
        
        # Start metrics processing thread
        self._metrics_thread = threading.Thread(
            target=self._metrics_loop,
            daemon=True
        )
        self._metrics_thread.start()
        
        # Start state processing thread
        self._state_thread = threading.Thread(
            target=self._state_loop,
            daemon=True
        )
        self._state_thread.start()
        
        self.logger.info("🔄 Processing threads started")
    
    def stop_processing(self) -> None:
        """Stop the processing threads."""
        self.is_processing = False
        
        if self._metrics_thread and self._metrics_thread.is_alive():
            self._metrics_thread.join(timeout=5.0)
        
        if self._state_thread and self._state_thread.is_alive():
            self._state_thread.join(timeout=5.0)
        
        self.logger.info("⏹️ Processing threads stopped")
    
    def update_training_state(self, **kwargs) -> None:
        """Update the current training state."""
        with self._state_lock:
            for key, value in kwargs.items():
                if hasattr(self.training_state, key):
                    setattr(self.training_state, key, value)
            
            # Queue state update
            self.state_queue.put(self.get_training_state())
    
    def record_metric(self, name: str, value: float,
                     metadata: Optional[Dict[str, Any]] = None) -> None:
        """Record a training metric."""
        try:
            # Update local metrics
            self.metrics.update(name, value)
            
            # Queue for processing
            self.metrics_queue.put({
                'name': name,
                'value': value,
                'metadata': metadata or {},
                'timestamp': time.time()
            })
            
            # Update stats collector
            self.stats_collector.record_metric(name, value)
            
        except Exception as e:
            self.error_handler.handle_error(
                e,
                severity=ErrorSeverity.ERROR,
                category=ErrorCategory.TRAINING,
                component="trainer_monitor_bridge"
            )
    
    def record_episode_end(self, reward: float, steps: int,
                         metadata: Optional[Dict[str, Any]] = None) -> None:
        """Record the end of a training episode."""
        try:
            # Update state
            with self._state_lock:
                self.training_state.episode += 1
                self.training_state.total_steps += steps
                self.training_state.episode_rewards.append(reward)
                
                # Keep only recent rewards
                if len(self.training_state.episode_rewards) > 100:
                    self.training_state.episode_rewards.pop(0)
            
            # Record metrics
            self.record_metric('episode_reward', reward)
            self.record_metric('episode_length', steps)
            
            # Log episode
            self.logger.training(
                f"Episode {self.training_state.episode} completed: "
                f"reward={reward:.2f}, steps={steps}"
            )
            
            # Notify stats collector
            self.stats_collector.record_episode_end(reward, steps)
            
            # Call episode end callback
            if self.on_episode_end:
                self.on_episode_end({
                    'episode': self.training_state.episode,
                    'reward': reward,
                    'steps': steps,
                    'total_steps': self.training_state.total_steps,
                    'metadata': metadata or {}
                })
            
        except Exception as e:
            self.error_handler.handle_error(
                e,
                severity=ErrorSeverity.ERROR,
                category=ErrorCategory.TRAINING,
                component="trainer_monitor_bridge"
            )
    
    def record_checkpoint(self, path: str,
                        metadata: Optional[Dict[str, Any]] = None) -> None:
        """Record a model checkpoint save."""
        try:
            # Update state
            with self._state_lock:
                self.training_state.checkpoint_path = path
                self.training_state.last_save_time = time.time()
            
            # Log checkpoint
            self.logger.training(f"Checkpoint saved: {path}")
            
            # Call checkpoint callback
            if self.on_checkpoint_save:
                self.on_checkpoint_save(path, metadata or {})
            
            # Publish checkpoint event
            if self.data_bus:
                self.data_bus.publish(
                    DataType.TRAINING_EVENT,
                    {
                        'event_type': 'checkpoint',
                        'path': path,
                        'timestamp': time.time(),
                        'metadata': metadata or {}
                    },
                    'trainer_monitor_bridge'
                )
            
        except Exception as e:
            self.error_handler.handle_error(
                e,
                severity=ErrorSeverity.ERROR,
                category=ErrorCategory.TRAINING,
                component="trainer_monitor_bridge"
            )
    
    def record_model_update(self, 
                          loss: Optional[float] = None,
                          policy_loss: Optional[float] = None,
                          value_loss: Optional[float] = None,
                          entropy: Optional[float] = None,
                          grad_norm: Optional[float] = None,
                          learning_rate: Optional[float] = None,
                          epsilon: Optional[float] = None) -> None:
        """Record model update metrics."""
        try:
            # Update state
            with self._state_lock:
                if loss is not None:
                    self.training_state.latest_loss = loss
                if policy_loss is not None:
                    self.training_state.latest_policy_loss = policy_loss
                if value_loss is not None:
                    self.training_state.latest_value_loss = value_loss
                if entropy is not None:
                    self.training_state.latest_entropy = entropy
                if learning_rate is not None:
                    self.training_state.learning_rate = learning_rate
                if epsilon is not None:
                    self.training_state.epsilon = epsilon
            
            # Record metrics
            metrics = {
                'loss': loss,
                'policy_loss': policy_loss,
                'value_loss': value_loss,
                'entropy': entropy,
                'grad_norm': grad_norm
            }
            
            for name, value in metrics.items():
                if value is not None:
                    self.record_metric(name, value)
            
            # Notify stats collector
            self.stats_collector.record_model_metrics(
                {k: v for k, v in metrics.items() if v is not None}
            )
            
        except Exception as e:
            self.error_handler.handle_error(
                e,
                severity=ErrorSeverity.ERROR,
                category=ErrorCategory.TRAINING,
                component="trainer_monitor_bridge"
            )
    
    def get_training_state(self) -> Dict[str, Any]:
        """Get the current training state."""
        with self._state_lock:
            return {
                'episode': self.training_state.episode,
                'total_steps': self.training_state.total_steps,
                'current_reward': self.training_state.current_reward,
                'recent_rewards': list(self.training_state.episode_rewards),
                'latest_loss': self.training_state.latest_loss,
                'latest_policy_loss': self.training_state.latest_policy_loss,
                'latest_value_loss': self.training_state.latest_value_loss,
                'latest_entropy': self.training_state.latest_entropy,
                'checkpoint_path': self.training_state.checkpoint_path,
                'learning_rate': self.training_state.learning_rate,
                'epsilon': self.training_state.epsilon,
                'last_save_time': self.training_state.last_save_time,
                'is_training': self.training_state.is_training,
                'metrics': self.metrics.get_all_stats()
            }
    
    def _metrics_loop(self) -> None:
        """Process metrics updates."""
        last_update = time.time()
        
        while self.is_processing:
            try:
                current_time = time.time()
                
                # Process all available metrics
                metrics_batch = []
                try:
                    while True:
                        metric = self.metrics_queue.get_nowait()
                        metrics_batch.append(metric)
                except:
                    pass
                
                # Publish metrics update if interval elapsed
                if metrics_batch and current_time - last_update >= self.metrics_update_interval:
                    if self.data_bus:
                        self.data_bus.publish(
                            DataType.TRAINING_METRICS,
                            {
                                'metrics': metrics_batch,
                                'timestamp': current_time
                            },
                            'trainer_monitor_bridge'
                        )
                    last_update = current_time
                
                time.sleep(0.1)  # Prevent tight loop
                
            except Exception as e:
                self.error_handler.handle_error(
                    e,
                    severity=ErrorSeverity.ERROR,
                    category=ErrorCategory.SYSTEM,
                    component="trainer_monitor_bridge"
                )
                time.sleep(1)  # Back off on error
    
    def _state_loop(self) -> None:
        """Process state updates."""
        last_update = time.time()
        
        while self.is_processing:
            try:
                current_time = time.time()
                
                # Process state updates
                latest_state = None
                try:
                    while True:
                        state = self.state_queue.get_nowait()
                        latest_state = state
                except:
                    pass
                
                # Publish state update if interval elapsed
                if latest_state and current_time - last_update >= self.state_update_interval:
                    if self.data_bus:
                        self.data_bus.publish(
                            DataType.TRAINING_STATE,
                            {
                                'state': latest_state,
                                'timestamp': current_time
                            },
                            'trainer_monitor_bridge'
                        )
                    last_update = current_time
                
                time.sleep(0.1)  # Prevent tight loop
                
            except Exception as e:
                self.error_handler.handle_error(
                    e,
                    severity=ErrorSeverity.ERROR,
                    category=ErrorCategory.SYSTEM,
                    component="trainer_monitor_bridge"
                )
                time.sleep(1)  # Back off on error
    
    def save_training_state(self, filepath: Optional[str] = None) -> None:
        """Save the current training state to a file."""
        try:
            if filepath is None:
                filepath = self.save_dir / f"training_state_{int(time.time())}.json"
            
            state_data = {
                'timestamp': datetime.now().isoformat(),
                'training_state': self.get_training_state(),
                'metrics': self.metrics.get_all_stats()
            }
            
            with open(filepath, 'w') as f:
                json.dump(state_data, f, indent=2)
            
            self.logger.info(f"💾 Training state saved to {filepath}")
            
        except Exception as e:
            self.error_handler.handle_error(
                e,
                severity=ErrorSeverity.ERROR,
                category=ErrorCategory.SYSTEM,
                component="trainer_monitor_bridge"
            )
    
    def register_callback(self, event_type: str, callback: Callable) -> None:
        """Register a callback for specific events."""
        if event_type == 'episode_end':
            self.on_episode_end = callback
        elif event_type == 'checkpoint_save':
            self.on_checkpoint_save = callback
    
    def __del__(self):
        """Cleanup on deletion."""
        self.stop_processing()
