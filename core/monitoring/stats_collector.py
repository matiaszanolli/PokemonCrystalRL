"""
stats_collector.py - Training statistics collection and analysis

Collects various statistics about the training process including rewards,
actions taken, performance metrics, and system resource usage.
"""

import time
import threading
from typing import Dict, List, Any, Optional, Union
from collections import deque
import psutil
import numpy as np
from dataclasses import dataclass, field

from core.monitoring.data_bus import DataType, get_data_bus
from core.error_handler import SafeOperation, error_boundary


@dataclass
class TrainingStats:
    """Container for training statistics"""
    episode: int
    total_reward: float
    steps: int
    actions_taken: Dict[str, int]
    avg_step_time: float
    memory_usage_mb: float
    timestamp: float = field(default_factory=time.time)


class StatsCollector:
    """Collects and analyzes training statistics"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.data_bus = get_data_bus()
        
        # Stats storage
        self._stats_history: List[TrainingStats] = []
        self._reward_window = deque(maxlen=window_size)
        self._step_times = deque(maxlen=window_size)
        self._memory_usage = deque(maxlen=window_size)
        self._action_counts: Dict[str, int] = {}
        
        # Performance tracking
        self._episode_start_time: Optional[float] = None
        self._last_update_time = time.time()
        self._current_episode = 0
        
        # Subscribe to events
        self.data_bus.subscribe(DataType.TRAINING_STATS, self._handle_training_stats)
        self.data_bus.subscribe(DataType.ACTION_TAKEN, self._handle_action)
        
    def start_episode(self) -> None:
        """Mark the start of a training episode"""
        self._episode_start_time = time.time()
        self._current_episode += 1
        self._action_counts.clear()
        
    def end_episode(self, total_reward: float) -> None:
        """Record end of episode statistics"""
        if not self._episode_start_time:
            return
            
        # Calculate episode stats
        end_time = time.time()
        episode_time = end_time - self._episode_start_time
        steps = sum(self._action_counts.values())
        
        if steps > 0:
            avg_step_time = episode_time / steps
        else:
            avg_step_time = 0
            
        # Get memory usage
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        
        # Create stats object
        stats = TrainingStats(
            episode=self._current_episode,
            total_reward=total_reward,
            steps=steps,
            actions_taken=self._action_counts.copy(),
            avg_step_time=avg_step_time,
            memory_usage_mb=memory_mb
        )
        
        # Update tracking
        self._stats_history.append(stats)
        self._reward_window.append(total_reward)
        self._step_times.append(avg_step_time)
        self._memory_usage.append(memory_mb)
        
        # Publish stats
        with SafeOperation("StatsCollector", "publish_episode_stats"):
            self.data_bus.publish(
                DataType.TRAINING_STATS,
                self.get_training_stats(),
                component="StatsCollector"
            )
            
        self._episode_start_time = None
        
    @error_boundary("StatsCollector")
    def _handle_training_stats(self, stats: Dict[str, Any]) -> None:
        """Handle training statistics updates"""
        if 'reward' in stats:
            self._reward_window.append(stats['reward'])
            
        if 'step_time' in stats:
            self._step_times.append(stats['step_time'])
            
    @error_boundary("StatsCollector")
    def _handle_action(self, action_data: Dict[str, Any]) -> None:
        """Handle action taken events"""
        if 'action' in action_data:
            action = action_data['action']
            self._action_counts[action] = self._action_counts.get(action, 0) + 1
            
    def get_training_stats(self) -> Dict[str, Any]:
        """Get current training statistics
        
        Returns:
            Dictionary containing various training metrics
        """
        now = time.time()
        
        # Calculate stats
        if self._reward_window:
            avg_reward = np.mean(self._reward_window)
            reward_std = np.std(self._reward_window)
        else:
            avg_reward = 0
            reward_std = 0
            
        if self._step_times:
            avg_step_time = np.mean(self._step_times)
            max_step_time = np.max(self._step_times)
        else:
            avg_step_time = 0
            max_step_time = 0
            
        total_steps = sum(self._action_counts.values())
        
        # Build stats dict
        stats = {
            'episode': self._current_episode,
            'total_steps': total_steps,
            'rewards': {
                'mean': float(avg_reward),
                'std': float(reward_std),
                'min': float(min(self._reward_window)) if self._reward_window else 0,
                'max': float(max(self._reward_window)) if self._reward_window else 0
            },
            'step_times': {
                'mean': float(avg_step_time),
                'max': float(max_step_time)
            },
            'actions': self._action_counts.copy(),
            'memory_usage': {
                'current': float(self._memory_usage[-1]) if self._memory_usage else 0,
                'mean': float(np.mean(self._memory_usage)) if self._memory_usage else 0,
                'max': float(np.max(self._memory_usage)) if self._memory_usage else 0
            },
            'timestamp': now,
            'time_since_last_update': now - self._last_update_time
        }
        
        self._last_update_time = now
        return stats
        
    def reset(self) -> None:
        """Reset all statistics"""
        self._stats_history.clear()
        self._reward_window.clear()
        self._step_times.clear()
        self._memory_usage.clear()
        self._action_counts.clear()
        self._current_episode = 0
        self._episode_start_time = None
        self._last_update_time = time.time()
