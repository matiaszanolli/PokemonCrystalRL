"""
Statistics Tracker - Training metrics and analytics tracking

Extracted from LLMTrainer to handle training statistics, performance metrics,
and analytics data collection.
"""

import time
import json
import logging
import threading
from typing import Dict, Any, List, Optional, Deque
from dataclasses import dataclass, field
from collections import deque, defaultdict
from datetime import datetime
import numpy as np


@dataclass
class TrainingSession:
    """Data structure for tracking a training session."""
    start_time: float
    end_time: Optional[float] = None
    total_actions: int = 0
    total_reward: float = 0.0
    llm_calls: int = 0
    badges_earned: int = 0
    levels_gained: int = 0
    unique_locations: int = 0
    errors: int = 0


@dataclass 
class PerformanceMetrics:
    """Performance metrics tracking."""
    actions_per_second: float = 0.0
    llm_response_time: float = 0.0
    reward_rate: float = 0.0
    exploration_rate: float = 0.0
    success_rate: float = 0.0
    stuck_rate: float = 0.0


class StatisticsTracker:
    """Tracks comprehensive training statistics and performance metrics."""
    
    def __init__(self, session_name: str = None):
        self.logger = logging.getLogger("StatisticsTracker")
        self.session_name = session_name or f"session_{int(time.time())}"
        
        # Current session tracking
        self.current_session = TrainingSession(start_time=time.time())
        self.session_history: List[TrainingSession] = []
        
        # Real-time metrics
        self.performance = PerformanceMetrics()
        
        # Detailed tracking
        self.action_history: Deque[Dict[str, Any]] = deque(maxlen=1000)
        self.reward_history: Deque[float] = deque(maxlen=1000)
        self.llm_history: Deque[Dict[str, Any]] = deque(maxlen=100)
        
        # Performance windows for rolling averages
        self.performance_windows = {
            'actions': deque(maxlen=100),
            'rewards': deque(maxlen=100), 
            'llm_times': deque(maxlen=50),
            'explorations': deque(maxlen=100),
            'successes': deque(maxlen=100)
        }
        
        # State tracking
        self.game_states = {
            'current_map': 0,
            'current_position': (0, 0),
            'badges': 0,
            'level': 0,
            'party_count': 0,
            'money': 0,
            'visited_maps': set(),
            'visited_positions': set()
        }
        
        # Error and issue tracking
        self.error_counts = defaultdict(int)
        self.stuck_episodes = 0
        self.recovery_events = 0
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Web stats for monitoring
        self.web_stats_history = {
            'reward_history': [],
            'action_history': [],
            'progression': [],
            'performance': []
        }
        
        self.logger.info(f"Statistics tracker initialized for session: {self.session_name}")
    
    def record_action(self, 
                     action: int,
                     source: str,
                     game_state: Dict[str, Any] = None,
                     reward: float = 0.0,
                     metadata: Dict[str, Any] = None) -> None:
        """Record an action taken during training.
        
        Args:
            action: Action code (1-8)
            source: Source of action ('llm', 'fallback', 'rule-based')
            game_state: Current game state
            reward: Reward received
            metadata: Additional metadata
        """
        with self._lock:
            timestamp = time.time()
            
            # Record action
            action_data = {
                'action': action,
                'source': source,
                'timestamp': timestamp,
                'reward': reward,
                'game_state': game_state.copy() if game_state else {},
                'metadata': metadata or {}
            }
            
            self.action_history.append(action_data)
            
            # Update session counters
            self.current_session.total_actions += 1
            self.current_session.total_reward += reward
            
            if source == 'llm':
                self.current_session.llm_calls += 1
            
            # Update performance windows
            self.performance_windows['actions'].append(timestamp)
            self.performance_windows['rewards'].append(reward)
            
            # Update game state tracking
            if game_state:
                self._update_game_state_tracking(game_state)
            
            # Update web history
            self._update_web_history(action_data)
    
    def record_llm_decision(self,
                           action: int,
                           response_time: float,
                           reasoning: str = "",
                           confidence: float = 0.0,
                           success: bool = True) -> None:
        """Record LLM decision details.
        
        Args:
            action: Action chosen by LLM
            response_time: Time taken for LLM response
            reasoning: LLM reasoning text
            confidence: Confidence score
            success: Whether decision was successful
        """
        with self._lock:
            llm_data = {
                'action': action,
                'response_time': response_time,
                'reasoning': reasoning,
                'confidence': confidence,
                'success': success,
                'timestamp': time.time()
            }
            
            self.llm_history.append(llm_data)
            self.performance_windows['llm_times'].append(response_time)
            self.performance_windows['successes'].append(1 if success else 0)
            
            # Update performance metrics
            self._update_performance_metrics()
    
    def record_exploration(self, new_location: bool = False, new_map: bool = False) -> None:
        """Record exploration events.
        
        Args:
            new_location: Whether this is a new location
            new_map: Whether this is a new map
        """
        with self._lock:
            self.performance_windows['explorations'].append(1 if new_location else 0)
            
            if new_map:
                self.current_session.unique_locations += 1
    
    def record_progress(self, 
                       badges: int = None,
                       level: int = None,
                       money: int = None,
                       party_count: int = None) -> None:
        """Record game progress updates.
        
        Args:
            badges: Current badge count
            level: Current player level
            money: Current money amount
            party_count: Current party size
        """
        with self._lock:
            progress_made = False
            
            if badges is not None and badges > self.game_states['badges']:
                self.current_session.badges_earned += (badges - self.game_states['badges'])
                self.game_states['badges'] = badges
                progress_made = True
                self.logger.info(f"Badge progress: {badges} badges")
            
            if level is not None and level > self.game_states['level']:
                self.current_session.levels_gained += (level - self.game_states['level'])
                self.game_states['level'] = level
                progress_made = True
                self.logger.info(f"Level progress: Level {level}")
            
            if money is not None:
                self.game_states['money'] = money
            
            if party_count is not None:
                self.game_states['party_count'] = party_count
            
            if progress_made:
                self._update_progression_history()
    
    def record_error(self, error_type: str, message: str = "", severity: str = "error") -> None:
        """Record error or issue.
        
        Args:
            error_type: Type of error (e.g., 'llm_failure', 'pyboy_crash')
            message: Error message
            severity: Error severity level
        """
        with self._lock:
            self.error_counts[error_type] += 1
            self.current_session.errors += 1
            
            self.logger.warning(f"Error recorded: {error_type} - {message}")
    
    def record_stuck_episode(self, duration: int, location: tuple) -> None:
        """Record stuck detection episode.
        
        Args:
            duration: How long agent was stuck
            location: Location where stuck occurred
        """
        with self._lock:
            self.stuck_episodes += 1
            self.logger.info(f"Stuck episode: {duration} actions at {location}")
    
    def record_recovery_event(self, event_type: str) -> None:
        """Record recovery/intervention event.
        
        Args:
            event_type: Type of recovery ('failsafe', 'restart', 'manual')
        """
        with self._lock:
            self.recovery_events += 1
            self.logger.info(f"Recovery event: {event_type}")
    
    def _update_game_state_tracking(self, game_state: Dict[str, Any]) -> None:
        """Update internal game state tracking."""
        # Update position tracking
        player_map = game_state.get('player_map', 0)
        player_x = game_state.get('player_x', 0)
        player_y = game_state.get('player_y', 0)
        
        if player_map != self.game_states['current_map']:
            self.game_states['visited_maps'].add(player_map)
            self.game_states['current_map'] = player_map
        
        position = (player_x, player_y)
        if position != self.game_states['current_position']:
            self.game_states['visited_positions'].add((player_map, player_x, player_y))
            self.game_states['current_position'] = position
    
    def _update_performance_metrics(self) -> None:
        """Update rolling performance metrics."""
        current_time = time.time()
        
        # Actions per second
        recent_actions = [t for t in self.performance_windows['actions'] 
                         if current_time - t <= 60]  # Last minute
        if len(recent_actions) > 1:
            self.performance.actions_per_second = len(recent_actions) / 60.0
        
        # LLM response time
        if self.performance_windows['llm_times']:
            self.performance.llm_response_time = np.mean(list(self.performance_windows['llm_times']))
        
        # Reward rate
        if self.performance_windows['rewards']:
            self.performance.reward_rate = np.mean(list(self.performance_windows['rewards']))
        
        # Exploration rate
        if self.performance_windows['explorations']:
            self.performance.exploration_rate = np.mean(list(self.performance_windows['explorations']))
        
        # Success rate
        if self.performance_windows['successes']:
            self.performance.success_rate = np.mean(list(self.performance_windows['successes']))
    
    def _update_web_history(self, action_data: Dict[str, Any]) -> None:
        """Update web monitoring history."""
        # Keep last 100 entries for web display
        if len(self.web_stats_history['action_history']) >= 100:
            self.web_stats_history['action_history'].pop(0)
        
        self.web_stats_history['action_history'].append({
            'action': action_data['action'],
            'source': action_data['source'], 
            'reward': action_data['reward'],
            'timestamp': action_data['timestamp']
        })
        
        # Update reward history
        if len(self.web_stats_history['reward_history']) >= 100:
            self.web_stats_history['reward_history'].pop(0)
        
        self.web_stats_history['reward_history'].append(action_data['reward'])
    
    def _update_progression_history(self) -> None:
        """Update progression tracking for web display."""
        progression_data = {
            'badges': self.game_states['badges'],
            'level': self.game_states['level'],
            'money': self.game_states['money'],
            'party_count': self.game_states['party_count'],
            'unique_maps': len(self.game_states['visited_maps']),
            'timestamp': time.time()
        }
        
        if len(self.web_stats_history['progression']) >= 50:
            self.web_stats_history['progression'].pop(0)
        
        self.web_stats_history['progression'].append(progression_data)
    
    def get_current_stats(self) -> Dict[str, Any]:
        """Get current comprehensive statistics.
        
        Returns:
            Dict: Current statistics for monitoring/web display
        """
        with self._lock:
            current_time = time.time()
            session_duration = current_time - self.current_session.start_time
            
            return {
                # Basic session stats
                'session_name': self.session_name,
                'start_time': self.current_session.start_time,
                'session_duration': session_duration,
                'total_actions': self.current_session.total_actions,
                'total_reward': self.current_session.total_reward,
                'llm_calls': self.current_session.llm_calls,
                
                # Performance metrics
                'actions_per_second': self.performance.actions_per_second,
                'llm_avg_time': self.performance.llm_response_time,
                'reward_rate': self.performance.reward_rate,
                'exploration_rate': self.performance.exploration_rate,
                'success_rate': self.performance.success_rate,
                
                # Game progress
                'badges_earned': self.current_session.badges_earned,
                'levels_gained': self.current_session.levels_gained,
                'current_badges': self.game_states['badges'],
                'current_level': self.game_states['level'],
                'current_money': self.game_states['money'],
                'party_count': self.game_states['party_count'],
                'unique_locations': len(self.game_states['visited_positions']),
                'unique_maps': len(self.game_states['visited_maps']),
                
                # Issues and recovery
                'total_errors': self.current_session.errors,
                'stuck_episodes': self.stuck_episodes,
                'recovery_events': self.recovery_events,
                
                # Recent activity (for web display)
                'recent_rewards': list(self.performance_windows['rewards'])[-20:],
                'recent_actions': [a['action'] for a in list(self.action_history)[-20:]],
                'recent_llm_times': list(self.performance_windows['llm_times'])[-10:],
                
                # Web history
                'web_stats_history': self.web_stats_history
            }
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary of current session.
        
        Returns:
            Dict: Session summary data
        """
        with self._lock:
            stats = self.get_current_stats()
            
            return {
                'session_name': self.session_name,
                'duration': stats['session_duration'],
                'actions': stats['total_actions'],
                'reward': stats['total_reward'],
                'llm_usage': f"{stats['llm_calls']}/{stats['total_actions']} ({(stats['llm_calls']/max(stats['total_actions'], 1)*100):.1f}%)",
                'badges': f"{stats['current_badges']} (+{stats['badges_earned']})",
                'level': f"{stats['current_level']} (+{stats['levels_gained']})",
                'exploration': f"{stats['unique_maps']} maps, {stats['unique_locations']} locations",
                'performance': f"{stats['actions_per_second']:.1f} acts/sec",
                'errors': stats['total_errors']
            }
    
    def end_session(self) -> TrainingSession:
        """End current session and return session data.
        
        Returns:
            TrainingSession: Completed session data
        """
        with self._lock:
            self.current_session.end_time = time.time()
            completed_session = self.current_session
            
            self.session_history.append(completed_session)
            self.logger.info(f"Session ended: {self.get_session_summary()}")
            
            return completed_session
    
    def save_statistics(self, filepath: str) -> bool:
        """Save statistics to file.
        
        Args:
            filepath: Path to save statistics
            
        Returns:
            bool: True if save successful
        """
        try:
            with self._lock:
                stats_data = {
                    'session_name': self.session_name,
                    'current_session': {
                        'start_time': self.current_session.start_time,
                        'end_time': self.current_session.end_time,
                        'total_actions': self.current_session.total_actions,
                        'total_reward': self.current_session.total_reward,
                        'llm_calls': self.current_session.llm_calls,
                        'badges_earned': self.current_session.badges_earned,
                        'levels_gained': self.current_session.levels_gained,
                        'unique_locations': self.current_session.unique_locations,
                        'errors': self.current_session.errors
                    },
                    'game_states': {
                        **self.game_states,
                        'visited_maps': list(self.game_states['visited_maps']),
                        'visited_positions': list(self.game_states['visited_positions'])
                    },
                    'error_counts': dict(self.error_counts),
                    'stuck_episodes': self.stuck_episodes,
                    'recovery_events': self.recovery_events,
                    'statistics': self.get_current_stats(),
                    'timestamp': datetime.now().isoformat()
                }
                
                with open(filepath, 'w') as f:
                    json.dump(stats_data, f, indent=2, default=str)
                
                self.logger.info(f"Statistics saved to {filepath}")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to save statistics: {e}")
            return False
    
    def load_statistics(self, filepath: str) -> bool:
        """Load statistics from file.
        
        Args:
            filepath: Path to load statistics
            
        Returns:
            bool: True if load successful
        """
        try:
            import os
            
            if not os.path.exists(filepath):
                return False
            
            with open(filepath, 'r') as f:
                stats_data = json.load(f)
            
            with self._lock:
                # Restore session data
                if 'current_session' in stats_data:
                    session_data = stats_data['current_session']
                    self.current_session = TrainingSession(
                        start_time=session_data['start_time'],
                        end_time=session_data.get('end_time'),
                        total_actions=session_data.get('total_actions', 0),
                        total_reward=session_data.get('total_reward', 0.0),
                        llm_calls=session_data.get('llm_calls', 0),
                        badges_earned=session_data.get('badges_earned', 0),
                        levels_gained=session_data.get('levels_gained', 0),
                        unique_locations=session_data.get('unique_locations', 0),
                        errors=session_data.get('errors', 0)
                    )
                
                # Restore game states
                if 'game_states' in stats_data:
                    gs_data = stats_data['game_states']
                    self.game_states.update(gs_data)
                    if 'visited_maps' in gs_data:
                        self.game_states['visited_maps'] = set(gs_data['visited_maps'])
                    if 'visited_positions' in gs_data:
                        self.game_states['visited_positions'] = set(tuple(pos) for pos in gs_data['visited_positions'])
                
                # Restore counters
                if 'error_counts' in stats_data:
                    self.error_counts.update(stats_data['error_counts'])
                self.stuck_episodes = stats_data.get('stuck_episodes', 0)
                self.recovery_events = stats_data.get('recovery_events', 0)
                
                self.logger.info(f"Statistics loaded from {filepath}")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to load statistics: {e}")
            return False