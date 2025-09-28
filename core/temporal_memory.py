#!/usr/bin/env python3
"""
Temporal Memory System for Hybrid LLM-RL Integration

Extends the existing ExperienceMemory system to support reinforcement learning
by tracking temporal sequences of states, actions, and rewards suitable for
RL algorithms while maintaining compatibility with LLM decision making.
"""

import json
import os
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Deque
from dataclasses import dataclass, asdict, field
from collections import deque
import time
import threading
from enum import Enum

from .experience_memory import ExperienceMemory, ExperienceEntry


class DecisionSource(Enum):
    """Source of the decision for tracking hybrid behavior."""
    LLM = "llm"
    RL = "rl"
    RULE_BASED = "rule_based"
    HYBRID = "hybrid"


@dataclass
class TemporalState:
    """Rich state representation for RL training."""
    # Game state features (normalized for RL)
    position: Tuple[int, int]
    map_id: int
    hp_ratio: float  # Current HP / Max HP
    level: int
    badges: int
    party_size: int
    money: int
    screen_state: str

    # Contextual features
    game_phase: str
    location_type: str
    in_battle: bool
    in_menu: bool

    # Temporal context
    timestep: int
    episode_time: float

    # Feature vector for RL (computed automatically)
    feature_vector: Optional[np.ndarray] = field(default=None, init=False)

    def __post_init__(self):
        """Generate feature vector for RL algorithms."""
        # Create normalized feature vector
        features = [
            self.position[0] / 255.0,  # Normalize screen coordinates
            self.position[1] / 255.0,
            self.map_id / 100.0,       # Normalize map ID
            self.hp_ratio,             # Already 0-1
            self.level / 100.0,        # Normalize level
            self.badges / 16.0,        # Max 16 badges
            self.party_size / 6.0,     # Max 6 Pokemon
            min(self.money / 999999.0, 1.0),  # Normalize money with cap

            # One-hot encode categorical features
            1.0 if self.in_battle else 0.0,
            1.0 if self.in_menu else 0.0,

            # Screen state encoding (simplified)
            hash(self.screen_state) % 10 / 10.0,  # Basic hash encoding

            # Temporal features
            self.timestep / 1000.0,    # Normalize timestep
            self.episode_time / 3600.0,  # Normalize episode time (hours)
        ]

        self.feature_vector = np.array(features, dtype=np.float32)


@dataclass
class TemporalExperience:
    """Single step in temporal sequence for RL training."""
    state: TemporalState
    action: int
    reward: float
    next_state: Optional[TemporalState]
    done: bool
    decision_source: DecisionSource
    llm_reasoning: Optional[str] = None
    llm_confidence: Optional[float] = None
    curriculum_level: Optional[int] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class Episode:
    """Complete episode for RL training."""
    episode_id: str
    save_state_id: Optional[str]  # Which save state started this episode
    curriculum_level: int
    experiences: List[TemporalExperience]
    total_reward: float
    total_steps: int
    success: bool
    duration: float
    start_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class TemporalMemoryBuffer:
    """
    Advanced memory system that bridges LLM reasoning and RL learning.

    Extends ExperienceMemory with temporal sequences, state-action-reward tracking,
    and hybrid decision support for both LLM strategic reasoning and RL optimization.
    """

    def __init__(self,
                 base_memory: ExperienceMemory,
                 buffer_size: int = 100000,
                 episode_buffer_size: int = 1000,
                 temporal_window: int = 10,
                 memory_file: str = "logs/temporal_memory.json"):
        """
        Initialize temporal memory buffer.

        Args:
            base_memory: Existing ExperienceMemory instance to extend
            buffer_size: Maximum number of temporal experiences to store
            episode_buffer_size: Maximum number of complete episodes to store
            temporal_window: Number of past states to consider for temporal patterns
            memory_file: File to persist temporal memory data
        """
        self.base_memory = base_memory
        self.memory_file = memory_file

        # Temporal experience buffer for RL training
        self.experience_buffer: Deque[TemporalExperience] = deque(maxlen=buffer_size)

        # Episode storage for curriculum learning and analysis
        self.episodes: Deque[Episode] = deque(maxlen=episode_buffer_size)

        # Current episode tracking
        self.current_episode: Optional[Episode] = None
        self.current_episode_experiences: List[TemporalExperience] = []

        # Temporal pattern tracking
        self.temporal_window = temporal_window
        self.state_history: Deque[TemporalState] = deque(maxlen=temporal_window)

        # Performance tracking
        self.rl_performance_history: List[Dict[str, float]] = []
        self.llm_performance_history: List[Dict[str, float]] = []

        # Thread safety
        self._lock = threading.RLock()

        self.load_temporal_memory()

    def start_episode(self,
                      save_state_id: Optional[str] = None,
                      curriculum_level: int = 0,
                      metadata: Dict[str, Any] = None) -> str:
        """Start a new episode for temporal tracking."""
        with self._lock:
            # Finish previous episode if exists
            if self.current_episode:
                self.end_episode()

            episode_id = f"ep_{int(time.time())}_{len(self.episodes)}"

            self.current_episode = Episode(
                episode_id=episode_id,
                save_state_id=save_state_id,
                curriculum_level=curriculum_level,
                experiences=[],
                total_reward=0.0,
                total_steps=0,
                success=False,
                duration=0.0,
                start_time=time.time(),
                metadata=metadata or {}
            )

            self.current_episode_experiences = []
            self.state_history.clear()

            return episode_id

    def record_temporal_experience(self,
                                   game_state: Dict[str, Any],
                                   screen_analysis: Dict[str, Any],
                                   action: int,
                                   reward: float,
                                   decision_source: DecisionSource,
                                   context: Dict[str, Any] = None,
                                   llm_reasoning: str = None,
                                   llm_confidence: float = None) -> None:
        """Record a temporal experience step."""
        with self._lock:
            if not self.current_episode:
                # Auto-start episode if none exists
                self.start_episode()

            # Create temporal state
            current_state = self._create_temporal_state(
                game_state, screen_analysis, context
            )

            # Get previous state for next_state reference
            previous_experience = (self.current_episode_experiences[-1]
                                 if self.current_episode_experiences else None)

            # Update previous experience with next_state
            if previous_experience and previous_experience.next_state is None:
                previous_experience.next_state = current_state

            # Create temporal experience
            experience = TemporalExperience(
                state=current_state,
                action=action,
                reward=reward,
                next_state=None,  # Will be set by next experience
                done=False,  # Will be set at episode end
                decision_source=decision_source,
                llm_reasoning=llm_reasoning,
                llm_confidence=llm_confidence,
                curriculum_level=self.current_episode.curriculum_level
            )

            # Add to current episode
            self.current_episode_experiences.append(experience)
            self.current_episode.total_reward += reward
            self.current_episode.total_steps += 1

            # Add to global buffer
            self.experience_buffer.append(experience)

            # Update state history for temporal patterns
            self.state_history.append(current_state)

            # Record in base memory for LLM integration
            situation_hash = self.base_memory.get_situation_hash(
                game_state, screen_analysis, context
            )
            # Convert action int to string for base memory compatibility
            action_str = self._action_int_to_string(action)
            self.base_memory.record_experience(
                situation_hash, [action_str], reward, context
            )

    def end_episode(self, success: bool = None) -> Optional[Episode]:
        """End the current episode and finalize temporal tracking."""
        with self._lock:
            if not self.current_episode:
                return None

            # Finalize episode
            self.current_episode.experiences = self.current_episode_experiences.copy()
            self.current_episode.duration = time.time() - self.current_episode.start_time

            # Auto-determine success if not provided
            if success is None:
                success = self._evaluate_episode_success()
            self.current_episode.success = success

            # Mark last experience as done
            if self.current_episode_experiences:
                self.current_episode_experiences[-1].done = True

            # Store completed episode
            completed_episode = self.current_episode
            self.episodes.append(completed_episode)

            # Update performance tracking
            self._update_performance_tracking(completed_episode)

            # Clear current episode
            self.current_episode = None
            self.current_episode_experiences = []

            return completed_episode

    def get_rl_batch(self, batch_size: int = 32) -> Optional[Dict[str, np.ndarray]]:
        """Get a batch of experiences for RL training."""
        with self._lock:
            if len(self.experience_buffer) < batch_size:
                return None

            # Sample random batch
            indices = np.random.choice(len(self.experience_buffer), batch_size, replace=False)
            batch_experiences = [self.experience_buffer[i] for i in indices]

            # Convert to RL-compatible format
            states = np.array([exp.state.feature_vector for exp in batch_experiences])
            actions = np.array([exp.action for exp in batch_experiences])
            rewards = np.array([exp.reward for exp in batch_experiences])
            next_states = np.array([
                exp.next_state.feature_vector if exp.next_state else np.zeros_like(exp.state.feature_vector)
                for exp in batch_experiences
            ])
            dones = np.array([exp.done for exp in batch_experiences])

            return {
                'states': states,
                'actions': actions,
                'rewards': rewards,
                'next_states': next_states,
                'dones': dones,
                'decision_sources': [exp.decision_source.value for exp in batch_experiences],
                'llm_confidences': [exp.llm_confidence or 0.0 for exp in batch_experiences]
            }

    def get_temporal_patterns(self,
                              pattern_length: int = 5,
                              min_frequency: int = 3) -> List[Dict[str, Any]]:
        """Extract temporal patterns for hybrid decision making."""
        with self._lock:
            patterns = {}

            # Extract patterns from completed episodes
            for episode in self.episodes:
                if len(episode.experiences) < pattern_length:
                    continue

                for i in range(len(episode.experiences) - pattern_length + 1):
                    # Extract state-action pattern
                    pattern_states = [exp.state for exp in episode.experiences[i:i+pattern_length]]
                    pattern_actions = [exp.action for exp in episode.experiences[i:i+pattern_length]]

                    # Create pattern signature
                    pattern_key = self._create_pattern_signature(pattern_states, pattern_actions)

                    if pattern_key not in patterns:
                        patterns[pattern_key] = {
                            'states': pattern_states,
                            'actions': pattern_actions,
                            'frequency': 0,
                            'success_rate': 0.0,
                            'average_reward': 0.0,
                            'decision_sources': []
                        }

                    pattern_data = patterns[pattern_key]
                    pattern_data['frequency'] += 1

                    # Calculate pattern outcome
                    pattern_reward = sum(exp.reward for exp in episode.experiences[i:i+pattern_length])
                    pattern_data['average_reward'] = (
                        (pattern_data['average_reward'] * (pattern_data['frequency'] - 1) + pattern_reward)
                        / pattern_data['frequency']
                    )

                    # Track decision sources
                    sources = [exp.decision_source.value for exp in episode.experiences[i:i+pattern_length]]
                    pattern_data['decision_sources'].append(sources)

            # Filter by frequency and return top patterns
            frequent_patterns = [
                pattern for pattern in patterns.values()
                if pattern['frequency'] >= min_frequency
            ]

            return sorted(frequent_patterns, key=lambda x: x['average_reward'], reverse=True)

    def get_llm_guidance_for_rl(self,
                                current_state: TemporalState,
                                context: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """Get LLM guidance to inform RL policy."""
        # Use base memory to get LLM recommendations
        game_state = self._temporal_state_to_game_state(current_state)
        screen_analysis = {'state': current_state.screen_state}

        situation_hash = self.base_memory.get_situation_hash(
            game_state, screen_analysis, context
        )

        recommended_actions = self.base_memory.get_recommended_actions(
            situation_hash, context
        )

        if recommended_actions:
            # Convert string actions to integers
            action_ints = [self._action_string_to_int(action) for action in recommended_actions]

            return {
                'recommended_actions': action_ints,
                'source': 'experience_memory',
                'confidence': 0.8  # Base confidence from experience
            }

        return None

    def _create_temporal_state(self,
                               game_state: Dict[str, Any],
                               screen_analysis: Dict[str, Any],
                               context: Dict[str, Any] = None) -> TemporalState:
        """Create a TemporalState from game data."""
        ctx = context or {}

        return TemporalState(
            position=(game_state.get('player_x', 0), game_state.get('player_y', 0)),
            map_id=game_state.get('map_id', 0),
            hp_ratio=game_state.get('hp_current', 1) / max(game_state.get('hp_max', 1), 1),
            level=game_state.get('level', 1),
            badges=game_state.get('badges', 0),
            party_size=game_state.get('party_size', 1),
            money=game_state.get('money', 0),
            screen_state=screen_analysis.get('state', 'unknown'),
            game_phase=ctx.get('phase', 'unknown'),
            location_type=ctx.get('location_type', 'unknown'),
            in_battle=game_state.get('in_battle', False),
            in_menu=screen_analysis.get('state') == 'menu',
            timestep=len(self.experience_buffer),
            episode_time=time.time() - (self.current_episode.start_time if self.current_episode else time.time())
        )

    def _evaluate_episode_success(self) -> bool:
        """Auto-evaluate episode success based on reward and patterns."""
        if not self.current_episode_experiences:
            return False

        total_reward = sum(exp.reward for exp in self.current_episode_experiences)
        avg_reward = total_reward / len(self.current_episode_experiences)

        # Simple heuristic: positive average reward indicates success
        return avg_reward > 0.1

    def _update_performance_tracking(self, episode: Episode) -> None:
        """Update performance metrics for LLM and RL decisions."""
        llm_experiences = [exp for exp in episode.experiences if exp.decision_source == DecisionSource.LLM]
        rl_experiences = [exp for exp in episode.experiences if exp.decision_source == DecisionSource.RL]

        if llm_experiences:
            llm_reward = sum(exp.reward for exp in llm_experiences) / len(llm_experiences)
            self.llm_performance_history.append({
                'episode_id': episode.episode_id,
                'average_reward': llm_reward,
                'decision_count': len(llm_experiences),
                'timestamp': time.time()
            })

        if rl_experiences:
            rl_reward = sum(exp.reward for exp in rl_experiences) / len(rl_experiences)
            self.rl_performance_history.append({
                'episode_id': episode.episode_id,
                'average_reward': rl_reward,
                'decision_count': len(rl_experiences),
                'timestamp': time.time()
            })

    def _create_pattern_signature(self, states: List[TemporalState], actions: List[int]) -> str:
        """Create a signature for a temporal pattern."""
        # Simplified pattern signature based on key state features
        signature_elements = []
        for state in states:
            signature_elements.extend([
                state.map_id,
                int(state.hp_ratio * 10),  # Discretize HP ratio
                state.in_battle,
                state.in_menu
            ])
        signature_elements.extend(actions)

        return str(hash(tuple(signature_elements)))

    def _action_int_to_string(self, action: int) -> str:
        """Convert action integer to string for base memory compatibility."""
        action_map = {
            0: "none", 1: "up", 2: "down", 3: "left", 4: "right",
            5: "a", 6: "b", 7: "start", 8: "select"
        }
        return action_map.get(action, "unknown")

    def _action_string_to_int(self, action: str) -> int:
        """Convert action string to integer for RL compatibility."""
        action_map = {
            "none": 0, "up": 1, "down": 2, "left": 3, "right": 4,
            "a": 5, "b": 6, "start": 7, "select": 8
        }
        return action_map.get(action.lower(), 0)

    def _temporal_state_to_game_state(self, temporal_state: TemporalState) -> Dict[str, Any]:
        """Convert TemporalState back to game_state dict for base memory compatibility."""
        return {
            'player_x': temporal_state.position[0],
            'player_y': temporal_state.position[1],
            'map_id': temporal_state.map_id,
            'hp_current': int(temporal_state.hp_ratio * 100),  # Approximate
            'hp_max': 100,  # Approximate
            'level': temporal_state.level,
            'badges': temporal_state.badges,
            'party_size': temporal_state.party_size,
            'money': temporal_state.money,
            'in_battle': temporal_state.in_battle
        }

    def save_temporal_memory(self) -> None:
        """Save temporal memory to disk."""
        with self._lock:
            os.makedirs(os.path.dirname(self.memory_file), exist_ok=True)

            # Convert to serializable format
            save_data = {
                'episodes': [asdict(episode) for episode in list(self.episodes)],
                'llm_performance': self.llm_performance_history,
                'rl_performance': self.rl_performance_history,
                'metadata': {
                    'save_time': time.time(),
                    'total_episodes': len(self.episodes),
                    'total_experiences': len(self.experience_buffer),
                    'buffer_size': self.experience_buffer.maxlen,
                    'temporal_window': self.temporal_window
                }
            }

            with open(self.memory_file, 'w') as f:
                json.dump(save_data, f, indent=2, default=str)  # default=str for numpy arrays

    def load_temporal_memory(self) -> None:
        """Load temporal memory from disk."""
        if not os.path.exists(self.memory_file):
            return

        try:
            with open(self.memory_file, 'r') as f:
                data = json.load(f)

            # Load episodes (reconstructing complex objects)
            self.episodes.clear()
            for episode_data in data.get('episodes', []):
                episode = Episode(**episode_data)
                self.episodes.append(episode)

            # Load performance history
            self.llm_performance_history = data.get('llm_performance', [])
            self.rl_performance_history = data.get('rl_performance', [])

            print(f"🧠 Loaded {len(self.episodes)} episodes and {len(self.experience_buffer)} temporal experiences")

        except Exception as e:
            print(f"⚠️ Failed to load temporal memory: {e}")
            self.episodes.clear()
            self.llm_performance_history = []
            self.rl_performance_history = []

    def get_temporal_stats(self) -> Dict[str, Any]:
        """Get comprehensive temporal memory statistics."""
        with self._lock:
            base_stats = self.base_memory.get_memory_stats()

            if not self.episodes:
                return {**base_stats, "temporal": {"episodes": 0, "experiences": 0}}

            successful_episodes = sum(1 for ep in self.episodes if ep.success)
            avg_episode_length = sum(len(ep.experiences) for ep in self.episodes) / len(self.episodes)
            avg_episode_reward = sum(ep.total_reward for ep in self.episodes) / len(self.episodes)

            # LLM vs RL performance comparison
            recent_llm_perf = self.llm_performance_history[-10:] if self.llm_performance_history else []
            recent_rl_perf = self.rl_performance_history[-10:] if self.rl_performance_history else []

            avg_llm_reward = (sum(p['average_reward'] for p in recent_llm_perf) / len(recent_llm_perf)) if recent_llm_perf else 0
            avg_rl_reward = (sum(p['average_reward'] for p in recent_rl_perf) / len(recent_rl_perf)) if recent_rl_perf else 0

            return {
                **base_stats,
                "temporal": {
                    "total_episodes": len(self.episodes),
                    "successful_episodes": successful_episodes,
                    "success_rate": successful_episodes / len(self.episodes),
                    "average_episode_length": avg_episode_length,
                    "average_episode_reward": avg_episode_reward,
                    "total_experiences": len(self.experience_buffer),
                    "llm_performance": {
                        "recent_average_reward": avg_llm_reward,
                        "total_decisions": len(self.llm_performance_history)
                    },
                    "rl_performance": {
                        "recent_average_reward": avg_rl_reward,
                        "total_decisions": len(self.rl_performance_history)
                    }
                }
            }