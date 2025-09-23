"""
Enhanced Stuck Detection and Recovery System

Advanced system for detecting when the AI is stuck and implementing
smart recovery strategies to break out of loops and unproductive patterns.
"""

import time
import logging
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Set
from dataclasses import dataclass
from collections import deque, Counter
from enum import Enum

logger = logging.getLogger(__name__)


class StuckType(Enum):
    """Types of stuck patterns we can detect."""
    POSITION_LOOP = "position_loop"         # Looping between same positions
    ACTION_REPETITION = "action_repetition" # Repeating same actions
    SCREEN_STUCK = "screen_stuck"          # Same screen content repeatedly
    DIALOGUE_LOOP = "dialogue_loop"        # Stuck in dialogue/menu
    BATTLE_STUCK = "battle_stuck"          # Stuck in battle with no progress
    EXPLORATION_STUCK = "exploration_stuck" # Not exploring new areas
    REWARD_PLATEAU = "reward_plateau"      # No reward progress
    NONE = "none"


@dataclass
class StuckPattern:
    """Represents a detected stuck pattern."""
    stuck_type: StuckType
    severity: float          # 0.0-1.0, higher = more stuck
    duration: int           # How long we've been stuck
    pattern_data: Dict[str, Any]  # Additional pattern-specific data
    recovery_suggestions: List[str]  # Suggested recovery actions


@dataclass
class RecoveryStrategy:
    """Represents a recovery strategy."""
    name: str
    actions: List[int]      # Sequence of actions to try
    priority: int           # Higher = try first
    conditions: Dict[str, Any]  # When to use this strategy
    success_rate: float     # Historical success rate


class EnhancedStuckDetector:
    """Advanced stuck detection with multiple detection methods."""

    def __init__(self):
        self.logger = logging.getLogger("EnhancedStuckDetector")

        # History tracking for pattern detection
        self.position_history = deque(maxlen=50)
        self.action_history = deque(maxlen=30)
        self.screen_hash_history = deque(maxlen=25)
        self.reward_history = deque(maxlen=40)
        self.state_history = deque(maxlen=20)

        # Detection thresholds
        self.thresholds = {
            'position_loop_threshold': 8,      # Same position repeated 8 times
            'action_repetition_threshold': 6,  # Same action 6 times in row
            'screen_stuck_threshold': 10,      # Same screen hash 10 times
            'exploration_stuck_threshold': 15,  # No new positions in 15 actions
            'reward_plateau_threshold': 20,     # No positive reward in 20 actions
            'severe_stuck_threshold': 0.8       # Severity above which we're "severely stuck"
        }

        # Recovery strategies
        self.recovery_strategies = self._initialize_recovery_strategies()

        # State tracking
        self.current_stuck_pattern = None
        self.stuck_start_time = None
        self.recovery_attempts = 0
        self.last_recovery_action_count = 0

        self.logger.info("Enhanced stuck detector initialized")

    def _initialize_recovery_strategies(self) -> List[RecoveryStrategy]:
        """Initialize various recovery strategies."""
        return [
            # Menu-based recovery (highest priority for dialogue/menu stuck)
            RecoveryStrategy(
                name="menu_escape",
                actions=[6, 6, 6, 7, 6],  # B, B, B, START, B
                priority=10,
                conditions={"stuck_types": [StuckType.DIALOGUE_LOOP, StuckType.SCREEN_STUCK]},
                success_rate=0.8
            ),

            # Direction randomization for position loops
            RecoveryStrategy(
                name="random_directions",
                actions=[4, 3, 1, 2, 4, 1],  # Random movement pattern
                priority=8,
                conditions={"stuck_types": [StuckType.POSITION_LOOP, StuckType.EXPLORATION_STUCK]},
                success_rate=0.7
            ),

            # Interaction attempts for general stuck situations
            RecoveryStrategy(
                name="interaction_sequence",
                actions=[5, 5, 6, 1, 5, 2, 5],  # A, A, B, UP, A, DOWN, A
                priority=6,
                conditions={"stuck_types": [StuckType.SCREEN_STUCK, StuckType.BATTLE_STUCK]},
                success_rate=0.6
            ),

            # Menu navigation for complex stuck situations
            RecoveryStrategy(
                name="menu_navigation",
                actions=[7, 1, 5, 6, 7, 2, 5, 6],  # START, UP, A, B, START, DOWN, A, B
                priority=7,
                conditions={"stuck_types": [StuckType.DIALOGUE_LOOP, StuckType.BATTLE_STUCK]},
                success_rate=0.5
            ),

            # Long sequence for severe stuck situations
            RecoveryStrategy(
                name="comprehensive_reset",
                actions=[6, 6, 6, 7, 8, 6, 1, 2, 3, 4, 5, 6],  # Complex sequence
                priority=9,
                conditions={"severity_threshold": 0.8},
                success_rate=0.4
            ),

            # Simple movement for light stuck situations
            RecoveryStrategy(
                name="gentle_movement",
                actions=[1, 4, 2, 3],  # UP, RIGHT, DOWN, LEFT
                priority=4,
                conditions={"severity_threshold": 0.3},
                success_rate=0.9
            )
        ]

    def update_history(self,
                      game_state: Dict[str, Any],
                      action: int,
                      reward: float,
                      screen_data: Optional[np.ndarray] = None) -> None:
        """Update all history tracking for pattern detection."""

        # Update position history
        position = (
            game_state.get('player_x', 0),
            game_state.get('player_y', 0),
            game_state.get('player_map', 0)
        )
        self.position_history.append(position)

        # Update action history
        self.action_history.append(action)

        # Update reward history
        self.reward_history.append(reward)

        # Update screen hash if available
        if screen_data is not None:
            screen_hash = hash(screen_data.tobytes())
            self.screen_hash_history.append(screen_hash)

        # Update state history
        state_info = {
            'in_battle': game_state.get('in_battle', False),
            'party_count': game_state.get('party_count', 0),
            'badges': game_state.get('badges', 0)
        }
        self.state_history.append(state_info)

    def detect_stuck_patterns(self, action_count: int) -> Optional[StuckPattern]:
        """Detect various types of stuck patterns."""

        # Don't detect stuck patterns too early
        if len(self.position_history) < 10:
            return None

        detected_patterns = []

        # Check for position loops
        position_pattern = self._detect_position_loop()
        if position_pattern:
            detected_patterns.append(position_pattern)

        # Check for action repetition
        action_pattern = self._detect_action_repetition()
        if action_pattern:
            detected_patterns.append(action_pattern)

        # Check for screen stuck
        screen_pattern = self._detect_screen_stuck()
        if screen_pattern:
            detected_patterns.append(screen_pattern)

        # Check for exploration stuck
        exploration_pattern = self._detect_exploration_stuck()
        if exploration_pattern:
            detected_patterns.append(exploration_pattern)

        # Check for reward plateau
        reward_pattern = self._detect_reward_plateau()
        if reward_pattern:
            detected_patterns.append(reward_pattern)

        # Check for dialogue/battle stuck
        state_pattern = self._detect_state_stuck()
        if state_pattern:
            detected_patterns.append(state_pattern)

        # Return the most severe pattern
        if detected_patterns:
            most_severe = max(detected_patterns, key=lambda p: p.severity)

            # Update stuck tracking
            if self.current_stuck_pattern is None:
                self.stuck_start_time = time.time()
                self.current_stuck_pattern = most_severe
            else:
                # Update duration
                self.current_stuck_pattern.duration = int(time.time() - self.stuck_start_time)

            return most_severe

        # No stuck pattern detected - reset tracking
        self.current_stuck_pattern = None
        self.stuck_start_time = None
        return None

    def _detect_position_loop(self) -> Optional[StuckPattern]:
        """Detect if agent is looping between same positions."""
        if len(self.position_history) < self.thresholds['position_loop_threshold']:
            return None

        recent_positions = list(self.position_history)[-self.thresholds['position_loop_threshold']:]
        position_counts = Counter(recent_positions)
        most_common = position_counts.most_common(1)[0]

        if most_common[1] >= self.thresholds['position_loop_threshold'] * 0.6:
            severity = min(1.0, most_common[1] / self.thresholds['position_loop_threshold'])

            return StuckPattern(
                stuck_type=StuckType.POSITION_LOOP,
                severity=severity,
                duration=0,
                pattern_data={'repeated_position': most_common[0], 'count': most_common[1]},
                recovery_suggestions=['Try different movement directions', 'Use menu actions']
            )

        return None

    def _detect_action_repetition(self) -> Optional[StuckPattern]:
        """Detect if agent is repeating the same action."""
        if len(self.action_history) < self.thresholds['action_repetition_threshold']:
            return None

        recent_actions = list(self.action_history)[-self.thresholds['action_repetition_threshold']:]
        action_counts = Counter(recent_actions)
        most_common = action_counts.most_common(1)[0]

        if most_common[1] >= self.thresholds['action_repetition_threshold'] * 0.7:
            severity = min(1.0, most_common[1] / self.thresholds['action_repetition_threshold'])

            return StuckPattern(
                stuck_type=StuckType.ACTION_REPETITION,
                severity=severity,
                duration=0,
                pattern_data={'repeated_action': most_common[0], 'count': most_common[1]},
                recovery_suggestions=['Vary action choices', 'Try menu or interaction actions']
            )

        return None

    def _detect_screen_stuck(self) -> Optional[StuckPattern]:
        """Detect if screen content isn't changing."""
        if len(self.screen_hash_history) < self.thresholds['screen_stuck_threshold']:
            return None

        recent_hashes = list(self.screen_hash_history)[-self.thresholds['screen_stuck_threshold']:]
        hash_counts = Counter(recent_hashes)
        most_common = hash_counts.most_common(1)[0]

        if most_common[1] >= self.thresholds['screen_stuck_threshold'] * 0.8:
            severity = min(1.0, most_common[1] / self.thresholds['screen_stuck_threshold'])

            return StuckPattern(
                stuck_type=StuckType.SCREEN_STUCK,
                severity=severity,
                duration=0,
                pattern_data={'screen_repetitions': most_common[1]},
                recovery_suggestions=['Press B to exit', 'Try menu navigation', 'Use different actions']
            )

        return None

    def _detect_exploration_stuck(self) -> Optional[StuckPattern]:
        """Detect if agent isn't exploring new areas."""
        if len(self.position_history) < self.thresholds['exploration_stuck_threshold']:
            return None

        recent_positions = list(self.position_history)[-self.thresholds['exploration_stuck_threshold']:]
        unique_positions = len(set(recent_positions))

        exploration_rate = unique_positions / len(recent_positions)

        if exploration_rate < 0.3:  # Less than 30% unique positions
            severity = 1.0 - exploration_rate

            return StuckPattern(
                stuck_type=StuckType.EXPLORATION_STUCK,
                severity=severity,
                duration=0,
                pattern_data={'exploration_rate': exploration_rate, 'unique_positions': unique_positions},
                recovery_suggestions=['Try new movement directions', 'Explore different areas']
            )

        return None

    def _detect_reward_plateau(self) -> Optional[StuckPattern]:
        """Detect if agent isn't making reward progress."""
        if len(self.reward_history) < self.thresholds['reward_plateau_threshold']:
            return None

        recent_rewards = list(self.reward_history)[-self.thresholds['reward_plateau_threshold']:]
        positive_rewards = [r for r in recent_rewards if r > 0]

        if len(positive_rewards) == 0:
            # No positive rewards in recent history
            avg_reward = sum(recent_rewards) / len(recent_rewards)
            severity = min(1.0, abs(avg_reward) / 5.0)  # Scale by typical reward magnitude

            return StuckPattern(
                stuck_type=StuckType.REWARD_PLATEAU,
                severity=severity,
                duration=0,
                pattern_data={'avg_reward': avg_reward, 'positive_count': 0},
                recovery_suggestions=['Try exploration', 'Interact with objects', 'Change strategy']
            )

        return None

    def _detect_state_stuck(self) -> Optional[StuckPattern]:
        """Detect if agent is stuck in battle or dialogue."""
        if len(self.state_history) < 8:
            return None

        recent_states = list(self.state_history)[-8:]

        # Check for battle stuck
        battle_states = [s['in_battle'] for s in recent_states]
        if all(battle_states):
            # Stuck in battle for 8+ actions
            return StuckPattern(
                stuck_type=StuckType.BATTLE_STUCK,
                severity=0.7,
                duration=0,
                pattern_data={'battle_duration': len(recent_states)},
                recovery_suggestions=['Try different battle actions', 'Use menu options']
            )

        # Check for dialogue stuck (would need more sophisticated detection)
        # For now, we rely on screen_stuck detection for this

        return None

    def get_recovery_strategy(self, stuck_pattern: StuckPattern) -> Optional[RecoveryStrategy]:
        """Get the best recovery strategy for a stuck pattern."""
        if stuck_pattern is None:
            return None

        # Filter strategies by conditions
        applicable_strategies = []

        for strategy in self.recovery_strategies:
            conditions = strategy.conditions

            # Check stuck type conditions
            if 'stuck_types' in conditions:
                if stuck_pattern.stuck_type in conditions['stuck_types']:
                    applicable_strategies.append(strategy)
                    continue

            # Check severity conditions
            if 'severity_threshold' in conditions:
                if stuck_pattern.severity >= conditions['severity_threshold']:
                    applicable_strategies.append(strategy)
                    continue

        # If no specific strategies apply, use general ones
        if not applicable_strategies:
            applicable_strategies = [s for s in self.recovery_strategies
                                   if 'stuck_types' not in s.conditions and
                                      'severity_threshold' not in s.conditions]

        # Sort by priority and success rate
        applicable_strategies.sort(key=lambda s: (s.priority, s.success_rate), reverse=True)

        return applicable_strategies[0] if applicable_strategies else None

    def should_trigger_recovery(self, stuck_pattern: StuckPattern, action_count: int) -> bool:
        """Determine if we should trigger recovery actions."""
        if stuck_pattern is None:
            return False

        # Don't trigger too frequently
        if action_count - self.last_recovery_action_count < 10:
            return False

        # Trigger based on severity
        if stuck_pattern.severity >= self.thresholds['severe_stuck_threshold']:
            return True

        # Trigger if stuck for a while
        if stuck_pattern.duration > 15:  # Stuck for 15+ seconds
            return True

        # Trigger if moderate stuck and we haven't tried recovery recently
        if stuck_pattern.severity >= 0.5 and action_count - self.last_recovery_action_count > 20:
            return True

        return False

    def execute_recovery(self, strategy: RecoveryStrategy, action_count: int) -> List[int]:
        """Execute a recovery strategy and return actions to take."""
        self.logger.info(f"Executing recovery strategy: {strategy.name}")
        self.recovery_attempts += 1
        self.last_recovery_action_count = action_count

        # Update strategy success rate based on usage
        # (This would be improved with actual success tracking)

        return strategy.actions

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about stuck detection."""
        return {
            'current_stuck': self.current_stuck_pattern.stuck_type.value if self.current_stuck_pattern else None,
            'current_severity': self.current_stuck_pattern.severity if self.current_stuck_pattern else 0.0,
            'stuck_duration': self.current_stuck_pattern.duration if self.current_stuck_pattern else 0,
            'recovery_attempts': self.recovery_attempts,
            'position_history_size': len(self.position_history),
            'action_history_size': len(self.action_history),
            'available_strategies': len(self.recovery_strategies)
        }

    def reset_stuck_state(self) -> None:
        """Reset stuck detection state (call when making progress)."""
        self.current_stuck_pattern = None
        self.stuck_start_time = None
        self.logger.debug("Stuck state reset - progress detected")