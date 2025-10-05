"""
Pokemon Crystal RL Reward Calculator

This module provides the component-based reward calculator implementation for
Pokemon Crystal RL training. The calculator composes multiple reward components
to evaluate game progress and provide appropriate reinforcement signals.

Features:
- Modular reward component system
- Comprehensive state validation
- Anti-farming and anti-glitch protection
- Screen state awareness
- Action-specific reward adjustments

The reward calculator implements the RewardCalculatorInterface and orchestrates
multiple reward components, each handling specific aspects of reward calculation.
"""

from typing import Dict, Tuple, Optional
from .interface import RewardCalculatorInterface


class PokemonRewardCalculator(RewardCalculatorInterface):
    """Sophisticated reward calculation for Pokemon Crystal using component system."""

    def __init__(self):
        self._last_screen_state = 'unknown'
        self._prev_screen_state = 'unknown'
        self._last_action = None

        # Initialize all reward components
        from .components.progress import HealthRewardComponent, LevelRewardComponent, BadgeRewardComponent
        from .components.movement import ExplorationRewardComponent, MovementRewardComponent, BlockedMovementComponent
        from .components.interaction import BattleRewardComponent, DialogueRewardComponent, MoneyRewardComponent, ProgressionRewardComponent

        self.components = [
            HealthRewardComponent(),
            LevelRewardComponent(),
            BadgeRewardComponent(),
            ExplorationRewardComponent(),
            MovementRewardComponent(),
            BlockedMovementComponent(),
            BattleRewardComponent(),
            DialogueRewardComponent(),
            MoneyRewardComponent(),
            ProgressionRewardComponent()
        ]

    @property
    def last_screen_state(self) -> str:
        return self._last_screen_state

    @last_screen_state.setter
    def last_screen_state(self, state: str):
        self._last_screen_state = state
        # Propagate to components
        for component in self.components:
            component.last_screen_state = state

    @property
    def prev_screen_state(self) -> str:
        return self._prev_screen_state

    @prev_screen_state.setter
    def prev_screen_state(self, state: str):
        self._prev_screen_state = state
        # Propagate to components
        for component in self.components:
            component.prev_screen_state = state

    @property
    def last_action(self) -> Optional[str]:
        return self._last_action

    @last_action.setter
    def last_action(self, action: Optional[str]):
        self._last_action = action
        # Propagate to components
        for component in self.components:
            component.last_action = action

    def calculate_reward(self, current_state: Dict, previous_state: Dict) -> Tuple[float, Dict[str, float]]:
        """Calculate comprehensive reward using all components.

        Args:
            current_state: Current game state dictionary
            previous_state: Previous game state dictionary

        Returns:
            tuple: (total_reward, reward_breakdown)
                - total_reward (float): Sum of all component rewards
                - reward_breakdown (dict): Detailed breakdown by component
        """
        total_reward = 0.0
        all_rewards = {}

        # Calculate rewards from each component
        for component in self.components:
            reward, details = component.calculate(current_state, previous_state)
            all_rewards.update(details)  # Merge component reward details
            total_reward += reward

        # Add time-based efficiency penalty
        time_penalty = -0.01
        all_rewards['time'] = time_penalty
        total_reward += time_penalty

        return total_reward, all_rewards

    def get_reward_summary(self, rewards: Dict[str, float]) -> str:
        """Get a human-readable summary of rewards.

        Args:
            rewards: Dictionary mapping reward categories to values

        Returns:
            str: Human-readable summary string
        """
        summary_parts = []
        for category, value in rewards.items():
            if abs(value) >= 0.01:  # Only show significant rewards (inclusive of 0.01)
                summary_parts.append(f"{category}: {value:+.2f}")

        return " | ".join(summary_parts) if summary_parts else "no rewards"
