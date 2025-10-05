"""
Random Walk Exploration Pattern

Implements a biased random walk for exploratory movement.
Balances randomness with tendency toward unexplored areas.
"""

import time
import random
from typing import Dict, Any, List, Tuple
from core.plugin_system import ExplorationPatternPlugin, PluginMetadata, PluginType


class RandomWalkPattern(ExplorationPatternPlugin):
    """Random walk pattern for unpredictable exploration"""

    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="random_walk",
            version="1.0.0",
            description="Random walk pattern for unpredictable exploration",
            author="Pokemon Crystal RL Team",
            plugin_type=PluginType.EXPLORATION_PATTERN,
            hot_swappable=True,
            priority=4,
            tags=["exploration", "random", "unpredictable", "official"]
        )

    def initialize(self) -> bool:
        self.logger.info("Initializing Random Walk Pattern")
        self.bias_towards_unvisited = self.config.get('bias_unvisited', True)
        self.visited_positions = set()
        self.direction_persistence = self.config.get('persistence', 3)  # Steps to continue in same direction
        self.current_direction = None
        self.steps_in_direction = 0
        random.seed(self.config.get('seed'))  # Allow reproducible randomness
        return True

    def shutdown(self) -> bool:
        self.logger.info("Shutting down Random Walk Pattern")
        return True

    def get_exploration_direction(self, game_state: Dict[str, Any], exploration_context: Dict[str, Any]) -> Dict[str, Any]:
        """Get next direction for random walk"""
        start_time = time.time()

        try:
            current_pos = (game_state.get('player_x', 0), game_state.get('player_y', 0))
            available_directions = exploration_context.get('available_directions', [])

            self.visited_positions.add(current_pos)

            # Get random direction with optional bias
            direction_result = self._get_random_direction(current_pos, available_directions)

            return {
                'direction': direction_result['direction'],
                'action': direction_result['action'],
                'confidence': direction_result['confidence'],
                'reasoning': direction_result['reasoning'],
                'pattern_type': 'random_walk',
                'visited_count': len(self.visited_positions),
                'persistence_remaining': max(0, self.direction_persistence - self.steps_in_direction)
            }

        finally:
            self._track_performance("get_exploration_direction", start_time)

    def update_exploration_state(self, game_state: Dict[str, Any], last_action: int) -> None:
        """Update random walk state"""
        action_to_direction = {1: 'up', 2: 'down', 3: 'left', 4: 'right'}
        last_direction = action_to_direction.get(last_action)

        if last_direction == self.current_direction:
            self.steps_in_direction += 1
        else:
            self.current_direction = last_direction
            self.steps_in_direction = 1

    def reset_exploration_pattern(self) -> None:
        """Reset random walk pattern"""
        self.visited_positions.clear()
        self.current_direction = None
        self.steps_in_direction = 0
        self.logger.info("Reset random walk pattern")

    def _get_random_direction(self, current_pos: Tuple[int, int], available_directions: List[str]) -> Dict[str, Any]:
        """Get random direction with optional bias"""

        if not available_directions:
            return {
                'direction': 'right',
                'action': 4,
                'confidence': 0.3,
                'reasoning': "No directions available: default"
            }

        # Apply direction persistence
        if (self.current_direction and
            self.current_direction in available_directions and
            self.steps_in_direction < self.direction_persistence):

            return {
                'direction': self.current_direction,
                'action': self._direction_to_action(self.current_direction),
                'confidence': 0.8,
                'reasoning': f"Random walk persistence: continuing {self.current_direction}"
            }

        # Choose new random direction
        if self.bias_towards_unvisited:
            direction = self._choose_biased_direction(current_pos, available_directions)
            reasoning = "Random walk with unvisited bias"
            confidence = 0.7
        else:
            direction = random.choice(available_directions)
            reasoning = "Pure random walk"
            confidence = 0.6

        self.current_direction = direction
        self.steps_in_direction = 0

        return {
            'direction': direction,
            'action': self._direction_to_action(direction),
            'confidence': confidence,
            'reasoning': f"{reasoning}: {direction}"
        }

    def _choose_biased_direction(self, current_pos: Tuple[int, int], available_directions: List[str]) -> str:
        """Choose direction biased towards unvisited areas"""
        direction_weights = []

        for direction in available_directions:
            # Calculate position if we move in this direction
            next_pos = self._get_next_position(current_pos, direction)

            # Weight based on whether position has been visited
            if next_pos in self.visited_positions:
                weight = 1  # Lower weight for visited positions
            else:
                weight = 3  # Higher weight for unvisited positions

            direction_weights.append((direction, weight))

        # Weighted random selection
        total_weight = sum(weight for _, weight in direction_weights)
        if total_weight == 0:
            return random.choice(available_directions)

        rand_value = random.uniform(0, total_weight)
        cumulative_weight = 0

        for direction, weight in direction_weights:
            cumulative_weight += weight
            if rand_value <= cumulative_weight:
                return direction

        return available_directions[0]  # Fallback

    def _get_next_position(self, current_pos: Tuple[int, int], direction: str) -> Tuple[int, int]:
        """Calculate next position based on direction"""
        x, y = current_pos

        if direction == 'up':
            return (x, y - 1)
        elif direction == 'down':
            return (x, y + 1)
        elif direction == 'left':
            return (x - 1, y)
        elif direction == 'right':
            return (x + 1, y)
        else:
            return current_pos

    def _direction_to_action(self, direction: str) -> int:
        """Convert direction to action"""
        direction_map = {
            'up': 1,
            'down': 2,
            'left': 3,
            'right': 4
        }
        return direction_map.get(direction, 4)