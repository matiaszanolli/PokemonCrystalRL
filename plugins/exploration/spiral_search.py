"""
Spiral Search Exploration Pattern

Implements a spiral search pattern that expands outward from a center point.
Useful for thorough exploration from a known starting location.
"""

import time
from typing import Dict, Any, List, Tuple
from core.plugin_system import ExplorationPatternPlugin, PluginMetadata, PluginType


class SpiralSearchPattern(ExplorationPatternPlugin):
    """Spiral search pattern for expanding outward exploration"""

    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="spiral_search",
            version="1.0.0",
            description="Spiral search pattern expanding outward from center",
            author="Pokemon Crystal RL Team",
            plugin_type=PluginType.EXPLORATION_PATTERN,
            hot_swappable=True,
            priority=7,
            tags=["exploration", "spiral", "expanding", "official"]
        )

    def initialize(self) -> bool:
        self.logger.info("Initializing Spiral Search Pattern")
        self.center_pos = None
        self.spiral_radius = 1
        self.current_direction = 'right'
        self.steps_in_direction = 0
        self.steps_needed = 1
        self.direction_changes = 0
        self.visited_positions = set()
        return True

    def shutdown(self) -> bool:
        self.logger.info("Shutting down Spiral Search Pattern")
        return True

    def get_exploration_direction(self, game_state: Dict[str, Any], exploration_context: Dict[str, Any]) -> Dict[str, Any]:
        """Get next direction for spiral search"""
        start_time = time.time()

        try:
            current_pos = (game_state.get('player_x', 0), game_state.get('player_y', 0))
            available_directions = exploration_context.get('available_directions', [])

            # Set center if not set
            if self.center_pos is None:
                self.center_pos = current_pos

            self.visited_positions.add(current_pos)

            # Get next spiral direction
            direction_result = self._get_spiral_direction(current_pos, available_directions)

            return {
                'direction': direction_result['direction'],
                'action': direction_result['action'],
                'confidence': direction_result['confidence'],
                'reasoning': direction_result['reasoning'],
                'pattern_type': 'spiral_search',
                'spiral_radius': self.spiral_radius,
                'distance_from_center': self._distance_from_center(current_pos)
            }

        finally:
            self._track_performance("get_exploration_direction", start_time)

    def update_exploration_state(self, game_state: Dict[str, Any], last_action: int) -> None:
        """Update spiral pattern state"""
        self.steps_in_direction += 1

        # Check if we need to change direction
        if self.steps_in_direction >= self.steps_needed:
            self._change_spiral_direction()

    def reset_exploration_pattern(self) -> None:
        """Reset spiral pattern"""
        self.center_pos = None
        self.spiral_radius = 1
        self.current_direction = 'right'
        self.steps_in_direction = 0
        self.steps_needed = 1
        self.direction_changes = 0
        self.visited_positions.clear()
        self.logger.info("Reset spiral search pattern")

    def _get_spiral_direction(self, current_pos: Tuple[int, int], available_directions: List[str]) -> Dict[str, Any]:
        """Get next direction in spiral pattern"""

        if self.current_direction in available_directions:
            return {
                'direction': self.current_direction,
                'action': self._direction_to_action(self.current_direction),
                'confidence': 0.9,
                'reasoning': f"Spiral search: {self.current_direction} (radius {self.spiral_radius})"
            }
        else:
            # Blocked, try to adapt
            return self._adapt_spiral_pattern(available_directions)

    def _change_spiral_direction(self) -> None:
        """Change direction in spiral pattern"""
        # Spiral direction sequence: right -> down -> left -> up -> right (with increasing steps)
        direction_sequence = ['right', 'down', 'left', 'up']
        current_index = direction_sequence.index(self.current_direction)
        next_index = (current_index + 1) % 4

        self.current_direction = direction_sequence[next_index]
        self.steps_in_direction = 0
        self.direction_changes += 1

        # Increase steps needed after every 2 direction changes (completing one "ring" of spiral)
        if self.direction_changes % 2 == 0:
            self.steps_needed += 1
            if self.direction_changes % 4 == 0:  # Completed full spiral ring
                self.spiral_radius += 1

    def _adapt_spiral_pattern(self, available_directions: List[str]) -> Dict[str, Any]:
        """Adapt spiral when blocked"""
        if available_directions:
            # Try next direction in spiral sequence
            direction_sequence = ['right', 'down', 'left', 'up']
            for direction in direction_sequence:
                if direction in available_directions:
                    return {
                        'direction': direction,
                        'action': self._direction_to_action(direction),
                        'confidence': 0.7,
                        'reasoning': f"Spiral adaptation: trying {direction}"
                    }

            # If no spiral directions available, pick any
            direction = available_directions[0]
            return {
                'direction': direction,
                'action': self._direction_to_action(direction),
                'confidence': 0.5,
                'reasoning': f"Spiral blocked: using {direction}"
            }
        else:
            return {
                'direction': 'right',
                'action': 4,
                'confidence': 0.3,
                'reasoning': "Spiral stuck: default movement"
            }

    def _distance_from_center(self, current_pos: Tuple[int, int]) -> float:
        """Calculate distance from spiral center"""
        if self.center_pos is None:
            return 0.0
        return ((current_pos[0] - self.center_pos[0])**2 + (current_pos[1] - self.center_pos[1])**2)**0.5

    def _direction_to_action(self, direction: str) -> int:
        """Convert direction to action"""
        direction_map = {
            'up': 1,
            'down': 2,
            'left': 3,
            'right': 4
        }
        return direction_map.get(direction, 4)
