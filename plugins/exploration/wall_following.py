"""
Wall Following Exploration Pattern

Implements a wall-following algorithm for systematic boundary exploration.
Follows walls on either the right or left side for complete area coverage.
"""

import time
from typing import Dict, Any, List, Tuple
from core.plugin_system import ExplorationPatternPlugin, PluginMetadata, PluginType


class WallFollowingPattern(ExplorationPatternPlugin):
    """Wall following pattern for systematic boundary exploration"""

    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="wall_following",
            version="1.0.0",
            description="Wall following pattern for boundary-based exploration",
            author="Pokemon Crystal RL Team",
            plugin_type=PluginType.EXPLORATION_PATTERN,
            hot_swappable=True,
            priority=6,
            tags=["exploration", "wall", "boundary", "official"]
        )

    def initialize(self) -> bool:
        self.logger.info("Initializing Wall Following Pattern")
        self.wall_side = self.config.get('wall_side', 'right')  # 'right' or 'left'
        self.current_direction = 'up'
        self.last_wall_contact = None
        self.visited_positions = set()
        self.following_wall = False
        return True

    def shutdown(self) -> bool:
        self.logger.info("Shutting down Wall Following Pattern")
        return True

    def get_exploration_direction(self, game_state: Dict[str, Any], exploration_context: Dict[str, Any]) -> Dict[str, Any]:
        """Get next direction for wall following"""
        start_time = time.time()

        try:
            current_pos = (game_state.get('player_x', 0), game_state.get('player_y', 0))
            available_directions = exploration_context.get('available_directions', [])
            blocked_directions = exploration_context.get('blocked_directions', [])

            self.visited_positions.add(current_pos)

            # Determine wall following direction
            direction_result = self._get_wall_following_direction(
                current_pos, available_directions, blocked_directions
            )

            return {
                'direction': direction_result['direction'],
                'action': direction_result['action'],
                'confidence': direction_result['confidence'],
                'reasoning': direction_result['reasoning'],
                'pattern_type': 'wall_following',
                'wall_side': self.wall_side,
                'following_wall': self.following_wall
            }

        finally:
            self._track_performance("get_exploration_direction", start_time)

    def update_exploration_state(self, game_state: Dict[str, Any], last_action: int) -> None:
        """Update wall following state"""
        # Update current direction based on last action
        action_to_direction = {1: 'up', 2: 'down', 3: 'left', 4: 'right'}
        if last_action in action_to_direction:
            self.current_direction = action_to_direction[last_action]

    def reset_exploration_pattern(self) -> None:
        """Reset wall following pattern"""
        self.current_direction = 'up'
        self.last_wall_contact = None
        self.visited_positions.clear()
        self.following_wall = False
        self.logger.info("Reset wall following pattern")

    def _get_wall_following_direction(self, current_pos: Tuple[int, int],
                                    available_directions: List[str],
                                    blocked_directions: List[str]) -> Dict[str, Any]:
        """Determine wall following direction"""

        if not self.following_wall:
            # Look for a wall to start following
            if blocked_directions:
                self.following_wall = True
                wall_direction = self._find_wall_direction(blocked_directions, available_directions)
                return {
                    'direction': wall_direction,
                    'action': self._direction_to_action(wall_direction),
                    'confidence': 0.8,
                    'reasoning': f"Starting wall following: {wall_direction}"
                }
            else:
                # No wall found, move to find one
                return self._search_for_wall(available_directions)
        else:
            # Currently following wall
            return self._continue_wall_following(available_directions, blocked_directions)

    def _find_wall_direction(self, blocked_directions: List[str], available_directions: List[str]) -> str:
        """Find initial direction to follow wall"""
        # Start by moving parallel to the wall
        if self.wall_side == 'right':
            # Keep wall on right side
            if 'down' in blocked_directions and 'right' in available_directions:
                return 'right'
            elif 'right' in blocked_directions and 'up' in available_directions:
                return 'up'
            elif 'up' in blocked_directions and 'left' in available_directions:
                return 'left'
            elif 'left' in blocked_directions and 'down' in available_directions:
                return 'down'

        # Default to any available direction
        return available_directions[0] if available_directions else 'right'

    def _continue_wall_following(self, available_directions: List[str], blocked_directions: List[str]) -> Dict[str, Any]:
        """Continue following the wall"""

        # Wall following algorithm: try to turn toward wall, then go straight, then turn away from wall
        if self.wall_side == 'right':
            preferred_order = self._get_right_wall_following_order()
        else:
            preferred_order = self._get_left_wall_following_order()

        for direction in preferred_order:
            if direction in available_directions:
                self.current_direction = direction
                return {
                    'direction': direction,
                    'action': self._direction_to_action(direction),
                    'confidence': 0.9,
                    'reasoning': f"Wall following ({self.wall_side} side): {direction}"
                }

        # If no preferred direction available, pick any
        if available_directions:
            direction = available_directions[0]
            return {
                'direction': direction,
                'action': self._direction_to_action(direction),
                'confidence': 0.5,
                'reasoning': f"Wall following fallback: {direction}"
            }
        else:
            return {
                'direction': 'right',
                'action': 4,
                'confidence': 0.3,
                'reasoning': "Wall following stuck: default movement"
            }

    def _get_right_wall_following_order(self) -> List[str]:
        """Get direction priority order for right wall following"""
        direction_order = {
            'up': ['right', 'up', 'left', 'down'],
            'right': ['down', 'right', 'up', 'left'],
            'down': ['left', 'down', 'right', 'up'],
            'left': ['up', 'left', 'down', 'right']
        }
        return direction_order.get(self.current_direction, ['right', 'up', 'left', 'down'])

    def _get_left_wall_following_order(self) -> List[str]:
        """Get direction priority order for left wall following"""
        direction_order = {
            'up': ['left', 'up', 'right', 'down'],
            'left': ['down', 'left', 'up', 'right'],
            'down': ['right', 'down', 'left', 'up'],
            'right': ['up', 'right', 'down', 'left']
        }
        return direction_order.get(self.current_direction, ['left', 'up', 'right', 'down'])

    def _search_for_wall(self, available_directions: List[str]) -> Dict[str, Any]:
        """Search for a wall to follow"""
        if available_directions:
            # Move in a direction to find a wall
            direction = available_directions[0]
            return {
                'direction': direction,
                'action': self._direction_to_action(direction),
                'confidence': 0.6,
                'reasoning': f"Searching for wall: moving {direction}"
            }
        else:
            return {
                'direction': 'right',
                'action': 4,
                'confidence': 0.3,
                'reasoning': "No wall found: default movement"
            }

    def _direction_to_action(self, direction: str) -> int:
        """Convert direction to action"""
        direction_map = {
            'up': 1,
            'down': 2,
            'left': 3,
            'right': 4
        }
        return direction_map.get(direction, 4)


