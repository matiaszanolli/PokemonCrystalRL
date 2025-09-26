"""
Official Exploration Pattern Plugins

This module contains official exploration pattern plugins that demonstrate
different approaches to map discovery and navigation.
"""

import time
import random
from typing import Dict, Any, List, Tuple
from collections import deque
from core.plugin_system import ExplorationPatternPlugin, PluginMetadata, PluginType


class SystematicSweepPattern(ExplorationPatternPlugin):
    """Systematic sweep exploration pattern for thorough map coverage"""

    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="systematic_sweep",
            version="1.0.0",
            description="Systematic sweep pattern for thorough map exploration",
            author="Pokemon Crystal RL Team",
            plugin_type=PluginType.EXPLORATION_PATTERN,
            hot_swappable=True,
            priority=8,
            tags=["exploration", "systematic", "thorough", "official"]
        )

    def initialize(self) -> bool:
        self.logger.info("Initializing Systematic Sweep Pattern")
        self.sweep_direction = 'right'  # Start sweeping right
        self.row_direction = 'down'     # When reaching edge, go down
        self.visited_positions = set()
        self.current_row = 0
        self.sweep_width = self.config.get('sweep_width', 20)
        self.pattern_state = 'sweeping'  # 'sweeping', 'row_change', 'reset'
        self.stuck_counter = 0
        self.max_stuck_attempts = 5
        return True

    def shutdown(self) -> bool:
        self.logger.info("Shutting down Systematic Sweep Pattern")
        return True

    def get_exploration_direction(self, game_state: Dict[str, Any], exploration_context: Dict[str, Any]) -> Dict[str, Any]:
        """Get next direction for systematic sweep"""
        start_time = time.time()

        try:
            current_pos = (game_state.get('player_x', 0), game_state.get('player_y', 0))
            available_directions = exploration_context.get('available_directions', [])

            # Track current position
            self.visited_positions.add(current_pos)

            # Determine next direction based on sweep pattern
            direction_result = self._get_sweep_direction(current_pos, available_directions, exploration_context)

            return {
                'direction': direction_result['direction'],
                'action': direction_result['action'],
                'confidence': direction_result['confidence'],
                'reasoning': direction_result['reasoning'],
                'pattern_type': 'systematic_sweep',
                'pattern_state': self.pattern_state,
                'coverage_estimate': len(self.visited_positions) / max(self.sweep_width * 20, 1)
            }

        finally:
            self._track_performance("get_exploration_direction", start_time)

    def update_exploration_state(self, game_state: Dict[str, Any], last_action: int) -> None:
        """Update sweep pattern state based on last action"""
        current_pos = (game_state.get('player_x', 0), game_state.get('player_y', 0))

        # Check if we're stuck
        if current_pos in self.visited_positions:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0

        # If stuck too long, change pattern
        if self.stuck_counter >= self.max_stuck_attempts:
            self._adjust_sweep_pattern()
            self.stuck_counter = 0

    def reset_exploration_pattern(self) -> None:
        """Reset sweep pattern to initial state"""
        self.sweep_direction = 'right'
        self.row_direction = 'down'
        self.visited_positions.clear()
        self.current_row = 0
        self.pattern_state = 'sweeping'
        self.stuck_counter = 0
        self.logger.info("Reset systematic sweep pattern")

    def _get_sweep_direction(self, current_pos: Tuple[int, int], available_directions: List[str], context: Dict[str, Any]) -> Dict[str, Any]:
        """Determine next sweep direction"""

        if self.pattern_state == 'sweeping':
            # Continue sweeping in current direction
            if self.sweep_direction in available_directions:
                return {
                    'direction': self.sweep_direction,
                    'action': self._direction_to_action(self.sweep_direction),
                    'confidence': 0.9,
                    'reasoning': f"Systematic sweep: continuing {self.sweep_direction}"
                }
            else:
                # Hit a wall, time to change rows
                self.pattern_state = 'row_change'
                return self._change_sweep_row(available_directions)

        elif self.pattern_state == 'row_change':
            # Move to next row
            if self.row_direction in available_directions:
                self.pattern_state = 'sweeping'
                self._reverse_sweep_direction()
                return {
                    'direction': self.row_direction,
                    'action': self._direction_to_action(self.row_direction),
                    'confidence': 0.8,
                    'reasoning': f"Systematic sweep: changing to next row ({self.row_direction})"
                }
            else:
                # Can't continue pattern, reset or adapt
                return self._adapt_sweep_pattern(available_directions)

        else:  # reset state
            return self._restart_sweep_pattern(available_directions)

    def _change_sweep_row(self, available_directions: List[str]) -> Dict[str, Any]:
        """Change to the next row in sweep pattern"""
        if self.row_direction in available_directions:
            return {
                'direction': self.row_direction,
                'action': self._direction_to_action(self.row_direction),
                'confidence': 0.8,
                'reasoning': f"Sweep row change: moving {self.row_direction}"
            }
        else:
            # Try opposite direction
            opposite_row = 'up' if self.row_direction == 'down' else 'down'
            if opposite_row in available_directions:
                self.row_direction = opposite_row
                return {
                    'direction': opposite_row,
                    'action': self._direction_to_action(opposite_row),
                    'confidence': 0.7,
                    'reasoning': f"Sweep adaptation: trying {opposite_row}"
                }
            else:
                return self._adapt_sweep_pattern(available_directions)

    def _reverse_sweep_direction(self) -> None:
        """Reverse the sweep direction for next row"""
        self.sweep_direction = 'left' if self.sweep_direction == 'right' else 'right'

    def _adapt_sweep_pattern(self, available_directions: List[str]) -> Dict[str, Any]:
        """Adapt when normal sweep pattern can't continue"""
        if available_directions:
            # Pick the direction that leads to least visited areas
            best_direction = available_directions[0]  # Default

            return {
                'direction': best_direction,
                'action': self._direction_to_action(best_direction),
                'confidence': 0.6,
                'reasoning': f"Sweep adaptation: trying {best_direction}"
            }
        else:
            return {
                'direction': 'right',
                'action': 4,
                'confidence': 0.3,
                'reasoning': "Sweep pattern stuck: default right movement"
            }

    def _restart_sweep_pattern(self, available_directions: List[str]) -> Dict[str, Any]:
        """Restart sweep pattern"""
        self.reset_exploration_pattern()
        if 'right' in available_directions:
            return {
                'direction': 'right',
                'action': 4,
                'confidence': 0.8,
                'reasoning': "Restarting systematic sweep pattern"
            }
        elif available_directions:
            return {
                'direction': available_directions[0],
                'action': self._direction_to_action(available_directions[0]),
                'confidence': 0.7,
                'reasoning': f"Restarting sweep with {available_directions[0]}"
            }
        else:
            return {
                'direction': 'right',
                'action': 4,
                'confidence': 0.3,
                'reasoning': "Default restart direction"
            }

    def _adjust_sweep_pattern(self) -> None:
        """Adjust sweep pattern when stuck"""
        self.pattern_state = 'reset'
        self.logger.debug("Adjusting sweep pattern due to being stuck")

    def _direction_to_action(self, direction: str) -> int:
        """Convert direction string to action number"""
        direction_map = {
            'up': 1,
            'down': 2,
            'left': 3,
            'right': 4
        }
        return direction_map.get(direction, 4)


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