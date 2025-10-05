"""
Systematic Sweep Exploration Pattern

Implements a systematic sweep pattern for thorough map coverage.
Sweeps horizontally, then changes rows when hitting obstacles.
"""

import time
from typing import Dict, Any, List, Tuple
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
