"""
Tests for Exploration Pattern Plugins

This module contains comprehensive tests for the official exploration pattern plugins
including systematic sweep, spiral search, wall following, and random walk patterns.
"""

import pytest
from unittest.mock import Mock, patch
from typing import Dict, Any

from plugins.exploration_patterns import (
    SystematicSweepPattern, SpiralSearchPattern, WallFollowingPattern, RandomWalkPattern
)
from core.plugin_system import PluginMetadata, PluginType


class TestSystematicSweepPattern:
    """Test SystematicSweepPattern plugin"""

    def setup_method(self):
        """Setup for each test"""
        self.pattern = SystematicSweepPattern()
        self.pattern.initialize()

    def test_metadata(self):
        """Test plugin metadata"""
        metadata = self.pattern.get_metadata()

        assert isinstance(metadata, PluginMetadata)
        assert metadata.name == "systematic_sweep"
        assert metadata.plugin_type == PluginType.EXPLORATION_PATTERN
        assert metadata.priority == 8
        assert "systematic" in metadata.tags
        assert metadata.hot_swappable is True

    def test_initialization(self):
        """Test plugin initialization"""
        config = {'sweep_width': 30}
        pattern = SystematicSweepPattern(config)
        result = pattern.initialize()

        assert result is True
        assert pattern.sweep_width == 30
        assert pattern.sweep_direction == 'right'
        assert pattern.row_direction == 'down'
        assert pattern.pattern_state == 'sweeping'

    def test_shutdown(self):
        """Test plugin shutdown"""
        result = self.pattern.shutdown()
        assert result is True

    def test_get_exploration_direction_sweeping(self):
        """Test exploration direction during normal sweeping"""
        game_state = {'player_x': 5, 'player_y': 10}
        exploration_context = {'available_directions': ['right', 'up', 'left']}

        result = self.pattern.get_exploration_direction(game_state, exploration_context)

        assert result['direction'] == 'right'
        assert result['action'] == 4
        assert result['confidence'] == 0.9
        assert 'systematic sweep' in result['reasoning']
        assert result['pattern_type'] == 'systematic_sweep'
        assert result['pattern_state'] == 'sweeping'

    def test_get_exploration_direction_hit_wall(self):
        """Test exploration direction when hitting a wall"""
        self.pattern.pattern_state = 'sweeping'
        self.pattern.sweep_direction = 'right'

        game_state = {'player_x': 5, 'player_y': 10}
        exploration_context = {'available_directions': ['up', 'down', 'left']}  # No right

        result = self.pattern.get_exploration_direction(game_state, exploration_context)

        # Should change to row change mode
        assert result['direction'] == 'down'  # row_direction
        assert result['action'] == 2
        assert 'changing to next row' in result['reasoning']

    def test_get_exploration_direction_row_change(self):
        """Test exploration direction during row change"""
        self.pattern.pattern_state = 'row_change'
        self.pattern.row_direction = 'down'

        game_state = {'player_x': 5, 'player_y': 10}
        exploration_context = {'available_directions': ['down', 'left']}

        result = self.pattern.get_exploration_direction(game_state, exploration_context)

        assert result['direction'] == 'down'
        assert 'changing to next row' in result['reasoning']

    def test_update_exploration_state_normal(self):
        """Test updating exploration state during normal movement"""
        game_state = {'player_x': 5, 'player_y': 10}

        self.pattern.update_exploration_state(game_state, 4)  # Right action

        # Should not be stuck since position is new
        assert self.pattern.stuck_counter == 0

    def test_update_exploration_state_stuck(self):
        """Test updating exploration state when stuck"""
        position = (5, 10)
        game_state = {'player_x': position[0], 'player_y': position[1]}

        # Mark position as visited
        self.pattern.visited_positions.add(position)

        # Update with same position multiple times
        for _ in range(6):  # More than max_stuck_attempts
            self.pattern.update_exploration_state(game_state, 4)

        # Should have reset pattern due to being stuck
        assert self.pattern.stuck_counter == 0  # Reset after adjustment

    def test_reset_exploration_pattern(self):
        """Test resetting exploration pattern"""
        # Modify some state
        self.pattern.sweep_direction = 'left'
        self.pattern.pattern_state = 'row_change'
        self.pattern.visited_positions.add((10, 15))
        self.pattern.stuck_counter = 3

        self.pattern.reset_exploration_pattern()

        # Should reset to initial state
        assert self.pattern.sweep_direction == 'right'
        assert self.pattern.row_direction == 'down'
        assert self.pattern.pattern_state == 'sweeping'
        assert len(self.pattern.visited_positions) == 0
        assert self.pattern.stuck_counter == 0

    def test_reverse_sweep_direction(self):
        """Test sweep direction reversal"""
        # Test right -> left
        self.pattern.sweep_direction = 'right'
        self.pattern._reverse_sweep_direction()
        assert self.pattern.sweep_direction == 'left'

        # Test left -> right
        self.pattern._reverse_sweep_direction()
        assert self.pattern.sweep_direction == 'right'

    def test_direction_to_action_mapping(self):
        """Test direction to action conversion"""
        assert self.pattern._direction_to_action('up') == 1
        assert self.pattern._direction_to_action('down') == 2
        assert self.pattern._direction_to_action('left') == 3
        assert self.pattern._direction_to_action('right') == 4
        assert self.pattern._direction_to_action('invalid') == 4  # Default


class TestSpiralSearchPattern:
    """Test SpiralSearchPattern plugin"""

    def setup_method(self):
        """Setup for each test"""
        self.pattern = SpiralSearchPattern()
        self.pattern.initialize()

    def test_metadata(self):
        """Test plugin metadata"""
        metadata = self.pattern.get_metadata()

        assert metadata.name == "spiral_search"
        assert metadata.plugin_type == PluginType.EXPLORATION_PATTERN
        assert metadata.priority == 7
        assert "spiral" in metadata.tags

    def test_initialization(self):
        """Test plugin initialization"""
        assert self.pattern.center_pos is None
        assert self.pattern.spiral_radius == 1
        assert self.pattern.current_direction == 'right'
        assert self.pattern.steps_in_direction == 0
        assert self.pattern.steps_needed == 1

    def test_get_exploration_direction_set_center(self):
        """Test exploration direction setting center position"""
        game_state = {'player_x': 10, 'player_y': 15}
        exploration_context = {'available_directions': ['right', 'up']}

        result = self.pattern.get_exploration_direction(game_state, exploration_context)

        # Should set center position on first call
        assert self.pattern.center_pos == (10, 15)
        assert result['direction'] == 'right'
        assert result['pattern_type'] == 'spiral_search'

    def test_get_exploration_direction_spiral_movement(self):
        """Test spiral movement pattern"""
        self.pattern.center_pos = (10, 15)
        self.pattern.current_direction = 'down'

        game_state = {'player_x': 11, 'player_y': 15}
        exploration_context = {'available_directions': ['down', 'left', 'up']}

        result = self.pattern.get_exploration_direction(game_state, exploration_context)

        assert result['direction'] == 'down'
        assert result['confidence'] == 0.9
        assert 'spiral search' in result['reasoning']
        assert 'radius' in result['reasoning']

    def test_get_exploration_direction_blocked(self):
        """Test exploration direction when blocked"""
        self.pattern.center_pos = (10, 15)
        self.pattern.current_direction = 'right'

        game_state = {'player_x': 10, 'player_y': 15}
        exploration_context = {'available_directions': ['up', 'down']}  # Right blocked

        result = self.pattern.get_exploration_direction(game_state, exploration_context)

        # Should adapt by choosing available direction
        assert result['direction'] in ['up', 'down']
        assert 'adaptation' in result['reasoning']

    def test_update_exploration_state(self):
        """Test updating spiral exploration state"""
        game_state = {'player_x': 10, 'player_y': 15}

        initial_steps = self.pattern.steps_in_direction
        self.pattern.update_exploration_state(game_state, 4)  # Right action

        assert self.pattern.steps_in_direction == initial_steps + 1

    def test_change_spiral_direction(self):
        """Test spiral direction changes"""
        # Test direction sequence: right -> down -> left -> up -> right
        initial_direction = 'right'
        self.pattern.current_direction = initial_direction
        self.pattern.steps_needed = 2
        self.pattern.steps_in_direction = 2

        self.pattern._change_spiral_direction()

        assert self.pattern.current_direction == 'down'
        assert self.pattern.steps_in_direction == 0
        assert self.pattern.direction_changes == 1

    def test_spiral_radius_increase(self):
        """Test spiral radius increases"""
        # Simulate completing one ring of spiral (4 direction changes)
        self.pattern.direction_changes = 4
        initial_radius = self.pattern.spiral_radius

        self.pattern._change_spiral_direction()

        # After completing a ring, radius should increase
        assert self.pattern.spiral_radius > initial_radius

    def test_distance_from_center(self):
        """Test distance calculation from center"""
        self.pattern.center_pos = (0, 0)

        # Test distance calculation
        distance1 = self.pattern._distance_from_center((3, 4))
        assert abs(distance1 - 5.0) < 0.001  # 3-4-5 triangle

        distance2 = self.pattern._distance_from_center((0, 0))
        assert distance2 == 0.0  # Same position

        # Test with no center set
        self.pattern.center_pos = None
        distance3 = self.pattern._distance_from_center((5, 5))
        assert distance3 == 0.0

    def test_reset_exploration_pattern(self):
        """Test resetting spiral pattern"""
        # Modify state
        self.pattern.center_pos = (10, 15)
        self.pattern.spiral_radius = 3
        self.pattern.current_direction = 'left'
        self.pattern.visited_positions.add((5, 5))

        self.pattern.reset_exploration_pattern()

        # Should reset to initial state
        assert self.pattern.center_pos is None
        assert self.pattern.spiral_radius == 1
        assert self.pattern.current_direction == 'right'
        assert len(self.pattern.visited_positions) == 0


class TestWallFollowingPattern:
    """Test WallFollowingPattern plugin"""

    def setup_method(self):
        """Setup for each test"""
        self.pattern = WallFollowingPattern()
        self.pattern.initialize()

    def test_metadata(self):
        """Test plugin metadata"""
        metadata = self.pattern.get_metadata()

        assert metadata.name == "wall_following"
        assert metadata.plugin_type == PluginType.EXPLORATION_PATTERN
        assert metadata.priority == 6
        assert "wall" in metadata.tags

    def test_initialization_with_config(self):
        """Test initialization with wall side configuration"""
        config = {'wall_side': 'left'}
        pattern = WallFollowingPattern(config)
        pattern.initialize()

        assert pattern.wall_side == 'left'
        assert pattern.current_direction == 'up'
        assert pattern.following_wall is False

    def test_get_exploration_direction_find_wall(self):
        """Test exploration direction when looking for wall"""
        self.pattern.following_wall = False

        game_state = {'player_x': 5, 'player_y': 10}
        exploration_context = {
            'available_directions': ['right', 'up'],
            'blocked_directions': ['left', 'down']
        }

        result = self.pattern.get_exploration_direction(game_state, exploration_context)

        # Should start following wall
        assert self.pattern.following_wall is True
        assert result['pattern_type'] == 'wall_following'
        assert result['wall_side'] == 'right'
        assert 'Starting wall following' in result['reasoning']

    def test_get_exploration_direction_no_wall(self):
        """Test exploration direction when no wall found"""
        self.pattern.following_wall = False

        game_state = {'player_x': 5, 'player_y': 10}
        exploration_context = {
            'available_directions': ['right', 'up', 'left', 'down'],
            'blocked_directions': []
        }

        result = self.pattern.get_exploration_direction(game_state, exploration_context)

        # Should search for wall
        assert result['direction'] in ['right', 'up', 'left', 'down']
        assert 'Searching for wall' in result['reasoning']

    def test_get_exploration_direction_following_wall(self):
        """Test exploration direction while following wall"""
        self.pattern.following_wall = True
        self.pattern.wall_side = 'right'
        self.pattern.current_direction = 'up'

        game_state = {'player_x': 5, 'player_y': 10}
        exploration_context = {
            'available_directions': ['up', 'left'],
            'blocked_directions': ['right', 'down']
        }

        result = self.pattern.get_exploration_direction(game_state, exploration_context)

        assert result['pattern_type'] == 'wall_following'
        assert result['following_wall'] is True
        assert 'Wall following' in result['reasoning']

    def test_update_exploration_state(self):
        """Test updating exploration state"""
        initial_direction = self.pattern.current_direction

        self.pattern.update_exploration_state({}, 4)  # Right action

        assert self.pattern.current_direction == 'right'

    def test_right_wall_following_order(self):
        """Test right wall following direction order"""
        # Test different current directions
        self.pattern.current_direction = 'up'
        order = self.pattern._get_right_wall_following_order()
        assert order == ['right', 'up', 'left', 'down']

        self.pattern.current_direction = 'right'
        order = self.pattern._get_right_wall_following_order()
        assert order == ['down', 'right', 'up', 'left']

    def test_left_wall_following_order(self):
        """Test left wall following direction order"""
        self.pattern.current_direction = 'up'
        order = self.pattern._get_left_wall_following_order()
        assert order == ['left', 'up', 'right', 'down']

        self.pattern.current_direction = 'left'
        order = self.pattern._get_left_wall_following_order()
        assert order == ['down', 'left', 'up', 'right']

    def test_find_wall_direction(self):
        """Test finding initial wall direction"""
        blocked_directions = ['down']
        available_directions = ['right', 'up', 'left']

        direction = self.pattern._find_wall_direction(blocked_directions, available_directions)
        assert direction in available_directions

    def test_reset_exploration_pattern(self):
        """Test resetting wall following pattern"""
        # Modify state
        self.pattern.following_wall = True
        self.pattern.current_direction = 'left'
        self.pattern.visited_positions.add((5, 5))

        self.pattern.reset_exploration_pattern()

        # Should reset to initial state
        assert self.pattern.current_direction == 'up'
        assert self.pattern.following_wall is False
        assert len(self.pattern.visited_positions) == 0


class TestRandomWalkPattern:
    """Test RandomWalkPattern plugin"""

    def setup_method(self):
        """Setup for each test"""
        self.pattern = RandomWalkPattern()
        self.pattern.initialize()

    def test_metadata(self):
        """Test plugin metadata"""
        metadata = self.pattern.get_metadata()

        assert metadata.name == "random_walk"
        assert metadata.plugin_type == PluginType.EXPLORATION_PATTERN
        assert metadata.priority == 4
        assert "random" in metadata.tags

    def test_initialization_with_config(self):
        """Test initialization with configuration"""
        config = {
            'bias_unvisited': False,
            'persistence': 5,
            'seed': 12345
        }
        pattern = RandomWalkPattern(config)
        pattern.initialize()

        assert pattern.bias_towards_unvisited is False
        assert pattern.direction_persistence == 5

    def test_get_exploration_direction_no_directions(self):
        """Test exploration direction with no available directions"""
        game_state = {'player_x': 5, 'player_y': 10}
        exploration_context = {'available_directions': []}

        result = self.pattern.get_exploration_direction(game_state, exploration_context)

        assert result['direction'] == 'right'
        assert result['action'] == 4
        assert result['confidence'] == 0.3
        assert 'No directions available' in result['reasoning']

    def test_get_exploration_direction_persistence(self):
        """Test direction persistence"""
        self.pattern.current_direction = 'up'
        self.pattern.steps_in_direction = 2
        self.pattern.direction_persistence = 3

        game_state = {'player_x': 5, 'player_y': 10}
        exploration_context = {'available_directions': ['up', 'right', 'left']}

        result = self.pattern.get_exploration_direction(game_state, exploration_context)

        # Should continue in current direction due to persistence
        assert result['direction'] == 'up'
        assert 'persistence' in result['reasoning']
        assert result['persistence_remaining'] == 1

    def test_get_exploration_direction_pure_random(self):
        """Test pure random direction selection"""
        self.pattern.bias_towards_unvisited = False
        self.pattern.current_direction = None  # Force new direction

        game_state = {'player_x': 5, 'player_y': 10}
        exploration_context = {'available_directions': ['up', 'right', 'left', 'down']}

        with patch('random.choice', return_value='up') as mock_choice:
            result = self.pattern.get_exploration_direction(game_state, exploration_context)

            assert result['direction'] == 'up'
            assert 'Pure random walk' in result['reasoning']
            mock_choice.assert_called_once()

    def test_get_exploration_direction_biased(self):
        """Test biased random direction selection"""
        self.pattern.bias_towards_unvisited = True
        self.pattern.current_direction = None
        # Add visited position
        self.pattern.visited_positions.add((5, 9))  # Up from current position

        game_state = {'player_x': 5, 'player_y': 10}
        exploration_context = {'available_directions': ['up', 'right']}

        result = self.pattern.get_exploration_direction(game_state, exploration_context)

        # Should prefer unvisited directions (right over up)
        assert result['direction'] in ['up', 'right']
        assert 'bias' in result['reasoning']

    def test_update_exploration_state(self):
        """Test updating random walk state"""
        initial_steps = self.pattern.steps_in_direction

        # Test continuing in same direction
        self.pattern.current_direction = 'right'
        self.pattern.update_exploration_state({}, 4)  # Right action
        assert self.pattern.steps_in_direction == initial_steps + 1

        # Test changing direction
        self.pattern.update_exploration_state({}, 1)  # Up action
        assert self.pattern.current_direction == 'up'
        assert self.pattern.steps_in_direction == 1

    def test_choose_biased_direction(self):
        """Test biased direction selection"""
        current_pos = (5, 10)
        available_directions = ['up', 'right', 'down']

        # Mark 'up' position as visited
        self.pattern.visited_positions.add((5, 9))

        with patch('random.uniform') as mock_uniform:
            # Mock random selection to prefer unvisited
            mock_uniform.return_value = 3.5  # Should select unvisited direction

            direction = self.pattern._choose_biased_direction(current_pos, available_directions)

            assert direction in ['right', 'down']  # Should avoid visited 'up'

    def test_get_next_position(self):
        """Test position calculation"""
        current_pos = (10, 15)

        # Test all directions
        assert self.pattern._get_next_position(current_pos, 'up') == (10, 14)
        assert self.pattern._get_next_position(current_pos, 'down') == (10, 16)
        assert self.pattern._get_next_position(current_pos, 'left') == (9, 15)
        assert self.pattern._get_next_position(current_pos, 'right') == (11, 15)
        assert self.pattern._get_next_position(current_pos, 'invalid') == (10, 15)

    def test_reset_exploration_pattern(self):
        """Test resetting random walk pattern"""
        # Modify state
        self.pattern.visited_positions.add((5, 10))
        self.pattern.current_direction = 'left'
        self.pattern.steps_in_direction = 3

        self.pattern.reset_exploration_pattern()

        # Should reset to initial state
        assert len(self.pattern.visited_positions) == 0
        assert self.pattern.current_direction is None
        assert self.pattern.steps_in_direction == 0


@pytest.mark.integration
class TestExplorationPatternIntegration:
    """Integration tests for exploration pattern plugins"""

    def test_all_patterns_consistency(self):
        """Test that all patterns implement required interface consistently"""
        patterns = [
            SystematicSweepPattern(),
            SpiralSearchPattern(),
            WallFollowingPattern(),
            RandomWalkPattern()
        ]

        game_state = {'player_x': 10, 'player_y': 15}
        exploration_context = {
            'available_directions': ['up', 'right', 'down'],
            'blocked_directions': ['left']
        }

        for pattern in patterns:
            pattern.initialize()

            # Test required methods
            result = pattern.get_exploration_direction(game_state, exploration_context)
            assert 'direction' in result
            assert 'action' in result
            assert 'confidence' in result
            assert 'reasoning' in result

            # Test update method (should not crash)
            pattern.update_exploration_state(game_state, result['action'])

            # Test reset method
            pattern.reset_exploration_pattern()

            pattern.shutdown()

    def test_pattern_performance_tracking(self):
        """Test that patterns track performance correctly"""
        pattern = SystematicSweepPattern()
        pattern.initialize()

        initial_calls = pattern.performance_stats['calls']

        # Make several calls
        for _ in range(5):
            pattern.get_exploration_direction({'player_x': 5, 'player_y': 10}, {'available_directions': ['right']})

        # Performance should be tracked
        assert pattern.performance_stats['calls'] > initial_calls
        assert pattern.performance_stats['total_time'] > 0

    def test_pattern_state_persistence(self):
        """Test that patterns maintain state correctly across calls"""
        # Test systematic sweep direction consistency
        sweep = SystematicSweepPattern()
        sweep.initialize()

        game_state = {'player_x': 10, 'player_y': 15}
        context = {'available_directions': ['right', 'up', 'left', 'down']}

        # Should start with right
        result1 = sweep.get_exploration_direction(game_state, context)
        assert result1['direction'] == 'right'

        # Should continue with right if available
        result2 = sweep.get_exploration_direction(game_state, context)
        assert result2['direction'] == 'right'

        # Test spiral pattern radius growth
        spiral = SpiralSearchPattern()
        spiral.initialize()
        spiral.center_pos = (10, 15)

        # Simulate completing a spiral ring
        for _ in range(8):  # Complete direction changes
            spiral._change_spiral_direction()

        assert spiral.spiral_radius > 1

    def test_pattern_adaptation_to_constraints(self):
        """Test how patterns adapt to movement constraints"""
        patterns = [
            SystematicSweepPattern(),
            SpiralSearchPattern(),
            WallFollowingPattern(),
            RandomWalkPattern()
        ]

        # Test with limited directions
        limited_context = {'available_directions': ['up'], 'blocked_directions': ['right', 'down', 'left']}
        game_state = {'player_x': 10, 'player_y': 15}

        for pattern in patterns:
            pattern.initialize()

            result = pattern.get_exploration_direction(game_state, limited_context)

            # Should adapt to constraints
            assert result['direction'] == 'up'  # Only available direction
            assert result['action'] == 1

    def test_pattern_error_recovery(self):
        """Test pattern error handling and recovery"""
        patterns = [
            SystematicSweepPattern(),
            SpiralSearchPattern(),
            WallFollowingPattern(),
            RandomWalkPattern()
        ]

        # Test with malformed input
        malformed_state = {'invalid_key': 'invalid_value'}
        empty_context = {}

        for pattern in patterns:
            pattern.initialize()

            # Should not crash with malformed input
            try:
                result = pattern.get_exploration_direction(malformed_state, empty_context)
                assert isinstance(result, dict)
                assert 'direction' in result
                assert 'action' in result
            except KeyError:
                # Some patterns might access specific keys, which is acceptable
                pass

    @patch('time.time')
    def test_concurrent_pattern_usage(self, mock_time):
        """Test patterns work correctly when used concurrently"""
        import threading
        import time as real_time

        mock_time.return_value = 1000.0

        pattern = SystematicSweepPattern()
        pattern.initialize()

        results = []
        errors = []

        def use_pattern():
            try:
                for i in range(10):
                    game_state = {'player_x': i, 'player_y': 5}
                    context = {'available_directions': ['right', 'up']}
                    result = pattern.get_exploration_direction(game_state, context)
                    results.append(result)
                    real_time.sleep(0.001)  # Small delay
            except Exception as e:
                errors.append(e)

        # Run multiple threads
        threads = [threading.Thread(target=use_pattern) for _ in range(3)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        # Should not have errors and should have collected results
        assert len(errors) == 0
        assert len(results) == 30  # 3 threads * 10 calls each

    def test_pattern_config_variations(self):
        """Test patterns with different configurations"""
        # Test systematic sweep with different widths
        sweep_narrow = SystematicSweepPattern({'sweep_width': 10})
        sweep_wide = SystematicSweepPattern({'sweep_width': 50})

        sweep_narrow.initialize()
        sweep_wide.initialize()

        assert sweep_narrow.sweep_width == 10
        assert sweep_wide.sweep_width == 50

        # Test wall following with different sides
        wall_right = WallFollowingPattern({'wall_side': 'right'})
        wall_left = WallFollowingPattern({'wall_side': 'left'})

        wall_right.initialize()
        wall_left.initialize()

        assert wall_right.wall_side == 'right'
        assert wall_left.wall_side == 'left'

        # Different side should produce different direction orders
        wall_right.current_direction = 'up'
        wall_left.current_direction = 'up'

        right_order = wall_right._get_right_wall_following_order()
        left_order = wall_left._get_left_wall_following_order()

        assert right_order != left_order

        # Test random walk with and without bias
        random_biased = RandomWalkPattern({'bias_unvisited': True})
        random_pure = RandomWalkPattern({'bias_unvisited': False})

        random_biased.initialize()
        random_pure.initialize()

        assert random_biased.bias_towards_unvisited is True
        assert random_pure.bias_towards_unvisited is False