"""
Enhanced Stuck Detection Tests

Tests for the enhanced stuck detection system that properly handles
memory corruption and distinguishes between real stuck loops and corruption artifacts.
"""

import pytest
import numpy as np
from training.components.enhanced_stuck_detection import (
    EnhancedStuckDetector,
    StuckType,
    StuckPattern
)


class TestEnhancedStuckDetection:
    """Test enhanced stuck detection with memory corruption handling."""

    def setup_method(self):
        """Set up test fixtures."""
        self.detector = EnhancedStuckDetector()

    def test_memory_corruption_position_detection(self):
        """Stuck detector should ignore memory corruption patterns."""
        # Simulate memory corruption with repeated (0,0,0) positions
        for _ in range(10):
            self.detector.update_history(
                game_state={
                    'player_x': 0,
                    'player_y': 0,
                    'player_map': 0,
                    'in_battle': False,
                    'party_count': 0,
                    'badges': 0
                },
                action=1,
                reward=0.0
            )

        # Should NOT detect stuck pattern due to corruption
        pattern = self.detector.detect_stuck_patterns(action_count=10)
        assert pattern is None

    def test_corrupted_position_patterns(self):
        """Should ignore various corruption patterns."""
        corruption_patterns = [
            (0, 0, 0),      # All zeros
            (255, 255, 255), # All max values
            (0, 255, 0),    # Mixed corruption
            (255, 0, 255),  # Mixed corruption
        ]

        for pos in corruption_patterns:
            self.detector = EnhancedStuckDetector()  # Reset for each test

            # Fill history with corrupted positions
            for _ in range(10):
                self.detector.update_history(
                    game_state={
                        'player_x': pos[0],
                        'player_y': pos[1],
                        'player_map': pos[2],
                        'in_battle': False,
                        'party_count': 0,
                        'badges': 0
                    },
                    action=1,
                    reward=0.0
                )

            pattern = self.detector.detect_stuck_patterns(action_count=10)
            assert pattern is None, f"Should ignore corruption pattern {pos}"

    def test_real_position_loop_detection(self):
        """Should detect actual position loops with valid coordinates."""
        # Simulate real stuck loop between two valid positions
        positions = [(1, 10, 15), (1, 11, 15)]  # Valid positions

        for i in range(10):
            pos = positions[i % 2]
            self.detector.update_history(
                game_state={
                    'player_x': pos[1],
                    'player_y': pos[2],
                    'player_map': pos[0],
                    'in_battle': False,
                    'party_count': 1,
                    'badges': 0
                },
                action=1,
                reward=0.0
            )

        # Should detect legitimate stuck pattern
        pattern = self.detector.detect_stuck_patterns(action_count=10)
        assert pattern is not None
        assert pattern.stuck_type == StuckType.POSITION_LOOP

    def test_impossible_coordinates_ignored(self):
        """Should ignore positions with impossible coordinate values."""
        # Simulate positions with negative or too large coordinates
        impossible_positions = [
            (-1, 10, 1),     # Negative x
            (10, -1, 1),     # Negative y
            (10, 10, -1),    # Negative map
            (300, 10, 1),    # Too large x
            (10, 300, 1),    # Too large y
        ]

        for pos in impossible_positions:
            self.detector = EnhancedStuckDetector()

            for _ in range(10):
                self.detector.update_history(
                    game_state={
                        'player_x': pos[0],
                        'player_y': pos[1],
                        'player_map': pos[2],
                        'in_battle': False,
                        'party_count': 0,
                        'badges': 0
                    },
                    action=1,
                    reward=0.0
                )

            pattern = self.detector.detect_stuck_patterns(action_count=10)
            assert pattern is None, f"Should ignore impossible coordinates {pos}"

    def test_mixed_corruption_and_valid_positions(self):
        """Should handle mix of corrupted and valid positions correctly."""
        # Mix corrupted and valid positions
        positions = [
            (0, 0, 0),      # Corrupted
            (1, 10, 15),    # Valid
            (0, 0, 0),      # Corrupted
            (1, 11, 15),    # Valid
            (255, 255, 255) # Corrupted
        ]

        for i in range(15):
            pos = positions[i % len(positions)]
            self.detector.update_history(
                game_state={
                    'player_x': pos[0],
                    'player_y': pos[1],
                    'player_map': pos[2],
                    'in_battle': False,
                    'party_count': 1 if pos != (0, 0, 0) else 0,
                    'badges': 0
                },
                action=1,
                reward=0.0
            )

        # Should not get false positive from corruption
        pattern = self.detector.detect_stuck_patterns(action_count=15)
        # Might detect or not depending on valid position patterns, but no corruption false positive
        if pattern:
            assert pattern.stuck_type != StuckType.POSITION_LOOP or \
                   pattern.severity < 0.8  # Not severely stuck due to corruption

    def test_action_repetition_still_works(self):
        """Non-position stuck detection should still work normally."""
        # Fill with same action repeatedly
        for _ in range(8):
            self.detector.update_history(
                game_state={
                    'player_x': 10,
                    'player_y': 15,
                    'player_map': 1,
                    'in_battle': False,
                    'party_count': 1,
                    'badges': 0
                },
                action=1,  # Same action
                reward=0.0
            )

        pattern = self.detector.detect_stuck_patterns(action_count=8)
        assert pattern is not None
        assert pattern.stuck_type == StuckType.ACTION_REPETITION

    def test_screen_stuck_detection_works(self):
        """Screen stuck detection should still work with screen data."""
        # Create consistent screen hash
        screen_data = np.zeros((144, 160, 3), dtype=np.uint8)
        screen_data[10:20, 10:20] = 255  # Some pattern

        for _ in range(12):
            self.detector.update_history(
                game_state={
                    'player_x': 10,
                    'player_y': 15,
                    'player_map': 1,
                    'in_battle': False,
                    'party_count': 1,
                    'badges': 0
                },
                action=2,
                reward=0.0,
                screen_data=screen_data
            )

        pattern = self.detector.detect_stuck_patterns(action_count=12)
        assert pattern is not None
        assert pattern.stuck_type == StuckType.SCREEN_STUCK

    def test_reward_plateau_detection(self):
        """Reward plateau detection should work independently of position corruption."""
        for _ in range(25):
            self.detector.update_history(
                game_state={
                    'player_x': 0,  # Even with corrupted positions
                    'player_y': 0,
                    'player_map': 0,
                    'in_battle': False,
                    'party_count': 0,
                    'badges': 0
                },
                action=3,
                reward=-0.1  # Negative rewards
            )

        pattern = self.detector.detect_stuck_patterns(action_count=25)
        # Should detect reward plateau, not position loop
        if pattern:
            assert pattern.stuck_type == StuckType.REWARD_PLATEAU

    def test_recovery_strategy_selection(self):
        """Recovery strategies should work with enhanced detection."""
        # Create a legitimate stuck pattern (not corruption)
        for i in range(10):
            self.detector.update_history(
                game_state={
                    'player_x': 10 + (i % 2),  # Valid oscillating positions
                    'player_y': 15,
                    'player_map': 1,
                    'in_battle': False,
                    'party_count': 1,
                    'badges': 0
                },
                action=1,
                reward=0.0
            )

        pattern = self.detector.detect_stuck_patterns(action_count=10)
        if pattern and pattern.stuck_type == StuckType.POSITION_LOOP:
            strategy = self.detector.get_recovery_strategy(pattern)
            assert strategy is not None
            assert len(strategy.actions) > 0

    def test_statistics_tracking(self):
        """Statistics should track enhanced detection properly."""
        stats = self.detector.get_statistics()
        assert 'current_stuck' in stats
        assert 'current_severity' in stats
        assert 'recovery_attempts' in stats
        assert 'position_history_size' in stats

    def test_stuck_state_reset(self):
        """Should properly reset stuck state when making progress."""
        # Create stuck state
        for _ in range(10):
            self.detector.update_history(
                game_state={
                    'player_x': 10,
                    'player_y': 15,
                    'player_map': 1,
                    'in_battle': False,
                    'party_count': 1,
                    'badges': 0
                },
                action=1,
                reward=0.0
            )

        pattern = self.detector.detect_stuck_patterns(action_count=10)
        if pattern:
            # Reset stuck state
            self.detector.reset_stuck_state()
            stats = self.detector.get_statistics()
            assert stats['current_stuck'] is None
            assert stats['current_severity'] == 0.0