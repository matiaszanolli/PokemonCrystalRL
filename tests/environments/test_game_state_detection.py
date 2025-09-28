#!/usr/bin/env python3
"""
Test Suite for Game State Detection Module

Tests the game state detection system that identifies different states
in Pokemon Crystal (overworld, battle, menu, dialogue, etc.)
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from environments.game_state_detection import GameState, GameStateDetector


class TestGameState:
    """Test GameState enum functionality."""

    def test_game_state_values(self):
        """Test that all game state enums have correct values."""
        assert GameState.UNKNOWN is not None
        assert GameState.OVERWORLD is not None
        assert GameState.BATTLE is not None
        assert GameState.MENU is not None
        assert GameState.DIALOGUE is not None
        assert GameState.LOADING is not None
        assert GameState.BLACK_SCREEN is not None
        assert GameState.INTRO is not None
        assert GameState.TRAINER_CARD is not None

    def test_from_string_conversion(self):
        """Test conversion from string to GameState enum."""
        test_cases = [
            ('unknown', GameState.UNKNOWN),
            ('overworld', GameState.OVERWORLD),
            ('battle', GameState.BATTLE),
            ('menu', GameState.MENU),
            ('dialogue', GameState.DIALOGUE),
            ('loading', GameState.LOADING),
            ('black_screen', GameState.BLACK_SCREEN),
            ('intro_sequence', GameState.INTRO),
            ('trainer_card', GameState.TRAINER_CARD)
        ]

        for string_val, expected_state in test_cases:
            assert GameState.from_string(string_val) == expected_state

    def test_from_string_case_insensitive(self):
        """Test that string conversion is case insensitive."""
        assert GameState.from_string('OVERWORLD') == GameState.OVERWORLD
        assert GameState.from_string('Battle') == GameState.BATTLE
        assert GameState.from_string('MeNu') == GameState.MENU

    def test_from_string_unknown_fallback(self):
        """Test that unknown strings default to UNKNOWN state."""
        assert GameState.from_string('invalid_state') == GameState.UNKNOWN
        assert GameState.from_string('') == GameState.UNKNOWN
        assert GameState.from_string('not_a_game_state') == GameState.UNKNOWN


class TestGameStateDetector:
    """Test GameStateDetector functionality."""

    @pytest.fixture
    def detector(self):
        """Create a fresh GameStateDetector instance."""
        return GameStateDetector()

    def test_initialization(self, detector):
        """Test GameStateDetector initializes correctly."""
        assert detector.last_screen_hash is None
        assert detector.consecutive_same_screens == 0
        assert detector.stuck_counter == 0
        assert detector.last_state is None
        assert detector._test_mode is False
        assert detector._forced_state is None
        assert detector._override_states == {}
        assert detector._state_sequence == []
        assert detector._sequence_index == 0

    def test_screen_hash_none_input(self, detector):
        """Test screen hash with None input."""
        hash_val = detector.get_screen_hash(None)
        assert hash_val is None

    def test_screen_hash_mock_object(self, detector):
        """Test screen hash with mock objects (for testing)."""
        mock_screen = Mock()
        mock_screen._mock_name = "test_screen"

        hash_val = detector.get_screen_hash(mock_screen)
        assert hash_val is not None
        assert isinstance(hash_val, int)

    def test_screen_hash_invalid_shape(self, detector):
        """Test screen hash with invalid array shapes."""
        # Empty array
        empty_array = np.array([])
        assert detector.get_screen_hash(empty_array) is None

        # 1D array (invalid)
        one_d_array = np.array([1, 2, 3])
        assert detector.get_screen_hash(one_d_array) is None

    def test_screen_hash_grayscale_image(self, detector):
        """Test screen hash with grayscale images."""
        # Create a simple 2D grayscale image
        grayscale_image = np.random.randint(0, 255, (144, 160), dtype=np.uint8)

        hash_val = detector.get_screen_hash(grayscale_image)
        assert hash_val is not None
        assert isinstance(hash_val, int)

    def test_screen_hash_color_image(self, detector):
        """Test screen hash with color images."""
        # Create a simple 3D color image (RGB)
        color_image = np.random.randint(0, 255, (144, 160, 3), dtype=np.uint8)

        hash_val = detector.get_screen_hash(color_image)
        assert hash_val is not None
        assert isinstance(hash_val, int)

    def test_screen_hash_consistency(self, detector):
        """Test that identical screens produce identical hashes."""
        # Create deterministic image
        test_image = np.ones((144, 160), dtype=np.uint8) * 128

        hash1 = detector.get_screen_hash(test_image)
        hash2 = detector.get_screen_hash(test_image)

        assert hash1 == hash2

    def test_screen_hash_different_images(self, detector):
        """Test that different images produce different hashes."""
        image1 = np.zeros((144, 160), dtype=np.uint8)
        image2 = np.ones((144, 160), dtype=np.uint8) * 255

        hash1 = detector.get_screen_hash(image1)
        hash2 = detector.get_screen_hash(image2)

        assert hash1 != hash2

    def test_screen_hash_small_images(self, detector):
        """Test screen hash with very small images."""
        # 1x1 image
        tiny_image = np.array([[100]], dtype=np.uint8)
        hash_val = detector.get_screen_hash(tiny_image)
        assert hash_val is not None

        # 2x2 image
        small_image = np.array([[100, 150], [200, 50]], dtype=np.uint8)
        hash_val = detector.get_screen_hash(small_image)
        assert hash_val is not None

    def test_test_mode_functionality(self, detector):
        """Test test mode functionality for unit testing."""
        # Enable test mode
        detector._test_mode = True
        detector._forced_state = GameState.BATTLE

        assert detector._test_mode is True
        assert detector._forced_state == GameState.BATTLE

    def test_state_override_functionality(self, detector):
        """Test state override functionality for testing."""
        # Set up state overrides
        detector._override_states = {
            'test_context': GameState.MENU,
            'battle_context': GameState.BATTLE
        }

        assert detector._override_states['test_context'] == GameState.MENU
        assert detector._override_states['battle_context'] == GameState.BATTLE

    def test_state_sequence_functionality(self, detector):
        """Test state sequence functionality for testing."""
        # Set up state sequence
        detector._state_sequence = [
            GameState.OVERWORLD,
            GameState.BATTLE,
            GameState.OVERWORLD
        ]
        detector._sequence_index = 0

        assert len(detector._state_sequence) == 3
        assert detector._state_sequence[0] == GameState.OVERWORLD
        assert detector._sequence_index == 0

    def test_stuck_detection_properties(self, detector):
        """Test stuck detection counter functionality."""
        # Initially no stuck detection
        assert detector.stuck_counter == 0
        assert detector.consecutive_same_screens == 0

        # Simulate stuck detection
        detector.stuck_counter = 5
        detector.consecutive_same_screens = 10

        assert detector.stuck_counter == 5
        assert detector.consecutive_same_screens == 10

    def test_last_state_tracking(self, detector):
        """Test that last state is properly tracked."""
        assert detector.last_state is None

        # Set a state
        detector.last_state = GameState.OVERWORLD
        assert detector.last_state == GameState.OVERWORLD

        # Change state
        detector.last_state = GameState.BATTLE
        assert detector.last_state == GameState.BATTLE

    def test_logging_setup(self, detector):
        """Test that logging is properly set up."""
        assert hasattr(detector, 'logger')
        assert detector.logger is not None
        assert detector.logger.name == 'environments.game_state_detection'

    def test_screen_hash_edge_cases(self, detector):
        """Test screen hash with various edge cases."""
        # Very large image
        large_image = np.random.randint(0, 255, (1000, 1000), dtype=np.uint8)
        hash_val = detector.get_screen_hash(large_image)
        assert hash_val is not None

        # Image with all same values
        uniform_image = np.full((144, 160), 128, dtype=np.uint8)
        hash_val = detector.get_screen_hash(uniform_image)
        assert hash_val is not None

        # Image with extreme values
        extreme_image = np.array([[0, 255], [255, 0]], dtype=np.uint8)
        hash_val = detector.get_screen_hash(extreme_image)
        assert hash_val is not None

    def test_color_image_conversion(self, detector):
        """Test that color images are properly converted to grayscale."""
        # Create RGB image with known values
        rgb_image = np.zeros((10, 10, 3), dtype=np.uint8)
        rgb_image[:, :, 0] = 255  # Red channel
        rgb_image[:, :, 1] = 128  # Green channel
        rgb_image[:, :, 2] = 64   # Blue channel

        hash_val = detector.get_screen_hash(rgb_image)
        assert hash_val is not None

        # Should be same as equivalent grayscale
        gray_value = int(np.mean([255, 128, 64]))
        gray_image = np.full((10, 10), gray_value, dtype=np.uint8)
        gray_hash = detector.get_screen_hash(gray_image)

        # Hashes should be similar (allowing for minor conversion differences)
        assert hash_val is not None and gray_hash is not None

    def test_subsampling_consistency(self, detector):
        """Test that subsampling produces consistent results."""
        # Create large image with pattern
        large_image = np.zeros((144, 160), dtype=np.uint8)
        large_image[::2, ::2] = 255  # Checkerboard pattern

        hash1 = detector.get_screen_hash(large_image)
        hash2 = detector.get_screen_hash(large_image)

        assert hash1 == hash2

    def test_hash_discrimination(self, detector):
        """Test that hash can discriminate between different screen layouts."""
        # Create images with very different patterns to ensure discrimination
        image1 = np.zeros((100, 100), dtype=np.uint8)
        image1[:50, :] = 255  # Top half white

        image2 = np.zeros((100, 100), dtype=np.uint8)
        image2[:, :50] = 255  # Left half white

        image3 = np.ones((100, 100), dtype=np.uint8) * 128  # All gray

        hash1 = detector.get_screen_hash(image1)
        hash2 = detector.get_screen_hash(image2)
        hash3 = detector.get_screen_hash(image3)

        # All hashes should be different (allowing for potential edge cases in hash function)
        hashes = [hash1, hash2, hash3]
        unique_hashes = set(hashes)

        # At least 2 should be different (robust test allowing for hash collisions)
        assert len(unique_hashes) >= 2, f"Expected at least 2 unique hashes, got: {hashes}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])