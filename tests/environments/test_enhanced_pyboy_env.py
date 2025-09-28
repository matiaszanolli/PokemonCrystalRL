#!/usr/bin/env python3
"""
Test Suite for Enhanced PyBoy Environment

Tests the enhanced Gymnasium environment with multi-modal observations,
action masking, and strategic integration for Pokemon Crystal RL.
"""

import pytest
import tempfile
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import gymnasium as gym
from gymnasium import spaces

from environments.enhanced_pyboy_env import EnhancedPyBoyPokemonCrystalEnv


class TestEnhancedPyBoyEnvironment:
    """Test EnhancedPyBoyPokemonCrystalEnv functionality."""

    @pytest.fixture
    def temp_rom_path(self):
        """Create temporary ROM file for testing."""
        with tempfile.NamedTemporaryFile(suffix='.gbc', delete=False) as f:
            f.write(b"fake_rom_data" * 1000)  # Make it reasonably sized
            return f.name

    @pytest.fixture
    def temp_save_state_path(self):
        """Create temporary save state file for testing."""
        with tempfile.NamedTemporaryFile(suffix='.ss1', delete=False) as f:
            f.write(b"fake_save_state_data" * 100)
            return f.name

    @pytest.fixture
    def env_config(self, temp_rom_path, temp_save_state_path):
        """Basic environment configuration for testing."""
        return {
            'rom_path': temp_rom_path,
            'save_state_path': temp_save_state_path,
            'max_steps': 100,
            'render_mode': None,
            'headless': True,
            'screen_size': (160, 144),
            'enable_action_masking': True,
            'enable_strategic_context': True,
            'history_window': 5,
            'observation_type': 'multi_modal'
        }

    @patch('environments.enhanced_pyboy_env.PyBoy')
    @patch('environments.enhanced_pyboy_env.StrategicContextBuilder')
    @patch('environments.enhanced_pyboy_env.GameStateAnalyzer')
    def test_initialization(self, mock_analyzer, mock_strategic, mock_pyboy, env_config):
        """Test environment initializes correctly."""
        env = EnhancedPyBoyPokemonCrystalEnv(**env_config)

        assert env.rom_path == env_config['rom_path']
        assert env.save_state_path == env_config['save_state_path']
        assert env.max_steps == env_config['max_steps']
        assert env.headless == env_config['headless']
        assert env.enable_action_masking == env_config['enable_action_masking']
        assert env.enable_strategic_context == env_config['enable_strategic_context']
        assert env.history_window == env_config['history_window']
        assert env.observation_type == env_config['observation_type']

    @patch('environments.enhanced_pyboy_env.PyBoy')
    @patch('environments.enhanced_pyboy_env.StrategicContextBuilder')
    @patch('environments.enhanced_pyboy_env.GameStateAnalyzer')
    def test_action_space(self, mock_analyzer, mock_strategic, mock_pyboy, env_config):
        """Test action space is correctly defined."""
        env = EnhancedPyBoyPokemonCrystalEnv(**env_config)

        assert isinstance(env.action_space, spaces.Discrete)
        assert env.action_space.n == 9  # 9 actions including no-op

    @patch('environments.enhanced_pyboy_env.PyBoy')
    @patch('environments.enhanced_pyboy_env.StrategicContextBuilder')
    @patch('environments.enhanced_pyboy_env.GameStateAnalyzer')
    def test_observation_space_setup(self, mock_analyzer, mock_strategic, mock_pyboy, env_config):
        """Test observation space is correctly set up."""
        env = EnhancedPyBoyPokemonCrystalEnv(**env_config)

        assert hasattr(env, 'observation_space')
        assert env.observation_space is not None

    @patch('environments.enhanced_pyboy_env.PyBoy')
    @patch('environments.enhanced_pyboy_env.StrategicContextBuilder')
    @patch('environments.enhanced_pyboy_env.GameStateAnalyzer')
    def test_component_initialization(self, mock_analyzer, mock_strategic, mock_pyboy, env_config):
        """Test that core components are initialized."""
        env = EnhancedPyBoyPokemonCrystalEnv(**env_config)

        # Core components should be initialized
        assert hasattr(env, 'game_state_analyzer')
        assert hasattr(env, 'strategic_context')

        # Strategic context should be created when enabled
        if env_config['enable_strategic_context']:
            mock_strategic.assert_called_once()
        else:
            mock_strategic.assert_not_called()

    @patch('environments.enhanced_pyboy_env.PyBoy')
    @patch('environments.enhanced_pyboy_env.StrategicContextBuilder')
    @patch('environments.enhanced_pyboy_env.GameStateAnalyzer')
    def test_state_tracking_initialization(self, mock_analyzer, mock_strategic, mock_pyboy, env_config):
        """Test that state tracking variables are initialized."""
        env = EnhancedPyBoyPokemonCrystalEnv(**env_config)

        assert env.step_count == 0
        assert env.episode_reward == 0
        assert env.last_game_state is None
        assert env.stuck_counter == 0
        assert isinstance(env.progress_milestones, set)
        assert len(env.progress_milestones) == 0

    @patch('environments.enhanced_pyboy_env.PyBoy')
    @patch('environments.enhanced_pyboy_env.StrategicContextBuilder')
    @patch('environments.enhanced_pyboy_env.GameStateAnalyzer')
    def test_history_tracking_initialization(self, mock_analyzer, mock_strategic, mock_pyboy, env_config):
        """Test that history tracking is properly initialized."""
        env = EnhancedPyBoyPokemonCrystalEnv(**env_config)

        assert hasattr(env, 'action_history')
        assert hasattr(env, 'observation_history')
        assert env.action_history.maxlen == env_config['history_window']
        assert env.observation_history.maxlen == env_config['history_window']
        assert len(env.action_history) == 0
        assert len(env.observation_history) == 0

    @patch('environments.enhanced_pyboy_env.PyBoy')
    @patch('environments.enhanced_pyboy_env.StrategicContextBuilder')
    @patch('environments.enhanced_pyboy_env.GameStateAnalyzer')
    def test_default_parameters(self, mock_analyzer, mock_strategic, mock_pyboy, temp_rom_path):
        """Test environment with default parameters."""
        env = EnhancedPyBoyPokemonCrystalEnv(rom_path=temp_rom_path)

        # Check default values
        assert env.max_steps == 10000
        assert env.headless is True
        assert env.screen_size == (160, 144)
        assert env.enable_action_masking is True
        assert env.enable_strategic_context is True
        assert env.history_window == 10
        assert env.observation_type == "multi_modal"

    @patch('environments.enhanced_pyboy_env.PyBoy')
    @patch('environments.enhanced_pyboy_env.StrategicContextBuilder')
    @patch('environments.enhanced_pyboy_env.GameStateAnalyzer')
    def test_strategic_context_disabled(self, mock_analyzer, mock_strategic, mock_pyboy, env_config):
        """Test environment with strategic context disabled."""
        env_config['enable_strategic_context'] = False
        env = EnhancedPyBoyPokemonCrystalEnv(**env_config)

        assert env.strategic_context is None
        mock_strategic.assert_not_called()

    @patch('environments.enhanced_pyboy_env.PyBoy')
    @patch('environments.enhanced_pyboy_env.StrategicContextBuilder')
    @patch('environments.enhanced_pyboy_env.GameStateAnalyzer')
    def test_observation_type_variations(self, mock_analyzer, mock_strategic, mock_pyboy, env_config):
        """Test different observation type configurations."""
        observation_types = ['multi_modal', 'state_only', 'screen_only', 'combined']

        for obs_type in observation_types:
            env_config['observation_type'] = obs_type
            env = EnhancedPyBoyPokemonCrystalEnv(**env_config)
            assert env.observation_type == obs_type

    @patch('environments.enhanced_pyboy_env.PyBoy')
    @patch('environments.enhanced_pyboy_env.StrategicContextBuilder')
    @patch('environments.enhanced_pyboy_env.GameStateAnalyzer')
    def test_screen_size_configuration(self, mock_analyzer, mock_strategic, mock_pyboy, env_config):
        """Test different screen size configurations."""
        screen_sizes = [(160, 144), (320, 288), (80, 72)]

        for size in screen_sizes:
            env_config['screen_size'] = size
            env = EnhancedPyBoyPokemonCrystalEnv(**env_config)
            assert env.screen_size == size

    @patch('environments.enhanced_pyboy_env.PyBoy')
    @patch('environments.enhanced_pyboy_env.StrategicContextBuilder')
    @patch('environments.enhanced_pyboy_env.GameStateAnalyzer')
    def test_history_window_configuration(self, mock_analyzer, mock_strategic, mock_pyboy, env_config):
        """Test different history window sizes."""
        window_sizes = [1, 5, 10, 20]

        for window_size in window_sizes:
            env_config['history_window'] = window_size
            env = EnhancedPyBoyPokemonCrystalEnv(**env_config)
            assert env.history_window == window_size
            assert env.action_history.maxlen == window_size
            assert env.observation_history.maxlen == window_size

    @patch('environments.enhanced_pyboy_env.PyBoy')
    @patch('environments.enhanced_pyboy_env.StrategicContextBuilder')
    @patch('environments.enhanced_pyboy_env.GameStateAnalyzer')
    def test_logging_setup(self, mock_analyzer, mock_strategic, mock_pyboy, env_config):
        """Test that logging is properly set up."""
        env = EnhancedPyBoyPokemonCrystalEnv(**env_config)

        assert hasattr(env, 'logger')
        assert env.logger is not None
        assert env.logger.name == "pokemon_trainer.enhanced_env"

    @patch('environments.enhanced_pyboy_env.PyBoy')
    @patch('environments.enhanced_pyboy_env.StrategicContextBuilder')
    @patch('environments.enhanced_pyboy_env.GameStateAnalyzer')
    def test_file_path_normalization(self, mock_analyzer, mock_strategic, mock_pyboy, env_config):
        """Test that file paths are properly normalized."""
        env = EnhancedPyBoyPokemonCrystalEnv(**env_config)

        assert Path(env.rom_path).is_absolute()
        assert Path(env.save_state_path).is_absolute()

    @patch('environments.enhanced_pyboy_env.PyBoy')
    @patch('environments.enhanced_pyboy_env.StrategicContextBuilder')
    @patch('environments.enhanced_pyboy_env.GameStateAnalyzer')
    def test_no_save_state_handling(self, mock_analyzer, mock_strategic, mock_pyboy, temp_rom_path):
        """Test environment handles no save state gracefully."""
        env = EnhancedPyBoyPokemonCrystalEnv(
            rom_path=temp_rom_path,
            save_state_path=None
        )

        assert env.save_state_path is None

    @patch('environments.enhanced_pyboy_env.PyBoy')
    @patch('environments.enhanced_pyboy_env.StrategicContextBuilder')
    @patch('environments.enhanced_pyboy_env.GameStateAnalyzer')
    def test_render_mode_configuration(self, mock_analyzer, mock_strategic, mock_pyboy, env_config):
        """Test different render mode configurations."""
        render_modes = [None, 'human', 'rgb_array']

        for mode in render_modes:
            env_config['render_mode'] = mode
            env = EnhancedPyBoyPokemonCrystalEnv(**env_config)
            assert env.render_mode == mode

    @patch('environments.enhanced_pyboy_env.PyBoy')
    @patch('environments.enhanced_pyboy_env.StrategicContextBuilder')
    @patch('environments.enhanced_pyboy_env.GameStateAnalyzer')
    def test_action_masking_configuration(self, mock_analyzer, mock_strategic, mock_pyboy, env_config):
        """Test action masking can be enabled/disabled."""
        # Test enabled
        env_config['enable_action_masking'] = True
        env = EnhancedPyBoyPokemonCrystalEnv(**env_config)
        assert env.enable_action_masking is True

        # Test disabled
        env_config['enable_action_masking'] = False
        env = EnhancedPyBoyPokemonCrystalEnv(**env_config)
        assert env.enable_action_masking is False

    @patch('environments.enhanced_pyboy_env.PyBoy')
    @patch('environments.enhanced_pyboy_env.StrategicContextBuilder')
    @patch('environments.enhanced_pyboy_env.GameStateAnalyzer')
    def test_pyboy_initialization_state(self, mock_analyzer, mock_strategic, mock_pyboy, env_config):
        """Test PyBoy emulator initialization state."""
        env = EnhancedPyBoyPokemonCrystalEnv(**env_config)

        # PyBoy should not be initialized until needed
        assert env.pyboy is None
        assert env.window_wrapper is None

    @patch('environments.enhanced_pyboy_env.PyBoy')
    @patch('environments.enhanced_pyboy_env.StrategicContextBuilder')
    @patch('environments.enhanced_pyboy_env.GameStateAnalyzer')
    def test_gymnasium_interface_compliance(self, mock_analyzer, mock_strategic, mock_pyboy, env_config):
        """Test that environment complies with Gymnasium interface."""
        env = EnhancedPyBoyPokemonCrystalEnv(**env_config)

        # Should inherit from gym.Env
        assert isinstance(env, gym.Env)

        # Should have required attributes
        assert hasattr(env, 'action_space')
        assert hasattr(env, 'observation_space')

        # Action space should be valid
        assert isinstance(env.action_space, spaces.Space)
        assert hasattr(env.action_space, 'sample')

        # Observation space should be valid
        assert isinstance(env.observation_space, spaces.Space)

    @patch('environments.enhanced_pyboy_env.PyBoy')
    @patch('environments.enhanced_pyboy_env.StrategicContextBuilder')
    @patch('environments.enhanced_pyboy_env.GameStateAnalyzer')
    def test_max_steps_configuration(self, mock_analyzer, mock_strategic, mock_pyboy, env_config):
        """Test different max steps configurations."""
        max_steps_values = [100, 1000, 10000, 50000]

        for max_steps in max_steps_values:
            env_config['max_steps'] = max_steps
            env = EnhancedPyBoyPokemonCrystalEnv(**env_config)
            assert env.max_steps == max_steps

    @patch('environments.enhanced_pyboy_env.PyBoy')
    @patch('environments.enhanced_pyboy_env.StrategicContextBuilder')
    @patch('environments.enhanced_pyboy_env.GameStateAnalyzer')
    def test_headless_mode_configuration(self, mock_analyzer, mock_strategic, mock_pyboy, env_config):
        """Test headless mode configuration."""
        # Test headless enabled
        env_config['headless'] = True
        env = EnhancedPyBoyPokemonCrystalEnv(**env_config)
        assert env.headless is True

        # Test headless disabled
        env_config['headless'] = False
        env = EnhancedPyBoyPokemonCrystalEnv(**env_config)
        assert env.headless is False


class TestObservationSpaceSetup:
    """Test observation space setup functionality."""

    @pytest.fixture
    def temp_rom_path(self):
        """Create temporary ROM file for testing."""
        with tempfile.NamedTemporaryFile(suffix='.gbc', delete=False) as f:
            f.write(b"fake_rom_data" * 1000)
            return f.name

    @patch('environments.enhanced_pyboy_env.PyBoy')
    @patch('environments.enhanced_pyboy_env.StrategicContextBuilder')
    @patch('environments.enhanced_pyboy_env.GameStateAnalyzer')
    def test_observation_space_creation(self, mock_analyzer, mock_strategic, mock_pyboy, temp_rom_path):
        """Test that observation space is created during initialization."""
        env = EnhancedPyBoyPokemonCrystalEnv(rom_path=temp_rom_path)

        # _setup_observation_space should be called during init
        assert hasattr(env, 'observation_space')
        assert env.observation_space is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])