#!/usr/bin/env python3
"""
Test Suite for PyBoy Environment

Tests the basic PyBoy-based environment wrapper for Pokemon Crystal RL training.
"""

import pytest
import tempfile
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import gymnasium as gym
from gymnasium import spaces
import collections

from environments.pyboy_env import PyBoyPokemonCrystalEnv


class TestPyBoyEnvironment:
    """Test PyBoyPokemonCrystalEnv functionality."""

    @pytest.fixture
    def temp_rom_path(self):
        """Create temporary ROM file for testing."""
        with tempfile.NamedTemporaryFile(suffix='.gbc', delete=False) as f:
            f.write(b"fake_rom_data" * 1000)
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
            'debug_mode': False,
            'enable_monitoring': False,
            'monitor_server_url': "http://localhost:5000"
        }

    @patch('environments.pyboy_env.PyBoy')
    @patch('environments.pyboy_env.MonitoringClient')
    def test_initialization(self, mock_monitoring, mock_pyboy, env_config):
        """Test environment initializes correctly."""
        env = PyBoyPokemonCrystalEnv(**env_config)

        assert env.rom_path == env_config['rom_path']
        assert env.save_state_path == env_config['save_state_path']
        assert env.max_steps == env_config['max_steps']
        assert env.render_mode == env_config['render_mode']
        assert env.headless == env_config['headless']
        assert env.debug_mode == env_config['debug_mode']
        assert env.enable_monitoring == env_config['enable_monitoring']

    @patch('environments.pyboy_env.PyBoy')
    @patch('environments.pyboy_env.MonitoringClient')
    def test_action_space(self, mock_monitoring, mock_pyboy, env_config):
        """Test action space is correctly defined."""
        env = PyBoyPokemonCrystalEnv(**env_config)

        assert isinstance(env.action_space, spaces.Discrete)
        assert env.action_space.n == 9  # 9 actions including no-op

    @patch('environments.pyboy_env.PyBoy')
    @patch('environments.pyboy_env.MonitoringClient')
    def test_observation_space(self, mock_monitoring, mock_pyboy, env_config):
        """Test observation space is correctly defined."""
        env = PyBoyPokemonCrystalEnv(**env_config)

        assert isinstance(env.observation_space, spaces.Box)
        assert env.observation_space.shape == (20,)
        assert env.observation_space.dtype == np.float32
        assert np.all(env.observation_space.low == 0)
        assert np.all(env.observation_space.high == 1)

    @patch('environments.pyboy_env.PyBoy')
    @patch('environments.pyboy_env.MonitoringClient')
    def test_state_tracking_initialization(self, mock_monitoring, mock_pyboy, env_config):
        """Test that state tracking variables are initialized."""
        env = PyBoyPokemonCrystalEnv(**env_config)

        assert env.step_count == 0
        assert env.episode_reward == 0
        assert env.previous_state is None
        assert env.current_state is None
        assert env.episode_number == 0

    @patch('environments.pyboy_env.PyBoy')
    @patch('environments.pyboy_env.MonitoringClient')
    def test_enhanced_state_tracking(self, mock_monitoring, mock_pyboy, env_config):
        """Test enhanced state tracking features."""
        env = PyBoyPokemonCrystalEnv(**env_config)

        assert env.consecutive_same_screens == 0
        assert env.last_screen_hash is None
        assert isinstance(env.recent_actions, collections.deque)
        assert env.recent_actions.maxlen == 10
        assert isinstance(env.game_state_history, collections.deque)
        assert env.game_state_history.maxlen == 5

    @patch('environments.pyboy_env.PyBoy')
    @patch('environments.pyboy_env.MonitoringClient')
    def test_action_mapping(self, mock_monitoring, mock_pyboy, env_config):
        """Test action mapping is correctly defined."""
        env = PyBoyPokemonCrystalEnv(**env_config)

        expected_mapping = {
            0: "NONE", 1: "UP", 2: "DOWN", 3: "LEFT", 4: "RIGHT",
            5: "A", 6: "B", 7: "START", 8: "SELECT"
        }

        assert env.action_map == expected_mapping

    @patch('environments.pyboy_env.PyBoy')
    @patch('environments.pyboy_env.MonitoringClient')
    def test_default_parameters(self, mock_monitoring, mock_pyboy, temp_rom_path):
        """Test environment with default parameters."""
        env = PyBoyPokemonCrystalEnv(rom_path=temp_rom_path)

        # Check default values
        assert env.max_steps == 10000
        assert env.render_mode is None
        assert env.headless is True
        assert env.debug_mode is False
        assert env.enable_monitoring is True

    @patch('environments.pyboy_env.PyBoy')
    @patch('environments.pyboy_env.MonitoringClient')
    def test_monitoring_disabled(self, mock_monitoring, mock_pyboy, env_config):
        """Test environment with monitoring disabled."""
        env_config['enable_monitoring'] = False
        env = PyBoyPokemonCrystalEnv(**env_config)

        assert env.enable_monitoring is False
        assert env.monitor is None
        mock_monitoring.assert_not_called()

    @patch('environments.pyboy_env.PyBoy')
    @patch('environments.pyboy_env.MonitoringClient')
    def test_monitoring_enabled_success(self, mock_monitoring, mock_pyboy, env_config):
        """Test environment with monitoring enabled and successful initialization."""
        env_config['enable_monitoring'] = True
        mock_client = Mock()
        mock_client.is_server_available.return_value = True
        mock_monitoring.return_value = mock_client

        env = PyBoyPokemonCrystalEnv(**env_config)

        assert env.enable_monitoring is True
        assert env.monitor is not None
        mock_monitoring.assert_called_once_with("http://localhost:5000", auto_start=True)

    @patch('environments.pyboy_env.PyBoy')
    @patch('environments.pyboy_env.MonitoringClient')
    def test_monitoring_enabled_failure(self, mock_monitoring, mock_pyboy, env_config):
        """Test environment with monitoring enabled but initialization fails."""
        env_config['enable_monitoring'] = True
        env_config['debug_mode'] = True
        mock_monitoring.side_effect = Exception("Connection failed")

        env = PyBoyPokemonCrystalEnv(**env_config)

        assert env.enable_monitoring is True
        assert env.monitor is None
        mock_monitoring.assert_called_once()

    @patch('environments.pyboy_env.PyBoy')
    @patch('environments.pyboy_env.MonitoringClient')
    def test_file_path_normalization(self, mock_monitoring, mock_pyboy, env_config):
        """Test that file paths are properly normalized."""
        env = PyBoyPokemonCrystalEnv(**env_config)

        assert Path(env.rom_path).is_absolute()
        assert Path(env.save_state_path).is_absolute()

    @patch('environments.pyboy_env.PyBoy')
    @patch('environments.pyboy_env.MonitoringClient')
    def test_no_save_state_handling(self, mock_monitoring, mock_pyboy, temp_rom_path):
        """Test environment handles no save state gracefully."""
        env = PyBoyPokemonCrystalEnv(
            rom_path=temp_rom_path,
            save_state_path=None,
            enable_monitoring=False
        )

        assert env.save_state_path is None

    @patch('environments.pyboy_env.PyBoy')
    @patch('environments.pyboy_env.MonitoringClient')
    def test_render_mode_configuration(self, mock_monitoring, mock_pyboy, env_config):
        """Test different render mode configurations."""
        render_modes = [None, 'human', 'rgb_array']

        for mode in render_modes:
            env_config['render_mode'] = mode
            env = PyBoyPokemonCrystalEnv(**env_config)
            assert env.render_mode == mode

    @patch('environments.pyboy_env.PyBoy')
    @patch('environments.pyboy_env.MonitoringClient')
    def test_debug_mode_configuration(self, mock_monitoring, mock_pyboy, env_config):
        """Test debug mode configuration."""
        # Test debug enabled
        env_config['debug_mode'] = True
        env = PyBoyPokemonCrystalEnv(**env_config)
        assert env.debug_mode is True

        # Test debug disabled
        env_config['debug_mode'] = False
        env = PyBoyPokemonCrystalEnv(**env_config)
        assert env.debug_mode is False

    @patch('environments.pyboy_env.PyBoy')
    @patch('environments.pyboy_env.MonitoringClient')
    def test_pyboy_initialization_state(self, mock_monitoring, mock_pyboy, env_config):
        """Test PyBoy emulator initialization state."""
        env = PyBoyPokemonCrystalEnv(**env_config)

        # PyBoy should not be initialized until needed
        assert env.pyboy is None
        assert env.window_wrapper is None

    @patch('environments.pyboy_env.PyBoy')
    @patch('environments.pyboy_env.MonitoringClient')
    def test_gymnasium_interface_compliance(self, mock_monitoring, mock_pyboy, env_config):
        """Test that environment complies with Gymnasium interface."""
        env = PyBoyPokemonCrystalEnv(**env_config)

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

    @patch('environments.pyboy_env.PyBoy')
    @patch('environments.pyboy_env.MonitoringClient')
    def test_max_steps_configuration(self, mock_monitoring, mock_pyboy, env_config):
        """Test different max steps configurations."""
        max_steps_values = [100, 1000, 10000, 50000]

        for max_steps in max_steps_values:
            env_config['max_steps'] = max_steps
            env = PyBoyPokemonCrystalEnv(**env_config)
            assert env.max_steps == max_steps

    @patch('environments.pyboy_env.PyBoy')
    @patch('environments.pyboy_env.MonitoringClient')
    def test_headless_mode_configuration(self, mock_monitoring, mock_pyboy, env_config):
        """Test headless mode configuration."""
        # Test headless enabled
        env_config['headless'] = True
        env = PyBoyPokemonCrystalEnv(**env_config)
        assert env.headless is True

        # Test headless disabled
        env_config['headless'] = False
        env = PyBoyPokemonCrystalEnv(**env_config)
        assert env.headless is False

    @patch('environments.pyboy_env.PyBoy')
    @patch('environments.pyboy_env.MonitoringClient')
    def test_monitor_server_url_configuration(self, mock_monitoring, mock_pyboy, env_config):
        """Test monitor server URL configuration."""
        urls = ["http://localhost:5000", "http://localhost:8080", "http://example.com:3000"]

        for url in urls:
            env_config['monitor_server_url'] = url
            env_config['enable_monitoring'] = True
            env = PyBoyPokemonCrystalEnv(**env_config)

            if env.monitor:
                mock_monitoring.assert_called_with(url, auto_start=True)

    @patch('environments.pyboy_env.PyBoy')
    @patch('environments.pyboy_env.MonitoringClient')
    def test_recent_actions_deque(self, mock_monitoring, mock_pyboy, env_config):
        """Test recent actions deque functionality."""
        env = PyBoyPokemonCrystalEnv(**env_config)

        # Should be empty initially
        assert len(env.recent_actions) == 0

        # Should have correct maxlen
        assert env.recent_actions.maxlen == 10

        # Test adding actions
        for i in range(15):  # Add more than maxlen
            env.recent_actions.append(i)

        # Should only keep last 10
        assert len(env.recent_actions) == 10
        assert list(env.recent_actions) == list(range(5, 15))

    @patch('environments.pyboy_env.PyBoy')
    @patch('environments.pyboy_env.MonitoringClient')
    def test_game_state_history_deque(self, mock_monitoring, mock_pyboy, env_config):
        """Test game state history deque functionality."""
        env = PyBoyPokemonCrystalEnv(**env_config)

        # Should be empty initially
        assert len(env.game_state_history) == 0

        # Should have correct maxlen
        assert env.game_state_history.maxlen == 5

        # Test adding states
        for i in range(8):  # Add more than maxlen
            env.game_state_history.append({'state': i})

        # Should only keep last 5
        assert len(env.game_state_history) == 5
        expected_states = [{'state': i} for i in range(3, 8)]
        assert list(env.game_state_history) == expected_states

    @patch('environments.pyboy_env.PyBoy')
    @patch('environments.pyboy_env.MonitoringClient')
    def test_screen_hash_tracking(self, mock_monitoring, mock_pyboy, env_config):
        """Test screen hash tracking for stuck detection."""
        env = PyBoyPokemonCrystalEnv(**env_config)

        # Initially no hash
        assert env.last_screen_hash is None
        assert env.consecutive_same_screens == 0

        # Simulate setting hash
        env.last_screen_hash = 12345
        env.consecutive_same_screens = 3

        assert env.last_screen_hash == 12345
        assert env.consecutive_same_screens == 3


class TestActionSpace:
    """Test action space specific functionality."""

    @pytest.fixture
    def temp_rom_path(self):
        """Create temporary ROM file for testing."""
        with tempfile.NamedTemporaryFile(suffix='.gbc', delete=False) as f:
            f.write(b"fake_rom_data" * 1000)
            return f.name

    @patch('environments.pyboy_env.PyBoy')
    @patch('environments.pyboy_env.MonitoringClient')
    def test_action_space_sampling(self, mock_monitoring, mock_pyboy, temp_rom_path):
        """Test that action space can sample valid actions."""
        env = PyBoyPokemonCrystalEnv(rom_path=temp_rom_path, enable_monitoring=False)

        # Sample multiple actions
        for _ in range(100):
            action = env.action_space.sample()
            assert 0 <= action < 9
            assert action in env.action_map

    @patch('environments.pyboy_env.PyBoy')
    @patch('environments.pyboy_env.MonitoringClient')
    def test_action_mapping_completeness(self, mock_monitoring, mock_pyboy, temp_rom_path):
        """Test that action mapping covers all possible actions."""
        env = PyBoyPokemonCrystalEnv(rom_path=temp_rom_path, enable_monitoring=False)

        # All actions in range should be mapped
        for action in range(9):
            assert action in env.action_map
            assert isinstance(env.action_map[action], str)
            assert len(env.action_map[action]) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])