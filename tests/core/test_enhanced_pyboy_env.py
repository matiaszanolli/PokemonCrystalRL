"""
Tests for the Enhanced PyBoy Pokemon Crystal Environment.
"""

import unittest
import tempfile
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from core.enhanced_pyboy_env import EnhancedPyBoyPokemonCrystalEnv


class TestEnhancedPyBoyEnvironment(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures with mocked PyBoy."""
        self.temp_dir = tempfile.mkdtemp()
        self.rom_path = Path(self.temp_dir) / "test.gbc"
        
        # Create dummy ROM file
        with open(self.rom_path, 'wb') as f:
            f.write(b'\x00' * 1024)  # Dummy ROM content
    
    def tearDown(self):
        """Clean up test resources."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    @patch('core.enhanced_pyboy_env.PyBoy')
    @patch('core.enhanced_pyboy_env.StrategicContextBuilder')
    def test_environment_initialization(self, mock_context_builder, mock_pyboy):
        """Test environment initialization with mocked dependencies."""
        # Mock PyBoy
        mock_pyboy_instance = Mock()
        mock_pyboy.return_value = mock_pyboy_instance
        mock_pyboy_instance.game_wrapper.return_value = Mock()
        
        # Mock context builder
        mock_context_instance = Mock()
        mock_context_builder.return_value = mock_context_instance
        
        # Create environment
        env = EnhancedPyBoyPokemonCrystalEnv(
            rom_path=str(self.rom_path),
            headless=True,
            observation_type="multi_modal"
        )
        
        # Verify initialization
        self.assertIsNotNone(env)
        self.assertEqual(env.observation_type, "multi_modal")
        self.assertTrue(env.headless)
        mock_pyboy.assert_called_once()
    
    @patch('core.enhanced_pyboy_env.PyBoy')
    @patch('core.enhanced_pyboy_env.StrategicContextBuilder')
    def test_observation_space_multi_modal(self, mock_context_builder, mock_pyboy):
        """Test multi-modal observation space structure."""
        # Setup mocks
        mock_pyboy_instance = Mock()
        mock_pyboy.return_value = mock_pyboy_instance
        mock_context_instance = Mock()
        mock_context_builder.return_value = mock_context_instance
        
        # Mock screen array
        mock_pyboy_instance.screen.screen_ndarray.return_value = np.zeros((144, 160, 3), dtype=np.uint8)
        
        # Create environment
        env = EnhancedPyBoyPokemonCrystalEnv(
            rom_path=str(self.rom_path),
            observation_type="multi_modal"
        )
        
        # Check observation space
        obs_space = env.observation_space
        self.assertIn('screen', obs_space.spaces)
        self.assertIn('state_variables', obs_space.spaces)
        self.assertIn('strategic_context', obs_space.spaces)
        
        # Check dimensions
        screen_shape = obs_space['screen'].shape
        self.assertEqual(screen_shape, (144, 160, 3))
        
        state_shape = obs_space['state_variables'].shape
        self.assertEqual(len(state_shape), 1)  # Should be 1D array
        self.assertGreater(state_shape[0], 0)  # Should have some variables
    
    @patch('core.enhanced_pyboy_env.PyBoy')
    @patch('core.enhanced_pyboy_env.StrategicContextBuilder')
    def test_action_space_configuration(self, mock_context_builder, mock_pyboy):
        """Test action space configuration."""
        # Setup mocks
        mock_pyboy_instance = Mock()
        mock_pyboy.return_value = mock_pyboy_instance
        mock_context_instance = Mock()
        mock_context_builder.return_value = mock_context_instance
        
        # Create environment
        env = EnhancedPyBoyPokemonCrystalEnv(rom_path=str(self.rom_path))
        
        # Check action space
        action_space = env.action_space
        self.assertEqual(action_space.n, 9)  # 8 PyBoy actions + no-op
    
    @patch('core.enhanced_pyboy_env.PyBoy')
    @patch('core.enhanced_pyboy_env.StrategicContextBuilder')
    @patch('core.enhanced_pyboy_env.get_memory_value')
    def test_reset_functionality(self, mock_get_memory, mock_context_builder, mock_pyboy):
        """Test environment reset functionality."""
        # Setup mocks
        mock_pyboy_instance = Mock()
        mock_pyboy.return_value = mock_pyboy_instance
        mock_pyboy_instance.screen.screen_ndarray.return_value = np.zeros((144, 160, 3), dtype=np.uint8)
        
        mock_context_instance = Mock()
        mock_context_builder.return_value = mock_context_instance
        mock_context_instance.build_context.return_value = {
            'current_goal': 'test_goal',
            'context_summary': 'test_summary',
            'action_suggestions': []
        }
        
        # Mock memory values
        mock_get_memory.return_value = 100
        
        # Create environment
        env = EnhancedPyBoyPokemonCrystalEnv(rom_path=str(self.rom_path))
        
        # Reset environment
        observation, info = env.reset()
        
        # Verify reset result
        self.assertIsInstance(observation, dict)
        self.assertIsInstance(info, dict)
        
        # Check observation structure for multi_modal
        if env.observation_type == "multi_modal":
            self.assertIn('screen', observation)
            self.assertIn('state_variables', observation)
            self.assertIn('strategic_context', observation)
    
    @patch('core.enhanced_pyboy_env.PyBoy')
    @patch('core.enhanced_pyboy_env.StrategicContextBuilder')
    @patch('core.enhanced_pyboy_env.get_memory_value')
    def test_step_functionality(self, mock_get_memory, mock_context_builder, mock_pyboy):
        """Test environment step functionality."""
        # Setup mocks
        mock_pyboy_instance = Mock()
        mock_pyboy.return_value = mock_pyboy_instance
        mock_pyboy_instance.screen.screen_ndarray.return_value = np.zeros((144, 160, 3), dtype=np.uint8)
        mock_pyboy_instance.tick.return_value = None
        
        mock_context_instance = Mock()
        mock_context_builder.return_value = mock_context_instance
        mock_context_instance.build_context.return_value = {
            'current_goal': 'test_goal',
            'context_summary': 'test_summary', 
            'action_suggestions': []
        }
        
        # Mock memory values
        mock_get_memory.return_value = 100
        
        # Create environment
        env = EnhancedPyBoyPokemonCrystalEnv(rom_path=str(self.rom_path))
        
        # Reset first
        env.reset()
        
        # Take a step
        action = 0  # A button
        observation, reward, terminated, truncated, info = env.step(action)
        
        # Verify step result
        self.assertIsInstance(observation, dict)
        self.assertIsInstance(reward, (int, float))
        self.assertIsInstance(terminated, bool)
        self.assertIsInstance(truncated, bool)
        self.assertIsInstance(info, dict)
        
        # Verify PyBoy was ticked
        mock_pyboy_instance.tick.assert_called()
    
    @patch('core.enhanced_pyboy_env.PyBoy')
    @patch('core.enhanced_pyboy_env.StrategicContextBuilder')
    def test_action_masking(self, mock_context_builder, mock_pyboy):
        """Test action masking functionality."""
        # Setup mocks
        mock_pyboy_instance = Mock()
        mock_pyboy.return_value = mock_pyboy_instance
        
        mock_context_instance = Mock()
        mock_context_builder.return_value = mock_context_instance
        
        # Create environment
        env = EnhancedPyBoyPokemonCrystalEnv(rom_path=str(self.rom_path))
        
        # Test action masking method
        game_state = {'in_battle': True, 'in_dialogue': False}
        action_mask = env._get_action_mask(game_state)
        
        # Verify mask is boolean array of correct size
        self.assertIsInstance(action_mask, np.ndarray)
        self.assertEqual(len(action_mask), env.action_space.n)
        self.assertTrue(all(isinstance(x, (bool, np.bool_)) for x in action_mask))
    
    @patch('core.enhanced_pyboy_env.PyBoy')
    @patch('core.enhanced_pyboy_env.StrategicContextBuilder')
    @patch('core.enhanced_pyboy_env.get_memory_value')
    def test_reward_calculation(self, mock_get_memory, mock_context_builder, mock_pyboy):
        """Test reward calculation system."""
        # Setup mocks
        mock_pyboy_instance = Mock()
        mock_pyboy.return_value = mock_pyboy_instance
        
        mock_context_instance = Mock()
        mock_context_builder.return_value = mock_context_instance
        
        # Mock memory values for different scenarios
        def memory_side_effect(addr):
            memory_map = {
                0xD16B: 10,   # Player X
                0xD16C: 10,   # Player Y
                0xD47C: 50,   # Player HP
                0xD47D: 100,  # Max HP
                0xD484: 5,    # Player level
                0xD35E: 0,    # Party size
            }
            return memory_map.get(addr, 0)
        
        mock_get_memory.side_effect = memory_side_effect
        
        # Create environment
        env = EnhancedPyBoyPokemonCrystalEnv(rom_path=str(self.rom_path))
        
        # Test reward calculation
        current_state = {'player_hp': 50, 'max_hp': 100, 'level': 5}
        previous_state = {'player_hp': 60, 'max_hp': 100, 'level': 4}
        action = 0
        
        reward = env._calculate_reward(current_state, previous_state, action)
        
        # Should receive some reward (level up positive, hp loss negative)
        self.assertIsInstance(reward, (int, float))
    
    @patch('core.enhanced_pyboy_env.PyBoy')
    @patch('core.enhanced_pyboy_env.StrategicContextBuilder')
    def test_observation_types(self, mock_context_builder, mock_pyboy):
        """Test different observation type configurations."""
        # Setup mocks
        mock_pyboy_instance = Mock()
        mock_pyboy.return_value = mock_pyboy_instance
        mock_pyboy_instance.screen.screen_ndarray.return_value = np.zeros((144, 160, 3), dtype=np.uint8)
        
        mock_context_instance = Mock()
        mock_context_builder.return_value = mock_context_instance
        
        # Test screen-only observation
        env_screen = EnhancedPyBoyPokemonCrystalEnv(
            rom_path=str(self.rom_path),
            observation_type="screen"
        )
        self.assertEqual(env_screen.observation_space.shape, (144, 160, 3))
        
        # Test state-only observation  
        env_state = EnhancedPyBoyPokemonCrystalEnv(
            rom_path=str(self.rom_path),
            observation_type="state"
        )
        self.assertEqual(len(env_state.observation_space.shape), 1)
        
        # Test multi-modal observation
        env_multi = EnhancedPyBoyPokemonCrystalEnv(
            rom_path=str(self.rom_path),
            observation_type="multi_modal"
        )
        self.assertIn('screen', env_multi.observation_space.spaces)
        self.assertIn('state_variables', env_multi.observation_space.spaces)
    
    @patch('core.enhanced_pyboy_env.PyBoy')
    @patch('core.enhanced_pyboy_env.StrategicContextBuilder')
    def test_close_functionality(self, mock_context_builder, mock_pyboy):
        """Test environment cleanup on close."""
        # Setup mocks
        mock_pyboy_instance = Mock()
        mock_pyboy.return_value = mock_pyboy_instance
        mock_pyboy_instance.stop = Mock()
        
        mock_context_instance = Mock()
        mock_context_builder.return_value = mock_context_instance
        
        # Create and close environment
        env = EnhancedPyBoyPokemonCrystalEnv(rom_path=str(self.rom_path))
        env.close()
        
        # Verify PyBoy was stopped
        mock_pyboy_instance.stop.assert_called_once()


class TestEnvironmentIntegration(unittest.TestCase):
    """Integration tests for environment components."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.rom_path = Path(self.temp_dir) / "test.gbc"
        
        # Create dummy ROM file
        with open(self.rom_path, 'wb') as f:
            f.write(b'\x00' * 1024)
    
    def tearDown(self):
        """Clean up test resources."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    @patch('core.enhanced_pyboy_env.PyBoy')
    @patch('core.enhanced_pyboy_env.StrategicContextBuilder')
    @patch('core.enhanced_pyboy_env.get_memory_value')
    def test_full_episode_simulation(self, mock_get_memory, mock_context_builder, mock_pyboy):
        """Test complete episode simulation."""
        # Setup comprehensive mocks
        mock_pyboy_instance = Mock()
        mock_pyboy.return_value = mock_pyboy_instance
        mock_pyboy_instance.screen.screen_ndarray.return_value = np.random.randint(0, 255, (144, 160, 3), dtype=np.uint8)
        
        mock_context_instance = Mock()
        mock_context_builder.return_value = mock_context_instance
        mock_context_instance.build_context.return_value = {
            'current_goal': 'explore',
            'context_summary': 'Player exploring',
            'action_suggestions': ['move_up', 'move_right']
        }
        
        # Mock progressive memory values
        step_count = 0
        def memory_side_effect(addr):
            nonlocal step_count
            base_values = {
                0xD16B: 10 + step_count % 5,  # Player X (moving)
                0xD16C: 10 + step_count % 5,  # Player Y (moving)  
                0xD47C: max(1, 100 - step_count),  # Player HP (decreasing)
                0xD47D: 100,  # Max HP
                0xD484: 1,    # Player level
                0xD35E: 1,    # Party size
            }
            return base_values.get(addr, 0)
        
        mock_get_memory.side_effect = memory_side_effect
        
        # Create environment
        env = EnhancedPyBoyPokemonCrystalEnv(rom_path=str(self.rom_path))
        
        # Run episode simulation
        observation, info = env.reset()
        total_reward = 0
        episode_length = 0
        
        for step in range(10):  # Short episode
            step_count = step
            action = step % env.action_space.n  # Cycle through actions
            observation, reward, terminated, truncated, info = env.step(action)
            
            total_reward += reward
            episode_length += 1
            
            # Verify observation structure
            if env.observation_type == "multi_modal":
                self.assertIn('screen', observation)
                self.assertIn('state_variables', observation)
                self.assertIn('strategic_context', observation)
            
            if terminated or truncated:
                break
        
        # Verify episode completed successfully
        self.assertGreater(episode_length, 0)
        self.assertIsInstance(total_reward, (int, float))
        
        # Clean up
        env.close()


if __name__ == '__main__':
    unittest.main()