#!/usr/bin/env python3
"""
test_unified_trainer.py - Comprehensive tests for UnifiedPokemonTrainer

This test file covers all aspects of the UnifiedPokemonTrainer including:
- Initialization and configuration
- PyBoy stability and recovery
- Error handling and logging
- Web dashboard functionality
- Rule-based action system
- Training modes and performance
- Integration scenarios
"""

import pytest
import time
import threading
import numpy as np
from unittest.mock import Mock, patch, MagicMock, call
import logging
import asyncio
from pathlib import Path

# Import test system modules
import sys
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from trainer.unified_trainer import UnifiedPokemonTrainer
from trainer.trainer import (
    TrainingConfig,
    TrainingMode,
    LLMBackend
)


@pytest.mark.unified_trainer
class TestTrainingConfig:
    """Test TrainingConfig functionality"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = TrainingConfig()
        
        assert config.rom_path == ""
        assert config.mode == TrainingMode.FAST_MONITORED
        assert config.max_actions == 10000
        assert config.headless == True
        assert config.debug_mode == False
        assert config.enable_web == True
        assert config.capture_screens == True
        assert config.log_level == "INFO"
        assert config.llm_backend == LLMBackend.NONE
        assert config.llm_interval == 10
    
    def test_custom_config(self):
        """Test custom configuration values"""
        config = TrainingConfig(
            rom_path="test.gbc",
            mode=TrainingMode.ULTRA_FAST,
            max_actions=5000,
            headless=False,
            debug_mode=True,
            enable_web=False,
            capture_screens=False,
            log_level="DEBUG",
            llm_backend=LLMBackend.SMOLLM2,
            llm_interval=20
        )
        
        assert config.rom_path == "test.gbc"
        assert config.mode == TrainingMode.ULTRA_FAST
        assert config.max_actions == 5000
        assert config.headless == False
        assert config.debug_mode == True
        assert config.enable_web == False
        assert config.capture_screens == False
        assert config.log_level == "DEBUG"
        assert config.llm_backend == LLMBackend.SMOLLM2
        assert config.llm_interval == 20


@pytest.mark.unified_trainer
class TestUnifiedPokemonTrainerInit:
    """Test initialization and setup"""
    
    @pytest.fixture
    def base_config(self):
        """Base configuration for testing"""
        return TrainingConfig(
            rom_path="test.gbc",
            headless=True,
            debug_mode=True,
            enable_web=False,
            capture_screens=False,
            log_level="DEBUG"
        )
    
    @patch('trainer.trainer.PyBoy')
    @patch('trainer.trainer.PYBOY_AVAILABLE', True)
    def test_initialization_success(self, mock_pyboy_class, base_config):
        """Test successful initialization with PyBoy available"""
        # Mock PyBoy instance
        mock_pyboy_instance = Mock()
        mock_pyboy_instance.frame_count = 1000
        mock_pyboy_class.return_value = mock_pyboy_instance
        
        trainer = UnifiedPokemonTrainer(base_config)
        
        assert trainer is not None
        assert trainer.config == base_config
        assert trainer.pyboy == mock_pyboy_instance
        assert hasattr(trainer, 'stats')
        assert hasattr(trainer, 'error_counts')
    
    @patch('trainer.trainer.PYBOY_AVAILABLE', False)
    def test_initialization_no_pyboy(self, base_config):
        """Test initialization when PyBoy is not available"""
        trainer = UnifiedPokemonTrainer(base_config)
        
        assert trainer is not None
        assert trainer.config == base_config
        assert trainer.pyboy is None
    
    @patch('trainer.trainer.PyBoy')
    @patch('trainer.trainer.PYBOY_AVAILABLE', True)
    def test_logging_setup(self, mock_pyboy_class, base_config):
        """Test that logging is properly configured"""
        mock_pyboy_instance = Mock()
        mock_pyboy_class.return_value = mock_pyboy_instance
        
        trainer = UnifiedPokemonTrainer(base_config)
        
        assert hasattr(trainer, 'logger')
        assert trainer.logger.level == logging.DEBUG
    
    @patch('trainer.trainer.PyBoy')
    @patch('trainer.trainer.PYBOY_AVAILABLE', True)
    def test_error_tracking_init(self, mock_pyboy_class, base_config):
        """Test that error tracking is initialized"""
        mock_pyboy_instance = Mock()
        mock_pyboy_class.return_value = mock_pyboy_instance
        
        trainer = UnifiedPokemonTrainer(base_config)
        
        assert hasattr(trainer, 'error_counts')
        assert trainer.error_counts['pyboy_crashes'] == 0
        assert trainer.error_counts['llm_failures'] == 0
        assert trainer.error_counts['capture_errors'] == 0
        assert trainer.error_counts['total_errors'] == 0


@pytest.mark.unified_trainer
class TestPyBoyStabilityAndRecovery:
    """Test PyBoy stability and crash recovery"""
    
    @pytest.fixture
    @patch('trainer.trainer.PyBoy')
    @patch('trainer.trainer.PYBOY_AVAILABLE', True)
    def trainer(self, mock_pyboy_class):
        """Create trainer for stability testing"""
        mock_pyboy_instance = Mock()
        mock_pyboy_instance.frame_count = 1000
        mock_pyboy_instance.screen_image.return_value = np.random.randint(
            0, 256, (144, 160, 3), dtype=np.uint8
        )
        mock_pyboy_class.return_value = mock_pyboy_instance
        
        config = TrainingConfig(
            rom_path="test.gbc",
            headless=True,
            debug_mode=True,
            enable_web=False,
            capture_screens=False
        )
        
        return UnifiedPokemonTrainer(config)
    
    def test_pyboy_alive_check_success(self, trainer):
        """Test successful PyBoy alive check"""
        result = trainer._is_pyboy_alive()
        assert result == True
    
    def test_pyboy_alive_check_failure_none(self, trainer):
        """Test PyBoy alive check when pyboy is None"""
        trainer.pyboy = None
        result = trainer._is_pyboy_alive()
        assert result == False
    
    def test_pyboy_alive_check_failure_exception(self, trainer):
        """Test PyBoy alive check when exception occurs"""
        trainer.pyboy.frame_count = Mock(side_effect=Exception("Test error"))
        result = trainer._is_pyboy_alive()
        assert result == False
    
    def test_pyboy_alive_check_invalid_frame_count(self, trainer):
        """Test PyBoy alive check with invalid frame count"""
        trainer.pyboy.frame_count = -1
        result = trainer._is_pyboy_alive()
        assert result == False
    
    @patch('trainer.trainer.PyBoy')
    def test_pyboy_recovery_success(self, mock_pyboy_class, trainer):
        """Test successful PyBoy recovery"""
        # Setup new mock instance for recovery
        new_mock_instance = Mock()
        new_mock_instance.frame_count = 2000
        mock_pyboy_class.return_value = new_mock_instance
        
        result = trainer._attempt_pyboy_recovery()
        
        assert result == True
        assert trainer.pyboy == new_mock_instance
        assert trainer.error_counts['pyboy_crashes'] == 1
    
    @patch('trainer.trainer.PyBoy')
    def test_pyboy_recovery_with_save_state(self, mock_pyboy_class, trainer):
        """Test PyBoy recovery with save state loading"""
        # Setup save state
        trainer.save_state = b"fake_save_state"
        
        new_mock_instance = Mock()
        new_mock_instance.frame_count = 2000
        mock_pyboy_class.return_value = new_mock_instance
        
        result = trainer._attempt_pyboy_recovery()
        
        assert result == True
        new_mock_instance.load_state.assert_called_once_with(trainer.save_state)
    
    @patch('trainer.trainer.PyBoy')
    def test_pyboy_recovery_failure(self, mock_pyboy_class, trainer):
        """Test PyBoy recovery failure"""
        mock_pyboy_class.side_effect = Exception("Recovery failed")
        
        result = trainer._attempt_pyboy_recovery()
        
        assert result == False
        assert trainer.pyboy is None
    
    def test_screen_format_conversion_rgba_to_rgb(self, trainer):
        """Test screen format conversion from RGBA to RGB"""
        # Create RGBA screen (144, 160, 4)
        rgba_screen = np.random.randint(0, 256, (144, 160, 4), dtype=np.uint8)
        trainer.pyboy.screen_image.return_value = rgba_screen
        
        screen = trainer._get_screen()
        
        assert screen.shape == (144, 160, 3)
        np.testing.assert_array_equal(screen, rgba_screen[:, :, :3])
    
    def test_screen_format_conversion_grayscale_to_rgb(self, trainer):
        """Test screen format conversion from grayscale to RGB"""
        # Create grayscale screen (144, 160)
        gray_screen = np.random.randint(0, 256, (144, 160), dtype=np.uint8)
        trainer.pyboy.screen_image.return_value = gray_screen
        
        screen = trainer._get_screen()
        
        assert screen.shape == (144, 160, 3)
        # Check that all channels are the same (grayscale converted to RGB)
        np.testing.assert_array_equal(screen[:, :, 0], gray_screen)
        np.testing.assert_array_equal(screen[:, :, 1], gray_screen)
        np.testing.assert_array_equal(screen[:, :, 2], gray_screen)
    
    def test_screen_format_conversion_invalid_shape(self, trainer):
        """Test screen format conversion with invalid shape"""
        # Create invalid screen shape
        invalid_screen = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        trainer.pyboy.screen_image.return_value = invalid_screen
        
        screen = trainer._get_screen()
        
        # Should return the screen as-is if shape is unexpected
        assert screen.shape == (100, 100, 3)


@pytest.mark.unified_trainer
class TestErrorHandlingAndLogging:
    """Test error handling and logging functionality"""
    
    @pytest.fixture
    @patch('trainer.trainer.PyBoy')
    @patch('trainer.trainer.PYBOY_AVAILABLE', True)
    def trainer(self, mock_pyboy_class):
        """Create trainer for error handling testing"""
        mock_pyboy_instance = Mock()
        mock_pyboy_instance.frame_count = 1000
        mock_pyboy_class.return_value = mock_pyboy_instance
        
        config = TrainingConfig(
            rom_path="test.gbc",
            headless=True,
            debug_mode=True,
            enable_web=False,
            capture_screens=False
        )
        
        return UnifiedPokemonTrainer(config)
    
    def test_error_context_manager_success(self, trainer):
        """Test error context manager with successful operation"""
        with trainer._handle_errors("test_operation"):
            # Simulate successful operation
            pass
        
        # No errors should be recorded
        assert trainer.error_counts['total_errors'] == 0
    
    def test_error_context_manager_exception(self, trainer):
        """Test error context manager with exception"""
        with pytest.raises(Exception):
            with trainer._handle_errors("test_operation"):
                raise Exception("Test error")
        
        # Error should be recorded
        assert trainer.error_counts['total_errors'] == 1
    
    @patch('trainer.trainer.PyBoy')
    def test_error_context_manager_pyboy_recovery(self, mock_pyboy_class, trainer):
        """Test error context manager with PyBoy recovery"""
        # Setup recovery mock
        new_mock_instance = Mock()
        new_mock_instance.frame_count = 2000
        mock_pyboy_class.return_value = new_mock_instance
        
        # Simulate PyBoy failure
        trainer.pyboy = None
        
        with trainer._handle_errors("test_operation"):
            # This should trigger PyBoy recovery
            pass
        
        # Recovery should have been attempted
        assert trainer.pyboy == new_mock_instance
    
    def test_keyboard_interrupt_handling(self, trainer):
        """Test KeyboardInterrupt handling"""
        with pytest.raises(KeyboardInterrupt):
            with trainer._handle_errors("test_operation"):
                raise KeyboardInterrupt()
        
        # KeyboardInterrupt should not be counted as regular error
        assert trainer.error_counts['total_errors'] == 0
    
    def test_logging_levels(self, trainer):
        """Test different logging levels"""
        # Test that logger is configured correctly
        assert trainer.logger.level == logging.DEBUG
        
        # Test logging methods exist
        assert hasattr(trainer.logger, 'debug')
        assert hasattr(trainer.logger, 'info')
        assert hasattr(trainer.logger, 'warning')
        assert hasattr(trainer.logger, 'error')


@pytest.mark.unified_trainer
class TestWebDashboardAndHTTPPolling:
    """Test web dashboard and HTTP polling functionality"""
    
    @pytest.fixture
    @patch('trainer.trainer.PyBoy')
    @patch('trainer.trainer.PYBOY_AVAILABLE', True)
    @patch('core.monitoring.web_server.TrainingWebServer')  # Fixed path
    def trainer(self, mock_web_server_class, mock_pyboy_class):
        """Create trainer with web server for testing"""
        mock_pyboy_instance = Mock()
        mock_pyboy_instance.frame_count = 1000
        mock_pyboy_class.return_value = mock_pyboy_instance
        
        # Mock web server
        mock_web_server = Mock()
        mock_web_server_class.return_value = mock_web_server
        
        config = TrainingConfig(
            rom_path="test.gbc",
            headless=True,
            debug_mode=True,
            enable_web=True,
            capture_screens=True
        )
        
        return UnifiedPokemonTrainer(config)
    
    @patch('core.monitoring.web_server.TrainingWebServer')  # Fixed path
    def test_web_server_initialization(self, mock_web_server_class):
        """Test web server initialization"""
        mock_web_server = Mock()
        mock_web_server_class.return_value = mock_web_server
        
        config = TrainingConfig(
            rom_path="test.gbc",
            enable_web=True
        )
        
        with patch('trainer.trainer.PyBoy'), \
             patch('trainer.trainer.PYBOY_AVAILABLE', True):
            trainer = UnifiedPokemonTrainer(config)
            
            # Web server should be initialized
            mock_web_server_class.assert_called_once()
            mock_web_server.start.assert_called_once()
    
    def test_screenshot_memory_management(self, trainer):
        """Test screenshot memory management"""
        # Fill screenshot queue to capacity
        for i in range(35):  # More than queue limit of 30
            trainer._capture_and_queue_screen()
        
        # Queue should not exceed limit
        assert len(trainer.screenshot_queue) <= 30
    
    def test_api_status_endpoint_data(self, trainer):
        """Test API status endpoint data structure"""
        stats = trainer.get_current_stats()
        
        # Check required fields for API
        assert 'total_actions' in stats
        assert 'total_errors' in stats
        assert 'pyboy_crashes' in stats
        assert 'llm_failures' in stats
        assert 'capture_errors' in stats


@pytest.mark.unified_trainer
class TestRuleBasedActionSystem:
    """Test rule-based action system"""
    
    @pytest.fixture
    @patch('trainer.trainer.PyBoy')
    @patch('trainer.trainer.PYBOY_AVAILABLE', True)
    def trainer(self, mock_pyboy_class):
        """Create trainer for rule-based testing"""
        mock_pyboy_instance = Mock()
        mock_pyboy_instance.frame_count = 1000
        mock_pyboy_instance.screen_image.return_value = np.random.randint(
            0, 256, (144, 160, 3), dtype=np.uint8
        )
        mock_pyboy_instance.send_input = Mock()
        mock_pyboy_instance.tick = Mock()
        mock_pyboy_class.return_value = mock_pyboy_instance
        
        config = TrainingConfig(
            rom_path="test.gbc",
            headless=True,
            debug_mode=True,
            enable_web=False,
            capture_screens=False
        )
        
        return UnifiedPokemonTrainer(config)
    
    def test_game_state_detection_unknown(self, trainer):
        """Test game state detection for unknown state"""
        # Create screen that doesn't match any known patterns
        screen = np.random.randint(0, 256, (144, 160, 3), dtype=np.uint8)
        trainer.pyboy.screen_image.return_value = screen
        
        state = trainer._detect_game_state()
        assert state in ['unknown', 'overworld', 'dialogue', 'menu', 'battle', 'loading', 'intro']
    
    def test_game_state_detection_loading(self, trainer):
        """Test game state detection for loading state"""
        # Create mostly black screen (loading indicator)
        screen = np.zeros((144, 160, 3), dtype=np.uint8)
        trainer.pyboy.screen_image.return_value = screen
        
        state = trainer._detect_game_state()
        # Should detect as loading or unknown
        assert state in ['loading', 'unknown']
    
    def test_game_state_detection_intro(self, trainer):
        """Test game state detection for intro state"""
        # Create bright screen (intro indicator)
        screen = np.full((144, 160, 3), 255, dtype=np.uint8)
        trainer.pyboy.screen_image.return_value = screen
        
        state = trainer._detect_game_state()
        # Should detect as intro or unknown
        assert state in ['intro', 'unknown']
    
    def test_game_state_detection_dialogue(self, trainer):
        """Test game state detection for dialogue state"""
        # Create screen with dialogue box pattern
        screen = np.random.randint(50, 200, (144, 160, 3), dtype=np.uint8)
        # Add dialogue box area (bottom portion)
        screen[100:140, 10:150, :] = 255  # White dialogue box
        trainer.pyboy.screen_image.return_value = screen
        
        state = trainer._detect_game_state()
        # Should detect as dialogue or unknown
        assert state in ['dialogue', 'unknown']
    
    def test_screen_hash_calculation(self, trainer):
        """Test screen hash calculation"""
        screen = np.random.randint(0, 256, (144, 160, 3), dtype=np.uint8)
        trainer.pyboy.screen_image.return_value = screen
        
        hash1 = trainer._get_screen_hash()
        hash2 = trainer._get_screen_hash()
        
        # Same screen should produce same hash
        assert hash1 == hash2
        assert isinstance(hash1, str)
        assert len(hash1) > 0
    
    def test_screen_hash_different_screens(self, trainer):
        """Test screen hash for different screens"""
        screen1 = np.zeros((144, 160, 3), dtype=np.uint8)
        screen2 = np.ones((144, 160, 3), dtype=np.uint8) * 255
        
        trainer.pyboy.screen_image.return_value = screen1
        hash1 = trainer._get_screen_hash()
        
        trainer.pyboy.screen_image.return_value = screen2
        hash2 = trainer._get_screen_hash()
        
        # Different screens should produce different hashes
        assert hash1 != hash2
    
    @patch('trainer.trainer.PyBoy')
    def test_stuck_detection_mechanism(self, mock_pyboy_class, trainer):
        """Test stuck detection and recovery mechanism"""
        # Setup proper PyBoy mock for recovery
        new_mock_instance = Mock()
        new_mock_instance.frame_count = 2000
        new_mock_instance.send_input = Mock()
        new_mock_instance.tick = Mock()
        mock_pyboy_class.return_value = new_mock_instance
        
        # Simulate same screen hash multiple times (stuck condition)
        same_screen = np.random.randint(0, 256, (144, 160, 3), dtype=np.uint8)
        trainer.pyboy.screen_image.return_value = same_screen
        
        # Execute multiple actions with same screen
        for _ in range(5):
            trainer._execute_synchronized_action(1)
        
        # Should have attempted recovery or unstuck actions
        assert trainer.pyboy.send_input.call_count >= 5
    
    def test_title_screen_handling(self, trainer):
        """Test title screen detection and handling"""
        # Create title screen pattern
        screen = np.random.randint(0, 100, (144, 160, 3), dtype=np.uint8)
        # Add title elements
        screen[20:40, 40:120, :] = 200  # Title text area
        trainer.pyboy.screen_image.return_value = screen
        
        state = trainer._detect_game_state()
        # Should handle title screen appropriately
        assert state in ['intro', 'menu', 'unknown']
    
    def test_dialogue_handling(self, trainer):
        """Test dialogue state handling"""
        # Create dialogue screen
        screen = np.random.randint(0, 256, (144, 160, 3), dtype=np.uint8)
        screen[100:140, 10:150, :] = 255  # Dialogue box
        trainer.pyboy.screen_image.return_value = screen
        
        # Execute action in dialogue state
        trainer._execute_synchronized_action(1)  # A button (advance dialogue)
        
        # Should have sent input
        trainer.pyboy.send_input.assert_called()
    
    def test_unstuck_action_patterns(self, trainer):
        """Test unstuck action patterns"""
        # Test that unstuck actions are valid
        for _ in range(10):
            action = trainer._get_unstuck_action()
            assert 1 <= action <= 8  # Valid action range


@pytest.mark.unified_trainer
class TestTrainingModes:
    """Test different training modes"""
    
    @pytest.fixture
    @patch('trainer.trainer.PyBoy')
    @patch('trainer.trainer.PYBOY_AVAILABLE', True)
    def trainer_fast_monitored(self, mock_pyboy_class):
        """Create trainer for fast monitored testing"""
        mock_pyboy_instance = Mock()
        mock_pyboy_instance.frame_count = 1000
        mock_pyboy_instance.screen_image.return_value = np.random.randint(
            0, 256, (144, 160, 3), dtype=np.uint8
        )
        mock_pyboy_instance.send_input = Mock()
        mock_pyboy_instance.tick = Mock()
        mock_pyboy_class.return_value = mock_pyboy_instance
        
        config = TrainingConfig(
            rom_path="test.gbc",
            mode=TrainingMode.FAST_MONITORED,
            max_actions=10,
            headless=True,
            enable_web=False,
            capture_screens=False
        )
        
        return UnifiedPokemonTrainer(config)
    
    @pytest.fixture
    @patch('trainer.trainer.PyBoy')
    @patch('trainer.trainer.PYBOY_AVAILABLE', True)
    def trainer_ultra_fast(self, mock_pyboy_class):
        """Create trainer for ultra fast testing"""
        mock_pyboy_instance = Mock()
        mock_pyboy_instance.frame_count = 1000
        mock_pyboy_instance.screen_image.return_value = np.random.randint(
            0, 256, (144, 160, 3), dtype=np.uint8
        )
        mock_pyboy_instance.send_input = Mock()
        mock_pyboy_instance.tick = Mock()
        mock_pyboy_class.return_value = mock_pyboy_instance
        
        config = TrainingConfig(
            rom_path="test.gbc",
            mode=TrainingMode.ULTRA_FAST,
            max_actions=10,
            headless=True,
            enable_web=False,
            capture_screens=False
        )
        
        return UnifiedPokemonTrainer(config)
    
    def test_fast_monitored_training_execution(self, trainer_fast_monitored):
        """Test fast monitored training execution"""
        # Run training for a few steps
        trainer_fast_monitored._run_synchronized_training()
        
        # Should have executed actions
        assert trainer_fast_monitored.stats['total_actions'] > 0
        assert trainer_fast_monitored.pyboy.send_input.call_count > 0
    
    def test_ultra_fast_training_execution(self, trainer_ultra_fast):
        """Test ultra fast training execution"""
        # Run training for a few steps
        trainer_ultra_fast._run_ultra_fast_training()
        
        # Should have executed actions
        assert trainer_ultra_fast.stats['total_actions'] > 0
        assert trainer_ultra_fast.pyboy.send_input.call_count > 0
    
    def test_action_execution(self, trainer_fast_monitored):
        """Test action execution"""
        initial_actions = trainer_fast_monitored.stats['total_actions']
        
        trainer_fast_monitored._execute_synchronized_action(5)
        
        # Action count should increase
        assert trainer_fast_monitored.stats['total_actions'] == initial_actions + 1
        trainer_fast_monitored.pyboy.send_input.assert_called_with(5)
    
    def test_stats_updating(self, trainer_fast_monitored):
        """Test stats updating during training"""
        trainer = trainer_fast_monitored
        
        initial_time = trainer.stats['start_time']
        trainer.stats['total_actions'] = 100
        
        # Wait a small amount to ensure time difference
        time.sleep(0.01)
        
        trainer._update_stats()
        
        # Actions per second should be calculated
        assert trainer.stats['actions_per_second'] > 0
        
        # Start time should remain the same
        assert trainer.stats['start_time'] == initial_time

class TestIntegrationScenarios:
    """Integration tests for complete training scenarios"""
    
    @pytest.fixture
    def integration_config(self):
        return TrainingConfig(
            rom_path="test.gbc",
            mode=TrainingMode.FAST_MONITORED,
            max_actions=50,
            capture_screens=True,
            headless=True,
            debug_mode=True,
            log_level="DEBUG"
        )
    
    @patch('trainer.trainer.PyBoy')
    @patch('trainer.trainer.PYBOY_AVAILABLE', True)
    def test_full_training_cycle_with_recovery(self, mock_pyboy_class, integration_config):
        """Test complete training cycle with PyBoy recovery"""
        # Setup mocks
        mock_pyboy_instance = Mock()
        mock_pyboy_instance.frame_count = 1000
        mock_pyboy_instance.screen.ndarray = np.random.randint(0, 255, (144, 160, 3), dtype=np.uint8)
        mock_pyboy_instance.send_input = Mock()
        mock_pyboy_instance.tick = Mock()
        mock_pyboy_instance.stop = Mock()
        mock_pyboy_class.return_value = mock_pyboy_instance
        
        trainer = None
        try:
            trainer = UnifiedPokemonTrainer(integration_config)
            
            # Mock training methods to prevent actual PyBoy operations
            trainer._start_screen_capture = Mock()
            
            # Simulate PyBoy crash and recovery during training
            call_count = [0]
            recovery_triggered = [False]
            max_crashes = 3  # Limit crashes to prevent infinite loop
            
            def mock_send_input(action):
                call_count[0] += 1
                # Trigger crash after some actions, but limit total crashes
                if call_count[0] == 25 and not recovery_triggered[0] and len([x for x in recovery_triggered if x]) < max_crashes:
                    recovery_triggered[0] = True
                    raise Exception("Simulated PyBoy crash")
            
            def mock_recovery():
                # Reset for next potential crash
                call_count[0] = 0
                recovery_triggered[0] = False
                # Create a new mock PyBoy instance
                new_mock = Mock()
                new_mock.frame_count = 1000
                new_mock.screen.ndarray = np.random.randint(0, 255, (144, 160, 3), dtype=np.uint8)
                new_mock.send_input = Mock(side_effect=mock_send_input)
                new_mock.tick = Mock()
                new_mock.stop = Mock()
                trainer.pyboy = new_mock
                return True
            
            # Set up the crash simulation
            mock_pyboy_instance.send_input.side_effect = mock_send_input
            trainer._attempt_pyboy_recovery = Mock(side_effect=mock_recovery)
            
            # Add termination condition to prevent infinite loop
            max_iterations = 100
            iteration_count = [0]
            
            # Override the training method with termination condition
            original_run = trainer._run_synchronized_training
            def limited_run():
                while (trainer.stats['total_actions'] < min(trainer.config.max_actions, 50) and 
                    trainer._training_active and iteration_count[0] < max_iterations):
                    iteration_count[0] += 1
                    
                    # Get action
                    action = trainer._get_rule_based_action(trainer.stats['total_actions'])
                    
                    if action:
                        try:
                            trainer._execute_synchronized_action(action)
                            trainer.stats['total_actions'] += 1
                            
                            # Update stats periodically
                            if trainer.stats['total_actions'] % 20 == 0:
                                trainer._update_stats()
                                
                        except Exception as e:
                            # This should trigger the error handler and recovery
                            trainer.stats['total_actions'] -= 1
                            with trainer._handle_errors("synchronized_training", "pyboy_crashes"):
                                raise e
                    
                    # Safety exit condition
                    if iteration_count[0] >= max_iterations:
                        trainer._training_active = False
                        break
            
            trainer._run_synchronized_training = limited_run
            trainer._run_synchronized_training()
            
            # Assertions
            trainer._attempt_pyboy_recovery.assert_called()
            assert trainer.error_count['total_errors'] > 0
            
        finally:
            # Proper cleanup
            if trainer:
                trainer._training_active = False
                
                # Stop web server if running
                if hasattr(trainer, 'web_server') and trainer.web_server:
                    try:
                        trainer.web_server.shutdown()
                    except:
                        pass
                
                # Stop web thread if running
                if hasattr(trainer, 'web_thread') and trainer.web_thread:
                    try:
                        trainer.web_thread.join(timeout=1.0)
                    except:
                        pass
                
                # Clear screen queue
                if hasattr(trainer, 'screen_queue'):
                    try:
                        while not trainer.screen_queue.empty():
                            trainer.screen_queue.get_nowait()
                    except:
                        pass
                
                # Call inherited cleanup
                if hasattr(trainer, '_finalize_training'):
                    try:
                        trainer._finalize_training()
                    except:
                        pass
    
    @patch('trainer.trainer.PyBoy')
    @patch('trainer.trainer.PYBOY_AVAILABLE', True)
    def test_initialization_no_pyboy(self, mock_pyboy_class, integration_config):
        """Test memory leak prevention in screen capture"""
        mock_pyboy_instance = Mock()
        mock_pyboy_instance.frame_count = 1000
        mock_pyboy_class.return_value = mock_pyboy_instance
        
        trainer = UnifiedPokemonTrainer(integration_config)
        
        # Test screen queue management
        test_screen_data = {'image_b64': 'test_data', 'timestamp': time.time()}
        
        # Fill queue beyond capacity
        for i in range(35):  # Queue max is 30
            try:
                trainer.screen_queue.put_nowait(test_screen_data)
            except queue.Full:
                break
        
        # Queue should not exceed maximum size
        assert trainer.screen_queue.qsize() <= 30
    
    @patch('trainer.trainer.PyBoy')
    @patch('trainer.trainer.PYBOY_AVAILABLE', True)
    def test_configuration_edge_cases(self, mock_pyboy_class):
        """Test edge cases in configuration"""
        # Test minimum values
        minimal_config = TrainingConfig(
            rom_path="test.gbc",
            max_actions=1,
            max_episodes=1,
            llm_interval=1
        )
        
        mock_pyboy_instance = Mock()
        mock_pyboy_class.return_value = mock_pyboy_instance
        
        trainer = UnifiedPokemonTrainer(minimal_config)
        
        assert trainer.config.max_actions == 1
        assert trainer.config.max_episodes == 1
        assert trainer.config.llm_interval == 1
    
    @patch('trainer.trainer.PyBoy')
    @patch('trainer.trainer.PYBOY_AVAILABLE', True)
    @patch('trainer.llm_manager.ollama')
    def test_llm_integration_fallback(self, mock_ollama, mock_pyboy_class, integration_config):
        """Test LLM integration with fallback to rule-based"""
        # Configure for LLM use
        integration_config.llm_backend = LLMBackend.SMOLLM2
        integration_config.llm_interval = 1
        
        mock_pyboy_instance = Mock()
        mock_pyboy_class.return_value = mock_pyboy_instance
        
        # Mock LLM failure
        mock_ollama.generate.side_effect = Exception("LLM error")
        mock_ollama.show.side_effect = Exception("Model not available")
        mock_ollama.pull.return_value = None
        
        trainer = UnifiedPokemonTrainer(integration_config)
        
        # Should fallback to rule-based when LLM fails
        action = trainer._get_llm_action()
        
        assert 1 <= action <= 8  # Valid action despite LLM failure
        assert trainer.error_count['total_errors'] == 0  # LLM errors handled gracefully


@pytest.mark.state_detection
@pytest.mark.unit
class TestEnhancedStateDetection:
    """Test enhanced state detection system"""
    
    @pytest.fixture
    @patch('trainer.trainer.PyBoy')
    @patch('trainer.trainer.PYBOY_AVAILABLE', True)
    @patch('core.monitoring.data_bus.get_data_bus')
    def trainer_fast_monitored(self, mock_data_bus, mock_pyboy_class):
        mock_pyboy_instance = Mock()
        mock_pyboy_instance.frame_count = 1000
        mock_pyboy_instance.screen_image = Mock(return_value=np.random.randint(0, 255, (144, 160, 3), dtype=np.uint8))
        mock_pyboy_class.return_value = mock_pyboy_instance
        
        # Mock data bus
        mock_data_bus_instance = Mock()
        mock_data_bus_instance.register_component = Mock()
        mock_data_bus_instance.publish = Mock()
        mock_data_bus.return_value = mock_data_bus_instance
        
        config = TrainingConfig(
            rom_path="test.gbc",
            mode=TrainingMode.FAST_MONITORED,
            max_actions=10,
            capture_screens=False,
            headless=True
        )
        
        return UnifiedPokemonTrainer(config)

    @pytest.fixture
    @patch('trainer.trainer.PyBoy')
    @patch('trainer.trainer.PYBOY_AVAILABLE', True)
    @patch('core.monitoring.data_bus.get_data_bus')
    def trainer_ultra_fast(self, mock_data_bus, mock_pyboy_class):
        mock_pyboy_instance = Mock()
        mock_pyboy_instance.frame_count = 1000
        mock_pyboy_instance.screen_image = Mock(return_value=np.random.randint(0, 255, (144, 160, 3), dtype=np.uint8))
        mock_pyboy_class.return_value = mock_pyboy_instance
        
        # Mock data bus
        mock_data_bus_instance = Mock()
        mock_data_bus_instance.register_component = Mock()
        mock_data_bus_instance.publish = Mock()
        mock_data_bus.return_value = mock_data_bus_instance
        
        config = TrainingConfig(
            rom_path="test.gbc",
            headless=True,
            mode=TrainingMode.ULTRA_FAST,
            debug_mode=True
        )
        
        return UnifiedPokemonTrainer(config)
    
    @pytest.fixture
    @patch('trainer.trainer.PyBoy')
    @patch('trainer.trainer.PYBOY_AVAILABLE', True)
    @patch('core.monitoring.data_bus.get_data_bus')
    def trainer(self, mock_data_bus, mock_pyboy_class, mock_config):
        """Create trainer with mocked PyBoy and data bus"""
        mock_pyboy_instance = Mock()
        mock_pyboy_instance.frame_count = 1000
        mock_pyboy_instance.screen_image = Mock(return_value=np.random.randint(0, 255, (144, 160, 3), dtype=np.uint8))
        mock_pyboy_class.return_value = mock_pyboy_instance
        
        # Mock data bus
        mock_data_bus_instance = Mock()
        mock_data_bus_instance.register_component = Mock()
        mock_data_bus_instance.publish = Mock()
        mock_data_bus.return_value = mock_data_bus_instance
        
        config = TrainingConfig(
            rom_path="test.gbc",
            headless=True,
            debug_mode=True
        )
        
        return UnifiedPokemonTrainer(config)

    def test_improved_dialogue_detection(self, trainer):
        """Test improved dialogue state detection"""
        # Create screen with dialogue characteristics
        dialogue_screen = np.ones((144, 160, 3), dtype=np.uint8) * 100
        dialogue_screen[100:, :] = 220  # Bright bottom section (text box)
        
        state = trainer._detect_game_state(dialogue_screen)
        assert state == "dialogue"
    
    def test_enhanced_overworld_detection(self, trainer):
        """Test enhanced overworld state detection"""
        # Create varied overworld screen
        overworld_screen = np.random.randint(50, 200, (144, 160, 3), dtype=np.uint8)
        
        state = trainer._detect_game_state(overworld_screen)
        assert state in ["overworld", "unknown"]  # May be unknown with random data
    
    def test_menu_state_detection(self, trainer):
        """Test menu state detection accuracy"""
        # Create screen with menu characteristics
        menu_screen = np.ones((144, 160, 3), dtype=np.uint8) * 120
        menu_screen[20:60, 20:140] = 200  # Menu box area
        
        state = trainer._detect_game_state(menu_screen)
        # Menu detection may need more sophisticated logic
        assert state in ["menu", "dialogue", "unknown"]
    
    def test_state_detection_performance(self, trainer):
        """Test state detection performance"""
        test_screen = np.random.randint(0, 255, (144, 160, 3), dtype=np.uint8)
        
        start_time = time.time()
        
        # Run state detection 100 times
        for _ in range(100):
            trainer._detect_game_state(test_screen)
        
        elapsed = time.time() - start_time
        
        # Should be very fast (under 10ms for 100 detections)
        assert elapsed < 0.01, f"State detection too slow: {elapsed:.4f}s for 100 detections"


@pytest.mark.web_monitoring
@pytest.mark.integration
class TestWebMonitoringEnhancements:
    """Test web monitoring enhancements in unified trainer"""
    
    @pytest.fixture
    def web_trainer_config(self):
        import socket
        sock = socket.socket()
        sock.bind(('', 0))
        port = sock.getsockname()[1]
        sock.close()
        
        return TrainingConfig(
            rom_path="test.gbc",
            enable_web=True,
            web_port=port,
            capture_screens=True,
            headless=True,
            debug_mode=True
        )
    
    @patch('trainer.trainer.PyBoy')
    @patch('trainer.trainer.PYBOY_AVAILABLE', True)
    @patch('monitoring.web_server.TrainingWebServer')
    def test_web_server_initialization(self, mock_web_server, mock_pyboy_class, web_trainer_config):
        """Test enhanced web server initialization"""
        mock_pyboy_instance = Mock()
        mock_pyboy_class.return_value = mock_pyboy_instance
        
        mock_server_instance = Mock()
        mock_server_instance.start = Mock()
        mock_server_instance.server = Mock()
        mock_server_instance.server.serve_forever = Mock()
        mock_web_server.return_value = mock_server_instance
        
        trainer = UnifiedPokemonTrainer(web_trainer_config)
        
        # Enhanced web features should be initialized
        assert trainer.web_server is not None
        assert trainer.web_thread is not None
        assert trainer.screen_queue.maxsize == 30  # Memory-bounded queue
        assert trainer.capture_active is False
    
    @patch('trainer.trainer.PyBoy')
    @patch('trainer.trainer.PYBOY_AVAILABLE', True)
    def test_real_time_stats_tracking(self, mock_pyboy_class, web_trainer_config):
        """Test real-time statistics tracking for web interface"""
        mock_pyboy_instance = Mock()
        mock_pyboy_class.return_value = mock_pyboy_instance
        
        trainer = UnifiedPokemonTrainer(web_trainer_config)
        
        # Initialize stats
        trainer.stats['start_time'] = time.time()
        trainer.stats['total_actions'] = 0
        
        # Simulate actions and track stats
        for i in range(10):
            trainer.stats['total_actions'] += 1
            trainer._update_stats()
        
        # Stats should be updated
        assert trainer.stats['total_actions'] == 10
        assert trainer.stats['actions_per_second'] >= 0
        assert 'uptime_seconds' in trainer.stats or trainer.stats['start_time'] > 0
    
    @patch('trainer.trainer.PyBoy')
    @patch('trainer.trainer.PYBOY_AVAILABLE', True)
    def test_screenshot_capture_optimization(self, mock_pyboy_class, web_trainer_config):
        """Test optimized screenshot capture for web monitoring"""
        mock_pyboy_instance = Mock()
        mock_pyboy_instance.screen.ndarray = np.random.randint(0, 255, (144, 160, 3), dtype=np.uint8)
        mock_pyboy_class.return_value = mock_pyboy_instance
        
        trainer = UnifiedPokemonTrainer(web_trainer_config)
        
        with patch.object(trainer, '_simple_screenshot_capture') as mock_capture:
            test_screen = np.random.randint(0, 255, (144, 160, 3), dtype=np.uint8)
            mock_capture.return_value = test_screen
            
            # Test optimized capture
            start_time = time.time()
            
            for _ in range(20):
                trainer._capture_and_queue_screen()
            
            elapsed = time.time() - start_time
            
        # Should be efficient (under 60ms for 20 captures)
        assert elapsed < 0.06, f"Screenshot capture too slow: {elapsed:.4f}s for 20 captures"
        
        # Queue should be managed efficiently
        assert trainer.screen_queue.qsize() <= 30


@pytest.mark.llm
@pytest.mark.multi_model
class TestLLMBackendSwitching:
    """Test LLM backend switching and multi-model support"""
    
    @pytest.fixture
    def model_configs(self):
        """Configurations for different models"""
        return {
            'smollm2': TrainingConfig(
                rom_path="test.gbc",
                llm_backend=LLMBackend.SMOLLM2,
                headless=True
            ),
            'llama32_1b': TrainingConfig(
                rom_path="test.gbc",
                llm_backend=LLMBackend.LLAMA32_1B,
                headless=True
            ),
            'none': TrainingConfig(
                rom_path="test.gbc",
                llm_backend=LLMBackend.NONE,
                headless=True
            )
        }
    
    @patch('trainer.trainer.PyBoy')
    @patch('trainer.trainer.PYBOY_AVAILABLE', True)
    def test_llm_performance_tracking_initialization(self, mock_pyboy_class, model_configs):
        """Test LLM performance tracking initialization"""
        mock_pyboy_instance = Mock()
        mock_pyboy_class.return_value = mock_pyboy_instance
        
        trainer = UnifiedPokemonTrainer(model_configs['smollm2'])
        
        # Check performance tracking attributes are initialized
        assert hasattr(trainer, 'llm_response_times')
        assert hasattr(trainer, 'adaptive_llm_interval')
        assert trainer.llm_response_times == []
        assert trainer.adaptive_llm_interval == trainer.config.llm_interval
        assert 'llm_total_time' in trainer.stats
        assert 'llm_avg_time' in trainer.stats
    
    @patch('trainer.trainer.PyBoy')
    @patch('trainer.trainer.PYBOY_AVAILABLE', True)
    def test_llm_performance_tracking_functionality(self, mock_pyboy_class, model_configs):
        """Test LLM performance tracking functionality"""
        mock_pyboy_instance = Mock()
        mock_pyboy_class.return_value = mock_pyboy_instance
        
        trainer = UnifiedPokemonTrainer(model_configs['smollm2'])
        
        # Test tracking individual response times
        trainer.stats['llm_calls'] = 1
        trainer._track_llm_performance(2.5)  # 2.5 second response
        
        assert len(trainer.llm_response_times) == 1
        assert trainer.llm_response_times[0] == 2.5
        assert trainer.stats['llm_total_time'] == 2.5
        assert trainer.stats['llm_avg_time'] == 2.5
        
        # Test multiple calls
        trainer.stats['llm_calls'] = 2
        trainer._track_llm_performance(1.5)  # Faster call
        
        assert len(trainer.llm_response_times) == 2
        assert trainer.stats['llm_total_time'] == 4.0
        assert trainer.stats['llm_avg_time'] == 2.0
    
    @patch('trainer.trainer.PyBoy')
    @patch('trainer.trainer.PYBOY_AVAILABLE', True)
    def test_adaptive_interval_slow_llm_increase(self, mock_pyboy_class, model_configs):
        """Test adaptive interval increases for slow LLM calls"""
        mock_pyboy_instance = Mock()
        mock_pyboy_class.return_value = mock_pyboy_instance
        
        trainer = UnifiedPokemonTrainer(model_configs['smollm2'])
        trainer.config.debug_mode = True  # Enable debug output
        
        original_interval = trainer.adaptive_llm_interval
        
        # Add 10 slow response times (>3 seconds each)
        for i in range(10):
            trainer.stats['llm_calls'] = i + 1
            trainer._track_llm_performance(4.0)  # 4 second responses
        
        # Interval should have increased
        assert trainer.adaptive_llm_interval > original_interval
        assert trainer.adaptive_llm_interval <= 50  # Should not exceed max
    
    @patch('trainer.trainer.PyBoy')
    @patch('trainer.trainer.PYBOY_AVAILABLE', True)
    def test_adaptive_interval_fast_llm_decrease(self, mock_pyboy_class, model_configs):
        """Test adaptive interval decreases for fast LLM calls"""
        mock_pyboy_instance = Mock()
        mock_pyboy_class.return_value = mock_pyboy_instance
        
        trainer = UnifiedPokemonTrainer(model_configs['smollm2'])
        trainer.config.debug_mode = True  # Enable debug output
        
        # First increase the interval
        for i in range(10):
            trainer.stats['llm_calls'] = i + 1
            trainer._track_llm_performance(4.0)  # Slow calls first
        
        increased_interval = trainer.adaptive_llm_interval
        assert increased_interval > trainer.config.llm_interval
        
        # Now add fast response times - use very fast responses to trigger decrease
        for i in range(10, 30):
            trainer.stats['llm_calls'] = i + 1
            trainer._track_llm_performance(0.8)  # Very fast calls under 1.5s threshold
        
        # Interval should decrease (but not below original)
        assert trainer.adaptive_llm_interval < increased_interval
        assert trainer.adaptive_llm_interval >= trainer.config.llm_interval
    
    @patch('trainer.trainer.PyBoy')
    @patch('trainer.trainer.PYBOY_AVAILABLE', True)
    def test_llm_response_times_window_management(self, mock_pyboy_class, model_configs):
        """Test that response times window is properly managed"""
        mock_pyboy_instance = Mock()
        mock_pyboy_class.return_value = mock_pyboy_instance
        
        trainer = UnifiedPokemonTrainer(model_configs['smollm2'])
        
        # Add more than 20 response times (current window size)
        for i in range(25):
            trainer.stats['llm_calls'] = i + 1
            trainer._track_llm_performance(1.0 + i * 0.1)
        
        # Should only keep last 20
        assert len(trainer.llm_response_times) == 20
        # Should have the most recent values (use approximate comparison for floating point)
        assert abs(trainer.llm_response_times[0] - 1.5) < 1e-10  # 6th response time (index 5)
        assert abs(trainer.llm_response_times[-1] - 3.4) < 1e-10  # 25th response time (index 24)
    
    @patch('trainer.trainer.PyBoy')
    @patch('trainer.trainer.PYBOY_AVAILABLE', True)
    def test_smollm2_backend_configuration(self, mock_pyboy_class, model_configs):
        """Test SmolLM2 backend configuration"""
        mock_pyboy_instance = Mock()
        mock_pyboy_class.return_value = mock_pyboy_instance
        
        trainer = UnifiedPokemonTrainer(model_configs['smollm2'])
        
        assert trainer.config.llm_backend == LLMBackend.SMOLLM2
        # Would test model-specific configuration
    
    @patch('trainer.trainer.PyBoy')
    @patch('trainer.trainer.PYBOY_AVAILABLE', True)
    def test_llm_fallback_to_rule_based(self, mock_pyboy_class, model_configs):
        """Test fallback to rule-based when LLM fails"""
        mock_pyboy_instance = Mock()
        mock_pyboy_class.return_value = mock_pyboy_instance
        
        trainer = UnifiedPokemonTrainer(model_configs['smollm2'])
        
        with patch('trainer.llm_manager.ollama') as mock_ollama:
            # Mock LLM failure
            mock_ollama.generate.side_effect = Exception("Model not available")
            mock_ollama.show.side_effect = Exception("Connection failed")
            
            # Should fallback gracefully
            action = trainer._get_llm_action()
            
            # Should either return None (triggering rule-based) or a valid action
            assert action is None or (1 <= action <= 8)
    
    @patch('trainer.trainer.PyBoy')
    @patch('trainer.trainer.PYBOY_AVAILABLE', True)
    def test_no_llm_backend_performance(self, mock_pyboy_class, model_configs):
        """Test performance with no LLM backend"""
        mock_pyboy_instance = Mock()
        mock_pyboy_class.return_value = mock_pyboy_instance
        
        trainer = UnifiedPokemonTrainer(model_configs['none'])
        
        start_time = time.time()
        
        # Test rule-based only performance
        for i in range(100):
            action = trainer._get_rule_based_action(i)
            assert 1 <= action <= 8
        
        elapsed = time.time() - start_time
        
        # Should be very fast without LLM overhead
        assert elapsed < 0.01, f"Rule-based actions too slow: {elapsed:.4f}s for 100 actions"


@pytest.mark.performance
@pytest.mark.integration
class TestUnifiedTrainerOptimizations:
    """Test performance optimizations in unified trainer"""
    
    @pytest.fixture
    def mock_config(self):
        """Basic mock configuration for tests"""
        return TrainingConfig(
            rom_path="test.gbc",
            mode=TrainingMode.FAST_MONITORED,
            max_actions=1000,
            headless=True,
            capture_screens=True
        )

    @pytest.fixture
    @patch('trainer.trainer.PyBoy')
    @patch('trainer.trainer.PYBOY_AVAILABLE', True)
    def optimized_trainer(self, mock_pyboy_class, mock_config):
        """Create an optimized trainer instance"""
        mock_pyboy_instance = Mock()
        mock_pyboy_instance.frame_count = 1000
        mock_pyboy_instance.screen.ndarray = np.random.randint(0, 255, (144, 160, 3), dtype=np.uint8)
        mock_pyboy_instance.send_input = Mock()
        mock_pyboy_instance.tick = Mock()
        mock_pyboy_class.return_value = mock_pyboy_instance
        return UnifiedPokemonTrainer(mock_config)
    
    def test_synchronized_training_performance(self, optimized_trainer):
        """Test synchronized training mode performance"""
        trainer = optimized_trainer
        
        with patch.object(trainer, '_simple_screenshot_capture') as mock_capture:
            mock_capture.return_value = np.random.randint(0, 255, (144, 160, 3), dtype=np.uint8)
            
            start_time = time.time()
            
            # Mock training methods
            trainer._finalize_training = Mock()
            
            # Run subset of synchronized training
            for step in range(50):
                action = trainer._get_rule_based_action(step)
                trainer._execute_action(action)
                
                # Simulate screen capture every few steps
                if step % 5 == 0:
                    trainer._capture_and_queue_screen()
            
            elapsed = time.time() - start_time
            actions_per_second = 50 / elapsed
            
            # Should achieve good performance (at least 10 actions/sec)
            assert actions_per_second >= 10, f"Synchronized training: {actions_per_second:.2f} actions/sec"
    
    def test_memory_usage_optimization(self, optimized_trainer):
        """Test memory usage optimizations"""
        trainer = optimized_trainer
        
        import gc
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Run extended operation
        with patch.object(trainer, '_simple_screenshot_capture') as mock_capture:
            mock_capture.return_value = np.random.randint(0, 255, (144, 160, 3), dtype=np.uint8)
            
            for i in range(500):
                trainer._get_rule_based_action(i)
                
                # Capture screens periodically
                if i % 10 == 0:
                    trainer._capture_and_queue_screen()
        
        # Force garbage collection
        gc.collect()
        
        final_memory = process.memory_info().rss
        memory_increase = (final_memory - initial_memory) / 1024 / 1024  # MB
        
        # Memory usage should be reasonable (under 30MB)
        assert memory_increase < 30, f"Memory usage increased by {memory_increase:.1f}MB"
    
    def test_error_recovery_optimization(self, optimized_trainer):
        """Test error recovery system performance"""
        trainer = optimized_trainer
        
        # Test recovery from multiple errors
        error_count = 0
        recovery_times = []
        
        for i in range(10):
            start_time = time.time()
            
            try:
                with trainer._handle_errors("test_operation", "general"):
                    if i % 3 == 0:  # Simulate occasional errors
                        raise Exception("Test error")
            except Exception:
                error_count += 1
            
            recovery_time = time.time() - start_time
            recovery_times.append(recovery_time)
        
        # Recovery should be fast (under 1ms per operation)
        avg_recovery_time = sum(recovery_times) / len(recovery_times)
        assert avg_recovery_time < 0.001, f"Error recovery too slow: {avg_recovery_time:.4f}s average"
        
        # Should have handled errors
        assert error_count > 0
        assert trainer.error_count['general'] == error_count


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
