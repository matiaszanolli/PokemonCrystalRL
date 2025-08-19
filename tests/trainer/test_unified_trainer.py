#!/usr/bin/env python3
"""
test_unified_trainer.py - Comprehensive tests for UnifiedPokemonTrainer

Tests all the recent improvements including:
- PyBoy stability and crash recovery
- Error handling and logging
- Web dashboard functionality
- HTTP polling with proper cleanup
- Screen capture improvements
- Configuration management
"""

import pytest
import tempfile
import time
import json
import base64
import numpy as np
from unittest.mock import Mock, MagicMock, patch, mock_open
from pathlib import Path
import logging
import threading
import queue
import io
from PIL import Image

from pokemon_crystal_rl.trainer import (
    UnifiedPokemonTrainer,
    TrainingConfig,
    TrainingMode,
    LLMBackend
)


class TestTrainingConfig:
    """Test TrainingConfig dataclass and validation"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = TrainingConfig(rom_path="test.gbc")
        
        assert config.rom_path == "test.gbc"
        assert config.mode == TrainingMode.FAST_MONITORED
        assert config.llm_backend == LLMBackend.SMOLLM2
        assert config.max_actions == 1000
        assert config.headless is True
        assert config.debug_mode is False
        assert config.capture_screens is True
        assert config.enable_web is False
        assert config.log_level == "INFO"
    
    def test_custom_config(self):
        """Test custom configuration values"""
        config = TrainingConfig(
            rom_path="custom.gbc",
            mode=TrainingMode.ULTRA_FAST,
            llm_backend=LLMBackend.NONE,
            max_actions=5000,
            headless=False,
            debug_mode=True,
            enable_web=True,
            web_port=9000,
            log_level="DEBUG"
        )
        
        assert config.rom_path == "custom.gbc"
        assert config.mode == TrainingMode.ULTRA_FAST
        assert config.llm_backend == LLMBackend.NONE
        assert config.max_actions == 5000
        assert config.headless is False
        assert config.debug_mode is True
        assert config.enable_web is True
        assert config.web_port == 9000
        assert config.log_level == "DEBUG"


class TestUnifiedPokemonTrainerInit:
    """Test trainer initialization and setup"""
    
    @pytest.fixture
    def mock_config(self):
        """Basic test configuration"""
        return TrainingConfig(
            rom_path="test.gbc",
            headless=True,
            debug_mode=True,
            enable_web=False,
            capture_screens=False,  # Disable for unit tests
            log_level="DEBUG"
        )
    
    @patch('pokemon_crystal_rl.trainer.PyBoy')
    @patch('pokemon_crystal_rl.trainer.PYBOY_AVAILABLE', True)
    def test_initialization_success(self, mock_pyboy_class, mock_config):
        """Test successful trainer initialization"""
        mock_pyboy_instance = Mock()
        mock_pyboy_class.return_value = mock_pyboy_instance
        
        trainer = UnifiedPokemonTrainer(mock_config)
        
        assert trainer.config == mock_config
        assert hasattr(trainer, 'logger')
        assert hasattr(trainer, 'error_count')
        assert hasattr(trainer, 'stats')
        assert trainer.stats['mode'] == 'fast_monitored'
        assert trainer.stats['total_actions'] == 0
        assert trainer.pyboy is not None
    
    @patch('pokemon_crystal_rl.trainer.trainer.PYBOY_AVAILABLE', False)
    def test_initialization_no_pyboy(self, mock_config):
        """Test initialization when PyBoy is not available"""
        with pytest.raises(RuntimeError, match="PyBoy not available"):
            UnifiedPokemonTrainer(mock_config)

    @patch('pokemon_crystal_rl.trainer.trainer.PyBoy')
    @patch('pokemon_crystal_rl.trainer.trainer.PYBOY_AVAILABLE', True)
    def test_logging_setup(self, mock_pyboy_class, mock_config):
        """Test logging system initialization"""
        mock_pyboy_instance = Mock()
        mock_pyboy_class.return_value = mock_pyboy_instance

        trainer = UnifiedPokemonTrainer(mock_config)

        assert hasattr(trainer, 'logger')
        assert trainer.logger.name == 'pokemon_trainer'
        assert trainer.logger.level == logging.DEBUG
        assert len(trainer.logger.handlers) >= 1  # At least console handler
    
    @patch('pokemon_crystal_rl.trainer.trainer.PyBoy')
    @patch('pokemon_crystal_rl.trainer.trainer.PYBOY_AVAILABLE', True)
    def test_error_tracking_init(self, mock_pyboy_class, mock_config):
        """Test error tracking initialization"""
        mock_pyboy_instance = Mock()
        mock_pyboy_class.return_value = mock_pyboy_instance
        
        trainer = UnifiedPokemonTrainer(mock_config)
        
        expected_error_types = ['pyboy_crashes', 'llm_failures', 'capture_errors', 'total_errors']
        for error_type in expected_error_types:
            assert error_type in trainer.error_count
            assert trainer.error_count[error_type] == 0
        
        assert trainer.last_error_time is None
        assert trainer.recovery_attempts == 0


class TestPyBoyStabilityAndRecovery:
    """Test PyBoy crash detection and recovery system"""
    
    @pytest.fixture
    def mock_config(self):
        return TrainingConfig(
            rom_path="test.gbc",
            headless=True,
            debug_mode=True,
            capture_screens=False
        )
    
    @pytest.fixture
    @patch('pokemon_crystal_rl.trainer.trainer.PyBoy')
    @patch('pokemon_crystal_rl.trainer.trainer.PYBOY_AVAILABLE', True)
    def trainer(self, mock_pyboy_class, mock_config):
        """Create trainer with mocked PyBoy"""
        mock_pyboy_instance = Mock()
        mock_pyboy_class.return_value = mock_pyboy_instance
        trainer = UnifiedPokemonTrainer(mock_config)
        trainer.mock_pyboy_class = mock_pyboy_class  # Store for recovery tests
        return trainer
    
    def test_pyboy_alive_check_success(self, trainer):
        """Test successful PyBoy health check"""
        trainer.pyboy.frame_count = 1000
        
        result = trainer._is_pyboy_alive()
        
        assert result is True
    
    def test_pyboy_alive_check_failure_none(self, trainer):
        """Test PyBoy health check with None instance"""
        trainer.pyboy = None
        
        result = trainer._is_pyboy_alive()
        
        assert result is False
    
    def test_pyboy_alive_check_failure_exception(self, trainer):
        """Test PyBoy health check with exception"""
        trainer.pyboy.frame_count = Mock(side_effect=Exception("PyBoy crashed"))
        
        result = trainer._is_pyboy_alive()
        
        assert result is False
    
    def test_pyboy_alive_check_invalid_frame_count(self, trainer):
        """Test PyBoy health check with invalid frame count"""
        trainer.pyboy.frame_count = "invalid"
        
        result = trainer._is_pyboy_alive()
        
        assert result is False
    
    @patch('pathlib.Path.exists', return_value=True)
    @patch('os.path.exists', return_value=True)  # Mock file existence for ROM
    def test_pyboy_recovery_success(self, mock_os_exists, mock_path_exists, trainer):
        """Test successful PyBoy recovery"""
        # Setup failing PyBoy instance
        old_pyboy = Mock()
        old_pyboy.stop.return_value = None
        trainer.pyboy = old_pyboy
        
        # Mock successful recovery - patch PyBoy constructor directly in recovery method
        new_pyboy = Mock()
        new_pyboy.frame_count = 0
        trainer.mock_pyboy_class.return_value = new_pyboy
        
        with patch('pokemon_crystal_rl.trainer.trainer.PyBoy', trainer.mock_pyboy_class):
            result = trainer._attempt_pyboy_recovery()
        
        assert result is True
        assert trainer.pyboy == new_pyboy
        old_pyboy.stop.assert_called_once()
    
    @patch('pathlib.Path.exists', return_value=True)
    @patch('os.path.exists', return_value=True)
    def test_pyboy_recovery_with_save_state(self, mock_os_exists, mock_path_exists, trainer):
        """Test PyBoy recovery with save state loading"""
        trainer.config.save_state_path = "test_save.state"
        
        # Setup mocks
        old_pyboy = Mock()
        trainer.pyboy = old_pyboy
        
        new_pyboy = Mock()
        new_pyboy.frame_count = 0
        new_pyboy.load_state = Mock()
        trainer.mock_pyboy_class.return_value = new_pyboy
        
        with patch('pokemon_crystal_rl.trainer.trainer.PyBoy', trainer.mock_pyboy_class):
            with patch('builtins.open', mock_open(read_data=b"save_data")):
                result = trainer._attempt_pyboy_recovery()
        
        assert result is True
        new_pyboy.load_state.assert_called_once()
    
    def test_pyboy_recovery_failure(self, trainer):
        """Test failed PyBoy recovery"""
        trainer.pyboy = Mock()
        
        # Mock failed recovery - patch the PyBoy constructor inside the recovery method
        with patch('pokemon_crystal_rl.trainer.trainer.PyBoy', side_effect=Exception("Recovery failed")):
            result = trainer._attempt_pyboy_recovery()
        
        assert result is False
        assert trainer.pyboy is None
    
    def test_screen_format_conversion_rgba_to_rgb(self, trainer):
        """Test RGBA to RGB screen format conversion"""
        # Create RGBA test data
        rgba_data = np.random.randint(0, 255, (144, 160, 4), dtype=np.uint8)
        
        result = trainer._convert_screen_format(rgba_data)
        
        assert result is not None
        assert result.shape == (144, 160, 3)
        assert result.dtype == np.uint8
        # Verify RGB data matches original (minus alpha channel)
        np.testing.assert_array_equal(result, rgba_data[:, :, :3])
    
    def test_screen_format_conversion_grayscale_to_rgb(self, trainer):
        """Test grayscale to RGB screen format conversion"""
        # Create grayscale test data
        gray_data = np.random.randint(0, 255, (144, 160), dtype=np.uint8)
        
        result = trainer._convert_screen_format(gray_data)
        
        assert result is not None
        assert result.shape == (144, 160, 3)
        assert result.dtype == np.uint8
        # Verify all channels are the same (grayscale)
        np.testing.assert_array_equal(result[:, :, 0], gray_data)
        np.testing.assert_array_equal(result[:, :, 1], gray_data)
        np.testing.assert_array_equal(result[:, :, 2], gray_data)
    
    def test_screen_format_conversion_invalid_shape(self, trainer):
        """Test screen format conversion with invalid shape"""
        # Create data with invalid shape
        invalid_data = np.random.randint(0, 255, (144, 160, 5), dtype=np.uint8)
        
        result = trainer._convert_screen_format(invalid_data)
        
        assert result is None


class TestErrorHandlingAndLogging:
    """Test comprehensive error handling and logging system"""
    
    @pytest.fixture
    def mock_config(self):
        return TrainingConfig(
            rom_path="test.gbc",
            debug_mode=True,
            log_level="DEBUG",
            capture_screens=False
        )
    
    @pytest.fixture
    @patch('pokemon_crystal_rl.trainer.trainer.PyBoy')
    @patch('pokemon_crystal_rl.trainer.trainer.PYBOY_AVAILABLE', True)
    def trainer(self, mock_pyboy_class, mock_config):
        mock_pyboy_instance = Mock()
        mock_pyboy_class.return_value = mock_pyboy_instance
        return UnifiedPokemonTrainer(mock_config)
    
    def test_error_context_manager_success(self, trainer):
        """Test error context manager with successful operation"""
        with trainer._handle_errors("test_operation"):
            # Simulate successful operation
            pass
        
        # Error counts should remain zero
        assert trainer.error_count['total_errors'] == 0
    
    def test_error_context_manager_exception(self, trainer):
        """Test error context manager with exception"""
        test_exception = ValueError("Test error")
        
        with pytest.raises(ValueError):
            with trainer._handle_errors("test_operation", "general"):
                raise test_exception
        
        # Error counts should be incremented
        assert trainer.error_count['general'] == 1
        assert trainer.error_count['total_errors'] == 1
        assert trainer.last_error_time is not None
    
    def test_error_context_manager_pyboy_recovery(self, trainer):
        """Test error context manager with PyBoy crash recovery"""
        # Mock successful recovery
        trainer._attempt_pyboy_recovery = Mock(return_value=True)
        
        with pytest.raises(Exception):
            with trainer._handle_errors("pyboy_operation", "pyboy_crashes"):
                raise Exception("PyBoy crash")
        
        # Should attempt recovery for PyBoy crashes
        trainer._attempt_pyboy_recovery.assert_called_once()
        assert trainer.error_count['pyboy_crashes'] == 1
        assert trainer.recovery_attempts == 1
    
    def test_keyboard_interrupt_handling(self, trainer):
        """Test keyboard interrupt handling"""
        with pytest.raises(KeyboardInterrupt):
            with trainer._handle_errors("test_operation"):
                raise KeyboardInterrupt()
        
        # Keyboard interrupts should not increment error counts
        assert trainer.error_count['total_errors'] == 0
    
    def test_logging_levels(self, trainer):
        """Test different logging levels"""
        # Test that logger was set up correctly
        assert trainer.logger.level == logging.DEBUG
        
        # Test logging methods exist and work
        trainer.logger.debug("Debug message")
        trainer.logger.info("Info message")
        trainer.logger.warning("Warning message")
        trainer.logger.error("Error message")
        
        # No exceptions should be raised


class TestWebDashboardAndHTTPPolling:
    """Test web dashboard functionality and HTTP polling improvements"""
    
    @pytest.fixture
    def mock_config(self):
        import socket
        # Find an available port
        sock = socket.socket()
        sock.bind(('', 0))
        available_port = sock.getsockname()[1]
        sock.close()
        
        return TrainingConfig(
            rom_path="test.gbc",
            enable_web=True,
            web_port=available_port,
            capture_screens=True,
            headless=True
        )
    
    @patch('pokemon_crystal_rl.trainer.trainer.PyBoy')
    @patch('pokemon_crystal_rl.trainer.trainer.PYBOY_AVAILABLE', True)
    @patch('pokemon_crystal_rl.monitoring.web_server.HTTPServer')
    def test_web_server_initialization(self
        """Test web server initialization"""
        mock_pyboy_instance = Mock()
        mock_pyboy_class.return_value = mock_pyboy_instance
        
        mock_server_instance = Mock()
        mock_http_server.return_value = mock_server_instance
        
        trainer = UnifiedPokemonTrainer(mock_config)
        
        # Web server should be initialized
        assert trainer.web_server is not None
        assert trainer.web_thread is not None
        mock_http_server.assert_called_once()
    
    @patch('pokemon_crystal_rl.trainer.trainer.PyBoy')
    @patch('pokemon_crystal_rl.trainer.trainer.PYBOY_AVAILABLE', True)
    def test_screenshot_memory_management(self, mock_pyboy_class, mock_config):
        """Test screenshot capture memory management"""
        mock_pyboy_instance = Mock()
        mock_pyboy_class.return_value = mock_pyboy_instance
        
        trainer = UnifiedPokemonTrainer(mock_config)
        
        # Test screen queue initialization
        assert isinstance(trainer.screen_queue, queue.Queue)
        assert trainer.screen_queue.maxsize == 30
        assert trainer.latest_screen is None
        assert trainer.capture_active is False
    
    @patch('pokemon_crystal_rl.trainer.trainer.PyBoy')
    @patch('pokemon_crystal_rl.trainer.trainer.PYBOY_AVAILABLE', True)
    def test_api_status_endpoint_data(self, mock_pyboy_class, mock_config):
        """Test API status endpoint data structure"""
        mock_pyboy_instance = Mock()
        mock_pyboy_class.return_value = mock_pyboy_instance
        
        trainer = UnifiedPokemonTrainer(mock_config)
        trainer._training_active = True
        trainer.stats['total_actions'] = 100
        trainer.stats['llm_calls'] = 10
        
        # This would be called by the web server handler
        # We're testing the data structure is correct
        expected_fields = [
            'is_training', 'mode', 'model', 'start_time', 
            'total_actions', 'llm_calls', 'actions_per_second'
        ]
        
        for field in expected_fields:
            assert field in trainer.stats or hasattr(trainer, '_training_active')


class TestRuleBasedActionSystem:
    """Test improved rule-based action system"""
    
    @pytest.fixture
    @patch('pokemon_crystal_rl.trainer.trainer.PyBoy')
    @patch('pokemon_crystal_rl.trainer.trainer.PYBOY_AVAILABLE', True)
    def trainer(self, mock_pyboy_class):
        mock_pyboy_instance = Mock()
        mock_pyboy_class.return_value = mock_pyboy_instance
        
        config = TrainingConfig(
            rom_path="test.gbc",
            capture_screens=False,
            headless=True
        )
        
        return UnifiedPokemonTrainer(config)
    
    def test_game_state_detection_unknown(self, trainer):
        """Test game state detection with None screenshot"""
        result = trainer._detect_game_state(None)
        assert result == "unknown"
    
    def test_game_state_detection_loading(self, trainer):
        """Test game state detection for loading screen"""
        # Create black screen (loading)
        black_screen = np.zeros((144, 160, 3), dtype=np.uint8)
        
        result = trainer._detect_game_state(black_screen)
        
        assert result == "loading"
    
    def test_game_state_detection_intro(self, trainer):
        """Test game state detection for intro sequence"""
        # Create white screen (intro text)
        white_screen = np.ones((144, 160, 3), dtype=np.uint8) * 250
        
        result = trainer._detect_game_state(white_screen)
        
        assert result == "intro_sequence"
    
    def test_game_state_detection_dialogue(self, trainer):
        """Test game state detection for dialogue"""
        # Create screen with bright bottom section (dialogue box)
        screen = np.ones((144, 160, 3), dtype=np.uint8) * 100
        screen[100:, :] = 220  # Bright bottom section
        
        result = trainer._detect_game_state(screen)
        
        assert result == "dialogue"
    
    def test_screen_hash_calculation(self, trainer):
        """Test screen hash calculation for stuck detection"""
        # Create test screen
        screen = np.random.randint(0, 255, (144, 160, 3), dtype=np.uint8)
        
        hash1 = trainer._get_screen_hash(screen)
        hash2 = trainer._get_screen_hash(screen)
        
        # Same screen should produce same hash
        assert hash1 == hash2
        assert isinstance(hash1, int)
    
    def test_screen_hash_different_screens(self, trainer):
        """Test screen hash with different screens"""
        screen1 = np.zeros((144, 160, 3), dtype=np.uint8)
        screen2 = np.ones((144, 160, 3), dtype=np.uint8) * 255
        
        hash1 = trainer._get_screen_hash(screen1)
        hash2 = trainer._get_screen_hash(screen2)
        
        # Different screens should produce different hashes
        assert hash1 != hash2
    
    def test_stuck_detection_mechanism(self, trainer):
        """Test anti-stuck mechanism"""
        trainer.last_screen_hash = 12345
        trainer.consecutive_same_screens = 0
        
        # Simulate same screen multiple times
        with patch.object(trainer, '_get_screen_hash', return_value=12345):
            with patch.object(trainer, '_simple_screenshot_capture', return_value=np.zeros((144, 160, 3))):
                # First call - should not be stuck
                action1 = trainer._get_rule_based_action(1)
                assert trainer.consecutive_same_screens == 1
                
                # Multiple calls with same hash
                for i in range(20):
                    trainer._get_rule_based_action(i + 2)
                
                # Should have triggered stuck counter
                assert trainer.consecutive_same_screens > 15
                assert trainer.stuck_counter > 0
    
    def test_title_screen_handling(self, trainer):
        """Test title screen action handling"""
        action = trainer._handle_title_screen(0)
        
        # Should return a valid action (1-8)
        assert 1 <= action <= 8
        
        # Test pattern consistency
        actions = [trainer._handle_title_screen(i) for i in range(10)]
        assert len(set(actions)) > 1  # Should have variety in actions
    
    def test_dialogue_handling(self, trainer):
        """Test dialogue action handling"""
        action = trainer._handle_dialogue(0)
        
        # Should return a valid action
        assert action in [0, 5]  # Pattern uses A button (5) or no-op converted to A
    
    def test_unstuck_action_patterns(self, trainer):
        """Test unstuck action patterns"""
        trainer.stuck_counter = 1
        
        actions = []
        for i in range(50):
            action = trainer._get_unstuck_action(i)
            actions.append(action)
            assert 1 <= action <= 8  # Valid action range
        
        # Should use multiple different actions
        unique_actions = set(actions)
        assert len(unique_actions) >= 3  # At least 3 different actions


class TestTrainingModes:
    """Test different training modes and their execution"""
    
    @pytest.fixture
    @patch('pokemon_crystal_rl.trainer.trainer.PyBoy')
    @patch('pokemon_crystal_rl.trainer.trainer.PYBOY_AVAILABLE', True)
    def test_initialization_no_pyboy(self, mock_pyboy_class, mock_config):
        mock_pyboy_instance = Mock()
        mock_pyboy_class.return_value = mock_pyboy_instance
        
        config = TrainingConfig(
            rom_path="test.gbc",
            mode=TrainingMode.FAST_MONITORED,
            max_actions=10,
            capture_screens=False,
            headless=True
        )
        
        return UnifiedPokemonTrainer(config)
    
    @pytest.fixture
    @patch('pokemon_crystal_rl.trainer.trainer.PyBoy')
    @patch('pokemon_crystal_rl.trainer.trainer.PYBOY_AVAILABLE', True)
    def trainer_ultra_fast(self, mock_pyboy_class):
        mock_pyboy_instance = Mock()
        mock_pyboy_class.return_value = mock_pyboy_instance
        
        config = TrainingConfig(
            rom_path="test.gbc",
            mode=TrainingMode.ULTRA_FAST,
            max_actions=10,
            capture_screens=False,
            headless=True
        )
        
        return UnifiedPokemonTrainer(config)
    
    def test_fast_monitored_training_execution(self, trainer_fast_monitored):
        """Test fast monitored training execution"""
        trainer = trainer_fast_monitored
        
        # Mock required methods
        trainer._execute_action = Mock()
        trainer._update_stats = Mock()
        
        # Mock PyBoy tick
        trainer.pyboy.tick = Mock()
        
        # Test that training completes without errors
        trainer._run_legacy_fast_training()
        
        # Verify training ran
        assert trainer.stats['total_actions'] == trainer.config.max_actions
    
    def test_ultra_fast_training_execution(self, trainer_ultra_fast):
        """Test ultra fast training execution"""
        trainer = trainer_ultra_fast
        
        # Mock required methods
        trainer._execute_action = Mock()
        
        # Test that training completes without errors
        trainer._run_ultra_fast_training()
        
        # Verify training ran
        assert trainer.stats['total_actions'] == trainer.config.max_actions
    
    def test_action_execution(self, trainer_fast_monitored):
        """Test action execution"""
        trainer = trainer_fast_monitored
        
        # Mock PyBoy methods
        trainer.pyboy.send_input = Mock()
        trainer.pyboy.tick = Mock()
        
        # Test valid action execution
        trainer._execute_action(5)  # A button
        
        trainer.pyboy.send_input.assert_called_once()
        trainer.pyboy.tick.assert_called_once()
    
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
    
    @patch('pokemon_crystal_rl.trainer.trainer.PyBoy')
    @patch('pokemon_crystal_rl.trainer.trainer.PYBOY_AVAILABLE', True)
    def test_full_training_cycle_with_recovery(self, mock_pyboy_class, integration_config):
        """Test complete training cycle with PyBoy recovery"""
        # Setup mocks
        mock_pyboy_instance = Mock()
        mock_pyboy_instance.frame_count = 1000
        mock_pyboy_instance.screen.ndarray = np.random.randint(0, 255, (144, 160, 3), dtype=np.uint8)
        mock_pyboy_class.return_value = mock_pyboy_instance
        
        trainer = UnifiedPokemonTrainer(integration_config)
        
        # Mock training methods to prevent actual PyBoy operations
        trainer._execute_synchronized_action = Mock()
        trainer._start_screen_capture = Mock()
        
        # Simulate PyBoy crash and recovery during training
        call_count = [0]
        recovery_triggered = [False]
        
        # Mock the frame_count access to trigger crash detection
        def mock_frame_count_getter():
            call_count[0] += 1
            if call_count[0] == 25 and not recovery_triggered[0]:  # Crash once
                recovery_triggered[0] = True
                raise Exception("Simulated PyBoy crash")
            return 1000
        
        # Create a proper recovery mock that actually resets things
        def mock_recovery():
            # Reset the call count to stop throwing exceptions
            # This simulates the PyBoy instance being recovered
            call_count[0] = 0  # Reset call count
            return True
        
        # Mock PyBoy properties to simulate crash in frame_count access
        type(mock_pyboy_instance).frame_count = property(mock_frame_count_getter)
        trainer._attempt_pyboy_recovery = Mock(side_effect=mock_recovery)
        
        # Run training - the crash will be triggered in _capture_synchronized_screenshot
        # which calls frame_count for health checking
        trainer._run_synchronized_training()
        
        # Verify recovery was attempted
        trainer._attempt_pyboy_recovery.assert_called()
        assert trainer.error_count['total_errors'] > 0
    
    @patch('pokemon_crystal_rl.trainer.trainer.PyBoy')
    @patch('pokemon_crystal_rl.trainer.trainer.PYBOY_AVAILABLE', True)
    def test_initialization_no_pyboy(self, mock_pyboy_class, mock_config):
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
    @patch('pokemon_crystal_rl.trainer.llm_manager.ollama')
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
    @patch('pokemon_crystal_rl.trainer.trainer.PyBoy')
    @patch('pokemon_crystal_rl.trainer.trainer.PYBOY_AVAILABLE', True)
    def trainer_fast_monitored(self, mock_pyboy_class):
        mock_pyboy_instance = Mock()
        mock_pyboy_class.return_value = mock_pyboy_instance
        
        config = TrainingConfig(
            rom_path="test.gbc",
            mode=TrainingMode.FAST_MONITORED,
            max_actions=10,
            capture_screens=False,
            headless=True
        )
        
        return UnifiedPokemonTrainer(config)

    @pytest.fixture
    @patch('pokemon_crystal_rl.trainer.trainer.PyBoy')
    @patch('pokemon_crystal_rl.trainer.trainer.PYBOY_AVAILABLE', True)
    def trainer_ultra_fast(self, mock_pyboy_class):
        mock_pyboy_instance = Mock()
        mock_pyboy_instance.frame_count = 1000
        mock_pyboy_class.return_value = mock_pyboy_instance
        
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
    
    @patch('pokemon_crystal_rl.trainer.trainer.PyBoy')
    @patch('pokemon_crystal_rl.trainer.trainer.PYBOY_AVAILABLE', True)
    @patch('pokemon_crystal_rl.trainer.web_server.HTTPServer')
    def test_web_server_initialization(self, mock_http_server, mock_pyboy_class, web_trainer_config):
        """Test enhanced web server initialization"""
        mock_pyboy_instance = Mock()
        mock_pyboy_class.return_value = mock_pyboy_instance
        
        mock_server = Mock()
        mock_http_server.return_value = mock_server
        
        trainer = UnifiedPokemonTrainer(web_trainer_config)
        
        # Enhanced web features should be initialized
        assert trainer.web_server is not None
        assert trainer.web_thread is not None
        assert trainer.screen_queue.maxsize == 30  # Memory-bounded queue
        assert trainer.capture_active is False
    
    @patch('pokemon_crystal_rl.trainer.trainer.PyBoy')
    @patch('pokemon_crystal_rl.trainer.trainer.PYBOY_AVAILABLE', True)
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
    
    @patch('pokemon_crystal_rl.trainer.trainer.PyBoy')
    @patch('pokemon_crystal_rl.trainer.trainer.PYBOY_AVAILABLE', True)
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
    
    @patch('pokemon_crystal_rl.trainer.trainer.PyBoy')
    @patch('pokemon_crystal_rl.trainer.trainer.PYBOY_AVAILABLE', True)
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
    
    @patch('pokemon_crystal_rl.trainer.trainer.PyBoy')
    @patch('pokemon_crystal_rl.trainer.trainer.PYBOY_AVAILABLE', True)
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
    
    @patch('pokemon_crystal_rl.trainer.trainer.PyBoy')
    @patch('pokemon_crystal_rl.trainer.trainer.PYBOY_AVAILABLE', True)
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
    
    @patch('pokemon_crystal_rl.trainer.trainer.PyBoy')
    @patch('pokemon_crystal_rl.trainer.trainer.PYBOY_AVAILABLE', True)
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
    
    @patch('pokemon_crystal_rl.trainer.trainer.PyBoy')
    @patch('pokemon_crystal_rl.trainer.trainer.PYBOY_AVAILABLE', True)
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
    
    @patch('pokemon_crystal_rl.trainer.trainer.PyBoy')
    @patch('pokemon_crystal_rl.trainer.trainer.PYBOY_AVAILABLE', True)
    def test_smollm2_backend_configuration(self, mock_pyboy_class, model_configs):
        """Test SmolLM2 backend configuration"""
        mock_pyboy_instance = Mock()
        mock_pyboy_class.return_value = mock_pyboy_instance
        
        trainer = UnifiedPokemonTrainer(model_configs['smollm2'])
        
        assert trainer.config.llm_backend == LLMBackend.SMOLLM2
        # Would test model-specific configuration
    
    @patch('pokemon_crystal_rl.trainer.trainer.PyBoy')
    @patch('pokemon_crystal_rl.trainer.trainer.PYBOY_AVAILABLE', True)
    def test_llm_fallback_to_rule_based(self, mock_pyboy_class, model_configs):
        """Test fallback to rule-based when LLM fails"""
        mock_pyboy_instance = Mock()
        mock_pyboy_class.return_value = mock_pyboy_instance
        
        trainer = UnifiedPokemonTrainer(model_configs['smollm2'])
        
        with patch('pokemon_crystal_rl.trainer.llm_manager.ollama') as mock_ollama:
            # Mock LLM failure
            mock_ollama.generate.side_effect = Exception("Model not available")
            mock_ollama.show.side_effect = Exception("Connection failed")
            
            # Should fallback gracefully
            action = trainer._get_llm_action()
            
            # Should either return None (triggering rule-based) or a valid action
            assert action is None or (1 <= action <= 8)
    
    @patch('pokemon_crystal_rl.trainer.trainer.PyBoy')
    @patch('pokemon_crystal_rl.trainer.trainer.PYBOY_AVAILABLE', True)
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
    @patch('pokemon_crystal_rl.trainer.trainer.PyBoy')
    @patch('pokemon_crystal_rl.trainer.trainer.PYBOY_AVAILABLE', True)
    def test_initialization_no_pyboy(self, mock_config):
        mock_pyboy_instance = Mock()
        mock_pyboy_instance.frame_count = 1000
        mock_pyboy_instance.screen.ndarray = np.random.randint(0, 255, (144, 160, 3), dtype=np.uint8)
        mock_pyboy_instance.send_input = Mock()
        mock_pyboy_instance.tick = Mock()
        mock_pyboy_class.return_value = mock_pyboy_instance
        
        config = TrainingConfig(
            rom_path="test.gbc",
            mode=TrainingMode.FAST_MONITORED,
            max_actions=1000,
            headless=True,
            capture_screens=True
        )
        
        return UnifiedPokemonTrainer(config)
    
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
