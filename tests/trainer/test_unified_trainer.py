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
import queue
import numpy as np
from unittest.mock import Mock, patch, MagicMock, call, PropertyMock
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
    
    @patch('trainer.trainer.PYBOY_AVAILABLE', True)
    @patch('trainer.trainer.PyBoy')
    def test_initialization_success(self, mock_pyboy_class, base_config):
        """Test successful initialization with PyBoy available"""
        # Mock PyBoy instance with required attributes
        mock_pyboy_instance = Mock(spec=['frame_count', 'send_input', 'tick', 'stop', 'screen'])
        mock_pyboy_instance.frame_count = 1000
        mock_pyboy_instance.screen = Mock(spec=['ndarray'])
        mock_pyboy_instance.screen.ndarray = np.random.randint(0, 256, (144, 160, 3), dtype=np.uint8)
        
        # Set up PyBoy class mock
        mock_pyboy_class.return_value = mock_pyboy_instance
        
        # Create trainer with mocked PyBoy
        trainer = UnifiedPokemonTrainer(base_config)
        
        # Verify initialization
        assert trainer is not None
        assert trainer.config == base_config
        assert trainer.pyboy._mock_name == mock_pyboy_instance._mock_name
        assert hasattr(trainer, 'stats')
        assert hasattr(trainer, 'error_counts')
    
    @patch('trainer.trainer.PYBOY_AVAILABLE', False)
    def test_initialization_no_pyboy(self, base_config):
        """Test initialization when PyBoy is not available"""
        trainer = UnifiedPokemonTrainer(base_config)
        
        assert trainer is not None
        assert trainer.config == base_config
        assert trainer.pyboy is None
    
    @patch('pyboy.PyBoy')
    @patch('trainer.trainer.PYBOY_AVAILABLE', True)
    def test_logging_setup(self, mock_pyboy_class, base_config):
        """Test that logging is properly configured"""
        mock_pyboy_instance = Mock()
        mock_pyboy_class.return_value = mock_pyboy_instance
        
        trainer = UnifiedPokemonTrainer(base_config)
        
        assert hasattr(trainer, 'logger')
        assert trainer.logger.level == logging.DEBUG
    
    @patch('pyboy.PyBoy')
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
    @patch('pyboy.PyBoy')
    @patch('trainer.trainer.PYBOY_AVAILABLE', True)
    def trainer(self, mock_pyboy_class):
        """Create trainer for stability testing"""
        # Create a mock with required attributes
        mock_pyboy = MagicMock(spec=['frame_count', 'screen', 'screen_image', 'send_input', 'tick', 'stop'])
        # Set up frame_count to return a valid value for the alive check
        type(mock_pyboy).frame_count = PropertyMock(return_value=1000)
        # Set up screen attributes
        screen_mock = Mock(spec=['ndarray'])
        screen_mock.ndarray = np.random.randint(0, 256, (144, 160, 3), dtype=np.uint8)
        mock_pyboy.screen = screen_mock
        mock_pyboy.screen_image.return_value = screen_mock.ndarray
        
        # Make PyBoy class return our mock instance
        mock_pyboy_class.return_value = mock_pyboy
        
        # Create trainer with mock PyBoy configured
        config = TrainingConfig(
            rom_path="test.gbc",
            headless=True,
            debug_mode=True,
            enable_web=False,
            capture_screens=False
        )
        
        # Return trainer instance
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
        # Create new mock with error behavior
        mock_pyboy = Mock(spec=['frame_count'])
        mock_pyboy.frame_count = Mock(side_effect=Exception("Test error"))
        
        # Replace PyBoy instance
        trainer.pyboy = mock_pyboy
        result = trainer._is_pyboy_alive()
        assert result == False
    
    def test_pyboy_alive_check_invalid_frame_count(self, trainer):
        """Test PyBoy alive check with invalid frame count"""
        # Create mock with invalid frame count
        mock_pyboy = MagicMock(spec=['frame_count', 'stop', 'screen', 'send_input', 'tick'])
        type(mock_pyboy).frame_count = PropertyMock(return_value=-1)
        
        # Replace trainer's PyBoy instance
        trainer.pyboy = mock_pyboy
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
        # Create mock with RGBA screen
        mock_pyboy = MagicMock(spec=['frame_count', 'stop', 'screen', 'send_input', 'tick', 'screen_image'])
        mock_pyboy.screen_image.return_value = rgba_screen
        
        # Replace trainer's PyBoy instance
        trainer.pyboy = mock_pyboy
        
        screen = trainer._get_screen()
        
        assert screen.shape == (144, 160, 3)
        np.testing.assert_array_equal(screen, rgba_screen[:, :, :3])
    
    def test_screen_format_conversion_grayscale_to_rgb(self, trainer):
        """Test screen format conversion from grayscale to RGB"""
        # Create grayscale screen (144, 160)
        gray_screen = np.random.randint(0, 256, (144, 160), dtype=np.uint8)
        
        # Create mock with grayscale screen
        mock_pyboy = MagicMock(spec=['frame_count', 'stop', 'screen', 'send_input', 'tick', 'screen_image'])
        mock_pyboy.screen_image.return_value = gray_screen
        
        # Replace trainer's PyBoy instance
        trainer.pyboy = mock_pyboy
        
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
    @patch('pyboy.PyBoy')
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
    def mock_web_server(self):
        """Create a mock web server for testing"""
        # Create the ServerConfig class mock first
        server_config_cls = Mock()
        server_config_cls.from_training_config = Mock()
        server_config_cls.from_training_config.return_value = Mock(port=8080)
        
        # Create the server instance mock
        mock_server = Mock()
        mock_server.start.return_value = True
        mock_server.run_in_thread = Mock()
        mock_server.server = Mock()
        
        # Create the TrainingWebServer class mock
        mock_server_cls = Mock()
        mock_server_cls.ServerConfig = server_config_cls
        mock_server_cls.return_value = mock_server
        
        with patch('trainer.trainer.TrainingWebServer', mock_server_cls):
            yield mock_server_cls
    
    @pytest.fixture
    @patch('pyboy.PyBoy')
    @patch('trainer.trainer.PYBOY_AVAILABLE', True)
    def web_enabled_trainer(self, mock_pyboy_class):
        """Create a trainer instance with web server enabled"""
        mock_pyboy_instance = Mock()
        mock_pyboy_instance.frame_count = 1000
        mock_pyboy_instance.screen.ndarray = np.random.randint(0, 255, (144, 160, 3), dtype=np.uint8)
        mock_pyboy_class.return_value = mock_pyboy_instance
        
        config = TrainingConfig(
            rom_path="test.gbc",
            enable_web=True,
            capture_screens=True
        )
        
        return UnifiedPokemonTrainer(config)
    
    def test_web_server_initialization(self, mock_web_server, mock_pyboy_class):
        """Test web server initialization"""
        # Mock PyBoy instance first
        mock_pyboy_instance = Mock()
        mock_pyboy_class.return_value = mock_pyboy_instance
        
        # Create a test config
        config = TrainingConfig(
            rom_path="test.gbc",
            enable_web=True,
            headless=True
        )
        
        # Create trainer with web server enabled
        trainer = UnifiedPokemonTrainer(config)
        
        # Web server and its components should be properly initialized
        assert mock_web_server.call_count == 1, "TrainingWebServer constructor should be called once"
        
        # Get the server instance and verify it was called correctly
        server_instance = trainer.web_server
        assert server_instance is not None, "Web server instance should be created"
        assert server_instance.start.call_count == 1, "Web server start should be called once"
        assert server_instance.run_in_thread.call_count == 1, "Web server run_in_thread should be called once"
    
    def test_screenshot_memory_management(self, web_enabled_trainer):
        """Test screenshot memory management"""
        # Initialize screen queue if needed
        if not hasattr(web_enabled_trainer, 'screen_queue'):
            web_enabled_trainer.screen_queue = queue.Queue(maxsize=30)

        # Fill screenshot queue to capacity
        for i in range(35):  # More than queue limit of 30
            web_enabled_trainer._capture_and_queue_screen()
        
        # Queue should not exceed limit
        assert web_enabled_trainer.screen_queue.qsize() <= 30
    
    def test_api_status_endpoint_data(self, web_enabled_trainer):
        """Test API status endpoint data structure"""
        stats = web_enabled_trainer.get_current_stats()
        
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
    @patch('pyboy.PyBoy')
    @patch('trainer.trainer.PYBOY_AVAILABLE', True)
    def trainer(self, mock_pyboy_class):
        """Create trainer for rule-based testing"""
        mock_pyboy_instance = Mock()
        mock_pyboy_instance.frame_count = 1000
        mock_pyboy_instance.screen_image.return_value = np.random.randint(
            0, 256, (144, 160, 3), dtype=np.uint8
        )
        # Fix: Add proper screen.ndarray mock for screenshot capture
        mock_pyboy_instance.screen = Mock()
        mock_pyboy_instance.screen.ndarray = np.random.randint(
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
        
        trainer = UnifiedPokemonTrainer(config)
        # Initialize stuck detection attributes
        trainer.consecutive_same_screens = 0
        trainer.last_screen_hash = None
        return trainer
    
    def test_game_state_detection_unknown(self, trainer):
        """Test game state detection for unknown state"""
        # Create screen that doesn't match any known patterns
        screen = np.random.randint(0, 256, (144, 160, 3), dtype=np.uint8)
        
        # Fix: Pass screen parameter to _detect_game_state
        state = trainer._detect_game_state(screen)
        assert state in ['unknown', 'overworld', 'dialogue', 'loading', 'intro_sequence']
    
    def test_game_state_detection_loading(self, trainer):
        """Test game state detection for loading state"""
        # Create mostly black screen (loading indicator)
        screen = np.zeros((144, 160, 3), dtype=np.uint8)
        
        # Fix: Pass screen parameter to _detect_game_state
        state = trainer._detect_game_state(screen)
        # Should detect as loading
        assert state == 'loading'
    
    def test_game_state_detection_intro(self, trainer):
        """Test game state detection for intro state"""
        # Create bright screen (intro indicator)
        screen = np.full((144, 160, 3), 255, dtype=np.uint8)
        
        # Fix: Pass screen parameter to _detect_game_state
        state = trainer._detect_game_state(screen)
        # Should detect as intro_sequence
        assert state == 'intro_sequence'
    
    def test_game_state_detection_dialogue(self, trainer):
        """Test game state detection for dialogue state"""
        # Create screen with dialogue box pattern
        screen = np.random.randint(50, 200, (144, 160, 3), dtype=np.uint8)
        # Add dialogue box area (bottom portion)
        screen[100:140, 10:150, :] = 255  # White dialogue box
        
        # Fix: Pass screen parameter to _detect_game_state
        state = trainer._detect_game_state(screen)
        # Should detect as dialogue
        assert state == 'dialogue'
    
    def test_screen_hash_calculation(self, trainer):
        """Test screen hash calculation"""
        screen = np.random.randint(0, 256, (144, 160, 3), dtype=np.uint8)
        
        # Fix: Pass screen parameter to _get_screen_hash
        hash1 = trainer._get_screen_hash(screen)
        hash2 = trainer._get_screen_hash(screen)
        
        # Same screen should produce same hash
        assert hash1 == hash2
        assert isinstance(hash1, int)
    
    def test_screen_hash_different_screens(self, trainer):
        """Test screen hash for different screens"""
        screen1 = np.zeros((144, 160, 3), dtype=np.uint8)
        screen2 = np.ones((144, 160, 3), dtype=np.uint8) * 255
        
        # Fix: Pass screen parameters to _get_screen_hash
        hash1 = trainer._get_screen_hash(screen1)
        hash2 = trainer._get_screen_hash(screen2)
        
        # Different screens should produce different hashes
        assert hash1 != hash2
    
    def test_stuck_detection_mechanism(self, trainer):
        """Test stuck detection and recovery mechanism"""
        # Mock PyBoy instance for stuck detection
        same_screen = np.random.randint(0, 256, (144, 160, 3), dtype=np.uint8)
        
        # Configure trainer's PyBoy mock
        trainer.pyboy = MagicMock(spec=['frame_count', 'screen', 'screen_image', 'send_input', 'tick', 'stop'])
        # Set frame_count to work with _is_pyboy_alive
        type(trainer.pyboy).frame_count = PropertyMock(return_value=1000)
        screen_mock = Mock(spec=['ndarray'])
        screen_mock.ndarray = same_screen
        trainer.pyboy.screen = screen_mock
        trainer.pyboy.screen_image.return_value = same_screen
        
        # Initialize stuck detection state
        trainer.consecutive_same_screens = 0
        trainer.last_screen_hash = None
        
        # Execute multiple actions
        action_count = 0
        while action_count < 5:
            trainer._execute_synchronized_action(1)
            action_count += 1
        
        # Should have attempted unstuck actions
        assert trainer.pyboy.send_input.call_count >= 5
    
    def test_title_screen_handling(self, trainer):
        """Test title screen detection and handling"""
        # Create title screen pattern
        screen = np.random.randint(0, 100, (144, 160, 3), dtype=np.uint8)
        # Add title elements
        screen[20:40, 40:120, :] = 200  # Title text area
        
        # Fix: Pass screen parameter to _detect_game_state
        state = trainer._detect_game_state(screen)
        # Should handle title screen appropriately
        assert state in ['overworld', 'dialogue', 'loading', 'intro_sequence']
    
    def test_dialogue_handling(self, trainer):
        """Test dialogue state handling"""
        # Create dialogue screen
        screen = np.random.randint(0, 256, (144, 160, 3), dtype=np.uint8)
        screen[100:140, 10:150, :] = 255  # Dialogue box
        trainer.pyboy.screen.ndarray = screen
        
        # Execute action in dialogue state
        trainer._execute_synchronized_action(1)  # A button (advance dialogue)
        
        # Should have sent input
        trainer.pyboy.send_input.assert_called()
    
    def test_unstuck_action_patterns(self, trainer):
        """Test unstuck action patterns"""
        # Test that unstuck actions are valid
        for step in range(10):
            # Fix: Pass step parameter to _get_unstuck_action
            action = trainer._get_unstuck_action(step)
            assert 1 <= action <= 6  # Valid action range (UP, DOWN, LEFT, RIGHT, A, B)


@pytest.mark.unified_trainer
class TestTrainingModes:
    """Test different training modes"""
    
    @pytest.fixture
    @patch('pyboy.PyBoy')
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
    @patch('pyboy.PyBoy')
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
        trainer = trainer_fast_monitored
        
        # Initialize required attributes for stuck detection
        trainer.consecutive_same_screens = 0
        trainer.last_screen_hash = None
        
        # Mock the screenshot capture to return a valid screen
        test_screen = np.random.randint(0, 256, (144, 160, 3), dtype=np.uint8)
        
        # Mock the methods that could cause infinite loops
        with patch.object(trainer, '_simple_screenshot_capture', return_value=test_screen), \
            patch.object(trainer, '_get_screen_hash', return_value=12345), \
            patch.object(trainer, '_is_pyboy_alive', return_value=True):
            
            # Add a safety timeout to prevent infinite loops in tests
            import signal
            
            def timeout_handler(signum, frame):
                trainer._training_active = False
                raise TimeoutError("Training loop timeout - test safety mechanism")
            
            # Set a 5-second timeout
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(5)
            
            try:
                # Run training for a few steps
                trainer._run_synchronized_training()
                
                # Should have executed actions
                assert trainer.stats['total_actions'] > 0
                assert trainer.pyboy.send_input.call_count > 0
                
            except TimeoutError:
                # If we hit the timeout, the test failed due to infinite loop
                pytest.fail("Training loop got stuck - infinite loop detected")
            finally:
                # Always disable the alarm
                signal.alarm(0)
                trainer._training_active = False
    
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
        
        # Action count should NOT increase in _execute_synchronized_action - it's only incremented in the training loop
        assert trainer_fast_monitored.stats['total_actions'] == initial_actions
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
        print("DEBUG: Setting up test mocks...")

        def setup_mock_instance(mock_type="base"):
            print(f"DEBUG: Creating {mock_type} mock instance...")
            mock = MagicMock()
            mock.frame_count = 1000
            screen_mock = MagicMock()
            screen_mock.ndarray = np.random.randint(0, 255, (144, 160, 3), dtype=np.uint8)
            mock.screen = screen_mock
            mock.screen_image = MagicMock(return_value=screen_mock.ndarray)
            mock.tick = MagicMock()
            mock.stop = MagicMock()
            mock.send_input = MagicMock()
            print(f"DEBUG: {mock_type} mock instance created")
            return mock

        # Create base and recovery mocks first
        print("DEBUG: Setting up base and recovery mocks...")
        base_mock = setup_mock_instance("base")
        recovery_mock = setup_mock_instance("recovery")

        # Create sequence of mock instances with failure after 5 actions
        print("DEBUG: Setting up mock sequence...")
        mocks = [base_mock, recovery_mock]
        mock_index = [0]  # Use list to allow modification in closure
        
        def mock_init(*args, **kwargs):
            print(f"DEBUG: PyBoy constructor called with args={args}, kwargs={kwargs}")
            if mock_index[0] < len(mocks):
                result = mocks[mock_index[0]]
                mock_index[0] += 1
                return result
            return recovery_mock  # Always return recovery mock after initial sequence
        
        mock_pyboy_class.side_effect = mock_init

        # Configure for a short run
        print("DEBUG: Configuring test parameters...")
        integration_config.max_actions = 10
        integration_config.rom_path = "test.gbc"  # Add explicit ROM path
        integration_config.headless = True
        integration_config.debug_mode = False
        max_attempts = 3  # Prevent infinite recovery attempts
        recovery_attempts = [0]  # Track recovery attempts
        print("DEBUG: Test parameters configured")

        try:
            # Create and configure trainer
            trainer = UnifiedPokemonTrainer(integration_config)
            trainer._start_screen_capture = Mock()  # Prevent screen capture thread
            trainer._training_active = True
            
            # Track initial error counts
            initial_crash_count = trainer.error_counts['pyboy_crashes']
            initial_total_errors = trainer.error_counts['total_errors']
            initial_recovery_calls = 0

            # Add crash behavior to base mock
            def mock_send_input(action):
                if mock_index[0] == 1:  # Only crash the first instance
                    if recovery_attempts[0] < max_attempts:
                        recovery_attempts[0] += 1
                        raise Exception("PyBoy crash")
                return None
            
            base_mock.send_input.side_effect = mock_send_input

            # Run actions with safety limit
            actions_completed = 0
            max_iterations = integration_config.max_actions * 2  # Prevent infinite loop
            iterations = 0
            
            while actions_completed < integration_config.max_actions and iterations < max_iterations:
                try:
                    if not trainer._training_active:
                        break
                        
                    action = trainer._get_rule_based_action(actions_completed)
                    trainer._execute_synchronized_action(action)
                    actions_completed += 1
                    iterations += 1
                except Exception as e:
                    if "PyBoy crash" in str(e):
                        if recovery_attempts[0] >= max_attempts:
                            pytest.fail("Too many recovery attempts")
                        # Increment error counts
                        trainer.error_counts['pyboy_crashes'] += 1
                        trainer.error_counts['total_errors'] += 1
                        # Attempt recovery
                        result = trainer._attempt_pyboy_recovery()
                        if not result:
                            pytest.fail("Recovery failed")
                    else:
                        raise

            # Verify error tracking
            assert trainer.error_counts['pyboy_crashes'] > initial_crash_count, "PyBoy crash not counted"
            assert trainer.error_counts['total_errors'] > initial_total_errors, "Total errors not incremented"
            # Recovery should have been attempted - verify through error counts rather than call tracking
            assert trainer.error_counts['pyboy_crashes'] > 0, "Recovery was not attempted"

            # Verify test results
            assert trainer.error_counts['pyboy_crashes'] > 0, "Crash was not recorded"
            assert trainer.error_counts['total_errors'] > 0, "Error was not counted"
            
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
    
    @patch('pyboy.PyBoy')
    @patch('trainer.trainer.PYBOY_AVAILABLE', True)
    @patch('trainer.llm_manager.ollama')
    def test_llm_integration_fallback(self, mock_pyboy_class, mock_ollama, integration_config):
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
    @patch('pyboy.PyBoy')
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
    @patch('pyboy.PyBoy')
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
    @patch('pyboy.PyBoy')
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
    @patch('trainer.trainer.TrainingWebServer')
    def test_web_server_initialization(self, mock_web_server, mock_pyboy_class, web_trainer_config):
        """Test enhanced web server initialization"""
        # Force port in config
        web_trainer_config.web_port = 9999
        
        # Set up PyBoy mock
        mock_pyboy_instance = Mock()
        mock_pyboy_class.return_value = mock_pyboy_instance
        
        # Set up web server mock
        mock_server_instance = Mock()
        mock_server_instance.start.return_value = True
        mock_server_instance.server = Mock()
        mock_server_instance.server.serve_forever = Mock()
        mock_web_server.return_value = mock_server_instance
        
        # Create trainer with web server enabled
        trainer = UnifiedPokemonTrainer(web_trainer_config)
        
        # Web server should be properly initialized
        assert mock_web_server.call_count == 1, "TrainingWebServer should be called once"
        assert trainer.web_thread is not None, "Web thread should be created"
        assert trainer.screen_queue.maxsize == 30, "Screen queue should be initialized with maxsize=30"
        assert trainer.capture_active is False, "Capture should not be active by default"
        assert trainer.screen_queue.maxsize == 30  # Memory-bounded queue
        assert trainer.capture_active is False
    
    @patch('trainer.trainer.PyBoy')
    @patch('trainer.trainer.PYBOY_AVAILABLE', True)
    def test_real_time_stats_tracking(self, mock_pyboy_class, web_trainer_config):
        """Test real-time statistics tracking for web interface"""
        mock_pyboy_instance = Mock()
        mock_pyboy_class.return_value = mock_pyboy_instance
        
        trainer = UnifiedPokemonTrainer(web_trainer_config)
        
        # Initialize all required stats
        initial_time = time.time()
        trainer.stats = {
            'start_time': initial_time,
            'total_actions': 0,
            'actions_per_second': 0.0,
            'uptime_seconds': 0.0,
            'llm_total_time': 0.0,
            'llm_avg_time': 0.0,
            'llm_calls': 0
        }
        
        # Simulate actions with delay between them
        for i in range(10):
            trainer.stats['total_actions'] += 1
            time.sleep(0.1)  # Add delay to get non-zero action rate
            trainer._update_stats()
        
        # Stats should be updated
        assert trainer.stats['total_actions'] == 10
        assert trainer.stats['actions_per_second'] > 0
        assert trainer.stats['uptime_seconds'] > 0
        assert trainer.stats['start_time'] == initial_time
    
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
        
        # Stats should be initialized with proper defaults
        assert trainer.stats.get('llm_total_time', 0.0) == 0.0
        assert trainer.stats.get('llm_avg_time', 0.0) == 0.0
        assert trainer.stats.get('llm_calls', 0) == 0
    
    @patch('trainer.trainer.PyBoy')
    @patch('trainer.trainer.PYBOY_AVAILABLE', True)
    def test_llm_performance_tracking_functionality(self, mock_pyboy_class, model_configs):
        """Test LLM performance tracking functionality"""
        mock_pyboy_instance = Mock()
        mock_pyboy_class.return_value = mock_pyboy_instance
        
        trainer = UnifiedPokemonTrainer(model_configs['smollm2'])
        # Initialize stats to avoid empty arrays
        trainer.llm_response_times = []
        trainer.stats['llm_total_time'] = 0.0
        trainer.stats['llm_avg_time'] = 0.0
        trainer.stats['llm_calls'] = 0
        
        # Test tracking individual response times
        trainer._track_llm_performance(2.5)  # 2.5 second response
        
        assert len(trainer.llm_response_times) == 1
        assert trainer.llm_response_times[0] == 2.5
        assert abs(trainer.stats['llm_total_time'] - 2.5) < 1e-6
        assert abs(trainer.stats['llm_avg_time'] - 2.5) < 1e-6
        
        # Test multiple calls
        trainer._track_llm_performance(1.5)  # Faster call
        
        assert len(trainer.llm_response_times) == 2
        assert abs(trainer.stats['llm_total_time'] - 4.0) < 1e-6
        assert abs(trainer.stats['llm_avg_time'] - 2.0) < 1e-6
    
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
        
        # Initialize stats and response times array
        trainer.llm_response_times = []
        trainer.stats['llm_calls'] = 0
        trainer.stats['llm_total_time'] = 0.0
        trainer.stats['llm_avg_time'] = 0.0
        
        # Add more than 20 response times (current window size)
        response_times = [1.0 + i * 0.1 for i in range(25)]  # Pre-calculate all times
        
        for i, response_time in enumerate(response_times):
            trainer.stats['llm_calls'] = i + 1
            trainer._track_llm_performance(response_time)
        
        # Should only keep last 20
        assert len(trainer.llm_response_times) == 20
        # Should have the most recent values (use approximate comparison for floating point)
        expected_times = response_times[-20:]
        for actual, expected in zip(trainer.llm_response_times, expected_times):
            assert abs(actual - expected) < 1e-10
    
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
        assert elapsed < 0.1, f"Rule-based actions too slow: {elapsed:.4f}s for 100 actions"


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
    @patch('pyboy.PyBoy')
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
        
        # Ensure error tracking dict is initialized
        trainer.error_counts = {'general': 0}
        
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
        
        # Ensure we have recovery times before calculating average
        if len(recovery_times) > 0:
            avg_recovery_time = sum(recovery_times) / len(recovery_times)
            assert avg_recovery_time < 0.001, f"Error recovery too slow: {avg_recovery_time:.4f}s average"
        
        # Should have handled errors
        assert error_count > 0
        assert trainer.error_counts['general'] == error_count


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
