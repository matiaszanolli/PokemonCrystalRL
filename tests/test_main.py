"""
Comprehensive tests for main.py entry point.

Tests the primary training script including argument parsing, system initialization,
error handling, and integration with all subsystems.
"""

import pytest
import argparse
import os
import tempfile
import signal
import logging
import sys
from unittest.mock import Mock, patch, MagicMock, call
from io import StringIO
import threading
import time

# Import the module under test
import main
from main import (
    parse_arguments,
    setup_logging,
    initialize_training_systems,
    graceful_shutdown,
    main as main_function
)


class TestArgumentParsing:
    """Test command line argument parsing."""

    def test_parse_arguments_minimal_required(self):
        """Test parsing with only required arguments."""
        test_args = ["test_rom.gbc"]

        with patch.object(sys, 'argv', ['main.py'] + test_args):
            args = parse_arguments()

        assert args.rom_path == "test_rom.gbc"
        assert args.max_actions == 5000  # default
        assert args.llm_model == "smollm2:1.7b"  # default
        assert args.enable_web is False  # default
        assert args.enable_dqn is False  # default

    def test_parse_arguments_all_options(self):
        """Test parsing with all available options."""
        test_args = [
            "test_rom.gbc",
            "--save-state", "test_save.state",
            "--max-actions", "1000",
            "--llm-model", "test-model",
            "--llm-base-url", "http://test:8080",
            "--llm-interval", "20",
            "--llm-temperature", "0.9",
            "--enable-web",
            "--web-port", "9090",
            "--web-host", "0.0.0.0",
            "--enable-dqn",
            "--dqn-model", "test_model.pth",
            "--dqn-learning-rate", "0.001",
            "--dqn-batch-size", "64",
            "--dqn-memory-size", "100000"
        ]

        with patch.object(sys, 'argv', ['main.py'] + test_args):
            args = parse_arguments()

        assert args.rom_path == "test_rom.gbc"
        assert args.save_state == "test_save.state"
        assert args.max_actions == 1000
        assert args.llm_model == "test-model"
        assert args.llm_base_url == "http://test:8080"
        assert args.llm_interval == 20
        assert args.llm_temperature == 0.9
        assert args.enable_web is True
        assert args.web_port == 9090
        assert args.web_host == "0.0.0.0"
        assert args.enable_dqn is True
        assert args.dqn_model == "test_model.pth"
        assert args.dqn_learning_rate == 0.001
        assert args.dqn_batch_size == 64
        assert args.dqn_memory_size == 100000

    def test_parse_arguments_missing_required(self):
        """Test that missing required arguments raise SystemExit."""
        with patch.object(sys, 'argv', ['main.py']):
            with pytest.raises(SystemExit):
                parse_arguments()

    def test_parse_arguments_invalid_types(self):
        """Test that invalid argument types raise SystemExit."""
        test_cases = [
            ["test_rom.gbc", "--max-actions", "invalid"],
            ["test_rom.gbc", "--llm-interval", "not_a_number"],
            ["test_rom.gbc", "--llm-temperature", "invalid_float"],
            ["test_rom.gbc", "--web-port", "not_a_port"],
        ]

        for test_args in test_cases:
            with patch.object(sys, 'argv', ['main.py'] + test_args):
                with pytest.raises(SystemExit):
                    parse_arguments()


class TestLoggingSetup:
    """Test logging configuration."""

    def test_setup_logging_creates_directory(self):
        """Test that setup_logging creates log directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = os.path.join(temp_dir, "test_logs")

            setup_logging(log_dir)

            assert os.path.exists(log_dir)
            assert os.path.isdir(log_dir)

    def test_setup_logging_creates_log_file(self):
        """Test that setup_logging creates log file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = os.path.join(temp_dir, "test_logs")

            setup_logging(log_dir)

            # Check that a log file was created
            log_files = [f for f in os.listdir(log_dir) if f.startswith("training_") and f.endswith(".log")]
            assert len(log_files) == 1

    def test_setup_logging_configures_logger(self):
        """Test that setup_logging properly configures logging."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = os.path.join(temp_dir, "test_logs")

            # Check that we can call setup_logging without error
            setup_logging(log_dir)

            # Test that we can log messages
            test_logger = logging.getLogger("test")
            test_logger.info("Test message")

            # Directory should exist
            assert os.path.exists(log_dir)


class TestSystemInitialization:
    """Test training system initialization."""

    @pytest.fixture
    def mock_args(self):
        """Create mock arguments for testing."""
        args = Mock()
        args.rom_path = "test_rom.gbc"
        args.save_state = None
        args.max_actions = 1000
        args.llm_model = "test-model"
        args.llm_base_url = "http://localhost:11434"
        args.llm_interval = 10
        args.llm_temperature = 0.7
        args.enable_web = False
        args.web_port = 8080
        args.web_host = "localhost"
        args.enable_dqn = False
        args.dqn_model = None
        args.dqn_learning_rate = 1e-4
        args.dqn_batch_size = 32
        args.dqn_memory_size = 50000
        args.dqn_training_freq = 4
        args.dqn_save_freq = 500
        args.log_dir = "logs"
        args.quiet = False
        return args

    @patch('main.LLMAgent')
    @patch('main.LLMTrainer')
    @patch('main.PokemonRewardCalculator')
    @patch('main.GameIntelligence')
    @patch('main.ExperienceMemory')
    @patch('main.StrategicContextBuilder')
    def test_initialize_training_systems_basic(self, mock_context, mock_memory,
                                               mock_intelligence, mock_reward,
                                               mock_trainer, mock_llm_agent, mock_args):
        """Test basic system initialization without DQN."""
        mock_trainer_instance = Mock()
        mock_trainer.return_value = mock_trainer_instance

        systems = initialize_training_systems(mock_args)

        # Check that all components were created
        mock_llm_agent.assert_called_once()
        mock_reward.assert_called_once()
        mock_intelligence.assert_called_once()
        mock_memory.assert_called_once()
        mock_context.assert_called_once()
        mock_trainer.assert_called_once()

        # Check that systems dictionary is properly structured
        assert 'trainer' in systems
        assert 'llm_agent' in systems
        assert 'reward_calculator' in systems
        assert 'game_intelligence' in systems
        assert systems['trainer'] == mock_trainer_instance

    @patch('main.DQNAgent')
    @patch('main.HybridAgent')
    @patch('main.LLMAgent')
    @patch('main.LLMTrainer')
    @patch('main.PokemonRewardCalculator')
    @patch('main.GameIntelligence')
    @patch('main.ExperienceMemory')
    @patch('main.StrategicContextBuilder')
    @patch('os.path.exists')
    def test_initialize_training_systems_with_dqn(self, mock_exists, mock_context,
                                                  mock_memory, mock_intelligence,
                                                  mock_reward, mock_trainer,
                                                  mock_llm_agent, mock_hybrid,
                                                  mock_dqn, mock_args):
        """Test system initialization with DQN enabled."""
        mock_args.enable_dqn = True
        mock_args.dqn_model = "test_model.pth"
        mock_exists.return_value = True

        mock_dqn_instance = Mock()
        mock_dqn.return_value = mock_dqn_instance
        mock_hybrid_instance = Mock()
        mock_hybrid.return_value = mock_hybrid_instance

        systems = initialize_training_systems(mock_args)

        # Check DQN agent creation
        mock_dqn.assert_called_once_with(
            state_size=32,
            action_size=8,
            learning_rate=mock_args.dqn_learning_rate,
            gamma=0.99,
            epsilon_start=0.9,
            epsilon_end=0.05,
            epsilon_decay=0.995,
            memory_size=mock_args.dqn_memory_size,
            batch_size=mock_args.dqn_batch_size,
            target_update=1000
        )

        # Check model loading
        mock_dqn_instance.load_model.assert_called_once_with("test_model.pth")

        # Check hybrid agent creation
        mock_hybrid.assert_called_once()

        assert 'dqn_agent' in systems
        assert 'hybrid_agent' in systems

    @patch('main.LLMAgent')
    @patch('main.LLMTrainer')
    @patch('main.PokemonRewardCalculator')
    @patch('main.GameIntelligence')
    @patch('main.ExperienceMemory')
    @patch('main.StrategicContextBuilder')
    def test_initialize_training_systems_trainer_config(self, mock_context, mock_memory,
                                                        mock_intelligence, mock_reward,
                                                        mock_trainer, mock_llm_agent, mock_args):
        """Test that trainer is initialized with correct configuration."""
        mock_trainer_instance = Mock()
        mock_trainer.return_value = mock_trainer_instance

        initialize_training_systems(mock_args)

        # Check trainer was called with correct arguments
        mock_trainer.assert_called_once()
        call_args = mock_trainer.call_args

        # Verify essential parameters are passed correctly
        assert call_args.kwargs['rom_path'] == mock_args.rom_path
        assert call_args.kwargs['max_actions'] == mock_args.max_actions
        assert call_args.kwargs['llm_model'] == mock_args.llm_model
        assert call_args.kwargs['enable_web'] == mock_args.enable_web
        assert call_args.kwargs['enable_dqn'] == mock_args.enable_dqn
        assert call_args.kwargs['show_progress'] == True  # ~quiet


class TestGracefulShutdown:
    """Test graceful shutdown functionality."""

    def test_graceful_shutdown_with_trainer(self):
        """Test graceful shutdown with trainer cleanup."""
        mock_trainer = Mock()
        systems = {'trainer': mock_trainer}

        graceful_shutdown(systems)

        mock_trainer.stop_training.assert_called_once()

    def test_graceful_shutdown_without_trainer(self):
        """Test graceful shutdown without trainer (shouldn't crash)."""
        systems = {}

        # Should not raise exception
        graceful_shutdown(systems)

    def test_graceful_shutdown_with_signal(self):
        """Test graceful shutdown with signal parameters."""
        mock_trainer = Mock()
        systems = {'trainer': mock_trainer}

        with pytest.raises(SystemExit):
            graceful_shutdown(systems, signal.SIGTERM, None)

        mock_trainer.stop_training.assert_called_once()

    def test_graceful_shutdown_trainer_exception(self):
        """Test graceful shutdown handles trainer cleanup exceptions."""
        mock_trainer = Mock()
        mock_trainer.stop_training.side_effect = Exception("Cleanup error")
        systems = {'trainer': mock_trainer}

        # Should not raise exception even if cleanup fails
        graceful_shutdown(systems)

        mock_trainer.stop_training.assert_called_once()


class TestMainFunction:
    """Test the main function integration."""

    @patch('main.graceful_shutdown')
    @patch('main.initialize_training_systems')
    @patch('main.setup_logging')
    @patch('main.parse_arguments')
    @patch('signal.signal')
    def test_main_function_success(self, mock_signal, mock_parse, mock_logging,
                                   mock_init, mock_shutdown):
        """Test successful main function execution."""
        # Setup mocks
        mock_args = Mock()
        mock_args.log_dir = "test_logs"
        mock_parse.return_value = mock_args

        mock_trainer = Mock()
        mock_trainer.training_thread = None
        mock_systems = {'trainer': mock_trainer}
        mock_init.return_value = mock_systems

        # Run main function
        main_function()

        # Verify call sequence
        mock_parse.assert_called_once()
        mock_logging.assert_called_once_with(mock_args.log_dir)
        mock_init.assert_called_once_with(mock_args)
        mock_trainer.start_training.assert_called_once()
        mock_shutdown.assert_called_once_with(mock_systems)

        # Verify signal handlers were set up
        assert mock_signal.call_count == 2
        signal_calls = [call[0][0] for call in mock_signal.call_args_list]
        assert signal.SIGINT in signal_calls
        assert signal.SIGTERM in signal_calls

    @patch('main.graceful_shutdown')
    @patch('main.initialize_training_systems')
    @patch('main.setup_logging')
    @patch('main.parse_arguments')
    def test_main_function_training_exception(self, mock_parse, mock_logging,
                                              mock_init, mock_shutdown):
        """Test main function handles training exceptions."""
        # Setup mocks
        mock_args = Mock()
        mock_args.log_dir = "test_logs"
        mock_parse.return_value = mock_args

        mock_trainer = Mock()
        mock_trainer.start_training.side_effect = Exception("Training failed")
        mock_systems = {'trainer': mock_trainer}
        mock_init.return_value = mock_systems

        # Run main function - should raise exception
        with pytest.raises(Exception, match="Training failed"):
            main_function()

        # Verify cleanup still happens
        mock_shutdown.assert_called_once_with(mock_systems)

    @patch('main.graceful_shutdown')
    @patch('main.initialize_training_systems')
    @patch('main.setup_logging')
    @patch('main.parse_arguments')
    def test_main_function_with_training_thread(self, mock_parse, mock_logging,
                                                mock_init, mock_shutdown):
        """Test main function waits for training thread completion."""
        # Setup mocks
        mock_args = Mock()
        mock_args.log_dir = "test_logs"
        mock_parse.return_value = mock_args

        mock_thread = Mock()
        mock_trainer = Mock()
        mock_trainer.training_thread = mock_thread
        mock_systems = {'trainer': mock_trainer}
        mock_init.return_value = mock_systems

        # Run main function
        main_function()

        # Verify thread join was called
        mock_thread.join.assert_called_once()
        mock_shutdown.assert_called_once_with(mock_systems)


class TestSignalHandling:
    """Test signal handling and graceful shutdown integration."""

    @patch('main.graceful_shutdown')
    @patch('main.initialize_training_systems')
    @patch('main.setup_logging')
    @patch('main.parse_arguments')
    def test_signal_handler_registration(self, mock_parse, mock_logging,
                                         mock_init, mock_shutdown):
        """Test that signal handlers are properly registered."""
        mock_args = Mock()
        mock_args.log_dir = "test_logs"
        mock_parse.return_value = mock_args

        mock_trainer = Mock()
        mock_trainer.training_thread = None
        mock_systems = {'trainer': mock_trainer}
        mock_init.return_value = mock_systems

        # Capture signal handlers
        original_sigint = signal.signal(signal.SIGINT, signal.SIG_DFL)
        original_sigterm = signal.signal(signal.SIGTERM, signal.SIG_DFL)

        try:
            main_function()

            # Verify handlers were set (they should be different from default)
            current_sigint = signal.signal(signal.SIGINT, signal.SIG_DFL)
            current_sigterm = signal.signal(signal.SIGTERM, signal.SIG_DFL)

            assert current_sigint != signal.SIG_DFL
            assert current_sigterm != signal.SIG_DFL

        finally:
            # Restore original handlers
            signal.signal(signal.SIGINT, original_sigint)
            signal.signal(signal.SIGTERM, original_sigterm)


@pytest.mark.integration
class TestMainIntegration:
    """Integration tests for main entry point."""

    def test_main_with_help_flag(self):
        """Test main script with help flag."""
        with patch.object(sys, 'argv', ['main.py', '--help']):
            with pytest.raises(SystemExit) as exc_info:
                parse_arguments()

            # Help should exit with code 0
            assert exc_info.value.code == 0

    @patch('main.initialize_training_systems')
    @patch('main.setup_logging')
    def test_main_minimal_integration(self, mock_logging, mock_init):
        """Test main function with minimal valid arguments."""
        mock_trainer = Mock()
        mock_trainer.training_thread = None
        mock_systems = {'trainer': mock_trainer}
        mock_init.return_value = mock_systems

        test_args = ['main.py', 'test_rom.gbc', '--max-actions', '10']

        with patch.object(sys, 'argv', test_args):
            main_function()

        # Verify basic flow completed
        mock_logging.assert_called_once()
        mock_init.assert_called_once()
        mock_trainer.start_training.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])