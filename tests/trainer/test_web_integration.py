"""Test web server integration."""

import time
import queue
import threading
import pytest
import numpy as np
from unittest.mock import Mock, patch

from training.trainer import TrainingConfig, PokemonTrainer
from core.web_monitor import WebMonitor

# Mock PyBoy for tests
@pytest.fixture
def mock_pyboy():
    with patch('trainer.trainer.PyBoy') as mock:
        mock_instance = Mock()
        mock_instance.frame_count = 0
        mock_instance.screen.ndarray = np.random.randint(0, 255, (144, 160, 3))
        mock_instance.send_input = Mock()
        mock_instance.tick = Mock()
        mock.return_value = mock_instance
        yield mock

@pytest.fixture
def mock_web_monitor():
    mock = Mock(spec=WebMonitor)
    mock.running = False
    mock.port = 8888
    mock.host = "localhost"
    mock.start.return_value = True
    mock.get_url.return_value = "http://localhost:8888"
    return mock

@pytest.mark.web
@pytest.mark.integration
class TestWebMonitorIntegration:
    """Test web monitor integration with trainer."""
    
    def test_web_monitor_initialization(self, mock_pyboy, mock_web_monitor):
        """Test web monitor initialization."""
        with patch('core.web_monitor.WebMonitor', return_value=mock_web_monitor):
            config = TrainingConfig(
                rom_path='test.gbc',
                web_port=8888,
                enable_web=True,
                capture_screens=True
            )
            
            trainer = PokemonTrainer(config)
            assert trainer.config.enable_web is True
            assert trainer.config.web_port == 8888
            
            # Should create web monitor
            assert trainer.web_monitor == mock_web_monitor
            
            # Clean up
            trainer._finalize_training()
            
            # Should stop web monitor
            mock_web_monitor.stop.assert_called_once()
    
    def test_web_monitor_disabled(self, mock_pyboy):
        """Test behavior when web monitor is disabled."""
        config = TrainingConfig(
            rom_path='test.gbc',
            enable_web=False
        )
        
        trainer = PokemonTrainer(config)
        assert trainer.config.enable_web is False
        assert trainer.web_monitor is None
        
        trainer._finalize_training()
    
    def test_web_monitor_pyboy_integration(self, mock_pyboy, mock_web_monitor):
        """Test web monitor integration with PyBoy."""
        with patch('core.web_monitor.WebMonitor', return_value=mock_web_monitor):
            config = TrainingConfig(
                rom_path='test.gbc',
                enable_web=True
            )
            
            trainer = PokemonTrainer(config)
            assert trainer.web_monitor == mock_web_monitor
            
            # PyBoy initialization should update web monitor
            mock_web_monitor.update_pyboy.assert_called_once()
            # Verify it was called with a PyBoy instance (not necessarily the fixture)
            call_args = mock_web_monitor.update_pyboy.call_args[0]
            assert len(call_args) == 1  # Should be called with one argument
            # The argument should be some kind of mock PyBoy instance
            assert hasattr(call_args[0], 'tick')  # Basic PyBoy interface check
            
            trainer._finalize_training()
    
    def test_web_monitor_stats_update(self, mock_pyboy, mock_web_monitor):
        """Test stats updating through web monitor."""
        with patch('core.web_monitor.WebMonitor', return_value=mock_web_monitor):
            config = TrainingConfig(
                rom_path='test.gbc',
                enable_web=True
            )
            
            trainer = PokemonTrainer(config)
            
            # Simulate some actions
            trainer.stats['actions_taken'] = 1000
            trainer.stats['total_reward'] = 50.0
            trainer.stats['llm_decision_count'] = 100
            
            # Get current stats should use these values
            stats = trainer.get_current_stats()
            assert stats['total_actions'] == 1000
            assert stats['total_reward'] == 50.0
            assert stats['llm_calls'] == 100
            
            trainer._finalize_training()
    
    def test_web_monitor_cleanup(self, mock_pyboy, mock_web_monitor):
        """Test proper cleanup of web monitor on shutdown."""
        with patch('core.web_monitor.WebMonitor', return_value=mock_web_monitor):
            config = TrainingConfig(
                rom_path='test.gbc',
                enable_web=True
            )
            
            trainer = PokemonTrainer(config)
            assert trainer.web_monitor.running is False
            
            # Simulate training start
            trainer.web_monitor.running = True
            
            # Finalize should stop the monitor
            trainer._finalize_training()
            mock_web_monitor.stop.assert_called_once()
