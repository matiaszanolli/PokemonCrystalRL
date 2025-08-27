"""Test web server integration."""

import time
import queue
import threading
import pytest
import numpy as np
from unittest.mock import Mock, patch

from trainer.trainer import TrainingConfig, PokemonTrainer
from .mock_web_server import MockWebServer, ServerConfig

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
def mock_web_server():
    with patch('trainer.trainer.WebServer', MockWebServer):
        yield MockWebServer

@pytest.mark.web
@pytest.mark.integration
class TestWebServerIntegration:
    """Test web server initialization and connection."""
    
    def test_web_server_initialization(self, mock_pyboy):
        """Test web server starts up correctly."""
        config = TrainingConfig(
            rom_path='test.gbc',
            web_port=8888,  # Use non-standard port
            enable_web=True,
            capture_screens=True
        )
        
        trainer = PokemonTrainer(config)
        assert trainer.web_server is not None
        assert trainer.web_thread is not None
        assert trainer.web_server._running
        assert trainer.web_server.config.port == 8888
        
        # Clean up
        trainer._finalize_training()
        time.sleep(0.1)  # Let server shut down
    
    def test_web_server_disabled(self, mock_pyboy):
        """Test web server doesn't start when disabled."""
        config = TrainingConfig(
            rom_path='test.gbc',
            enable_web=False
        )
        
        trainer = PokemonTrainer(config)
        assert trainer.web_server is None
        assert trainer.web_thread is None
        
        trainer._finalize_training()
    
    def test_screenshot_memory_management(self, mock_pyboy):
        """Test screenshot memory handling."""
        config = TrainingConfig(
            rom_path='test.gbc',
            enable_web=True,
            capture_screens=True,
            capture_fps=30
        )
        
        trainer = PokemonTrainer(config)
        assert trainer.screen_queue is not None
        assert trainer.screen_queue.maxsize == 30
        
        # Fill queue
        screens = []
        for _ in range(35):  # Try to overflow
            screen = np.random.randint(0, 255, (144, 160, 3))
            trainer._capture_and_queue_screen()
            screens.append(screen)
            
        # Queue should maintain size limit
        assert trainer.screen_queue.qsize() <= 30
        assert not trainer.screen_queue.full()
        
        # Clean up
        trainer._finalize_training()
    
    def test_web_server_recovery(self, mock_pyboy):
        """Test web server handles failures gracefully."""
        config = TrainingConfig(
            rom_path='test.gbc',
            enable_web=True,
            web_port=8889
        )
        
        # Force server creation failure first
        with patch('monitoring.web_server.WebServer.start') as mock_start:
            mock_start.side_effect = Exception("Server start failed")
            trainer = PokemonTrainer(config)
            assert trainer.web_server is None
            assert trainer.web_thread is None
        
        # Now create working server
        trainer = PokemonTrainer(config)
        assert trainer.web_server is not None
        assert trainer.web_thread is not None
        assert trainer.web_server._running
        
        trainer._finalize_training()
    
    def test_web_server_port_retry(self, mock_pyboy):
        """Test web server retries with different ports."""
        config = TrainingConfig(
            rom_path='test.gbc',
            enable_web=True,
            web_port=8890
        )
        
        # Create first server
        trainer1 = PokemonTrainer(config)
        assert trainer1.web_server is not None
        assert trainer1.web_server.config.port == 8890
        
        # Try to create second server on same port
        trainer2 = PokemonTrainer(config)
        assert trainer2.web_server is not None
        assert trainer2.web_server.config.port > 8890
        
        # Clean up
        trainer1._finalize_training()
        trainer2._finalize_training()
        time.sleep(0.1)  # Let servers shut down
