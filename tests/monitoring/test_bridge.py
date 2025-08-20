#!/usr/bin/env python3
"""
test_bridge.py - Unit tests for trainer-web bridge

Tests basic functionality and error handling of the bridge component.
"""

import pytest
import time
import numpy as np
import socket
from unittest.mock import Mock, patch
import threading
import tempfile
from pathlib import Path

def get_free_port():
    """Get a free port number."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port

from core.monitoring.bridge import TrainerWebBridge
from trainer import (
    PokemonTrainer,
    TrainingConfig,
    TrainingMode,
    LLMBackend
)


@pytest.fixture
def test_port():
    """Get random port for testing"""
    return get_free_port()

@pytest.fixture
def mock_config(test_port):
    """Base test configuration"""
    return TrainingConfig(
        rom_path="test.gbc",
        enable_web=True,
        web_port=test_port,  # Use random port for testing
        capture_screens=True,
        headless=True,
        debug_mode=True
    )


@pytest.fixture
def mock_trainer(mock_config):
    """Create mock trainer"""
    trainer = Mock()
    trainer.config = mock_config
    trainer.stats = {
        'total_actions': 0,
        'total_episodes': 0,
        'start_time': time.time(),
        'actions_per_second': 0.0,
        'llm_calls': 0,
        'llm_avg_time': 0.0
    }
    trainer.latest_screen = np.zeros((144, 160, 3), dtype=np.uint8)
    trainer._training_active = True
    
    return trainer


@pytest.mark.monitoring
class TestTrainerWebBridge:
    """Test bridge functionality"""

    @pytest.fixture(autouse=True)
    def setup(self, mock_trainer, test_port):
        """Setup test environment."""
        self.mock_trainer = mock_trainer
        self.test_port = test_port
        self.bridge = TrainerWebBridge(
            trainer=mock_trainer,
            port=test_port,
            host='127.0.0.1',
            debug=False
        )
        yield
        if self.bridge.active:
            self.bridge.stop()

    def test_bridge_initialization(self):
        """Test bridge initialization"""
        assert self.bridge.port == self.test_port
        assert not self.bridge.active
        assert self.bridge.error_count == 0
        assert self.bridge.transfer_stats['screenshots_sent'] == 0

    def test_screenshot_validation(self):
        """Test screenshot validation"""
        # Valid screen
        valid_screen = np.random.randint(0, 255, (144, 160, 3), dtype=np.uint8)
        assert self.bridge._validate_screenshot(valid_screen)
        
        # Invalid screens
        assert not self.bridge._validate_screenshot(None)
        assert not self.bridge._validate_screenshot(np.zeros((144, 160), dtype=np.uint8))  # Wrong shape
        assert not self.bridge._validate_screenshot(np.zeros((144, 160, 3), dtype=np.uint8))  # All black

    def test_stats_collection(self):
        """Test stats collection"""
        # Update mock trainer stats
        self.mock_trainer.stats.update({
            'total_actions': 100,
            'total_episodes': 2,
            'actions_per_second': 25.0
        })
        
        stats = self.bridge._get_session_stats()
        
        assert stats['basic']['total_steps'] == 100
        assert stats['basic']['total_episodes'] == 2
        assert stats['performance']['avg_actions_per_second'] == 25.0
        assert 'system' in stats
        assert 'timestamp' in stats

    def test_error_handling(self):
        """Test error handling and rate limiting"""
        # Generate multiple errors
        for i in range(10):
            self.bridge._handle_error(f"Test error {i}")
        
        assert self.bridge.error_count == 10
        assert self.bridge.transfer_stats['errors'] == 10
        assert 'Test error' in self.bridge.transfer_stats['last_error']['message']

    def test_server_lifecycle(self):
        """Test server start/stop"""
        # Start server
        self.bridge.start()
        assert self.bridge.active
        assert self.bridge.bridge_thread is not None
        assert self.bridge.cleanup_thread is not None
        
        # Let it run briefly
        time.sleep(0.1)
        
        # Stop server
        self.bridge.stop()
        assert not self.bridge.active
        
        # Check final stats
        assert self.bridge.transfer_stats['start_time'] is not None
        assert 'screenshots_sent' in self.bridge.transfer_stats
        assert 'errors' in self.bridge.transfer_stats

    def test_screenshot_transfer(self):
        """Test screenshot transfer"""
        # Create test screen
        test_screen = np.random.randint(0, 255, (144, 160, 3), dtype=np.uint8)
        self.mock_trainer.latest_screen = test_screen
        
        screenshot = self.bridge._get_current_screenshot()
        assert screenshot is not None
        assert len(screenshot) > 0  # Should have base64 data
        
        # Check stats
        assert self.bridge.transfer_stats['screenshots_sent'] == 1
        assert self.bridge.last_screen_update > 0

    def test_cleanup_worker(self):
        """Test cleanup worker"""
        # Start bridge
        self.bridge.start()
        
        # Add some errors
        for _ in range(5):
            self.bridge._handle_error("Test error")
        
        original_count = self.bridge.error_count
        
        # Let cleanup run
        time.sleep(0.1)
        
        # Verify cleanup
        self.bridge.stop()
        assert self.bridge.error_count <= original_count

    def test_bridge_status(self):
        """Test bridge status information"""
        self.bridge.start()
        
        status = self.bridge._get_bridge_status()
        
        assert status['active']
        assert 'uptime' in status
        assert 'screenshots_sent' in status
        assert 'error_count' in status
        assert 'screen_fps' in status
        assert 'last_error' in status
        
        self.bridge.stop()

    def test_error_recovery(self):
        """Test error recovery"""
        # Simulate screenshot error
        self.mock_trainer.latest_screen = None
        screenshot = self.bridge._get_current_screenshot()
        
        assert screenshot is None
        assert self.bridge.error_count == 0  # Should not count as error
        
        # Simulate processing error
        self.mock_trainer.latest_screen = "invalid"
        screenshot = self.bridge._get_current_screenshot()
        
        assert screenshot is None
        assert self.bridge.error_count > 0  # Should count as error

    def test_flask_routes(self):
        """Test Flask routes"""
        with self.bridge.app.test_client() as client:
            # Test dashboard
            response = client.get('/')
            assert response.status_code == 200
            
            # Test API routes
            response = client.get('/api/screenshot/current')
            assert response.status_code == 200
            assert 'screenshot' in response.json
            
            response = client.get('/api/session/stats')
            assert response.status_code == 200
            assert 'basic' in response.json
            assert 'performance' in response.json
            
            response = client.get('/api/bridge/status')
            assert response.status_code == 200
            assert 'active' in response.json
            assert 'uptime' in response.json
