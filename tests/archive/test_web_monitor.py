#!/usr/bin/env python3
"""
test_web_monitor.py - Unit tests for the WebMonitor class

Tests the core functionality of the WebMonitor component:
- Screen capture and streaming
- Stats tracking and API endpoints
- Server initialization and cleanup
- Error handling
"""

import pytest
import json
import time
from datetime import datetime
import numpy as np
import base64
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile

pytestmark = pytest.mark.skip("WebMonitor archive tests disabled - testing archived functionality")
import threading
import queue
import requests
from PIL import Image
import io

from core.web_monitor import WebMonitor, ScreenCapture, WebMonitorHandler


class MockPyBoy:
    """Mock PyBoy for testing screen capture."""
    
    def __init__(self):
        self.screen = Mock()
        self.screen.ndarray = np.random.randint(0, 255, (144, 160, 3), dtype=np.uint8)


class MockTrainer:
    """Mock trainer for testing WebMonitor."""
    
    def __init__(self):
        self._start_time = time.time()
        self.stats = {
            'actions_taken': 1000,
            'training_time': 3600,
            'actions_per_second': 2.5,
            'llm_decision_count': 100,
            'total_reward': 50.0,
            'player_level': 5,
            'badges_total': 1,
            'start_time': datetime.now().isoformat()
        }
        
        self.llm_decisions = [
            {'action': 'up', 'reasoning': 'Moving to exit', 'timestamp': time.time()},
            {'action': 'a', 'reasoning': 'Interact with NPC', 'timestamp': time.time()}
        ]
        
    @property
    def start_time(self):
        return self._start_time


@pytest.fixture
def mock_pyboy():
    """Create mock PyBoy instance."""
    return MockPyBoy()


@pytest.fixture
def mock_trainer():
    """Create mock trainer instance."""
    return MockTrainer()


@pytest.fixture
def web_monitor(mock_trainer):
    """Create WebMonitor instance for testing."""
    # Start with a high port to avoid conflicts with common services
    start_port = 9000
    monitor = None
    for port in range(start_port, start_port + 100):  # Try 100 ports
        try:
            monitor = WebMonitor(mock_trainer, port, "localhost")
            break
        except:
            continue
    
    if monitor is None:
        raise RuntimeError("Could not find available port for web monitor")
        
    yield monitor
    try:
        monitor.stop()
    except:
        pass


class TestScreenCapture:
    """Test screen capture functionality."""
    
    def test_screen_capture_initialization(self, mock_pyboy):
        """Test screen capture initialization."""
        capture = ScreenCapture(mock_pyboy)
        assert capture.pyboy == mock_pyboy
        assert capture.latest_screen is None
        assert not capture.capture_active
        assert isinstance(capture.stats, dict)
    
    def test_screen_capture_start_stop(self, mock_pyboy):
        """Test starting and stopping screen capture."""
        # Configure mock_pyboy.screen.ndarray with proper shape attribute
        screen_array = np.random.randint(0, 255, (144, 160, 3), dtype=np.uint8)
        mock_pyboy.screen.ndarray = screen_array
        
        capture = ScreenCapture(mock_pyboy)
        
        # Start capture
        capture.start_capture()
        assert capture.capture_active
        assert capture.capture_thread is not None
        assert capture.capture_thread.is_alive()
        
        # Let it capture a few frames
        time.sleep(0.5)
        assert capture.stats['frames_captured'] > 0
        assert capture.latest_screen is not None
        
        # Stop capture
        capture.stop_capture()
        assert not capture.capture_active
        time.sleep(0.2)
        assert not capture.capture_thread.is_alive()
    
    def test_get_latest_screen(self, mock_pyboy):
        """Test getting latest screen data."""
        capture = ScreenCapture(mock_pyboy)
        capture.start_capture()
        
        # Wait for first capture with increasing delay if needed
        for _ in range(5):  # Try up to 5 times
            time.sleep(0.3)
            if capture.latest_screen is not None:
                break
        
        # Get screen bytes
        screen_bytes = capture.get_latest_screen_bytes()
        assert screen_bytes is not None
        assert len(screen_bytes) > 0
        
        # Verify it's a valid PNG
        image = Image.open(io.BytesIO(screen_bytes))
        assert image.format == "PNG"
        assert image.size == (320, 288)  # Expected resize dimensions
        
        # Get screen data
        screen_data = capture.get_latest_screen_data()
        assert screen_data is not None
        assert 'image_b64' in screen_data
        assert 'timestamp' in screen_data
        assert 'size' in screen_data
        assert 'frame_id' in screen_data
        
        capture.stop_capture()


class TestWebMonitor:
    """Test WebMonitor functionality."""
    
    def test_monitor_initialization(self, mock_trainer):
        """Test WebMonitor initialization."""
        monitor = WebMonitor(mock_trainer, 8080, "localhost")
        assert monitor.trainer == mock_trainer
        assert monitor.port == 8080
        assert monitor.host == "localhost"
        assert monitor.screen_capture is not None
        assert not monitor.running
    
    def test_server_start_stop(self, web_monitor):
        """Test starting and stopping the web server."""
        assert not web_monitor.running
        
        # Start server
        success = web_monitor.start()
        assert success
        assert web_monitor.running
        assert web_monitor.server is not None
        assert web_monitor.server_thread is not None
        assert web_monitor.server_thread.is_alive()
        
        # Verify server is responsive
        response = requests.get(f"http://localhost:{web_monitor.port}/api/status")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "running"
        
        # Stop server
        web_monitor.stop()
        assert not web_monitor.running
        time.sleep(0.2)
        assert not web_monitor.server_thread.is_alive()
        
        # Verify server is stopped
        with pytest.raises(requests.ConnectionError):
            requests.get(f"http://localhost:{web_monitor.port}/api/status")
    
    def test_api_endpoints(self, web_monitor, mock_pyboy):
        """Test API endpoints."""
        web_monitor.update_pyboy(mock_pyboy)
        web_monitor.start()
        base_url = f"http://localhost:{web_monitor.port}"
        
        try:
            # Test /api/stats
            response = requests.get(f"{base_url}/api/stats")
            assert response.status_code == 200
            stats = response.json()
            assert stats["total_actions"] == 1000
            assert stats["llm_calls"] == 100
            assert stats["total_reward"] == 50.0
            
            # Test /api/screenshot
            response = requests.get(f"{base_url}/api/screenshot")
            assert response.status_code == 200
            assert response.headers["Content-Type"] == "image/png"
            
            # Test /api/llm_decisions
            response = requests.get(f"{base_url}/api/llm_decisions")
            assert response.status_code == 200
            decisions = response.json()
            assert "recent_decisions" in decisions
            assert len(decisions["recent_decisions"]) == 2
            
            # Test /api/status
            response = requests.get(f"{base_url}/api/status")
            assert response.status_code == 200
            status = response.json()
            assert status["status"] == "running"
            assert status["screen_capture_active"] is True
            
        finally:
            web_monitor.stop()
    
    def test_dashboard_html(self, web_monitor):
        """Test dashboard HTML endpoint."""
        web_monitor.start()
        
        try:
            response = requests.get(f"http://localhost:{web_monitor.port}/")
            assert response.status_code == 200
            assert "text/html" in response.headers["Content-Type"]
            
            # Check for key dashboard elements
            html = response.text
            assert "Pokemon Crystal LLM RL Training Dashboard" in html
            assert "Training Statistics" in html
            assert "Live Game Screen" in html
            assert "Game State" in html
            assert "Recent LLM Decisions" in html
            
        finally:
            web_monitor.stop()
    
    def test_error_handling(self, web_monitor):
        """Test error handling in requests."""
        web_monitor.start()
        
        try:
            # Test non-existent endpoint
            response = requests.get(f"http://localhost:{web_monitor.port}/api/nonexistent")
            assert response.status_code == 404
            
            # Test invalid screenshot request (before capture starts)
            web_monitor.screen_capture.stop_capture()
            response = requests.get(f"http://localhost:{web_monitor.port}/api/screenshot")
            assert response.status_code == 200  # Should return blank image
            image = Image.open(io.BytesIO(response.content))
            assert image.size == (320, 288)
            
        finally:
            web_monitor.stop()
    
    def test_port_management(self, mock_trainer):
        """Test port selection and conflict handling."""
        # Start first monitor
        monitor1 = WebMonitor(mock_trainer, 8080, "localhost")
        assert monitor1.start()
        
        # Try to start second monitor on same port
        monitor2 = WebMonitor(mock_trainer, 8080, "localhost")
        # Should pick a different port
        assert monitor2.port != 8080
        assert monitor2.start()
        
        # Clean up
        monitor1.stop()
        monitor2.stop()


class TestWebMonitorIntegration:
    """Test WebMonitor integration with trainer."""
    
    def test_pyboy_update(self, web_monitor, mock_pyboy):
        """Test updating PyBoy instance."""
        web_monitor.start()
        assert not web_monitor.screen_capture.capture_active
        
        # Update with PyBoy
        web_monitor.update_pyboy(mock_pyboy)
        assert web_monitor.screen_capture.pyboy == mock_pyboy
        assert web_monitor.screen_capture.capture_active
        
        # Verify screenshot capture works
        time.sleep(0.3)
        response = requests.get(f"http://localhost:{web_monitor.port}/api/screenshot")
        assert response.status_code == 200
        image = Image.open(io.BytesIO(response.content))
        assert image.size == (320, 288)
        
        web_monitor.stop()
    
    def test_trainer_stats_update(self, web_monitor, mock_trainer):
        """Test trainer stats updates."""
        web_monitor.start()
        
        # Initial stats
        response = requests.get(f"http://localhost:{web_monitor.port}/api/stats")
        initial_stats = response.json()
        assert initial_stats["total_actions"] == 1000
        
        # Update trainer stats
        mock_trainer.stats["actions_taken"] = 2000
        mock_trainer.stats["total_reward"] = 100.0
        
        # Get updated stats
        response = requests.get(f"http://localhost:{web_monitor.port}/api/stats")
        updated_stats = response.json()
        assert updated_stats["total_actions"] == 2000
        assert updated_stats["total_reward"] == 100.0
        
        web_monitor.stop()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
