"""Integration tests for web functionality of unified monitoring system.

⚠️  DEPRECATED: These tests use the legacy monitoring system.
Use tests/web_dashboard/ for unified dashboard tests.
"""

import pytest
import json
import asyncio
import tempfile
import requests
import socketio
import time
import threading
import io
import numpy as np
from PIL import Image
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch

# Skip all tests in this module since it's for legacy system
pytestmark = pytest.mark.skip(reason="Legacy monitoring system removed - use web_dashboard tests instead")

@pytest.fixture
def temp_dir():
    """Create temporary directory for test data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)

@pytest.fixture
def mock_trainer():
    """Create mock trainer instance."""
    trainer = Mock()
    trainer._start_time = time.time()
    trainer.start_time = property(lambda self: self._start_time)
    trainer.stats = {
        'total_actions': 1000,
        'training_time': 3600,
        'actions_per_second': 2.5,
        'llm_calls': 100,
        'total_reward': 50.0,
        'player_level': 5,
        'badges_total': 1
    }
    trainer.llm_decisions = [
        {'action': 'up', 'reasoning': 'Moving to exit', 'timestamp': time.time()},
        {'action': 'a', 'reasoning': 'Interact with NPC', 'timestamp': time.time()}
    ]
    return trainer

@pytest.fixture
def available_port():
    """Get an available port for testing."""
    import socket
    sock = socket.socket()
    sock.bind(('', 0))
    port = sock.getsockname()[1]
    sock.close()
    return port

@pytest.fixture
def error_handler():
    """Create error handler for testing."""
    from monitoring.error_handler import ErrorHandler, ErrorSeverity, ErrorCategory
    return ErrorHandler()

@pytest.fixture
def web_monitor(mock_trainer, available_port):
    """Create and start web monitor instance."""
    monitor = WebMonitor(mock_trainer, available_port, "localhost")
    assert monitor.start()  # Start the monitor
    
    # Wait for server to become responsive
    start_time = time.time()
    max_wait = 5  # Maximum wait time in seconds
    while time.time() - start_time < max_wait:
        try:
            response = requests.get(f"http://localhost:{monitor.port}/api/status", timeout=1)
            if response.status_code == 200:
                break
        except (requests.ConnectionError, requests.Timeout):
            time.sleep(0.1)
    else:
        raise TimeoutError(f"Server did not become responsive within timeout period")
    
    try:
        yield monitor
    finally:
        try:
            monitor.stop()
            time.sleep(0.5)  # Allow time for cleanup
        except Exception as e:
            print(f"Warning: Error during monitor cleanup: {e}")

from ..fixtures.socket_helpers import temp_server, SocketTestManager

@pytest.mark.integration
@pytest.mark.web
class TestWebIntegration:
    @pytest.fixture(scope='class')
    def socket_manager(self):
        manager = SocketTestManager()
        yield manager
        manager.cleanup()
        
    @pytest.fixture
    def test_port(self, socket_manager):
        """Get test port for each test"""
        with temp_server() as port:
            yield port
    """Test web interface integration with LLMTrainer."""
    
    def test_status_endpoint(self, web_monitor):
        """Test status API endpoint."""
        response = requests.get(f"http://localhost:{web_monitor.port}/api/status")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "running"
        assert "screen_capture_active" in data
        assert data["screen_capture_active"] == False  # No PyBoy instance yet
        assert "uptime" in data
    
    def test_stats_endpoint(self, web_monitor, mock_trainer):
        """Test stats API endpoint."""
        response = requests.get(f"http://localhost:{web_monitor.port}/api/stats")
        assert response.status_code == 200
        
        data = response.json()
        assert data["total_actions"] == mock_trainer.stats["total_actions"]
        assert data["llm_calls"] == mock_trainer.stats["llm_calls"]
        assert data["total_reward"] == mock_trainer.stats["total_reward"]

        # Update stats
        mock_trainer.stats["total_actions"] = 2000
        mock_trainer.stats["total_reward"] = 100.0
        
        response = requests.get(f"http://localhost:{web_monitor.port}/api/stats")
        data = response.json()
        assert data["total_actions"] == 2000
        assert data["total_reward"] == 100.0
    
    def test_screenshot_endpoint(self, web_monitor):
        """Test screenshot API endpoint."""
        response = requests.get(f"http://localhost:{web_monitor.port}/api/screenshot")
        assert response.status_code == 200
        
        # Should return blank image since no PyBoy instance
        image = Image.open(io.BytesIO(response.content))
        assert image.size == (320, 288)  # Expected dimensions after resize
        
        # Add mock PyBoy with screen
        mock_pyboy = Mock()
        mock_pyboy.screen = Mock()
        # Create actual numpy array to avoid shape attribute issues
        screen_array = np.random.randint(0, 255, (144, 160, 3), dtype=np.uint8)
        mock_pyboy.screen.ndarray = screen_array
        
        web_monitor.update_pyboy(mock_pyboy)
        time.sleep(0.3)  # Wait for first capture
        
        response = requests.get(f"http://localhost:{web_monitor.port}/api/screenshot")
        assert response.status_code == 200
        assert response.headers["Content-Type"] == "image/png"
        
        image = Image.open(io.BytesIO(response.content))
        assert image.size == (320, 288)
    
    def test_llm_decisions_endpoint(self, web_monitor, mock_trainer):
        """Test LLM decisions API endpoint."""
        response = requests.get(f"http://localhost:{web_monitor.port}/api/llm_decisions")
        assert response.status_code == 200
        
        data = response.json()
        assert "recent_decisions" in data
        assert len(data["recent_decisions"]) == 2  # From mock trainer
        
        # Verify decision content
        decisions = data["recent_decisions"]
        assert decisions[0]["action"] == "a"
        assert decisions[0]["reasoning"] == "Interact with NPC"
        assert decisions[1]["action"] == "up"
        assert decisions[1]["reasoning"] == "Moving to exit"
    
    def test_stats_update_interval(self, web_monitor, mock_trainer):
        """Test stats update interval works properly."""
        # Get initial stats
        response = requests.get(f"http://localhost:{web_monitor.port}/api/stats")
        initial_stats = response.json()
        
        # Update trainer stats
        mock_trainer.stats["total_actions"] = 3000
        mock_trainer.stats["llm_calls"] = 200
        mock_trainer.stats["total_reward"] = 150.0
        
        # Stats should update within 1 second
        time.sleep(1.0)
        
        response = requests.get(f"http://localhost:{web_monitor.port}/api/stats")
        updated_stats = response.json()
        
        assert updated_stats["total_actions"] == 3000
        assert updated_stats["llm_calls"] == 200
        assert updated_stats["total_reward"] == 150.0
    
    def test_dashboard_loads(self, web_monitor):
        """Test that the dashboard UI loads properly."""
        response = requests.get(f"http://localhost:{web_monitor.port}/")
        assert response.status_code == 200
        assert "text/html" in response.headers["Content-Type"]
        
        html = response.text
        # Check for key dashboard elements
        required_elements = [
            "Pokemon Crystal LLM RL Training Dashboard",
            "Training Statistics",
            "Live Game Screen",
            "Game State",
            "Recent LLM Decisions",
            "Live Memory Debug"
        ]
        
        for element in required_elements:
            assert element in html, f"Missing dashboard element: {element}"
    
    def test_screen_update_interval(self, web_monitor):
        """Test screen update interval."""
        # Add mock PyBoy with screen
        mock_pyboy = Mock()
        mock_pyboy.screen = Mock()
        # Create actual numpy array to avoid shape attribute issues
        screen_array = np.random.randint(0, 255, (144, 160, 3), dtype=np.uint8)
        mock_pyboy.screen.ndarray = screen_array
        
        web_monitor.update_pyboy(mock_pyboy)
        
        # Should capture at ~5 FPS
        time.sleep(0.5)
        capture_stats = web_monitor.screen_capture.stats
        
        # Should have captured 2-3 frames in 0.5 seconds at 5 FPS
        assert 1 <= capture_stats['frames_captured'] <= 3
    
    def test_concurrent_requests(self, web_monitor):
        """Test handling concurrent API requests."""
        request_results = []
        request_errors = []
        
        def make_requests(endpoint):
            try:
                for _ in range(10):
                    response = requests.get(f"http://localhost:{web_monitor.port}{endpoint}")
                    request_results.append((endpoint, response.status_code))
                    time.sleep(0.01)  # Small delay
            except Exception as e:
                request_errors.append((endpoint, str(e)))
        
        # Create threads for different endpoints
        threads = [
            threading.Thread(target=make_requests, args=("/api/stats",)),
            threading.Thread(target=make_requests, args=("/api/screenshot",)),
            threading.Thread(target=make_requests, args=("/api/llm_decisions",))
        ]
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        assert not request_errors, f"Errors during concurrent requests: {request_errors}"
        assert len(request_results) == 30  # 10 requests * 3 endpoints
        
        # All requests should have succeeded
        for _, status_code in request_results:
            assert status_code == 200
    
    def test_port_conflict_resolution(self):
        """Test handling of port conflicts."""
        mock_trainer = Mock()
        
        # Start first monitor
        port = 8888
        monitor1 = WebMonitor(mock_trainer, port, "localhost")
        assert monitor1.start()
        assert monitor1.port == port
        
        # Try to start second monitor on same port
        monitor2 = WebMonitor(mock_trainer, port, "localhost")
        assert monitor2.port != port  # Should have picked different port
        assert monitor2.start()
        
        # Verify both are running
        assert requests.get(f"http://localhost:{monitor1.port}/api/status").status_code == 200
        assert requests.get(f"http://localhost:{monitor2.port}/api/status").status_code == 200
        
        # Clean up
        monitor1.stop()
        monitor2.stop()
    
    def test_error_handling(self, web_monitor):
        """Test error handling in web monitor."""
        # Test non-existent endpoint
        response = requests.get(f"http://localhost:{web_monitor.port}/api/nonexistent")
        assert response.status_code == 404
        
        # Test missing PyBoy
        response = requests.get(f"http://localhost:{web_monitor.port}/api/screenshot")
        assert response.status_code == 200  # Should return blank image
        
        # Test malformed requests
        response = requests.post(
            f"http://localhost:{web_monitor.port}/api/status",
            data="invalid json"
        )
        assert response.status_code in [400, 404, 405]  # Depending on implementation
