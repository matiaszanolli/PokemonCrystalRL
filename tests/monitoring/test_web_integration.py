"""Integration tests for web functionality of unified monitoring system."""

import pytest
import json
import asyncio
import tempfile
import requests
import socketio
import time
import numpy as np
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch

from monitoring import (
    UnifiedMonitor,
    MonitorConfig,
)
from monitoring.web_monitor import TrainingState

@pytest.fixture
def temp_dir():
    """Create temporary directory for test data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)

@pytest.fixture
def test_config(temp_dir):
    """Create test configuration."""
    # Find an available port
    import socket
    sock = socket.socket()
    sock.bind(('', 0))
    available_port = sock.getsockname()[1]
    sock.close()
    
    return MonitorConfig(
        db_path=str(temp_dir / "test.db"),
        data_dir=str(temp_dir / "data"),
        static_dir=str(temp_dir / "static"),
        web_port=available_port,  # Use dynamic port
        update_interval=0.1,
        snapshot_interval=0.5,
        max_events=1000,
        max_snapshots=10,
        debug=True
    )

@pytest.fixture
def error_handler():
    """Create error handler for testing."""
    from monitoring.error_handler import ErrorHandler, ErrorSeverity, ErrorCategory
    return ErrorHandler()

@pytest.fixture
def monitor(test_config, error_handler):
    """Create and start monitor instance."""
    monitor = UnifiedMonitor(config=test_config)
    monitor.error_handler = error_handler
    
    # Start training - this will call _ensure_server_started() automatically
    # which starts the server in a background thread
    monitor.start_training(config={"test": True})
    
    import time
    import requests
    
    # Wait for server to become responsive
    start_time = time.time()
    max_wait = 10  # Maximum wait time in seconds
    while time.time() - start_time < max_wait:
        try:
            response = requests.get(f"http://localhost:{monitor.config.web_port}/api/status", timeout=1)
            if response.status_code == 200:
                break
        except (requests.ConnectionError, requests.Timeout):
            time.sleep(0.1)
    else:
        raise TimeoutError(f"Server on port {monitor.config.web_port} did not become responsive within timeout period")
    
    try:
        yield monitor
    finally:
        # Comprehensive cleanup
        try:
            # Stop training and monitoring (this will also stop the server thread)
            monitor.stop_training()
            monitor.stop_monitoring()
            
            # Give the system time to release the port
            time.sleep(1.0)
            
        except Exception as e:
            print(f"Warning: Error during monitor cleanup: {e}")

@pytest.mark.integration
@pytest.mark.web
class TestWebIntegration:
    """Test web interface integration."""
    
    def test_status_endpoint(self, monitor):
        """Test status API endpoint."""
        port = monitor.config.web_port if monitor.config else monitor.port
        response = requests.get(
            f"http://localhost:{port}/api/status"
        )
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "running"
        assert data["monitoring"] is True
        assert "current_run_id" in data
        assert data["current_run_id"] == monitor.current_run_id
    
    def test_metrics_endpoint(self, monitor):
        """Test metrics API endpoint."""
        # Add some metrics
        metrics = {
            "loss": 0.5,
            "accuracy": 0.8,
            "reward": 1.0
        }
        monitor.update_metrics(metrics)
        
        # Get metrics
        port = monitor.config.web_port if monitor.config else monitor.port
        response = requests.get(
            f"http://localhost:{port}/api/metrics"
        )
        assert response.status_code == 200
        
        data = response.json()
        assert "metrics" in data
        assert data["metrics"]["loss"] == 0.5
        assert data["metrics"]["accuracy"] == 0.8
        
        # Test historical metrics
        response = requests.get(
            f"http://localhost:{port}/api/metrics/history",
            params={"metric": "loss", "minutes": 5}
        )
        assert response.status_code == 200
        
        data = response.json()
        assert "data" in data
        assert len(data["data"]) > 0
        assert "statistics" in data
    
    def test_screenshot_endpoint(self, monitor):
        """Test screenshot API endpoint."""
        # Add a screenshot
        screenshot = np.random.randint(0, 255, (144, 160, 3), dtype=np.uint8)
        monitor.update_screenshot(screenshot)
        
        # Get screenshot
        port = monitor.config.web_port if monitor.config else monitor.port
        response = requests.get(
            f"http://localhost:{port}/api/screenshot"
        )
        assert response.status_code == 200
        
        data = response.json()
        assert "image" in data
        assert data["image"].startswith("data:image/png;base64,")
        assert "timestamp" in data
        assert "dimensions" in data
    
    def test_websocket_connection(self, monitor):
        """Test Socket.IO connectivity."""
        # Create Socket.IO client
        socket = socketio.Client()
        port = monitor.config.web_port if monitor.config else monitor.port
        socket.connect(f"http://localhost:{port}")

        # Initialize response holder
        response_data = None
        done = False
        def handle_response(data):
            nonlocal response_data, done
            response_data = data
            done = True

        # Register handler and emit test message
        socket.on('test', handle_response)
        socket.emit('test', {'message': 'test'})

        # Wait for response with timeout
        start = time.time()
        while not done and time.time() - start < 5:
            time.sleep(0.1)
        
        socket.disconnect()
        
        assert response_data is not None
        assert 'type' in response_data
        assert response_data['type'] == 'test'
        assert 'timestamp' in response_data
    
    def test_real_time_updates(self, monitor):
        """Test real-time updates via Socket.IO."""
        socket = socketio.Client()
        port = monitor.config.web_port if monitor.config else monitor.port
        socket.connect(f"http://localhost:{port}")

        updates_received = []
        done = False

        def handle_metrics(data):
            updates_received.append(('metrics', data))
            check_done()

        def handle_episode(data):
            updates_received.append(('episode', data))
            check_done()

        def check_done():
            nonlocal done
            done = len(updates_received) >= 2

        socket.on('metrics_update', handle_metrics)
        socket.on('episode_update', handle_episode)

        # Update metrics
        monitor.update_metrics({
            "loss": 0.5,
            "accuracy": 0.8
        })

        # Update episode
        monitor.update_episode(
            episode=0,
            total_reward=10.0,
            steps=100,
            success=True
        )

        # Wait for updates with timeout
        start = time.time()
        while not done and time.time() - start < 5:
            time.sleep(0.1)

        socket.disconnect()

        assert len(updates_received) >= 2

        # Get metrics and episode updates
        metrics_payload = next(data for type, data in updates_received if type == 'metrics')
        episode_payload = next(data for type, data in updates_received if type == 'episode')

        # metrics_payload is the full event data; access nested structure if present
        metrics = metrics_payload.get('data', {}).get('metrics') or metrics_payload
        assert metrics['loss'] == 0.5
        assert metrics['accuracy'] == 0.8

        # Episode payload may be nested as well
        episode = episode_payload.get('data') or episode_payload
        assert episode['episode'] == 0
        assert episode['total_reward'] == 10.0
    
    def test_error_reporting(self, monitor):
        """Test error reporting via web interface."""
        # Generate error
        error = Exception("Test error")
        from monitoring.error_handler import ErrorSeverity, ErrorCategory
        monitor.error_handler.handle_error(
            error=error,
            message="Test error message",
            severity=ErrorSeverity.ERROR,
            category=ErrorCategory.SYSTEM,
            component="monitor"
        )
        
        # Check error endpoint
        port = monitor.config.web_port if monitor.config else monitor.port
        response = requests.get(
            f"http://localhost:{port}/api/errors"
        )
        assert response.status_code == 200
        
        data = response.json()
        assert "errors" in data
        assert len(data["errors"]) == 1
        error = data["errors"][0]
        assert error["message"] == "Test error message"
        assert error["component"] == "monitor"
    
    def test_system_metrics(self, monitor):
        """Test system metrics reporting."""
        port = monitor.config.web_port if monitor.config else monitor.port
        response = requests.get(
            f"http://localhost:{port}/api/system"
        )
        assert response.status_code == 200
        
        data = response.json()
        assert "cpu_percent" in data
        assert "memory_percent" in data
        assert "disk_usage" in data
        assert "timestamp" in data
    
    def test_training_control(self, monitor):
        """Test training control via web interface."""
        port = monitor.config.web_port if monitor.config else monitor.port
        # Pause training
        response = requests.post(
            f"http://localhost:{port}/api/training/control",
            json={"action": "pause"}
        )
        assert response.status_code == 200
        assert monitor.training_state == TrainingState.PAUSED
        
        # Resume training
        response = requests.post(
            f"http://localhost:{port}/api/training/control",
            json={"action": "resume"}
        )
        assert response.status_code == 200
        assert monitor.training_state == TrainingState.RUNNING
    
    def test_static_files(self, monitor, test_config):
        """Test static file serving."""
        # Create test static file
        static_dir = Path(test_config.static_dir)
        static_dir.mkdir(parents=True, exist_ok=True)
        test_file = static_dir / "test.txt"
        test_file.write_text("test content")
        
        # Get file
        port = monitor.config.web_port if monitor.config else monitor.port
        response = requests.get(
            f"http://localhost:{port}/static/test.txt"
        )
        assert response.status_code == 200
        assert response.text == "test content"
    
    def test_dashboard_html(self, monitor):
        """Test dashboard HTML endpoint."""
        port = monitor.config.web_port if monitor.config else monitor.port
        response = requests.get(
            f"http://localhost:{port}/"
        )
        assert response.status_code == 200
        assert "text/html" in response.headers["Content-Type"]
        assert "Pokemon Crystal RL Training Monitor" in response.text
