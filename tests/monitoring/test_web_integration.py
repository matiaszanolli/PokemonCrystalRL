"""Integration tests for web functionality of unified monitoring system."""

import pytest
import json
import asyncio
import websockets
import tempfile
import requests
import time
import numpy as np
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch

from monitoring import (
    UnifiedMonitor,
    MonitorConfig,
)
from monitoring.trainer_monitor_bridge import TrainingState

@pytest.fixture
def temp_dir():
    """Create temporary directory for test data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)

@pytest.fixture
def test_config(temp_dir):
    """Create test configuration."""
    return MonitorConfig(
        db_path=str(temp_dir / "test.db"),
        data_dir=str(temp_dir / "data"),
        static_dir=str(temp_dir / "static"),
        web_port=8099,  # Use non-standard port
        update_interval=0.1,
        snapshot_interval=0.5,
        max_events=1000,
        max_snapshots=10,
        debug=True
    )

@pytest.fixture
def monitor(test_config):
    """Create and start monitor instance."""
    monitor = UnifiedMonitor(config=test_config)
    monitor.start_training(config={"test": True})
    
    # Start the web server in a separate thread
    import threading
    import time
    
    def run_server():
        monitor.run(debug=False)
    
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    
    # Wait for server to start
    time.sleep(2)
    
    try:
        yield monitor
    finally:
        monitor.stop_training()

@pytest.mark.integration
@pytest.mark.web
class TestWebIntegration:
    """Test web interface integration."""
    
    def test_status_endpoint(self, monitor):
        """Test status API endpoint."""
        response = requests.get(
            f"http://localhost:{monitor.port}/api/status"
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
        response = requests.get(
            f"http://localhost:{monitor.port}/api/metrics"
        )
        assert response.status_code == 200
        
        data = response.json()
        assert "metrics" in data
        assert data["metrics"]["loss"] == 0.5
        assert data["metrics"]["accuracy"] == 0.8
        
        # Test historical metrics
        response = requests.get(
            f"http://localhost:{monitor.port}/api/metrics/history",
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
        response = requests.get(
            f"http://localhost:{monitor.port}/api/screenshot"
        )
        assert response.status_code == 200
        
        data = response.json()
        assert "image" in data
        assert data["image"].startswith("data:image/png;base64,")
        assert "timestamp" in data
        assert "dimensions" in data
    
    @pytest.mark.asyncio
    async def test_websocket_connection(self, monitor):
        """Test WebSocket connectivity."""
        uri = f"ws://localhost:{monitor.port}/ws"
        
        async with websockets.connect(uri) as websocket:
            # Send test message
            await websocket.send(json.dumps({
                "type": "test",
                "data": {"message": "test"}
            }))
            
            # Wait for response
            response = await websocket.recv()
            data = json.loads(response)
            assert data["type"] == "test"
            assert "timestamp" in data
    
    @pytest.mark.asyncio
    async def test_real_time_updates(self, monitor):
        """Test real-time updates via WebSocket."""
        uri = f"ws://localhost:{monitor.port}/ws"
        
        updates_received = []
        
        async with websockets.connect(uri) as websocket:
            # Update metrics
            monitor.update_metrics({
                "loss": 0.5,
                "accuracy": 0.8
            })
            
            # Should receive update
            response = await websocket.recv()
            data = json.loads(response)
            updates_received.append(data)
            
            assert data["type"] == "metrics_update"
            assert data["data"]["metrics"]["loss"] == 0.5
            
            # Update episode
            monitor.update_episode(
                episode=0,
                total_reward=10.0,
                steps=100,
                success=True
            )
            
            # Should receive update
            response = await websocket.recv()
            data = json.loads(response)
            updates_received.append(data)
            
            assert data["type"] == "episode_update"
            assert data["data"]["total_reward"] == 10.0
            
        assert len(updates_received) >= 2
    
    def test_error_reporting(self, monitor):
        """Test error reporting via web interface."""
        # Generate error
        error = Exception("Test error")
        monitor.error_handler.handle_error(
            error=error,
            message="Test error message",
            severity="error",
            category="test",
            component="monitor"
        )
        
        # Check error endpoint
        response = requests.get(
            f"http://localhost:{monitor.port}/api/errors"
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
        response = requests.get(
            f"http://localhost:{monitor.port}/api/system"
        )
        assert response.status_code == 200
        
        data = response.json()
        assert "cpu_percent" in data
        assert "memory_percent" in data
        assert "disk_usage" in data
        assert "timestamp" in data
    
    def test_training_control(self, monitor):
        """Test training control via web interface."""
        # Pause training
        response = requests.post(
            f"http://localhost:{monitor.port}/api/training/control",
            json={"action": "pause"}
        )
        assert response.status_code == 200
        assert monitor.training_state == TrainingState.PAUSED
        
        # Resume training
        response = requests.post(
            f"http://localhost:{monitor.port}/api/training/control",
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
        response = requests.get(
            f"http://localhost:{monitor.port}/static/test.txt"
        )
        assert response.status_code == 200
        assert response.text == "test content"
    
    def test_dashboard_html(self, monitor):
        """Test dashboard HTML endpoint."""
        response = requests.get(
            f"http://localhost:{monitor.port}/"
        )
        assert response.status_code == 200
        assert "text/html" in response.headers["Content-Type"]
        assert "Pokemon Crystal RL Monitor" in response.text
