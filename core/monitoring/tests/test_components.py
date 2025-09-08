"""
Tests for Monitoring Components

This module provides comprehensive tests for the core monitoring components:
- Screen capture
- Metrics collection
- Web server
"""

import time
import threading
import numpy as np
import pytest
from unittest.mock import Mock, MagicMock

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from core.monitoring.base import (
    MetricDefinition,
    ScreenCaptureConfig,
    ComponentError
)
from monitoring.components.capture import ScreenCapture
from monitoring.components.metrics import MetricsCollector
from monitoring.components.web import WebServer

# Test Utilities

class MockPyBoy:
    """Mock PyBoy implementation for testing."""
    def __init__(self):
        self.screen = Mock()
        self._frame_count = 0
        
    def get_frame(self):
        """Get a test frame."""
        self._frame_count += 1
        return np.zeros((144, 160, 4), dtype=np.uint8)

# Fixtures

@pytest.fixture
def mock_pyboy():
    """Create mock PyBoy instance."""
    pyboy = MockPyBoy()
    pyboy.screen.ndarray = pyboy.get_frame
    return pyboy

@pytest.fixture
def screen_capture():
    """Create screen capture component."""
    config = ScreenCaptureConfig(
        buffer_size=5,
        frame_rate=10.0,
        compression_quality=85
    )
    return ScreenCapture(config)

@pytest.fixture
def metrics_collector(tmp_path):
    """Create metrics collector component."""
    storage_path = tmp_path / "metrics"
    return MetricsCollector(
        retention_hours=24.0,
        storage_path=str(storage_path)
    )

@pytest.fixture
def web_server():
    """Create web server component."""
    return WebServer(host="localhost", port=0)  # Port 0 = random port

# Screen Capture Tests

def test_screen_capture_init():
    """Test screen capture initialization."""
    config = ScreenCaptureConfig(buffer_size=5)
    capture = ScreenCapture(config)
    assert capture.config.buffer_size == 5
    assert not capture._running
    assert len(capture._frame_buffer) == 0

def test_screen_capture_pyboy_required(screen_capture):
    """Test PyBoy instance is required."""
    with pytest.raises(ComponentError):
        screen_capture.start()

def test_screen_capture_start_stop(screen_capture, mock_pyboy):
    """Test screen capture start/stop."""
    screen_capture.set_pyboy(mock_pyboy)
    assert screen_capture.start()
    assert screen_capture._running
    assert screen_capture._capture_thread is not None
    
    assert screen_capture.stop()
    assert not screen_capture._running
    assert not screen_capture._capture_thread.is_alive()

def test_screen_capture_frame_formats(screen_capture, mock_pyboy):
    """Test different frame output formats."""
    screen_capture.set_pyboy(mock_pyboy)
    screen_capture.start()
    
    # Wait for first frame
    time.sleep(0.2)
    
    # Test formats
    frame = screen_capture.get_frame("numpy")
    assert isinstance(frame, np.ndarray)
    assert frame.shape == (144, 160, 4)
    
    frame = screen_capture.get_frame("base64_png")
    assert isinstance(frame, str)
    assert frame.startswith("iVBORw0KG")

# Metrics Tests

def test_metrics_init(metrics_collector):
    """Test metrics collector initialization."""
    assert metrics_collector.retention_hours == 24.0
    assert not metrics_collector._running
    assert len(metrics_collector.metrics) == 0

def test_metrics_register(metrics_collector):
    """Test metric registration."""
    definition = MetricDefinition(
        name="test_metric",
        type="counter",
        description="Test metric"
    )
    assert metrics_collector.register_metric(definition)
    assert "test_metric" in metrics_collector.definitions
    assert "test_metric" in metrics_collector.metrics

def test_metrics_record(metrics_collector):
    """Test recording metrics."""
    definition = MetricDefinition(
        name="test_metric",
        type="gauge",
        description="Test metric"
    )
    metrics_collector.register_metric(definition)
    
    assert metrics_collector.record_metric("test_metric", 42.0)
    assert metrics_collector.get_latest("test_metric") == 42.0
    
    stats = metrics_collector.get_statistics("test_metric")
    assert stats["count"] == 1
    assert stats["mean"] == 42.0

def test_metrics_persistence(metrics_collector):
    """Test metric persistence."""
    definition = MetricDefinition(
        name="test_metric",
        type="counter",
        description="Test metric"
    )
    metrics_collector.register_metric(definition)
    metrics_collector.record_metric("test_metric", 42.0)
    
    # Save and reload
    assert metrics_collector.save_metrics()
    
    new_collector = MetricsCollector(
        storage_path=metrics_collector.storage_path
    )
    assert new_collector.load_metrics()
    assert new_collector.get_latest("test_metric") == 42.0

# Web Server Tests

def test_web_server_init(web_server):
    """Test web server initialization."""
    assert not web_server._running
    assert web_server.ws_clients == 0

def test_web_server_start_stop(web_server):
    """Test web server start/stop."""
    assert web_server.start()
    assert web_server._running
    assert web_server._server_thread is not None
    
    assert web_server.stop()
    assert not web_server._running

def test_web_server_status(web_server):
    """Test web server status."""
    status = web_server.get_status()
    assert not status["running"]
    assert status["requests_handled"] == 0
    assert status["ws_messages_sent"] == 0

def test_web_server_updates(web_server):
    """Test publishing updates."""
    assert web_server.start()
    
    # Test publishing update
    assert web_server.publish_update("test_event", {"value": 42})
    assert web_server.ws_messages_sent == 1

# Integration Tests

def test_component_integration(screen_capture, metrics_collector, web_server, mock_pyboy):
    """Test components working together."""
    # Start components
    screen_capture.set_pyboy(mock_pyboy)
    assert screen_capture.start()
    assert metrics_collector.start()
    assert web_server.start()
    
    # Register metrics
    fps_metric = MetricDefinition(
        name="capture_fps",
        type="gauge",
        description="Screen capture FPS"
    )
    metrics_collector.register_metric(fps_metric)
    
    # Run for a bit
    time.sleep(0.5)
    
    # Get frame and publish metrics
    frame = screen_capture.get_frame("base64_png")
    assert frame is not None
    
    status = screen_capture.get_status()
    metrics_collector.record_metric("capture_fps", status["current_fps"])
    
    # Publish update
    web_server.publish_update("frame", {
        "image": frame,
        "fps": status["current_fps"]
    })
    
    # Verify
    assert metrics_collector.get_latest("capture_fps") is not None
    assert web_server.ws_messages_sent > 0
    
    # Stop components
    assert screen_capture.stop()
    assert metrics_collector.stop()
    assert web_server.stop()
