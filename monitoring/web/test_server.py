"""
Tests for the monitoring web server.

These tests cover:
1. Basic server initialization and configuration
2. Route handling (/, /status, API endpoints)
3. WebSocket functionality
4. Component integration (ScreenCapture, MetricsCollector)
5. Status management and updates
"""

import pytest
import json
import time
import numpy as np
from unittest.mock import Mock, patch
from flask import url_for
from flask.testing import FlaskClient
from flask_socketio import SocketIOTestClient

from .server import MonitoringServer, WebServerConfig
from ..components.capture import ScreenCapture
from ..components.metrics import MetricsCollector

@pytest.fixture
def config():
    """Server configuration for testing."""
    return WebServerConfig(
        host="localhost",
        port=5000,
        debug=True,
        frame_buffer_size=1,
        frame_quality=85
    )

@pytest.fixture
def server(config):
    """Create test server instance."""
    server = MonitoringServer(config)
    server.app.testing = True  # Enable testing mode
    server.app.config['ENV'] = 'testing'  # Set test environment
    return server

@pytest.fixture
def client(server):
    """Create Flask test client."""
    server.app.config['TESTING'] = True
    with server.app.test_client() as client:
        yield client

@pytest.fixture
def socketio_client(server):
    """Create SocketIO test client."""
    return SocketIOTestClient(
        server.app,
        server.socketio,
        flask_test_client=server.app.test_client()
    )

@pytest.fixture
def mock_screen_capture():
    """Mock screen capture component."""
    capture = Mock(spec=ScreenCapture)
    # Create a simple test frame
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    frame[10:50, 10:50] = 255  # White square
    capture.get_frame.return_value = frame
    return capture

@pytest.fixture
def mock_metrics_collector():
    """Mock metrics collector component."""
    collector = Mock(spec=MetricsCollector)
    collector.get_metrics.return_value = {
        "reward": [1.0, 2.0, 3.0],
        "steps": [100, 200, 300]
    }
    return collector

def test_server_init(server):
    """Test server initialization."""
    assert server.app is not None
    assert server.socketio is not None
    assert server.status_manager is not None
    assert server.event_manager is not None
    assert not server._running

def test_dashboard_route(client):
    """Test dashboard page route."""
    response = client.get('/')
    assert response.status_code == 200
    assert b'Pokemon Crystal RL Monitor' in response.data
    assert b'<title>Pokemon Crystal RL Monitor</title>' in response.data

def test_status_route(client):
    """Test status page route."""
    response = client.get('/status')
    assert response.status_code == 200
    assert b'<title>Monitor Status</title>' in response.data
    assert b'<h1 class="text-2xl font-bold">Monitor Status</h1>' in response.data

def test_api_status(client, server):
    """Test status API endpoint."""
    response = client.get(f"{server.config.api_prefix}/status")
    assert response.status_code == 200
    data = json.loads(response.data)
    assert "running" in data
    assert "connections" in data
    assert "frames_sent" in data
    assert "updates_sent" in data

def test_api_metrics(client, server, mock_metrics_collector):
    """Test metrics API endpoint."""
    server.set_metrics_collector(mock_metrics_collector)
    response = client.get(f"{server.config.api_prefix}/metrics")
    assert response.status_code == 200
    data = json.loads(response.data)
    assert "reward" in data
    assert "steps" in data

def test_api_frame(client, server, mock_screen_capture):
    """Test frame API endpoint."""
    server.set_screen_capture(mock_screen_capture)
    response = client.get(f"{server.config.api_prefix}/frame?format=jpeg")
    assert response.status_code == 200
    assert response.mimetype == "image/jpeg"

def test_websocket_connect(socketio_client):
    """Test WebSocket connection."""
    assert socketio_client.is_connected()
    received = socketio_client.get_received()
    assert len(received) > 0
    assert received[0]['name'] == 'status'
    assert received[0]['args'][0]['connected'] is True

def test_websocket_frame(socketio_client, server, mock_screen_capture):
    """Test WebSocket frame streaming."""
    server.set_screen_capture(mock_screen_capture)
    socketio_client.emit('request_frame')
    received = socketio_client.get_received()
    assert len(received) > 0
    frame_event = next(
        (ev for ev in received if ev['name'] == 'frame'),
        None
    )
    assert frame_event is not None
    assert isinstance(frame_event['args'][0], bytes)

def test_websocket_metrics(socketio_client, server, mock_metrics_collector):
    """Test WebSocket metrics updates."""
    server.set_metrics_collector(mock_metrics_collector)
    socketio_client.emit('subscribe_metrics', {
        'metrics': ['reward', 'steps'],
        'since': time.time() - 3600
    })
    received = socketio_client.get_received()
    assert len(received) > 0
    metrics_event = next(
        (ev for ev in received if ev['name'] == 'metrics'),
        None
    )
    assert metrics_event is not None
    assert 'reward' in metrics_event['args'][0]
    assert 'steps' in metrics_event['args'][0]

def test_server_start_stop(server):
    """Test server start/stop functionality."""
    assert server.start()
    assert server._running
    assert server._update_thread is not None
    assert server._server_thread is not None
    
    assert server.stop()
    assert not server._running
    assert not server._update_thread.is_alive()
    assert not server._server_thread.is_alive()

def test_component_integration(server, mock_screen_capture, mock_metrics_collector):
    """Test component integration."""
    server.set_screen_capture(mock_screen_capture)
    server.set_metrics_collector(mock_metrics_collector)
    
    assert server.screen_capture is mock_screen_capture
    assert server.metrics_collector is mock_metrics_collector
    
    # Test frame capture integration
    frame = server.screen_capture.get_frame("raw")
    assert frame is not None
    assert isinstance(frame, np.ndarray)
    
    # Test metrics integration
    metrics = server.metrics_collector.get_metrics()
    assert metrics is not None
    assert "reward" in metrics
    assert "steps" in metrics
