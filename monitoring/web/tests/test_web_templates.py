"""
Tests for web templates functionality.

Tests verify that the templates correctly:
1. Use shared JavaScript utilities
2. Handle WebSocket events
3. Update resource visualizations
4. Handle logging
5. Maintain responsiveness
"""

import os
import pytest
import json
from flask import Flask
from flask.testing import FlaskClient
from flask_socketio import SocketIO
import base64
from unittest.mock import MagicMock, patch

from monitoring.web.server import MonitoringServer, WebServerConfig
from monitoring.data.bus import get_data_bus

# Configuration for testing
TEST_CONFIG = WebServerConfig(
    host="localhost",
    port=8080,
    debug=True,
    enable_api=True,
    enable_websocket=True,
    enable_metrics=True,
    template_dir="templates",
    static_dir="static"
)

@pytest.fixture
def app():
    """Create Flask test app."""
    app = Flask(__name__,
                template_folder=os.path.join(os.path.dirname(__file__), "../templates"),
                static_folder=os.path.join(os.path.dirname(__file__), "../static"))
    app.config["TESTING"] = True
    return app

@pytest.fixture
def test_client(app):
    """Create test client."""
    return app.test_client()

@pytest.fixture
def socketio(app):
    """Create SocketIO for testing."""
    return SocketIO(app, async_mode="threading")

@pytest.fixture
def monitor_server(app, socketio):
    """Create MonitoringServer instance."""
    server = MonitoringServer(TEST_CONFIG)
    server.app = app
    server.socketio = socketio
    return server

def test_dashboard_template(test_client):
    """Test dashboard.html template."""
    response = test_client.get("/")
    assert response.status_code == 200
    html = response.data.decode()
    
    # Verify shared.js is included
    assert 'src="/static/js/shared.js"' in html
    
    # Verify key UI elements exist
    assert 'id="game-screen"' in html
    assert 'id="total-actions"' in html
    assert 'id="llm-decisions"' in html
    assert 'id="memory-debug"' in html

def test_status_template(test_client):
    """Test status.html template."""
    response = test_client.get("/status")
    assert response.status_code == 200
    html = response.data.decode()
    
    # Verify shared.js is included
    assert 'src="/static/js/shared.js"' in html
    
    # Verify key UI elements exist
    assert 'id="cpuBar"' in html
    assert 'id="memoryBar"' in html
    assert 'id="storageBar"' in html
    assert 'id="errorLog"' in html

def test_websocket_frame_events(monitor_server, socketio):
    """Test WebSocket frame event handling."""
    client = socketio.test_client(monitor_server.app)
    
    # Send test frame
    test_frame = base64.b64encode(b"test frame data").decode()
    monitor_server.socketio.emit("frame", {"data": test_frame})
    
    received = client.get_received()
    assert len(received) > 0
    assert received[0]["name"] == "frame"
    assert "data" in received[0]["args"][0]

def test_resource_updates(monitor_server, socketio):
    """Test resource usage updates."""
    client = socketio.test_client(monitor_server.app)
    
    # Send test resource data
    test_data = {
        "cpu_percent": 50.0,
        "memory_usage_mb": 1024,
        "storage_usage_mb": 2048
    }
    monitor_server.socketio.emit("status", test_data)
    
    received = client.get_received()
    assert len(received) > 0
    assert received[0]["name"] == "status"
    resource_data = received[0]["args"][0]
    assert resource_data["cpu_percent"] == 50.0
    assert resource_data["memory_usage_mb"] == 1024

def test_logging_functionality(monitor_server, socketio):
    """Test log entry handling."""
    client = socketio.test_client(monitor_server.app)
    
    # Send test error log
    test_error = {"message": "Test error message", "level": "error"}
    monitor_server.socketio.emit("error", test_error)
    
    received = client.get_received()
    assert len(received) > 0
    assert received[0]["name"] == "error"
    error_data = received[0]["args"][0]
    assert error_data["message"] == "Test error message"
