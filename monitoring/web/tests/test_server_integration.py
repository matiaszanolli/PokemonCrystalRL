"""Integration tests for the monitoring web server.

This module tests the full request/response cycle through the Flask app,
including REST endpoints, WebSocket events, and component interactions.
"""
import json
import time
import threading
import numpy as np
import pytest
from datetime import datetime, timezone
from unittest.mock import Mock, MagicMock, patch

from monitoring.web.server import MonitoringServer, WebServerConfig
from monitoring.web.managers import StatusManager, EventManager
from monitoring.web.services.frame import FrameService
from monitoring.web.services.metrics import MetricsService
from monitoring.components.capture import ScreenCapture
from monitoring.components.metrics import MetricsCollector

pytestmark = [pytest.mark.web, pytest.mark.web_monitoring, pytest.mark.integration]

@pytest.fixture(scope="module")
def mock_components():
    """Create mock components for testing."""
    screen_capture = MagicMock(spec=ScreenCapture)
    screen_capture.pyboy = Mock()
    screen_capture.capture_active = True
    screen_capture.get_latest_screen_bytes = Mock(return_value=b'test_image_data')
    screen_capture.get_latest_screen_data = Mock(return_value={'width': 160, 'height': 144})
    screen_capture.get_frame = Mock(return_value=np.zeros((144, 160, 3), dtype=np.uint8))
    
    metrics_collector = Mock(spec=MetricsCollector)
    metrics_collector.get_metrics.return_value = {
        'fps': 30.0,
        'memory_usage': 100.0
    }
    
    return {
        'screen_capture': screen_capture,
        'metrics_collector': metrics_collector
    }
from monitoring.web.managers import StatusManager, EventManager
from monitoring.web.services.frame import FrameService
from monitoring.web.services.metrics import MetricsService
from monitoring.components.capture import ScreenCapture
from monitoring.components.metrics import MetricsCollector


class TestMonitoringServerSetup:
    """Tests for server initialization and configuration."""
    
    def test_server_creation_with_default_config(self):
        """Test server creation with default configuration."""
        server = MonitoringServer()
        config = server.config
        
        assert config.host == "localhost"
        assert config.port == 8080
        assert not config.debug
        assert config.enable_api
        assert config.enable_websocket
        assert config.enable_metrics
        
        # Check essential components
        assert isinstance(server.status_manager, StatusManager)
        assert isinstance(server.event_manager, EventManager)
        assert isinstance(server.frame_service, FrameService)
        assert isinstance(server.metrics_service, MetricsService)
    
    def test_server_creation_with_custom_config(self):
        """Test server creation with custom configuration."""
        config = WebServerConfig(
            host="0.0.0.0",
            port=5000,
            debug=True,
            frame_buffer_size=5,
            frame_quality=90,
            update_interval=0.05
        )
        
        server = MonitoringServer(config)
        
        assert server.config.host == "0.0.0.0"
        assert server.config.port == 5000
        assert server.config.debug is True
        assert server.config.frame_buffer_size == 5
        assert server.config.frame_quality == 90
        assert server.config.update_interval == 0.05




@pytest.fixture
def test_server(mock_components):
    """Create a test server instance with mocked components."""
    config = WebServerConfig(
        host="localhost",
        port=5000,
        debug=False,
        update_interval=0.1
    )
    
    server = MonitoringServer(config)
    server.set_screen_capture(mock_components['screen_capture'])
    server.set_metrics_collector(mock_components['metrics_collector'])
    
    # Configure test client
    server.app.config['TESTING'] = True
    client = server.app.test_client()
    
    return {
        'server': server,
        'client': client
    }


class TestRESTEndpoints:
    """Integration tests for REST API endpoints."""
    
    def test_status_endpoint(self, test_server):
        """Test the server status endpoint."""
        response = test_server['client'].get('/api/v1/status')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert 'running' in data
        assert 'frame_service' in data
        assert 'metrics_service' in data
    
    def test_training_stats_endpoint(self, test_server):
        """Test the training stats endpoint."""
        response = test_server['client'].get('/api/v1/training/stats')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert 'total_actions' in data
        assert 'actions_per_second' in data
        assert 'llm_calls' in data
        assert 'total_reward' in data
    
    def test_system_status_endpoint(self, test_server):
        """Test the system status endpoint."""
        response = test_server['client'].get('/api/v1/system/status')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert data['status'] == 'running'
        assert 'uptime' in data
        assert data['screen_capture_active'] is True
    
    def test_game_frame_endpoint(self, test_server, mock_components):
        """Test the game frame endpoint."""
        # Test PNG format
        response = test_server['client'].get('/api/v1/game/frame?format=png')
        assert response.status_code == 200
        assert response.mimetype == 'image/png'
        assert response.data == b'test_image_data'
        
        # Test JPEG format
        response = test_server['client'].get('/api/v1/game/frame?format=jpeg')
        assert response.status_code == 200
        assert response.mimetype == 'image/jpeg'
        assert response.data == b'test_image_data'
        
        # Test invalid format
        response = test_server['client'].get('/api/v1/game/frame?format=invalid')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'error' in data
    
    def test_metrics_endpoint(self, test_server, mock_components):
        """Test the metrics endpoint."""
        response = test_server['client'].get('/api/v1/metrics')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert 'fps' in data
        assert 'memory_usage' in data
    
    def test_training_toggle_endpoint(self, test_server):
        """Test the training toggle endpoint."""
        response = test_server['client'].post('/api/v1/training/toggle')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert data['success'] is True
    
    def test_environment_reset_endpoint(self, test_server):
        """Test the environment reset endpoint."""
        response = test_server['client'].post('/api/v1/environment/reset')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert data['success'] is True


class TestWebSocketCommunication:
    """Integration tests for WebSocket communication."""
    
    @pytest.fixture
    def socket_client(self, test_server):
        """Create a test WebSocket client."""
        return test_server['server'].socketio.test_client(test_server['server'].app)
    
    def test_client_connection(self, socket_client, test_server):
        """Test client connection event handling."""
        assert socket_client.is_connected()
        # Status manager should track the connection
        assert test_server['server'].status_manager.get_connected_clients() > 0
    
    def test_metrics_subscription(self, socket_client, mock_components):
        """Test metrics subscription and updates."""
        # Subscribe to metrics
        socket_client.emit('subscribe_metrics', {
            'metrics': ['fps', 'memory_usage'],
            'since': time.time() - 3600
        })
        
        # Wait for response
        response = socket_client.get_received()
        assert len(response) > 0
        
        # Verify metrics data
        metrics_events = [r for r in response if r['name'] == 'metrics']
        assert len(metrics_events) > 0
        data = metrics_events[0]['args'][0]
        assert 'fps' in data
        assert 'memory_usage' in data
    
    def test_frame_request(self, socket_client, mock_components):
        """Test frame request handling."""
        # Request a frame
        socket_client.emit('request_frame')
        
        # Verify the request was handled
        mock_components['screen_capture'].get_latest_screen_bytes.assert_called()
        
        # In a real scenario, we'd verify frame data was sent
        # but in test mode, the frame service is mocked


class TestServerLifecycle:
    """Integration tests for server lifecycle management."""
    
    def test_server_start_stop(self, test_server):
        """Test server start and stop operations."""
        server = test_server['server']
        
        # Start server
        assert server.start() is True
        assert server._running is True
        assert isinstance(server._update_thread, threading.Thread)
        assert isinstance(server._server_thread, threading.Thread)
        
        # Small delay to ensure threads are running
        time.sleep(0.2)
        assert server._update_thread.is_alive()
        assert server._server_thread.is_alive()
        
        # Stop server
        assert server.stop() is True
        assert server._running is False
        
        # Verify threads stopped
        time.sleep(0.2)
        assert not server._update_thread.is_alive()
        assert not server._server_thread.is_alive()
    
    def test_component_updates(self, test_server, mock_components):
        """Test handling component updates."""
        server = test_server['server']
        
        # Update screen capture
        new_screen_capture = Mock(spec=ScreenCapture)
        server.set_screen_capture(new_screen_capture)
        assert server.screen_capture == new_screen_capture
        assert server.frame_service.screen_capture == new_screen_capture
        assert server.system_api.screen_capture == new_screen_capture
        assert server.game_api.screen_capture == new_screen_capture
        
        # Update metrics collector
        new_metrics_collector = Mock(spec=MetricsCollector)
        server.set_metrics_collector(new_metrics_collector)
        assert server.metrics_collector == new_metrics_collector
        assert server.metrics_service.metrics_collector == new_metrics_collector
    
    def test_error_handling(self, test_server, mock_components):
        """Test error handling in server operations."""
        server = test_server['server']
        client = test_server['client']
        
        # Test API error handling
        mock_components['screen_capture'].get_latest_screen_bytes.side_effect = Exception("Test error")
        response = client.get('/api/v1/game/frame')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'error' in data
        
        # Test metrics error handling
        mock_components['metrics_collector'].get_metrics.side_effect = Exception("Test error")
        response = client.get('/api/v1/metrics')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert not data  # Should return empty dict on error
