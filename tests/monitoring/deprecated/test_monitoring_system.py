"""
Unit tests for the Pokemon Crystal RL monitoring system.

Tests the interaction between:
- Web server
- Web interface
- Web monitor
- Data bus integration
- Error handling
"""

import pytest
import asyncio
import json
import time
import threading
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import os
import numpy as np
import websockets

from monitoring import UnifiedMonitor, MonitorConfig
from monitoring.web_interface import WebInterface
from monitoring import UnifiedMonitor, MonitorConfig
from monitoring.data_bus import DataType
from monitoring.error_handler import ErrorCategory, ErrorSeverity


@pytest.fixture
def mock_data_bus():
    """Create a mock data bus."""
    mock = Mock()
    mock.publish = Mock()
    mock.subscribe = Mock()
    return mock


@pytest.fixture
def temp_dir():
    """Create a temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_server_config():
    """Create a mock server configuration."""
    return ServerConfig(
        host="localhost",
        port=8080,
        static_dir=str(temp_dir()),
        debug=True
    )


@pytest.fixture
def mock_monitor_config():
    """Create a mock monitor configuration."""
    return MonitorConfig(
        host="localhost",
        port=8080,
        static_dir=str(temp_dir()),
        data_dir=str(temp_dir()),
        update_interval=0.1,
        snapshot_interval=1.0,
        max_events=10,
        max_snapshots=5,
        debug=True
    )


class TestWebServer:
    """Test web server functionality."""
    
    @pytest.mark.asyncio
    async def test_server_initialization(self, mock_server_config):
        """Test server initialization."""
        server = WebServer(mock_server_config)
        assert server.config == mock_server_config
        assert not server.is_running
        assert server.start_time is None
    
    @pytest.mark.asyncio
    async def test_server_start_stop(self, mock_server_config):
        """Test server start and stop."""
        server = WebServer(mock_server_config)
        
        # Start server
        await server.start()
        assert server.is_running
        assert server.start_time is not None
        
        # Stop server
        await server.shutdown()
        assert not server.is_running
    
    @pytest.mark.asyncio
    async def test_websocket_connection(self, mock_server_config):
        """Test WebSocket connection handling."""
        server = WebServer(mock_server_config)
        await server.start()
        
        # Connect WebSocket client
        uri = f"ws://{server.config.host}:{server.config.port}/ws"
        async with websockets.connect(uri) as websocket:
            # Send test message
            await websocket.send(json.dumps({
                "type": "test",
                "data": {"message": "Hello"}
            }))
            
            # Should receive echo response
            response = await websocket.recv()
            data = json.loads(response)
            assert data["type"] == "test"
            assert data["data"]["message"] == "Hello"
        
        await server.shutdown()
    
    @pytest.mark.asyncio
    async def test_api_endpoints(self, mock_server_config, aiohttp_client):
        """Test HTTP API endpoints."""
        server = WebServer(mock_server_config)
        client = await aiohttp_client(server.app)
        
        # Test status endpoint
        resp = await client.get("/api/status")
        assert resp.status == 200
        data = await resp.json()
        assert "status" in data
        assert "uptime" in data
        
        # Test metrics endpoint
        resp = await client.get("/api/metrics")
        assert resp.status == 200
        data = await resp.json()
        assert "metrics" in data
        
        # Test events endpoint
        resp = await client.get("/api/events")
        assert resp.status == 200
        data = await resp.json()
        assert "events" in data
        
        await server.shutdown()


class TestWebInterface:
    """Test web interface functionality."""
    
    def test_interface_initialization(self, temp_dir):
        """Test interface initialization."""
        interface = WebInterface(str(temp_dir))
        assert interface.static_dir == temp_dir
        assert len(interface.components) > 0
    
    def test_component_building(self, temp_dir):
        """Test building React components."""
        interface = WebInterface(str(temp_dir))
        interface.build_components()
        
        # Check component files were created
        components_dir = temp_dir / "components"
        assert components_dir.exists()
        assert any(components_dir.glob("*.jsx"))
        
        # Check metadata file
        metadata_file = temp_dir / "components.json"
        assert metadata_file.exists()
        
        with open(metadata_file) as f:
            metadata = json.load(f)
            assert len(metadata) > 0
    
    def test_component_validation(self, temp_dir):
        """Test component validation."""
        interface = WebInterface(str(temp_dir))
        
        # Test valid component
        assert interface.validate_component("Dashboard")
        
        # Test invalid component
        assert not interface.validate_component("NonExistent")
    
    def test_component_metadata(self, temp_dir):
        """Test component metadata handling."""
        interface = WebInterface(str(temp_dir))
        
        # Save metadata
        interface.save_component_metadata()
        
        # Load metadata
        metadata = interface.load_component_metadata()
        assert len(metadata) > 0
        assert "Dashboard" in metadata


class TestUnifiedMonitor:
    """Test web monitor functionality."""
    
    def test_monitor_initialization(self, mock_monitor_config, mock_data_bus):
        """Test monitor initialization."""
        with patch("monitoring.unified_monitor.get_data_bus",
                  return_value=mock_data_bus):
            monitor = UnifiedMonitor(mock_monitor_config)
            
            assert monitor.config == mock_monitor_config
            assert not monitor.is_monitoring
            assert monitor.error_handler is not None
    
    @pytest.mark.asyncio
    async def test_monitor_start_stop(self, mock_monitor_config, mock_data_bus):
        """Test monitor start and stop."""
        with patch("monitoring.unified_monitor.get_data_bus",
                  return_value=mock_data_bus):
            monitor = UnifiedMonitor(mock_monitor_config)
            
            # Start monitor
            monitor.start_monitoring()
            assert monitor.is_monitoring
            
            # Stop monitor
            monitor.stop_monitoring()
            assert not monitor.is_monitoring
    
    def test_event_handling(self, mock_monitor_config, mock_data_bus):
        """Test event handling."""
        with patch("monitoring.unified_monitor.get_data_bus",
                  return_value=mock_data_bus):
            monitor = UnifiedMonitor(mock_monitor_config)
            
            # Test metric update
            monitor.update_stats({"accuracy": 0.8})
            assert monitor.current_stats["accuracy"] == 0.8


class TestMonitoringIntegration:
    """Test monitoring system integration."""
    
    @pytest.mark.asyncio
    async def test_full_monitoring_stack(self, mock_monitor_config, mock_data_bus,
                                       temp_dir):
        """Test full monitoring stack integration."""
        with patch("monitoring.unified_monitor.get_data_bus",
                  return_value=mock_data_bus):
            # Initialize components
            monitor = UnifiedMonitor(mock_monitor_config)
            monitor.start_monitoring()
            
            # Simulate training updates
            for i in range(5):
                monitor.update_stats({
                    "episode": i,
                    "reward": i * 0.5,
                    "steps": i * 100
                })
                monitor.add_performance_metric("loss", 1.0 - (i * 0.1))
                monitor.add_performance_metric("accuracy", i * 0.2)
                
                await asyncio.sleep(0.1)
            
            # Check data was recorded
            assert len(monitor.current_stats) > 0
            
            monitor.stop_monitoring()


class TestErrorHandling:
    """Test error handling in monitoring system."""
    
    def test_error_handling_web_server(self, mock_server_config):
        """Test error handling in web server."""
        server = WebServer(mock_server_config)
        
        # Simulate error in request handling
        error = Exception("Test error")
        server.error_handler.handle_error(
            error,
            severity=ErrorSeverity.ERROR,
            category=ErrorCategory.SYSTEM,
            component="web_server"
        )
        
        # Error should be logged and handled
        assert True  # No exception raised
    
    def test_error_handling_unified_monitor(self, mock_monitor_config, mock_data_bus):
        """Test error handling in web monitor."""
        with patch("monitoring.unified_monitor.get_data_bus",
                  return_value=mock_data_bus):
            monitor = UnifiedMonitor(mock_monitor_config)
            
            # Simulate error handling
            error = Exception("Test error")
            monitor.error_handler.handle_error(
                error,
                severity=ErrorSeverity.ERROR,
                category=ErrorCategory.SYSTEM,
                component="monitor"
            )
            
            # Error should be handled
            assert monitor.error_handler is not None
    
    @pytest.mark.asyncio
    async def test_error_recovery(self, mock_monitor_config, mock_data_bus):
        """Test error recovery mechanisms."""
        with patch("monitoring.unified_monitor.get_data_bus",
                  return_value=mock_data_bus):
            monitor = UnifiedMonitor(mock_monitor_config)
            monitor.start_monitoring()
            
            # Simulate recoverable error
            error = Exception("Recoverable error")
            monitor.error_handler.handle_error(
                error,
                severity=ErrorSeverity.WARNING,
                category=ErrorCategory.SYSTEM,
                component="test"
            )
            
            # Monitor should still be running
            assert monitor.is_monitoring
            
            # Simulate critical error
            error = Exception("Critical error")
            monitor.error_handler.handle_error(
                error,
                severity=ErrorSeverity.CRITICAL,
                category=ErrorCategory.SYSTEM,
                component="test"
            )
            
            # Monitor should still be running (UnifiedMonitor handles critical errors differently)
            await asyncio.sleep(0.1)
            assert monitor.is_monitoring
            
            monitor.stop_monitoring()


if __name__ == "__main__":
    pytest.main(["-v", __file__])
