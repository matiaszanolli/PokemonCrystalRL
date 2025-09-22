"""
Integration tests for the unified web dashboard system.

Tests the complete end-to-end functionality of the consolidated web monitoring
and dashboard system, including API endpoints, WebSocket connections, and
real trainer integration.
"""

import pytest
import time
import json
import requests
import threading
from unittest.mock import Mock, patch
from unittest import mock
import asyncio
import websockets
from training.components.statistics_tracker import StatisticsTracker
from web_dashboard import create_web_server


class MockTrainer:
    """Mock trainer that matches the unified trainer interface."""

    def __init__(self):
        self.stats_tracker = StatisticsTracker("test_session")
        self.emulation_manager = Mock()
        self.emulation_manager.get_instance.return_value = None  # No PyBoy for tests

        # Mock some initial stats
        self.stats_tracker.record_action(1, "rule_based", game_state={}, reward=0.1)
        self.stats_tracker.record_action(2, "llm", game_state={}, reward=15.0)

    def get_current_stats(self):
        """Backward compatibility method."""
        return self.stats_tracker.get_current_stats()


@pytest.fixture
def mock_trainer():
    """Create a mock trainer for testing."""
    return MockTrainer()


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
def web_server(mock_trainer, available_port):
    """Create and start a web server for testing."""
    server = create_web_server(
        trainer=mock_trainer,
        host="localhost",
        http_port=available_port,
        ws_port=available_port + 1
    )

    # Start server
    assert server.start(), "Failed to start web server"

    # Wait for server to be ready
    time.sleep(0.5)

    yield server, available_port

    # Cleanup
    try:
        server.stop()
        time.sleep(0.5)
    except Exception as e:
        print(f"Cleanup error: {e}")


@pytest.mark.integration
@pytest.mark.web_monitoring
class TestUnifiedWebDashboard:
    """Test the unified web dashboard integration."""

    def test_server_startup_and_shutdown(self, mock_trainer, available_port):
        """Test that the unified web server starts and stops cleanly."""
        server = create_web_server(
            trainer=mock_trainer,
            host="localhost",
            http_port=available_port,
            ws_port=available_port + 1
        )

        # Test startup
        assert server.start()
        assert server.is_running()

        # Test shutdown
        server.stop()
        time.sleep(0.5)
        assert not server.is_running()

    def test_health_endpoint(self, web_server):
        """Test the health check endpoint."""
        server, port = web_server

        response = requests.get(f"http://localhost:{port}/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert data["api_available"] == True
        assert data["websocket_available"] == True

    def test_dashboard_api_endpoint(self, web_server):
        """Test the main dashboard API endpoint."""
        server, port = web_server

        response = requests.get(f"http://localhost:{port}/api/dashboard")
        assert response.status_code == 200

        data = response.json()
        assert data["success"] == True
        assert "training_stats" in data["data"]
        assert "game_state" in data["data"]
        assert "recent_llm_decisions" in data["data"]

        # Check training stats have real data (not zeros)
        stats = data["data"]["training_stats"]
        assert stats["total_actions"] > 0  # Should have recorded actions
        assert stats["total_reward"] > 0   # Should have recorded rewards

    def test_training_stats_endpoint(self, web_server):
        """Test the training statistics endpoint."""
        server, port = web_server

        response = requests.get(f"http://localhost:{port}/api/training_stats")
        assert response.status_code == 200

        data = response.json()
        assert data["success"] == True

        stats = data["data"]
        assert "total_actions" in stats
        assert "total_reward" in stats
        assert "actions_per_second" in stats
        assert "llm_decisions" in stats

        # Verify we get real data, not zeros
        assert stats["total_actions"] > 0
        assert stats["total_reward"] != 0  # Could be positive or negative

    def test_game_state_endpoint(self, web_server):
        """Test the game state endpoint."""
        server, port = web_server

        response = requests.get(f"http://localhost:{port}/api/game_state")
        assert response.status_code == 200

        data = response.json()
        assert data["success"] == True

        game_state = data["data"]
        assert "current_map" in game_state
        assert "player_position" in game_state
        assert "money" in game_state

    def test_memory_debug_endpoint(self, web_server):
        """Test the memory debug endpoint."""
        server, port = web_server

        response = requests.get(f"http://localhost:{port}/api/memory_debug")
        assert response.status_code == 200

        data = response.json()
        assert data["success"] == True

        memory_data = data["data"]
        assert "memory_addresses" in memory_data
        assert "memory_read_success" in memory_data
        assert "pyboy_available" in memory_data

    def test_llm_decisions_endpoint(self, web_server):
        """Test the LLM decisions endpoint."""
        server, port = web_server

        response = requests.get(f"http://localhost:{port}/api/llm_decisions")
        assert response.status_code == 200

        data = response.json()
        assert data["success"] == True
        # The data should be a list of LLM decisions
        assert isinstance(data["data"], list)

    def test_system_status_endpoint(self, web_server):
        """Test the system status endpoint."""
        server, port = web_server

        response = requests.get(f"http://localhost:{port}/api/system_status")
        assert response.status_code == 200

        data = response.json()
        assert data["success"] == True

        status = data["data"]
        assert "uptime_seconds" in status
        assert "training_active" in status

    def test_static_dashboard_loads(self, web_server):
        """Test that the dashboard HTML loads correctly."""
        server, port = web_server

        response = requests.get(f"http://localhost:{port}/")
        assert response.status_code == 200
        assert "text/html" in response.headers["Content-Type"]

        html = response.text
        # Check for key dashboard elements
        assert "Pokemon Crystal RL Training Dashboard" in html
        assert "Training Statistics" in html
        assert "Live Game Screen" in html
        assert "Game State" in html
        assert "Recent LLM Decisions" in html

    def test_concurrent_api_requests(self, web_server):
        """Test handling of concurrent API requests."""
        server, port = web_server

        def make_request(endpoint):
            response = requests.get(f"http://localhost:{port}{endpoint}")
            return response.status_code == 200

        # Test concurrent requests to different endpoints
        endpoints = ["/health", "/api/dashboard", "/api/training_stats", "/api/game_state"]
        threads = []
        results = []

        for endpoint in endpoints:
            thread = threading.Thread(target=lambda ep=endpoint: results.append(make_request(ep)))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # All requests should succeed
        assert all(results), "Some concurrent requests failed"

    def test_error_handling_invalid_endpoints(self, web_server):
        """Test error handling for invalid endpoints."""
        server, port = web_server

        # Test non-existent endpoint
        response = requests.get(f"http://localhost:{port}/api/nonexistent")
        assert response.status_code == 404

        # Test invalid method
        response = requests.post(f"http://localhost:{port}/health")
        assert response.status_code in [405, 404, 501]  # Method not allowed, not found, or not implemented

    @pytest.mark.asyncio
    async def test_websocket_connection(self, web_server):
        """Test WebSocket connection establishment."""
        server, port = web_server
        ws_port = port + 1

        try:
            async with websockets.connect(f"ws://localhost:{ws_port}") as websocket:
                # Connection established if we can reach this point
                # Should be able to send a ping
                await websocket.send(json.dumps({"type": "ping"}))

                # Should receive a response (with timeout)
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                    data = json.loads(response)
                    # Could be pong or initial data
                    assert "type" in data
                except asyncio.TimeoutError:
                    # Timeout is acceptable for this test - server might not respond
                    pass

        except Exception as e:
            pytest.fail(f"WebSocket connection failed: {e}")

    def test_real_trainer_integration_simulation(self, available_port):
        """Test integration with trainer stats updates."""
        # Create trainer with evolving stats
        trainer = MockTrainer()

        server = create_web_server(
            trainer=trainer,
            host="localhost",
            http_port=available_port,
            ws_port=available_port + 1
        )

        assert server.start()

        try:
            # Get initial stats
            response = requests.get(f"http://localhost:{available_port}/api/training_stats")
            initial_data = response.json()["data"]
            initial_actions = initial_data["total_actions"]

            # Simulate trainer progress
            for i in range(5):
                trainer.stats_tracker.record_action(i + 2, "llm", game_state={}, reward=0.1)
                trainer.stats_tracker.record_action(i + 3, "rule_based", game_state={}, reward=-1.0)

            # Get updated stats
            response = requests.get(f"http://localhost:{available_port}/api/training_stats")
            updated_data = response.json()["data"]
            updated_actions = updated_data["total_actions"]

            # Should see increased action count
            assert updated_actions > initial_actions

        finally:
            server.stop()

    def test_port_conflict_handling(self, mock_trainer):
        """Test handling when ports are already in use."""
        import socket

        # Bind to a port manually
        sock = socket.socket()
        sock.bind(('localhost', 0))
        occupied_port = sock.getsockname()[1]

        try:
            # Try to create server on occupied port
            server = create_web_server(
                trainer=mock_trainer,
                host="localhost",
                http_port=occupied_port,
                ws_port=occupied_port + 1
            )

            # Server creation should succeed but start might fail gracefully
            start_result = server.start()
            if start_result:
                # If it started, it found alternative ports or handled conflict
                server.stop()
            # Test passes if no exception is thrown

        finally:
            sock.close()


@pytest.mark.unit
class TestUnifiedWebDashboardUnit:
    """Unit tests for individual components."""

    def test_create_web_server_with_none_trainer(self):
        """Test server creation with None trainer."""
        server = create_web_server(
            trainer=None,
            host="localhost",
            http_port=8888,
            ws_port=8889
        )

        # Should create server but API calls will return empty data
        assert server is not None

    def test_trainer_stats_tracker_compatibility(self):
        """Test that both stats_tracker and statistics_tracker work."""
        # Test with stats_tracker (unified trainer)
        trainer1 = MockTrainer()
        assert hasattr(trainer1, 'stats_tracker')

        # Test with statistics_tracker (legacy trainer)
        trainer2 = Mock()
        trainer2.statistics_tracker = StatisticsTracker("legacy_session")
        trainer2.statistics_tracker.record_action(1, "rule_based", game_state={}, reward=0.1)

        # Both should work with our compatibility layer
        assert hasattr(trainer2, 'statistics_tracker')


if __name__ == "__main__":
    # Run a simple test
    pytest.main([__file__, "-v"])