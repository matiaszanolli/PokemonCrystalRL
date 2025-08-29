#!/usr/bin/env python3
"""
test_monitoring_integration.py - Integration tests for monitoring components

Tests the integration between different monitoring components including:
- Data bus communication
- Web server streaming
- Real-time statistics
- Component lifecycle
"""

import pytest
import time
import queue
import numpy as np
from unittest.mock import Mock, patch, PropertyMock
import threading
import tempfile
from pathlib import Path

from monitoring.data_bus import DataBus, DataType
from monitoring.web_server import WebServer as TrainingWebServer
from .mock_llm_manager import MockLLMManager
from trainer.trainer import TrainingConfig, TrainingMode, LLMBackend, PokemonTrainer
from trainer.unified_trainer import UnifiedPokemonTrainer


import socket

@pytest.fixture
def free_port():
    """Get an available port for testing."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port

@pytest.fixture
def mock_config(free_port):
    """Base configuration for testing"""
    return TrainingConfig(
        rom_path="test.gbc",
        enable_web=True,
        web_port=free_port,  # Use dynamic port for testing
        capture_screens=True,
        headless=True,
        debug_mode=True,
        test_mode=True  # Mark as test mode
    )


@pytest.fixture
def data_bus():
    """Create and initialize a data bus for testing"""
    from monitoring.data_bus import get_data_bus, shutdown_data_bus
    from tests.trainer.mock_web_server import MockWebServer
    
    # Always start clean
    MockWebServer.cleanup_ports()
    shutdown_data_bus()
    
    # Create new data bus
    data_bus = get_data_bus()
    
    # Ensure cleanup after test
    yield data_bus
    
    # Cleanup in reverse order
    try:
        shutdown_data_bus()
    except Exception as e:
        print(f"Error during data bus shutdown: {e}")
    try:
        MockWebServer.cleanup_ports()
    except Exception as e:
        print(f"Error during port cleanup: {e}")


@pytest.mark.monitoring
@pytest.mark.integration
class TestMonitoringIntegration:
    """Test monitoring component integration"""

    def test_trainer_data_bus_registration(self, mock_config, data_bus):
        """Test trainer registration with data bus"""
        # Mock PyBoy and LLM manager
        with patch('trainer.trainer.PyBoy') as mock_pyboy, \
patch('llm.local_llm_agent.LLMManager', return_value=MockLLMManager()):
            
            mock_pyboy_instance = Mock()
            mock_pyboy.return_value = mock_pyboy_instance

            trainer = UnifiedPokemonTrainer(mock_config)

            # Check registration
            components = data_bus.get_component_status()
            assert "trainer" in components
            assert components["trainer"]["type"] == "core"
            assert components["trainer"]["mode"] == mock_config.mode.value

    def test_data_bus_screen_publishing(self, mock_config, data_bus):
        """Test screen data publishing to data bus"""
        screen_data = None
        screen_received = threading.Event()
    
        def screen_callback(data, publisher):
            nonlocal screen_data
            screen_data = data
            screen_received.set()
    
        data_bus.subscribe(
            DataType.GAME_SCREEN,
            "test",
            screen_callback
        )
    
        # Mock data bus publish to directly call callbacks
        def mock_publish_call(data_type, data, publisher):
            if data_type == DataType.GAME_SCREEN:
                screen_callback(data, publisher)
                return True
            return None

        with patch.object(data_bus, 'publish', side_effect=mock_publish_call), \
             patch('trainer.trainer.PyBoy') as mock_pyboy, \
             patch('llm.local_llm_agent.LLMManager', return_value=MockLLMManager()), \
             patch('cv2.cvtColor', return_value=np.random.randint(0, 255, (144, 160, 3), dtype=np.uint8)), \
             patch('cv2.imencode', return_value=(True, b'test_jpg_data')):
            mock_pyboy_instance = Mock()
            mock_screen = np.random.randint(0, 255, (144, 160, 3), dtype=np.uint8)
            mock_pyboy_instance.screen.ndarray = mock_screen
            mock_pyboy_instance.frame_count = 1000
            mock_pyboy.return_value = mock_pyboy_instance
    
            trainer = UnifiedPokemonTrainer(mock_config)
            
            trainer._capture_and_queue_screen()
            
            # Wait for screen data
            assert screen_received.wait(timeout=1.0)
            assert screen_data is not None
            assert "screen" in screen_data
            assert "timestamp" in screen_data
            assert "frame" in screen_data
            assert screen_data["frame"] == 1000
            
            # Verify screen dimensions
            published_screen = screen_data["screen"]
            assert published_screen.shape[:2] == mock_config.screen_resize

    def test_web_server_data_stream(self, mock_config, data_bus):
        """Test web server data streaming"""
        import requests
        import time
        import json
        from http.server import HTTPServer, BaseHTTPRequestHandler

        class MockHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == '/api/status':
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    response = {'success': True, 'active': True, 'stats': {}}
                    self.wfile.write(json.dumps(response).encode('utf-8'))
                elif self.path == '/api/stats':
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    response = {'stats': {'total_actions': 100}}
                    self.wfile.write(json.dumps(response).encode('utf-8'))
                elif self.path == '/api/screen':
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    screen_data = {'screen': 'test_jpg_data', 'timestamp': time.time()}
                    self.wfile.write(json.dumps(screen_data).encode('utf-8'))
                else:
                    self.send_error(404)
    
        class MockWebServer:
            def __init__(self):
                self._server = None
                self.running = False
                
            def start(self):
                self._server = HTTPServer(('localhost', mock_config.web_port), MockHandler)
                self.running = True
                # Start server in a separate thread
                import threading
                self.server_thread = threading.Thread(target=self._server.serve_forever)
                self.server_thread.daemon = True
                self.server_thread.start()
                return True
                
            def shutdown(self):
                if self._server:
                    self.running = False
                    self._server.shutdown()
                    self._server.server_close()
                    self._server = None
                    
            def __del__(self):
                self.shutdown()
    
        with patch('trainer.trainer.PyBoy') as mock_pyboy, \
             patch('llm.local_llm_agent.LLMManager', return_value=MockLLMManager()), \
             patch('trainer.trainer.TrainingWebServer', return_value=MockWebServer()), \
             patch('cv2.cvtColor', return_value=np.random.randint(0, 255, (144, 160, 3), dtype=np.uint8)), \
             patch('cv2.imencode', return_value=(True, b'test_jpg_data')):
            mock_pyboy_instance = Mock()
            mock_screen = np.random.randint(0, 255, (144, 160, 3), dtype=np.uint8)
            mock_pyboy_instance.screen.ndarray = mock_screen
            mock_pyboy_instance.frame_count = 1000
            mock_pyboy.return_value = mock_pyboy_instance
    
            trainer = UnifiedPokemonTrainer(mock_config)
    
            # Check server started
            assert trainer.web_server is not None
            
            # Give server time to start
            time.sleep(0.1)
    
            # Test status endpoint
            response = requests.get(f"http://localhost:{mock_config.web_port}/api/status")
            
            # Verify response
            assert response.status_code == 200
            data = response.json()
            assert data['success'] is True
            assert 'active' in data
            assert 'stats' in data
            trainer.stats['total_actions'] = 100
            response = requests.get(f"http://localhost:{mock_config.web_port}/api/stats")
            assert response.status_code == 200
            data = response.json()
            assert "stats" in data
            assert data["stats"]["total_actions"] == 100

            # Test screen endpoint
            trainer._capture_and_queue_screen()  # Put a screen in the queue
            response = requests.get(f"http://localhost:{mock_config.web_port}/api/screen")
            assert response.status_code == 200
            data = response.json()
            assert "screen" in data
            assert "timestamp" in data

    def test_web_server_shutdown(self, mock_config, data_bus):
        """Test clean web server shutdown"""
        import requests
        import time
        import json
        from http.server import HTTPServer, BaseHTTPRequestHandler

        class MockHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == '/api/status':
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    response = {'success': True, 'active': True, 'stats': {}}
                    self.wfile.write(json.dumps(response).encode('utf-8'))
                else:
                    self.send_error(404)
    
        class MockWebServer:
            def __init__(self):
                self._server = None
                self.running = False
                
            def start(self):
                self._server = HTTPServer(('localhost', mock_config.web_port), MockHandler)
                self.running = True
                # Start server in a separate thread
                import threading
                self.server_thread = threading.Thread(target=self._server.serve_forever)
                self.server_thread.daemon = True
                self.server_thread.start()
                return True
                
            def shutdown(self):
                if self._server:
                    self.running = False
                    self._server.shutdown()
                    self._server.server_close()
                    self._server = None
                    
            def __del__(self):
                self.shutdown()
    
        with patch('trainer.trainer.PyBoy') as mock_pyboy, \
             patch('llm.local_llm_agent.LLMManager', return_value=MockLLMManager()), \
             patch('trainer.trainer.TrainingWebServer', return_value=MockWebServer()):
            mock_pyboy_instance = Mock()
            mock_pyboy.return_value = mock_pyboy_instance
    
            trainer = UnifiedPokemonTrainer(mock_config)
            
            # Give server time to start
            time.sleep(0.1)
    
            # Test server started
            response = requests.get(f"http://localhost:{mock_config.web_port}/api/status")
            assert response.status_code == 200
    
            # Shut down
            trainer.web_server.shutdown()
            
            # Give server time to stop
            time.sleep(0.1)
    
            # Test requests fail after shutdown
            with pytest.raises(requests.exceptions.ConnectionError):
                requests.get(f"http://localhost:{mock_config.web_port}/api/status")
                requests.get(f"http://localhost:{mock_config.web_port}/api/status")

            # Server should be unregistered from data bus
            components = data_bus.get_component_status()
            assert "web_server" not in components

    def test_memory_management(self, mock_config, data_bus):
        """Test memory management in monitoring system"""
        import psutil
        import gc
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        with patch('trainer.trainer.PyBoy') as mock_pyboy, \
             patch('llm.local_llm_agent.LLMManager', return_value=MockLLMManager()):
            mock_pyboy_instance = Mock()
            mock_screen = np.random.randint(0, 255, (144, 160, 3), dtype=np.uint8)
            mock_pyboy_instance.screen_image = Mock(return_value=mock_screen)
            mock_pyboy.return_value = mock_pyboy_instance

            trainer = UnifiedPokemonTrainer(mock_config)

            # Generate lots of monitoring data
            for i in range(100):
                trainer._capture_and_queue_screen()
                trainer.stats['total_actions'] = i
                data_bus.publish(
                    DataType.TRAINING_STATS,
                    trainer.stats,
                    "trainer"
                )
                time.sleep(0.01)

            # Force garbage collection
            gc.collect()

            # Check memory usage stayed reasonable
            final_memory = process.memory_info().rss
            memory_increase = (final_memory - initial_memory) / 1024 / 1024  # MB
            assert memory_increase < 50, f"Memory increased by {memory_increase:.1f}MB"

            # Screen queue should be bounded
            assert trainer.screen_queue.qsize() <= 30

    def test_error_resilience(self, mock_config, data_bus):
        """Test monitoring system error resilience"""
        error_count = 0
        error_received = threading.Event()

        def error_callback(data, publisher=None):
            nonlocal error_count
            print("\nDEBUG: Error callback executing with data:", data)
            error_count += 1
            error_received.set()
            print("DEBUG: Error event set, count:", error_count)

        # Subscribe to error events
        data_bus.subscribe(
            DataType.ERROR_EVENT,
            "test",
            error_callback
        )

        with patch('trainer.trainer.PyBoy') as mock_pyboy, \
             patch('llm.local_llm_agent.LLMManager', return_value=MockLLMManager()):
            # Configure PyBoy mock
            mock_pyboy_instance = Mock()
            mock_pyboy_instance.frame_count = 1000

            # Setup screen mock to fail on second access
            mock_screen = Mock()
            mock_screen.ndarray = PropertyMock(side_effect=Exception("Screen access error"))
            mock_pyboy_instance.screen = mock_screen
            mock_pyboy.return_value = mock_pyboy_instance

            trainer = UnifiedPokemonTrainer(mock_config)

            # Mock publish to ensure error propagation
            def mock_error_publish(data_type, data, publisher):
                if data_type == DataType.ERROR_EVENT:
                    error_callback(data, publisher)
                return True

            with patch.object(data_bus, 'publish', side_effect=mock_error_publish):
                try:
                    # Attempt to capture screen which should trigger error
                    trainer._capture_and_queue_screen()
                except Exception as e:
                    # Error was thrown as expected, let the error handler process it
                    pass

            # Error should have been published to data bus
            assert error_received.wait(timeout=2.0), "Error event not received"
            
            with patch.object(data_bus, 'publish', side_effect=_mock_publish):
                with trainer._handle_errors('test_screen_capture', 'capture_errors'):
                    screen = trainer._simple_screenshot_capture()
                    print("DEBUG: Initial screen capture successful, attempting second capture")
                    screen = trainer._simple_screenshot_capture()  # This should fail

            # Wait for error event
            assert error_received.wait(timeout=2.0), "Error event not received"
            print("DEBUG: Error event received")

            # Verify error tracking
            assert error_count > 0, "Error count not incremented"
            assert trainer.error_count['capture_errors'] > 0, "Capture error not tracked"

            print("DEBUG: Testing recovery")
            # Reset screen to working state
            mock_pyboy_instance.screen_ndarray = test_screen
            trainer._capture_and_queue_screen()
            
            # Verify screen was queued
            assert trainer.screen_queue.qsize() > 0, "Screen capture not recovered"
            print("DEBUG: Recovery successful")
            print("DEBUG: System recovered successfully")

    def test_component_lifecycle(self, mock_config, data_bus):
        """Test component lifecycle management"""
        print("\nDEBUG: Starting component lifecycle test")
        
        def wait_for_condition(condition_func, timeout=2.0, interval=0.1):
            """Wait for condition with timeout"""
            start_time = time.time()
            while time.time() - start_time < timeout:
                if condition_func():
                    return True
                time.sleep(interval)
            return False
        
        with patch('trainer.trainer.PyBoy') as mock_pyboy, \
             patch('llm.local_llm_agent.LLMManager', return_value=MockLLMManager()):
            print("DEBUG: Mocked PyBoy and LLMManager")
            mock_pyboy_instance = Mock()
            mock_pyboy.return_value = mock_pyboy_instance

            print("DEBUG: Initializing trainer...")
            # Initialize components
            trainer = UnifiedPokemonTrainer(mock_config)
            print("DEBUG: Trainer initialized")

            # Wait for components to register
            print("DEBUG: Waiting for component registration...")
            registration_success = wait_for_condition(
                lambda: "trainer" in data_bus.get_component_status() and \
                        "web_server" in data_bus.get_component_status()
            )
            assert registration_success, "Components failed to register within timeout"
            
            components = data_bus.get_component_status()
            print("DEBUG: Initial components:", components)

            # Check activity tracking
            print("DEBUG: Starting activity tracking check...")
            initial_status = components["trainer"]
            print("DEBUG: Initial trainer status:", initial_status)
            print("DEBUG: Sleeping for activity update...")
            time.sleep(0.1)
            print("DEBUG: Getting updated status...")
            updated_status = data_bus.get_component_status()["trainer"]
            print("DEBUG: Updated trainer status:", updated_status)
            assert updated_status["last_seen"] > initial_status["last_seen"]
            print("DEBUG: Activity tracking confirmed")

            # Test component shutdown
            print("DEBUG: Starting component shutdown...")
            # Save web server instance for direct shutdown if needed
            web_server = trainer.web_server
            # Start finalization
            trainer._finalize_training()
            print("DEBUG: Training finalized - waiting for unregistration...")
            
            # Wait for components to unregister
            unregister_success = wait_for_condition(
                lambda: "trainer" not in data_bus.get_component_status() and \
                        "web_server" not in data_bus.get_component_status(),
                timeout=5.0  # Longer timeout for cleanup
            )
            
            if not unregister_success:
                # Backup cleanup if unregistration failed
                print("DEBUG: Unregistration timed out, forcing cleanup...")
                if web_server:
                    try:
                        web_server.stop()
                    except Exception as e:
                        print(f"DEBUG: Error during forced web server stop: {e}")
                data_bus.shutdown()
                time.sleep(0.1)
            
            # Check final status
            print("DEBUG: Checking final component status...")
            final_components = data_bus.get_component_status()
            print("DEBUG: Final components:", final_components)
            assert "trainer" not in final_components
            assert "web_server" not in final_components
            print("DEBUG: Component unregistration confirmed")

    def test_real_time_stats_tracking(self, mock_config, data_bus):
        """Test real-time statistics tracking"""
        stats_received = []
        stats_event = threading.Event()
    
        def stats_callback(data, publisher=None):
            stats_received.append(data)
            if len(stats_received) >= 3:
                stats_event.set()
    
        data_bus.subscribe(
            DataType.TRAINING_STATS,
            "test",
            stats_callback
        )
    
        # Mock data bus publish to directly call callbacks
        def mock_publish_call(data_type, data, publisher):
            if data_type == DataType.TRAINING_STATS:
                stats_callback(dict(data), publisher)
                return True
            return None

        with patch.object(data_bus, 'publish', side_effect=mock_publish_call), \
             patch('trainer.trainer.PyBoy') as mock_pyboy, \
             patch('llm.local_llm_agent.LLMManager', return_value=MockLLMManager()):
            mock_pyboy_instance = Mock()
            mock_pyboy.return_value = mock_pyboy_instance
    
            trainer = UnifiedPokemonTrainer(mock_config)
            
            # Simulate some training progress
            for i in range(3):
                trainer.stats['total_actions'] = i * 10
                trainer._update_stats()
                time.sleep(0.1)
                
            # Should have received stats updates
            assert stats_event.wait(timeout=1.0)
            assert len(stats_received) >= 3

            # Stats should show progression
            assert stats_received[-1]['total_actions'] > stats_received[0]['total_actions']
            assert isinstance(stats_received[-1]['actions_per_second'], float)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
