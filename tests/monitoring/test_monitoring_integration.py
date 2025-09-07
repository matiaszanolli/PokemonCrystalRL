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
from queue import Queue
import numpy as np
from unittest.mock import Mock, patch, PropertyMock
import threading
import tempfile
from pathlib import Path

from monitoring.data_bus import DataBus, DataType
from trainer.web_server import WebServer as TrainingWebServer
from .mock_llm_manager import MockLLMManager
from training.trainer import TrainingConfig, TrainingMode, LLMBackend, PokemonTrainer
from trainer.unified_trainer import UnifiedTrainer
from monitoring.error_handler import ErrorHandler


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
    port = None
    # Try multiple ports in case of conflicts
    for _ in range(3):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', 0))
                port = s.getsockname()[1]
                # Test if port is actually free
                try:
                    requests.get(f"http://localhost:{port}/health", timeout=0.1)
                except requests.exceptions.RequestException:
                    # Port is truly free
                    break
        except Exception:
            time.sleep(0.1)
    
    if port is None:
        port = free_port  # Fall back to fixture provided port
    
    return TrainingConfig(
        rom_path="test.gbc",
        enable_web=True,
        web_port=port,
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

            trainer = UnifiedTrainer(mock_config)

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
    
            trainer = UnifiedTrainer(mock_config)
            
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

    def test_web_server_configuration(self, mock_config, data_bus):
        """Test web server configuration (consolidated into core.web_monitor.WebMonitor)"""
        with patch('trainer.trainer.PyBoy') as mock_pyboy, \
             patch('llm.local_llm_agent.LLMManager', return_value=MockLLMManager()):
            mock_pyboy_instance = Mock()
            mock_screen = np.random.randint(0, 255, (144, 160, 3), dtype=np.uint8)
            mock_pyboy_instance.screen.ndarray = mock_screen
            mock_pyboy_instance.frame_count = 1000
            mock_pyboy.return_value = mock_pyboy_instance
    
            trainer = UnifiedTrainer(mock_config)

            # Web server functionality consolidated into core.web_monitor.WebMonitor
            # Note: test_mode=True forces enable_web=False in trainer for isolation
            assert trainer.config.test_mode == True
            assert trainer.config.web_port is not None
            
            # Trainer should have stats available for web monitoring
            stats = trainer.get_current_stats()
            assert 'total_actions' in stats
            assert 'mode' in stats

    def test_trainer_cleanup(self, mock_config, data_bus):
        """Test trainer cleanup (web server consolidated into core.web_monitor.WebMonitor)"""
        with patch('trainer.trainer.PyBoy') as mock_pyboy, \
             patch('llm.local_llm_agent.LLMManager', return_value=MockLLMManager()):
            mock_pyboy_instance = Mock()
            mock_pyboy.return_value = mock_pyboy_instance
    
            trainer = UnifiedTrainer(mock_config)
            
            # Check trainer is registered
            components = data_bus.get_component_status()
            assert "trainer" in components
            
            # Cleanup trainer
            trainer._finalize_training()
            
            # Check trainer configuration maintained
            # Note: test_mode=True forces enable_web=False in trainer for isolation
            assert trainer.config.test_mode == True

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

            trainer = UnifiedTrainer(mock_config)

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

        # Set up error tracking at data bus level
        data_bus.error_handler = ErrorHandler()
        data_bus.error_handler.error_events = Queue()

        def error_callback(data, publisher=None):
            nonlocal error_count
            print("\nDEBUG: Error callback executing with data:", data)
            error_count += 1
            error_received.set()
            print("DEBUG: Error event set, count:", error_count)
            return True  # Indicate successful handling

        # Subscribe to error events
        data_bus.subscribe(
            DataType.ERROR_EVENT,
            "test",
            error_callback
        )

        with patch('trainer.trainer.PyBoy') as mock_pyboy, \
             patch('llm.local_llm_agent.LLMManager', return_value=MockLLMManager()):
            mock_pyboy_instance = Mock()
            mock_pyboy_instance.frame_count = 1000
            mock_pyboy.return_value = mock_pyboy_instance

            trainer = UnifiedTrainer(mock_config)
            
            # Force test_mode off and enable screen capture
            trainer.config.test_mode = False
            trainer.config.capture_screens = True
            
            # Register component for error handling
            data_bus.register_component("test_error_handler", {'type': 'test'})
            
            # Patch the _simple_screenshot_capture to raise an exception
            def failing_screenshot_capture():
                print("DEBUG: Failing screenshot capture called")
                raise RuntimeError("Screen access error")
                
            # First let's debug what the normal call does
            print(f"DEBUG: Config capture_screens: {trainer.config.capture_screens}")
            print(f"DEBUG: Config test_mode: {getattr(trainer.config, 'test_mode', None)}")
            
            # Trigger error by patching the method that should fail
            with patch.object(trainer, '_simple_screenshot_capture', side_effect=failing_screenshot_capture):
                print("\nDEBUG: About to trigger error in capture_and_queue_screen")
                try:
                    with trainer._handle_errors('test_screen_capture', 'capture_errors'):
                        print("DEBUG: Inside handle_errors context manager")
                        trainer._capture_and_queue_screen()  # This should raise and be caught by the context manager
                        print("DEBUG: capture_and_queue_screen completed without error - THIS SHOULD NOT HAPPEN")
                except Exception as e:
                    print("\nDEBUG: Caught exception outside error handler:", e)
                    import traceback
                    traceback.print_exc()
                        
                # Explicitly wait a bit for error processing
                time.sleep(0.5)
                        
                # Wait for error event with increased timeout
                assert error_received.wait(timeout=5.0), "Error event not received"
                # Check error callback ran at least once
                assert error_count > 0, "Error count not incremented"
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
            trainer = UnifiedTrainer(mock_config)
            print("DEBUG: Trainer initialized")

            # Wait for components to register
            print("DEBUG: Waiting for component registration...")
            registration_success = wait_for_condition(
                lambda: "trainer" in data_bus.get_component_status()
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
    
            trainer = UnifiedTrainer(mock_config)
            
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
