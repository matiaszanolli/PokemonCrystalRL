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
from unittest.mock import Mock, patch
import threading
import tempfile
from pathlib import Path

from monitoring.data_bus import DataBus, DataType, init_data_bus
from monitoring.web_server import TrainingWebServer
from .mock_llm_manager import MockLLMManager
from trainer import (
    PokemonTrainer,
    TrainingConfig,
    TrainingMode,
    LLMBackend
)


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
        debug_mode=True
    )


@pytest.fixture
def data_bus():
    """Create and initialize a data bus for testing"""
    data_bus = init_data_bus()
    yield data_bus
    data_bus.shutdown()


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

        with patch('trainer.trainer.PyBoy') as mock_pyboy, \
             patch('llm.local_llm_agent.LLMManager', return_value=MockLLMManager()):
            mock_pyboy_instance = Mock()
            mock_screen = np.random.randint(0, 255, (144, 160, 3), dtype=np.uint8)
            mock_pyboy_instance.screen_image = Mock(return_value=mock_screen)
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

        with patch('trainer.trainer.PyBoy') as mock_pyboy, \
             patch('llm.local_llm_agent.LLMManager', return_value=MockLLMManager()):
            mock_pyboy_instance = Mock()
            mock_screen = np.random.randint(0, 255, (144, 160, 3), dtype=np.uint8)
            mock_pyboy_instance.screen_image = Mock(return_value=mock_screen)
            mock_pyboy_instance.frame_count = 1000
            mock_pyboy.return_value = mock_pyboy_instance

            trainer = UnifiedPokemonTrainer(mock_config)

            # Check server started
            assert trainer.web_server is not None

            # Test status endpoint
            response = requests.get(f"http://localhost:{mock_config.web_port}/api/status")
            assert response.status_code == 200
            data = response.json()
            assert "status" in data
            assert "components" in data
            assert "trainer" in data["components"]

            # Test stats endpoint
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

        with patch('trainer.trainer.PyBoy') as mock_pyboy, \
             patch('llm.local_llm_agent.LLMManager', return_value=MockLLMManager()):
            mock_pyboy_instance = Mock()
            mock_pyboy.return_value = mock_pyboy_instance

            trainer = UnifiedPokemonTrainer(mock_config)
            
            # Test server started
            response = requests.get(f"http://localhost:{mock_config.web_port}/api/status")
            assert response.status_code == 200

            # Shutdown server
            trainer.web_server.shutdown()
            time.sleep(0.1)  # Give server time to stop

            # Should not be able to connect after shutdown
            with pytest.raises(requests.exceptions.ConnectionError):
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

        def error_callback(data, publisher):
            nonlocal error_count
            error_count += 1
            error_received.set()

        data_bus.subscribe(
            DataType.ERROR,
            "test",
            error_callback
        )

        with patch('trainer.trainer.PyBoy') as mock_pyboy, \
             patch('llm.local_llm_agent.LLMManager', return_value=MockLLMManager()):
            mock_pyboy_instance = Mock()
            mock_screen = np.random.randint(0, 255, (144, 160, 3), dtype=np.uint8)
            mock_pyboy_instance.screen_image = Mock(return_value=mock_screen)
            mock_pyboy.return_value = mock_pyboy_instance

            trainer = UnifiedPokemonTrainer(mock_config)

            # Trigger screen capture error
            mock_pyboy_instance.screen_image.side_effect = Exception("Screen error")
            trainer._capture_and_queue_screen()

            # Error should be published
            assert error_received.wait(timeout=1.0)
            assert error_count > 0
            assert trainer.error_count['capture_errors'] > 0

            # System should still be functional
            mock_pyboy_instance.screen_image = Mock(return_value=mock_screen)
            trainer._capture_and_queue_screen()
            assert trainer.screen_queue.qsize() > 0

    def test_component_lifecycle(self, mock_config, data_bus):
        """Test component lifecycle management"""
        with patch('trainer.trainer.PyBoy') as mock_pyboy, \
             patch('llm.local_llm_agent.LLMManager', return_value=MockLLMManager()):
            mock_pyboy_instance = Mock()
            mock_pyboy.return_value = mock_pyboy_instance

            # Initialize components
            trainer = UnifiedPokemonTrainer(mock_config)

            # Check initial registrations
            components = data_bus.get_component_status()
            assert "trainer" in components
            assert "web_server" in components

            # Check activity tracking
            initial_status = components["trainer"]
            time.sleep(0.1)
            updated_status = data_bus.get_component_status()["trainer"]
            assert updated_status["last_seen"] > initial_status["last_seen"]

            # Test component shutdown
            trainer._finalize_training()

            # Components should be unregistered
            final_components = data_bus.get_component_status()
            assert "trainer" not in final_components
            assert "web_server" not in final_components

    def test_real_time_stats_tracking(self, mock_config, data_bus):
        """Test real-time statistics tracking"""
        stats_received = []
        stats_event = threading.Event()

        def stats_callback(data, publisher):
            stats_received.append(data)
            if len(stats_received) >= 3:
                stats_event.set()

        data_bus.subscribe(
            DataType.TRAINING_STATS,
            "test",
            stats_callback
        )

        with patch('trainer.trainer.PyBoy') as mock_pyboy, \
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
