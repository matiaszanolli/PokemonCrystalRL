"""Tests for the Pokemon Crystal RL UnifiedMonitor."""

import pytest
import asyncio
import json
import time
import threading
from unittest.mock import Mock, patch
from pathlib import Path
import tempfile
import os
import numpy as np
import websockets

from monitoring import UnifiedMonitor, MonitorConfig
from monitoring.error_handler import ErrorSeverity, RecoveryStrategy

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
def mock_monitor_config(temp_dir):
    """Create a mock monitor configuration."""
    return MonitorConfig(
        web_port=8080,
        db_path=str(temp_dir / "monitoring.db"),
        data_dir=str(temp_dir / "data"),
        static_dir=str(temp_dir / "static"),
        update_interval=0.1,
        snapshot_interval=1.0,
        max_events=10,
        max_snapshots=5,
        debug=True
    )


class TestUnifiedMonitor:
    """Test UnifiedMonitor functionality."""
    
    def test_monitor_initialization(self, mock_monitor_config, mock_data_bus):
        """Test monitor initialization."""
        with patch("monitoring.unified_monitor.get_data_bus",
                  return_value=mock_data_bus):
            monitor = UnifiedMonitor(config=mock_monitor_config)
            
            assert monitor.config == mock_monitor_config
            assert not monitor.is_monitoring
            assert monitor.error_handler is not None
    
    def test_start_stop_monitoring(self, mock_monitor_config, mock_data_bus):
        """Test start and stop monitoring."""
        with patch("monitoring.unified_monitor.get_data_bus",
                  return_value=mock_data_bus):
            monitor = UnifiedMonitor(config=mock_monitor_config)
            
            # Start monitor
            monitor.start_monitoring()
            assert monitor.is_monitoring
            
            # Stop monitor
            monitor.stop_monitoring()
            assert not monitor.is_monitoring
    
    def test_screenshot_update(self, mock_monitor_config, mock_data_bus):
        """Test screenshot update functionality."""
        with patch("monitoring.unified_monitor.get_data_bus",
                  return_value=mock_data_bus):
            monitor = UnifiedMonitor(config=mock_monitor_config)
            
            # Create test screenshot
            screenshot = np.random.randint(0, 255, (144, 160, 3), dtype=np.uint8)
            
            # Test updating screenshot
            monitor.update_screenshot(screenshot)
            
            # Verify screenshot queue has data
            assert not monitor.screen_queue.empty()
            screenshot_data = monitor.screen_queue.get()
            assert 'image' in screenshot_data
            assert 'timestamp' in screenshot_data
            assert 'dimensions' in screenshot_data
    
    def test_text_update(self, mock_monitor_config, mock_data_bus):
        """Test text recognition update."""
        with patch("monitoring.unified_monitor.get_data_bus",
                  return_value=mock_data_bus):
            monitor = UnifiedMonitor(config=mock_monitor_config)
            
            test_text = "TEST TEXT"
            test_location = "menu"
            
            monitor.update_text(test_text, test_location)
            
            assert len(monitor.recent_text) == 1
            text_data = monitor.recent_text[0]
            assert text_data['text'] == test_text
            assert text_data['location'] == test_location
            assert text_data['timestamp'] is not None
    
    def test_action_update(self, mock_monitor_config, mock_data_bus):
        """Test action history update."""
        with patch("monitoring.unified_monitor.get_data_bus",
                  return_value=mock_data_bus):
            monitor = UnifiedMonitor(config=mock_monitor_config)
            
            test_action = "MOVE_RIGHT"
            test_details = "Moving to next tile"
            
            monitor.update_action(test_action, test_details)
            
            assert len(monitor.recent_actions) == 1
            action_data = monitor.recent_actions[0]
            assert action_data['action'] == test_action
            assert action_data['details'] == test_details
            assert action_data['timestamp'] is not None
    
    def test_decision_update(self, mock_monitor_config, mock_data_bus):
        """Test decision history update."""
        with patch("monitoring.unified_monitor.get_data_bus",
                  return_value=mock_data_bus):
            monitor = UnifiedMonitor(config=mock_monitor_config)
            
            test_decision = {
                'action': 'MOVE_RIGHT',
                'confidence': 0.95,
                'reasoning': 'Path to objective'
            }
            
            monitor.update_decision(test_decision)
            
            assert len(monitor.recent_decisions) == 1
            decision_data = monitor.recent_decisions[0]
            assert decision_data['action'] == test_decision['action']
            assert decision_data['confidence'] == test_decision['confidence']
            assert decision_data['reasoning'] == test_decision['reasoning']
            assert 'timestamp' in decision_data
    
    def test_memory_cleanup(self, mock_monitor_config, mock_data_bus):
        """Test memory management cleanup."""
        with patch("monitoring.unified_monitor.get_data_bus",
                  return_value=mock_data_bus):
            monitor = UnifiedMonitor(config=mock_monitor_config)
            
            # Add lots of text entries
            for i in range(1100):
                monitor.update_text(f"Text {i}")
            
            # Force cleanup
            monitor.cleanup_interval = 0  # Set to 0 to force immediate cleanup
            monitor.cleanup()
            
            # Verify cleanup worked
            assert len(monitor.text_frequency) <= 1000
    
    def test_system_stats(self, mock_monitor_config, mock_data_bus):
        """Test system statistics collection."""
        with patch("monitoring.unified_monitor.get_data_bus",
                  return_value=mock_data_bus):
            monitor = UnifiedMonitor(config=mock_monitor_config)
            
            stats = monitor._get_system_stats()
            if stats:  # If psutil is available
                assert 'cpu_percent' in stats
                assert 'memory_percent' in stats
                assert 'disk_usage' in stats


class TestErrorHandling:
    """Test error handling in monitoring system."""
    
    def test_error_handling(self, mock_monitor_config, mock_data_bus):
        """Test error handling."""
        with patch("monitoring.unified_monitor.get_data_bus",
                  return_value=mock_data_bus):
            monitor = UnifiedMonitor(config=mock_monitor_config)
            
            # Simulate error handling
            error = Exception("Test error")
            monitor.error_handler.handle_error(
                error,
                severity=ErrorSeverity.ERROR,
                strategy=RecoveryStrategy.RETRY,
                component="monitor"
            )
            
            # Error should be handled
            assert monitor.error_handler is not None
    
    def test_error_recovery(self, mock_monitor_config, mock_data_bus):
        """Test error recovery mechanisms."""
        with patch("monitoring.unified_monitor.get_data_bus",
                  return_value=mock_data_bus):
            monitor = UnifiedMonitor(config=mock_monitor_config)
            monitor.start_monitoring()
            
            # Simulate recoverable error
            error = Exception("Recoverable error")
            monitor.error_handler.handle_error(
                error,
                severity=ErrorSeverity.WARNING,
                strategy=RecoveryStrategy.RETRY,
                component="test"
            )
            
            # Monitor should still be running
            assert monitor.is_monitoring
            
            # Simulate critical error
            error = Exception("Critical error")
            monitor.error_handler.handle_error(
                error,
                severity=ErrorSeverity.CRITICAL,
                strategy=RecoveryStrategy.ABORT,
                component="test"
            )
            
            # Monitor should still be running (UnifiedMonitor handles critical errors differently)
            assert monitor.is_monitoring
            
            monitor.stop_monitoring()


class TestTrainingControl:
    """Test training control functionality."""
    
    @pytest.fixture
    def monitor(self, mock_monitor_config, mock_data_bus):
        """Create a configured monitor for testing."""
        with patch("monitoring.unified_monitor.get_data_bus",
                  return_value=mock_data_bus):
            return UnifiedMonitor(config=mock_monitor_config)
    
    def test_start_training(self, monitor):
        """Test starting a new training run."""
        config = {'learning_rate': 0.001, 'batch_size': 32}
        run_id = monitor.start_training(config)
        
        assert run_id is not None if monitor.db else True
        assert monitor.training_state == monitor.training_state.RUNNING
        assert monitor.is_monitoring
    
    def test_stop_training(self, monitor):
        """Test stopping a training run."""
        run_id = monitor.start_training()
        monitor.stop_training(final_reward=100.0)
        
        assert monitor.training_state == monitor.training_state.COMPLETED
        assert not monitor.is_monitoring
    
    def test_pause_resume_training(self, monitor):
        """Test pausing and resuming a training run."""
        monitor.start_training()
        
        # Test pause
        monitor.pause_training()
        assert monitor.training_state == monitor.training_state.PAUSED
        
        # Test resume
        monitor.resume_training()
        assert monitor.training_state == monitor.training_state.RUNNING
    
    def test_cannot_start_while_running(self, monitor):
        """Test that starting training fails when already running."""
        monitor.start_training()
        
        with pytest.raises(RuntimeError, match="Training already in progress"):
            monitor.start_training()
    
    def test_update_training_metrics(self, monitor):
        """Test updating training metrics."""
        run_id = monitor.start_training()
        metrics = {
            'loss': 0.5,
            'reward': 10.0,
            'epsilon': 0.1
        }
        
        monitor.update_metrics(metrics)
        
        # Check metrics were stored
        for name, value in metrics.items():
            assert name in monitor.performance_metrics
            assert value in monitor.performance_metrics[name]
    
    def test_update_training_episode(self, monitor):
        """Test updating episode information."""
        run_id = monitor.start_training()
        monitor.update_episode(
            episode=1,
            total_reward=100.0,
            steps=500,
            success=True,
            metadata={'level': 'route_1'}
        )
        
        # Check episode was recorded
        assert len(monitor.episode_data) == 1
        episode = monitor.episode_data[0]
        assert episode['episode'] == 1
        assert episode['total_reward'] == 100.0
        assert episode['steps'] == 500
        assert episode['success']
        assert episode['metadata']['level'] == 'route_1'


if __name__ == "__main__":
    pytest.main(["-v", __file__])
