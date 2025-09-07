"""Tests for web monitoring services.

This module contains comprehensive tests for frame and metrics services:
- Frame processing and compression
- Metrics collection and history tracking
- Performance measurements
- Error handling
"""

import time
import pytest
import numpy as np
import cv2
from unittest.mock import Mock, patch, MagicMock
from queue import Queue
from collections import deque

# Mock constant for testing
cv2.IMWRITE_JPEG_QUALITY = 1
cv2.IMWRITE_JPEG_OPTIMIZE = 2
cv2.IMWRITE_JPEG_PROGRESSIVE = 3
from queue import Queue
from collections import deque

from monitoring.web.services.frame import FrameService, FrameConfig
from monitoring.web.services.metrics import (
    MetricsService, MetricsConfig, MetricHistory
)
from monitoring.components.capture import ScreenCapture
from monitoring.components.metrics import MetricsCollector


class TestFrameService:
    """Tests for frame processing service."""
    
    @pytest.fixture
    def frame_service(self):
        """Create frame service instance."""
        config = FrameConfig(
            buffer_size=2,
            quality=85,
            target_fps=30,
            optimize=True
        )
        return FrameService(config)
    
    @pytest.fixture
    def test_frame(self):
        """Create test frame data."""
        # Create simple 160x144 RGB test frame
        return np.zeros((144, 160, 3), dtype=np.uint8)
    
    def test_frame_service_initialization(self, frame_service):
        """Test frame service creation and configuration."""
        assert frame_service.config.buffer_size == 2
        assert frame_service.config.quality == 85
        assert frame_service.config.target_fps == 30
        assert frame_service.config.optimize is True
        assert frame_service.config.progressive is False
        
        assert isinstance(frame_service._frame_queue, Queue)
        assert frame_service._frame_queue.maxsize == 2
        
        assert frame_service.frames_captured == 0
        assert frame_service.frames_sent == 0
        assert frame_service.current_fps == 0.0
    
    def test_quality_settings(self, frame_service):
        """Test frame quality adjustment."""
        frame_service.set_quality('low')
        assert frame_service.config.quality == 50
        
        frame_service.set_quality('medium')
        assert frame_service.config.quality == 85
        
        frame_service.set_quality('high')
        assert frame_service.config.quality == 95
        
        # Invalid quality falls back to medium
        frame_service.set_quality('invalid')
        assert frame_service.config.quality == 85
    
    def test_frame_processing(self, frame_service, test_frame):
        """Test frame processing pipeline."""
        with patch('cv2.imencode') as mock_imencode:
            # Mock successful JPEG encoding
            mock_imencode.return_value = (True, np.array([1, 2, 3]))
            
            # Process frame
            frame_data = frame_service.process_frame(test_frame)
            
            # Verify encoding was called with correct parameters
            mock_imencode.assert_called_once()
            args, kwargs = mock_imencode.call_args
            
            # Check encoding parameters
            assert args[0] == '.jpg'
            assert isinstance(args[1], np.ndarray)
            params = args[2]
            assert params[0] == cv2.IMWRITE_JPEG_QUALITY
            assert params[1] == 85  # quality
            assert params[2] == cv2.IMWRITE_JPEG_OPTIMIZE
            assert params[3] == 1   # optimize=True
            
            # Check frame was processed
            assert frame_data is not None
            assert frame_service.frames_captured == 1
            assert frame_service.current_fps > 0
    
    def test_frame_processing_error(self, frame_service, test_frame):
        """Test frame processing error handling."""
        with patch('cv2.imencode') as mock_imencode:
            # Mock encoding failure
            mock_imencode.return_value = (False, None)
            
            # Process should return None on failure
            frame_data = frame_service.process_frame(test_frame)
            assert frame_data is None
    
    def test_frame_sending(self, frame_service):
        """Test frame sending with WebSocket."""
        test_data = b'test_frame_data'
        
        with patch('flask_socketio.emit') as mock_emit:
            # Send frame
            success = frame_service.send_frame(test_data)
            
            # Verify emission
            assert success is True
            mock_emit.assert_called_once_with(
                'frame',
                test_data,
                binary=True
            )
            assert frame_service.frames_sent == 1
    
    def test_frame_request_handling(self, frame_service):
        """Test frame request handling."""
        mock_capture = Mock(spec=ScreenCapture)
        test_frame = np.zeros((144, 160, 3), dtype=np.uint8)
        mock_capture.get_frame.return_value = test_frame
        
        frame_service.set_screen_capture(mock_capture)
        
        with patch.object(frame_service, 'process_frame') as mock_process:
            with patch.object(frame_service, 'send_frame') as mock_send:
                mock_process.return_value = b'processed_frame'
                
                # Handle request
                frame_service.handle_frame_request()
                
                # Verify processing pipeline
                mock_capture.get_frame.assert_called_once_with("raw")
                mock_process.assert_called_once()
                mock_send.assert_called_once_with(b'processed_frame')
    
    def test_performance_tracking(self, frame_service, test_frame):
        """Test FPS and performance tracking."""
        # Process multiple frames
        for _ in range(30):
            frame_service.process_frame(test_frame)
            time.sleep(0.01)  # Small delay
        
        status = frame_service.get_status()
        assert status['frames_captured'] == 30
        assert 20 <= status['current_fps'] <= 120  # Reasonable FPS range


class TestMetricsService:
    """Tests for metrics service."""
    
    @pytest.fixture
    def metrics_service(self):
        """Create metrics service instance."""
        config = MetricsConfig(
            history_size=1000,
            update_interval=1.0,
            retention_hours=24.0
        )
        return MetricsService(config)
    
    @pytest.fixture
    def mock_collector(self):
        """Create mock metrics collector."""
        collector = Mock(spec=MetricsCollector)
        collector.get_metrics.return_value = {
            'cpu_percent': 50.0,
            'memory_usage_mb': 100.0,
            'network_bytes_sec': 1000.0
        }
        return collector
    
    def test_metric_history(self):
        """Test metric history tracking."""
        history = MetricHistory()
        
        # Add values with timestamps
        current_time = time.time()
        history.add(1.0, current_time - 2)
        history.add(2.0, current_time - 1)
        history.add(3.0, current_time)
        
        # Get recent values
        values = history.get_since(current_time - 1.5)
        assert len(values) == 2
        assert values[0]['value'] == 2.0
        assert values[1]['value'] == 3.0
        
        # Clear history
        history.clear()
        assert len(history.values) == 0
        assert len(history.timestamps) == 0
    
    def test_metrics_initialization(self, metrics_service):
        """Test metrics service initialization."""
        assert isinstance(metrics_service.config, MetricsConfig)
        assert metrics_service.config.history_size == 1000
        assert metrics_service.config.update_interval == 1.0
        
        # Check history initialization
        assert 'reward' in metrics_service._history
        assert 'steps' in metrics_service._history
        assert 'cpu_percent' in metrics_service._history
        assert isinstance(metrics_service._history['reward'], MetricHistory)
    
    def test_training_metrics_update(self, metrics_service):
        """Test training metrics updates."""
        training_data = {
            'episode': 1,
            'total_reward': 100.0,
            'total_steps': 500,
            'experience': 1000,
            'exploration': 0.5
        }
        
        metrics_service.update_training_metrics(training_data)
        
        # Check current metrics
        metrics = metrics_service.get_metrics()
        assert metrics['total_reward'] == 100.0
        assert metrics['total_steps'] == 500
        
        # Check histories
        reward_history = metrics_service._history['reward'].values
        assert len(reward_history) == 1
        assert reward_history[0] == 100.0
        
        steps_history = metrics_service._history['steps'].values
        assert len(steps_history) == 1
        assert steps_history[0] == 500
    
    def test_game_metrics_update(self, metrics_service):
        """Test game metrics updates."""
        game_data = {
            'party_count': 3,
            'badges_total': 2,
            'money': 1000
        }
        
        metrics_service.update_game_metrics(game_data)
        
        # Check current metrics
        metrics = metrics_service.get_metrics()
        assert metrics['party_count'] == 3
        assert metrics['badges_total'] == 2
        assert metrics['money'] == 1000
        
        # Check histories
        pokemon_history = metrics_service._history['pokemon_count'].values
        assert len(pokemon_history) == 1
        assert pokemon_history[0] == 3
        
        badges_history = metrics_service._history['badge_count'].values
        assert len(badges_history) == 1
        assert badges_history[0] == 2
    
    def test_resource_metrics_update(self, metrics_service):
        """Test resource metrics updates."""
        resource_data = {
            'cpu_percent': 50.0,
            'memory_usage_mb': 100.0,
            'network_bytes_sec': 1000.0
        }
        
        metrics_service.update_resource_metrics(resource_data)
        
        # Check current metrics
        metrics = metrics_service.get_metrics()
        assert metrics['cpu_percent'] == 50.0
        assert metrics['memory_usage_mb'] == 100.0
        assert metrics['network_bytes_sec'] == 1000.0
        
        # Check histories
        cpu_history = metrics_service._history['cpu_percent'].values
        assert len(cpu_history) == 1
        assert cpu_history[0] == 50.0
        
        memory_history = metrics_service._history['memory_usage'].values
        assert len(memory_history) == 1
        assert memory_history[0] == 100.0
    
    def test_metrics_filtering(self, metrics_service):
        """Test metrics filtering by name."""
        # Add various metrics
        metrics_service.update_training_metrics({
            'total_reward': 100.0,
            'total_steps': 500
        })
        metrics_service.update_game_metrics({
            'party_count': 3,
            'money': 1000
        })
        
        # Get specific metrics
        filtered = metrics_service.get_metrics(
            names=['total_reward', 'party_count']
        )
        assert len(filtered) == 2
        assert filtered['total_reward'] == 100.0
        assert filtered['party_count'] == 3
        
        # Get with history
        since_time = time.time() - 3600
        with_history = metrics_service.get_metrics(
            names=['total_reward'],
            since=since_time
        )
        assert 'total_reward' in with_history
        assert 'total_reward_history' in with_history
    
    def test_chart_data(self, metrics_service):
        """Test chart data formatting."""
        # Add metrics data
        metrics_service.update_training_metrics({
            'total_reward': 100.0,
            'experience': 1000,
            'exploration': 0.5
        })
        
        # Get chart data
        chart_data = metrics_service.get_chart_data()
        
        assert 'reward_history' in chart_data
        assert len(chart_data['reward_history']) == 1
        assert chart_data['reward_history'][0] == 100.0
        
        assert 'progress' in chart_data
        assert len(chart_data['progress']) == 1
        assert chart_data['progress'][0]['experience'] == 1000
        assert chart_data['progress'][0]['exploration'] == 0.5
    
    def test_status_tracking(self, metrics_service):
        """Test metrics service status tracking."""
        metrics_service.update_training_metrics({
            'total_reward': 100.0,
            'total_steps': 500
        })
        
        status = metrics_service.get_status()
        assert status['running'] is False
        assert status['active_metrics'] == 2
        assert status['total_recorded'] == 2
        assert isinstance(status['last_update'], float)
    
    def test_clear_metrics(self, metrics_service):
        """Test metrics clearing."""
        # Add some metrics
        metrics_service.update_training_metrics({
            'total_reward': 100.0,
            'total_steps': 500
        })
        metrics_service.update_game_metrics({
            'party_count': 3,
            'money': 1000
        })
        
        # Clear all metrics
        metrics_service.clear()
        
        # Verify cleared state
        assert len(metrics_service._metrics) == 0
        assert len(metrics_service._active_metrics) == 0
        assert metrics_service._total_recorded == 0
        
        # Check histories cleared
        for history in metrics_service._history.values():
            assert len(history.values) == 0
            assert len(history.timestamps) == 0
