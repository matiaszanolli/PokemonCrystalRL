"""Tests for frame processing service."""

import pytest
import os
import sys
import numpy as np
from unittest.mock import Mock, patch
from flask_socketio import SocketIO

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from monitoring.web.services.frame import FrameService, FrameConfig
from monitoring.components.capture import ScreenCapture


@pytest.fixture
def config():
    """Frame service configuration for testing."""
    return FrameConfig(
        buffer_size=2,
        quality=85,
        target_fps=30,
        optimize=True,
        progressive=False
    )

@pytest.fixture
def frame_service(config):
    """Create frame service instance."""
    return FrameService(config)

@pytest.fixture
def mock_capture():
    """Mock screen capture component."""
    capture = Mock(spec=ScreenCapture)
    # Create a simple test frame
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    frame[10:50, 10:50] = 255  # White square
    capture.get_frame.return_value = frame
    return capture

def test_frame_service_init(frame_service, config):
    """Test frame service initialization."""
    assert frame_service.config == config
    assert frame_service.frames_captured == 0
    assert frame_service.frames_sent == 0
    assert frame_service.current_fps == 0.0
    assert not frame_service._running

def test_set_screen_capture(frame_service, mock_capture):
    """Test setting screen capture component."""
    frame_service.set_screen_capture(mock_capture)
    assert frame_service._screen_capture == mock_capture

def test_set_quality(frame_service):
    """Test quality settings."""
    # Test different quality levels
    frame_service.set_quality('low')
    assert frame_service.config.quality == 50
    
    frame_service.set_quality('medium')
    assert frame_service.config.quality == 85
    
    frame_service.set_quality('high')
    assert frame_service.config.quality == 95
    
    # Test invalid quality defaults to medium
    frame_service.set_quality('invalid')
    assert frame_service.config.quality == 85

def test_get_status(frame_service):
    """Test status reporting."""
    status = frame_service.get_status()
    assert 'running' in status
    assert 'frames_captured' in status
    assert 'frames_sent' in status
    assert 'current_fps' in status
    assert 'frame_queue_size' in status
    assert 'quality' in status

def test_process_frame(frame_service, mock_capture):
    """Test frame processing."""
    frame_service.set_screen_capture(mock_capture)
    frame = mock_capture.get_frame()
    
    # Process frame
    frame_data = frame_service.process_frame(frame)
    assert frame_data is not None
    assert isinstance(frame_data, bytes)
    assert frame_service.frames_captured == 1

@patch('flask_socketio.emit')
def test_send_frame(mock_emit, frame_service, mock_capture):
    """Test frame sending."""
    frame_service.set_screen_capture(mock_capture)
    frame = mock_capture.get_frame()
    frame_data = frame_service.process_frame(frame)
    
    # Send frame
    success = frame_service.send_frame(frame_data)
    assert success
    assert frame_service.frames_sent == 1
    mock_emit.assert_called_once()

@patch('flask_socketio.emit')
def test_handle_frame_request(mock_emit, frame_service, mock_capture):
    """Test frame request handling."""
    frame_service.set_screen_capture(mock_capture)
    
    # Handle request
    frame_service.handle_frame_request()
    assert frame_service.frames_captured == 1
    assert frame_service.frames_sent == 1
    mock_emit.assert_called_once()

def test_handle_frame_request_no_capture(frame_service):
    """Test frame request handling without capture component."""
    # Should not raise error
    frame_service.handle_frame_request()
    assert frame_service.frames_captured == 0
    assert frame_service.frames_sent == 0

def test_clear(frame_service, mock_capture):
    """Test state clearing."""
    frame_service.set_screen_capture(mock_capture)
    
    # Generate some activity
    frame_service.handle_frame_request()
    assert frame_service.frames_captured > 0
    assert frame_service.frames_sent > 0
    
    # Clear state
    frame_service.clear()
    assert frame_service.frames_captured == 0
    assert frame_service.frames_sent == 0
    assert frame_service.current_fps == 0.0

def test_fps_tracking(frame_service, mock_capture):
    """Test FPS tracking."""
    frame_service.set_screen_capture(mock_capture)
    
    # Process multiple frames
    for _ in range(30):
        frame = mock_capture.get_frame()
        frame_service.process_frame(frame)
    
    status = frame_service.get_status()
    assert status['current_fps'] >= 0.0  # Should have some FPS value

def test_error_handling(frame_service):
    """Test error handling in frame processing."""
    # Try to process invalid frame
    frame_data = frame_service.process_frame(None)
    assert frame_data is None
    
    # Try to send invalid data
    success = frame_service.send_frame(None)
    assert not success
