"""Test screen capture system."""

import time
import pytest
import numpy as np
from queue import Empty

from .mock_screen_capture import MockScreenCapture, MockCaptureConfig

@pytest.fixture
def capture_config() -> MockCaptureConfig:
    """Create test capture config."""
    return MockCaptureConfig(
        enabled=True,
        fps=30,
        width=160,
        height=144,
        channels=3,
        queue_size=10
    )

@pytest.fixture
def screen_capture(capture_config):
    """Create test screen capture."""
    capture = MockScreenCapture(capture_config)
    capture.start()
    yield capture
    capture.stop()

@pytest.mark.monitoring
class TestScreenCapture:
    """Test screen capture functionality."""
    
    def test_capture_initialization(self, screen_capture):
        """Test capture initialization."""
        assert screen_capture._running
        assert screen_capture.config.width == 160
        assert screen_capture.config.height == 144
        
    def test_frame_generation(self, screen_capture):
        """Test frame generation."""
        frame = screen_capture.get_current_frame()
        assert frame is not None
        assert frame.shape == (144, 160, 3)
        assert frame.dtype == np.uint8
        
    def test_frame_queueing(self, screen_capture):
        """Test frame queue handling."""
        frame = screen_capture.get_current_frame()
        frame_data = {
            'image': frame,
            'timestamp': time.time(),
            'frame': 1
        }
        
        success = screen_capture.queue_frame(frame_data)
        assert success
        assert screen_capture.latest_screen == frame_data
        assert not screen_capture.screen_queue.empty()
        
    def test_queue_overflow(self, screen_capture):
        """Test queue overflow handling."""
        # Fill queue to max
        for i in range(screen_capture.config.queue_size + 5):
            frame = screen_capture.get_current_frame()
            frame_data = {
                'image': frame,
                'timestamp': time.time(),
                'frame': i
            }
            screen_capture.queue_frame(frame_data)
            
        # Queue should maintain size limit
        assert screen_capture.screen_queue.qsize() == screen_capture.config.queue_size
        assert screen_capture.latest_screen['frame'] == screen_capture.config.queue_size + 4
        
    def test_mock_frames(self, screen_capture):
        """Test pre-loaded mock frames."""
        mock_frames = [
            np.random.randint(0, 255, (144, 160, 3), dtype=np.uint8)
            for _ in range(3)
        ]
        screen_capture.load_mock_frames(mock_frames)
        
        # Should cycle through mock frames
        for i in range(5):
            frame = screen_capture.get_current_frame()
            assert np.array_equal(frame, mock_frames[i % 3])
            
        screen_capture.clear_mock_frames()
        frame = screen_capture.get_current_frame()
        assert not any(np.array_equal(frame, mock) for mock in mock_frames)
        
    def test_capture_disabled(self, capture_config):
        """Test disabled capture."""
        capture_config.enabled = False
        capture = MockScreenCapture(capture_config)
        assert not capture.start()
        assert capture.get_current_frame() is None
        
    def test_capture_timing(self, screen_capture):
        """Test capture timing tracking."""
        for _ in range(3):
            screen_capture.get_current_frame()
            time.sleep(0.01)
            
        assert len(screen_capture.capture_calls) == 3
        # Verify reasonable intervals
        intervals = np.diff(screen_capture.capture_calls)
        assert all(interval >= 0.01 for interval in intervals)
        
    def test_reset_state(self, screen_capture):
        """Test state reset."""
        # Generate some state
        frame = screen_capture.get_current_frame()
        screen_capture.queue_frame({'image': frame, 'frame': 1})
        screen_capture.load_mock_frames([frame])
        
        screen_capture.reset_state()
        assert len(screen_capture.capture_calls) == 0
        assert screen_capture.error_count == 0
        assert len(screen_capture.mock_frames) == 0
        assert screen_capture.latest_screen is None
        assert screen_capture.screen_queue.empty()
