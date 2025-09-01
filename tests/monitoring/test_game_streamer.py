#!/usr/bin/env python3
"""
test_game_streamer.py - Comprehensive tests for GameStreamComponent

Tests the game streaming system including:
- PyBoy screen capture integration
- Frame buffering and compression
- Multiple output formats (PNG, JPEG, base64)
- Memory leak prevention and cleanup
- Performance monitoring and health checks
- Error handling and recovery
- Thread safety
"""

import pytest
import time
import threading
import queue
import numpy as np
import base64
from unittest.mock import Mock, patch, MagicMock, PropertyMock
from PIL import Image
import io

from monitoring.game_streamer import GameStreamComponent
from monitoring.data_bus import DataType


@pytest.mark.streaming
@pytest.mark.unit
class TestGameStreamComponentInit:
    """Test GameStreamComponent initialization and configuration"""
    
    def test_default_initialization(self):
        """Test default initialization parameters"""
        streamer = GameStreamComponent()
        
        assert streamer.buffer_size == 10
        assert streamer.compression_quality == 85
        assert streamer.max_frame_rate == 10.0
        assert streamer.enable_data_bus == True
        assert streamer._capture_active == False
        assert streamer._frames_captured == 0
        assert streamer._frames_served == 0
        assert streamer._capture_errors == 0
        assert streamer._consecutive_errors == 0
        assert streamer._pyboy is None
        
        streamer.shutdown()
    
    def test_custom_initialization(self):
        """Test initialization with custom parameters"""
        streamer = GameStreamComponent(
            buffer_size=20,
            compression_quality=75,
            max_frame_rate=15.0,
            enable_data_bus=False
        )
        
        assert streamer.buffer_size == 20
        assert streamer.compression_quality == 75
        assert streamer.max_frame_rate == 15.0
        assert streamer.enable_data_bus == False
        assert streamer.data_bus is None
        
        streamer.shutdown()
    
    @patch('monitoring.game_streamer.get_data_bus')
    def test_data_bus_integration(self, mock_get_data_bus):
        """Test data bus registration and integration"""
        mock_bus = Mock()
        mock_get_data_bus.return_value = mock_bus
        
        streamer = GameStreamComponent(enable_data_bus=True)
        
        assert streamer.data_bus == mock_bus
        mock_bus.register_component.assert_called_once_with(
            "game_streamer", 
            {
                "type": "streaming",
                "buffer_size": 10,
                "max_fps": 10.0
            }
        )
        
        streamer.shutdown()


@pytest.mark.streaming
@pytest.mark.unit
class TestPyBoyIntegration:
    """Test PyBoy integration and screen capture"""
    
    @pytest.fixture
    def streamer(self):
        """Create streamer instance for testing"""
        streamer = GameStreamComponent(enable_data_bus=False)
        yield streamer
        streamer.shutdown()
    
    @pytest.fixture
    def mock_pyboy(self):
        """Create mock PyBoy instance"""
        mock_pyboy = Mock()
        mock_screen = Mock()
        test_array = np.random.randint(0, 255, (144, 160, 3), dtype=np.uint8)
        mock_screen.ndarray = test_array
        mock_pyboy.screen = mock_screen
        return mock_pyboy
    
    def test_set_pyboy_instance(self, streamer, mock_pyboy):
        """Test setting PyBoy instance"""
        streamer.set_pyboy_instance(mock_pyboy)
        
        assert streamer._pyboy == mock_pyboy
    
    def test_start_streaming_without_pyboy(self, streamer):
        """Test that streaming fails without PyBoy instance"""
        result = streamer.start_streaming()
        
        assert result == False
        assert streamer._capture_active == False
    
    def test_start_streaming_with_pyboy(self, streamer, mock_pyboy):
        """Test successful streaming startup with PyBoy"""
        streamer.set_pyboy_instance(mock_pyboy)
        
        result = streamer.start_streaming()
        
        assert result == True
        assert streamer._capture_active == True
        assert streamer._capture_thread is not None
        assert streamer._capture_thread.is_alive()
        
        # Clean up
        streamer.stop_streaming()
    
    def test_stop_streaming(self, streamer, mock_pyboy):
        """Test stopping streaming process"""
        streamer.set_pyboy_instance(mock_pyboy)
        streamer.start_streaming()
        
        # Verify it started
        assert streamer._capture_active == True
        
        streamer.stop_streaming()
        
        # Verify it stopped
        assert streamer._capture_active == False
        # Thread should be joined
        time.sleep(0.1)  # Give thread time to shut down
    
    def test_capture_frame_success(self, streamer, mock_pyboy):
        """Test successful frame capture from PyBoy"""
        streamer.set_pyboy_instance(mock_pyboy)
        
        frame = streamer._capture_frame()
        
        assert frame is not None
        assert isinstance(frame, np.ndarray)
        assert frame.shape == (144, 160, 3)
        assert streamer._frames_captured == 1
        assert len(streamer._capture_times) == 1
    
    def test_capture_frame_empty_screen(self, streamer, mock_pyboy):
        """Test frame capture with empty screen"""
        mock_pyboy.screen.ndarray = None
        streamer.set_pyboy_instance(mock_pyboy)
        
        frame = streamer._capture_frame()
        
        assert frame is None
    
    def test_capture_frame_error(self, streamer):
        """Test frame capture error handling"""
        mock_pyboy = Mock()
        mock_screen = Mock()
        # The ndarray property should raise an exception when accessed
        type(mock_screen).ndarray = PropertyMock(side_effect=Exception("Screen error"))
        mock_pyboy.screen = mock_screen
        streamer.set_pyboy_instance(mock_pyboy)
        
        with pytest.raises(Exception):
            streamer._capture_frame()


@pytest.mark.streaming
@pytest.mark.unit
class TestFrameFormats:
    """Test different frame output formats"""
    
    @pytest.fixture
    def streamer_with_frame(self):
        """Create streamer with a test frame loaded"""
        streamer = GameStreamComponent(enable_data_bus=False)
        test_frame = np.random.randint(0, 255, (144, 160, 3), dtype=np.uint8)
        streamer._latest_frame = test_frame
        yield streamer
        streamer.shutdown()
    
    def test_get_frame_numpy(self, streamer_with_frame):
        """Test getting frame in numpy format"""
        frame = streamer_with_frame.get_latest_frame("numpy")
        
        assert frame is not None
        assert isinstance(frame, np.ndarray)
        assert frame.shape == (144, 160, 3)
        assert streamer_with_frame._frames_served == 1
    
    def test_get_frame_png(self, streamer_with_frame):
        """Test getting frame in PNG format"""
        frame = streamer_with_frame.get_latest_frame("png")
        
        assert frame is not None
        assert isinstance(frame, bytes)
        # Verify it's valid PNG data
        img = Image.open(io.BytesIO(frame))
        assert img.format == 'PNG'
        assert img.size == (160, 144)  # PIL uses (width, height)
    
    def test_get_frame_jpeg(self, streamer_with_frame):
        """Test getting frame in JPEG format"""
        frame = streamer_with_frame.get_latest_frame("jpeg")
        
        assert frame is not None
        assert isinstance(frame, bytes)
        # Verify it's valid JPEG data
        img = Image.open(io.BytesIO(frame))
        assert img.format == 'JPEG'
        assert img.size == (160, 144)
    
    def test_get_frame_base64_png(self, streamer_with_frame):
        """Test getting frame in base64 PNG format"""
        frame = streamer_with_frame.get_latest_frame("base64_png")
        
        assert frame is not None
        assert isinstance(frame, str)
        # Verify it's valid base64 data
        decoded = base64.b64decode(frame)
        img = Image.open(io.BytesIO(decoded))
        assert img.format == 'PNG'
    
    def test_get_frame_base64_jpeg(self, streamer_with_frame):
        """Test getting frame in base64 JPEG format"""
        frame = streamer_with_frame.get_latest_frame("base64_jpeg")
        
        assert frame is not None
        assert isinstance(frame, str)
        # Verify it's valid base64 data
        decoded = base64.b64decode(frame)
        img = Image.open(io.BytesIO(decoded))
        assert img.format == 'JPEG'
    
    def test_get_frame_rgba_conversion(self, streamer_with_frame):
        """Test RGBA to RGB conversion for JPEG compatibility"""
        # Create RGBA frame
        rgba_frame = np.random.randint(0, 255, (144, 160, 4), dtype=np.uint8)
        streamer_with_frame._latest_frame = rgba_frame
        
        frame = streamer_with_frame.get_latest_frame("jpeg")
        
        assert frame is not None
        # Should successfully convert RGBA to RGB for JPEG
        img = Image.open(io.BytesIO(frame))
        assert img.format == 'JPEG'
        assert img.mode == 'RGB'
    
    def test_get_frame_no_frame_available(self, streamer_with_frame):
        """Test getting frame when no frame is available"""
        streamer_with_frame._latest_frame = None
        
        frame = streamer_with_frame.get_latest_frame("numpy")
        
        assert frame is None
    
    def test_get_frame_invalid_format(self, streamer_with_frame):
        """Test getting frame with invalid format"""
        frame = streamer_with_frame.get_latest_frame("invalid_format")
        
        assert frame is None
    
    def test_get_frame_conversion_error(self, streamer_with_frame):
        """Test frame conversion error handling"""
        # Create an invalid frame that will cause conversion errors
        invalid_frame = np.array([[]])  # Empty frame
        streamer_with_frame._latest_frame = invalid_frame
        
        frame = streamer_with_frame.get_latest_frame("png")
        
        assert frame is None


@pytest.mark.streaming
@pytest.mark.unit
class TestFrameBuffering:
    """Test frame buffering system"""
    
    @pytest.fixture
    def streamer(self):
        """Create streamer with small buffer for testing"""
        streamer = GameStreamComponent(buffer_size=3, enable_data_bus=False)
        yield streamer
        streamer.shutdown()
    
    def test_frame_buffer_normal_operation(self, streamer):
        """Test normal frame buffering operation"""
        frames = []
        for i in range(2):
            frame = np.full((144, 160, 3), i, dtype=np.uint8)
            frames.append(frame)
            streamer._process_frame(frame)
        
        assert streamer._frame_buffer.qsize() == 2
        assert streamer._latest_frame is not None
        assert np.array_equal(streamer._latest_frame, frames[-1])
    
    def test_frame_buffer_overflow(self, streamer):
        """Test frame buffer overflow handling"""
        # Fill buffer beyond capacity
        for i in range(5):
            frame = np.full((144, 160, 3), i, dtype=np.uint8)
            streamer._process_frame(frame)
        
        # Buffer should not exceed max size
        assert streamer._frame_buffer.qsize() <= streamer.buffer_size
        # Latest frame should still be available
        assert streamer._latest_frame is not None
    
    def test_clear_buffer(self, streamer):
        """Test clearing the frame buffer"""
        # Add frames to buffer
        for i in range(3):
            frame = np.full((144, 160, 3), i, dtype=np.uint8)
            streamer._process_frame(frame)
        
        assert streamer._frame_buffer.qsize() == 3
        
        streamer.clear_buffer()
        
        assert streamer._frame_buffer.empty()
    
    def test_frame_metadata(self, streamer):
        """Test frame metadata generation"""
        test_frame = np.random.randint(50, 200, (144, 160, 3), dtype=np.uint8)
        streamer._process_frame(test_frame)
        
        info = streamer.get_frame_info()
        
        assert 'timestamp' in info
        assert info['shape'] == (144, 160, 3)
        assert info['dtype'] == 'uint8'
        assert info['size_bytes'] == test_frame.nbytes
        assert 50 <= info['min_value'] <= 200
        assert 50 <= info['max_value'] <= 200
        assert isinstance(info['mean_value'], float)


@pytest.mark.streaming
@pytest.mark.performance
class TestPerformanceAndStats:
    """Test performance monitoring and statistics"""
    
    @pytest.fixture
    def streamer(self):
        """Create streamer for performance testing"""
        streamer = GameStreamComponent(enable_data_bus=False)
        yield streamer
        streamer.shutdown()
    
    @pytest.fixture
    def mock_pyboy(self):
        """Create mock PyBoy instance for performance tests"""
        mock_pyboy = Mock()
        mock_screen = Mock()
        test_array = np.random.randint(0, 255, (144, 160, 3), dtype=np.uint8)
        mock_screen.ndarray = test_array
        mock_pyboy.screen = mock_screen
        return mock_pyboy
    
    def test_performance_stats_initial(self, streamer):
        """Test initial performance statistics"""
        stats = streamer.get_performance_stats()
        
        assert stats['frames_captured'] == 0
        assert stats['frames_served'] == 0
        assert stats['capture_errors'] == 0
        assert stats['error_rate'] == 0.0
        assert stats['current_fps'] == 0.0
        assert stats['target_fps'] == 10.0
        assert stats['buffer_utilization'] == 0.0
        assert stats['is_streaming'] == False
        assert stats['consecutive_errors'] == 0
    
    def test_performance_stats_after_frames(self, streamer):
        """Test performance statistics after processing frames"""
        mock_pyboy = Mock()
        mock_pyboy.screen.ndarray = np.random.randint(0, 255, (144, 160, 3), dtype=np.uint8)
        streamer.set_pyboy_instance(mock_pyboy)
        
        # Capture some frames
        for _ in range(5):
            streamer._capture_frame()
        
        # Serve some frames
        test_frame = np.random.randint(0, 255, (144, 160, 3), dtype=np.uint8)
        streamer._latest_frame = test_frame
        for _ in range(3):
            streamer.get_latest_frame("numpy")
        
        stats = streamer.get_performance_stats()
        
        assert stats['frames_captured'] == 5
        assert stats['frames_served'] == 3
        assert stats['capture_errors'] == 0
        assert stats['error_rate'] == 0.0
        assert len(streamer._capture_times) == 5
    
    def test_health_check_inactive(self, streamer):
        """Test health check when streaming is inactive"""
        assert streamer.is_healthy() == False
    
    def test_health_check_active_no_activity(self, streamer, mock_pyboy):
        """Test health check when active but no recent activity"""
        streamer.set_pyboy_instance(mock_pyboy)
        streamer._capture_active = True
        # No recent capture time
        
        assert streamer.is_healthy() == False
    
    def test_health_check_active_recent_activity(self, streamer):
        """Test health check when active with recent activity"""
        streamer._capture_active = True
        streamer._consecutive_errors = 0
        streamer._last_capture_time = time.time()  # Recent activity
        
        assert streamer.is_healthy() == True
    
    def test_health_check_too_many_errors(self, streamer):
        """Test health check with too many consecutive errors"""
        streamer._capture_active = True
        streamer._consecutive_errors = 10  # Exceeds max
        streamer._last_capture_time = time.time()
        
        assert streamer.is_healthy() == False


@pytest.mark.streaming
@pytest.mark.unit
class TestErrorHandling:
    """Test error handling and recovery"""
    
    @pytest.fixture
    def streamer(self):
        """Create streamer for error testing"""
        streamer = GameStreamComponent(enable_data_bus=False)
        yield streamer
        streamer.shutdown()
    
    def test_capture_error_handling(self, streamer):
        """Test capture error handling and counting"""
        initial_errors = streamer._capture_errors
        initial_consecutive = streamer._consecutive_errors
        
        error = Exception("Test capture error")
        streamer._handle_capture_error(error)
        
        assert streamer._capture_errors == initial_errors + 1
        assert streamer._consecutive_errors == initial_consecutive + 1
    
    def test_error_backoff_escalation(self, streamer):
        """Test error backoff time escalation"""
        # Simulate multiple consecutive errors
        error = Exception("Persistent error")
        
        for i in range(3):
            streamer._handle_capture_error(error)
        
        # Should have escalating consecutive errors
        assert streamer._consecutive_errors == 3
        assert streamer._capture_errors == 3
    
    @patch('monitoring.game_streamer.get_data_bus')
    def test_error_reporting_to_data_bus(self, mock_get_data_bus):
        """Test error reporting to data bus"""
        mock_bus = Mock()
        mock_get_data_bus.return_value = mock_bus
        
        streamer = GameStreamComponent(enable_data_bus=True)
        
        # Cause enough errors to trigger data bus reporting
        error = Exception("Critical error")
        for _ in range(6):  # Exceed max consecutive errors
            streamer._handle_capture_error(error)
        
        # Should have published error to data bus
        mock_bus.publish.assert_called()
        calls = mock_bus.publish.call_args_list
        error_call = None
        for call in calls:
            if call[0][0] == DataType.ERROR_EVENT:
                error_call = call
                break
        
        assert error_call is not None
        error_data = error_call[0][1]
        assert error_data['component'] == 'game_streamer'
        assert 'consecutive_errors' in error_data
        assert error_data['severity'] in ['high', 'medium']
        
        streamer.shutdown()


@pytest.mark.streaming
@pytest.mark.integration
class TestStreamingIntegration:
    """Test complete streaming system integration"""
    
    @pytest.fixture
    def mock_pyboy(self):
        """Create comprehensive mock PyBoy instance"""
        mock_pyboy = Mock()
        mock_screen = Mock()
        
        # Create sequence of different test frames
        frames = []
        for i in range(10):
            frame = np.full((144, 160, 3), i * 25, dtype=np.uint8)
            frames.append(frame)
        
        # Create a cycling iterator that will never run out
        frame_cycle = []
        def get_frame():
            if not frame_cycle:
                frame_cycle.extend(frames * 10)  # Initialize with many copies
            return frame_cycle.pop(0)  # Return first frame and remove it
        
        mock_screen.ndarray = get_frame
        mock_pyboy.screen = mock_screen
        return mock_pyboy
    
    @pytest.fixture
    def streamer(self):
        """Create streamer for integration testing"""
        streamer = GameStreamComponent(
            buffer_size=5,
            max_frame_rate=30.0,  # Higher rate for testing
            enable_data_bus=False
        )
        yield streamer
        streamer.shutdown()
    
    def test_full_streaming_cycle(self, streamer, mock_pyboy):
        """Test complete streaming cycle"""
        streamer.set_pyboy_instance(mock_pyboy)
        
        # Start streaming
        result = streamer.start_streaming()
        assert result == True
        
        # Let it run for a short time
        time.sleep(0.2)
        
        # Check that frames are being captured
        assert streamer._frames_captured > 0
        
        # Get frames in different formats
        numpy_frame = streamer.get_latest_frame("numpy")
        png_frame = streamer.get_latest_frame("png")
        jpeg_frame = streamer.get_latest_frame("jpeg")
        
        assert numpy_frame is not None
        assert png_frame is not None
        assert jpeg_frame is not None
        
        # Check performance stats
        stats = streamer.get_performance_stats()
        assert stats['frames_captured'] > 0
        assert stats['frames_served'] > 0
        assert stats['is_streaming'] == True
        
        # Check health
        assert streamer.is_healthy() == True
        
        # Stop streaming
        streamer.stop_streaming()
        assert streamer._capture_active == False
    
    def test_streaming_with_errors_and_recovery(self, streamer):
        """Test streaming with errors and recovery"""
        mock_pyboy = Mock()
        mock_screen = Mock()
        
        # Create a sequence: success, error, success, error, success...
        call_count = [0]
        def screen_side_effect():
            call_count[0] += 1
            if call_count[0] % 3 == 0:  # Every 3rd call fails
                raise Exception("Intermittent screen error")
            return np.random.randint(0, 255, (144, 160, 3), dtype=np.uint8)
        
        mock_screen.ndarray = Mock(side_effect=screen_side_effect)
        mock_pyboy.screen = mock_screen
        
        streamer.set_pyboy_instance(mock_pyboy)
        streamer.start_streaming()
        
        # Let it run and encounter errors
        time.sleep(0.3)
        
        # Should have captured some frames despite errors
        assert streamer._frames_captured > 0
        assert streamer._capture_errors > 0
        
        # Should still be healthy if not too many consecutive errors
        stats = streamer.get_performance_stats()
        assert stats['error_rate'] > 0.0
        
        streamer.stop_streaming()
    
    def test_memory_management_long_run(self, streamer, mock_pyboy):
        """Test memory management during longer streaming session"""
        import gc
        import psutil
        import os
        
        # Get initial memory
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        streamer.set_pyboy_instance(mock_pyboy)
        streamer.start_streaming()
        
        # Run for longer period with many frame captures
        for _ in range(20):
            time.sleep(0.01)
            # Request frames in different formats to stress memory
            if streamer._latest_frame is not None:
                streamer.get_latest_frame("numpy")
                streamer.get_latest_frame("png")
                streamer.get_latest_frame("base64_jpeg")
        
        streamer.stop_streaming()
        
        # Force garbage collection
        gc.collect()
        
        # Check final memory
        final_memory = process.memory_info().rss
        memory_increase = (final_memory - initial_memory) / 1024 / 1024  # MB
        
        # Memory increase should be reasonable (under 20MB for this test)
        assert memory_increase < 20, f"Memory usage increased by {memory_increase:.1f}MB"
    
    def test_thread_safety(self, streamer, mock_pyboy):
        """Test thread safety of frame access"""
        streamer.set_pyboy_instance(mock_pyboy)
        streamer.start_streaming()
        
        # Let some frames be captured
        time.sleep(0.1)
        
        results = []
        errors = []
        
        def frame_getter():
            try:
                for _ in range(10):
                    frame = streamer.get_latest_frame("numpy")
                    if frame is not None:
                        results.append(frame.shape)
                    time.sleep(0.01)
            except Exception as e:
                errors.append(e)
        
        # Start multiple threads accessing frames
        threads = []
        for _ in range(3):
            t = threading.Thread(target=frame_getter)
            threads.append(t)
            t.start()
        
        # Wait for threads to complete
        for t in threads:
            t.join()
        
        # Should have gotten frames without errors
        assert len(errors) == 0, f"Thread safety errors: {errors}"
        assert len(results) > 0, "Should have captured frames from multiple threads"
        
        streamer.stop_streaming()


@pytest.mark.streaming
@pytest.mark.cleanup
class TestCleanupAndShutdown:
    """Test cleanup and shutdown procedures"""
    
    def test_shutdown_inactive_streamer(self):
        """Test shutdown of inactive streamer"""
        streamer = GameStreamComponent(enable_data_bus=False)
        
        # Should shutdown cleanly even if never used
        streamer.shutdown()
        
        assert streamer._latest_frame is None
        assert len(streamer._latest_frame_data) == 0
        assert streamer._pyboy is None
    
    def test_shutdown_active_streamer(self):
        """Test shutdown of active streaming system"""
        mock_pyboy = Mock()
        mock_pyboy.screen.ndarray = np.random.randint(0, 255, (144, 160, 3), dtype=np.uint8)
        
        streamer = GameStreamComponent(enable_data_bus=False)
        streamer.set_pyboy_instance(mock_pyboy)
        streamer.start_streaming()
        
        # Let it run briefly
        time.sleep(0.1)
        
        # Shutdown should stop streaming and clean up
        streamer.shutdown()
        
        assert streamer._capture_active == False
        assert streamer._latest_frame is None
        assert streamer._pyboy is None
    
    @patch('monitoring.game_streamer.get_data_bus')
    def test_shutdown_with_data_bus(self, mock_get_data_bus):
        """Test shutdown with data bus integration"""
        mock_bus = Mock()
        mock_get_data_bus.return_value = mock_bus
        
        streamer = GameStreamComponent(enable_data_bus=True)
        streamer.shutdown()
        
        # Should notify data bus of shutdown
        mock_bus.publish.assert_called()
        calls = mock_bus.publish.call_args_list
        shutdown_call = None
        for call in calls:
            if call[0][0] == DataType.COMPONENT_STATUS:
                shutdown_call = call
                break
        
        assert shutdown_call is not None
        status_data = shutdown_call[0][1]
        assert status_data['component'] == 'game_streamer'
        assert status_data['status'] == 'shutdown'


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-m", "streaming"])
