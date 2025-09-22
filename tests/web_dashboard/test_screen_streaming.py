"""
Integration tests for web dashboard screen streaming functionality.

Tests the critical screen streaming fixes implemented in September 2024.
"""

import pytest
import requests
import time
import tempfile
from PIL import Image
from io import BytesIO
from unittest.mock import Mock, MagicMock, patch
import numpy as np

from web_dashboard.websocket_handler import WebSocketHandler
from web_dashboard.server import UnifiedWebServer


@pytest.mark.web_monitoring
class TestScreenStreaming:
    """Test screen streaming functionality with proper PyBoy mocking."""

    def test_websocket_handler_screen_capture(self):
        """Test WebSocket handler can capture screen from PyBoy."""
        # Mock trainer with PyBoy instance
        mock_trainer = Mock()
        mock_pyboy = Mock()
        mock_emulation_manager = Mock()

        # Setup mock screen data (160x144 RGB array - Game Boy screen size)
        mock_screen_array = np.random.randint(0, 255, (144, 160, 3), dtype=np.uint8)
        mock_pyboy.screen.ndarray = mock_screen_array

        mock_emulation_manager.get_instance.return_value = mock_pyboy
        mock_trainer.emulation_manager = mock_emulation_manager

        # Create WebSocket handler
        handler = WebSocketHandler(trainer=mock_trainer)

        # Test screen capture
        screen_data = handler._get_current_screen()

        # Verify screen data is captured
        assert screen_data is not None
        assert screen_data.startswith("data:image/png;base64,")
        assert handler.latest_screen_data is not None
        assert len(handler.latest_screen_data) > 0

    def test_websocket_handler_update_for_http(self):
        """Test update_screen_for_http method populates latest_screen_data."""
        # Mock trainer setup
        mock_trainer = Mock()
        mock_pyboy = Mock()
        mock_emulation_manager = Mock()

        mock_screen_array = np.random.randint(0, 255, (144, 160, 3), dtype=np.uint8)
        mock_pyboy.screen.ndarray = mock_screen_array

        mock_emulation_manager.get_instance.return_value = mock_pyboy
        mock_trainer.emulation_manager = mock_emulation_manager

        handler = WebSocketHandler(trainer=mock_trainer)

        # Initially no screen data
        assert handler.latest_screen_data is None

        # Call update method
        handler.update_screen_for_http()

        # Verify screen data is now populated
        assert handler.latest_screen_data is not None
        assert len(handler.latest_screen_data) > 0

    def test_screen_scaling_and_format(self):
        """Test screen is properly scaled and formatted."""
        mock_trainer = Mock()
        mock_pyboy = Mock()
        mock_emulation_manager = Mock()

        # Original Game Boy resolution: 160x144
        mock_screen_array = np.random.randint(0, 255, (144, 160, 3), dtype=np.uint8)
        mock_pyboy.screen.ndarray = mock_screen_array

        mock_emulation_manager.get_instance.return_value = mock_pyboy
        mock_trainer.emulation_manager = mock_emulation_manager

        handler = WebSocketHandler(trainer=mock_trainer)
        screen_data = handler._get_current_screen()

        # Decode the base64 image to verify scaling
        import base64
        base64_data = screen_data.split(',')[1]
        image_bytes = base64.b64decode(base64_data)
        image = Image.open(BytesIO(image_bytes))

        # Verify scaled dimensions (should be 320x288 - 2x scale)
        assert image.size == (320, 288)
        assert image.format == 'PNG'

    def test_fallback_without_trainer(self):
        """Test handler gracefully handles missing trainer."""
        handler = WebSocketHandler(trainer=None)

        screen_data = handler._get_current_screen()
        assert screen_data is None
        assert handler.latest_screen_data is None

        # update_screen_for_http should not crash
        handler.update_screen_for_http()
        assert handler.latest_screen_data is None

    def test_server_screen_endpoint_integration(self):
        """Test server properly calls WebSocket handler for screen updates."""
        # Mock components
        mock_websocket_handler = Mock()
        mock_websocket_handler.update_screen_for_http = Mock()
        mock_websocket_handler.get_latest_screen = Mock(return_value=b'mock_png_data')

        # Test the _serve_screen logic directly
        from web_dashboard.server import UnifiedHttpHandler

        # Create a mock handler with just the methods we need
        handler = Mock(spec=UnifiedHttpHandler)
        handler.websocket_handler = mock_websocket_handler
        handler.send_response = Mock()
        handler.send_header = Mock()
        handler._set_cors_headers = Mock()
        handler.end_headers = Mock()
        handler.wfile = Mock()
        handler._serve_placeholder_image = Mock()

        # Import and call the actual _serve_screen method
        UnifiedHttpHandler._serve_screen(handler)

        # Verify the fix: update_screen_for_http was called
        mock_websocket_handler.update_screen_for_http.assert_called_once()
        mock_websocket_handler.get_latest_screen.assert_called_once()

        # Verify response was sent
        handler.send_response.assert_called_with(200)
        handler.send_header.assert_called_with('Content-type', 'image/png')

    @pytest.mark.integration
    def test_end_to_end_screen_capture_flow(self):
        """Test complete screen capture flow from PyBoy to HTTP response."""
        # Mock complete trainer setup
        mock_trainer = Mock()
        mock_pyboy = Mock()
        mock_emulation_manager = Mock()
        mock_stats_tracker = Mock()

        # Setup screen data
        mock_screen_array = np.random.randint(0, 255, (144, 160, 3), dtype=np.uint8)
        mock_pyboy.screen.ndarray = mock_screen_array

        mock_emulation_manager.get_instance.return_value = mock_pyboy
        mock_trainer.emulation_manager = mock_emulation_manager
        mock_trainer.stats_tracker = mock_stats_tracker
        mock_stats_tracker.get_current_stats.return_value = {}

        # Create components
        handler = WebSocketHandler(trainer=mock_trainer)

        # Simulate HTTP request for screen
        handler.update_screen_for_http()
        screen_bytes = handler.get_latest_screen()

        # Verify complete flow
        assert screen_bytes is not None
        assert len(screen_bytes) > 0

        # Verify it's a valid PNG
        image = Image.open(BytesIO(screen_bytes))
        assert image.format == 'PNG'
        assert image.size == (320, 288)

        # Multiple calls should work (testing for memory leaks)
        for _ in range(5):
            handler.update_screen_for_http()
            new_screen_bytes = handler.get_latest_screen()
            assert new_screen_bytes is not None
            assert len(new_screen_bytes) > 0


@pytest.mark.web_monitoring
class TestScreenStreamingErrorHandling:
    """Test error handling in screen streaming."""

    def test_pyboy_screen_access_error(self):
        """Test graceful handling when PyBoy screen access fails."""
        mock_trainer = Mock()
        mock_pyboy = Mock()
        mock_emulation_manager = Mock()

        # Simulate PyBoy screen access error
        mock_pyboy.screen.ndarray = Mock(side_effect=Exception("Screen access failed"))

        mock_emulation_manager.get_instance.return_value = mock_pyboy
        mock_trainer.emulation_manager = mock_emulation_manager

        handler = WebSocketHandler(trainer=mock_trainer)

        # Should handle error gracefully
        screen_data = handler._get_current_screen()
        assert screen_data is None
        assert handler.latest_screen_data is None

    def test_invalid_screen_array_format(self):
        """Test handling of unexpected screen array formats."""
        mock_trainer = Mock()
        mock_pyboy = Mock()
        mock_emulation_manager = Mock()

        # Invalid screen array (will cause PIL to fail during resize)
        mock_pyboy.screen.ndarray = np.array([[1, 2], [3, 4]])  # 2x2 array, too small

        mock_emulation_manager.get_instance.return_value = mock_pyboy
        mock_trainer.emulation_manager = mock_emulation_manager

        handler = WebSocketHandler(trainer=mock_trainer)

        # Should handle error gracefully - the fromarray or resize might fail
        screen_data = handler._get_current_screen()
        # Could be None due to error, or might succeed with minimal image
        # Just ensure it doesn't crash the system
        assert screen_data is None or isinstance(screen_data, str)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])