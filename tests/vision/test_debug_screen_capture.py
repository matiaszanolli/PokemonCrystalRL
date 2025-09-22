import unittest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import sys
import os
from PIL import Image
import io
import base64

from vision.debug.debug_screen_capture import (
    _test_pyboy_screen_methods, main, PYBOY_AVAILABLE
)

class TestDebugScreenCapture(unittest.TestCase):
    """Test debug screen capture utility functions"""

    def setUp(self):
        """Set up test fixtures"""
        self.rom_path = "test_rom.gbc"
        
    @patch('vision.debug.debug_screen_capture.PyBoy')
    def test_pyboy_screen_methods_basic(self, mock_pyboy):
        """Test basic functionality of pyboy screen methods"""
        _test_pyboy_screen_methods(self.rom_path)
        mock_pyboy.assert_called_once()


class TestPyBoyAvailability(unittest.TestCase):
    """Test PyBoy availability detection"""
    
    def test_pyboy_available_constant(self):
        """Test that PYBOY_AVAILABLE is a boolean"""
        self.assertIsInstance(PYBOY_AVAILABLE, bool)
    
    @patch('vision.debug.debug_screen_capture.PyBoy')
    def test_pyboy_import_success(self, mock_pyboy):
        """Test successful PyBoy import"""
        self.assertIsNotNone(mock_pyboy)
        
    def test_pyboy_not_available_message(self):
        """Test that when PyBoy is not available, proper message is displayed"""
        pass


class TestPyBoyScreenMethods(unittest.TestCase):
    """Test PyBoy screen capture method testing"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.rom_path = "test_rom.gbc"
        
        # Create mock PyBoy instance
        self.mock_pyboy = Mock()
        
        # Mock screen with ndarray property
        self.mock_screen = Mock()
        test_image_data = np.random.randint(0, 256, (144, 160, 3), dtype=np.uint8)
        self.mock_screen.ndarray = test_image_data
        
        # Mock screen image method
        test_pil_image = Image.fromarray(test_image_data)
        self.mock_screen.image.return_value = test_pil_image
        
        # Set up mock PyBoy with screen
        self.mock_pyboy.screen = self.mock_screen
        self.mock_pyboy.tick = Mock()
        self.mock_pyboy.stop = Mock()
        
    def test_pyboy_not_available(self):
        """Test behavior when PyBoy is not available"""
        with patch('vision.debug.debug_screen_capture.PYBOY_AVAILABLE', False):
            result = _test_pyboy_screen_methods(self.rom_path)
            self.assertIsNone(result)
    
    @patch('vision.debug.debug_screen_capture.PyBoy')
    @patch('vision.debug.debug_screen_capture.PYBOY_AVAILABLE', True)
    def test_pyboy_initialization(self, mock_pyboy_class):
        """Test PyBoy initialization"""
        mock_pyboy_class.return_value = self.mock_pyboy
        
        _test_pyboy_screen_methods(self.rom_path)
        
        mock_pyboy_class.assert_called_once_with(
            self.rom_path, window="null", debug=False
        )
        
        self.assertGreaterEqual(self.mock_pyboy.tick.call_count, 10)
        self.mock_pyboy.stop.assert_called_once()

    @patch('vision.debug.debug_screen_capture.PyBoy')
    @patch('vision.debug.debug_screen_capture.PYBOY_AVAILABLE', True)
    def test_screen_capture_success(self, mock_pyboy_class):
        """Test successful screen capture"""
        mock_pyboy_class.return_value = self.mock_pyboy
        _test_pyboy_screen_methods(self.rom_path)
        self.assertTrue(self.mock_pyboy.tick.called)

if __name__ == '__main__':
    unittest.main()
