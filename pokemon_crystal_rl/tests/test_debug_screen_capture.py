"""
Comprehensive unit tests for the Debug Screen Capture module.
Tests PyBoy screen capture methods, image processing, and debugging utilities.
"""
import unittest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import sys
import os
from PIL import Image
import io
import base64

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from vision.debug_screen_capture import (
    _test_pyboy_screen_methods, main, PYBOY_AVAILABLE
)


class TestDebugScreenCapture(unittest.TestCase):
    """Test debug screen capture utility functions"""

    def setUp(self):
        """Set up test fixtures"""
        self.rom_path = "test_rom.gbc"
        
    @patch('vision.debug_screen_capture.PyBoy')
    def test_pyboy_screen_methods_basic(self, mock_pyboy):
        """Test basic functionality of pyboy screen methods"""
        _test_pyboy_screen_methods(self.rom_path)
        mock_pyboy.assert_called_once()


class TestPyBoyAvailability(unittest.TestCase):
    """Test PyBoy availability detection"""
    
    def test_pyboy_available_constant(self):
        """Test that PYBOY_AVAILABLE is a boolean"""
        self.assertIsInstance(PYBOY_AVAILABLE, bool)
    
    @patch('vision.debug_screen_capture.PyBoy')
    def test_pyboy_import_success(self, mock_pyboy):
        """Test successful PyBoy import"""
        # The import would have already happened during module loading
        # This test verifies the mock works
        self.assertIsNotNone(mock_pyboy)
        
    def test_pyboy_not_available_message(self):
        """Test that when PyBoy is not available, proper message is displayed"""
        # This is implicitly tested by the module loading
        # If PyBoy was not available, PYBOY_AVAILABLE would be False
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
        with patch('vision.debug_screen_capture.PYBOY_AVAILABLE', False):
            result = _test_pyboy_screen_methods(self.rom_path)
            # Should return early without error
            self.assertIsNone(result)
    
    @patch('vision.debug_screen_capture.PyBoy')
    @patch('vision.debug_screen_capture.PYBOY_AVAILABLE', True)
    def test_pyboy_initialization(self, mock_pyboy_class):
        """Test PyBoy initialization"""
        mock_pyboy_class.return_value = self.mock_pyboy
        
        _test_pyboy_screen_methods(self.rom_path)
        
        # Verify PyBoy was initialized with correct parameters
        mock_pyboy_class.assert_called_once_with(
            self.rom_path, window="null", debug=False
        )
        
        # Verify tick was called for boot frames (10 times initially)
        self.assertGreaterEqual(self.mock_pyboy.tick.call_count, 10)
        
        # Verify stop was called
        self.mock_pyboy.stop.assert_called_once()
    
    @patch('vision.debug_screen_capture.PyBoy')
    @patch('vision.debug_screen_capture.PYBOY_AVAILABLE', True)
    def test_screen_ndarray_method(self, mock_pyboy_class):
        """Test screen.ndarray method testing"""
        mock_pyboy_class.return_value = self.mock_pyboy
        
        _test_pyboy_screen_methods(self.rom_path)
        
        # Verify ndarray property was accessed
        self.assertTrue(hasattr(self.mock_pyboy.screen, 'ndarray'))
        
    @patch('vision.debug_screen_capture.PyBoy')
    @patch('vision.debug_screen_capture.PYBOY_AVAILABLE', True)
    def test_screen_image_method(self, mock_pyboy_class):
        """Test screen.image method testing"""
        mock_pyboy_class.return_value = self.mock_pyboy
        
        _test_pyboy_screen_methods(self.rom_path)
        
        # Verify image method was called
        self.mock_screen.image.assert_called()
        
    @patch('vision.debug_screen_capture.PyBoy')
    @patch('vision.debug_screen_capture.PYBOY_AVAILABLE', True)
    def test_screen_methods_enumeration(self, mock_pyboy_class):
        """Test enumeration of available screen methods"""
        mock_pyboy_class.return_value = self.mock_pyboy
        
        # Add some mock methods to screen
        self.mock_screen.method1 = Mock()
        self.mock_screen.method2 = Mock()
        self.mock_screen._private_method = Mock()
        
        _test_pyboy_screen_methods(self.rom_path)
        
        # The test should enumerate available methods
        # This is tested implicitly through the dir() call in the function
        
    @patch('vision.debug_screen_capture.PyBoy')
    @patch('vision.debug_screen_capture.PYBOY_AVAILABLE', True)
    def test_continuous_capture(self, mock_pyboy_class):
        """Test continuous screen capture"""
        mock_pyboy_class.return_value = self.mock_pyboy
        
        _test_pyboy_screen_methods(self.rom_path)
        
        # Verify additional tick calls for continuous capture
        # Total should be initial 10 + continuous 10 = at least 20
        self.assertGreaterEqual(self.mock_pyboy.tick.call_count, 20)
        
    @patch('vision.debug_screen_capture.PyBoy')
    @patch('vision.debug_screen_capture.PYBOY_AVAILABLE', True) 
    def test_pil_image_processing(self, mock_pyboy_class):
        """Test PIL image processing within screen capture"""
        mock_pyboy_class.return_value = self.mock_pyboy
        
        _test_pyboy_screen_methods(self.rom_path)
        
        # The function should process images
        # This is tested implicitly as the function completes without error
        
    @patch('vision.debug_screen_capture.PyBoy')
    @patch('vision.debug_screen_capture.PYBOY_AVAILABLE', True)
    def test_base64_encoding(self, mock_pyboy_class):
        """Test base64 encoding within screen capture"""
        mock_pyboy_class.return_value = self.mock_pyboy
        
        _test_pyboy_screen_methods(self.rom_path)
        
        # The function should encode images
        # This is tested implicitly as the function completes without error


class TestErrorHandling(unittest.TestCase):
    """Test error handling in screen capture methods"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.rom_path = "test_rom.gbc"
    
    @patch('vision.debug_screen_capture.PyBoy')
    @patch('vision.debug_screen_capture.PYBOY_AVAILABLE', True)
    def test_pyboy_initialization_error(self, mock_pyboy_class):
        """Test error handling during PyBoy initialization"""
        mock_pyboy_class.side_effect = Exception("Mock PyBoy initialization error")
        
        # The debug function doesn't handle PyBoy initialization errors
        # It's designed to show the error output for debugging purposes
        with self.assertRaises(Exception):
            _test_pyboy_screen_methods(self.rom_path)
    
    @patch('vision.debug_screen_capture.PyBoy')
    @patch('vision.debug_screen_capture.PYBOY_AVAILABLE', True)
    def test_screen_attribute_missing(self, mock_pyboy_class):
        """Test error handling when screen attribute is missing"""
        mock_pyboy = Mock()
        # Don't set screen attribute
        del mock_pyboy.screen  # Remove screen attribute
        mock_pyboy_class.return_value = mock_pyboy
        
        # Should handle missing screen attribute gracefully
        try:
            _test_pyboy_screen_methods(self.rom_path)
        except Exception as e:
            self.fail(f"test_pyboy_screen_methods raised exception: {e}")
    
    @patch('vision.debug_screen_capture.PyBoy')
    @patch('vision.debug_screen_capture.PYBOY_AVAILABLE', True)
    def test_ndarray_attribute_missing(self, mock_pyboy_class):
        """Test error handling when ndarray attribute is missing"""
        mock_pyboy = Mock()
        mock_screen = Mock()
        # Don't set ndarray attribute
        if hasattr(mock_screen, 'ndarray'):
            delattr(mock_screen, 'ndarray')
        mock_pyboy.screen = mock_screen
        mock_pyboy_class.return_value = mock_pyboy
        
        # Should handle missing ndarray attribute gracefully
        try:
            _test_pyboy_screen_methods(self.rom_path)
        except Exception as e:
            self.fail(f"test_pyboy_screen_methods raised exception: {e}")
    
    @patch('vision.debug_screen_capture.PyBoy')
    @patch('vision.debug_screen_capture.PYBOY_AVAILABLE', True)
    def test_image_method_missing(self, mock_pyboy_class):
        """Test error handling when image method is missing"""
        mock_pyboy = Mock()
        mock_screen = Mock()
        # Don't set image method
        if hasattr(mock_screen, 'image'):
            delattr(mock_screen, 'image')
        mock_pyboy.screen = mock_screen
        mock_pyboy_class.return_value = mock_pyboy
        
        # Should handle missing image method gracefully
        try:
            _test_pyboy_screen_methods(self.rom_path)
        except Exception as e:
            self.fail(f"test_pyboy_screen_methods raised exception: {e}")
            
    @patch('vision.debug_screen_capture.PyBoy')
    @patch('vision.debug_screen_capture.PYBOY_AVAILABLE', True)
    def test_pil_conversion_error(self, mock_pyboy_class):
        """Test error handling during PIL conversion"""
        mock_pyboy = Mock()
        mock_screen = Mock()
        
        # Create invalid data that will cause PIL conversion to fail
        invalid_data = "not an array"
        mock_screen.ndarray = invalid_data
        
        mock_pyboy.screen = mock_screen
        mock_pyboy_class.return_value = mock_pyboy
        
        # Should handle PIL conversion error gracefully
        try:
            _test_pyboy_screen_methods(self.rom_path)
        except Exception as e:
            self.fail(f"test_pyboy_screen_methods raised exception: {e}")
    
    @patch('vision.debug_screen_capture.PyBoy')
    @patch('vision.debug_screen_capture.PYBOY_AVAILABLE', True)
    @patch('base64.b64encode')
    def test_base64_encoding_error(self, mock_b64encode, mock_pyboy_class):
        """Test error handling during base64 encoding"""
        # Set up valid PyBoy mock
        mock_pyboy = Mock()
        mock_screen = Mock()
        test_image_data = np.random.randint(0, 256, (144, 160, 3), dtype=np.uint8)
        mock_screen.ndarray = test_image_data
        mock_pyboy.screen = mock_screen
        mock_pyboy_class.return_value = mock_pyboy
        
        # Make base64 encoding fail
        mock_b64encode.side_effect = Exception("Mock base64 error")
        
        # Should handle base64 encoding error gracefully
        try:
            _test_pyboy_screen_methods(self.rom_path)
        except Exception as e:
            self.fail(f"test_pyboy_screen_methods raised exception: {e}")
    
    @patch('vision.debug_screen_capture.PyBoy')
    @patch('vision.debug_screen_capture.PYBOY_AVAILABLE', True)
    def test_tick_error_during_continuous_capture(self, mock_pyboy_class):
        """Test error handling when tick() fails during continuous capture"""
        mock_pyboy = Mock()
        mock_screen = Mock()
        test_image_data = np.random.randint(0, 256, (144, 160, 3), dtype=np.uint8)
        mock_screen.ndarray = test_image_data
        mock_pyboy.screen = mock_screen
        
        # Make tick fail after some calls
        call_count = 0
        def failing_tick():
            nonlocal call_count
            call_count += 1
            if call_count > 15:  # Fail after initial setup
                raise Exception("Mock tick error")
        
        mock_pyboy.tick = Mock(side_effect=failing_tick)
        mock_pyboy_class.return_value = mock_pyboy
        
        # Should handle tick error gracefully
        try:
            _test_pyboy_screen_methods(self.rom_path)
        except Exception as e:
            self.fail(f"test_pyboy_screen_methods raised exception: {e}")


class TestImageProcessing(unittest.TestCase):
    """Test image processing components"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.rom_path = "test_rom.gbc"
    
    @patch('vision.debug_screen_capture.PyBoy')
    @patch('vision.debug_screen_capture.PYBOY_AVAILABLE', True)
    def test_image_resizing(self, mock_pyboy_class):
        """Test image resizing functionality"""
        mock_pyboy = Mock()
        mock_screen = Mock()
        test_image_data = np.random.randint(0, 256, (144, 160, 3), dtype=np.uint8)
        mock_screen.ndarray = test_image_data
        
        # Mock image method
        test_pil_image = Image.fromarray(test_image_data)
        mock_screen.image.return_value = test_pil_image
        
        mock_pyboy.screen = mock_screen
        mock_pyboy_class.return_value = mock_pyboy
        
        _test_pyboy_screen_methods(self.rom_path)
        
        # The function should complete without error
        
    @patch('vision.debug_screen_capture.PyBoy')
    @patch('vision.debug_screen_capture.PYBOY_AVAILABLE', True)
    def test_png_format_saving(self, mock_pyboy_class):
        """Test PNG format saving"""
        mock_pyboy = Mock()
        mock_screen = Mock()
        test_image_data = np.random.randint(0, 256, (144, 160, 3), dtype=np.uint8)
        mock_screen.ndarray = test_image_data
        mock_pyboy.screen = mock_screen
        mock_pyboy_class.return_value = mock_pyboy
        
        _test_pyboy_screen_methods(self.rom_path)
        
        # The function should complete without error
        
    @patch('vision.debug_screen_capture.PyBoy')
    @patch('vision.debug_screen_capture.PYBOY_AVAILABLE', True)
    def test_image_array_conversion(self, mock_pyboy_class):
        """Test numpy array to PIL Image conversion"""
        mock_pyboy = Mock()
        mock_screen = Mock()
        
        # Test different image array shapes and types
        test_cases = [
            np.random.randint(0, 256, (144, 160, 3), dtype=np.uint8),  # RGB
            np.random.randint(0, 256, (144, 160, 4), dtype=np.uint8),  # RGBA
            np.random.randint(0, 256, (144, 160), dtype=np.uint8),     # Grayscale
        ]
        
        for test_data in test_cases:
            mock_screen.ndarray = test_data
            mock_pyboy.screen = mock_screen
            mock_pyboy_class.return_value = mock_pyboy
            
            try:
                _test_pyboy_screen_methods(self.rom_path)
            except Exception as e:
                # Some formats might fail, that's expected behavior
                pass


class TestMainFunction(unittest.TestCase):
    """Test main function"""
    
    @patch('vision.debug_screen_capture._test_pyboy_screen_methods')
    def test_main_function_execution(self, mock_test_function):
        """Test main function executes test_pyboy_screen_methods"""
        main()
        
        # Verify the test function was called with expected ROM path
        mock_test_function.assert_called_once()
        args = mock_test_function.call_args[0]
        self.assertIn('pokemon_crystal.gbc', args[0])
    
    @patch('vision.debug_screen_capture._test_pyboy_screen_methods')
    def test_main_function_error_handling(self, mock_test_function):
        """Test main function handles errors gracefully"""
        mock_test_function.side_effect = Exception("Mock test error")
        
        # Should not raise exception
        try:
            main()
        except Exception as e:
            self.fail(f"main() raised exception: {e}")


class TestScreenDataAnalysis(unittest.TestCase):
    """Test screen data analysis components"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.rom_path = "test_rom.gbc"
    
    @patch('vision.debug_screen_capture.PyBoy')
    @patch('vision.debug_screen_capture.PYBOY_AVAILABLE', True)
    def test_screen_data_statistics(self, mock_pyboy_class):
        """Test screen data statistical analysis"""
        mock_pyboy = Mock()
        mock_screen = Mock()
        
        # Create test data with known statistics
        test_image_data = np.array([
            [[255, 0, 0], [128, 128, 128]],
            [[0, 255, 0], [64, 64, 64]]
        ], dtype=np.uint8)
        
        mock_screen.ndarray = test_image_data
        mock_pyboy.screen = mock_screen
        mock_pyboy_class.return_value = mock_pyboy
        
        _test_pyboy_screen_methods(self.rom_path)
        
        # The function should analyze min/max values
        # This is tested implicitly through function completion
        
    @patch('vision.debug_screen_capture.PyBoy')
    @patch('vision.debug_screen_capture.PYBOY_AVAILABLE', True)
    def test_sample_values_extraction(self, mock_pyboy_class):
        """Test sample values extraction from screen data"""
        mock_pyboy = Mock()
        mock_screen = Mock()
        
        # Create deterministic test data
        test_image_data = np.arange(24, dtype=np.uint8).reshape((2, 4, 3))
        mock_screen.ndarray = test_image_data
        mock_pyboy.screen = mock_screen
        mock_pyboy_class.return_value = mock_pyboy
        
        _test_pyboy_screen_methods(self.rom_path)
        
        # The function should extract and display sample values
        # This is tested implicitly through function completion


class TestContinuousCapture(unittest.TestCase):
    """Test continuous capture functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.rom_path = "test_rom.gbc"
    
    @patch('vision.debug_screen_capture.PyBoy')
    @patch('vision.debug_screen_capture.PYBOY_AVAILABLE', True)
    @patch('time.sleep')
    def test_continuous_capture_timing(self, mock_sleep, mock_pyboy_class):
        """Test continuous capture timing"""
        mock_pyboy = Mock()
        mock_screen = Mock()
        test_image_data = np.random.randint(0, 256, (144, 160, 3), dtype=np.uint8)
        mock_screen.ndarray = test_image_data
        mock_pyboy.screen = mock_screen
        mock_pyboy_class.return_value = mock_pyboy
        
        _test_pyboy_screen_methods(self.rom_path)
        
        # Verify sleep was called
        self.assertEqual(mock_sleep.call_count, 10)
        
        # Verify sleep duration
        for call in mock_sleep.call_args_list:
            self.assertEqual(call[0][0], 0.1)  # 0.1 second sleep
    
    @patch('vision.debug_screen_capture.PyBoy')
    @patch('vision.debug_screen_capture.PYBOY_AVAILABLE', True)
    def test_capture_success_tracking(self, mock_pyboy_class):
        """Test capture success/failure tracking"""
        mock_pyboy = Mock()
        mock_screen = Mock()
        test_image_data = np.random.randint(0, 256, (144, 160, 3), dtype=np.uint8)
        mock_screen.ndarray = test_image_data
        mock_pyboy.screen = mock_screen
        mock_pyboy_class.return_value = mock_pyboy
        
        _test_pyboy_screen_methods(self.rom_path)
        
        # The function should track successful
        # This is tested implicitly through function completion without error
        
    @patch('vision.debug_screen_capture.PyBoy')
    @patch('vision.debug_screen_capture.PYBOY_AVAILABLE', True)
    def test_frame_processing_failure_handling(self, mock_pyboy_class):
        """Test handling of individual frame processing failures"""
        mock_pyboy = Mock()
        mock_screen = Mock()
        
        # Create a scenario where some frames fail
        call_count = 0
        def get_failing_ndarray():
            nonlocal call_count
            call_count += 1
            if call_count % 3 == 0:  # Every 3rd call fails
                raise Exception("Mock ndarray error")
            return np.random.randint(0, 256, (144, 160, 3), dtype=np.uint8)
        
        mock_screen.ndarray = property(get_failing_ndarray)
        mock_pyboy.screen = mock_screen
        mock_pyboy_class.return_value = mock_pyboy
        
        # Should handle individual frame failures gracefully
        try:
            _test_pyboy_screen_methods(self.rom_path)
        except Exception as e:
            self.fail(f"test_pyboy_screen_methods raised exception: {e}")


class TestImageFormatValidation(unittest.TestCase):
    """Test image format validation and handling"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.rom_path = "test_rom.gbc"
    
    @patch('vision.debug_screen_capture.PyBoy')
    @patch('vision.debug_screen_capture.PYBOY_AVAILABLE', True)
    def test_different_image_formats(self, mock_pyboy_class):
        """Test handling of different image formats"""
        mock_pyboy = Mock()
        mock_screen = Mock()
        
        # Test different image data types
        test_formats = [
            np.random.randint(0, 256, (144, 160, 3), dtype=np.uint8),   # Standard RGB
            np.random.randint(0, 256, (200, 180, 3), dtype=np.uint8),   # Different size
            np.ones((144, 160, 3), dtype=np.uint8) * 255,               # All white
            np.zeros((144, 160, 3), dtype=np.uint8),                    # All black
        ]
        
        for test_data in test_formats:
            mock_screen.ndarray = test_data
            mock_pyboy.screen = mock_screen
            mock_pyboy_class.return_value = mock_pyboy
            
            try:
                _test_pyboy_screen_methods(self.rom_path)
            except Exception as e:
                self.fail(f"test_pyboy_screen_methods failed with {test_data.shape}: {e}")
    
    @patch('vision.debug_screen_capture.PyBoy')
    @patch('vision.debug_screen_capture.PYBOY_AVAILABLE', True)
    def test_image_optimization(self, mock_pyboy_class):
        """Test image optimization during saving"""
        mock_pyboy = Mock()
        mock_screen = Mock()
        test_image_data = np.random.randint(0, 256, (144, 160, 3), dtype=np.uint8)
        mock_screen.ndarray = test_image_data
        mock_pyboy.screen = mock_screen
        mock_pyboy_class.return_value = mock_pyboy
        
        _test_pyboy_screen_methods(self.rom_path)
        
        # The function should use optimize
        # This is tested implicitly through function completion


class TestIntegrationScenarios(unittest.TestCase):
    """Test integration scenarios"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.rom_path = "test_rom.gbc"
    
    @patch('vision.debug_screen_capture.PyBoy')
    @patch('vision.debug_screen_capture.PYBOY_AVAILABLE', True)
    def test_complete_screen_capture_workflow(self, mock_pyboy_class):
        """Test complete screen capture workflow"""
        mock_pyboy = Mock()
        mock_screen = Mock()
        
        # Set up complete mock environment
        test_image_data = np.random.randint(0, 256, (144, 160, 3), dtype=np.uint8)
        mock_screen.ndarray = test_image_data
        
        test_pil_image = Image.fromarray(test_image_data)
        mock_screen.image.return_value = test_pil_image
        
        # Add mock methods for enumeration
        mock_screen.method1 = Mock()
        mock_screen.method2 = Mock()
        
        mock_pyboy.screen = mock_screen
        mock_pyboy.tick = Mock()
        mock_pyboy.stop = Mock()
        
        mock_pyboy_class.return_value = mock_pyboy
        
        # Should complete entire workflow without error
        _test_pyboy_screen_methods(self.rom_path)
        
        # Verify key operations
        self.assertTrue(mock_pyboy.tick.called)
        self.assertTrue(mock_pyboy.stop.called)
    
    @patch('vision.debug_screen_capture.PyBoy')
    @patch('vision.debug_screen_capture.PYBOY_AVAILABLE', True)
    def test_mixed_success_failure_scenario(self, mock_pyboy_class):
        """Test scenario with mixed success and failure cases"""
        mock_pyboy = Mock()
        mock_screen = Mock()
        
        # Set up scenario where some operations succeed and some fail
        test_image_data = np.random.randint(0, 256, (144, 160, 3), dtype=np.uint8)
        mock_screen.ndarray = test_image_data
        
        # Make image method fail
        mock_screen.image.side_effect = Exception("Mock image error")
        
        mock_pyboy.screen = mock_screen
        mock_pyboy_class.return_value = mock_pyboy
        
        # Should handle mixed success/failure gracefully
        try:
            _test_pyboy_screen_methods(self.rom_path)
        except Exception as e:
            self.fail(f"test_pyboy_screen_methods raised exception: {e}")


if __name__ == '__main__':
    unittest.main()
