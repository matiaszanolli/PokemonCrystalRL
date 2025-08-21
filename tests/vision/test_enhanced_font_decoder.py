#!/usr/bin/env python3
"""
Test suite for the enhanced font decoder module.
"""

import unittest
import numpy as np
import os
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
import cv2

from vision.enhanced_font_decoder import ROMFontDecoder


class TestROMFontDecoder(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        # Create a mock decoder without loading actual ROM
        with patch('vision.enhanced_font_decoder.PokemonCrystalFontExtractor'):
            self.decoder = ROMFontDecoder()
    
    def tearDown(self):
        """Clean up test fixtures"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test initialization without any paths"""
        with patch('vision.enhanced_font_decoder.PokemonCrystalFontExtractor'):
            decoder = ROMFontDecoder()
        
        self.assertIsInstance(decoder.font_templates, dict)
        self.assertIsInstance(decoder.recognition_stats, dict)
        self.assertIsInstance(decoder.template_cache, dict)
        self.assertEqual(decoder.cache_size, 1000)
    
    def test_initialization_with_template_path(self):
        """Test initialization with template path"""
        template_path = os.path.join(self.temp_dir, "test_templates.npz")
        
        # Create mock template file
        mock_templates = {'A': np.ones((8, 8), dtype=np.uint8) * 255}
        np.savez_compressed(template_path, **mock_templates)
        
        with patch('vision.enhanced_font_decoder.PokemonCrystalFontExtractor') as mock_extractor:
            mock_instance = MagicMock()
            mock_instance.load_font_templates.return_value = mock_templates
            mock_extractor.return_value = mock_instance
            
            decoder = ROMFontDecoder(template_path=template_path)
            
        self.assertEqual(decoder.template_path, template_path)
    
    def test_character_recognition(self):
        """Test basic character recognition"""
        test_char = np.zeros((8, 8), dtype=np.uint8)
        test_char[2:6, 2:6] = 255  # Simple box pattern
        
        char, confidence = self.decoder.recognize_character(test_char)
        
        self.assertIsInstance(char, str)
        self.assertIsInstance(confidence, float)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
    
    def test_text_region_decoding(self):
        """Test text region decoding"""
        # Create a simple 2x4 character test region
        test_region = np.zeros((16, 32), dtype=np.uint8)
        test_region[0:8, 0:8] = 255  # First character
        test_region[0:8, 8:16] = 255  # Second character
        
        text = self.decoder.decode_text_region(test_region)
        self.assertIsInstance(text, str)
    
    def test_error_handling(self):
        """Test error handling"""
        invalid_inputs = [
            None, 
            np.array([]),
            np.zeros((4, 4)),
            np.zeros((8, 8, 4))
        ]
        
        for invalid in invalid_inputs:
            char, conf = self.decoder.recognize_character(invalid)
            # Invalid inputs get normalized to zeros, which matches space character
            self.assertEqual(char, ' ')
            # Confidence should be high since zeros perfectly match space template
            self.assertGreaterEqual(conf, 0.9)
    
    def test_gbc_palette_integration(self):
        """Test Game Boy Color palette integration"""
        with patch('vision.enhanced_font_decoder.GBC_PALETTE_AVAILABLE', True):
            # Mock GBC palette with complete response
            mock_palette = MagicMock()
            mock_palette.analyze_text_region_colors.return_value = {
                'text_style': 'dark_on_light',
                'mean_brightness': 150,
                'is_high_contrast': True,  # Add missing key
                'unique_colors': 4,        # Add missing key
                'detected_palette': 'default'  # Add missing key
            }
            mock_palette.enhance_template_for_lighting.return_value = np.zeros((8, 8))
            
            self.decoder.gbc_palette = mock_palette
            
            region = np.zeros((8, 8), dtype=np.uint8)
            result = self.decoder.decode_text_region_with_palette(region)
            
            self.assertIsInstance(result, str)
            mock_palette.analyze_text_region_colors.assert_called_once()


    def test_cache_management(self):
        """Test cache management"""
        # Fill cache beyond capacity
        for i in range(self.decoder.cache_size + 10):
            tile = np.full((8, 8), i % 256, dtype=np.uint8)
            self.decoder._cache_character_result(tile, chr(65 + i % 26), 0.9)
        
        # Cache should not exceed capacity
        self.assertLessEqual(len(self.decoder.template_cache), self.decoder.cache_size)
        
        # Clear cache
        self.decoder.clear_cache()
        self.assertEqual(len(self.decoder.template_cache), 0)


if __name__ == '__main__':
    unittest.main()
