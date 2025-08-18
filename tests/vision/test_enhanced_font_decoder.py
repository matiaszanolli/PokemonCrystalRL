"""
Comprehensive unit tests for the Enhanced Font Decoder module.
Tests ROM-based font decoding, template matching, caching, and Game Boy Color palette integration.
"""
import unittest
import numpy as np
import os
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Tuple, Optional
import cv2

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from vision.enhanced_font_decoder import ROMFontDecoder, test_rom_font_decoder


class TestROMFontDecoder(unittest.TestCase):
    """Test ROMFontDecoder class"""
    
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
        """Test decoder initialization"""
        with patch('vision.enhanced_font_decoder.PokemonCrystalFontExtractor'):
            decoder = ROMFontDecoder()
            
        self.assertIsInstance(decoder.font_templates, dict)
        self.assertIsInstance(decoder.font_variations, dict)
        self.assertIsInstance(decoder.recognition_stats, dict)
        self.assertIsInstance(decoder.template_cache, dict)
        self.assertIsInstance(decoder.region_cache, dict)
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
        
    def test_initialization_with_rom_path(self):
        """Test initialization with ROM path"""
        rom_path = "/test/rom.gbc"
        
        with patch('vision.enhanced_font_decoder.PokemonCrystalFontExtractor'):
            decoder = ROMFontDecoder(rom_path=rom_path)
            
        self.assertEqual(decoder.rom_path, rom_path)


class TestFallbackTemplateCreation(unittest.TestCase):
    """Test fallback template creation methods"""
    
    def setUp(self):
        """Set up test fixtures"""
        with patch('vision.enhanced_font_decoder.PokemonCrystalFontExtractor'):
            self.decoder = ROMFontDecoder()
            
    def test_create_fallback_templates(self):
        """Test creating fallback templates"""
        self.decoder._create_fallback_templates()
        
        expected_chars = {' ', 'A', 'B', 'O', 'K', 'E', '!', '?'}
        self.assertTrue(expected_chars.issubset(set(self.decoder.font_templates.keys())))
        
        # Check template shapes
        for char in expected_chars:
            self.assertEqual(self.decoder.font_templates[char].shape, (8, 8))
            self.assertEqual(self.decoder.font_templates[char].dtype, np.uint8)
            
    def test_create_letter_a(self):
        """Test creating letter 'A' template"""
        template = self.decoder._create_letter_a()
        
        self.assertEqual(template.shape, (8, 8))
        self.assertEqual(template.dtype, np.uint8)
        # Should have some non-zero pixels
        self.assertGreater(np.sum(template > 0), 0)
        
    def test_create_letter_templates(self):
        """Test creating individual letter templates"""
        templates = {
            'B': self.decoder._create_letter_b(),
            'O': self.decoder._create_letter_o(),
            'K': self.decoder._create_letter_k(),
            'E': self.decoder._create_letter_e()
        }
        
        for char, template in templates.items():
            self.assertEqual(template.shape, (8, 8), f"Template {char} wrong shape")
            self.assertEqual(template.dtype, np.uint8, f"Template {char} wrong dtype")
            self.assertGreater(np.sum(template > 0), 0, f"Template {char} is empty")
            
    def test_create_punctuation_templates(self):
        """Test creating punctuation templates"""
        exclamation = self.decoder._create_exclamation()
        question = self.decoder._create_question()
        
        self.assertEqual(exclamation.shape, (8, 8))
        self.assertEqual(question.shape, (8, 8))
        self.assertGreater(np.sum(exclamation > 0), 0)
        self.assertGreater(np.sum(question > 0), 0)


class TestTileNormalization(unittest.TestCase):
    """Test tile normalization and preprocessing"""
    
    def setUp(self):
        """Set up test fixtures"""
        with patch('vision.enhanced_font_decoder.PokemonCrystalFontExtractor'):
            self.decoder = ROMFontDecoder()
            
    def test_normalize_tile_correct_size(self):
        """Test normalizing tile with correct size"""
        tile = np.random.randint(0, 256, (8, 8), dtype=np.uint8)
        normalized = self.decoder._normalize_tile(tile)
        
        self.assertEqual(normalized.shape, (8, 8))
        self.assertEqual(normalized.dtype, np.uint8)
        self.assertTrue(np.all((normalized == 0) | (normalized == 255)))
        
    def test_normalize_tile_wrong_size(self):
        """Test normalizing tile with wrong size"""
        tile = np.random.randint(0, 256, (16, 16), dtype=np.uint8)
        normalized = self.decoder._normalize_tile(tile)
        
        self.assertEqual(normalized.shape, (8, 8))
        self.assertEqual(normalized.dtype, np.uint8)
        
    def test_normalize_tile_binary_input(self):
        """Test normalizing tile with binary input"""
        tile = np.random.randint(0, 2, (8, 8), dtype=np.uint8)
        normalized = self.decoder._normalize_tile(tile)
        
        self.assertEqual(normalized.shape, (8, 8))
        self.assertEqual(normalized.dtype, np.uint8)
        
    def test_normalize_tile_threshold(self):
        """Test tile normalization threshold"""
        # Test with values above and below threshold
        tile = np.full((8, 8), 200, dtype=np.uint8)  # Above 128
        normalized = self.decoder._normalize_tile(tile)
        self.assertTrue(np.all(normalized == 255))
        
        tile = np.full((8, 8), 50, dtype=np.uint8)   # Below 128
        normalized = self.decoder._normalize_tile(tile)
        self.assertTrue(np.all(normalized == 0))


class TestHashingAndCaching(unittest.TestCase):
    """Test hashing and caching functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        with patch('vision.enhanced_font_decoder.PokemonCrystalFontExtractor'):
            self.decoder = ROMFontDecoder()
            
    def test_hash_tile(self):
        """Test tile hashing"""
        tile = np.random.randint(0, 256, (8, 8), dtype=np.uint8)
        hash1 = self.decoder._hash_tile(tile)
        hash2 = self.decoder._hash_tile(tile)
        
        # Same tile should produce same hash
        self.assertEqual(hash1, hash2)
        self.assertIsInstance(hash1, str)
        self.assertEqual(len(hash1), 32)  # MD5 hash length
        
    def test_hash_tile_different_tiles(self):
        """Test hashing different tiles produces different hashes"""
        tile1 = np.zeros((8, 8), dtype=np.uint8)
        tile2 = np.ones((8, 8), dtype=np.uint8)
        
        hash1 = self.decoder._hash_tile(tile1)
        hash2 = self.decoder._hash_tile(tile2)
        
        self.assertNotEqual(hash1, hash2)
        
    def test_hash_region(self):
        """Test region hashing"""
        region = np.random.randint(0, 256, (16, 32), dtype=np.uint8)
        hash1 = self.decoder._hash_region(region)
        hash2 = self.decoder._hash_region(region)
        
        self.assertEqual(hash1, hash2)
        self.assertIsInstance(hash1, str)
        self.assertEqual(len(hash1), 32)
        
    def test_hash_region_color_input(self):
        """Test region hashing with color input"""
        region = np.random.randint(0, 256, (16, 32, 3), dtype=np.uint8)
        hash_val = self.decoder._hash_region(region)
        
        self.assertIsInstance(hash_val, str)
        self.assertEqual(len(hash_val), 32)
        
    def test_cache_character_result(self):
        """Test caching character recognition results"""
        tile = np.random.randint(0, 256, (8, 8), dtype=np.uint8)
        
        self.decoder._cache_character_result(tile, 'A', 0.95)
        
        cached_result = self.decoder._get_cached_character(tile)
        self.assertIsNotNone(cached_result)
        self.assertEqual(cached_result[0], 'A')
        self.assertEqual(cached_result[1], 0.95)
        
    def test_cache_lru_eviction(self):
        """Test LRU cache eviction"""
        # Fill cache beyond capacity
        for i in range(self.decoder.cache_size + 10):
            tile = np.full((8, 8), i % 256, dtype=np.uint8)
            self.decoder._cache_character_result(tile, chr(65 + i % 26), 0.9)
        
        # Cache should not exceed capacity
        self.assertLessEqual(len(self.decoder.template_cache), self.decoder.cache_size)
        
    def test_cache_region_result(self):
        """Test caching text region results"""
        region = np.random.randint(0, 256, (16, 32), dtype=np.uint8)
        
        self.decoder._cache_region_result(region, "HELLO")
        
        cached_result = self.decoder._get_cached_region(region)
        self.assertEqual(cached_result, "HELLO")
        
    def test_clear_cache(self):
        """Test clearing caches"""
        # Add some items to cache
        tile = np.random.randint(0, 256, (8, 8), dtype=np.uint8)
        region = np.random.randint(0, 256, (16, 32), dtype=np.uint8)
        
        self.decoder._cache_character_result(tile, 'A', 0.9)
        self.decoder._cache_region_result(region, "TEST")
        
        # Clear cache
        self.decoder.clear_cache()
        
        self.assertEqual(len(self.decoder.template_cache), 0)
        self.assertEqual(len(self.decoder.region_cache), 0)
        
    def test_get_cache_stats(self):
        """Test getting cache statistics"""
        # Perform some operations
        tile = np.random.randint(0, 256, (8, 8), dtype=np.uint8)
        self.decoder._cache_character_result(tile, 'A', 0.9)
        self.decoder._get_cached_character(tile)
        
        stats = self.decoder.get_cache_stats()
        
        self.assertIsInstance(stats, dict)
        self.assertIn('template_cache_size', stats)
        self.assertIn('cache_hits', stats)
        self.assertIn('cache_misses', stats)
        self.assertIn('hit_rate', stats)


class TestMatchScoring(unittest.TestCase):
    """Test character matching and scoring"""
    
    def setUp(self):
        """Set up test fixtures"""
        with patch('vision.enhanced_font_decoder.PokemonCrystalFontExtractor'):
            self.decoder = ROMFontDecoder()
        # Add some test templates
        self.decoder.font_templates['A'] = np.eye(8, dtype=np.uint8) * 255
        self.decoder.font_templates['B'] = np.ones((8, 8), dtype=np.uint8) * 255
        
    def test_calculate_match_score_identical(self):
        """Test match scoring with identical tiles"""
        template = np.eye(8, dtype=np.uint8) * 255
        tile = template.copy()
        
        score = self.decoder._calculate_match_score(tile, template)
        
        self.assertGreater(score, 0.8)  # Should be high confidence
        self.assertLessEqual(score, 1.0)
        
    def test_calculate_match_score_different(self):
        """Test match scoring with different tiles"""
        template = np.eye(8, dtype=np.uint8) * 255
        tile = np.ones((8, 8), dtype=np.uint8) * 255
        
        score = self.decoder._calculate_match_score(tile, template)
        
        self.assertGreater(score, 0.0)
        self.assertLess(score, 1.0)
        
    def test_calculate_match_score_empty_tiles(self):
        """Test match scoring with empty tiles"""
        template = np.zeros((8, 8), dtype=np.uint8)
        tile = np.zeros((8, 8), dtype=np.uint8)
        
        score = self.decoder._calculate_match_score(tile, template)
        
        self.assertGreater(score, 0.0)  # Should have some score for matching empty
        
    def test_calculate_match_score_cv2_error_handling(self):
        """Test match scoring with OpenCV error handling"""
        template = np.ones((8, 8), dtype=np.uint8) * 255
        tile = np.ones((8, 8), dtype=np.uint8) * 255
        
        with patch('cv2.matchTemplate', side_effect=cv2.error("Mock error")):
            score = self.decoder._calculate_match_score(tile, template)
            
        # Should still return a score despite CV2 error
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)


class TestCharacterRecognition(unittest.TestCase):
    """Test character recognition functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        with patch('vision.enhanced_font_decoder.PokemonCrystalFontExtractor'):
            self.decoder = ROMFontDecoder()
        # Add test templates
        self.decoder.font_templates['A'] = np.eye(8, dtype=np.uint8) * 255
        self.decoder.font_templates['B'] = np.ones((8, 8), dtype=np.uint8) * 255
        
    def test_recognize_character_no_templates(self):
        """Test character recognition with no templates"""
        decoder = ROMFontDecoder.__new__(ROMFontDecoder)
        decoder.font_templates = {}
        decoder.recognition_stats = {
            'total_attempts': 0,
            'successful_matches': 0,
            'failed_matches': 0,
            'confidence_scores': [],
            'cache_hits': 0,
            'cache_misses': 0
        }
        decoder.template_cache = {}
        
        tile = np.random.randint(0, 256, (8, 8), dtype=np.uint8)
        char, confidence = decoder.recognize_character(tile)
        
        self.assertEqual(char, '?')
        self.assertEqual(confidence, 0.0)
        
    def test_recognize_character_with_cache(self):
        """Test character recognition using cache"""
        tile = np.eye(8, dtype=np.uint8) * 255
        
        # First recognition (cache miss)
        char1, conf1 = self.decoder.recognize_character(tile)
        
        # Second recognition (cache hit)
        char2, conf2 = self.decoder.recognize_character(tile)
        
        self.assertEqual(char1, char2)
        self.assertEqual(conf1, conf2)
        
    def test_recognize_character_confidence_threshold(self):
        """Test character recognition with confidence threshold"""
        tile = np.random.randint(0, 50, (8, 8), dtype=np.uint8)  # Low match
        
        char, confidence = self.decoder.recognize_character(tile, min_confidence=0.8)
        
        if confidence < 0.8:
            self.assertEqual(char, '?')
        
    def test_recognize_character_statistics(self):
        """Test character recognition statistics tracking"""
        initial_attempts = self.decoder.recognition_stats['total_attempts']
        
        tile = np.eye(8, dtype=np.uint8) * 255
        self.decoder.recognize_character(tile)
        
        self.assertEqual(
            self.decoder.recognition_stats['total_attempts'], 
            initial_attempts + 1
        )
        self.assertTrue(len(self.decoder.recognition_stats['confidence_scores']) > 0)


class TestTextRegionDecoding(unittest.TestCase):
    """Test text region decoding functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        with patch('vision.enhanced_font_decoder.PokemonCrystalFontExtractor'):
            self.decoder = ROMFontDecoder()
        # Add test templates
        self.decoder.font_templates['A'] = np.eye(8, dtype=np.uint8) * 255
        self.decoder.font_templates['B'] = np.ones((8, 8), dtype=np.uint8) * 255
        self.decoder.font_templates[' '] = np.zeros((8, 8), dtype=np.uint8)
        
    def test_decode_text_region_none_input(self):
        """Test decoding with None input"""
        result = self.decoder.decode_text_region(None)
        self.assertEqual(result, "")
        
    def test_decode_text_region_invalid_input(self):
        """Test decoding with invalid input"""
        # Non-ndarray input
        result = self.decoder.decode_text_region("invalid")
        self.assertEqual(result, "")
        
        # Empty array
        result = self.decoder.decode_text_region(np.array([]))
        self.assertEqual(result, "")
        
        # Wrong dimensions
        result = self.decoder.decode_text_region(np.random.rand(5))
        self.assertEqual(result, "")
        
    def test_decode_text_region_too_small(self):
        """Test decoding region too small for characters"""
        small_region = np.random.randint(0, 256, (4, 4), dtype=np.uint8)
        result = self.decoder.decode_text_region(small_region)
        self.assertEqual(result, "")
        
    def test_decode_text_region_color_input(self):
        """Test decoding color region"""
        # Create 3-channel color region
        color_region = np.random.randint(0, 256, (16, 16, 3), dtype=np.uint8)
        result = self.decoder.decode_text_region(color_region)
        
        # Should not crash and return string
        self.assertIsInstance(result, str)
        
    def test_decode_text_region_invalid_color_channels(self):
        """Test decoding with invalid color channels"""
        # Wrong number of channels
        invalid_region = np.random.randint(0, 256, (16, 16, 4), dtype=np.uint8)
        result = self.decoder.decode_text_region(invalid_region)
        self.assertEqual(result, "")
        
    def test_decode_text_region_cv2_error(self):
        """Test decoding with OpenCV color conversion error"""
        region = np.random.randint(0, 256, (16, 16, 3), dtype=np.uint8)
        
        with patch('cv2.cvtColor', side_effect=cv2.error("Mock error")):
            result = self.decoder.decode_text_region(region)
            
        self.assertEqual(result, "")
        
    def test_decode_text_region_single_character(self):
        """Test decoding single character region"""
        # Create 8x8 region with 'A' pattern
        region = np.eye(8, dtype=np.uint8) * 255
        result = self.decoder.decode_text_region(region)
        
        self.assertIsInstance(result, str)
        
    def test_decode_text_region_multi_character(self):
        """Test decoding multi-character region"""
        # Create 8x16 region (1 row, 2 characters)
        region = np.zeros((8, 16), dtype=np.uint8)
        region[:, 0:8] = np.eye(8, dtype=np.uint8) * 255  # 'A' pattern
        region[:, 8:16] = np.ones((8, 8), dtype=np.uint8) * 255  # 'B' pattern
        
        result = self.decoder.decode_text_region(region)
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) >= 0)
        
    def test_decode_text_region_multi_line(self):
        """Test decoding multi-line region"""
        # Create 16x8 region (2 rows, 1 character each)
        region = np.zeros((16, 8), dtype=np.uint8)
        region[0:8, :] = np.eye(8, dtype=np.uint8) * 255    # First row
        region[8:16, :] = np.ones((8, 8), dtype=np.uint8) * 255  # Second row
        
        result = self.decoder.decode_text_region(region)
        self.assertIsInstance(result, str)
        
    def test_decode_text_region_empty_lines(self):
        """Test handling of empty lines in decoding"""
        # Create region that would produce empty lines
        region = np.zeros((16, 16), dtype=np.uint8)
        # Only fill first character position
        region[0:8, 0:8] = np.eye(8, dtype=np.uint8) * 255
        
        result = self.decoder.decode_text_region(region)
        # Should not have trailing newlines or empty strings
        self.assertFalse(result.endswith('\n'))


class TestGameBoyColorPaletteIntegration(unittest.TestCase):
    """Test Game Boy Color palette integration"""
    
    def setUp(self):
        """Set up test fixtures"""
        with patch('vision.enhanced_font_decoder.PokemonCrystalFontExtractor'):
            self.decoder = ROMFontDecoder()
            
    def test_decode_text_region_with_palette_no_gbc(self):
        """Test palette decoding when GBC support not available"""
        self.decoder.gbc_palette = None
        
        region = np.zeros((8, 8), dtype=np.uint8)
        result = self.decoder.decode_text_region_with_palette(region)
        
        self.assertIsInstance(result, str)
        
    @patch('vision.enhanced_font_decoder.GBC_PALETTE_AVAILABLE', True)
    def test_decode_text_region_with_palette_available(self):
        """Test palette decoding when GBC support is available"""
        # Mock GBC palette
        mock_palette = MagicMock()
        mock_palette.analyze_text_region_colors.return_value = {
            'is_high_contrast': True,
            'detected_palette': 'text_white',
            'text_style': 'light_on_dark',
            'mean_brightness': 150
        }
        mock_palette.enhance_template_for_lighting.return_value = np.zeros((8, 8))
        
        self.decoder.gbc_palette = mock_palette
        
        region = np.zeros((8, 8), dtype=np.uint8)
        result = self.decoder.decode_text_region_with_palette(region)
        
        self.assertIsInstance(result, str)
        mock_palette.analyze_text_region_colors.assert_called_once()
        
    def test_decode_text_region_with_palette_invalid_input(self):
        """Test palette decoding with invalid input"""
        # Test all the same invalid inputs as regular decoding
        invalid_inputs = [None, "invalid", np.array([]), np.random.rand(5)]
        
        for invalid_input in invalid_inputs:
            result = self.decoder.decode_text_region_with_palette(invalid_input)
            self.assertEqual(result, "")
            
    @patch('vision.enhanced_font_decoder.GBC_PALETTE_AVAILABLE', True)
    def test_decode_text_region_with_palette_low_contrast(self):
        """Test palette decoding with low contrast enhancement"""
        mock_palette = MagicMock()
        mock_palette.analyze_text_region_colors.return_value = {
            'is_high_contrast': False,
            'detected_palette': 'text_black',
            'text_style': 'dark_on_light',
            'mean_brightness': 50  # Dark
        }
        enhanced_region = np.ones((8, 8), dtype=np.uint8) * 255
        mock_palette.enhance_template_for_lighting.return_value = enhanced_region
        
        self.decoder.gbc_palette = mock_palette
        
        region = np.zeros((8, 8), dtype=np.uint8)
        result = self.decoder.decode_text_region_with_palette(region)
        
        # Use ANY to handle numpy arrays in mock assertions
        from unittest.mock import ANY
        mock_palette.enhance_template_for_lighting.assert_called_with(ANY, 'dark')
        
    def test_get_palette_analysis_no_gbc(self):
        """Test getting palette analysis when GBC not available"""
        self.decoder.gbc_palette = None
        
        region = np.zeros((8, 8), dtype=np.uint8)
        result = self.decoder.get_palette_analysis(region)
        
        self.assertEqual(result, {})
        
    @patch('vision.enhanced_font_decoder.GBC_PALETTE_AVAILABLE', True)
    def test_get_palette_analysis_available(self):
        """Test getting palette analysis when GBC available"""
        mock_palette = MagicMock()
        expected_analysis = {'test': 'data'}
        mock_palette.analyze_text_region_colors.return_value = expected_analysis
        
        self.decoder.gbc_palette = mock_palette
        
        region = np.zeros((8, 8), dtype=np.uint8)
        result = self.decoder.get_palette_analysis(region)
        
        self.assertEqual(result, expected_analysis)
        
    def test_optimize_templates_for_palette_no_gbc(self):
        """Test template optimization when GBC not available"""
        self.decoder.gbc_palette = None
        
        self.decoder.optimize_templates_for_palette("test_palette")
        
        # Should not crash, just print warning
        self.assertEqual(len(self.decoder.font_variations), 0)
        
    @patch('vision.enhanced_font_decoder.GBC_PALETTE_AVAILABLE', True)  
    def test_optimize_templates_for_palette_unknown(self):
        """Test template optimization with unknown palette"""
        mock_palette = MagicMock()
        mock_palette.list_palettes.return_value = ['palette1', 'palette2']
        
        self.decoder.gbc_palette = mock_palette
        self.decoder.font_templates = {'A': np.eye(8, dtype=np.uint8)}
        
        self.decoder.optimize_templates_for_palette("unknown_palette")
        
        # Should not add variations for unknown palette
        self.assertNotIn("unknown_palette", self.decoder.font_variations)
        
    @patch('vision.enhanced_font_decoder.GBC_PALETTE_AVAILABLE', True)
    def test_optimize_templates_for_palette_success(self):
        """Test successful template optimization"""
        mock_palette = MagicMock()
        mock_palette.list_palettes.return_value = ['text_white']
        optimized_template = np.ones((8, 8), dtype=np.uint8) * 200
        mock_palette.create_color_aware_template.return_value = optimized_template
        
        self.decoder.gbc_palette = mock_palette
        self.decoder.font_templates = {'A': np.eye(8, dtype=np.uint8)}
        
        self.decoder.optimize_templates_for_palette("text_white")
        
        # Should add variations
        self.assertIn("text_white", self.decoder.font_variations)
        self.assertIn("A_text_white", self.decoder.font_variations["text_white"])


class TestStatisticsAndUtilities(unittest.TestCase):
    """Test statistics and utility functions"""
    
    def setUp(self):
        """Set up test fixtures"""
        with patch('vision.enhanced_font_decoder.PokemonCrystalFontExtractor'):
            self.decoder = ROMFontDecoder()
            
    def test_get_recognition_stats_empty(self):
        """Test getting recognition stats with no data"""
        stats = self.decoder.get_recognition_stats()
        
        self.assertIsInstance(stats, dict)
        self.assertEqual(stats['total_attempts'], 0)
        self.assertEqual(stats['successful_matches'], 0)
        self.assertEqual(stats['failed_matches'], 0)
        self.assertEqual(stats['success_rate'], 0.0)
        self.assertEqual(stats['average_confidence'], 0.0)
        
    def test_get_recognition_stats_with_data(self):
        """Test getting recognition stats with data"""
        # Manually add some stats
        self.decoder.recognition_stats['total_attempts'] = 10
        self.decoder.recognition_stats['successful_matches'] = 7
        self.decoder.recognition_stats['failed_matches'] = 3
        self.decoder.recognition_stats['confidence_scores'] = [0.8, 0.9, 0.7, 0.6, 0.95]
        
        stats = self.decoder.get_recognition_stats()
        
        self.assertEqual(stats['total_attempts'], 10)
        self.assertEqual(stats['success_rate'], 0.7)
        self.assertAlmostEqual(stats['average_confidence'], 0.79, places=1)
        self.assertEqual(stats['min_confidence'], 0.6)
        self.assertEqual(stats['max_confidence'], 0.95)
        
    def test_reset_stats(self):
        """Test resetting recognition statistics"""
        # Add some data
        self.decoder.recognition_stats['total_attempts'] = 10
        self.decoder.recognition_stats['confidence_scores'] = [0.8, 0.9]
        
        self.decoder.reset_stats()
        
        self.assertEqual(self.decoder.recognition_stats['total_attempts'], 0)
        self.assertEqual(len(self.decoder.recognition_stats['confidence_scores']), 0)
        
    def test_save_templates(self):
        """Test saving templates to file"""
        self.decoder.font_templates = {'A': np.eye(8, dtype=np.uint8)}
        
        temp_file = os.path.join(tempfile.gettempdir(), "test_templates.npz")
        
        try:
            result = self.decoder.save_templates(temp_file)
            
            self.assertTrue(result)
            self.assertTrue(os.path.exists(temp_file))
            
            # Verify file contents
            loaded_data = np.load(temp_file)
            self.assertIn('A', loaded_data.files)
            
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)
                
    def test_save_templates_error(self):
        """Test saving templates with I/O error"""
        self.decoder.font_templates = {'A': np.eye(8, dtype=np.uint8)}
        
        # Try to save to invalid path
        result = self.decoder.save_templates("/invalid/path/test.npz")
        
        self.assertFalse(result)
        
    def test_add_custom_template(self):
        """Test adding custom template"""
        template = np.ones((8, 8), dtype=np.uint8) * 200
        
        self.decoder.add_custom_template('X', template)
        
        self.assertIn('X', self.decoder.font_templates)
        # Template should be normalized
        self.assertEqual(self.decoder.font_templates['X'].dtype, np.uint8)
        
    def test_add_custom_template_wrong_shape(self):
        """Test adding custom template with wrong shape"""
        template = np.ones((16, 16), dtype=np.uint8)
        initial_count = len(self.decoder.font_templates)
        
        self.decoder.add_custom_template('Y', template)
        
        # Should not add template with wrong shape
        self.assertEqual(len(self.decoder.font_templates), initial_count)
        
    def test_preview_template_exists(self):
        """Test previewing existing template"""
        self.decoder.font_templates['A'] = np.eye(8, dtype=np.uint8) * 255
        
        # Should not raise exception
        try:
            self.decoder.preview_template('A')
        except Exception as e:
            self.fail(f"Preview template raised exception: {e}")
            
    def test_preview_template_not_exists(self):
        """Test previewing non-existent template"""
        # Should not raise exception
        try:
            self.decoder.preview_template('Z')
        except Exception as e:
            self.fail(f"Preview template raised exception: {e}")


class TestROMFontDecoderIntegration(unittest.TestCase):
    """Integration tests for ROM font decoder"""
    
    def setUp(self):
        """Set up integration test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up integration test fixtures"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            
    def test_complete_decoding_workflow(self):
        """Test complete decoding workflow"""
        # Mock extractor
        mock_extractor = MagicMock()
        mock_extractor.load_font_templates.return_value = {
            'A': np.eye(8, dtype=np.uint8) * 255,
            'B': np.ones((8, 8), dtype=np.uint8) * 255,
            ' ': np.zeros((8, 8), dtype=np.uint8)
        }
        
        with patch('vision.enhanced_font_decoder.PokemonCrystalFontExtractor', return_value=mock_extractor):
            decoder = ROMFontDecoder()
            
        # Test character recognition
        test_tile = np.eye(8, dtype=np.uint8) * 255
        char, confidence = decoder.recognize_character(test_tile)
        
        self.assertIsInstance(char, str)
        self.assertIsInstance(confidence, float)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
        
        # Test region decoding
        test_region = np.zeros((8, 16), dtype=np.uint8)
        test_region[:, 0:8] = np.eye(8) * 255  # Should match 'A'
        test_region[:, 8:16] = np.ones((8, 8)) * 255  # Should match 'B'
        
        decoded_text = decoder.decode_text_region(test_region)
        self.assertIsInstance(decoded_text, str)
        
        # Test statistics
        stats = decoder.get_recognition_stats()
        self.assertGreater(stats['total_attempts'], 0)
        
    def test_template_loading_fallback(self):
        """Test template loading with fallback"""
        # Mock extractor that returns empty templates
        mock_extractor = MagicMock()
        mock_extractor.load_font_templates.return_value = {}
        
        with patch('vision.enhanced_font_decoder.PokemonCrystalFontExtractor', return_value=mock_extractor):
            decoder = ROMFontDecoder()
            
        # Should have fallback templates
        self.assertGreater(len(decoder.font_templates), 0)
        self.assertIn('A', decoder.font_templates)
        self.assertIn(' ', decoder.font_templates)
        
    def test_test_rom_font_decoder_function(self):
        """Test the standalone test function"""
        # Should not raise exception
        try:
            with patch('vision.enhanced_font_decoder.ROMFontDecoder') as mock_decoder_class:
                mock_decoder = MagicMock()
                mock_decoder.font_templates = {'A': np.eye(8), 'O': np.ones((8, 8)), ' ': np.zeros((8, 8)), '!': np.eye(8)}
                mock_decoder.recognize_character.return_value = ('A', 0.95)
                mock_decoder.decode_text_region.return_value = "OK"
                mock_decoder.get_recognition_stats.return_value = {
                    'total_attempts': 3,
                    'success_rate': 1.0,
                    'average_confidence': 0.95
                }
                mock_decoder_class.return_value = mock_decoder
                
                test_rom_font_decoder()
                
        except Exception as e:
            self.fail(f"test_rom_font_decoder raised exception: {e}")


if __name__ == '__main__':
    unittest.main()
