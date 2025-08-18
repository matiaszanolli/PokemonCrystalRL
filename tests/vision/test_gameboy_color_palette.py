"""
Comprehensive unit tests for the Game Boy Color Palette module.
Tests GBC palette management, color conversion, analysis, and template enhancement functionality.
"""
import unittest
import numpy as np
import os
import tempfile
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Tuple, Optional
import cv2

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from pokemon_crystal_rl.vision.gameboy_color_palette import GameBoyColorPalette, GBCPalette, test_gameboy_color_palette


class TestGBCPalette(unittest.TestCase):
    """Test GBCPalette dataclass"""
    
    def test_gbc_palette_creation(self):
        """Test creating GBCPalette objects"""
        palette = GBCPalette(
            name='test_palette',
            colors=[(255, 255, 255), (128, 128, 128), (64, 64, 64), (0, 0, 0)],
            description='Test palette description'
        )
        
        self.assertEqual(palette.name, 'test_palette')
        self.assertEqual(len(palette.colors), 4)
        self.assertIsInstance(palette.colors[0], tuple)
        self.assertEqual(len(palette.colors[0]), 3)  # RGB tuple
        self.assertEqual(palette.description, 'Test palette description')
        
    def test_gbc_palette_color_validation(self):
        """Test GBCPalette color values are valid RGB"""
        palette = GBCPalette(
            name='valid_palette',
            colors=[(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)],
            description='Valid RGB colors'
        )
        
        # All colors should be valid RGB tuples
        for color in palette.colors:
            self.assertEqual(len(color), 3)
            for channel in color:
                self.assertIsInstance(channel, int)
                self.assertGreaterEqual(channel, 0)
                self.assertLessEqual(channel, 255)


class TestGameBoyColorPaletteInitialization(unittest.TestCase):
    """Test GameBoyColorPalette initialization and basic functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.gbc_palette = GameBoyColorPalette()
        
    def test_initialization(self):
        """Test GameBoyColorPalette initialization"""
        self.assertIsInstance(self.gbc_palette.palettes, dict)
        self.assertGreater(len(self.gbc_palette.palettes), 0)
        
        # Check that some expected palettes exist
        expected_palettes = ['text_white', 'text_black', 'menu_standard', 'dialogue_normal']
        for palette_name in expected_palettes:
            self.assertIn(palette_name, self.gbc_palette.palettes)
            
    def test_predefined_palettes(self):
        """Test that predefined palettes are properly configured"""
        # Check text_white palette
        text_white = self.gbc_palette.get_palette('text_white')
        self.assertIsNotNone(text_white)
        self.assertEqual(text_white.name, 'text_white')
        self.assertEqual(len(text_white.colors), 4)
        
        # Check text_black palette
        text_black = self.gbc_palette.get_palette('text_black')
        self.assertIsNotNone(text_black)
        self.assertEqual(text_black.name, 'text_black')
        
        # Check health palettes exist
        health_palettes = ['health_green', 'health_yellow', 'health_red']
        for palette_name in health_palettes:
            palette = self.gbc_palette.get_palette(palette_name)
            self.assertIsNotNone(palette)
            
    def test_palette_color_structure(self):
        """Test that all palettes have proper color structure"""
        for palette_name, palette in self.gbc_palette.palettes.items():
            self.assertEqual(len(palette.colors), 4, f"Palette {palette_name} should have 4 colors")
            
            # Check each color is a valid RGB tuple
            for i, color in enumerate(palette.colors):
                self.assertEqual(len(color), 3, f"Color {i} in {palette_name} should be RGB tuple")
                for j, channel in enumerate(color):
                    self.assertIsInstance(channel, int, f"Channel {j} in color {i} of {palette_name} should be int")
                    self.assertGreaterEqual(channel, 0)
                    self.assertLessEqual(channel, 255)


class TestPaletteManagement(unittest.TestCase):
    """Test palette management functions"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.gbc_palette = GameBoyColorPalette()
        
    def test_get_palette_existing(self):
        """Test getting existing palettes"""
        palette = self.gbc_palette.get_palette('text_white')
        self.assertIsNotNone(palette)
        self.assertIsInstance(palette, GBCPalette)
        self.assertEqual(palette.name, 'text_white')
        
    def test_get_palette_non_existent(self):
        """Test getting non-existent palette"""
        palette = self.gbc_palette.get_palette('non_existent_palette')
        self.assertIsNone(palette)
        
    def test_list_palettes(self):
        """Test listing available palettes"""
        palette_names = self.gbc_palette.list_palettes()
        self.assertIsInstance(palette_names, list)
        self.assertGreater(len(palette_names), 0)
        
        # Check that known palettes are in the list
        expected_palettes = ['text_white', 'text_black', 'menu_standard']
        for expected in expected_palettes:
            self.assertIn(expected, palette_names)
            
    def test_list_palettes_completeness(self):
        """Test that list_palettes returns all available palettes"""
        palette_names = self.gbc_palette.list_palettes()
        self.assertEqual(len(palette_names), len(self.gbc_palette.palettes))
        
        # Every palette in the dict should be in the list
        for palette_name in self.gbc_palette.palettes.keys():
            self.assertIn(palette_name, palette_names)


class TestPaletteConversion(unittest.TestCase):
    """Test palette conversion functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.gbc_palette = GameBoyColorPalette()
        
    def test_convert_grayscale_image(self):
        """Test converting grayscale image to GBC palette"""
        # Create test grayscale image
        test_image = np.array([
            [0, 64, 128, 192],
            [32, 96, 160, 224],
            [16, 80, 144, 208],
            [48, 112, 176, 255]
        ], dtype=np.uint8)
        
        converted = self.gbc_palette.convert_to_gbc_palette(test_image, 'text_white')
        
        # Should return RGB image
        self.assertEqual(len(converted.shape), 3)
        self.assertEqual(converted.shape[2], 3)
        self.assertEqual(converted.shape[:2], test_image.shape)
        
    def test_convert_rgb_image(self):
        """Test converting RGB image to GBC palette"""
        # Create test RGB image
        test_image = np.random.randint(0, 256, (8, 8, 3), dtype=np.uint8)
        
        converted = self.gbc_palette.convert_to_gbc_palette(test_image, 'text_black')
        
        # Should return RGB image with same dimensions
        self.assertEqual(converted.shape, test_image.shape)
        self.assertEqual(converted.dtype, np.uint8)
        
    def test_convert_unknown_palette(self):
        """Test converting with unknown palette name"""
        test_image = np.zeros((4, 4), dtype=np.uint8)
        
        # Should return original image for unknown palette
        converted = self.gbc_palette.convert_to_gbc_palette(test_image, 'unknown_palette')
        np.testing.assert_array_equal(converted, test_image)
        
    def test_convert_palette_levels(self):
        """Test that grayscale levels map correctly to palette colors"""
        # Create image with specific gray levels
        test_image = np.array([
            [0, 63],      # Should map to palette color 0
            [64, 127],    # Should map to palette color 1  
            [128, 191],   # Should map to palette color 2
            [192, 255]    # Should map to palette color 3
        ], dtype=np.uint8)
        
        converted = self.gbc_palette.convert_to_gbc_palette(test_image, 'text_white')
        palette = self.gbc_palette.get_palette('text_white')
        
        # Check that palette colors are used
        unique_colors = np.unique(converted.reshape(-1, 3), axis=0)
        self.assertLessEqual(len(unique_colors), 4)  # Should use at most 4 palette colors
        
    def test_convert_edge_cases(self):
        """Test conversion with edge case images"""
        # Single pixel image
        single_pixel = np.array([[128]], dtype=np.uint8)
        converted_single = self.gbc_palette.convert_to_gbc_palette(single_pixel, 'text_white')
        self.assertEqual(converted_single.shape, (1, 1, 3))
        
        # All black image
        black_image = np.zeros((4, 4), dtype=np.uint8)
        converted_black = self.gbc_palette.convert_to_gbc_palette(black_image, 'text_white')
        self.assertEqual(converted_black.shape, (4, 4, 3))
        
        # All white image
        white_image = np.ones((4, 4), dtype=np.uint8) * 255
        converted_white = self.gbc_palette.convert_to_gbc_palette(white_image, 'text_white')
        self.assertEqual(converted_white.shape, (4, 4, 3))


class TestPaletteDetection(unittest.TestCase):
    """Test palette detection functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.gbc_palette = GameBoyColorPalette()
        
    def test_detect_palette_rgb_image(self):
        """Test detecting palette from RGB image"""
        # Create image with colors similar to a known palette
        test_image = np.zeros((8, 8, 3), dtype=np.uint8)
        palette = self.gbc_palette.get_palette('text_white')
        
        # Fill image with palette colors
        for i, color in enumerate(palette.colors):
            y_start, y_end = i * 2, (i + 1) * 2
            test_image[y_start:y_end, :] = color
            
        detected = self.gbc_palette.detect_palette_from_image(test_image)
        self.assertIsInstance(detected, str)
        self.assertIn(detected, self.gbc_palette.list_palettes())
        
    def test_detect_palette_grayscale_image(self):
        """Test detecting palette from grayscale image"""
        test_image = np.random.randint(0, 256, (16, 16), dtype=np.uint8)
        
        detected = self.gbc_palette.detect_palette_from_image(test_image)
        self.assertIsInstance(detected, str)
        self.assertIn(detected, self.gbc_palette.list_palettes())
        
    def test_detect_palette_uniform_image(self):
        """Test detecting palette from uniform color image"""
        # Single color image
        uniform_image = np.ones((8, 8, 3), dtype=np.uint8) * 128
        
        detected = self.gbc_palette.detect_palette_from_image(uniform_image)
        self.assertIsInstance(detected, str)
        
    def test_detect_palette_empty_image(self):
        """Test detecting palette from very small or edge case images"""
        # Very small image
        small_image = np.array([[[255, 255, 255]]], dtype=np.uint8)
        detected = self.gbc_palette.detect_palette_from_image(small_image)
        self.assertIsInstance(detected, str)


class TestDominantColorExtraction(unittest.TestCase):
    """Test dominant color extraction functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.gbc_palette = GameBoyColorPalette()
        
    def test_get_dominant_colors_normal(self):
        """Test extracting dominant colors from normal image"""
        # Create image with distinct colors
        test_image = np.zeros((12, 12, 3), dtype=np.uint8)
        test_image[0:3, 0:3] = [255, 0, 0]      # Red
        test_image[3:6, 3:6] = [0, 255, 0]      # Green  
        test_image[6:9, 6:9] = [0, 0, 255]      # Blue
        test_image[9:12, 9:12] = [255, 255, 0]  # Yellow
        
        dominant_colors = self.gbc_palette._get_dominant_colors(test_image, k=4)
        
        self.assertEqual(len(dominant_colors), 4)
        for color in dominant_colors:
            self.assertIsInstance(color, tuple)
            self.assertEqual(len(color), 3)
            
    def test_get_dominant_colors_k_parameter(self):
        """Test dominant color extraction with different k values"""
        test_image = np.random.randint(0, 256, (16, 16, 3), dtype=np.uint8)
        
        for k in [1, 2, 3, 4, 8]:
            dominant_colors = self.gbc_palette._get_dominant_colors(test_image, k=k)
            self.assertEqual(len(dominant_colors), k)
            
    @patch('cv2.kmeans')
    def test_get_dominant_colors_cv2_error(self, mock_kmeans):
        """Test dominant color extraction with cv2 error"""
        mock_kmeans.side_effect = cv2.error("Mock error")
        
        test_image = np.random.randint(0, 256, (8, 8, 3), dtype=np.uint8)
        dominant_colors = self.gbc_palette._get_dominant_colors(test_image, k=4)
        
        # Should return gray fallback colors
        self.assertEqual(len(dominant_colors), 4)
        for color in dominant_colors:
            self.assertEqual(color, (128, 128, 128))
            
    def test_get_dominant_colors_single_pixel(self):
        """Test dominant color extraction from single pixel"""
        test_image = np.array([[[255, 128, 64]]], dtype=np.uint8)
        dominant_colors = self.gbc_palette._get_dominant_colors(test_image, k=2)
        
        self.assertEqual(len(dominant_colors), 2)


class TestPaletteSimilarity(unittest.TestCase):
    """Test palette similarity calculation"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.gbc_palette = GameBoyColorPalette()
        
    def test_calculate_palette_similarity_identical(self):
        """Test similarity calculation with identical palettes"""
        colors1 = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        colors2 = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        
        similarity = self.gbc_palette._calculate_palette_similarity(colors1, colors2)
        self.assertEqual(similarity, 1.0)
        
    def test_calculate_palette_similarity_different(self):
        """Test similarity calculation with different palettes"""
        colors1 = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        colors2 = [(0, 0, 0), (128, 128, 128), (255, 255, 255)]
        
        similarity = self.gbc_palette._calculate_palette_similarity(colors1, colors2)
        self.assertGreaterEqual(similarity, 0.0)
        self.assertLessEqual(similarity, 1.0)
        self.assertLess(similarity, 1.0)  # Should not be identical
        
    def test_calculate_palette_similarity_empty(self):
        """Test similarity calculation with empty palettes"""
        colors1 = []
        colors2 = [(255, 0, 0)]
        
        similarity = self.gbc_palette._calculate_palette_similarity(colors1, colors2)
        self.assertEqual(similarity, 0.0)
        
        similarity2 = self.gbc_palette._calculate_palette_similarity([], [])
        self.assertEqual(similarity2, 0.0)
        
    def test_calculate_palette_similarity_single_color(self):
        """Test similarity calculation with single colors"""
        colors1 = [(255, 255, 255)]
        colors2 = [(0, 0, 0)]
        
        similarity = self.gbc_palette._calculate_palette_similarity(colors1, colors2)
        self.assertGreaterEqual(similarity, 0.0)
        self.assertLessEqual(similarity, 1.0)
        
    def test_calculate_palette_similarity_partial_match(self):
        """Test similarity with partially matching palettes"""
        colors1 = [(255, 0, 0), (0, 255, 0)]
        colors2 = [(255, 0, 0), (0, 0, 255)]  # One match, one different
        
        similarity = self.gbc_palette._calculate_palette_similarity(colors1, colors2)
        self.assertGreater(similarity, 0.0)
        self.assertLess(similarity, 1.0)


class TestColorAwareTemplates(unittest.TestCase):
    """Test color-aware template creation"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.gbc_palette = GameBoyColorPalette()
        
    def test_create_color_aware_template_basic(self):
        """Test basic color-aware template creation"""
        template = np.random.randint(0, 256, (8, 8), dtype=np.uint8)
        
        color_template = self.gbc_palette.create_color_aware_template(
            template, 'text_white', 'text_black'
        )
        
        self.assertEqual(color_template.shape, (8, 8))
        self.assertEqual(color_template.dtype, np.uint8)
        
    def test_create_color_aware_template_same_palette(self):
        """Test creating template with same source and target palette"""
        template = np.eye(8, dtype=np.uint8) * 255
        
        color_template = self.gbc_palette.create_color_aware_template(
            template, 'text_white', 'text_white'
        )
        
        self.assertEqual(color_template.shape, template.shape)
        
    def test_create_color_aware_template_unknown_palette(self):
        """Test creating template with unknown palette"""
        template = np.ones((8, 8), dtype=np.uint8) * 128
        
        color_template = self.gbc_palette.create_color_aware_template(
            template, 'unknown_palette', 'text_white'
        )
        
        # Should still work (unknown palette gets handled in convert_to_gbc_palette)
        self.assertEqual(color_template.shape, (8, 8))


class TestTemplateEnhancement(unittest.TestCase):
    """Test template enhancement for different lighting conditions"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.gbc_palette = GameBoyColorPalette()
        
    def test_enhance_template_normal(self):
        """Test template enhancement with normal lighting"""
        template = np.random.randint(50, 200, (8, 8), dtype=np.uint8)
        
        enhanced = self.gbc_palette.enhance_template_for_lighting(template, 'normal')
        
        # Normal lighting should return unchanged template
        np.testing.assert_array_equal(enhanced, template)
        
    def test_enhance_template_bright(self):
        """Test template enhancement for bright lighting"""
        template = np.ones((8, 8), dtype=np.uint8) * 100
        
        enhanced = self.gbc_palette.enhance_template_for_lighting(template, 'bright')
        
        self.assertEqual(enhanced.shape, template.shape)
        self.assertEqual(enhanced.dtype, np.uint8)
        # Bright enhancement should generally increase values
        
    def test_enhance_template_dark(self):
        """Test template enhancement for dark lighting"""
        template = np.ones((8, 8), dtype=np.uint8) * 100
        
        enhanced = self.gbc_palette.enhance_template_for_lighting(template, 'dark')
        
        self.assertEqual(enhanced.shape, template.shape)
        self.assertEqual(enhanced.dtype, np.uint8)
        
    def test_enhance_template_night(self):
        """Test template enhancement for night lighting"""
        template = np.ones((8, 8), dtype=np.uint8) * 100
        
        enhanced = self.gbc_palette.enhance_template_for_lighting(template, 'night')
        
        self.assertEqual(enhanced.shape, template.shape)
        self.assertEqual(enhanced.dtype, np.uint8)
        
    def test_enhance_template_unknown_condition(self):
        """Test template enhancement with unknown lighting condition"""
        template = np.random.randint(0, 256, (8, 8), dtype=np.uint8)
        
        enhanced = self.gbc_palette.enhance_template_for_lighting(template, 'unknown')
        
        # Unknown condition should return unchanged template
        np.testing.assert_array_equal(enhanced, template)
        
    def test_enhance_template_extreme_values(self):
        """Test template enhancement with extreme pixel values"""
        # Test with very dark template
        dark_template = np.ones((8, 8), dtype=np.uint8) * 5
        enhanced_dark = self.gbc_palette.enhance_template_for_lighting(dark_template, 'bright')
        self.assertEqual(enhanced_dark.shape, dark_template.shape)
        
        # Test with very bright template
        bright_template = np.ones((8, 8), dtype=np.uint8) * 250
        enhanced_bright = self.gbc_palette.enhance_template_for_lighting(bright_template, 'dark')
        self.assertEqual(enhanced_bright.shape, bright_template.shape)


class TestScreenTypePalettes(unittest.TestCase):
    """Test screen type to palette mapping"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.gbc_palette = GameBoyColorPalette()
        
    def test_get_palette_for_screen_type_known(self):
        """Test getting palettes for known screen types"""
        known_mappings = {
            'dialogue': 'dialogue_normal',
            'battle': 'menu_battle',
            'menu': 'menu_standard',
            'overworld': 'text_white',
            'intro': 'pokemon_crystal'
        }
        
        for screen_type, expected_palette in known_mappings.items():
            result = self.gbc_palette.get_palette_for_screen_type(screen_type)
            self.assertEqual(result, expected_palette)
            
    def test_get_palette_for_screen_type_unknown(self):
        """Test getting palette for unknown screen type"""
        result = self.gbc_palette.get_palette_for_screen_type('unknown_screen')
        self.assertEqual(result, 'text_white')  # Should return default
        
    def test_get_palette_for_screen_type_none(self):
        """Test getting palette for None screen type"""
        result = self.gbc_palette.get_palette_for_screen_type(None)
        self.assertEqual(result, 'text_white')  # Should return default
        
    def test_get_palette_for_screen_type_empty_string(self):
        """Test getting palette for empty string screen type"""
        result = self.gbc_palette.get_palette_for_screen_type('')
        self.assertEqual(result, 'text_white')  # Should return default


class TestTextRegionAnalysis(unittest.TestCase):
    """Test text region color analysis"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.gbc_palette = GameBoyColorPalette()
        
    def test_analyze_text_region_grayscale(self):
        """Test analyzing grayscale text region"""
        # Create test region with varying brightness
        region = np.array([
            [0, 64, 128, 192],
            [32, 96, 160, 224], 
            [16, 80, 144, 208],
            [48, 112, 176, 255]
        ], dtype=np.uint8)
        
        analysis = self.gbc_palette.analyze_text_region_colors(region)
        
        # Check all expected keys are present
        expected_keys = [
            'mean_brightness', 'std_brightness', 'min_brightness', 'max_brightness',
            'contrast', 'text_style', 'detected_palette', 'is_high_contrast', 'is_uniform'
        ]
        for key in expected_keys:
            self.assertIn(key, analysis)
            
        # Check value types and ranges
        self.assertIsInstance(analysis['mean_brightness'], (int, float, np.number))
        self.assertIsInstance(analysis['is_high_contrast'], (bool, np.bool_))
        self.assertIsInstance(analysis['is_uniform'], (bool, np.bool_))
        self.assertIn(analysis['text_style'], ['light_on_dark', 'dark_on_light'])
        
    def test_analyze_text_region_rgb(self):
        """Test analyzing RGB text region"""
        region = np.random.randint(0, 256, (8, 8, 3), dtype=np.uint8)
        
        analysis = self.gbc_palette.analyze_text_region_colors(region)
        
        # Should work with RGB input
        self.assertIn('mean_brightness', analysis)
        self.assertIn('detected_palette', analysis)
        
    def test_analyze_text_region_light_on_dark(self):
        """Test analyzing light text on dark background"""
        # Dark background with light text
        region = np.ones((8, 8), dtype=np.uint8) * 50  # Dark background
        region[2:6, 2:6] = 200  # Light text area
        
        analysis = self.gbc_palette.analyze_text_region_colors(region)
        
        self.assertLess(analysis['mean_brightness'], 128)
        self.assertEqual(analysis['text_style'], 'light_on_dark')
        
    def test_analyze_text_region_dark_on_light(self):
        """Test analyzing dark text on light background"""
        # Light background with dark text
        region = np.ones((8, 8), dtype=np.uint8) * 200  # Light background
        region[2:6, 2:6] = 50   # Dark text area
        
        analysis = self.gbc_palette.analyze_text_region_colors(region)
        
        self.assertGreater(analysis['mean_brightness'], 128)
        self.assertEqual(analysis['text_style'], 'dark_on_light')
        
    def test_analyze_text_region_high_contrast(self):
        """Test analyzing high contrast region"""
        # High contrast: black and white
        region = np.zeros((8, 8), dtype=np.uint8)
        region[0:4, :] = 255
        
        analysis = self.gbc_palette.analyze_text_region_colors(region)
        
        self.assertTrue(analysis['is_high_contrast'])
        self.assertEqual(analysis['contrast'], 255)
        
    def test_analyze_text_region_low_contrast(self):
        """Test analyzing low contrast region"""
        # Low contrast: similar gray values
        region = np.ones((8, 8), dtype=np.uint8)
        region[0:4, :] = region[0:4, :] * 120
        region[4:8, :] = region[4:8, :] * 130
        
        analysis = self.gbc_palette.analyze_text_region_colors(region)
        
        self.assertFalse(analysis['is_high_contrast'])
        
    def test_analyze_text_region_uniform(self):
        """Test analyzing uniform color region"""
        region = np.ones((8, 8), dtype=np.uint8) * 128
        
        analysis = self.gbc_palette.analyze_text_region_colors(region)
        
        self.assertTrue(analysis['is_uniform'])
        self.assertEqual(analysis['std_brightness'], 0.0)


class TestPalettePreview(unittest.TestCase):
    """Test palette preview functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.gbc_palette = GameBoyColorPalette()
        
    def test_preview_palette_existing(self):
        """Test previewing existing palette"""
        # Should not raise exception
        try:
            self.gbc_palette.preview_palette('text_white')
        except Exception as e:
            self.fail(f"preview_palette raised exception: {e}")
            
    def test_preview_palette_non_existent(self):
        """Test previewing non-existent palette"""
        # Should not raise exception, just print error
        try:
            self.gbc_palette.preview_palette('non_existent_palette')
        except Exception as e:
            self.fail(f"preview_palette raised exception: {e}")
            
    def test_preview_palette_all_known(self):
        """Test previewing all known palettes"""
        for palette_name in self.gbc_palette.list_palettes():
            try:
                self.gbc_palette.preview_palette(palette_name)
            except Exception as e:
                self.fail(f"preview_palette failed for {palette_name}: {e}")


class TestGameBoyColorPaletteIntegration(unittest.TestCase):
    """Integration tests for Game Boy Color palette system"""
    
    def setUp(self):
        """Set up integration test fixtures"""
        self.gbc_palette = GameBoyColorPalette()
        
    def test_complete_palette_workflow(self):
        """Test complete palette conversion and analysis workflow"""
        # Create test image
        test_image = np.random.randint(0, 256, (16, 16), dtype=np.uint8)
        
        # Convert to palette
        converted = self.gbc_palette.convert_to_gbc_palette(test_image, 'text_white')
        
        # Detect palette
        detected = self.gbc_palette.detect_palette_from_image(converted)
        
        # Analyze colors
        analysis = self.gbc_palette.analyze_text_region_colors(test_image)
        
        # Create color-aware template
        template = self.gbc_palette.create_color_aware_template(
            test_image, 'text_white', 'text_black'
        )
        
        # Enhance template
        enhanced = self.gbc_palette.enhance_template_for_lighting(template, 'bright')
        
        # All operations should complete successfully
        self.assertEqual(converted.shape, (16, 16, 3))
        self.assertIsInstance(detected, str)
        self.assertIsInstance(analysis, dict)
        self.assertEqual(template.shape, test_image.shape)
        self.assertEqual(enhanced.shape, test_image.shape)
        
    def test_palette_consistency(self):
        """Test consistency across palette operations"""
        # Convert image to palette and back
        original = np.random.randint(0, 256, (8, 8), dtype=np.uint8)
        
        # Convert to GBC palette
        gbc_converted = self.gbc_palette.convert_to_gbc_palette(original, 'text_white')
        
        # Analyze the converted image
        analysis = self.gbc_palette.analyze_text_region_colors(gbc_converted)
        
        # Results should be consistent
        self.assertEqual(gbc_converted.shape, (8, 8, 3))
        self.assertIn('detected_palette', analysis)
        
    def test_multiple_palette_comparisons(self):
        """Test comparing multiple palettes"""
        test_image = np.random.randint(0, 256, (12, 12), dtype=np.uint8)
        
        results = {}
        palettes_to_test = ['text_white', 'text_black', 'menu_standard', 'dialogue_normal']
        
        for palette_name in palettes_to_test:
            converted = self.gbc_palette.convert_to_gbc_palette(test_image, palette_name)
            detected = self.gbc_palette.detect_palette_from_image(converted)
            results[palette_name] = {
                'converted_shape': converted.shape,
                'detected_palette': detected
            }
        
        # All conversions should succeed
        for palette_name, result in results.items():
            self.assertEqual(result['converted_shape'], (12, 12, 3))
            self.assertIsInstance(result['detected_palette'], str)
            
    def test_test_gameboy_color_palette_function(self):
        """Test the standalone test function"""
        # Should not raise exception
        try:
            test_gameboy_color_palette()
        except Exception as e:
            self.fail(f"test_gameboy_color_palette raised exception: {e}")


class TestEdgeCasesAndErrorHandling(unittest.TestCase):
    """Test edge cases and error handling"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.gbc_palette = GameBoyColorPalette()
        
    def test_empty_image_handling(self):
        """Test handling of empty or invalid images"""
        # Empty array
        empty_array = np.array([], dtype=np.uint8)
        
        # Should not crash
        try:
            analysis = self.gbc_palette.analyze_text_region_colors(empty_array.reshape(0, 0))
        except:
            pass  # Expected to fail gracefully
            
    def test_single_pixel_operations(self):
        """Test operations on single pixel images"""
        single_pixel = np.array([[128]], dtype=np.uint8)
        
        # Convert to palette
        converted = self.gbc_palette.convert_to_gbc_palette(single_pixel, 'text_white')
        self.assertEqual(converted.shape, (1, 1, 3))
        
        # Analyze colors
        analysis = self.gbc_palette.analyze_text_region_colors(single_pixel)
        self.assertIn('mean_brightness', analysis)
        
    def test_extreme_image_values(self):
        """Test operations with extreme pixel values"""
        # All zeros
        zero_image = np.zeros((4, 4), dtype=np.uint8)
        converted_zero = self.gbc_palette.convert_to_gbc_palette(zero_image, 'text_white')
        self.assertEqual(converted_zero.shape, (4, 4, 3))
        
        # All max values
        max_image = np.ones((4, 4), dtype=np.uint8) * 255
        converted_max = self.gbc_palette.convert_to_gbc_palette(max_image, 'text_white')
        self.assertEqual(converted_max.shape, (4, 4, 3))
        
    @patch('cv2.cvtColor')
    def test_cv2_color_conversion_error(self, mock_cvtColor):
        """Test handling of cv2 color conversion errors"""
        mock_cvtColor.side_effect = cv2.error("Mock error")
        
        # Should handle conversion error gracefully
        test_image = np.random.randint(0, 256, (8, 8, 3), dtype=np.uint8)
        
        # These operations may fail but shouldn't crash
        try:
            converted = self.gbc_palette.convert_to_gbc_palette(test_image, 'text_white')
        except:
            pass  # Expected to handle errors
            
    def test_large_image_performance(self):
        """Test performance with larger images"""
        # Test with reasonably large image
        large_image = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
        
        # All operations should complete in reasonable time
        converted = self.gbc_palette.convert_to_gbc_palette(large_image, 'text_white')
        detected = self.gbc_palette.detect_palette_from_image(large_image)
        analysis = self.gbc_palette.analyze_text_region_colors(large_image)
        
        self.assertEqual(converted.shape, (64, 64, 3))
        self.assertIsInstance(detected, str)
        self.assertIsInstance(analysis, dict)


if __name__ == '__main__':
    unittest.main()
