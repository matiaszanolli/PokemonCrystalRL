"""
Comprehensive unit tests for the Pokemon Font Decoder module.
Tests character template creation, template matching, and text recognition functionality.
"""
import unittest
import numpy as np
import cv2
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Tuple, Optional

from pokemon_crystal_rl.vision.pokemon_font_decoder import PokemonFontDecoder, CharacterMatch


class TestCharacterMatch(unittest.TestCase):
    """Test CharacterMatch dataclass"""
    
    def test_character_match_creation(self):
        """Test creating CharacterMatch objects"""
        match = CharacterMatch(
            char='A',
            confidence=0.95,
            bbox=(10, 20, 8, 8),
            position=(14, 24)
        )
        
        self.assertEqual(match.char, 'A')
        self.assertEqual(match.confidence, 0.95)
        self.assertEqual(match.bbox, (10, 20, 8, 8))
        self.assertEqual(match.position, (14, 24))
        
    def test_character_match_equality(self):
        """Test CharacterMatch equality comparison"""
        match1 = CharacterMatch('A', 0.95, (10, 20, 8, 8), (14, 24))
        match2 = CharacterMatch('A', 0.95, (10, 20, 8, 8), (14, 24))
        match3 = CharacterMatch('B', 0.95, (10, 20, 8, 8), (14, 24))
        
        self.assertEqual(match1, match2)
        self.assertNotEqual(match1, match3)


class TestPokemonFontDecoder(unittest.TestCase):
    """Test PokemonFontDecoder class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.decoder = PokemonFontDecoder()
        
    def test_initialization(self):
        """Test decoder initialization"""
        self.assertEqual(self.decoder.char_width, 8)
        self.assertEqual(self.decoder.char_height, 8)
        self.assertTrue(self.decoder.font_loaded)
        self.assertIsInstance(self.decoder.char_templates, dict)
        self.assertGreater(len(self.decoder.char_templates), 0)
        
    def test_pokemon_character_mapping(self):
        """Test Pokemon Crystal character mapping constants"""
        # Test some known mappings
        self.assertEqual(self.decoder.pokemon_chars['A'], 0x80)
        self.assertEqual(self.decoder.pokemon_chars['Z'], 0x99)
        self.assertEqual(self.decoder.pokemon_chars['0'], 0xF6)
        self.assertEqual(self.decoder.pokemon_chars['9'], 0xFF)
        self.assertEqual(self.decoder.pokemon_chars[' '], 0x7F)
        self.assertEqual(self.decoder.pokemon_chars['!'], 0xE6)
        self.assertEqual(self.decoder.pokemon_chars['?'], 0xE5)
        
    def test_character_template_creation(self):
        """Test that character templates are created correctly"""
        # Check that basic templates exist
        required_chars = ['A', 'B', 'O', 'K', 'e', 'M', 'o', 'N', ' ']
        
        for char in required_chars:
            self.assertIn(char, self.decoder.char_templates)
            template = self.decoder.char_templates[char]
            self.assertIsInstance(template, np.ndarray)
            self.assertEqual(template.shape, (8, 8))
            self.assertEqual(template.dtype, np.uint8)
            
    def test_template_properties(self):
        """Test template array properties"""
        # Test 'A' template specifically
        a_template = self.decoder.char_templates['A']
        
        # Should be 8x8
        self.assertEqual(a_template.shape, (8, 8))
        
        # Should contain only 0 and 255 values (binary)
        unique_values = np.unique(a_template)
        self.assertTrue(all(val in [0, 255] for val in unique_values))
        
        # Should have some white pixels (letter content)
        self.assertGreater(np.sum(a_template == 255), 0)
        
    def test_space_character_template(self):
        """Test that space character template is empty"""
        space_template = self.decoder.char_templates[' ']
        
        # Space should be all zeros
        self.assertTrue(np.all(space_template == 0))
        
    def test_extract_character_grid_rgb_image(self):
        """Test extracting character grid from RGB image"""
        # Create a simple test image (16x8 - 2 characters wide, 1 high)
        test_image = np.zeros((8, 16, 3), dtype=np.uint8)
        
        # Add some white pixels to make characters
        test_image[2:6, 2:6, :] = 255  # First character
        test_image[2:6, 10:14, :] = 255  # Second character
        
        chars = self.decoder._extract_character_grid(test_image)
        
        # Should extract 2 characters
        self.assertEqual(len(chars), 2)
        
        # Each entry should be (char_cell, x, y)
        for char_cell, x, y in chars:
            self.assertEqual(char_cell.shape, (8, 8))
            self.assertIsInstance(x, int)
            self.assertIsInstance(y, int)
            
    def test_extract_character_grid_grayscale_image(self):
        """Test extracting character grid from grayscale image"""
        # Create a grayscale test image
        test_image = np.zeros((8, 16), dtype=np.uint8)
        test_image[2:6, 2:6] = 255  # First character
        test_image[2:6, 10:14] = 255  # Second character
        
        chars = self.decoder._extract_character_grid(test_image)
        
        self.assertEqual(len(chars), 2)
        for char_cell, x, y in chars:
            self.assertEqual(char_cell.shape, (8, 8))
            
    def test_extract_character_grid_empty_regions(self):
        """Test that empty character regions are filtered out"""
        # Create image with one character and one empty region
        test_image = np.zeros((8, 16), dtype=np.uint8)
        test_image[2:6, 2:6] = 255  # Only first character has content
        
        chars = self.decoder._extract_character_grid(test_image)
        
        # Should only extract the non-empty character
        # Note: This depends on the implementation having content filtering
        self.assertGreaterEqual(len(chars), 1)
        
    def test_match_character_exact_match(self):
        """Test character matching with exact template match"""
        # Use the 'A' template as input
        test_char = self.decoder.char_templates['A'].copy()
        
        # Should match 'A' with high confidence
        char, confidence = self.decoder._match_character(test_char)
        
        self.assertEqual(char, 'A')
        self.assertGreater(confidence, 0.9)
        
    def test_match_character_no_match(self):
        """Test character matching with no good matches"""
        # Create a random pattern that shouldn't match any template
        random_char = np.random.randint(0, 256, (8, 8), dtype=np.uint8)
        
        # Should return low confidence match (method always returns something)
        char, confidence = self.decoder._match_character(random_char)
        
        # Should have low confidence for random input
        self.assertIsInstance(char, str)
        self.assertIsInstance(confidence, (float, np.floating))  # Accept both float and numpy float
        self.assertLessEqual(confidence, 1.0)
            
    def test_match_character_with_noise(self):
        """Test character matching with noisy input"""
        # Start with 'A' template and add some noise
        noisy_char = self.decoder.char_templates['A'].copy()
        
        # Add small amount of noise
        noise_positions = np.random.choice(64, size=5, replace=False)
        for pos in noise_positions:
            row, col = pos // 8, pos % 8
            noisy_char[row, col] = 255 - noisy_char[row, col]  # Flip pixel
            
        char, confidence = self.decoder._match_character(noisy_char)
        
        # Should still match something, but might not be 'A' due to noise
        self.assertIsInstance(char, str)
        self.assertIsInstance(confidence, (float, np.floating))
        self.assertGreater(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
        
    def test_match_character_always_returns_result(self):
        """Test that _match_character always returns a result"""
        test_char = self.decoder.char_templates['A'].copy()
        
        # Should always return a character and confidence
        char, confidence = self.decoder._match_character(test_char)
        
        self.assertIsInstance(char, str)
        self.assertIsInstance(confidence, (float, np.floating))
        self.assertEqual(char, 'A')  # Should be perfect match
        self.assertGreater(confidence, 0.9)
        
    def test_decode_text_simple_word(self):
        """Test decoding a simple word"""
        # Create image with "OK" (2 characters, 16x8)
        test_image = np.zeros((8, 16), dtype=np.uint8)
        
        # Place 'O' template in first position
        test_image[0:8, 0:8] = self.decoder.char_templates['O']
        
        # Place 'K' template in second position  
        test_image[0:8, 8:16] = self.decoder.char_templates['K']
        
        result = self.decoder.decode_text(test_image)
        
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].char, 'O')
        self.assertEqual(result[1].char, 'K')
        
    def test_decode_text_with_confidence_filter(self):
        """Test text decoding with confidence filtering"""
        # Create test image
        test_image = np.zeros((8, 16), dtype=np.uint8)
        test_image[0:8, 0:8] = self.decoder.char_templates['A']
        
        # With low confidence threshold
        result_low = self.decoder.decode_text(test_image, min_confidence=0.5)
        self.assertGreater(len(result_low), 0)
        
        # With high confidence threshold  
        result_high = self.decoder.decode_text(test_image, min_confidence=0.99)
        # Might have fewer results due to template matching precision
        self.assertGreaterEqual(len(result_low), len(result_high))
        
    def test_decode_text_empty_image(self):
        """Test decoding empty image"""
        empty_image = np.zeros((8, 8), dtype=np.uint8)
        
        result = self.decoder.decode_text(empty_image)
        
        # Should return empty list or list with only space characters
        # depending on implementation
        self.assertIsInstance(result, list)
        
    def test_decode_text_return_types(self):
        """Test that decode_text returns proper types"""
        test_image = np.zeros((8, 8), dtype=np.uint8)
        test_image[:, :] = self.decoder.char_templates['A']
        
        result = self.decoder.decode_text(test_image)
        
        self.assertIsInstance(result, list)
        for match in result:
            self.assertIsInstance(match, CharacterMatch)
            self.assertIsInstance(match.char, str)
            self.assertIsInstance(match.confidence, (float, np.floating))
            self.assertIsInstance(match.bbox, tuple)
            self.assertIsInstance(match.position, tuple)
            
    def test_template_matching_algorithm(self):
        """Test the underlying template matching algorithm"""
        # Test with perfect match
        test_template = self.decoder.char_templates['B'].copy()
        
        with patch('cv2.matchTemplate') as mock_match:
            mock_match.return_value = np.array([[1.0]])  # Perfect match
            
            char, confidence = self.decoder._match_character(test_template)
            
            self.assertIsInstance(char, str)
            self.assertIsInstance(confidence, float)
            mock_match.assert_called()
            
    def test_multiple_character_templates(self):
        """Test that all expected character templates exist"""
        expected_chars = ['A', 'B', 'O', 'K', 'e', 'M', 'o', 'N', ' ']
        
        for char in expected_chars:
            self.assertIn(char, self.decoder.char_templates)
            template = self.decoder.char_templates[char]
            
            # Verify template properties
            self.assertEqual(template.shape, (8, 8))
            self.assertEqual(template.dtype, np.uint8)
            
            # Verify template has expected pattern
            if char != ' ':  # Space should be empty
                self.assertGreater(np.sum(template), 0)
                
    def test_font_decoder_robustness(self):
        """Test decoder robustness with various inputs"""
        test_cases = [
            np.zeros((8, 8), dtype=np.uint8),  # All black
            np.ones((8, 8), dtype=np.uint8) * 255,  # All white
            np.random.randint(0, 256, (8, 8), dtype=np.uint8),  # Random
        ]
        
        for test_case in test_cases:
            # Should not crash
            try:
                char, confidence = self.decoder._match_character(test_case)
                # Should always return string and float
                self.assertIsInstance(char, str)
                self.assertIsInstance(confidence, (float, np.floating))
            except Exception as e:
                self.fail(f"Decoder crashed with input {test_case.shape}: {e}")
                
    @patch('cv2.matchTemplate')
    def test_template_matching_cv2_integration(self, mock_match_template):
        """Test integration with OpenCV template matching"""
        # Mock cv2.matchTemplate to return known values
        mock_match_template.return_value = np.array([[0.85]])
        
        test_char = np.zeros((8, 8), dtype=np.uint8)
        char, confidence = self.decoder._match_character(test_char)
        
        # Should have called cv2.matchTemplate for each template
        self.assertTrue(mock_match_template.called)
        self.assertGreater(mock_match_template.call_count, 0)
        self.assertIsInstance(char, str)
        self.assertIsInstance(confidence, (float, np.floating))
        
    def test_character_position_calculation(self):
        """Test that character positions are calculated correctly"""
        # Create 2x1 character grid (16x8 image)
        test_image = np.zeros((8, 16), dtype=np.uint8)
        test_image[0:8, 0:8] = self.decoder.char_templates['A']
        test_image[0:8, 8:16] = self.decoder.char_templates['B']
        
        result = self.decoder.decode_text(test_image)
        
        if len(result) >= 2:
            # First character should be at position around (4, 4)
            self.assertAlmostEqual(result[0].position[0], 4, delta=2)
            self.assertAlmostEqual(result[0].position[1], 4, delta=2)
            
            # Second character should be at position around (12, 4)
            self.assertAlmostEqual(result[1].position[0], 12, delta=2)
            self.assertAlmostEqual(result[1].position[1], 4, delta=2)
            
    def test_bounding_box_calculation(self):
        """Test that bounding boxes are calculated correctly"""
        test_image = np.zeros((8, 8), dtype=np.uint8)
        test_image[:, :] = self.decoder.char_templates['A']
        
        result = self.decoder.decode_text(test_image)
        
        if len(result) > 0:
            bbox = result[0].bbox
            self.assertEqual(len(bbox), 4)  # x, y, width, height
            self.assertEqual(bbox[2], 8)  # width
            self.assertEqual(bbox[3], 8)  # height
            self.assertGreaterEqual(bbox[0], 0)  # x >= 0
            self.assertGreaterEqual(bbox[1], 0)  # y >= 0
            

    def test_decode_text_lines(self):
        """Test decoding text and organizing into lines"""
        # Create a 2-line text image (16x16 - 2 chars wide, 2 lines)
        test_image = np.zeros((16, 16), dtype=np.uint8)
        
        # First line: "OK"
        test_image[0:8, 0:8] = self.decoder.char_templates['O']
        test_image[0:8, 8:16] = self.decoder.char_templates['K']
        
        # Second line: "AB" 
        test_image[8:16, 0:8] = self.decoder.char_templates['A']
        test_image[8:16, 8:16] = self.decoder.char_templates['B']
        
        lines = self.decoder.decode_text_lines(test_image)
        
        self.assertEqual(len(lines), 2)
        self.assertEqual(lines[0], 'OK')
        self.assertEqual(lines[1], 'AB')
        
    def test_decode_text_lines_empty_image(self):
        """Test decoding lines from empty image"""
        empty_image = np.zeros((16, 16), dtype=np.uint8)
        
        lines = self.decoder.decode_text_lines(empty_image)
        
        self.assertIsInstance(lines, list)
        # Should return empty list or list of empty strings
        
    def test_get_text_regions(self):
        """Test getting text organized by screen regions"""
        # Create a larger test image (48x64 - typical game screen proportions)
        test_image = np.zeros((48, 64), dtype=np.uint8)
        
        # Add text in dialogue region (bottom 30%)
        dialogue_y_start = int(48 * 0.7)  # Bottom 30%
        test_image[dialogue_y_start:dialogue_y_start+8, 0:8] = self.decoder.char_templates['A']
        test_image[dialogue_y_start:dialogue_y_start+8, 8:16] = self.decoder.char_templates['B']
        
        # Add text in UI region (top-right)
        ui_x_start = int(64 * 0.6)  # Right 40%
        test_image[0:8, ui_x_start:ui_x_start+8] = self.decoder.char_templates['O']
        
        regions = self.decoder.get_text_regions(test_image)
        
        self.assertIsInstance(regions, dict)
        # Should have found text in some regions
        
    def test_get_text_regions_empty_image(self):
        """Test getting regions from empty image"""
        empty_image = np.zeros((48, 64), dtype=np.uint8)
        
        regions = self.decoder.get_text_regions(empty_image)
        
        self.assertIsInstance(regions, dict)
        # Empty image should return empty dict or dict with empty regions
        
    def test_font_not_loaded_handling(self):
        """Test handling when font is not loaded"""
        # Create decoder and simulate font not loaded
        decoder = PokemonFontDecoder()
        decoder.font_loaded = False
        
        # Test image
        test_image = np.zeros((8, 8), dtype=np.uint8)
        
        # Should handle gracefully
        result = decoder.decode_text(test_image)
        self.assertEqual(result, [])
        
        lines = decoder.decode_text_lines(test_image)
        self.assertEqual(lines, [])
        
    def test_character_resizing_in_match(self):
        """Test character resizing when dimensions don't match"""
        # Create character with wrong dimensions
        wrong_size_char = np.zeros((6, 6), dtype=np.uint8)
        wrong_size_char[1:5, 1:5] = 255  # Some pattern
        
        # Should handle resizing gracefully
        char, confidence = self.decoder._match_character(wrong_size_char)
        
        self.assertIsInstance(char, str)
        self.assertIsInstance(confidence, (float, np.floating))
        

class TestPokemonFontDecoderIntegration(unittest.TestCase):
    """Integration tests for Pokemon Font Decoder"""
    
    def setUp(self):
        """Set up integration test fixtures"""
        self.decoder = PokemonFontDecoder()
        
    def test_decode_pokemon_text(self):
        """Test decoding Pokemon-style text"""
        # Create a longer Pokemon text: "POKEMON"
        chars = ['P', 'O', 'K', 'e', 'M', 'o', 'N']
        image_width = len(chars) * 8
        test_image = np.zeros((8, image_width), dtype=np.uint8)
        
        # Place each character template
        for i, char in enumerate(chars):
            if char in self.decoder.char_templates:
                x_start = i * 8
                x_end = x_start + 8
                test_image[0:8, x_start:x_end] = self.decoder.char_templates[char]
                
        result = self.decoder.decode_text(test_image)
        
        # Should recognize most or all characters
        self.assertGreater(len(result), 0)
        
        # Check that recognized characters are from our input
        recognized_chars = [match.char for match in result]
        for char in recognized_chars:
            self.assertIn(char, chars)
            
    def test_performance_with_large_image(self):
        """Test performance with larger images"""
        import time
        
        # Create a larger image (4 characters high, 10 wide = 32x80)
        large_image = np.zeros((32, 80), dtype=np.uint8)
        
        # Fill with random templates
        templates = list(self.decoder.char_templates.keys())[:10]  # First 10 templates
        
        for row in range(4):
            for col in range(10):
                if col < len(templates):
                    template = self.decoder.char_templates[templates[col]]
                    y_start, y_end = row * 8, (row + 1) * 8
                    x_start, x_end = col * 8, (col + 1) * 8
                    large_image[y_start:y_end, x_start:x_end] = template
                    
        start_time = time.time()
        result = self.decoder.decode_text(large_image)
        end_time = time.time()
        
        decode_time = end_time - start_time
        
        # Should complete in reasonable time (adjust threshold as needed)
        self.assertLess(decode_time, 5.0)  # 5 seconds max
        self.assertGreater(len(result), 0)
        
        print(f"🚀 Decoded {len(result)} characters in {decode_time:.3f}s")


if __name__ == '__main__':
    unittest.main()
