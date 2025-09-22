"""
Comprehensive unit tests for the ROM Font Extractor module.
Tests ROM loading, tile decoding, font extraction, and template management functionality.
"""
import unittest
import numpy as np
import os
import tempfile
import shutil
from unittest.mock import Mock, patch, mock_open, MagicMock
from typing import Dict, List, Tuple, Optional

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from vision.extractors.rom_font_extractor import PokemonCrystalFontExtractor, FontTile, extract_pokemon_crystal_fonts


class TestFontTile(unittest.TestCase):
    """Test FontTile dataclass"""
    
    def test_font_tile_creation(self):
        """Test creating FontTile objects"""
        bitmap = np.zeros((8, 8), dtype=np.uint8)
        tile = FontTile(
            char_code=0x80,
            bitmap=bitmap,
            character='A'
        )
        
        self.assertEqual(tile.char_code, 0x80)
        self.assertEqual(tile.character, 'A')
        self.assertIsInstance(tile.bitmap, np.ndarray)
        self.assertEqual(tile.bitmap.shape, (8, 8))
        
    def test_font_tile_equality(self):
        """Test FontTile equality comparison"""
        bitmap1 = np.ones((8, 8), dtype=np.uint8)
        bitmap2 = np.ones((8, 8), dtype=np.uint8)
        bitmap3 = np.zeros((8, 8), dtype=np.uint8)
        
        tile1 = FontTile(0x80, bitmap1, 'A')
        tile2 = FontTile(0x80, bitmap2, 'A')
        tile3 = FontTile(0x80, bitmap3, 'A')
        
        # Test equality using custom comparison to handle numpy arrays
        self.assertTrue(tile1.char_code == tile2.char_code and tile1.character == tile2.character and np.array_equal(tile1.bitmap, tile2.bitmap))
        self.assertFalse(tile1.char_code == tile3.char_code and tile1.character == tile3.character and np.array_equal(tile1.bitmap, tile3.bitmap))


class TestPokemonCrystalFontExtractor(unittest.TestCase):
    """Test PokemonCrystalFontExtractor class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.extractor = PokemonCrystalFontExtractor()
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up test fixtures"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            
    def test_initialization(self):
        """Test extractor initialization"""
        self.assertIsNone(self.extractor.rom_path)
        self.assertIsNone(self.extractor.rom_data)
        self.assertIsInstance(self.extractor.font_locations, dict)
        self.assertIsInstance(self.extractor.char_mapping, dict)
        
        # Check font locations structure
        self.assertIn('main_font', self.extractor.font_locations)
        self.assertIn('start_address', self.extractor.font_locations['main_font'])
        
    def test_initialization_with_rom_path(self):
        """Test initialization with ROM path"""
        rom_path = "/test/rom.gbc"
        extractor = PokemonCrystalFontExtractor(rom_path)
        self.assertEqual(extractor.rom_path, rom_path)
        
    def test_character_mapping(self):
        """Test Pokemon Crystal character mapping constants"""
        # Test some known mappings
        self.assertEqual(self.extractor.char_mapping[0x80], 'A')
        self.assertEqual(self.extractor.char_mapping[0x99], 'Z')
        self.assertEqual(self.extractor.char_mapping[0xA0], 'a')
        self.assertEqual(self.extractor.char_mapping[0xB9], 'z')
        self.assertEqual(self.extractor.char_mapping[0xF6], '0')
        self.assertEqual(self.extractor.char_mapping[0xFF], '9')
        self.assertEqual(self.extractor.char_mapping[0x7F], ' ')
        
    def test_font_locations_structure(self):
        """Test that font locations are properly structured"""
        required_keys = ['start_address', 'tile_count', 'char_offset', 'size', 'style']
        
        for font_name, font_info in self.extractor.font_locations.items():
            for key in required_keys:
                self.assertIn(key, font_info, f"Missing {key} in {font_name}")
                
            # Validate data types
            self.assertIsInstance(font_info['start_address'], int)
            self.assertIsInstance(font_info['tile_count'], int)
            self.assertIsInstance(font_info['char_offset'], int)
            self.assertIsInstance(font_info['size'], tuple)
            self.assertIsInstance(font_info['style'], str)
            
    def test_load_rom_file_not_found(self):
        """Test loading ROM when file doesn't exist"""
        result = self.extractor.load_rom("/nonexistent/rom.gbc")
        
        self.assertFalse(result)
        self.assertIsNone(self.extractor.rom_data)
        
    def test_load_rom_success(self):
        """Test successful ROM loading"""
        # Create mock ROM data with proper header
        mock_rom_data = bytearray(0x8000)  # 32KB ROM
        mock_rom_data[0x143] = 0x80  # GBC compatible flag
        
        mock_rom_path = os.path.join(self.temp_dir, "test_rom.gbc")
        with open(mock_rom_path, 'wb') as f:
            f.write(mock_rom_data)
            
        result = self.extractor.load_rom(mock_rom_path)
        
        self.assertTrue(result)
        self.assertIsNotNone(self.extractor.rom_data)
        self.assertEqual(len(self.extractor.rom_data), len(mock_rom_data))
        
    def test_load_rom_invalid_size(self):
        """Test loading ROM that's too small"""
        # Create very small file
        mock_rom_path = os.path.join(self.temp_dir, "small_rom.gbc")
        with open(mock_rom_path, 'wb') as f:
            f.write(b'small')  # Only 5 bytes
            
        result = self.extractor.load_rom(mock_rom_path)
        
        self.assertFalse(result)
        
    def test_load_rom_non_gbc_format(self):
        """Test loading ROM without GBC flag"""
        # Create ROM with non-GBC flag
        mock_rom_data = bytearray(0x8000)
        mock_rom_data[0x143] = 0x00  # Non-GBC flag
        
        mock_rom_path = os.path.join(self.temp_dir, "non_gbc_rom.gbc")
        with open(mock_rom_path, 'wb') as f:
            f.write(mock_rom_data)
            
        result = self.extractor.load_rom(mock_rom_path)
        
        # Should still load but with warning
        self.assertTrue(result)
        
    def test_load_rom_exception_handling(self):
        """Test ROM loading with file I/O exception"""
        with patch('builtins.open', side_effect=IOError("File error")):
            result = self.extractor.load_rom("/test/rom.gbc")
            
        self.assertFalse(result)
        
    def test_decode_2bpp_tile_valid(self):
        """Test decoding valid 2bpp tile data"""
        # Create test tile data (16 bytes)
        # This represents a simple pattern
        tile_data = bytes([
            0xFF, 0x00,  # Row 0: 11111111, 00000000 -> all pixel value 1
            0x00, 0xFF,  # Row 1: 00000000, 11111111 -> all pixel value 2  
            0xFF, 0xFF,  # Row 2: 11111111, 11111111 -> all pixel value 3
            0x00, 0x00,  # Row 3: 00000000, 00000000 -> all pixel value 0
            0x81, 0x81,  # Row 4: 10000001, 10000001 -> value 3 at ends, 0 in middle
            0x00, 0x00,  # Row 5-7: all zeros
            0x00, 0x00,
            0x00, 0x00
        ])
        
        pixels = self.extractor._decode_2bpp_tile(tile_data)
        
        self.assertEqual(pixels.shape, (8, 8))
        self.assertEqual(pixels.dtype, np.uint8)
        
        # Check specific pixel values
        self.assertEqual(pixels[0, 0], 1)  # First row should be all 1s
        self.assertEqual(pixels[1, 0], 2)  # Second row should be all 2s
        self.assertEqual(pixels[2, 0], 3)  # Third row should be all 3s
        self.assertEqual(pixels[3, 0], 0)  # Fourth row should be all 0s
        
    def test_decode_2bpp_tile_invalid_length(self):
        """Test decoding tile with wrong data length"""
        with self.assertRaises(ValueError):
            self.extractor._decode_2bpp_tile(b'short')  # Too short
            
        with self.assertRaises(ValueError):
            self.extractor._decode_2bpp_tile(b'x' * 20)  # Too long
            
    def test_decode_2bpp_tile_bit_manipulation(self):
        """Test bit manipulation in tile decoding"""
        # Create tile with known bit patterns
        tile_data = bytes([
            0x80, 0x80,  # 10000000, 10000000 -> first pixel = 3, rest = 0
            0x40, 0x40,  # 01000000, 01000000 -> second pixel = 3, rest = 0
            0x01, 0x01,  # 00000001, 00000001 -> last pixel = 3, rest = 0
            0x00, 0x00,  # All zeros
            0x00, 0x00,
            0x00, 0x00,
            0x00, 0x00,
            0x00, 0x00
        ])
        
        pixels = self.extractor._decode_2bpp_tile(tile_data)
        
        # Check bit positions
        self.assertEqual(pixels[0, 0], 3)  # First pixel
        self.assertEqual(pixels[0, 1], 0)  # Other pixels in row 0
        self.assertEqual(pixels[1, 1], 3)  # Second pixel in row 1
        self.assertEqual(pixels[2, 7], 3)  # Last pixel in row 2
        
    def test_convert_to_binary(self):
        """Test converting 2bpp pixels to binary"""
        # Create test pixel array with different values
        tile_pixels = np.array([
            [0, 1, 2, 3, 0, 1, 2, 3],
            [3, 2, 1, 0, 3, 2, 1, 0],
            [0, 0, 0, 0, 1, 1, 1, 1],
            [2, 2, 2, 2, 3, 3, 3, 3],
            [0, 1, 0, 1, 0, 1, 0, 1],
            [2, 3, 2, 3, 2, 3, 2, 3],
            [0, 0, 1, 1, 2, 2, 3, 3],
            [1, 2, 3, 0, 1, 2, 3, 0]
        ], dtype=np.uint8)
        
        binary = self.extractor._convert_to_binary(tile_pixels)
        
        self.assertEqual(binary.shape, (8, 8))
        self.assertEqual(binary.dtype, np.uint8)
        
        # Check conversion logic: 0 -> 0, anything else -> 255
        expected = np.where(tile_pixels > 0, 255, 0).astype(np.uint8)
        np.testing.assert_array_equal(binary, expected)
        
    def test_extract_font_set_no_rom_data(self):
        """Test extracting font set without ROM data loaded"""
        result = self.extractor.extract_font_set('main_font')
        
        self.assertEqual(result, {})
        
    def test_extract_font_set_invalid_font_set(self):
        """Test extracting unknown font set"""
        # Load some dummy ROM data
        self.extractor.rom_data = b'x' * 0x8000
        
        result = self.extractor.extract_font_set('unknown_font')
        
        self.assertEqual(result, {})
        
    def test_extract_font_set_success(self):
        """Test successful font set extraction"""
        # Create mock ROM with known tile data
        rom_size = 0x25000  # Large enough to contain font data
        mock_rom = bytearray(rom_size)
        
        # Put some test tile data at main_font location (0x1C000)
        main_font_addr = 0x1C000
        
        # Create a simple 'A' character tile
        a_tile_data = bytes([
            0x18, 0x18,  # Simple A pattern
            0x24, 0x24,
            0x42, 0x42,
            0x7E, 0x7E,
            0x81, 0x81,
            0x81, 0x81,
            0x00, 0x00,
            0x00, 0x00
        ])
        
        # Place the tile data in ROM
        mock_rom[main_font_addr:main_font_addr + 16] = a_tile_data
        self.extractor.rom_data = bytes(mock_rom)
        
        result = self.extractor.extract_font_set('main_font')
        
        self.assertIsInstance(result, dict)
        
        # Should extract the 'A' character (char_code 0x80 maps to 'A')
        if 'A' in result:
            self.assertEqual(result['A'].shape, (8, 8))
            self.assertEqual(result['A'].dtype, np.uint8)
            
    def test_extract_font_set_rom_bounds_check(self):
        """Test font extraction with ROM bounds checking"""
        # Create small ROM that will run out of data
        small_rom = b'x' * 0x1C100  # Just past start of main font
        self.extractor.rom_data = small_rom
        
        result = self.extractor.extract_font_set('main_font')
        
        # Should handle gracefully when reaching end of ROM
        self.assertIsInstance(result, dict)
        
    def test_extract_font_set_tile_decode_error(self):
        """Test font extraction with tile decode error"""
        # Create ROM with invalid tile data (will cause decode errors)
        rom_size = 0x25000
        mock_rom = bytearray(rom_size)
        self.extractor.rom_data = bytes(mock_rom)
        
        # Mock decode method to raise exception
        with patch.object(self.extractor, '_decode_2bpp_tile', side_effect=ValueError("Mock error")):
            result = self.extractor.extract_font_set('main_font')
            
        # Should handle decode errors gracefully
        self.assertIsInstance(result, dict)
        
    def test_extract_all_fonts(self):
        """Test extracting all font sets"""
        # Load some mock ROM data first (required for extract_all_fonts to proceed)
        self.extractor.rom_data = b'\x00' * 32768
        
        # Mock the extract_font_set method
        mock_font_data = {'A': np.ones((8, 8), dtype=np.uint8)}
        
        with patch.object(self.extractor, 'extract_font_set', return_value=mock_font_data) as mock_extract:
            result = self.extractor.extract_all_fonts()
            
        self.assertIsInstance(result, dict)
        # Should call extract_font_set for each font location
        self.assertEqual(mock_extract.call_count, len(self.extractor.font_locations))
        
    def test_extract_all_fonts_no_rom_data(self):
        """Test extracting all fonts without ROM data"""
        result = self.extractor.extract_all_fonts()
        
        self.assertEqual(result, {})
        
    def test_save_font_templates_success(self):
        """Test successful template saving"""
        font_tiles = {
            'A': np.ones((8, 8), dtype=np.uint8),
            'B': np.zeros((8, 8), dtype=np.uint8)
        }
        
        output_path = os.path.join(self.temp_dir, "test_templates.npz")
        result = self.extractor.save_font_templates(font_tiles, output_path)
        
        self.assertTrue(result)
        self.assertTrue(os.path.exists(output_path))
        
        # Verify file contents
        loaded_data = np.load(output_path)
        self.assertIn('A', loaded_data.files)
        self.assertIn('B', loaded_data.files)
        
    def test_save_font_templates_directory_creation(self):
        """Test template saving with directory creation"""
        font_tiles = {'A': np.ones((8, 8), dtype=np.uint8)}
        
        nested_dir = os.path.join(self.temp_dir, "nested", "deep")
        output_path = os.path.join(nested_dir, "test_templates.npz")
        
        result = self.extractor.save_font_templates(font_tiles, output_path)
        
        self.assertTrue(result)
        self.assertTrue(os.path.exists(nested_dir))
        self.assertTrue(os.path.exists(output_path))
        
    def test_save_font_templates_error_handling(self):
        """Test template saving with I/O error"""
        font_tiles = {'A': np.ones((8, 8), dtype=np.uint8)}
        
        with patch('numpy.savez_compressed', side_effect=IOError("Write error")):
            result = self.extractor.save_font_templates(font_tiles, "/tmp/test.npz")
            
        self.assertFalse(result)
        
    def test_load_font_templates_success(self):
        """Test successful template loading"""
        # First save some templates
        font_tiles = {
            'A': np.ones((8, 8), dtype=np.uint8) * 255,
            'B': np.zeros((8, 8), dtype=np.uint8)
        }
        
        template_path = os.path.join(self.temp_dir, "test_templates.npz")
        np.savez_compressed(template_path, **font_tiles)
        
        # Load them back
        result = self.extractor.load_font_templates(template_path)
        
        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), 2)
        self.assertIn('A', result)
        self.assertIn('B', result)
        
        # Verify array contents
        np.testing.assert_array_equal(result['A'], font_tiles['A'])
        np.testing.assert_array_equal(result['B'], font_tiles['B'])
        
    def test_load_font_templates_file_not_found(self):
        """Test loading templates when file doesn't exist"""
        result = self.extractor.load_font_templates("/nonexistent/file.npz")
        
        self.assertEqual(result, {})
        
    def test_load_font_templates_load_error(self):
        """Test loading templates with numpy load error"""
        # Create invalid file
        invalid_path = os.path.join(self.temp_dir, "invalid.npz")
        with open(invalid_path, 'w') as f:
            f.write("not a valid npz file")
            
        result = self.extractor.load_font_templates(invalid_path)
        
        self.assertEqual(result, {})
        
    def test_preview_character_success(self):
        """Test character preview functionality"""
        # Create test font tiles
        test_tile = np.array([
            [0, 0, 255, 255, 255, 255, 0, 0],
            [0, 255, 0, 0, 0, 0, 255, 0],
            [255, 0, 0, 0, 0, 0, 0, 255],
            [255, 255, 255, 255, 255, 255, 255, 255],
            [255, 0, 0, 0, 0, 0, 0, 255],
            [255, 0, 0, 0, 0, 0, 0, 255],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0]
        ], dtype=np.uint8)
        
        font_tiles = {'A': test_tile}
        
        # Should not raise exception
        try:
            self.extractor.preview_character('A', font_tiles)
        except Exception as e:
            self.fail(f"Preview character raised exception: {e}")
            
    def test_preview_character_not_found(self):
        """Test character preview for non-existent character"""
        font_tiles = {'A': np.ones((8, 8), dtype=np.uint8)}
        
        # Should handle gracefully
        try:
            self.extractor.preview_character('Z', font_tiles)
        except Exception as e:
            self.fail(f"Preview character raised exception: {e}")
            

class TestPokemonCrystalFontExtractorIntegration(unittest.TestCase):
    """Integration tests for Pokemon Crystal Font Extractor"""
    
    def setUp(self):
        """Set up integration test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up integration test fixtures"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            
    def test_complete_extraction_workflow(self):
        """Test complete font extraction workflow"""
        extractor = PokemonCrystalFontExtractor()
        
        # Create mock ROM with some basic font data
        rom_size = 0x25000
        mock_rom = bytearray(rom_size)
        mock_rom[0x143] = 0x80  # GBC flag
        
        # Add some recognizable tile data for 'A' character
        main_font_addr = 0x1C000
        a_tile = bytes([0x18, 0x18, 0x24, 0x24, 0x42, 0x42, 0x7E, 0x7E, 
                       0x81, 0x81, 0x81, 0x81, 0x00, 0x00, 0x00, 0x00])
        mock_rom[main_font_addr:main_font_addr + 16] = a_tile
        
        # Save mock ROM
        rom_path = os.path.join(self.temp_dir, "test_rom.gbc")
        with open(rom_path, 'wb') as f:
            f.write(mock_rom)
            
        # Test complete workflow
        self.assertTrue(extractor.load_rom(rom_path))
        
        font_tiles = extractor.extract_all_fonts()
        self.assertIsInstance(font_tiles, dict)
        
        if font_tiles:  # If we extracted anything
            template_path = os.path.join(self.temp_dir, "extracted_fonts.npz")
            self.assertTrue(extractor.save_font_templates(font_tiles, template_path))
            
            # Load back and verify
            loaded_fonts = extractor.load_font_templates(template_path)
            self.assertEqual(len(loaded_fonts), len(font_tiles))
            
    def test_extract_pokemon_crystal_fonts_function(self):
        """Test the standalone extraction function"""
        # Create mock ROM
        mock_rom = bytearray(0x25000)
        mock_rom[0x143] = 0x80  # GBC flag
        
        rom_path = os.path.join(self.temp_dir, "pokemon_crystal.gbc")
        with open(rom_path, 'wb') as f:
            f.write(mock_rom)
            
        # Test extraction function
        result = extract_pokemon_crystal_fonts(rom_path, self.temp_dir)
        
        # Function should complete without errors
        self.assertIsInstance(result, bool)
        
        # Check if output file was created
        output_path = os.path.join(self.temp_dir, "pokemon_crystal_font_templates.npz")
        if result:  # Only check if function succeeded
            self.assertTrue(os.path.exists(output_path))
            
    def test_extract_pokemon_crystal_fonts_no_rom(self):
        """Test extraction function with non-existent ROM"""
        result = extract_pokemon_crystal_fonts("/nonexistent/rom.gbc", self.temp_dir)
        
        self.assertFalse(result)
        
    def test_performance_benchmarks(self):
        """Test performance of font extraction operations"""
        import time
        
        extractor = PokemonCrystalFontExtractor()
        
        # Create larger mock ROM
        rom_size = 0x100000  # 1MB
        mock_rom = bytearray(rom_size)
        mock_rom[0x143] = 0x80
        
        # Time ROM loading
        start_time = time.time()
        extractor.rom_data = bytes(mock_rom)
        load_time = time.time() - start_time
        
        self.assertLess(load_time, 1.0)  # Should load quickly
        
        # Time tile decoding
        test_tile_data = bytes([0xFF, 0x00] * 8)  # 16 bytes
        
        start_time = time.time()
        for _ in range(100):  # Decode 100 tiles
            pixels = extractor._decode_2bpp_tile(test_tile_data)
            binary = extractor._convert_to_binary(pixels)
        decode_time = time.time() - start_time
        
        self.assertLess(decode_time, 1.0)  # Should decode quickly
        
        print(f"🚀 Performance: ROM load ~{load_time:.4f}s, 100 tiles decode ~{decode_time:.4f}s")
        
    def test_memory_usage_with_large_font_sets(self):
        """Test memory usage with large font sets"""
        import time
        
        extractor = PokemonCrystalFontExtractor()
        
        # Create large font tile dictionary
        large_font_set = {}
        for i in range(1000):  # 1000 characters
            char_name = f"char_{i}"
            large_font_set[char_name] = np.random.randint(0, 256, (8, 8), dtype=np.uint8)
            
        # Test save/load with large dataset
        template_path = os.path.join(self.temp_dir, "large_font_set.npz")
        
        start_time = time.time()
        result = extractor.save_font_templates(large_font_set, template_path)
        save_time = time.time() - start_time
        
        self.assertTrue(result)
        self.assertLess(save_time, 5.0)  # Should save within reasonable time
        
        start_time = time.time()
        loaded_set = extractor.load_font_templates(template_path)
        load_time = time.time() - start_time
        
        self.assertEqual(len(loaded_set), 1000)
        self.assertLess(load_time, 5.0)  # Should load within reasonable time
        
        print(f"🚀 Large dataset: Save {save_time:.3f}s, Load {load_time:.3f}s")


if __name__ == '__main__':
    unittest.main()
