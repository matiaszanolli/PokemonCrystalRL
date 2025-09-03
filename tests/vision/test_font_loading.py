#!/usr/bin/env python3
"""
Test script for font data loading functionality
"""

import sys
import os
sys.path.append('.')

from vision.pokemon_font_decoder import PokemonFontDecoder
import numpy as np

def test_font_loading():
    """Test the font data loading functionality"""
    print("🧪 Testing Pokemon Font Decoder - Font Data Loading")
    print("=" * 60)
    
    # Test 1: Initialize decoder without font data (should create default)
    print("\n📝 Test 1: Initialize decoder without font data")
    decoder = PokemonFontDecoder()
    
    if decoder.is_font_loaded():
        print("✅ Font data loaded successfully")
        print(f"📊 Available characters: {len(decoder.get_available_characters())}")
        print(f"🔤 Sample characters: {', '.join(decoder.get_available_characters()[:10])}")
    else:
        print("❌ No font data loaded")
        return False
    
    # Test 2: Test character template retrieval
    print("\n📝 Test 2: Test character template retrieval")
    test_chars = ['A', 'B', 'O', '0', ' ']
    
    for char in test_chars:
        template = decoder.get_character_template(char)
        if template is not None:
            print(f"✅ Template for '{char}': shape {template.shape}, dtype {template.dtype}")
            # Verify it's 8x8 and binary
            if template.shape == (8, 8) and template.dtype == np.uint8:
                print(f"   ✓ Valid template format")
            else:
                print(f"   ❌ Invalid template format")
        else:
            print(f"❌ No template found for '{char}'")
    
    # Test 3: Test loading from specific path
    print("\n📝 Test 3: Test loading from specific path")
    
    # Create a test font file
    test_templates = {
        'X': np.array([
            [255, 0, 0, 0, 0, 0, 255, 0],
            [0, 255, 0, 0, 0, 255, 0, 0],
            [0, 0, 255, 0, 255, 0, 0, 0],
            [0, 0, 0, 255, 0, 0, 0, 0],
            [0, 0, 255, 0, 255, 0, 0, 0],
            [0, 255, 0, 0, 0, 255, 0, 0],
            [255, 0, 0, 0, 0, 0, 255, 0],
            [0, 0, 0, 0, 0, 0, 0, 0]
        ], dtype=np.uint8),
        'Y': np.array([
            [255, 0, 0, 0, 0, 0, 255, 0],
            [0, 255, 0, 0, 0, 255, 0, 0],
            [0, 0, 255, 0, 255, 0, 0, 0],
            [0, 0, 0, 255, 0, 0, 0, 0],
            [0, 0, 0, 255, 0, 0, 0, 0],
            [0, 0, 0, 255, 0, 0, 0, 0],
            [0, 0, 0, 255, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0]
        ], dtype=np.uint8)
    }
    
    test_path = "outputs/test_font_templates.npz"
    os.makedirs("outputs", exist_ok=True)
    np.savez_compressed(test_path, **test_templates)
    print(f"💾 Created test font file: {test_path}")
    
    # Test loading from specific path
    decoder2 = PokemonFontDecoder(test_path)
    
    if decoder2.is_font_loaded():
        print("✅ Successfully loaded from specific path")
        chars = decoder2.get_available_characters()
        print(f"📊 Loaded characters: {chars}")
        
        # Test the loaded templates
        for char in ['X', 'Y']:
            template = decoder2.get_character_template(char)
            if template is not None:
                print(f"✅ Template for '{char}' loaded correctly")
            else:
                print(f"❌ Template for '{char}' not found")
    else:
        print("❌ Failed to load from specific path")
    
    # Test 4: Test error handling
    print("\n📝 Test 4: Test error handling")
    
    # Test with non-existent file
    decoder3 = PokemonFontDecoder("nonexistent_file.npz")
    if not decoder3.is_font_loaded():
        print("✅ Correctly handled non-existent file")
    else:
        print("❌ Should not have loaded non-existent file")
    
    # Test with wrong file extension
    decoder4 = PokemonFontDecoder("test_file.txt")
    if not decoder4.is_font_loaded():
        print("✅ Correctly handled wrong file extension")
    else:
        print("❌ Should not have loaded wrong file type")
    
    # Test 5: Test visual preview of a character
    print("\n📝 Test 5: Visual preview of character template")
    template = decoder.get_character_template('A')
    if template is not None:
        print("Character 'A' template:")
        print("┌" + "─" * 16 + "┐")
        for row in template:
            line = "│"
            for pixel in row:
                line += "██" if pixel > 0 else "  "
            line += "│"
            print(line)
        print("└" + "─" * 16 + "┘")
    
    # Clean up test file
    if os.path.exists(test_path):
        os.remove(test_path)
        print(f"🧹 Cleaned up test file: {test_path}")
    
    print("\n🎉 Font loading tests completed!")
    return True

if __name__ == "__main__":
    success = test_font_loading()
    if success:
        print("\n✅ All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)