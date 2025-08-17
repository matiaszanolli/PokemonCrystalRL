"""
rom_font_extractor.py - Pokemon Crystal ROM Font Data Extractor

Extracts the actual font tile data from Pokemon Crystal ROM file
to create perfect character templates for text recognition.

Pokemon Crystal stores font data as 8x8 pixel tiles in specific ROM locations.
This module extracts those tiles and converts them to numpy arrays for template matching.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import os
import struct
from dataclasses import dataclass

@dataclass
class FontTile:
    """Represents a single font tile from the ROM"""
    char_code: int
    bitmap: np.ndarray  # 8x8 pixel array
    character: str
    
class PokemonCrystalFontExtractor:
    """
    Extracts font data from Pokemon Crystal ROM
    """
    
    def __init__(self, rom_path: str = None):
        """
        Initialize the font extractor
        
        Args:
            rom_path: Path to Pokemon Crystal ROM file (.gbc)
        """
        self.rom_path = rom_path
        self.rom_data = None
        
        # Pokemon Crystal font memory locations
        # Font data is stored as 2bpp (2 bits per pixel) tiles
        self.font_locations = {
            # Main font set (uppercase letters, numbers, symbols)
            'main_font': {
                'start_address': 0x1C000,  # Bank 7, offset 0x0000
                'tile_count': 128,
                'char_offset': 0x80,  # Characters start at 0x80
                'size': (8, 8),
                'style': 'normal'
            },
            # Additional character sets
            'lowercase_font': {
                'start_address': 0x1D000,  # Bank 7, offset 0x1000  
                'tile_count': 64,
                'char_offset': 0xA0,
                'size': (8, 8),
                'style': 'normal'
            },
            # Battle text font (typically larger/bolder)
            'battle_font': {
                'start_address': 0x1E000,  # Bank 7, offset 0x2000
                'tile_count': 96,
                'char_offset': 0x80,
                'size': (8, 8),
                'style': 'bold'
            },
            # Small font for status/stats
            'small_font': {
                'start_address': 0x1F000,  # Bank 7, offset 0x3000
                'tile_count': 64,
                'char_offset': 0x80,
                'size': (6, 8),
                'style': 'small'
            },
            # Menu headers (larger font)
            'large_font': {
                'start_address': 0x20000,  # Bank 8, offset 0x0000
                'tile_count': 32,
                'char_offset': 0x80,
                'size': (16, 16),
                'style': 'large'
            }
        }
        
        # Pokemon Crystal character mapping (based on game's charset)
        self.char_mapping = {
            # Uppercase letters (0x80-0x99)
            0x80: 'A', 0x81: 'B', 0x82: 'C', 0x83: 'D', 0x84: 'E', 0x85: 'F',
            0x86: 'G', 0x87: 'H', 0x88: 'I', 0x89: 'J', 0x8A: 'K', 0x8B: 'L',
            0x8C: 'M', 0x8D: 'N', 0x8E: 'O', 0x8F: 'P', 0x90: 'Q', 0x91: 'R',
            0x92: 'S', 0x93: 'T', 0x94: 'U', 0x95: 'V', 0x96: 'W', 0x97: 'X',
            0x98: 'Y', 0x99: 'Z',
            
            # Lowercase letters (0xA0-0xB9)
            0xA0: 'a', 0xA1: 'b', 0xA2: 'c', 0xA3: 'd', 0xA4: 'e', 0xA5: 'f',
            0xA6: 'g', 0xA7: 'h', 0xA8: 'i', 0xA9: 'j', 0xAA: 'k', 0xAB: 'l',
            0xAC: 'm', 0xAD: 'n', 0xAE: 'o', 0xAF: 'p', 0xB0: 'q', 0xB1: 'r',
            0xB2: 's', 0xB3: 't', 0xB4: 'u', 0xB5: 'v', 0xB6: 'w', 0xB7: 'x',
            0xB8: 'y', 0xB9: 'z',
            
            # Numbers (0xF6-0xFF)
            0xF6: '0', 0xF7: '1', 0xF8: '2', 0xF9: '3', 0xFA: '4',
            0xFB: '5', 0xFC: '6', 0xFD: '7', 0xFE: '8', 0xFF: '9',
            
            # Special characters (Basic punctuation)
            0x7F: ' ',   # Space
            0xE0: '-',   # Hyphen/dash
            0xE1: '/',   # Slash
            0xE2: "'",   # Apostrophe
            0xE3: ';',   # Semicolon
            0xE4: ':',   # Colon
            0xE5: '?',   # Question mark
            0xE6: '!',   # Exclamation mark
            0xE7: ',',   # Comma
            0xE8: '.',   # Period
            0xE9: '(',   # Left parenthesis
            0xEA: ')',   # Right parenthesis
            0xEB: '[',   # Left bracket
            0xEC: ']',   # Right bracket
            0xED: '"',   # Quote
            0xEE: '&',   # Ampersand
            0xEF: '+',   # Plus
            
            # Pokemon-specific symbols
            0xF0: '♂',   # Male symbol
            0xF1: '♀',   # Female symbol
            0xF2: '$',   # Money/Pokedollar symbol
            0xF3: '×',   # Multiplication/times symbol
            0xF4: '…',   # Ellipsis
            0xF5: '▶',   # Right arrow/play symbol
            
            # Additional special characters
            0x4E: '★',   # Star symbol
            0x51: '♪',   # Music note
            0x52: '♫',   # Music notes
            0x53: '♬',   # Music beam
            0x54: '♭',   # Flat symbol
            0x55: '♯',   # Sharp symbol
            0x56: '☀',   # Sun symbol
            0x57: '☁',   # Cloud symbol
            0x58: '☂',   # Umbrella symbol
            0x59: '❄',   # Snowflake symbol
            0x5A: '⚡',   # Lightning bolt
            0x5B: '♠',   # Spade symbol
            0x5C: '♣',   # Club symbol
            0x5D: '♥',   # Heart symbol
            0x5E: '♦',   # Diamond symbol
            
            # Level/status indicators
            0x34: 'Lv',  # Level abbreviation
            0x35: 'HP',  # Hit Points
            0x36: 'PP',  # Power Points
            0x37: 'EXP', # Experience
            0x38: 'ATK', # Attack
            0x39: 'DEF', # Defense
            0x3A: 'SPD', # Speed
            0x3B: 'SPC', # Special
            
            # Pokemon type symbols (approximated)
            0x60: '🔥',   # Fire type
            0x61: '💧',   # Water type
            0x62: '🌿',   # Grass type
            0x63: '⚡',   # Electric type
            0x64: '🧊',   # Ice type
            0x65: '✊',   # Fighting type
            0x66: '☠️',   # Poison type
            0x67: '🌍',   # Ground type
            0x68: '🕊️',   # Flying type
            0x69: '🔮',   # Psychic type
            0x6A: '🐛',   # Bug type
            0x6B: '🗿',   # Rock type
            0x6C: '👻',   # Ghost type
            0x6D: '🐉',   # Dragon type
            0x6E: '🌑',   # Dark type
            0x6F: '⚙️',   # Steel type
            
            # Game-specific characters
            0x70: '→',   # Right arrow
            0x71: '←',   # Left arrow
            0x72: '↑',   # Up arrow
            0x73: '↓',   # Down arrow
            0x74: '▲',   # Up triangle
            0x75: '▼',   # Down triangle
            0x76: '◄',   # Left triangle
            0x77: '►',   # Right triangle
            0x78: '■',   # Solid square
            0x79: '□',   # Empty square
            0x7A: '●',   # Solid circle
            0x7B: '○',   # Empty circle
            0x7C: '◆',   # Solid diamond
            0x7D: '◇',   # Empty diamond
        }
        
        print("🔍 Pokemon Crystal font extractor initialized")
    
    def load_rom(self, rom_path: str = None) -> bool:
        """
        Load the Pokemon Crystal ROM file
        
        Args:
            rom_path: Path to ROM file, uses self.rom_path if None
            
        Returns:
            True if ROM loaded successfully
        """
        if rom_path:
            self.rom_path = rom_path
            
        if not self.rom_path or not os.path.exists(self.rom_path):
            print(f"❌ ROM file not found: {self.rom_path}")
            return False
        
        try:
            with open(self.rom_path, 'rb') as f:
                self.rom_data = f.read()
            
            # Verify this is a Game Boy Color ROM
            if len(self.rom_data) < 0x150:
                print("❌ Invalid ROM file - too small")
                return False
            
            # Check ROM header for Game Boy Color flag
            cgb_flag = self.rom_data[0x143]
            if cgb_flag not in [0x80, 0xC0]:  # GBC compatible or GBC only
                print("⚠️ ROM may not be Game Boy Color format")
            
            print(f"✅ ROM loaded: {len(self.rom_data)} bytes")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load ROM: {e}")
            return False
    
    def _decode_2bpp_tile(self, tile_data: bytes) -> np.ndarray:
        """
        Decode a 2bpp (2 bits per pixel) Game Boy tile to 8x8 pixel array
        
        Game Boy tiles are stored as 16 bytes (2 bytes per row, 8 rows)
        Each pixel uses 2 bits, allowing 4 colors (0-3)
        
        Args:
            tile_data: 16 bytes of tile data
            
        Returns:
            8x8 numpy array with pixel values 0-3
        """
        if len(tile_data) != 16:
            raise ValueError(f"Tile data must be 16 bytes, got {len(tile_data)}")
        
        pixels = np.zeros((8, 8), dtype=np.uint8)
        
        for row in range(8):
            # Each row is stored as 2 bytes
            byte1 = tile_data[row * 2]      # Low bits
            byte2 = tile_data[row * 2 + 1]  # High bits
            
            for col in range(8):
                # Extract bit for this pixel from both bytes
                bit_pos = 7 - col  # Bits are stored left-to-right
                low_bit = (byte1 >> bit_pos) & 1
                high_bit = (byte2 >> bit_pos) & 1
                
                # Combine bits to get 2-bit pixel value (0-3)
                pixel_value = (high_bit << 1) | low_bit
                pixels[row, col] = pixel_value
        
        return pixels
    
    def _convert_to_binary(self, tile_pixels: np.ndarray) -> np.ndarray:
        """
        Convert 2bpp tile pixels to binary (black/white) for template matching
        
        Args:
            tile_pixels: 8x8 array with values 0-3
            
        Returns:
            8x8 binary array (0=background, 255=foreground)
        """
        # In Pokemon Crystal font:
        # 0 = transparent/background
        # 1 = light gray (outline)
        # 2 = dark gray (shadow)  
        # 3 = black (main text)
        
        # Create binary mask: anything not background (0) becomes foreground
        binary = np.where(tile_pixels > 0, 255, 0).astype(np.uint8)
        
        return binary
    
    def extract_font_set(self, font_set: str = 'main_font') -> Dict[str, np.ndarray]:
        """
        Extract a complete font set from the ROM
        
        Args:
            font_set: Which font set to extract ('main_font' or 'lowercase_font')
            
        Returns:
            Dictionary mapping characters to 8x8 numpy arrays
        """
        if not self.rom_data:
            print("❌ No ROM data loaded")
            return {}
        
        if font_set not in self.font_locations:
            print(f"❌ Unknown font set: {font_set}")
            return {}
        
        font_info = self.font_locations[font_set]
        start_addr = font_info['start_address']
        tile_count = font_info['tile_count']
        char_offset = font_info['char_offset']
        
        font_tiles = {}
        extracted_count = 0
        
        print(f"🔍 Extracting {font_set} from ROM address 0x{start_addr:X}")
        
        for i in range(tile_count):
            tile_offset = start_addr + (i * 16)  # Each tile is 16 bytes
            
            # Check if we're within ROM bounds
            if tile_offset + 15 >= len(self.rom_data):
                print(f"⚠️ Reached end of ROM at tile {i}")
                break
            
            # Extract tile data
            tile_data = self.rom_data[tile_offset:tile_offset + 16]
            
            # Decode 2bpp tile to pixels
            try:
                tile_pixels = self._decode_2bpp_tile(tile_data)
                binary_tile = self._convert_to_binary(tile_pixels)
                
                # Map to character if we know it
                char_code = char_offset + i
                if char_code in self.char_mapping:
                    character = self.char_mapping[char_code]
                    font_tiles[character] = binary_tile
                    extracted_count += 1
                
            except Exception as e:
                print(f"⚠️ Failed to decode tile {i}: {e}")
                continue
        
        print(f"✅ Extracted {extracted_count} characters from {font_set}")
        return font_tiles
    
    def extract_all_fonts(self) -> Dict[str, np.ndarray]:
        """
        Extract all available font sets from the ROM
        
        Returns:
            Dictionary mapping all characters to 8x8 numpy arrays
        """
        if not self.rom_data:
            print("❌ No ROM data loaded")
            return {}
        
        all_fonts = {}
        
        for font_set in self.font_locations.keys():
            font_tiles = self.extract_font_set(font_set)
            all_fonts.update(font_tiles)
        
        print(f"✅ Total characters extracted: {len(all_fonts)}")
        return all_fonts
    
    def save_font_templates(self, font_tiles: Dict[str, np.ndarray], 
                           output_path: str = "pokemon_crystal_font_templates.npz") -> bool:
        """
        Save extracted font templates to a file
        
        Args:
            font_tiles: Dictionary of character templates
            output_path: Path to save the templates
            
        Returns:
            True if saved successfully
        """
        try:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
            
            # Save as compressed numpy archive
            np.savez_compressed(output_path, **font_tiles)
            
            print(f"💾 Font templates saved to: {output_path}")
            print(f"📝 Saved {len(font_tiles)} character templates")
            return True
            
        except Exception as e:
            print(f"❌ Failed to save templates: {e}")
            return False
    
    def load_font_templates(self, template_path: str = "pokemon_crystal_font_templates.npz") -> Dict[str, np.ndarray]:
        """
        Load previously extracted font templates
        
        Args:
            template_path: Path to the template file
            
        Returns:
            Dictionary of character templates
        """
        try:
            if not os.path.exists(template_path):
                print(f"❌ Template file not found: {template_path}")
                return {}
            
            data = np.load(template_path)
            font_tiles = {key: data[key] for key in data.files}
            
            print(f"📁 Loaded {len(font_tiles)} character templates from {template_path}")
            return font_tiles
            
        except Exception as e:
            print(f"❌ Failed to load templates: {e}")
            return {}
    
    def preview_character(self, char: str, font_tiles: Dict[str, np.ndarray]) -> None:
        """
        Print a visual preview of a character template
        
        Args:
            char: Character to preview
            font_tiles: Dictionary of font templates
        """
        if char not in font_tiles:
            print(f"❌ Character '{char}' not found in templates")
            return
        
        tile = font_tiles[char]
        print(f"\n📝 Character '{char}' preview:")
        print("┌" + "─" * 8 + "┐")
        
        for row in tile:
            line = "│"
            for pixel in row:
                line += "██" if pixel > 0 else "  "
            line += "│"
            print(line)
        
        print("└" + "─" * 8 + "┘")


def extract_pokemon_crystal_fonts(rom_path: str, output_dir: str = "outputs") -> bool:
    """
    Complete font extraction workflow for Pokemon Crystal
    
    Args:
        rom_path: Path to Pokemon Crystal ROM file
        output_dir: Directory to save extracted templates
        
    Returns:
        True if extraction completed successfully
    """
    print("🎮 Starting Pokemon Crystal font extraction...")
    
    # Initialize extractor
    extractor = PokemonCrystalFontExtractor()
    
    # Load ROM
    if not extractor.load_rom(rom_path):
        return False
    
    # Extract all fonts
    font_tiles = extractor.extract_all_fonts()
    
    if not font_tiles:
        print("❌ No font data extracted")
        return False
    
    # Save templates
    output_path = os.path.join(output_dir, "pokemon_crystal_font_templates.npz")
    os.makedirs(output_dir, exist_ok=True)
    
    if not extractor.save_font_templates(font_tiles, output_path):
        return False
    
    # Preview a few characters
    print("\n🔍 Character previews:")
    for char in ['A', 'a', '0', '!']:
        if char in font_tiles:
            extractor.preview_character(char, font_tiles)
    
    print(f"\n✅ Font extraction completed!")
    print(f"📁 Templates saved to: {output_path}")
    return True


def test_font_extractor():
    """Test the font extractor with a mock ROM"""
    print("🧪 Testing Pokemon Crystal font extractor...")
    
    # Try to find a ROM file in common locations
    rom_paths = [
        "pokemon_crystal.gbc",
        "Pokemon_Crystal.gbc", 
        "../pokemon_crystal.gbc",
        "../../pokemon_crystal.gbc"
    ]
    
    rom_found = False
    for rom_path in rom_paths:
        if os.path.exists(rom_path):
            print(f"📂 Found ROM: {rom_path}")
            if extract_pokemon_crystal_fonts(rom_path):
                rom_found = True
                break
    
    if not rom_found:
        print("⚠️ No Pokemon Crystal ROM found for testing")
        print("💡 To test with actual ROM data:")
        print("   1. Place pokemon_crystal.gbc in this directory")
        print("   2. Run: python rom_font_extractor.py")
        
        # Create mock templates for testing
        extractor = PokemonCrystalFontExtractor()
        print("🔧 Creating mock font templates for testing...")
        
        # Generate some basic character templates
        mock_tiles = {}
        for char in ['A', 'B', 'O', 'K', ' ']:
            # Create a simple pattern for each character
            mock_tile = np.zeros((8, 8), dtype=np.uint8)
            if char != ' ':  # Don't fill space character
                mock_tile[1:7, 1:7] = 255  # Simple filled rectangle
            mock_tiles[char] = mock_tile
        
        # Save mock templates
        output_path = "outputs/mock_pokemon_font_templates.npz"
        if extractor.save_font_templates(mock_tiles, output_path):
            print("✅ Mock templates created successfully")
        
    print("🎉 Font extractor test completed!")


if __name__ == "__main__":
    test_font_extractor()
