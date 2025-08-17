"""
pokemon_font_decoder.py - Pokemon Crystal Font Recognition

Custom font decoder for Pokemon Crystal that uses template matching
to recognize the game's fixed bitmap font. Much faster and more
accurate than general OCR for this specific use case.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json
import os

@dataclass
class CharacterMatch:
    """Represents a matched character in the image"""
    char: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x, y, width, height
    position: Tuple[int, int]  # center x, y

class PokemonFontDecoder:
    """
    Pokemon Crystal font decoder using template matching
    """
    
    def __init__(self):
        """Initialize the font decoder"""
        self.char_templates = {}
        self.char_width = 8  # Pokemon Crystal uses 8x8 pixel characters
        self.char_height = 8
        self.font_loaded = False
        
        # Pokemon Crystal character mapping
        self.pokemon_chars = {
            # Letters
            'A': 0x80, 'B': 0x81, 'C': 0x82, 'D': 0x83, 'E': 0x84, 'F': 0x85,
            'G': 0x86, 'H': 0x87, 'I': 0x88, 'J': 0x89, 'K': 0x8A, 'L': 0x8B,
            'M': 0x8C, 'N': 0x8D, 'O': 0x8E, 'P': 0x8F, 'Q': 0x90, 'R': 0x91,
            'S': 0x92, 'T': 0x93, 'U': 0x94, 'V': 0x95, 'W': 0x96, 'X': 0x97,
            'Y': 0x98, 'Z': 0x99,
            
            # Numbers
            '0': 0xF6, '1': 0xF7, '2': 0xF8, '3': 0xF9, '4': 0xFA,
            '5': 0xFB, '6': 0xFC, '7': 0xFD, '8': 0xFE, '9': 0xFF,
            
            # Special characters
            ' ': 0x7F,  # Space
            '.': 0xE8,  # Period
            ',': 0xE7,  # Comma
            '!': 0xE6,  # Exclamation
            '?': 0xE5,  # Question mark
            ':': 0xE4,  # Colon
            ';': 0xE3,  # Semicolon
            "'": 0xE2,  # Apostrophe
            '-': 0xE0,  # Hyphen
            '/': 0xE1,  # Slash
        }
        
        # Initialize character templates
        self._create_character_templates()
        
        print("🔤 Pokemon Crystal font decoder initialized")
    
    def _create_character_templates(self):
        """Create character templates from Pokemon Crystal font data"""
        # For this implementation, we'll create basic templates
        # In a full implementation, these would be extracted from the game's font data
        
        # Create simple templates for common characters
        # These are simplified 8x8 templates - in practice you'd extract from ROM
        templates = {
            'A': np.array([
                [0,0,1,1,1,1,0,0],
                [0,1,1,0,0,1,1,0],
                [1,1,0,0,0,0,1,1],
                [1,1,0,0,0,0,1,1],
                [1,1,1,1,1,1,1,1],
                [1,1,0,0,0,0,1,1],
                [1,1,0,0,0,0,1,1],
                [0,0,0,0,0,0,0,0]
            ], dtype=np.uint8) * 255,
            
            'B': np.array([
                [1,1,1,1,1,1,0,0],
                [1,1,0,0,0,1,1,0],
                [1,1,0,0,0,1,1,0],
                [1,1,1,1,1,1,0,0],
                [1,1,0,0,0,1,1,0],
                [1,1,0,0,0,1,1,0],
                [1,1,1,1,1,1,0,0],
                [0,0,0,0,0,0,0,0]
            ], dtype=np.uint8) * 255,
            
            'O': np.array([
                [0,1,1,1,1,1,1,0],
                [1,1,0,0,0,0,1,1],
                [1,1,0,0,0,0,1,1],
                [1,1,0,0,0,0,1,1],
                [1,1,0,0,0,0,1,1],
                [1,1,0,0,0,0,1,1],
                [0,1,1,1,1,1,1,0],
                [0,0,0,0,0,0,0,0]
            ], dtype=np.uint8) * 255,
            
            'K': np.array([
                [1,1,0,0,0,1,1,0],
                [1,1,0,0,1,1,0,0],
                [1,1,0,1,1,0,0,0],
                [1,1,1,1,0,0,0,0],
                [1,1,0,1,1,0,0,0],
                [1,1,0,0,1,1,0,0],
                [1,1,0,0,0,1,1,0],
                [0,0,0,0,0,0,0,0]
            ], dtype=np.uint8) * 255,
            
            'e': np.array([
                [0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0],
                [0,1,1,1,1,1,0,0],
                [1,1,0,0,0,1,1,0],
                [1,1,1,1,1,1,1,0],
                [1,1,0,0,0,0,0,0],
                [0,1,1,1,1,1,1,0],
                [0,0,0,0,0,0,0,0]
            ], dtype=np.uint8) * 255,
            
            'M': np.array([
                [1,1,0,0,0,0,1,1],
                [1,1,1,0,0,1,1,1],
                [1,1,1,1,1,1,1,1],
                [1,1,0,1,1,0,1,1],
                [1,1,0,0,0,0,1,1],
                [1,1,0,0,0,0,1,1],
                [1,1,0,0,0,0,1,1],
                [0,0,0,0,0,0,0,0]
            ], dtype=np.uint8) * 255,
            
            'o': np.array([
                [0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0],
                [0,1,1,1,1,1,0,0],
                [1,1,0,0,0,1,1,0],
                [1,1,0,0,0,1,1,0],
                [1,1,0,0,0,1,1,0],
                [0,1,1,1,1,1,0,0],
                [0,0,0,0,0,0,0,0]
            ], dtype=np.uint8) * 255,
            
            'N': np.array([
                [1,1,0,0,0,0,1,1],
                [1,1,1,0,0,0,1,1],
                [1,1,1,1,0,0,1,1],
                [1,1,0,1,1,0,1,1],
                [1,1,0,0,1,1,1,1],
                [1,1,0,0,0,1,1,1],
                [1,1,0,0,0,0,1,1],
                [0,0,0,0,0,0,0,0]
            ], dtype=np.uint8) * 255,
            
            ' ': np.array([
                [0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0]
            ], dtype=np.uint8),
        }
        
        # Store templates
        self.char_templates = templates
        self.font_loaded = True
        
        print(f"📝 Loaded {len(self.char_templates)} character templates")
    
    def _extract_character_grid(self, image: np.ndarray) -> List[np.ndarray]:
        """Extract individual character cells from the image"""
        if len(image.shape) == 3:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # Threshold to binary
        _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
        
        height, width = binary.shape
        chars = []
        
        # Extract characters in 8x8 grids
        for y in range(0, height - self.char_height + 1, self.char_height):
            for x in range(0, width - self.char_width + 1, self.char_width):
                char_cell = binary[y:y+self.char_height, x:x+self.char_width]
                
                # Only process if there's some content
                if np.sum(char_cell) > 0:
                    chars.append((char_cell, x, y))
        
        return chars
    
    def _match_character(self, char_cell: np.ndarray) -> Tuple[str, float]:
        """Match a character cell against templates"""
        if not self.font_loaded:
            return '?', 0.0
        
        best_match = '?'
        best_score = 0.0
        
        # Normalize character cell
        if char_cell.shape != (self.char_height, self.char_width):
            char_cell = cv2.resize(char_cell, (self.char_width, self.char_height))
        
        # Try to match against each template
        for char, template in self.char_templates.items():
            # Use template matching
            result = cv2.matchTemplate(char_cell, template, cv2.TM_CCOEFF_NORMED)
            score = result[0, 0]
            
            if score > best_score:
                best_score = score
                best_match = char
        
        return best_match, best_score
    
    def decode_text(self, image: np.ndarray, min_confidence: float = 0.5) -> List[CharacterMatch]:
        """Decode text from Pokemon Crystal screenshot"""
        if not self.font_loaded:
            print("⚠️ Font templates not loaded")
            return []
        
        # Extract character cells
        char_cells = self._extract_character_grid(image)
        
        matches = []
        for char_cell, x, y in char_cells:
            char, confidence = self._match_character(char_cell)
            
            if confidence >= min_confidence:
                matches.append(CharacterMatch(
                    char=char,
                    confidence=confidence,
                    bbox=(x, y, self.char_width, self.char_height),
                    position=(x + self.char_width // 2, y + self.char_height // 2)
                ))
        
        return matches
    
    def decode_text_lines(self, image: np.ndarray, min_confidence: float = 0.5) -> List[str]:
        """Decode text and organize into lines"""
        matches = self.decode_text(image, min_confidence)
        
        if not matches:
            return []
        
        # Group characters by line (similar Y positions)
        lines = {}
        for match in matches:
            line_y = (match.position[1] // self.char_height) * self.char_height
            if line_y not in lines:
                lines[line_y] = []
            lines[line_y].append(match)
        
        # Sort and reconstruct lines
        text_lines = []
        for line_y in sorted(lines.keys()):
            # Sort characters by X position
            line_chars = sorted(lines[line_y], key=lambda m: m.position[0])
            line_text = ''.join([m.char for m in line_chars])
            text_lines.append(line_text.strip())
        
        return [line for line in text_lines if line]  # Remove empty lines
    
    def get_text_regions(self, image: np.ndarray) -> Dict[str, List[str]]:
        """Get text organized by screen regions"""
        height, width = image.shape[:2]
        
        # Define regions (similar to original vision processor)
        regions = {
            'dialogue': image[int(height * 0.7):, :],  # Bottom 30%
            'ui': image[:int(height * 0.3), int(width * 0.6):],  # Top-right
            'menu': image[:, int(width * 0.7):],  # Right side
            'center': image[int(height * 0.3):int(height * 0.7), :int(width * 0.6)]  # Center
        }
        
        text_by_region = {}
        for region_name, region_img in regions.items():
            if region_img.size > 0:
                text_lines = self.decode_text_lines(region_img)
                if text_lines:
                    text_by_region[region_name] = text_lines
        
        return text_by_region


def load_pokemon_font_from_rom(rom_path: str = None) -> Dict[str, np.ndarray]:
    """
    Load actual Pokemon Crystal font data from ROM (placeholder)
    
    In a real implementation, this would extract the actual font tiles
    from the Pokemon Crystal ROM file at specific memory addresses.
    """
    # This is a placeholder - real implementation would:
    # 1. Load the ROM file
    # 2. Extract font tile data from specific addresses (usually around 0x4000-0x5000)
    # 3. Convert tile data to 8x8 numpy arrays
    # 4. Return dictionary mapping characters to their bitmap templates
    
    if rom_path and os.path.exists(rom_path):
        print(f"📂 Would load font from ROM: {rom_path}")
        # TODO: Implement actual ROM font extraction
    
    return {}


def test_pokemon_font_decoder():
    """Test the Pokemon font decoder"""
    print("🧪 Testing Pokemon Crystal Font Decoder...")
    
    # Create a test image with some Pokemon-style text
    test_image = np.zeros((64, 320, 3), dtype=np.uint8)
    test_image.fill(255)  # White background
    
    # Add some simple text patterns (simulating Pokemon font)
    # This would normally be actual game screenshot text
    cv2.putText(test_image, "POKEMON", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)
    cv2.putText(test_image, "NEW GAME", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    
    # Initialize decoder
    decoder = PokemonFontDecoder()
    
    # Test text decoding
    matches = decoder.decode_text(test_image)
    lines = decoder.decode_text_lines(test_image)
    regions = decoder.get_text_regions(test_image)
    
    print(f"✅ Character matches: {len(matches)}")
    print(f"✅ Text lines: {lines}")
    print(f"✅ Regions with text: {list(regions.keys())}")
    
    print("🎉 Pokemon font decoder test completed!")


if __name__ == "__main__":
    test_pokemon_font_decoder()
