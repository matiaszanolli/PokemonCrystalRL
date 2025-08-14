"""
enhanced_font_decoder.py - Pokemon Crystal ROM-Based Font Decoder

Uses actual font tiles extracted from Pokemon Crystal ROM for perfect text recognition.
This replaces the custom pattern matching with precise template matching using real game fonts.
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Set
import os
from collections import defaultdict
import json

from rom_font_extractor import PokemonCrystalFontExtractor


class ROMFontDecoder:
    """
    Enhanced font decoder using actual Pokemon Crystal ROM font data
    """
    
    def __init__(self, template_path: str = None, rom_path: str = None):
        """
        Initialize the ROM-based font decoder
        
        Args:
            template_path: Path to extracted font templates (.npz file)
            rom_path: Path to Pokemon Crystal ROM file (for fresh extraction)
        """
        self.font_templates: Dict[str, np.ndarray] = {}
        self.template_path = template_path
        self.rom_path = rom_path
        
        # Character recognition statistics
        self.recognition_stats = {
            'total_attempts': 0,
            'successful_matches': 0,
            'failed_matches': 0,
            'confidence_scores': []
        }
        
        # Load or extract font templates
        self._load_font_templates()
        
        print(f"🎮 ROM Font Decoder initialized with {len(self.font_templates)} characters")
    
    def _load_font_templates(self) -> bool:
        """
        Load font templates from file or extract from ROM
        
        Returns:
            True if templates loaded successfully
        """
        # Try to load existing templates first
        if self.template_path and os.path.exists(self.template_path):
            extractor = PokemonCrystalFontExtractor()
            self.font_templates = extractor.load_font_templates(self.template_path)
            
            if self.font_templates:
                print(f"📁 Loaded existing templates: {len(self.font_templates)} characters")
                return True
        
        # If no templates or ROM path provided, try standard locations
        if not self.rom_path:
            rom_candidates = [
                "pokemon_crystal.gbc",
                "Pokemon_Crystal.gbc",
                "../pokemon_crystal.gbc", 
                "../../pokemon_crystal.gbc",
                "outputs/pokemon_crystal_font_templates.npz",
                "pokemon_crystal_font_templates.npz"
            ]
            
            # Check for existing template files
            for candidate in rom_candidates:
                if candidate.endswith('.npz') and os.path.exists(candidate):
                    extractor = PokemonCrystalFontExtractor()
                    self.font_templates = extractor.load_font_templates(candidate)
                    if self.font_templates:
                        print(f"📁 Loaded templates from: {candidate}")
                        return True
            
            # Check for ROM files
            for candidate in rom_candidates:
                if candidate.endswith('.gbc') and os.path.exists(candidate):
                    self.rom_path = candidate
                    break
        
        # Extract from ROM if available
        if self.rom_path and os.path.exists(self.rom_path):
            print(f"🔍 Extracting fonts from ROM: {self.rom_path}")
            extractor = PokemonCrystalFontExtractor()
            
            if extractor.load_rom(self.rom_path):
                self.font_templates = extractor.extract_all_fonts()
                
                # Save extracted templates
                output_path = "outputs/pokemon_crystal_font_templates.npz"
                extractor.save_font_templates(self.font_templates, output_path)
                
                if self.font_templates:
                    print(f"✅ Extracted {len(self.font_templates)} characters from ROM")
                    return True
        
        # Fall back to creating basic templates
        print("⚠️ No ROM or templates found, creating basic fallback templates")
        self._create_fallback_templates()
        return len(self.font_templates) > 0
    
    def _create_fallback_templates(self) -> None:
        """Create basic fallback templates when ROM data isn't available"""
        # Create simple 8x8 templates for common characters
        fallback_chars = {
            ' ': np.zeros((8, 8), dtype=np.uint8),  # Space
            'A': self._create_letter_a(),
            'B': self._create_letter_b(),
            'O': self._create_letter_o(),
            'K': self._create_letter_k(),
            'E': self._create_letter_e(),
            '!': self._create_exclamation(),
            '?': self._create_question(),
        }
        
        self.font_templates.update(fallback_chars)
        print(f"🔧 Created {len(fallback_chars)} fallback templates")
    
    def _create_letter_a(self) -> np.ndarray:
        """Create basic 'A' template"""
        a = np.zeros((8, 8), dtype=np.uint8)
        a[1:7, 2:6] = 255  # Main body
        a[3, 2:6] = 0      # Horizontal line gap
        return a
    
    def _create_letter_b(self) -> np.ndarray:
        """Create basic 'B' template"""
        b = np.zeros((8, 8), dtype=np.uint8)
        b[1:7, 1:5] = 255  # Main body
        return b
    
    def _create_letter_o(self) -> np.ndarray:
        """Create basic 'O' template"""
        o = np.zeros((8, 8), dtype=np.uint8)
        o[2:6, 2:6] = 255  # Outer rectangle
        o[3:5, 3:5] = 0    # Inner hollow
        return o
    
    def _create_letter_k(self) -> np.ndarray:
        """Create basic 'K' template"""
        k = np.zeros((8, 8), dtype=np.uint8)
        k[1:7, 1] = 255    # Vertical line
        k[3, 2:5] = 255    # Horizontal center
        k[2, 3] = 255      # Upper diagonal
        k[4, 3] = 255      # Lower diagonal
        return k
    
    def _create_letter_e(self) -> np.ndarray:
        """Create basic 'E' template"""
        e = np.zeros((8, 8), dtype=np.uint8)
        e[1:7, 1:5] = 255  # Main rectangle
        e[2:6, 3:5] = 0    # Right side hollow
        e[3, 3:5] = 255    # Middle horizontal line
        return e
    
    def _create_exclamation(self) -> np.ndarray:
        """Create basic '!' template"""
        ex = np.zeros((8, 8), dtype=np.uint8)
        ex[1:5, 3:5] = 255  # Top part
        ex[6, 3:5] = 255    # Bottom dot
        return ex
    
    def _create_question(self) -> np.ndarray:
        """Create basic '?' template"""
        q = np.zeros((8, 8), dtype=np.uint8)
        q[1:3, 2:6] = 255   # Top curve
        q[3:5, 4:6] = 255   # Right side
        q[6, 3:5] = 255     # Bottom dot
        return q
    
    def _normalize_tile(self, tile: np.ndarray) -> np.ndarray:
        """
        Normalize a tile for template matching
        
        Args:
            tile: 8x8 pixel tile
            
        Returns:
            Normalized binary tile
        """
        if tile.shape != (8, 8):
            tile = cv2.resize(tile, (8, 8), interpolation=cv2.INTER_NEAREST)
        
        # Convert to binary
        if tile.max() > 1:
            binary = np.where(tile > 128, 255, 0).astype(np.uint8)
        else:
            binary = (tile * 255).astype(np.uint8)
        
        return binary
    
    def _calculate_match_score(self, tile: np.ndarray, template: np.ndarray) -> float:
        """
        Calculate similarity score between tile and template
        
        Args:
            tile: Input tile to match
            template: Character template
            
        Returns:
            Similarity score (0.0 to 1.0, higher is better)
        """
        # Normalize both tiles
        tile_norm = self._normalize_tile(tile)
        template_norm = self._normalize_tile(template)
        
        # Calculate multiple similarity metrics
        scores = []
        
        # 1. Template matching with normalized cross-correlation
        try:
            result = cv2.matchTemplate(tile_norm, template_norm, cv2.TM_CCOEFF_NORMED)
            ncc_score = float(result[0, 0]) if result.size > 0 else 0.0
            scores.append(max(0.0, ncc_score))
        except:
            scores.append(0.0)
        
        # 2. Pixel-wise exact match percentage
        exact_matches = np.sum(tile_norm == template_norm)
        exact_score = exact_matches / 64.0  # 8x8 = 64 pixels
        scores.append(exact_score)
        
        # 3. Structural similarity (simplified)
        # Count foreground pixels in both
        tile_fg = np.sum(tile_norm > 0)
        template_fg = np.sum(template_norm > 0)
        
        if tile_fg == 0 and template_fg == 0:
            struct_score = 1.0  # Both empty
        elif tile_fg == 0 or template_fg == 0:
            struct_score = 0.0  # One empty, one not
        else:
            # Ratio similarity
            ratio = min(tile_fg, template_fg) / max(tile_fg, template_fg)
            struct_score = ratio
        
        scores.append(struct_score)
        
        # Weighted average of all scores
        weights = [0.5, 0.3, 0.2]  # Favor template matching
        final_score = sum(w * s for w, s in zip(weights, scores))
        
        return final_score
    
    def recognize_character(self, tile: np.ndarray, min_confidence: float = 0.6) -> Tuple[str, float]:
        """
        Recognize a single character from an 8x8 tile
        
        Args:
            tile: 8x8 pixel tile to recognize
            min_confidence: Minimum confidence threshold
            
        Returns:
            Tuple of (character, confidence_score)
        """
        self.recognition_stats['total_attempts'] += 1
        
        if not self.font_templates:
            return '?', 0.0
        
        best_char = '?'
        best_score = 0.0
        
        # Test against all templates
        for char, template in self.font_templates.items():
            score = self._calculate_match_score(tile, template)
            
            if score > best_score:
                best_score = score
                best_char = char
        
        # Record statistics
        self.recognition_stats['confidence_scores'].append(best_score)
        
        if best_score >= min_confidence:
            self.recognition_stats['successful_matches'] += 1
            return best_char, best_score
        else:
            self.recognition_stats['failed_matches'] += 1
            return '?', best_score
    
    def decode_text_region(self, text_region: np.ndarray, char_width: int = 8, 
                          char_height: int = 8, min_confidence: float = 0.6) -> str:
        """
        Decode text from a text region using ROM font templates
        
        Args:
            text_region: Image region containing text
            char_width: Width of each character in pixels
            char_height: Height of each character in pixels
            min_confidence: Minimum confidence for character recognition
            
        Returns:
            Decoded text string
        """
        if text_region is None or text_region.size == 0:
            return ""
        
        # Convert to grayscale if needed
        if len(text_region.shape) == 3:
            text_region = cv2.cvtColor(text_region, cv2.COLOR_BGR2GRAY)
        
        height, width = text_region.shape
        
        # Calculate number of characters that fit
        chars_x = width // char_width
        chars_y = height // char_height
        
        if chars_x == 0 or chars_y == 0:
            return ""
        
        decoded_lines = []
        
        # Process each row of characters
        for row in range(chars_y):
            line_chars = []
            
            # Process each character in the row
            for col in range(chars_x):
                # Extract character tile
                y_start = row * char_height
                y_end = y_start + char_height
                x_start = col * char_width
                x_end = x_start + char_width
                
                char_tile = text_region[y_start:y_end, x_start:x_end]
                
                # Recognize character
                char, confidence = self.recognize_character(char_tile, min_confidence)
                line_chars.append(char)
            
            # Join characters and clean up the line
            line = ''.join(line_chars).rstrip()
            if line:  # Only add non-empty lines
                decoded_lines.append(line)
        
        # Join lines with newlines
        result = '\n'.join(decoded_lines).strip()
        return result
    
    def get_recognition_stats(self) -> Dict:
        """Get recognition performance statistics"""
        stats = self.recognition_stats.copy()
        
        if stats['confidence_scores']:
            stats['average_confidence'] = np.mean(stats['confidence_scores'])
            stats['min_confidence'] = np.min(stats['confidence_scores'])
            stats['max_confidence'] = np.max(stats['confidence_scores'])
        else:
            stats['average_confidence'] = 0.0
            stats['min_confidence'] = 0.0
            stats['max_confidence'] = 0.0
        
        if stats['total_attempts'] > 0:
            stats['success_rate'] = stats['successful_matches'] / stats['total_attempts']
        else:
            stats['success_rate'] = 0.0
        
        return stats
    
    def reset_stats(self) -> None:
        """Reset recognition statistics"""
        self.recognition_stats = {
            'total_attempts': 0,
            'successful_matches': 0,
            'failed_matches': 0,
            'confidence_scores': []
        }
    
    def save_templates(self, output_path: str) -> bool:
        """
        Save current font templates to file
        
        Args:
            output_path: Path to save templates
            
        Returns:
            True if saved successfully
        """
        try:
            os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
            np.savez_compressed(output_path, **self.font_templates)
            print(f"💾 Font templates saved to: {output_path}")
            return True
        except Exception as e:
            print(f"❌ Failed to save templates: {e}")
            return False
    
    def add_custom_template(self, char: str, template: np.ndarray) -> None:
        """
        Add a custom character template
        
        Args:
            char: Character to add
            template: 8x8 template array
        """
        if template.shape != (8, 8):
            print(f"⚠️ Template for '{char}' must be 8x8, got {template.shape}")
            return
        
        self.font_templates[char] = self._normalize_tile(template)
        print(f"✅ Added custom template for '{char}'")
    
    def preview_template(self, char: str) -> None:
        """
        Print a visual preview of a character template
        
        Args:
            char: Character to preview
        """
        if char not in self.font_templates:
            print(f"❌ No template found for '{char}'")
            return
        
        template = self.font_templates[char]
        print(f"\n📝 Template for '{char}':")
        print("┌────────┐")
        
        for row in template:
            line = "│"
            for pixel in row:
                line += "██" if pixel > 0 else "  "
            line += "│"
            print(line)
        
        print("└────────┘")


def test_rom_font_decoder():
    """Test the ROM font decoder"""
    print("🧪 Testing ROM Font Decoder...")
    
    # Initialize decoder
    decoder = ROMFontDecoder()
    
    # Test character recognition with a mock tile
    test_tile = np.zeros((8, 8), dtype=np.uint8)
    test_tile[1:7, 2:6] = 255  # Simple rectangle
    
    char, confidence = decoder.recognize_character(test_tile)
    print(f"🔍 Test tile recognized as: '{char}' (confidence: {confidence:.2f})")
    
    # Preview some templates
    if decoder.font_templates:
        print("\n🔍 Template previews:")
        for char in ['A', 'O', ' ', '!']:
            if char in decoder.font_templates:
                decoder.preview_template(char)
    
    # Test text region decoding
    mock_region = np.zeros((16, 32), dtype=np.uint8)  # 2x4 characters
    mock_region[0:8, 0:8] = decoder.font_templates.get('O', np.zeros((8, 8)))
    mock_region[0:8, 8:16] = decoder.font_templates.get('K', np.zeros((8, 8)))
    
    decoded = decoder.decode_text_region(mock_region)
    print(f"\n📝 Mock region decoded as: '{decoded}'")
    
    # Show statistics
    stats = decoder.get_recognition_stats()
    print(f"\n📊 Recognition Stats:")
    print(f"   Total attempts: {stats['total_attempts']}")
    print(f"   Success rate: {stats['success_rate']:.2%}")
    print(f"   Average confidence: {stats['average_confidence']:.2f}")
    
    print("✅ ROM Font Decoder test completed!")


if __name__ == "__main__":
    test_rom_font_decoder()
