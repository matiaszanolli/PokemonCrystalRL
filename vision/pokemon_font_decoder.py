"""Pokemon font decoder module for text recognition.

This module provides functionality to decode text from Pokemon Crystal screenshots
using a custom-trained font decoder that recognizes the game's font glyphs.
"""

from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict
import numpy as np
import os
from pathlib import Path

@dataclass
class CharacterMatch:
    """A matched character from screen text."""
    char: str  
    confidence: float
    x: int
    y: int
    width: int
    height: int

class PokemonFontDecoder:
    """Decoder for Pokemon game font text."""
    
    def __init__(self, font_data_path: Optional[str] = None):
        """Initialize the font decoder.
        
        Args:
            font_data_path: Optional path to font data file
        """
        self.font_data = {}
        
        # Try to load font data if path provided
        if font_data_path:
            if not self.load_font_data(font_data_path):
                print("⚠️ Failed to load provided font data, decoder initialized empty")
        
        # If no font data loaded, try common default locations
        elif not self.font_data:
            default_paths = [
                "outputs/pokemon_crystal_font_templates.npz",
                "outputs/default_pokemon_font_templates.npz",
                "pokemon_crystal_font_templates.npz",
                "font_templates.npz"
            ]
            
            for path in default_paths:
                if os.path.exists(path):
                    print(f"🔍 Found font data at: {path}")
                    if self.load_font_data(path):
                        break
            
            # If still no font data, create default templates
            if not self.font_data:
                print("🔧 No font data found, creating default templates...")
                default_path = "outputs/default_pokemon_font_templates.npz"
                if self.create_default_font_data(default_path):
                    self.load_font_data(default_path)
    
    def load_font_data(self, path: str) -> bool:
        """Load font data from file.
        
        Args:
            path: Path to font data file (.npz format)
            
        Returns:
            True if font data loaded successfully, False otherwise
        """
        try:
            # Check if file exists
            if not os.path.exists(path):
                print(f"❌ Font data file not found: {path}")
                return False
            
            # Check file extension
            if not path.lower().endswith('.npz'):
                print(f"❌ Font data file must be .npz format, got: {path}")
                return False
            
            # Load the compressed numpy archive
            print(f"📁 Loading font data from: {path}")
            data = np.load(path)
            
            # Convert to dictionary format
            self.font_data = {}
            for key in data.files:
                char_template = data[key]
                
                # Validate template dimensions
                if char_template.shape != (8, 8):
                    print(f"⚠️ Invalid template size for '{key}': {char_template.shape}, expected (8, 8)")
                    continue
                
                # Ensure binary format (0 or 255)
                if char_template.dtype != np.uint8:
                    char_template = char_template.astype(np.uint8)
                
                # Store the character template
                self.font_data[key] = char_template
            
            print(f"✅ Loaded {len(self.font_data)} character templates")
            
            # Print some loaded characters for verification
            if self.font_data:
                sample_chars = list(self.font_data.keys())[:10]
                print(f"📝 Sample characters: {', '.join(sample_chars)}")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to load font data: {e}")
            self.font_data = {}
            return False
    
    def get_character_template(self, char: str) -> Optional[np.ndarray]:
        """Get the template for a specific character.
        
        Args:
            char: Character to get template for
            
        Returns:
            8x8 numpy array template, or None if character not found
        """
        return self.font_data.get(char, None)
    
    def get_available_characters(self) -> List[str]:
        """Get list of all available characters in the font data.
        
        Returns:
            List of character strings
        """
        return list(self.font_data.keys())
    
    def is_font_loaded(self) -> bool:
        """Check if font data is loaded.
        
        Returns:
            True if font data is available
        """
        return len(self.font_data) > 0
    
    def create_default_font_data(self, output_path: str = "outputs/default_pokemon_font_templates.npz") -> bool:
        """Create a basic default font data file for testing.
        
        Args:
            output_path: Path to save the default font data
            
        Returns:
            True if default font data created successfully
        """
        try:
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Create basic character templates
            default_templates = {}
            
            # Create simple patterns for common characters
            patterns = {
                'A': [
                    [0, 0, 255, 255, 255, 255, 0, 0],
                    [0, 255, 255, 0, 0, 255, 255, 0],
                    [255, 255, 0, 0, 0, 0, 255, 255],
                    [255, 255, 0, 0, 0, 0, 255, 255],
                    [255, 255, 255, 255, 255, 255, 255, 255],
                    [255, 255, 0, 0, 0, 0, 255, 255],
                    [255, 255, 0, 0, 0, 0, 255, 255],
                    [0, 0, 0, 0, 0, 0, 0, 0]
                ],
                'B': [
                    [255, 255, 255, 255, 255, 255, 0, 0],
                    [255, 255, 0, 0, 0, 255, 255, 0],
                    [255, 255, 0, 0, 0, 255, 255, 0],
                    [255, 255, 255, 255, 255, 255, 0, 0],
                    [255, 255, 0, 0, 0, 255, 255, 0],
                    [255, 255, 0, 0, 0, 255, 255, 0],
                    [255, 255, 255, 255, 255, 255, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0]
                ],
                'O': [
                    [0, 255, 255, 255, 255, 255, 255, 0],
                    [255, 255, 0, 0, 0, 0, 255, 255],
                    [255, 255, 0, 0, 0, 0, 255, 255],
                    [255, 255, 0, 0, 0, 0, 255, 255],
                    [255, 255, 0, 0, 0, 0, 255, 255],
                    [255, 255, 0, 0, 0, 0, 255, 255],
                    [0, 255, 255, 255, 255, 255, 255, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0]
                ],
                '0': [
                    [0, 255, 255, 255, 255, 255, 255, 0],
                    [255, 255, 0, 0, 0, 255, 255, 255],
                    [255, 255, 0, 0, 255, 255, 255, 255],
                    [255, 255, 0, 255, 255, 0, 255, 255],
                    [255, 255, 255, 255, 0, 0, 255, 255],
                    [255, 255, 255, 0, 0, 0, 255, 255],
                    [0, 255, 255, 255, 255, 255, 255, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0]
                ],
                ' ': [
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0]
                ]
            }
            
            # Convert patterns to numpy arrays
            for char, pattern in patterns.items():
                default_templates[char] = np.array(pattern, dtype=np.uint8)
            
            # Add some basic numbers and letters
            for i, char in enumerate(['1', '2', '3', '4', '5']):
                # Create simple vertical line patterns for numbers
                template = np.zeros((8, 8), dtype=np.uint8)
                template[1:7, 3:5] = 255  # Vertical line
                if i > 0:  # Add some variation
                    template[1, 2:6] = 255  # Top line
                default_templates[char] = template
            
            # Save as compressed numpy archive
            np.savez_compressed(output_path, **default_templates)
            
            print(f"💾 Default font templates saved to: {output_path}")
            print(f"📝 Created {len(default_templates)} character templates")
            return True
            
        except Exception as e:
            print(f"❌ Failed to create default font data: {e}")
            return False
    
    def detect_text(self, screenshot: np.ndarray) -> List[CharacterMatch]:
        """Detect text in screenshot and return character matches.
        
        Args:
            screenshot: RGB numpy array of size (144, 160, 3)
        
        Returns:
            List of CharacterMatch objects found in the image
        """
        # TODO: Implement text detection
        if not self.is_font_loaded():
            print("⚠️ No font data loaded for text detection")
            return []
        
        # Placeholder implementation - will be implemented in the next TODO item
        print(f"🔍 Text detection called with screenshot shape: {screenshot.shape}")
        print(f"📝 Available characters: {len(self.font_data)}")
        return []
