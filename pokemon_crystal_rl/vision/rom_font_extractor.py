"""ROM font extractor for Pokemon Crystal.

This module extracts font data directly from the Pokemon Crystal ROM file.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, BinaryIO
import numpy as np

@dataclass
class FontTile:
    """A single font tile from the ROM."""
    char: str
    data: np.ndarray
    width: int
    height: int

def extract_pokemon_crystal_fonts(rom_path: str) -> List[FontTile]:
    """Extract fonts from Pokemon Crystal ROM.
    
    Args:
        rom_path: Path to Pokemon Crystal ROM file
    
    Returns:
        List of FontTile objects
    """
    # TODO: Implement font extraction
    return []

def test_font_extractor(rom_path: str):
    """Test the font extractor on a ROM file."""
    tiles = extract_pokemon_crystal_fonts(rom_path)
    print(f"Extracted {len(tiles)} font tiles")

class PokemonCrystalFontExtractor:
    """Extracts font data from Pokemon Crystal ROM."""
    
    def __init__(self, rom_path: str):
        """Initialize the font extractor.
        
        Args:
            rom_path: Path to Pokemon Crystal ROM
        """
        self.rom_path = rom_path
        self.tiles: List[FontTile] = []
    
    def extract_fonts(self) -> List[FontTile]:
        """Extract fonts from the ROM file."""
        self.tiles = extract_pokemon_crystal_fonts(self.rom_path)
        return self.tiles
