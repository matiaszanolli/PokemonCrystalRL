"""
Vision module for Pokemon Crystal RL

Contains image processing, font decoding, and screen capture utilities.
"""

from vision.vision_processor import DetectedText, GameUIElement, VisualContext, PokemonVisionProcessor
from .gameboy_color_palette import GameBoyColorPalette
from .vision_enhanced_training import VisionEnhancedTrainingSession
from .pokemon_font_decoder import PokemonFontDecoder, CharacterMatch
from .rom_font_extractor import PokemonCrystalFontExtractor, FontTile, extract_pokemon_crystal_fonts

from .debug_screen_capture import _test_pyboy_screen_methods as test_pyboy_screen_methods, PYBOY_AVAILABLE

__all__ = [
    'DetectedText',
    'GameUIElement',
    'VisualContext',
    'PokemonVisionProcessor'
    'GameBoyColorPalette',
    'VisionEnhancedTrainingSession',
    'PokemonFontDecoder',
    'CharacterMatch',
    'PokemonCrystalFontExtractor',
    'FontTile',
    'extract_pokemon_crystal_fonts',
    'test_pyboy_screen_methods',
    'PYBOY_AVAILABLE',
]
