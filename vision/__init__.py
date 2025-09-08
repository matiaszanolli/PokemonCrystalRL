"""
Pokemon Crystal Vision Processing

This package provides vision-related utilities for the Pokemon Crystal RL project,
including screen capture, image processing, and color palette management.

Reorganized structure:
- core/: Main vision processing and font decoding
- extractors/: ROM font extraction and Game Boy Color palette
- training/: Vision-enhanced training integration  
- debug/: Debug utilities
"""

# Import main functionality for backward compatibility
from .core import ROMFontDecoder, UnifiedVisionProcessor
from .extractors import PokemonCrystalFontExtractor, GameBoyColorPalette
from .training import VisionEnhancedTrainingSession

__all__ = [
    'ROMFontDecoder',
    'UnifiedVisionProcessor', 
    'PokemonCrystalFontExtractor',
    'GameBoyColorPalette',
    'VisionEnhancedTrainingSession'
]
