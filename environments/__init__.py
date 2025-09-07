"""
Environment Management Package

This package contains all PyBoy environment management components including:
- PyBoy environment wrappers
- Game state detection  
- Environment configuration
- State management utilities
"""

from .pyboy_env import PyBoyPokemonCrystalEnv
from .enhanced_pyboy_env import EnhancedPyBoyPokemonCrystalEnv  
from .pyboy_state_detector import PyBoyStateDetector
from .game_state_detection import GameStateDetector

__all__ = [
    'PyBoyPokemonCrystalEnv',
    'EnhancedPyBoyPokemonCrystalEnv',
    'PyBoyStateDetector', 
    'GameStateDetector'
]