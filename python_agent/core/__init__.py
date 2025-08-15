"""
Core module for Pokemon Crystal RL

Contains core game environments, memory mapping, and base classes.
"""

# Import memory_map first as it has no external dependencies
from .memory_map import MEMORY_ADDRESSES

# Import PyBoy environment (dependencies are now properly handled)
from .pyboy_env import PyBoyPokemonCrystalEnv

__all__ = ['MEMORY_ADDRESSES', 'PyBoyPokemonCrystalEnv']

# Try to import optional modules
try:
    from .env import *
except ImportError:
    pass
