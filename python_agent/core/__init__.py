"""
Core module for Pokemon Crystal RL

Contains core game environments, memory mapping, and base classes.
"""

# Import memory_map first as it has no external dependencies
from .memory_map import MEMORY_ADDRESSES

__all__ = ['MEMORY_ADDRESSES']

# Try to import modules with external dependencies
try:
    from .pyboy_env import PyBoyPokemonCrystalEnv
    __all__.append('PyBoyPokemonCrystalEnv')
except ImportError:
    pass

try:
    from .env import *
except ImportError:
    pass
