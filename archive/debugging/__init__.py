"""
Pokemon Crystal RL - Python Agent Package

A sophisticated AI training system that combines reinforcement learning with 
local Large Language Models to play Pokemon Crystal intelligently.
"""

__version__ = "1.0.0"
__author__ = "Pokemon Crystal RL Team"

# Core exports
from .core import PyBoyPokemonCrystalEnv, MEMORY_ADDRESSES
from .agents import LocalLLMPokemonAgent, EnhancedLLMPokemonAgent

__all__ = [
    'PyBoyPokemonCrystalEnv',
    'MEMORY_ADDRESSES',
    'LocalLLMPokemonAgent',
    'EnhancedLLMPokemonAgent',
]
