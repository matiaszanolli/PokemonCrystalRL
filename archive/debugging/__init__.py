"""
Pokemon Crystal RL - Python Agent Package

A sophisticated AI training system that combines reinforcement learning with 
local Large Language Models to play Pokemon Crystal intelligently.
"""

__version__ = "1.0.0"
__author__ = "Pokemon Crystal RL Team"

# Core exports
from environments.pyboy_env import PyBoyPokemonCrystalEnv
from environments.state.memory_map import MEMORY_ADDRESSES
from archive.local_llm_agent import LocalLLMPokemonAgent as LocalLLMPokemonAgent
from archive.local_llm_agent import LocalLLMPokemonAgent as EnhancedLLMPokemonAgent

__all__ = [
    'PyBoyPokemonCrystalEnv',
    'MEMORY_ADDRESSES',
    'LocalLLMPokemonAgent',
    'EnhancedLLMPokemonAgent',
]
