"""
Pokemon Crystal RL - Reinforcement Learning for Pokemon Crystal

This package provides interfaces and tools for training AI agents to play Pokemon Crystal.
Includes core functionality, vision processing, and monitoring capabilities.

Also includes LLM-based agents when their dependencies are available.
"""

# Core components
# Import core components that should always be available
from pokemon_crystal_rl.core.pyboy_env import PyBoyPokemonCrystalEnv
from pokemon_crystal_rl.vision.vision_enhanced_training import VisionEnhancedTrainingSession
from pokemon_crystal_rl.monitoring.monitoring_client import MonitoringClient

__all__ = [
    'PyBoyPokemonCrystalEnv',
    'VisionEnhancedTrainingSession',
    'MonitoringClient',
]

# Try to import agents with external dependencies
try:
    from pokemon_crystal_rl.llm.local_llm_agent import LocalLLMPokemonAgent
    __all__.append('LocalLLMPokemonAgent')
except ImportError:
    pass

try:
    from pokemon_crystal_rl.llm.enhanced_llm_agent import EnhancedLLMPokemonAgent
    __all__.append('EnhancedLLMPokemonAgent')
except ImportError:
    pass
