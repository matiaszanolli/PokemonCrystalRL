"""
Pokemon Crystal RL - Reinforcement Learning for Pokemon Crystal

This package provides interfaces and tools for training AI agents to play Pokemon Crystal.
Includes core functionality, vision processing, and monitoring capabilities.

Also includes LLM-based agents when their dependencies are available.
"""

# Core components
# Import core components that should always be available
from core.game_states import (
    PyBoyGameState,
    STATE_UI_ELEMENTS,
    STATE_TRANSITION_REWARDS
)
from core.pyboy_env import PyBoyPokemonCrystalEnv
from vision.vision_enhanced_training import VisionEnhancedTrainingSession
from monitoring.monitoring_client import MonitoringClient

__all__ = [
    'PyBoyGameState',
    'STATE_UI_ELEMENTS',
    'STATE_TRANSITION_REWARDS',
    'PyBoyPokemonCrystalEnv',
    'VisionEnhancedTrainingSession',
    'MonitoringClient',
]

# Try to import agents with external dependencies
try:
    from llm.local_llm_agent import LocalLLMPokemonAgent
    __all__.append('LocalLLMPokemonAgent')
except ImportError:
    pass

try:
    from llm.enhanced_llm_agent import EnhancedLLMPokemonAgent
    __all__.append('EnhancedLLMPokemonAgent')
except ImportError:
    pass
