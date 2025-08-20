"""
Pokemon Crystal RL Trainer - Core Training Module

A comprehensive training module that orchestrates reinforcement learning
training for Pokemon Crystal with support for different training modes,
LLM integration, web monitoring, and optimized video streaming.
"""

# Import all public components
from .trainer import PokemonTrainer, TrainingConfig, TrainingMode, LLMBackend
from .unified_trainer import UnifiedPokemonTrainer
from .game_state_detection import GameStateDetector
from .llm_manager import LLMManager
from .dialogue_state_machine import DialogueStateMachine, DialogueState, NPCType
from .semantic_context_system import SemanticContextSystem, GameContext, DialogueIntent

# Version info
__version__ = "2.0.0"

__all__ = [
    'PokemonTrainer',
    'UnifiedPokemonTrainer',
    'TrainingConfig',
    'TrainingMode',
    'LLMBackend',
    'GameStateDetector',
    'LLMManager',
    'DialogueStateMachine',
    'DialogueState',
    'NPCType',
    'SemanticContextSystem',
    'GameContext',
    'DialogueIntent',
]
