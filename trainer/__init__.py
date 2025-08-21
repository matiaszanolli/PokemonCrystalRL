"""
Pokemon Crystal Trainer Module

This module provides the core training systems and LLM-based decision making
for the Pokemon Crystal RL agent.
"""

from .trainer import TrainingMode, TrainingConfig, LLMBackend, PyBoy, PYBOY_AVAILABLE, PokemonTrainer
from .unified_trainer import UnifiedPokemonTrainer
from .dialogue_state_machine import DialogueStateMachine, DialogueState
from .game_state_detection import GameStateDetector
from .llm_manager import LLMManager
from .training_strategies import TrainingStrategy, CurriculumStrategy
from .game_state_detection import GameStateDetector

__all__ = [
    'TrainingMode',
    'TrainingConfig',
    'UnifiedPokemonTrainer',
    'DialogueStateMachine',
    'DialogueState',
    'GameStateDetector',
    'LLMManager',
    'LLMBackend',
    'PokemonTrainer',
    'TrainingStrategy',
    'CurriculumStrategy'
]
