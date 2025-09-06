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
# from .pokemon_trainer import LLMPokemonTrainer as LegacyLLMPokemonTrainer  # Disabled due to missing dependencies
from .llm_pokemon_trainer import LLMPokemonTrainer
# from .llm import LLMAgent  # Disabled due to import issues
from .rewards import PokemonRewardCalculator
from .monitoring import WebMonitor
from core.choice_recognition import ChoiceRecognitionSystem, ChoiceType, ChoicePosition, ChoiceContext, RecognizedChoice

__all__ = [
    'TrainingMode',
    'TrainingConfig',
    'UnifiedPokemonTrainer',
    'LLMPokemonTrainer',
    # 'LegacyLLMPokemonTrainer',  # Disabled due to missing dependencies
    'DialogueStateMachine',
    'DialogueState',
    'GameStateDetector',
    'LLMManager',
    # 'LLMAgent',  # Disabled due to import issues
    'LLMBackend',
    'PokemonTrainer',
    'PokemonRewardCalculator',
    'WebMonitor',
    'TrainingStrategy',
    'CurriculumStrategy'
]
