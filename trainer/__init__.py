"""
Pokemon Crystal Trainer Module

This module provides the core training systems and LLM-based decision making
for the Pokemon Crystal RL agent.
"""

# Import from new training package
from training.trainer import TrainingMode, TrainingConfig, LLMBackend, PyBoy, PYBOY_AVAILABLE, PokemonTrainer
from training.unified_trainer import UnifiedPokemonTrainer
from training.llm_pokemon_trainer import LLMPokemonTrainer
from training.strategies import TrainingStrategy, CurriculumStrategy

# Import from environments package  
from environments.game_state_detection import GameStateDetector

# Keep local imports
from .dialogue_state_machine import DialogueStateMachine, DialogueState
from .llm_manager import LLMManager
# from .llm import LLMAgent  # Disabled due to import issues
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
    'WebMonitor',
    'TrainingStrategy',
    'CurriculumStrategy'
]
