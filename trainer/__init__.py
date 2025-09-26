"""
Pokemon Crystal Trainer Module

This module provides the core training systems and LLM-based decision making
for the Pokemon Crystal RL agent.
"""

# Import from new training package
from training.trainer import TrainingMode, TrainingConfig, LLMBackend, PyBoy, PYBOY_AVAILABLE, PokemonTrainer
from training.unified_pokemon_trainer import UnifiedPokemonTrainer
# Backward compatibility alias
UnifiedTrainer = UnifiedPokemonTrainer
# from training.llm_pokemon_trainer import LLMTrainer  # Archived
from training.unified_pokemon_trainer import create_llm_trainer
LLMTrainer = create_llm_trainer  # Backward compatibility factory
from training.config.training_strategies import TrainingStrategy, CurriculumStrategy

# Import from environments package  
from environments.game_state_detection import GameStateDetector

# Keep local imports
from core.dialogue_state_machine import DialogueStateMachine, DialogueState
from .llm_manager import LLMManager
# LLMAgent now imported from agents.llm_agent
from .monitoring import WebMonitor
from core.choice_recognition import ChoiceRecognitionSystem, ChoiceType, ChoicePosition, ChoiceContext, RecognizedChoice

__all__ = [
    'TrainingMode',
    'TrainingConfig',
    'UnifiedPokemonTrainer',
    'UnifiedTrainer',  # Backward compatibility alias
    'LLMTrainer',
    # 'LegacyLLMPokemonTrainer',  # Disabled due to missing dependencies
    'DialogueStateMachine',
    'DialogueState',
    'GameStateDetector',
    'LLMManager',
    # LLMAgent available from agents.llm_agent
    'LLMBackend',
    'PokemonTrainer',
    'WebMonitor',
    'TrainingStrategy',
    'CurriculumStrategy'
]
