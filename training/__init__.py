"""
Training Orchestration Package

This package contains all training orchestration components including:
- Main training coordinators
- Training strategies and modes
- Training configuration management  
- Training loop controllers
"""

from .config.training_strategies import TrainingStrategy
from .config.training_modes import TrainingMode
from .trainer import PokemonTrainer, TrainingConfig, TrainingMode as TrainerTrainingMode, LLMBackend
from .unified_pokemon_trainer import UnifiedPokemonTrainer
# Backward compatibility alias
UnifiedTrainer = UnifiedPokemonTrainer
# from .llm_pokemon_trainer import LLMTrainer  # Archived - use UnifiedPokemonTrainer instead
from .unified_pokemon_trainer import create_llm_trainer
LLMTrainer = create_llm_trainer  # Backward compatibility factory
# AdvancedHybridTrainer removed - experimental/unused

__all__ = [
    'TrainingStrategy',
    'TrainingMode', 
    'PokemonTrainer',
    'TrainingConfig',
    'TrainerTrainingMode',
    'LLMBackend',
    'UnifiedPokemonTrainer',
    'UnifiedTrainer',  # Backward compatibility alias
    'LLMTrainer',
# 'AdvancedHybridTrainer'  # Removed - experimental
]