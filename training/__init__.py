"""
Training Orchestration Package

This package contains all training orchestration components including:
- Main training coordinators
- Training strategies and modes
- Training configuration management  
- Training loop controllers
"""

from .strategies import TrainingStrategy
from .modes import TrainingMode
from .trainer import PokemonTrainer, TrainingConfig, TrainingMode as TrainerTrainingMode, LLMBackend
from .unified_trainer import UnifiedTrainer
from .llm_pokemon_trainer import LLMTrainer
# AdvancedHybridTrainer removed - experimental/unused

__all__ = [
    'TrainingStrategy',
    'TrainingMode', 
    'PokemonTrainer',
    'TrainingConfig',
    'TrainerTrainingMode',
    'LLMBackend',
    'UnifiedTrainer',
    'LLMTrainer',
# 'AdvancedHybridTrainer'  # Removed - experimental
]