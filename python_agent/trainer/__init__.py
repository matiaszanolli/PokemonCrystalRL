"""
Pokemon Crystal RL Trainer Package

A modular, extensible training system for Pokemon Crystal using PyBoy and LLMs.
"""

from .config import TrainingConfig, TrainingMode, LLMBackend
from .trainer import UnifiedPokemonTrainer
from .game_state import GameStateDetector
from .llm_manager import LLMManager
from .web_server import TrainingWebServer
from .training_strategies import TrainingStrategyManager

__version__ = "2.0.0"
__author__ = "Pokemon Crystal RL Team"

__all__ = [
    "TrainingConfig",
    "TrainingMode", 
    "LLMBackend",
    "UnifiedPokemonTrainer",
    "GameStateDetector",
    "LLMManager",
    "TrainingWebServer",
    "TrainingStrategyManager"
]
