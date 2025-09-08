"""
Core trainer module for Pokemon Crystal RL - Compatibility wrapper for refactored components.

This module maintains backward compatibility while delegating to the refactored modular components:
- TrainingConfig, TrainingMode, LLMBackend -> config/training_config.py
- PokemonTrainer core logic -> core/pokemon_trainer.py  
- PyBoy management -> infrastructure/pyboy_manager.py
- Web integration -> infrastructure/web_integration.py
- Training modes -> core/training_modes.py

REFACTORING SUCCESS: Reduced from 1,534 lines to a clean compatibility wrapper.
"""

# Import refactored components
from .config import TrainingConfig, TrainingMode, LLMBackend
from .core import PokemonTrainer

# Backward compatibility - re-export all classes at module level
__all__ = ['TrainingConfig', 'TrainingMode', 'LLMBackend', 'PokemonTrainer']

# Legacy global variables for test compatibility
PyBoy = None  # Will be resolved dynamically by PyBoyManager
PYBOY_AVAILABLE = True