"""Domain-specific APIs for web monitoring.

This package provides clean API layers for different monitoring domains:
- Training statistics and control
- System status and resources
- Game state and metrics
"""

from .training import TrainingAPI
from .system import SystemAPI
from .game import GameAPI

__all__ = ['TrainingAPI', 'SystemAPI', 'GameAPI']
