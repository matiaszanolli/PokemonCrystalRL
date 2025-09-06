"""
Pokemon Crystal RL State Management Package

This package contains all state-related functionality:
- State machine definitions and transitions
- State variable dictionaries and mappings
- Game state analysis and strategic assessment
"""

from .machine import PyBoyGameState, STATE_UI_ELEMENTS, STATE_TRANSITION_REWARDS, VALID_STATE_TRANSITIONS
from .variables import StateVariable, StateVariableDictionary, VariableType, ImpactCategory
from .analyzer import GameStateAnalyzer, GameStateAnalysis, GamePhase, SituationCriticality

__all__ = [
    # State machine
    'PyBoyGameState',
    'STATE_UI_ELEMENTS',
    'STATE_TRANSITION_REWARDS', 
    'VALID_STATE_TRANSITIONS',
    
    # State variables
    'StateVariable',
    'StateVariableDictionary',
    'VariableType',
    'ImpactCategory',
    
    # State analysis
    'GameStateAnalyzer',
    'GameStateAnalysis',
    'GamePhase',
    'SituationCriticality',
]