"""
Game Intelligence Module

Modular game intelligence system for Pokemon Crystal RL.

This module provides:
- Location analysis and strategic recommendations
- Progress tracking and objective determination
- Battle strategy and type effectiveness
- Inventory and item management
- Coordinated game context analysis

Architecture:
- location.py: Location types, context, and analysis
- progression.py: Game phase and goal tracking
- battle.py: Battle strategy and move recommendations
- inventory.py: Item management and usage strategies
- orchestrator.py: Main GameIntelligence coordinator
"""

# Location intelligence
from .location import (
    LocationType,
    IntelligenceGameContext,
    GameContext,  # Backward compatibility alias
    ActionPlan,
    LocationAnalyzer
)

# Progression tracking
from .progression import ProgressTracker

# Battle strategy
from .battle import BattleStrategy

# Inventory management
from .inventory import InventoryManager

# Main orchestrator
from .orchestrator import GameIntelligence

__all__ = [
    # Location
    'LocationType',
    'IntelligenceGameContext',
    'GameContext',
    'ActionPlan',
    'LocationAnalyzer',
    # Progression
    'ProgressTracker',
    # Battle
    'BattleStrategy',
    # Inventory
    'InventoryManager',
    # Orchestrator
    'GameIntelligence',
]
