#!/usr/bin/env python3
"""
Core Game Intelligence Module for Pokemon Crystal

This module has been refactored into a modular package structure.
All functionality is now available through core.intelligence submodules.

For backward compatibility, all classes and types are re-exported here.

New modular structure:
- core/intelligence/location.py - Location analysis and context
- core/intelligence/progression.py - Progress tracking
- core/intelligence/battle.py - Battle strategy
- core/intelligence/inventory.py - Item management
- core/intelligence/orchestrator.py - Main coordinator

Migration guide:
- Old: from core.game_intelligence import GameIntelligence
- New: from core.intelligence import GameIntelligence (or use this file)
"""

# Re-export all public classes and types for backward compatibility
from .intelligence import (
    # Location intelligence
    LocationType,
    IntelligenceGameContext,
    GameContext,
    ActionPlan,
    LocationAnalyzer,
    # Progression tracking
    ProgressTracker,
    # Battle strategy
    BattleStrategy,
    # Inventory management
    InventoryManager,
    # Main orchestrator
    GameIntelligence,
)

__all__ = [
    'LocationType',
    'IntelligenceGameContext',
    'GameContext',
    'ActionPlan',
    'LocationAnalyzer',
    'ProgressTracker',
    'BattleStrategy',
    'InventoryManager',
    'GameIntelligence',
]
