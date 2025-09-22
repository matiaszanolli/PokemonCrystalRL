"""
Decision Analysis Package

Refactored decision analysis system split into focused modules:
- models: Data models and enums
- database: Database operations and persistence
- pattern_detector: Pattern recognition algorithms
- analyzer: Main decision analyzer orchestrator

This package maintains backward compatibility while providing 
better separation of concerns.
"""

from .models import OutcomeType, PatternType, DecisionRecord, DecisionPattern
from .database import DecisionDatabase
from .pattern_detector import PatternDetector
from .analyzer import DecisionHistoryAnalyzer

# Maintain backward compatibility
__all__ = ['DecisionHistoryAnalyzer', 'OutcomeType', 'PatternType', 'DecisionRecord', 'DecisionPattern']