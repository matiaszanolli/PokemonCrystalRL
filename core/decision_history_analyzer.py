#!/usr/bin/env python3
"""
Decision History Analyzer - REFACTORED COMPATIBILITY WRAPPER

### REFACTORING NOTICE ###
This module has been refactored into a modular package structure for better
maintainability and separation of concerns. The functionality is now split across:

- core.decision_analysis.models: Data models and enums
- core.decision_analysis.database: Database operations and persistence  
- core.decision_analysis.pattern_detector: Pattern recognition algorithms
- core.decision_analysis.analyzer: Main decision analyzer orchestrator

This file now serves as a compatibility wrapper to maintain backward compatibility.
All functionality has been preserved while providing cleaner architecture.
"""

# Import the new modular components
from .decision_analysis import (
    DecisionHistoryAnalyzer, 
    OutcomeType, 
    PatternType, 
    DecisionRecord, 
    DecisionPattern
)

# Legacy compatibility - re-export everything that was previously available
__all__ = ['DecisionHistoryAnalyzer', 'OutcomeType', 'PatternType', 'DecisionRecord', 'DecisionPattern']