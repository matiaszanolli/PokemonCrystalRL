#!/usr/bin/env python3
"""Shared type definitions for Pokemon Crystal RL.

This module contains shared data classes and enums used across multiple modules
to avoid circular dependencies.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum


class PyBoyGameState(Enum):
    """Current game state from visual analysis."""
    UNKNOWN = "unknown"
    BATTLE = "battle"
    MENU = "menu"
    DIALOGUE = "dialogue"
    OVERWORLD = "overworld"


@dataclass
class DetectedText:
    """Detected text in the game screen"""
    text: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    location: str  # 'menu', 'dialogue', 'ui', 'world'


@dataclass
class GameUIElement:
    """UI element detected in the game screen"""
    element_type: str  # 'healthbar', 'menu', 'dialogue_box', 'battle_ui'
    bbox: Tuple[int, int, int, int]
    confidence: float


@dataclass
class VisualContext:
    """Complete visual analysis of a game screenshot"""
    screen_type: str  # 'overworld', 'battle', 'menu', 'dialogue', 'intro'
    detected_text: List[DetectedText]
    ui_elements: List[GameUIElement]
    dominant_colors: List[Tuple[int, int, int]]
    game_phase: str  # 'intro', 'gameplay', 'battle', 'menu_navigation'
    visual_summary: str
