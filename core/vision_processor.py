"""
vision_processor.py - Computer Vision for Pokemon Crystal Agent

This module processes PyBoy screenshots to extract visual context
for the LLM agent, including text recognition, UI elements detection,
and game state analysis.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

# Visual context classes
@dataclass
class DetectedText:
    """Represents detected text in the game"""
    text: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    location: str  # 'menu', 'dialogue', 'ui', 'world'


@dataclass
class GameUIElement:
    """Represents a UI element detected in the game"""
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
