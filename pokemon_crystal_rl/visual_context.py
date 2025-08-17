#!/usr/bin/env python3
"""
Visual context class for Pokemon Crystal game state detection.
"""

from typing import List, Optional
import numpy as np

class VisualContext:
    """
    A class that holds the current visual context of the game, including the screen state,
    UI elements, current game phase, and any detected text.
    """
    
    def __init__(self, screen: np.ndarray, ui_elements: List[dict], game_phase: str = "unknown", screen_type: Optional[str] = None):
        """
        Initialize a new visual context.
        
        Args:
            screen (np.ndarray): The current screen as a numpy array of shape (144, 160, 3)
            ui_elements (List[dict]): List of detected UI elements, each containing position and type information
            game_phase (str): The current phase of the game (e.g. "battle", "overworld", "dialogue")
            screen_type (str, optional): The specific type of screen being displayed
        """
        self.screen = screen
        self.ui_elements = ui_elements or []
        self.game_phase = game_phase
        self.screen_type = screen_type or game_phase

    def __eq__(self, other):
        """Compare visual contexts based on their game phase and screen type."""
        if not isinstance(other, VisualContext):
            return False
        return self.game_phase == other.game_phase and self.screen_type == other.screen_type

    def __repr__(self):
        """String representation of the visual context."""
        return f"VisualContext(phase='{self.game_phase}', type='{self.screen_type}', elements={len(self.ui_elements)})"
