"""
"""
game_state.py - Game state detection and management system.

Note: This module is deprecated. Use game_states.py instead.
"""

This module handles detecting different game states (battle, menu, dialogue, etc.)
based on screen content and other indicators.
"""

from enum import Enum
import numpy as np
from typing import Optional, Dict, Any


class PyBoyGameState(Enum):
    """Possible game states"""
    UNKNOWN = "unknown"
    LOADING = "loading"
    INTRO_SEQUENCE = "intro_sequence"
    TITLE_SCREEN = "title_screen"
    MENU = "menu"
    DIALOGUE = "dialogue"
    BATTLE = "battle"
    OVERWORLD = "overworld"


class GameStateDetector:
    """Detects current game state based on screen analysis"""
    
    def __init__(self, debug_mode: bool = False):
        self.debug_mode = debug_mode
        self.last_screen_hash = None
        self.consecutive_same_screens = 0
        self.stuck_counter = 0
    
    def detect_game_state(self, screenshot: np.ndarray) -> str:
        """
        Detect game state from screenshot
        
        Args:
            screenshot: numpy array of screen pixels
            
        Returns:
            String indicating game state
        """
        if screenshot is None or screenshot.size == 0:
            return "unknown"

        # Check if image has reasonable dimensions
        if len(screenshot.shape) != 3 or screenshot.shape[0] != 144 or screenshot.shape[1] != 160:
            return "unknown"

        # Use moderate sampling to improve speed
        sample_screenshot = screenshot[::4, ::4]
        mean_brightness = np.mean(sample_screenshot)
        color_variance = np.var(sample_screenshot)

        # Very dark screen = loading
        if mean_brightness < 10:
            return "loading"

        # Very bright screen = intro sequence
        if mean_brightness > 240:
            return "intro_sequence"

        # Look for dialogue box (bright bottom section with text variance)
        height = screenshot.shape[0]
        bottom_section = screenshot[int(height * 0.8)::2, ::2]  # Sample every other pixel
        if bottom_section.size > 0:
            bottom_mean = np.mean(bottom_section)
            bottom_variance = np.var(bottom_section)

            if bottom_mean > 200 and bottom_variance > 100:  # Bright with text variance
                return "dialogue"

        # Title screen - usually bright with low variance
        if mean_brightness >= 200 and color_variance < 100:
            return "title_screen"

        # Battle screens often have distinctive brightness levels
        if abs(mean_brightness - 180.0) < 1.0 or (95 <= mean_brightness <= 105):
            return "battle"

        # Menu screens tend to have medium brightness
        if 120 <= mean_brightness <= 180:
            return "menu"

        # Overworld has various characteristics depending on location
        if color_variance > 800 and 50 < mean_brightness < 200:  # High detail outside
            return "overworld"
        elif color_variance > 400 and 60 < mean_brightness < 180:  # Medium detail inside
            return "overworld"
        elif color_variance > 200 and 40 < mean_brightness < 220:  # Simple areas
            return "overworld"

        # If nothing else matches
        return "unknown"

    def get_screen_hash(self, screenshot: Optional[np.ndarray]) -> Optional[int]:
        """Calculate a hash of the screen for stuck detection"""
        if screenshot is None or screenshot.size == 0:
            return None

        try:
            # Simple hash based on downsampled grayscale image
            gray = np.mean(screenshot, axis=2).astype(np.uint8)
            downsampled = gray[::4, ::4]  # 36x40
            hash_value = hash(downsampled.tobytes())
            return hash_value
        except Exception:
            return None

    def is_stuck(self, threshold: int = 20) -> bool:
        """
        Check if we're stuck based on consecutive identical screens
        
        Args:
            threshold: Number of identical screens to consider stuck
            
        Returns:
            True if stuck, False otherwise
        """
        return self.consecutive_same_screens >= threshold
