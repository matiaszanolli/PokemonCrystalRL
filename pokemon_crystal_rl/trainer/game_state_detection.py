"""
Game State Detection Module

Handles detection and tracking of game states in Pokemon Crystal.
"""

from enum import Enum, auto
from typing import Dict, List, Optional, Any
import numpy as np
import logging

class GameState(Enum):
    """Game states in Pokemon Crystal."""
    UNKNOWN = auto()
    OVERWORLD = auto()  
    BATTLE = auto()
    MENU = auto()
    DIALOGUE = auto()
    LOADING = auto()
    BLACK_SCREEN = auto()
    INTRO = auto()
    TRAINER_CARD = auto()

class GameStateDetector:
    """Detects game states using visual and other cues."""

    def __init__(self, debug: bool = False):
        """Initialize game state detector.
        
        Args:
            debug: Enable debug logging
        """
        self.logger = logging.getLogger(__name__)
        self.debug = debug
        self.current_state = GameState.UNKNOWN
        self.previous_state = GameState.UNKNOWN
        self.state_duration = 0
        self.state_history: List[GameState] = []
        self.max_history = 100

        self.stats = {
            'total_detections': 0,
            'state_changes': 0,
            'state_durations': {state.name: 0 for state in GameState},
            'uncertain_detections': 0
        }

    def detect_state(self, screen: np.ndarray, overlay_data: Dict[str, Any] = None) -> GameState:
        """Detect game state from screen and overlay data.
        
        Args:
            screen: Screenshot as numpy array
            overlay_data: Optional dictionary with UI info
            
        Returns:
            Detected game state
        """
        self.stats['total_detections'] += 1

        if screen is None:
            return self._update_state(GameState.UNKNOWN)

        # Basic checks
        if not self._is_valid_screen(screen):
            return self._update_state(GameState.UNKNOWN)

        # State detection logic
        if self._is_black_screen(screen):
            return self._update_state(GameState.BLACK_SCREEN)

        if self._is_loading_screen(screen):
            return self._update_state(GameState.LOADING)

        if self._has_battle_ui(screen, overlay_data):
            return self._update_state(GameState.BATTLE)

        if self._has_dialogue_box(screen, overlay_data):
            return self._update_state(GameState.DIALOGUE)

        if self._has_menu_ui(screen, overlay_data):
            return self._update_state(GameState.MENU)

        # Default to overworld if nothing specific detected
        return self._update_state(GameState.OVERWORLD)

    def get_state_duration(self) -> int:
        """Get duration of current state in frames."""
        return self.state_duration

    def get_state_history(self) -> List[GameState]:
        """Get list of recent states."""
        return self.state_history.copy()

    def get_stats(self) -> Dict[str, Any]:
        """Get detection statistics."""
        return self.stats.copy()

    def _update_state(self, new_state: GameState) -> GameState:
        """Update state tracking and history."""
        if new_state == self.current_state:
            self.state_duration += 1
        else:
            self.previous_state = self.current_state
            self.current_state = new_state
            self.state_duration = 1
            self.stats['state_changes'] += 1
            self.stats['state_durations'][new_state.name] += 1

        self.state_history.append(new_state)
        if len(self.state_history) > self.max_history:
            self.state_history.pop(0)

        return new_state

    def _is_valid_screen(self, screen: np.ndarray) -> bool:
        """Check if screen data is valid."""
        if screen is None:
            return False
        if len(screen.shape) != 3:
            return False
        height, width, channels = screen.shape
        return height >= 144 and width >= 160 and channels == 3

    def _is_black_screen(self, screen: np.ndarray) -> bool:
        """Check if screen is completely black."""
        return np.mean(screen) < 30

    def _is_loading_screen(self, screen: np.ndarray) -> bool:
        """Check if screen is a loading screen."""
        mean_bright = np.mean(screen)
        std_dev = np.std(screen)
        return mean_bright < 40 and std_dev < 20

    def _has_battle_ui(self, screen: np.ndarray, overlay_data: Optional[Dict[str, Any]]) -> bool:
        """Check for battle UI elements."""
        if overlay_data and 'battle_ui' in overlay_data:
            return True

        # Simple check for health bars
        top_area = screen[:40, :, :]
        green_mask = (top_area[:, :, 1] > 180) & (top_area[:, :, 0] < 100)
        return np.sum(green_mask) > 100

    def _has_dialogue_box(self, screen: np.ndarray, overlay_data: Optional[Dict[str, Any]]) -> bool:
        """Check for dialogue box."""
        if overlay_data and 'dialogue_box' in overlay_data:
            return True

        # Check bottom portion for text box
        bottom_third = screen[100:, :, :]
        return np.mean(bottom_third) > np.mean(screen) * 1.2

    def _has_menu_ui(self, screen: np.ndarray, overlay_data: Optional[Dict[str, Any]]) -> bool:
        """Check for menu UI elements."""
        if overlay_data and 'menu_ui' in overlay_data:
            return True

        # Check right side for menu
        right_side = screen[:, -50:, :]
        return (np.mean(right_side) < 100) or (np.mean(right_side) > 200)
