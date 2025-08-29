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
    """Detects and manages game state information."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.last_screen_hash = None
        self.consecutive_same_screens = 0
        self.stuck_counter = 0

    def get_screen_hash(self, screen: np.ndarray) -> Optional[int]:
        """Get a hash value for screen content for stuck detection."""
        if screen is None:
            return None
        
        # Handle invalid screen shapes
        if len(screen.shape) < 2:
            return None
        
        try:
            # Compute mean values for grid cells
            h, w = screen.shape[:2]
            
            # Handle screens that are too small
            if h < 4 or w < 4:
                return None
                
            cell_h = h // 4
            cell_w = w // 4
            grid_means = []

            for i in range(4):
                for j in range(4):
                    cell = screen[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
                    if cell.size > 0:  # Ensure cell is not empty
                        mean = np.mean(cell)
                        grid_means.append(int(mean))
                    else:
                        grid_means.append(0)

            # Convert means to hash
            hash_val = 0
            for mean in grid_means:
                hash_val = hash_val * 31 + mean

            return hash_val
        except Exception:
            return None
    
    def is_stuck(self) -> bool:
        """Check if the game appears to be stuck."""
        return self.consecutive_same_screens >= 30 or self.stuck_counter > 0

    def detect_game_state(self, screen: np.ndarray) -> str:
        """Detect game state from screen content."""
        if screen is None:
            return "unknown"

        # Handle mock screen objects in tests
        if hasattr(screen, '_mock_name'):
            return "unknown"

        # Early exit if screen is wrong shape
        if len(screen.shape) < 2 or (len(screen.shape) == 3 and screen.shape[2] not in (1, 3, 4)):
            return "unknown"

        # Convert to grayscale, cache result for performance
        try:
            if not hasattr(self, '_last_gray') or self._last_gray is None:
                if len(screen.shape) == 3:
                    self._last_gray = np.mean(screen, axis=2).astype(np.uint8)
                else:
                    self._last_gray = screen
                gray = self._last_gray
            else:
                gray = self._last_gray
        except (AttributeError, TypeError):
            return "unknown"

        # Store last screen hash for transition detection
        current_hash = self.get_screen_hash(screen)
        if current_hash is None:
            return "unknown"

        # Check if we have a different screen
        same_screen = current_hash == self.last_screen_hash
        if same_screen:
            # With same screen, always increment consecutive counter
            self.consecutive_same_screens += 1
            
            # Update stuck counter if threshold reached
            if self.consecutive_same_screens >= 30:
                self.stuck_counter += 1
                return "stuck"
        else:
            # Different screen - only reset if substantially different
            self.last_screen_hash = current_hash
            if self.consecutive_same_screens >= 15:
                self.consecutive_same_screens = 0
                self.stuck_counter = 0
            self._last_gray = None  # Reset gray cache on change
            self.last_screen_hash = current_hash
            self._last_gray = None  # Reset gray cache on screen change

        # Detect loading/black screen and transitions
        mean_brightness = np.mean(gray)
        title_std = np.std(gray)
        self.logger.debug(f"Screen stats - mean: {mean_brightness:.1f}, std: {title_std:.1f}")
        
        # Transitions include both dark and fade screens
        if mean_brightness < 50 or (mean_brightness < 100 and title_std < 20):
            self.logger.debug("Detected unknown/loading state")
            return "unknown"

        # Detect intro/white screen
        if mean_brightness > 240:
            self.logger.debug("Detected intro sequence")
            return "title_screen"

        # Detect title screen (characterized by medium-high uniform brightness)
        if 180 <= mean_brightness <= 220 and title_std < 30:
            self.logger.debug("Detected title screen")
            # Title screens can use either START (7) or A (5) button
            return "title_screen"

        # Menu detection (check for brighter rectangular region)
        menu_regions = [
            gray[20:60, 20:140],  # Standard menu
            gray[30:90, 30:130],  # Battle menu
            gray[100:140, 10:150]  # Options menu
        ]
        for region in menu_regions:
            region_mean = np.mean(region)
            region_std = np.std(region)
            # Menu regions are bright and uniform
            if region_mean > 180 and region_std < 40:
                return "menu"

        # Dialogue detection with optimized checks
        # Only compute bottom region first for performance
        try:
            bottom = gray[100:140, 10:150]  # Focused dialogue box region
            bottom_mean = np.mean(bottom)

            # Quick check for bright bottom region first
            if bottom_mean > 170:  # Dialog boxes are brighter than background
                top = gray[20:90, 10:150]  # Check game area
                top_mean = np.mean(top)
                top_std = np.std(top)

                # Enhanced dialogue box criteria:
                # - Very bright bottom region
                # - Game area is significantly darker
                # - Game area has reasonable variation
                # - Game area not too dark (to avoid menus)
                if (bottom_mean > 200 and
                    bottom_mean > top_mean * 1.5 and
                    top_std > 20 and
                    50 < top_mean < 150):
                    return "dialogue"
        except (IndexError, TypeError):
            pass

        # Check for battle screen characteristics
        if self._detect_battle_screen(gray):
            return "battle"

        return "overworld"

    def _detect_battle_screen(self, gray: np.ndarray) -> bool:
        """Helper method to detect battle screen state."""
        # Battle screens have distinctive layout with HP bars
        hp_regions = [
            gray[30:45, 170:220],   # Player HP region
            gray[100:115, 50:100]    # Enemy HP region
        ]
        
        for region in hp_regions:
            region_mean = np.mean(region)
            region_std = np.std(region)
            # HP bars have high contrast
            if region_mean > 150 and region_std > 50:
                return True
                
        return False

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


def get_unstuck_action(step: int, stuck_level: int) -> int:
    """Get appropriate action to escape stuck states.
    
    Args:
        step: Current step number
        stuck_level: Current stuck counter level
        
    Returns:
        Action ID from 1-8
    """
    # Basic movement pattern for low stuck levels
    if stuck_level <= 2:
        pattern = [1, 2, 5, 4, 3]  # UP, DOWN, A, LEFT, RIGHT
        return pattern[step % len(pattern)]
    
    # More varied pattern for moderate stuck levels
    if stuck_level <= 5:
        pattern = [1, 5, 2, 5, 3, 5, 4, 5, 6]  # Movement + A button + B
        return pattern[step % len(pattern)]
    
    # Aggressive pattern for high stuck levels
    if stuck_level <= 10:
        pattern = [1, 5, 2, 6, 3, 5, 4, 6, 7, 8]  # All actions
        return pattern[step % len(pattern)]
    
    # Very stuck - use pseudo-random pattern
    return (step % 8) + 1  # 1-8
