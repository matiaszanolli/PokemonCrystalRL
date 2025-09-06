"""
Pokemon Crystal Screen Analysis Utilities

Functions for analyzing Game Boy screen states and content.
"""

from typing import Dict, Tuple
import numpy as np
import logging
from config.constants import SCREEN_STATES, SCREEN_DIMENSIONS

logger = logging.getLogger(__name__)

def analyze_screen_state(screen_array: np.ndarray) -> Dict[str, float]:
    """
    Analyze current screen state with robust detection.
    
    Args:
        screen_array: Numpy array of Game Boy screen pixels

    Returns:
        Dict with screen state analysis:
        - state: Current screen state (menu, battle, overworld, etc.)
        - variance: Screen pixel variance
        - colors: Number of unique colors
        - brightness: Average screen brightness
    """
    try:
        # Calculate core metrics
        variance = float(np.var(screen_array.astype(np.float32)))
        unique_colors = len(np.unique(screen_array.reshape(-1, screen_array.shape[-1]), axis=0))
        brightness = float(np.mean(screen_array.astype(np.float32)))
        
        # Determine screen state based on metrics
        state = _determine_screen_state(variance, unique_colors, brightness)
        
        return {
            'state': state,
            'variance': variance,
            'colors': unique_colors,
            'brightness': brightness
        }
        
    except Exception as e:
        logger.error(f"Error analyzing screen: {str(e)}")
        return {
            'state': SCREEN_STATES['UNKNOWN'],
            'variance': 0,
            'colors': 0,
            'brightness': 0
        }

def _determine_screen_state(variance: float, unique_colors: int, brightness: float) -> str:
    """
    Determine screen state based on analyzed metrics.
    
    Uses a combination of variance, color count, and brightness to identify different
    game states like menus, battles, overworld, etc.
    """
    # CRITICAL: Very few colors (2-3) almost always means menu/battle/evolution
    # This prevents false positives where menus are classified as overworld
    if unique_colors <= 3:
        # Very few colors - definitely not overworld
        if variance < 50:
            return SCREEN_STATES['LOADING']  # Solid colors or very simple screen
        elif brightness > 200:
            return SCREEN_STATES['DIALOGUE']  # High brightness with few colors = dialogue box
        else:
            return SCREEN_STATES['MENU']  # Low brightness with few colors = menu/battle/evolution
            
    # Very low variance = loading/transition screen
    if variance < 50:
        return SCREEN_STATES['LOADING']
        
    # Very high variance with many colors = battle screen (lots of sprites/effects)
    if variance > 20000 and unique_colors > 8:
        return SCREEN_STATES['BATTLE']
        
    # Medium-high variance with many colors = overworld
    if variance > 3000 and unique_colors > 10:
        return SCREEN_STATES['OVERWORLD']
        
    # Low variance patterns
    if variance < 3000:
        # Further distinguish between menu and dialogue
        if brightness > 200 and unique_colors < 8:
            # Very bright with few colors = likely dialogue box
            return SCREEN_STATES['DIALOGUE']
        elif unique_colors < 6:
            # Few colors = menu system
            return SCREEN_STATES['MENU']
        elif variance > 500 and unique_colors >= 8:
            # Some variance with multiple colors = likely settings/menu
            return SCREEN_STATES['SETTINGS_MENU']
        else:
            # Default to menu for low variance screens
            return SCREEN_STATES['MENU']
            
    # Medium variance with reasonable colors - could be overworld
    if unique_colors > 8:
        return SCREEN_STATES['OVERWORLD']
    else:
        return SCREEN_STATES['MENU']  # Conservative: few colors = likely menu

def has_menu_indicators(screen_array: np.ndarray) -> bool:
    """
    Check for visual indicators of menu presence.
    
    Looks for common menu elements like borders, arrows, etc.
    """
    try:
        # Check for menu borders (vertical lines)
        if np.any(np.all(screen_array[:, 0:2] == 255, axis=2)) and \
           np.any(np.all(screen_array[:, -2:] == 255, axis=2)):
            return True
            
        # Check for menu arrows (> shape at edges)
        arrow_pattern = np.array([
            [255, 255, 0],
            [255, 0, 255],
            [255, 255, 0]
        ])
        for y in range(screen_array.shape[0] - 2):
            if np.array_equal(screen_array[y:y+3, 0:3], arrow_pattern):
                return True
                
        return False
        
    except Exception as e:
        logger.error(f"Error checking menu indicators: {str(e)}")
        return False

def detect_dialogue_box(screen_array: np.ndarray) -> bool:
    """
    Check for presence of dialogue box in lower screen area.
    
    Looks for characteristic white box pattern in bottom portion.
    """
    try:
        # Focus on bottom third of screen where dialogue appears
        dialogue_area = screen_array[SCREEN_DIMENSIONS['HEIGHT']*2//3:, :, :]
        
        # Check for large white rectangular area
        white_pixels = np.all(dialogue_area > 240, axis=2)
        if np.mean(white_pixels) > 0.6:  # Over 60% white in dialogue area
            return True
            
        return False
        
    except Exception as e:
        logger.error(f"Error detecting dialogue box: {str(e)}")
        return False

def detect_battle_sprites(screen_array: np.ndarray) -> Tuple[bool, float]:
    """
    Check for presence of battle sprites and calculate battle intensity.
    
    Returns:
        Tuple of (is_battle: bool, intensity: float)
        intensity measures sprite movement/effects (0-1)
    """
    try:
        # Split screen into player and enemy areas
        player_area = screen_array[SCREEN_DIMENSIONS['HEIGHT']//2:, :SCREEN_DIMENSIONS['WIDTH']//2]
        enemy_area = screen_array[:SCREEN_DIMENSIONS['HEIGHT']//2, SCREEN_DIMENSIONS['WIDTH']//2:]
        
        # Calculate sprite presence in battle areas
        player_variance = float(np.var(player_area))
        enemy_variance = float(np.var(enemy_area))
        
        # High variance in both areas suggests active battle
        is_battle = player_variance > 5000 and enemy_variance > 5000
        
        # Calculate battle intensity from 0-1
        intensity = min((player_variance + enemy_variance) / 50000, 1.0)
        
        return is_battle, intensity
        
    except Exception as e:
        logger.error(f"Error detecting battle sprites: {str(e)}")
        return False, 0.0

def is_screen_transitioning(current: np.ndarray, previous: np.ndarray) -> bool:
    """
    Detect if screen is in transition between states.
    
    Args:
        current: Current frame array
        previous: Previous frame array
        
    Returns:
        bool indicating if screen is transitioning
    """
    try:
        if previous is None:
            return False
            
        # Calculate frame difference
        diff = np.abs(current.astype(np.float32) - previous.astype(np.float32))
        avg_diff = float(np.mean(diff))
        
        # High average difference indicates transition
        return avg_diff > 50.0
        
    except Exception as e:
        logger.error(f"Error detecting screen transition: {str(e)}")
        return False
