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
    # Check for dialogue box first since it has a distinctive pattern
    has_dialogue = detect_dialogue_box(screen_array)
    if has_dialogue:
        return SCREEN_STATES['DIALOGUE']
        
    if unique_colors <= 3:
        # Very few colors - definitely not overworld
        if variance < 50:
            return SCREEN_STATES['LOADING']  # Solid colors or very simple screen
        else:
            return SCREEN_STATES['MENU']  # Few colors typically means menu
            
    # Very low variance = loading/transition screen
    if variance < 50:
        return SCREEN_STATES['LOADING']
        
    # Check for battle indicators
    is_battle, _ = detect_battle_sprites(screen_array)
    if is_battle:
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
    
    Looks for common menu elements like borders, frames, etc.
    """
    try:
        # Look for horizontal menu borders
        horizontal_lines = np.all(screen_array > 200, axis=2)
        top_border = np.any(horizontal_lines[10:30, :])  # Top menu area
        bottom_border = np.any(horizontal_lines[-30:-10, :])  # Bottom menu area
        
        # Look for vertical menu borders or side indicators
        vertical_lines = np.all(screen_array > 200, axis=2)
        left_border = np.any(vertical_lines[:, 10:30])  # Left menu area
        right_border = np.any(vertical_lines[:, -30:-10])  # Right menu area
        
        # Check for rectangular menu frame
        has_frame = (top_border and bottom_border) or (left_border and right_border)
        if has_frame:
            return True
            
        # Check for menu selection highlighting
        # Look for bright horizontal bars that could be selection indicators
        line_heights = np.mean(horizontal_lines, axis=1)
        has_highlight = np.any(line_heights > 0.7)  # At least one very bright line
        if has_highlight:
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
        
        # Check for large white or very bright rectangular area
        bright_pixels = np.all(dialogue_area > 180, axis=2)
        if np.mean(bright_pixels) > 0.4:  # Over 40% bright in dialogue area
            # Verify box-like shape by checking edges
            top_edge = bright_pixels[0:2, :]
            bottom_edge = bright_pixels[-2:, :]
            if np.mean(top_edge) > 0.7 and np.mean(bottom_edge) > 0.7:
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
        
        # Look for battle UI elements (health bars, etc)
        has_battle_ui = has_menu_indicators(screen_array)
        
        # High variance in both areas suggests active battle
        is_battle = (player_variance > 3000 and enemy_variance > 3000) or \
                   (has_battle_ui and (player_variance > 2000 or enemy_variance > 2000))
        
        # Calculate battle intensity from 0-1
        intensity = min((player_variance + enemy_variance) / 30000, 1.0)
        
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
