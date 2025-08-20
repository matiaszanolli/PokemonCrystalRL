#!/usr/bin/env python3
"""
Game state detection module for Pokemon Crystal RL agent
"""

from typing import Dict, List, Any
import numpy as np
from core.game_states import PyBoyGameState, STATE_UI_ELEMENTS

def detect_game_state(visual_context) -> PyBoyGameState:
    """Detect current game state from visual context"""
    # Handle invalid, empty, or unknown contexts
    if not visual_context:
        return PyBoyGameState.UNKNOWN
    
    if not hasattr(visual_context, 'screen_type') or not hasattr(visual_context, 'ui_elements'):
        return PyBoyGameState.UNKNOWN
        
    # Handle invalid screen dimensions or completely black/noisy screens
    if hasattr(visual_context, 'screen_type') and visual_context.screen_type == 'unknown':
        return PyBoyGameState.UNKNOWN
        
    # Check for UI elements characteristic of each state
    ui_elements = set(ui.element_type for ui in visual_context.ui_elements)
    
    # Battle state detection: requires health bars
    if 'healthbar' in ui_elements:
        return PyBoyGameState.BATTLE
    
    # Menu state detection: requires menu box
    if 'menu_box' in ui_elements:
        return PyBoyGameState.MENU
    
    # Dialogue detection: requires dialogue box
    if 'dialogue_box' in ui_elements:
        return PyBoyGameState.DIALOGUE
    
    # Fallback to game phase hints if no UI elements were definitive
    if hasattr(visual_context, 'game_phase'):
        phase = visual_context.game_phase.lower()
        if "battle" in phase:
            return PyBoyGameState.BATTLE
        elif "menu" in phase:
            return PyBoyGameState.MENU
        elif "dialogue" in phase:
            return PyBoyGameState.DIALOGUE
        elif "overworld" in phase:
            return PyBoyGameState.OVERWORLD
        elif "title" in phase:
            return PyBoyGameState.TITLE_SCREEN
    
    return PyBoyGameState.UNKNOWN

def find_ui_elements(screen: np.ndarray) -> List[Dict[str, Any]]:
    """Find UI elements in the game screen"""
    elements = []
    
    # Validate input
    if screen is None or screen.size == 0:
        return elements
        
    # Check dimensions
    if len(screen.shape) != 3 or screen.shape[2] != 3:
        return elements
        
    height, width = screen.shape[:2]
    # Check for exact Game Boy dimensions (144x160)
    if height != 144 or width != 160:
        return elements
        
    try:
        # Battle UI: health bars (specifically check green channel)
        upper_region = screen[15:25, 20:100]
        green_channel = upper_region[:, :, 1]
        
        # Check for health bar region characteristics
        mean_green = np.mean(green_channel)
        std_green = np.std(green_channel)
        if mean_green > 200 and std_green < 50:  # Strong, uniform green
            elements.append({
                "element_type": "healthbar",
                "confidence": 0.95,
                "bbox": (20, 15, 100, 25)
            })
        
        # Skip further processing if screen is completely black
        mean_intensity = np.mean(screen)
        if mean_intensity < 5:
            return elements
        
        # Dialogue box: light colored box at bottom with dark border
        lower_region = screen[100:140, 10:150]
        if lower_region.size > 0:
            # Calculate brightness and contrast
            region_mean = np.mean(lower_region, axis=2)  # Average across color channels
            max_brightness = np.max(region_mean)  # Get brightest part (dialogue box)
            min_brightness = np.min(region_mean)  # Get darkest part (text/border)
            contrast = np.std(lower_region)
            brightness_diff = max_brightness - min_brightness
            
            # Detect dialogue box with more flexible criteria
            has_bright_area = max_brightness > 180  # Bright dialogue box
            has_contrast = contrast > 20  # Some text contrast
            has_structure = brightness_diff > 50  # Clear difference between box and background
            
            if has_bright_area and has_contrast and has_structure:
                elements.append({
                    "element_type": "dialogue_box",
                    "confidence": 0.9,
                    "bbox": (10, 100, 150, 140)
                })
        
        # Menu box: high contrast area on right side with borders
        menu_region = screen[20:100, 100:150]
        if menu_region.size > 0:
            contrast = np.std(menu_region)
            brightness_var = np.var(np.mean(menu_region, axis=2))
            if contrast > 50 and contrast < 120 and brightness_var > 100:  # High contrast with structure
                elements.append({
                    "element_type": "menu_box",
                    "confidence": 0.85,
                    "bbox": (100, 20, 150, 100)
                })
    
    except (IndexError, ValueError) as e:
        # Handle any array access errors from invalid dimensions
        print(f"Error detecting UI elements: {e}")
        return elements
    
    return elements
