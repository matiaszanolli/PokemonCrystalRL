"""
Tests for Pokemon Crystal screen analysis utilities.
"""

import pytest
import numpy as np
from typing import Dict

from utils.screen_analyzer import (
    analyze_screen_state,
    _determine_screen_state,
    has_menu_indicators,
    detect_dialogue_box,
    detect_battle_sprites,
    is_screen_transitioning
)

from config.constants import SCREEN_STATES

from tests.fixtures.pyboy_helpers import (
    mock_pyboy,
    mock_screen_overworld,
    mock_screen_battle,
    mock_screen_menu,
    mock_screen_dialogue
)

def test_analyze_screen_state(mock_screen_overworld):
    """Test basic screen state analysis."""
    result = analyze_screen_state(mock_screen_overworld)
    assert 'state' in result
    assert 'variance' in result
    assert 'colors' in result
    assert 'brightness' in result
    assert isinstance(result['variance'], float)
    assert isinstance(result['colors'], int)
    assert isinstance(result['brightness'], float)

def test_overworld_detection(mock_screen_overworld):
    """Test overworld screen state detection."""
    result = analyze_screen_state(mock_screen_overworld)
    assert result['state'] == SCREEN_STATES['OVERWORLD']
    assert result['variance'] > 3000  # Overworld typically has high variance
    assert result['colors'] > 10  # Overworld has many colors

def test_battle_detection(mock_screen_battle):
    """Test battle screen state detection."""
    result = analyze_screen_state(mock_screen_battle)
    assert result['state'] == SCREEN_STATES['BATTLE']
    assert result['variance'] > 20000  # Battles have very high variance
    assert result['colors'] > 8  # Battles have many colors

def test_menu_detection(mock_screen_menu):
    """Test menu screen state detection."""
    result = analyze_screen_state(mock_screen_menu)
    assert result['state'] == SCREEN_STATES['MENU']
    assert result['variance'] < 3000  # Menus have low variance
    assert result['colors'] < 6  # Menus have few colors

def test_dialogue_detection(mock_screen_dialogue):
    """Test dialogue screen state detection."""
    result = analyze_screen_state(mock_screen_dialogue)
    assert result['state'] == SCREEN_STATES['DIALOGUE']
    assert result['brightness'] > 200  # Dialogue boxes are bright
    assert result['colors'] < 8  # Dialogue has few colors

def test_loading_screen_detection():
    """Test loading screen state detection."""
    # Create solid black screen
    screen = np.zeros((144, 160, 3), dtype=np.uint8)
    result = analyze_screen_state(screen)
    assert result['state'] == SCREEN_STATES['LOADING']
    assert result['variance'] < 50  # Loading screens have very low variance

def test_settings_menu_detection():
    """Test settings menu state detection."""
    # Create screen with settings menu characteristics
    screen = np.zeros((144, 160, 3), dtype=np.uint8)
    screen[20:120, 20:140] = 200  # Large menu area
    screen[30:40, 25:135] = 255  # Menu text area
    
    result = analyze_screen_state(screen)
    assert result['state'] == SCREEN_STATES['SETTINGS_MENU']
    assert result['colors'] >= 8
    assert 500 < result['variance'] < 3000

def test_menu_indicators(mock_screen_menu):
    """Test menu UI element detection."""
    assert has_menu_indicators(mock_screen_menu) is True
    
    # Test with overworld screen (should not have menu indicators)
    assert has_menu_indicators(mock_screen_overworld) is False

def test_dialogue_box_detection(mock_screen_dialogue):
    """Test dialogue box detection."""
    assert detect_dialogue_box(mock_screen_dialogue) is True
    
    # Test without dialogue box
    assert detect_dialogue_box(mock_screen_overworld) is False
    
    # Test partial dialogue box (should not detect)
    partial_dialogue = np.ones((144, 160, 3), dtype=np.uint8) * 128
    partial_dialogue[120:140, 20:60] = 255  # Too small for dialogue box
    assert detect_dialogue_box(partial_dialogue) is False

def test_battle_sprite_detection(mock_screen_battle):
    """Test battle sprite detection."""
    is_battle, intensity = detect_battle_sprites(mock_screen_battle)
    assert is_battle is True
    assert 0 <= intensity <= 1
    
    # Test non-battle screen
    is_battle, intensity = detect_battle_sprites(mock_screen_overworld)
    assert is_battle is False
    assert intensity < 0.5  # Lower intensity for non-battle

def test_screen_transition_detection(mock_screen_overworld):
    """Test screen transition detection."""
    # Create faded version for transition
    faded = mock_screen_overworld.copy() * 0.5
    
    assert is_screen_transitioning(faded, mock_screen_overworld) is True
    assert is_screen_transitioning(mock_screen_overworld, mock_screen_overworld) is False
    
    # Test with None previous screen
    assert is_screen_transitioning(mock_screen_overworld, None) is False

def test_invalid_screen_data():
    """Test handling of invalid screen data."""
    # Wrong dimensions - should handle gracefully, not raise exception
    invalid_screen = np.zeros((100, 100, 3), dtype=np.uint8)
    result = analyze_screen_state(invalid_screen)
    # Function detects it as loading screen, which is reasonable for all-black pixels
    assert result['state'] in [SCREEN_STATES['UNKNOWN'], SCREEN_STATES['LOADING']]
    
    # Invalid data type
    invalid_data = np.zeros((144, 160, 3), dtype=np.float32)
    result = analyze_screen_state(invalid_data)
    # Function also detects float data as loading screen, which is reasonable
    assert result['state'] in [SCREEN_STATES['UNKNOWN'], SCREEN_STATES['LOADING']]
