"""Test suite for utility functions."""

import pytest
import numpy as np
from utils.utils import calculate_reward
from utils.preprocess_state import preprocess_state


def test_calculate_reward_no_previous_state():
    """Test reward calculation with no previous state."""
    current_state = {
        'player': {'x': 10, 'y': 20},
        'party': [{'hp': 100, 'level': 5}]
    }
    reward = calculate_reward(current_state)
    assert reward == 0.0


def test_calculate_reward_movement():
    """Test reward calculation for movement."""
    prev_state = {'player_x': 10, 'player_y': 20}
    curr_state = {'player_x': 11, 'player_y': 20}
    
    reward = calculate_reward(curr_state, prev_state)
    assert reward == 0.1  # Movement reward
    
    # No movement
    curr_state = {'player_x': 10, 'player_y': 20}
    reward = calculate_reward(curr_state, prev_state)
    assert reward == 0.0


def test_calculate_reward_map_change():
    """Test reward calculation for map changes."""
    prev_state = {'player_map': 1}
    curr_state = {'player_map': 2}
    
    reward = calculate_reward(curr_state, prev_state)
    assert reward == 1.0  # Map change reward


def test_calculate_reward_money():
    """Test reward calculation for money changes."""
    prev_state = {'money': 1000}
    
    # Gaining money
    curr_state = {'money': 1200}
    reward = calculate_reward(curr_state, prev_state)
    assert reward == 0.5
    
    # Losing money
    curr_state = {'money': 800}
    reward = calculate_reward(curr_state, prev_state)
    assert reward == -0.2


def test_calculate_reward_battle():
    """Test reward calculation for battle states."""
    prev_state = {
        'in_battle': False,
        'enemy_hp': 100
    }
    
    # Battle start
    curr_state = {
        'in_battle': True,
        'enemy_hp': 100
    }
    reward = calculate_reward(curr_state, prev_state)
    assert reward == 0.5
    
    # Dealing damage
    prev_state = curr_state.copy()
    curr_state = {
        'in_battle': True,
        'enemy_hp': 80
    }
    reward = calculate_reward(curr_state, prev_state)
    assert reward == 0.3
    
    # Battle end
    curr_state = {
        'in_battle': False,
        'enemy_hp': 0
    }
    reward = calculate_reward(curr_state, prev_state)
    assert reward == 1.0


def test_calculate_reward_pokemon_level():
    """Test reward calculation for Pokemon leveling up."""
    prev_state = {
        'party': [
            {'level': 5, 'hp': 20},
            {'level': 10, 'hp': 30}
        ]
    }
    curr_state = {
        'party': [
            {'level': 6, 'hp': 22},  # Level up
            {'level': 10, 'hp': 30}  # No change
        ]
    }
    
    reward = calculate_reward(curr_state, prev_state)
    assert reward == 2.0  # Level up reward


def test_calculate_reward_badges():
    """Test reward calculation for obtaining badges."""
    prev_state = {'badges': 2}
    curr_state = {'badges': 3}
    
    reward = calculate_reward(curr_state, prev_state)
    assert reward == 10.0  # Badge reward


def test_calculate_reward_stuck_penalty():
    """Test reward calculation for being stuck."""
    curr_state = {'consecutive_same_screens': 51}
    
    reward = calculate_reward(curr_state, {})
    assert reward == -0.1  # Stuck penalty


def test_calculate_reward_dialog():
    """Test reward calculation for dialog progression."""
    prev_state = {'text_box_active': True}
    curr_state = {'text_box_active': False}
    
    reward = calculate_reward(curr_state, prev_state)
    assert reward == 0.2  # Dialog progression reward


def test_preprocess_state_grayscale():
    """Test state preprocessing with grayscale input."""
    screen = np.ones((144, 160), dtype=np.uint8) * 128
    processed = preprocess_state(screen)
    
    assert processed.shape == (144, 160, 3)
    assert processed.dtype == np.uint8
    assert np.all(processed == 128)


def test_preprocess_state_rgba():
    """Test state preprocessing with RGBA input."""
    screen = np.ones((144, 160, 4), dtype=np.uint8) * 128
    processed = preprocess_state(screen)
    
    assert processed.shape == (144, 160, 3)
    assert processed.dtype == np.uint8
    assert np.all(processed == 128)


def test_preprocess_state_rgb():
    """Test state preprocessing with RGB input."""
    screen = np.ones((144, 160, 3), dtype=np.uint8) * 128
    processed = preprocess_state(screen)
    
    assert processed.shape == (144, 160, 3)
    assert processed.dtype == np.uint8
    assert np.all(processed == 128)


def test_preprocess_state_invalid():
    """Test state preprocessing with invalid input."""
    # Invalid shape
    screen = np.ones((144, 160, 5), dtype=np.uint8)
    with pytest.raises(ValueError):
        preprocess_state(screen)
    
    # Invalid type
    screen = "not an array"
    with pytest.raises(ValueError):
        preprocess_state(screen)


def test_preprocess_state_float():
    """Test state preprocessing with float input."""
    screen = np.ones((144, 160, 3), dtype=np.float32) * 0.5
    processed = preprocess_state(screen)
    
    assert processed.shape == (144, 160, 3)
    assert processed.dtype == np.uint8
    assert np.all(processed == 127)  # 0.5 * 255 ≈ 127
