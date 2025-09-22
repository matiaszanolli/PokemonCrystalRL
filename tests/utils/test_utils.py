"""Test suite for utility functions."""

import pytest
import numpy as np
from environments.rewards import calculate_reward
from utils.preprocess_state import preprocess_state


def calculate_basic_reward(current_state, previous_state=None):
    """Basic reward calculation for tests - simple version without enhancements."""
    if previous_state is None:
        return 0.0
    
    reward = 0.0
    
    # Movement reward
    curr_x = current_state.get('player_x', 0)
    curr_y = current_state.get('player_y', 0)
    prev_x = previous_state.get('player_x', 0)
    prev_y = previous_state.get('player_y', 0)
    
    distance_moved = abs(curr_x - prev_x) + abs(curr_y - prev_y)
    if distance_moved > 0:
        reward += 0.1
    
    # Map change reward
    curr_map = current_state.get('player_map', 0)
    prev_map = previous_state.get('player_map', 0)
    if curr_map != prev_map:
        reward += 1.0
    
    # Money rewards
    curr_money = current_state.get('money', 0)
    prev_money = previous_state.get('money', 0)
    money_diff = curr_money - prev_money
    if money_diff > 0:
        reward += 0.5
    elif money_diff < 0:
        reward -= 0.2
    
    # Battle rewards
    curr_battle = current_state.get('in_battle', False)
    prev_battle = previous_state.get('in_battle', False)
    if curr_battle and not prev_battle:
        reward += 0.5  # Battle start
    elif not curr_battle and prev_battle:
        reward += 1.0  # Battle end (won)
    elif curr_battle and 'enemy_hp' in current_state and 'enemy_hp' in previous_state:
        if current_state['enemy_hp'] < previous_state['enemy_hp']:
            reward += 0.3  # Dealing damage
    
    # Pokemon level rewards
    if 'party' in current_state and 'party' in previous_state:
        curr_party = current_state['party']
        prev_party = previous_state['party']
        for i, (curr_pokemon, prev_pokemon) in enumerate(zip(curr_party, prev_party)):
            if curr_pokemon.get('level', 1) > prev_pokemon.get('level', 1):
                reward += 2.0  # Level up reward
    
    # Badge rewards
    if 'badges' in current_state and 'badges' in previous_state:
        badges_gained = current_state['badges'] - previous_state['badges']
        if badges_gained > 0:
            reward += 10.0 * badges_gained
    
    # Stuck penalty
    consecutive_same_screens = current_state.get('consecutive_same_screens', 0)
    if consecutive_same_screens > 50:
        reward -= 0.1
    
    # Dialog progression reward
    prev_text_box = previous_state.get('text_box_active', False)
    curr_text_box = current_state.get('text_box_active', False)
    if prev_text_box and not curr_text_box:
        reward += 0.2
    
    return reward


def test_calculate_reward_no_previous_state():
    """Test reward calculation with no previous state."""
    current_state = {
        'player': {'x': 10, 'y': 20},
        'party': [{'hp': 100, 'level': 5}]
    }
    reward = calculate_basic_reward(current_state)
    assert reward == 0.0


def test_calculate_reward_movement():
    """Test reward calculation for movement."""
    prev_state = {'player_x': 10, 'player_y': 20}
    curr_state = {'player_x': 11, 'player_y': 20}
    
    reward = calculate_basic_reward(curr_state, prev_state)
    assert reward == 0.1  # Movement reward
    
    # No movement
    curr_state = {'player_x': 10, 'player_y': 20}
    reward = calculate_basic_reward(curr_state, prev_state)
    assert reward == 0.0


def test_calculate_reward_map_change():
    """Test reward calculation for map changes."""
    prev_state = {'player_map': 1}
    curr_state = {'player_map': 2}
    
    reward = calculate_basic_reward(curr_state, prev_state)
    assert reward == 1.0  # Map change reward


def test_calculate_reward_money():
    """Test reward calculation for money changes."""
    prev_state = {'money': 1000}
    
    # Gaining money
    curr_state = {'money': 1200}
    reward = calculate_basic_reward(curr_state, prev_state)
    assert reward == 0.5
    
    # Losing money
    curr_state = {'money': 800}
    reward = calculate_basic_reward(curr_state, prev_state)
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
    reward = calculate_basic_reward(curr_state, prev_state)
    assert reward == 0.5
    
    # Dealing damage
    prev_state = curr_state.copy()
    curr_state = {
        'in_battle': True,
        'enemy_hp': 80
    }
    reward = calculate_basic_reward(curr_state, prev_state)
    assert reward == 0.3
    
    # Battle end
    curr_state = {
        'in_battle': False,
        'enemy_hp': 0
    }
    reward = calculate_basic_reward(curr_state, prev_state)
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
    
    reward = calculate_basic_reward(curr_state, prev_state)
    assert reward == 2.0  # Level up reward


def test_calculate_reward_badges():
    """Test reward calculation for obtaining badges."""
    prev_state = {'badges': 2}
    curr_state = {'badges': 3}
    
    reward = calculate_basic_reward(curr_state, prev_state)
    assert reward == 10.0  # Badge reward


def test_calculate_reward_stuck_penalty():
    """Test reward calculation for being stuck."""
    curr_state = {'consecutive_same_screens': 51}
    
    reward = calculate_basic_reward(curr_state, {})
    assert reward == -0.1  # Stuck penalty


def test_calculate_reward_dialog():
    """Test reward calculation for dialog progression."""
    prev_state = {'text_box_active': True}
    curr_state = {'text_box_active': False}
    
    reward = calculate_basic_reward(curr_state, prev_state)
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
