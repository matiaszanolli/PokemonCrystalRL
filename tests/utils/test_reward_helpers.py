"""
Tests for Pokemon Crystal reward calculation utilities.
"""

import pytest
from typing import Dict, Set
from copy import deepcopy

from utils.reward_helpers import (
    calculate_hp_reward,
    calculate_level_reward,
    calculate_badge_reward,
    calculate_exploration_reward,
    calculate_blocked_movement_penalty,
    get_reward_summary
)

from tests.fixtures.pyboy_helpers import create_game_state
from config.constants import SCREEN_STATES, REWARD_VALUES

@pytest.fixture
def basic_game_state():
    """Basic game state with a Pokemon."""
    return create_game_state(
        party_count=1,
        player_level=5,
        player_hp=20,
        player_max_hp=24,
        player_status=0
    )

@pytest.fixture
def exploration_state():
    """Game state for testing exploration rewards."""
    return create_game_state(
        player_map=26,  # New Bark Town
        player_x=10,
        player_y=12,
        party_count=1
    )

def test_calculate_hp_reward_empty():
    """Test HP rewards with no Pokemon."""
    current = create_game_state()
    previous = create_game_state()
    assert calculate_hp_reward(current, previous) == 0.0

def test_calculate_hp_reward_healing():
    """Test HP rewards for healing."""
    current = create_game_state(
        party_count=1,
        player_hp=20,
        player_max_hp=24
    )
    previous = create_game_state(
        party_count=1,
        player_hp=16,
        player_max_hp=24
    )
    
    reward = calculate_hp_reward(current, previous)
    assert reward > 0
    assert reward == ((20/24 - 16/24) * 5.0)  # Healing reward

def test_calculate_hp_reward_damage():
    """Test HP rewards for taking damage."""
    current = create_game_state(
        party_count=1,
        player_hp=16,
        player_max_hp=24
    )
    previous = create_game_state(
        party_count=1,
        player_hp=20,
        player_max_hp=24
    )
    
    reward = calculate_hp_reward(current, previous)
    assert reward < 0
    assert reward == ((16/24 - 20/24) * 10.0)  # Damage penalty

def test_calculate_level_reward_invalid():
    """Test level rewards with invalid states."""
    current = create_game_state(player_level=0)
    previous = create_game_state(player_level=0)
    
    # No Pokemon = no reward
    assert calculate_level_reward(current, previous) == 0.0
    
    # Invalid levels = no reward
    current['party_count'] = 1
    current['player_level'] = 101
    assert calculate_level_reward(current, previous) == 0.0

def test_calculate_level_reward_gain():
    """Test level rewards for gaining levels."""
    current = create_game_state(
        party_count=1,
        player_level=6,
        player_hp=25,
        player_max_hp=30
    )
    previous = create_game_state(
        party_count=1,
        player_level=5,
        player_hp=20,
        player_max_hp=24
    )
    
    reward = calculate_level_reward(current, previous)
    assert reward > 0
    assert reward == 50.0  # One level gain

def test_calculate_level_reward_screen_validation():
    """Test level rewards respect screen state."""
    current = create_game_state(
        party_count=1,
        player_level=6
    )
    previous = create_game_state(
        party_count=1,
        player_level=5
    )
    
    # Only reward in overworld
    assert calculate_level_reward(
        current, previous,
        SCREEN_STATES['OVERWORLD'],
        SCREEN_STATES['OVERWORLD']
    ) == 50.0
    
    # No reward in menus
    assert calculate_level_reward(
        current, previous,
        SCREEN_STATES['MENU'],
        SCREEN_STATES['MENU']
    ) == 0.0

def test_calculate_badge_reward():
    """Test badge earning rewards."""
    current = create_game_state(
        party_count=1,
        badges=0x01,  # First badge
        badges_total=1
    )
    previous = create_game_state(
        party_count=1,
        badges=0x00,
        badges_total=0
    )
    
    milestones = set()
    reward = calculate_badge_reward(
        current, previous,
        SCREEN_STATES['OVERWORLD'],
        SCREEN_STATES['OVERWORLD'],
        milestones
    )
    assert reward == REWARD_VALUES['BADGE']
    assert len(milestones) == 1  # Should track milestone
    
    # No duplicate rewards
    reward = calculate_badge_reward(
        current, previous,
        SCREEN_STATES['OVERWORLD'],
        SCREEN_STATES['OVERWORLD'],
        milestones
    )
    assert reward == 0.0

def test_calculate_exploration_reward():
    """Test exploration rewards."""
    current = create_game_state(
        player_map=26,
        player_x=10,
        player_y=12
    )
    previous = create_game_state(
        player_map=26,
        player_x=10,
        player_y=11
    )
    
    visited_maps = set()
    visited_locations = set()
    
    # First visit to location
    reward = calculate_exploration_reward(
        current, previous,
        SCREEN_STATES['OVERWORLD'],
        step_counter=100,
        visited_maps=visited_maps,
        visited_locations=visited_locations
    )
    assert reward == REWARD_VALUES['NEW_LOCATION']
    assert len(visited_locations) == 1
    
    # Already visited
    reward = calculate_exploration_reward(
        current, previous,
        SCREEN_STATES['OVERWORLD'],
        step_counter=101,
        visited_maps=visited_maps,
        visited_locations=visited_locations
    )
    assert reward == 0.0

def test_calculate_exploration_reward_new_map():
    """Test rewards for discovering new maps."""
    current = create_game_state(
        player_map=27,  # Prof Elm's Lab
        player_x=5,
        player_y=5
    )
    previous = create_game_state(
        player_map=26,  # New Bark Town
        player_x=15,
        player_y=5
    )
    
    visited_maps = set()
    visited_locations = set()
    
    reward = calculate_exploration_reward(
        current, previous,
        SCREEN_STATES['OVERWORLD'],
        step_counter=100,
        visited_maps=visited_maps,
        visited_locations=visited_locations
    )
    assert reward == REWARD_VALUES['NEW_MAP']
    assert 27 in visited_maps

def test_calculate_blocked_movement_penalty():
    """Test penalties for trying blocked movements."""
    current = create_game_state(
        player_map=26,
        player_x=10,
        player_y=12
    )
    previous = deepcopy(current)  # Same position = blocked
    
    blocked_tracker = {}
    
    # First blocked attempt
    penalty = calculate_blocked_movement_penalty(
        current, previous,
        last_action='up',
        curr_screen=SCREEN_STATES['OVERWORLD'],
        blocked_tracker=blocked_tracker
    )
    assert penalty == -0.005
    
    # Repeated blocked attempts
    for _ in range(4):
        penalty = calculate_blocked_movement_penalty(
            current, previous,
            last_action='up',
            curr_screen=SCREEN_STATES['OVERWORLD'],
            blocked_tracker=blocked_tracker
        )
    
    # Should reach max penalty
    assert penalty == REWARD_VALUES['MAX_BLOCKED_PENALTY']

def test_get_reward_summary():
    """Test reward summary formatting."""
    rewards = {
        'level': 50.0,
        'health': -5.0,
        'exploration': 0.1,
        'time': -0.01  # Too small to show
    }
    
    summary = get_reward_summary(rewards)
    assert 'level: +50.00' in summary
    assert 'health: -5.00' in summary
    assert 'exploration: +0.10' in summary
    assert 'time' not in summary  # Too small
    
    # Empty rewards
    assert get_reward_summary({}) == 'no rewards'
