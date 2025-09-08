"""
Unit Tests for Reward Components

This module provides comprehensive tests for the reward component system,
testing both individual components and their integration in the calculator.
"""

from typing import Dict

import os
import sys
import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from rewards.component import RewardComponent
from rewards.components.interaction import (
    BattleRewardComponent,
    DialogueRewardComponent,
    MoneyRewardComponent,
    ProgressionRewardComponent
)
from rewards.components.movement import (
    BlockedMovementComponent,
    ExplorationRewardComponent,
    MovementRewardComponent,
)
from rewards.components.progress import (
    BadgeRewardComponent,
    HealthRewardComponent,
    LevelRewardComponent,
)

# Test utilities and fixtures

@pytest.fixture
def base_state() -> Dict:
    """Provide a base game state with valid ranges."""
    return {
        'party_count': 1,
        'player_map': 1,
        'player_x': 100,
        'player_y': 100,
        'money': 1000,
        'badges_total': 0,
        'badges': 0,
        'in_battle': 0,
        'player_level': 5,
        'player_hp': 20,
        'player_max_hp': 24
    }

def setup_component_state(component: RewardComponent, screen_state: str = 'overworld'):
    """Set up component with screen state."""
    component.last_screen_state = screen_state
    component.prev_screen_state = screen_state

# Health Component Tests

def test_health_reward_no_pokemon():
    """Should not give health rewards without Pokemon."""
    component = HealthRewardComponent()
    current = {'party_count': 0}
    previous = {'party_count': 0}
    
    reward, details = component.calculate(current, previous)
    assert reward == 0.0
    assert not details

def test_health_reward_healing():
    """Should reward healing."""
    component = HealthRewardComponent()
    current = {
        'party_count': 1,
        'player_hp': 20,
        'player_max_hp': 24
    }
    previous = {
        'party_count': 1,
        'player_hp': 12,
        'player_max_hp': 24
    }
    
    reward, details = component.calculate(current, previous)
    assert reward > 0.0
    assert 'healing' in details

def test_health_reward_damage():
    """Should penalize taking damage."""
    component = HealthRewardComponent()
    current = {
        'party_count': 1,
        'player_hp': 12,
        'player_max_hp': 24
    }
    previous = {
        'party_count': 1,
        'player_hp': 20,
        'player_max_hp': 24
    }
    
    reward, details = component.calculate(current, previous)
    assert reward < 0.0
    assert 'damage' in details

# Level Component Tests

def test_level_reward_invalid_screen():
    """Should not give level rewards in non-overworld states."""
    component = LevelRewardComponent()
    setup_component_state(component, 'menu')
    
    current = {
        'party_count': 1,
        'player_level': 6
    }
    previous = {
        'party_count': 1,
        'player_level': 5
    }
    
    reward, details = component.calculate(current, previous)
    assert reward == 0.0
    assert not details

def test_level_reward_valid_increase():
    """Should reward valid level increases."""
    component = LevelRewardComponent()
    setup_component_state(component, 'overworld')
    
    current = {
        'party_count': 1,
        'player_level': 6,
        'player_hp': 25,
        'player_max_hp': 28
    }
    previous = {
        'party_count': 1,
        'player_level': 5,
        'player_hp': 20,
        'player_max_hp': 24
    }
    
    reward, details = component.calculate(current, previous)
    assert reward > 0.0
    assert 'level_up' in details

def test_level_reward_impossible_spike():
    """Should reject impossible level spikes."""
    component = LevelRewardComponent()
    setup_component_state(component, 'overworld')
    
    current = {
        'party_count': 1,
        'player_level': 50
    }
    previous = {
        'party_count': 1,
        'player_level': 5
    }
    
    reward, details = component.calculate(current, previous)
    assert reward == 0.0
    assert not details

# Badge Component Tests

def test_badge_reward_first_badge():
    """Should give big reward for first badge."""
    component = BadgeRewardComponent()
    setup_component_state(component, 'overworld')
    
    current = {
        'party_count': 1,
        'badges_total': 1,
        'badges': 1
    }
    previous = {
        'party_count': 1,
        'badges_total': 0,
        'badges': 0
    }
    
    reward, details = component.calculate(current, previous)
    assert reward > 0.0
    assert 'badge_earned' in details

def test_badge_reward_invalid_early():
    """Should reject badges without Pokemon."""
    component = BadgeRewardComponent()
    setup_component_state(component, 'overworld')
    
    current = {
        'party_count': 0,
        'badges_total': 1,
        'badges': 1
    }
    previous = {
        'party_count': 0,
        'badges_total': 0,
        'badges': 0
    }
    
    reward, details = component.calculate(current, previous)
    assert reward == 0.0
    assert not details

# Exploration Component Tests

def test_exploration_new_map():
    """Should reward entering new maps."""
    component = ExplorationRewardComponent()
    setup_component_state(component, 'overworld')
    
    current = {
        'player_map': 2,
        'player_x': 5,
        'player_y': 5
    }
    previous = {
        'player_map': 1,
        'player_x': 250,
        'player_y': 250
    }
    
    reward, details = component.calculate(current, previous)
    assert reward > 0.0
    assert 'new_map' in details

def test_exploration_reject_suspicious_jump():
    """Should reject suspicious map transitions."""
    component = ExplorationRewardComponent()
    setup_component_state(component, 'overworld')
    
    current = {
        'player_map': 100,
        'player_x': 5,
        'player_y': 5
    }
    previous = {
        'player_map': 1,
        'player_x': 5,
        'player_y': 5
    }
    
    reward, details = component.calculate(current, previous)
    assert reward == 0.0
    assert not details

# Movement Component Tests

def test_movement_basic_step():
    """Should reward basic movement."""
    component = MovementRewardComponent()
    setup_component_state(component, 'overworld')
    
    current = {
        'player_map': 1,
        'player_x': 101,
        'player_y': 100
    }
    previous = {
        'player_map': 1,
        'player_x': 100,
        'player_y': 100
    }
    
    reward, details = component.calculate(current, previous)
    assert reward > 0.0
    assert 'movement' in details

def test_movement_farming_penalty():
    """Should penalize movement farming."""
    component = MovementRewardComponent()
    setup_component_state(component, 'overworld')
    
    # Oscillate between two positions
    positions = [
        {'player_map': 1, 'player_x': 100, 'player_y': 100},
        {'player_map': 1, 'player_x': 101, 'player_y': 100}
    ]
    
    # Move back and forth several times
    rewards = []
    for i in range(10):
        curr = positions[i % 2]
        prev = positions[(i + 1) % 2]
        reward, _ = component.calculate(curr, prev)
        rewards.append(reward)
    
    # Later rewards should be more negative
    assert rewards[-1] < rewards[0]

# Battle Component Tests

def test_battle_start_reward():
    """Should reward starting battles."""
    component = BattleRewardComponent()
    
    current = {
        'in_battle': 1,
        'player_hp': 20,
        'player_max_hp': 24
    }
    previous = {
        'in_battle': 0,
        'player_hp': 20,
        'player_max_hp': 24
    }
    
    reward, details = component.calculate(current, previous)
    assert reward > 0.0
    assert 'battle_start' in details

def test_battle_victory_reward():
    """Should reward winning battles."""
    component = BattleRewardComponent()
    
    current = {
        'in_battle': 0,
        'player_hp': 20,
        'player_max_hp': 24
    }
    previous = {
        'in_battle': 1,
        'player_hp': 20,
        'player_max_hp': 24
    }
    
    reward, details = component.calculate(current, previous)
    assert reward > 0.0
    assert 'battle_victory' in details

def test_battle_loss_penalty():
    """Should penalize losing battles."""
    component = BattleRewardComponent()
    
    current = {
        'in_battle': 0,
        'player_hp': 5,
        'player_max_hp': 24
    }
    previous = {
        'in_battle': 1,
        'player_hp': 5,
        'player_max_hp': 24
    }
    
    reward, details = component.calculate(current, previous)
    assert reward < 0.0
    assert 'battle_loss' in details

# Money Component Tests

def test_money_gain_reward():
    """Should reward earning money."""
    component = MoneyRewardComponent()
    setup_component_state(component, 'overworld')
    
    current = {'money': 1200}
    previous = {'money': 1000}
    
    reward, details = component.calculate(current, previous)
    assert reward > 0.0
    assert 'money_gain' in details

def test_money_spend_small_penalty():
    """Should apply small penalty for spending money."""
    component = MoneyRewardComponent()
    setup_component_state(component, 'overworld')
    
    current = {'money': 800}
    previous = {'money': 1000}
    
    reward, details = component.calculate(current, previous)
    assert reward < 0.0
    assert 'money_spent' in details

def test_money_reject_suspicious():
    """Should reject suspicious money changes."""
    component = MoneyRewardComponent()
    setup_component_state(component, 'overworld')
    
    current = {'money': 100000}
    previous = {'money': 1000}
    
    reward, details = component.calculate(current, previous)
    assert reward == 0.0
    assert not details

# Progression Component Tests

def test_first_pokemon_reward():
    """Should give big reward for first Pokemon."""
    component = ProgressionRewardComponent()
    setup_component_state(component, 'overworld')
    
    current = {
        'party_count': 1,
        'player_level': 5,
        'player_hp': 20,
        'player_max_hp': 24
    }
    previous = {
        'party_count': 0,
        'player_level': 0,
        'player_hp': 0,
        'player_max_hp': 0
    }
    
    reward, details = component.calculate(current, previous)
    assert reward > 0.0
    assert 'first_pokemon' in details

def test_additional_pokemon_reward():
    """Should reward catching additional Pokemon."""
    component = ProgressionRewardComponent()
    setup_component_state(component, 'overworld')
    
    current = {
        'party_count': 2,
        'player_level': 10,
        'player_hp': 30,
        'player_max_hp': 35
    }
    previous = {
        'party_count': 1,
        'player_level': 10,
        'player_hp': 30,
        'player_max_hp': 35
    }
    
    reward, details = component.calculate(current, previous)
    assert reward > 0.0
    assert 'new_pokemon' in details

def test_reject_invalid_progression():
    """Should reject invalid progression states."""
    component = ProgressionRewardComponent()
    setup_component_state(component, 'overworld')
    
    current = {
        'party_count': 1,
        'player_level': 0,  # Invalid level
        'player_hp': 20,
        'player_max_hp': 24
    }
    previous = {
        'party_count': 0,
        'player_level': 0,
        'player_hp': 0,
        'player_max_hp': 0
    }
    
    reward, details = component.calculate(current, previous)
    assert reward == 0.0
    assert not details
