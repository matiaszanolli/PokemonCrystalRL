"""
Integration Tests for PokemonRewardCalculator

This module tests the complete reward calculator, ensuring components work
together correctly and the calculator properly implements the interface.
"""

from typing import Dict

import os
import sys
import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from rewards.calculator import PokemonRewardCalculator
from rewards.interface import RewardCalculatorInterface

@pytest.fixture
def calculator():
    """Create a fresh reward calculator instance."""
    return PokemonRewardCalculator()

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

# Interface Tests

def test_implements_interface(calculator):
    """Calculator should implement RewardCalculatorInterface."""
    assert isinstance(calculator, RewardCalculatorInterface)

def test_screen_state_propagation(calculator):
    """Screen state changes should propagate to all components."""
    calculator.last_screen_state = 'battle'
    calculator.prev_screen_state = 'overworld'
    
    # Check all components received the state
    for component in calculator.components:
        assert component.last_screen_state == 'battle'
        assert component.prev_screen_state == 'overworld'

def test_action_propagation(calculator):
    """Action changes should propagate to all components."""
    calculator.last_action = 'up'
    
    # Check all components received the action
    for component in calculator.components:
        assert component.last_action == 'up'

# Reward Calculation Tests

def test_empty_state_safety(calculator):
    """Should safely handle empty states."""
    reward, details = calculator.calculate_reward({}, {})
    assert reward <= 0  # Should only get time penalty
    assert 'time' in details

def test_typical_gameplay_rewards(calculator, base_state):
    """Should give appropriate rewards for typical gameplay."""
    calculator.last_screen_state = 'overworld'
    calculator.prev_screen_state = 'overworld'
    
    # Move forward and gain some health
    current = base_state.copy()
    current.update({
        'player_x': 101,  # Moved
        'player_hp': 22   # Healed
    })
    previous = base_state.copy()
    
    reward, details = calculator.calculate_reward(current, previous)
    
    # Should get movement and healing rewards
    assert reward > 0
    assert any(k in details for k in ['movement', 'healing'])

def test_major_milestone_rewards(calculator, base_state):
    """Should give big rewards for major milestones."""
    calculator.last_screen_state = 'overworld'
    calculator.prev_screen_state = 'overworld'
    
    # Get first badge and level up
    current = base_state.copy()
    current.update({
        'badges_total': 1,
        'badges': 1,
        'player_level': 6
    })
    previous = base_state.copy()
    
    reward, details = calculator.calculate_reward(current, previous)
    
    # Should get substantial rewards
    assert reward > 100  # Major milestone rewards are large
    assert any(k in details for k in ['badge_earned', 'level_up'])

def test_early_game_progression(calculator):
    """Should properly reward early game progression."""
    calculator.last_screen_state = 'overworld'
    calculator.prev_screen_state = 'overworld'
    
    # Get first Pokemon
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
    
    reward, details = calculator.calculate_reward(current, previous)
    
    # Should get first Pokemon reward
    assert reward > 50  # First Pokemon is a major milestone
    assert 'first_pokemon' in details

def test_battle_sequence_rewards(calculator, base_state):
    """Should properly reward battle sequences."""
    # Enter battle
    current = base_state.copy()
    current['in_battle'] = 1
    previous = base_state.copy()
    
    calculator.last_screen_state = 'battle'
    calculator.prev_screen_state = 'overworld'
    
    reward1, details1 = calculator.calculate_reward(current, previous)
    assert reward1 > 0
    assert 'battle_start' in details1
    
    # Win battle
    current = base_state.copy()
    current['in_battle'] = 0
    previous = base_state.copy()
    previous['in_battle'] = 1
    
    calculator.last_screen_state = 'overworld'
    calculator.prev_screen_state = 'battle'
    
    reward2, details2 = calculator.calculate_reward(current, previous)
    assert reward2 > 0
    assert 'battle_victory' in details2

def test_anti_farming_protection(calculator, base_state):
    """Should penalize reward farming behavior."""
    calculator.last_screen_state = 'overworld'
    calculator.prev_screen_state = 'overworld'
    
    # Oscillate between two positions
    pos1 = base_state.copy()
    pos2 = base_state.copy()
    pos2['player_x'] = 101
    
    rewards = []
    # Move back and forth several times
    for i in range(10):
        curr = pos1 if i % 2 == 0 else pos2
        prev = pos2 if i % 2 == 0 else pos1
        reward, _ = calculator.calculate_reward(curr, prev)
        rewards.append(reward)
    
    # Later rewards should be more negative (farming penalty)
    assert rewards[-1] < rewards[0]

def test_reward_summary_formatting(calculator):
    """Should format reward summaries correctly."""
    rewards = {
        'movement': 0.01,
        'healing': 0.5,
        'tiny': 0.001,  # Should be excluded (too small)
        'battle_victory': 20.0
    }
    
    summary = calculator.get_reward_summary(rewards)
    
    # Should format significant rewards
    assert 'movement: +0.01' in summary
    assert 'healing: +0.50' in summary
    assert 'battle_victory: +20.00' in summary
    # Should exclude insignificant rewards
    assert 'tiny' not in summary
