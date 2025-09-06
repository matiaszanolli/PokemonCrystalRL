"""
Tests for Pokemon Crystal action parsing utilities.
"""

import pytest
from typing import Dict, List

from utils.action_parser import (
    parse_action_response,
    get_allowed_action,
    is_action_allowed,
    get_exploration_fallback,
    get_context_specific_action,
    get_exploration_pattern_action,
    is_action_safe
)

from tests.fixtures.pyboy_helpers import create_game_state
from config.constants import SCREEN_STATES, AVAILABLE_ACTIONS

def test_parse_action_response():
    """Test LLM response parsing."""
    # Test ACTION: pattern
    response = "I think we should... ACTION: up to move forward"
    assert parse_action_response(response) == "up"
    
    # Test direct action word
    assert parse_action_response("move down") == "down"
    
    # Test synonyms
    assert parse_action_response("go north") == "up"
    assert parse_action_response("interact with NPC") == "a"
    assert parse_action_response("escape from battle") == "b"
    
    # Test invalid/missing action
    assert parse_action_response("do nothing") == "a"

def test_is_action_allowed():
    """Test action permission checking."""
    # Test with no Pokemon
    state = create_game_state(party_count=0)
    assert is_action_allowed('up', state) is True
    assert is_action_allowed('a', state) is True
    assert is_action_allowed('start', state) is False
    assert is_action_allowed('select', state) is False
    
    # Test with Pokemon
    state = create_game_state(party_count=1)
    assert is_action_allowed('start', state) is True
    assert is_action_allowed('select', state) is True

def test_get_allowed_action():
    """Test getting allowed alternative actions."""
    state = create_game_state(party_count=0)
    
    # Test forbidden action replacement
    alt = get_allowed_action('start', state, SCREEN_STATES['OVERWORLD'])
    assert alt in AVAILABLE_ACTIONS['MOVEMENT'] + ['a']
    
    # Test battle state
    state['in_battle'] = 1
    assert get_allowed_action('select', state, SCREEN_STATES['BATTLE']) == 'a'
    
    # Test dialogue state
    assert get_allowed_action('start', state, SCREEN_STATES['DIALOGUE']) == 'a'
    
    # Test menu state
    assert get_allowed_action('select', state, SCREEN_STATES['MENU']) == 'b'

def test_get_exploration_fallback():
    """Test exploration fallback actions."""
    # Test with no recent actions
    action = get_exploration_fallback()
    assert action in AVAILABLE_ACTIONS['MOVEMENT'] + ['a']
    
    # Test avoiding recent actions
    recent = ['up', 'up', 'a']
    action = get_exploration_fallback(recent)
    assert action not in recent
    
    # Test with all actions recently used
    recent = ['up', 'down', 'left', 'right', 'a']
    action = get_exploration_fallback(recent)
    assert action in AVAILABLE_ACTIONS['MOVEMENT'] + ['a']

def test_get_context_specific_action():
    """Test context-aware action selection."""
    state = create_game_state()
    recent = ['up', 'down']
    
    # Test battle state
    state['in_battle'] = 1
    action, _ = get_context_specific_action(SCREEN_STATES['BATTLE'], state, recent)
    assert action == 'a'
    
    # Test dialogue state
    state['in_battle'] = 0
    action, _ = get_context_specific_action(SCREEN_STATES['DIALOGUE'], state, recent)
    assert action == 'a'
    
    # Test settings menu state
    action, _ = get_context_specific_action(SCREEN_STATES['SETTINGS_MENU'], state, recent)
    assert action == 'b'
    
    # Test menu after START
    action, _ = get_context_specific_action(SCREEN_STATES['MENU'], state, ['START'])
    assert action == 'b'
    
    # Test overworld exploration
    action, _ = get_context_specific_action(SCREEN_STATES['OVERWORLD'], state, recent)
    assert action in AVAILABLE_ACTIONS['MOVEMENT'] + ['a']

def test_get_exploration_pattern_action():
    """Test exploration pattern action selection."""
    # Test pattern start
    assert get_exploration_pattern_action([]) in ['up', 'a']
    
    # Test pattern continuation
    assert get_exploration_pattern_action(['up']) == 'up'
    assert get_exploration_pattern_action(['up', 'up']) == 'a'
    
    # Test pattern wrap-around
    pattern = ['up', 'up', 'a', 'right', 'right', 'a', 'down', 'down', 'a', 'left', 'left', 'a']
    for i in range(len(pattern) * 2):
        recent = pattern[max(0, i-3):i]
        action = get_exploration_pattern_action(recent)
        expected = pattern[i % len(pattern)]
        assert action == expected

def test_is_action_safe():
    """Test action safety checking."""
    state = create_game_state()
    
    # Test forbidden actions
    assert is_action_safe('start', state, SCREEN_STATES['OVERWORLD']) is False
    
    # Test stuck detection
    stuck_count = {'up': 5}
    assert is_action_safe('up', state, SCREEN_STATES['OVERWORLD'], stuck_count) is False
    
    # Test settings menu
    assert is_action_safe('b', state, SCREEN_STATES['SETTINGS_MENU']) is True
    assert is_action_safe('a', state, SCREEN_STATES['SETTINGS_MENU']) is False
    
    # Test menu START prevention
    assert is_action_safe('START', state, SCREEN_STATES['MENU']) is False
    
    # Test normal actions
    assert is_action_safe('a', state, SCREEN_STATES['OVERWORLD']) is True
    assert is_action_safe('up', state, SCREEN_STATES['OVERWORLD']) is True
