"""Test suite for game states."""

import pytest
from core.state.machine import PyBoyGameState


def test_state_creation():
    """Test creating game states."""
    assert PyBoyGameState.OVERWORLD.value == "overworld"
    assert PyBoyGameState.BATTLE.value == "battle"
    assert PyBoyGameState.MENU.value == "menu"
    assert PyBoyGameState.DIALOGUE.value == "dialogue"


def test_from_str():
    """Test converting strings to game states."""
    assert PyBoyGameState.from_str("overworld") == PyBoyGameState.OVERWORLD
    assert PyBoyGameState.from_str("battle") == PyBoyGameState.BATTLE
    assert PyBoyGameState.from_str("menu") == PyBoyGameState.MENU
    assert PyBoyGameState.from_str("dialogue") == PyBoyGameState.DIALOGUE
    
    # Test invalid state
    assert PyBoyGameState.from_str("invalid_state") == PyBoyGameState.UNKNOWN


def test_allows_input():
    """Test allows_input property."""
    assert PyBoyGameState.OVERWORLD.allows_input is True
    assert PyBoyGameState.BATTLE.allows_input is True
    assert PyBoyGameState.MENU.allows_input is True
    assert PyBoyGameState.DIALOGUE.allows_input is True
    
    assert PyBoyGameState.LOADING.allows_input is False
    assert PyBoyGameState.INTRO_SEQUENCE.allows_input is False
    assert PyBoyGameState.CUTSCENE.allows_input is False
    assert PyBoyGameState.EVOLUTION.allows_input is False


def test_requires_action():
    """Test requires_action property."""
    assert PyBoyGameState.TITLE_SCREEN.requires_action is True
    assert PyBoyGameState.NEW_GAME.requires_action is True
    assert PyBoyGameState.CONTINUE.requires_action is True
    assert PyBoyGameState.DIALOGUE.requires_action is True
    assert PyBoyGameState.MENU.requires_action is True
    
    assert PyBoyGameState.OVERWORLD.requires_action is False
    assert PyBoyGameState.LOADING.requires_action is False
    assert PyBoyGameState.BATTLE.requires_action is False


def test_is_interactive():
    """Test is_interactive property."""
    assert PyBoyGameState.OVERWORLD.is_interactive is True
    assert PyBoyGameState.BATTLE.is_interactive is True
    assert PyBoyGameState.MENU.is_interactive is True
    assert PyBoyGameState.PC_BOX.is_interactive is True
    assert PyBoyGameState.PAUSE.is_interactive is True
    
    assert PyBoyGameState.LOADING.is_interactive is False
    assert PyBoyGameState.CUTSCENE.is_interactive is False
    assert PyBoyGameState.DIALOGUE.is_interactive is False


def test_is_transitional():
    """Test is_transitional property."""
    assert PyBoyGameState.LOADING.is_transitional is True
    assert PyBoyGameState.INTRO_SEQUENCE.is_transitional is True
    assert PyBoyGameState.CUTSCENE.is_transitional is True
    assert PyBoyGameState.EVOLUTION.is_transitional is True
    
    assert PyBoyGameState.OVERWORLD.is_transitional is False
    assert PyBoyGameState.BATTLE.is_transitional is False
    assert PyBoyGameState.MENU.is_transitional is False
