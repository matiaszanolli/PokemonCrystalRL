#!/usr/bin/env python3
"""
conftest.py - PyTest configuration and shared fixtures for Pokemon Crystal RL Agent tests

This module provides common fixtures and test utilities for all test modules.
"""

import pytest
import tempfile
import sqlite3
from pathlib import Path
from typing import Dict, List, Any
from unittest.mock import Mock, MagicMock

# Import the modules we're testing
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vision_processor import DetectedText, VisualContext
from semantic_context_system import SemanticContextSystem, GameContext
from dialogue_state_machine import DialogueStateMachine, DialogueState, NPCType
from choice_recognition_system import (
    ChoiceRecognitionSystem, 
    RecognizedChoice, 
    ChoiceContext,
    ChoiceType,
    ChoicePosition
)


# ============================================================================
# Database Fixtures
# ============================================================================

@pytest.fixture
def temp_db():
    """Create a temporary database for testing"""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    
    yield db_path
    
    # Cleanup
    Path(db_path).unlink(missing_ok=True)


@pytest.fixture
def temp_choice_db():
    """Create a temporary choice recognition database for testing"""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    
    # Initialize the database schema
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS choice_recognitions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                dialogue_text TEXT NOT NULL,
                recognized_choices TEXT NOT NULL,
                chosen_action TEXT,
                success BOOLEAN,
                confidence REAL,
                context_data TEXT
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS pattern_effectiveness (
                pattern_id TEXT PRIMARY KEY,
                usage_count INTEGER DEFAULT 0,
                success_count INTEGER DEFAULT 0,
                average_confidence REAL DEFAULT 0.5,
                last_used TEXT
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS action_mappings (
                choice_text TEXT,
                ui_layout TEXT,
                action_sequence TEXT,
                success_count INTEGER DEFAULT 0,
                failure_count INTEGER DEFAULT 0,
                last_used TEXT,
                PRIMARY KEY (choice_text, ui_layout, action_sequence)
            )
        """)
        conn.commit()
    
    yield db_path
    
    # Cleanup
    Path(db_path).unlink(missing_ok=True)


# ============================================================================
# Vision Processing Fixtures
# ============================================================================

@pytest.fixture
def sample_detected_texts():
    """Sample detected text objects for testing"""
    return [
        DetectedText("Hello! I'm Professor Elm!", 0.95, (10, 20, 200, 40), "dialogue"),
        DetectedText("Would you like a Pokemon?", 0.90, (10, 50, 180, 70), "dialogue"),
        DetectedText("Yes", 0.98, (20, 90, 50, 110), "choice"),
        DetectedText("No", 0.97, (20, 120, 40, 140), "choice"),
    ]


@pytest.fixture
def sample_visual_context(sample_detected_texts):
    """Sample visual context for testing"""
    return VisualContext(
        screen_type="dialogue",
        detected_text=sample_detected_texts,
        ui_elements=[],
        dominant_colors=[(255, 255, 255), (0, 0, 0)],
        game_phase="dialogue_interaction",
        visual_summary="Professor Elm asking about Pokemon"
    )


@pytest.fixture
def pokemon_selection_texts():
    """Sample Pokemon selection detected texts"""
    return [
        DetectedText("Choose your starter Pokemon!", 0.95, (10, 20, 220, 40), "dialogue"),
        DetectedText("Cyndaquil", 0.98, (20, 60, 90, 80), "choice"),
        DetectedText("Totodile", 0.97, (100, 60, 160, 80), "choice"),
        DetectedText("Chikorita", 0.96, (170, 60, 240, 80), "choice"),
    ]


@pytest.fixture
def battle_choice_texts():
    """Sample battle choice detected texts"""
    return [
        DetectedText("What will Cyndaquil do?", 0.95, (10, 20, 180, 40), "dialogue"),
        DetectedText("Fight", 0.98, (20, 60, 60, 80), "choice"),
        DetectedText("Bag", 0.97, (80, 60, 110, 80), "choice"),
        DetectedText("Pokemon", 0.96, (20, 90, 80, 110), "choice"),
        DetectedText("Run", 0.95, (90, 90, 120, 110), "choice"),
    ]


# ============================================================================
# Semantic Context Fixtures
# ============================================================================

@pytest.fixture
def sample_game_context():
    """Sample game context for testing"""
    return GameContext(
        current_objective="get_starter_pokemon",
        player_progress={"badges": 0, "pokemon_count": 1, "story_flags": ["met_elm"]},
        location_info={"current_map": "elm_lab", "region": "johto"},
        recent_events=["entered_elm_lab", "spoke_to_elm"],
        active_quests=["get_starter_pokemon"]
    )


@pytest.fixture
def advanced_game_context():
    """Advanced game context for testing"""
    return GameContext(
        current_objective="beat_bugsy",
        player_progress={"badges": 2, "pokemon_count": 6, "story_flags": ["met_elm", "got_starter", "beat_falkner"]},
        location_info={"current_map": "azalea_gym", "region": "johto"},
        recent_events=["entered_gym", "challenged_bugsy"],
        active_quests=["beat_bugsy", "find_team_rocket"]
    )


@pytest.fixture
def semantic_system(temp_db):
    """Create a semantic context system for testing"""
    return SemanticContextSystem(db_path=temp_db)


# ============================================================================
# Dialogue State Machine Fixtures
# ============================================================================

@pytest.fixture
def dialogue_machine(temp_db):
    """Create a dialogue state machine for testing"""
    return DialogueStateMachine(db_path=temp_db)


@pytest.fixture
def sample_choice_context():
    """Sample choice context for testing"""
    return ChoiceContext(
        dialogue_text="Would you like to continue?",
        screen_type="dialogue",
        npc_type="professor",
        current_objective="get_starter_pokemon",
        conversation_history=["Hello!", "I study Pokemon."],
        ui_layout="standard_dialogue"
    )


# ============================================================================
# Choice Recognition Fixtures
# ============================================================================

@pytest.fixture
def choice_system(temp_choice_db):
    """Create a choice recognition system for testing"""
    return ChoiceRecognitionSystem(db_path=temp_choice_db)


@pytest.fixture
def sample_recognized_choices():
    """Sample recognized choices for testing"""
    return [
        RecognizedChoice(
            text="Yes",
            choice_type=ChoiceType.YES_NO,
            position=ChoicePosition.TOP,
            action_mapping=["A"],
            confidence=0.95,
            priority=80,
            expected_outcome="accept_or_confirm",
            context_tags=["positive_response", "type_yes_no"],
            ui_coordinates=(20, 90, 50, 110)
        ),
        RecognizedChoice(
            text="No",
            choice_type=ChoiceType.YES_NO,
            position=ChoicePosition.BOTTOM,
            action_mapping=["B"],
            confidence=0.90,
            priority=40,
            expected_outcome="decline_or_cancel",
            context_tags=["negative_response", "type_yes_no"],
            ui_coordinates=(20, 120, 40, 140)
        )
    ]


@pytest.fixture
def pokemon_choices():
    """Sample Pokemon selection choices"""
    return [
        RecognizedChoice(
            text="Cyndaquil",
            choice_type=ChoiceType.POKEMON_SELECTION,
            position=ChoicePosition.LEFT,
            action_mapping=["A"],
            confidence=0.98,
            priority=90,
            expected_outcome="select_fire_starter",
            context_tags=["pokemon_choice", "fire_type"],
            ui_coordinates=(20, 60, 90, 80)
        ),
        RecognizedChoice(
            text="Totodile",
            choice_type=ChoiceType.POKEMON_SELECTION,
            position=ChoicePosition.CENTER,
            action_mapping=["DOWN", "A"],
            confidence=0.97,
            priority=70,
            expected_outcome="select_water_starter",
            context_tags=["pokemon_choice", "water_type"],
            ui_coordinates=(100, 60, 160, 80)
        ),
        RecognizedChoice(
            text="Chikorita",
            choice_type=ChoiceType.POKEMON_SELECTION,
            position=ChoicePosition.RIGHT,
            action_mapping=["DOWN", "DOWN", "A"],
            confidence=0.96,
            priority=50,
            expected_outcome="select_grass_starter",
            context_tags=["pokemon_choice", "grass_type"],
            ui_coordinates=(170, 60, 240, 80)
        )
    ]


# ============================================================================
# Mock Fixtures
# ============================================================================

@pytest.fixture
def mock_rom_processor():
    """Mock ROM processor for testing"""
    mock = Mock()
    mock.decode_text.return_value = "Sample decoded text"
    mock.is_available = True
    return mock


@pytest.fixture
def mock_vision_processor():
    """Mock vision processor for testing"""
    mock = Mock()
    mock.process_frame.return_value = Mock(
        detected_text=[
            DetectedText("Sample text", 0.9, (10, 10, 100, 30), "dialogue")
        ],
        ui_elements=[],
        screen_type="dialogue"
    )
    return mock


# ============================================================================
# Test Data Fixtures
# ============================================================================

@pytest.fixture
def dialogue_scenarios():
    """Common dialogue test scenarios"""
    return {
        "professor_elm_intro": {
            "npc_type": NPCType.PROFESSOR,
            "dialogue_text": "Hello! I'm Professor Elm! I study Pokemon behavior.",
            "expected_intent": "introduction",
            "context": "elm_lab"
        },
        "nurse_healing": {
            "npc_type": NPCType.GENERIC,
            "dialogue_text": "Welcome to the Pokemon Center! Would you like me to heal your Pokemon?",
            "expected_intent": "healing_request",
            "context": "pokemon_center"
        },
        "gym_leader_challenge": {
            "npc_type": NPCType.GYM_LEADER,
            "dialogue_text": "I'm Falkner, the Violet Gym Leader! Are you ready to battle?",
            "expected_intent": "gym_challenge",
            "context": "violet_gym"
        },
        "shop_keeper": {
            "npc_type": NPCType.SHOPKEEPER,
            "dialogue_text": "Welcome to the Poke Mart! What can I get for you today?",
            "expected_intent": "shop_interaction",
            "context": "poke_mart"
        }
    }


@pytest.fixture
def choice_scenarios():
    """Common choice recognition test scenarios"""
    return {
        "yes_no_simple": {
            "dialogue": "Would you like to continue?",
            "choices": ["Yes", "No"],
            "expected_types": [ChoiceType.YES_NO, ChoiceType.YES_NO],
            "expected_actions": [["A"], ["B"]]
        },
        "pokemon_starter": {
            "dialogue": "Choose your starter Pokemon!",
            "choices": ["Cyndaquil", "Totodile", "Chikorita"],
            "expected_types": [ChoiceType.POKEMON_SELECTION] * 3,
            "expected_actions": [["A"], ["DOWN", "A"], ["DOWN", "DOWN", "A"]]
        },
        "battle_menu": {
            "dialogue": "What will your Pokemon do?",
            "choices": ["Fight", "Bag", "Pokemon", "Run"],
            "expected_types": [ChoiceType.MENU_SELECTION] * 4,
            "expected_actions": [["A"], ["DOWN", "A"], ["DOWN", "DOWN", "A"], ["DOWN", "DOWN", "DOWN", "A"]]
        }
    }


# ============================================================================
# Utility Functions
# ============================================================================

def create_test_visual_context(detected_texts: List[DetectedText], screen_type: str = "dialogue") -> VisualContext:
    """Create a visual context for testing"""
    return VisualContext(
        screen_type=screen_type,
        detected_text=detected_texts,
        ui_elements=[],
        dominant_colors=[(255, 255, 255)],
        game_phase="dialogue_interaction",
        visual_summary="Test dialogue scenario"
    )


def assert_choice_properties(choice: RecognizedChoice, expected_text: str, expected_type: ChoiceType):
    """Assert basic properties of a recognized choice"""
    assert choice.text == expected_text
    assert choice.choice_type == expected_type
    assert 0.0 <= choice.confidence <= 1.0
    assert choice.priority >= 0
    assert len(choice.action_mapping) > 0
    assert choice.expected_outcome is not None
    assert len(choice.context_tags) > 0


# ============================================================================
# Parametrize Fixtures
# ============================================================================

@pytest.fixture(params=[
    {"badges": 0, "pokemon_count": 1},
    {"badges": 3, "pokemon_count": 6},
    {"badges": 8, "pokemon_count": 12}
])
def game_progress_scenarios(request):
    """Different game progress scenarios for testing"""
    return request.param


@pytest.fixture(params=[
    NPCType.PROFESSOR,
    NPCType.FAMILY,
    NPCType.GYM_LEADER,
    NPCType.SHOPKEEPER,
    NPCType.TRAINER,
    NPCType.GENERIC
])
def npc_types(request):
    """Different NPC types for testing"""
    return request.param
