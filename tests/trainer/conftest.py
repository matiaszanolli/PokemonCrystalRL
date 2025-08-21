"""Common test fixtures and mocks."""

import pytest
from unittest.mock import Mock, patch
import tempfile
from pathlib import Path
from unittest.mock import Mock
from core.choice_recognition_system import (
    ChoiceRecognitionSystem, 
    RecognizedChoice, 
    ChoiceContext,
    ChoiceType,
    ChoicePosition
)
from vision.vision_processor import DetectedText, VisualContext
from trainer.trainer import TrainingConfig, TrainingMode, LLMBackend, PokemonTrainer

@pytest.fixture
def base_config():
    """Create a base training config for tests."""
    return TrainingConfig(
        rom_path="test.gbc",
        mode=TrainingMode.FAST_MONITORED,
        llm_backend=LLMBackend.SMOLLM2,
        headless=True,
        debug_mode=True
    )

@pytest.fixture
def trainer_fast_monitored(base_config):
    """Create a trainer fixture with fast monitored mode."""
    with patch('trainer.trainer.PyBoy') as mock_pyboy_class:
        mock_pyboy_instance = Mock()
        mock_pyboy_class.return_value = mock_pyboy_instance
        trainer = PokemonTrainer(base_config)
        return trainer

@pytest.fixture
def trainer_ultra_fast(base_config):
    """Create a trainer fixture with ultra fast mode."""
    config = base_config
    config.mode = TrainingMode.ULTRA_FAST
    config.max_actions = 10
    config.capture_screens = False

    with patch('trainer.trainer.PyBoy') as mock_pyboy_class:
        mock_pyboy_instance = Mock()
        mock_pyboy_class.return_value = mock_pyboy_instance
        trainer = PokemonTrainer(config)
        return trainer

@pytest.fixture(autouse=True)
def mock_llm_manager(monkeypatch):
    """Mock LocalLLMPokemonAgent to avoid import errors."""
    mock_agent = Mock()
    mock_agent.return_value = Mock()
    monkeypatch.setattr('trainer.llm_manager.LLMManager', mock_agent)

@pytest.fixture
def choice_system():
    """Create a choice recognition system for testing."""
    # Use a temporary database for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "test_choice_patterns.db"
        system = ChoiceRecognitionSystem(str(db_path))
        yield system

@pytest.fixture
def sample_visual_context():
    """Sample visual context for dialogue testing."""
    text1 = DetectedText("Hello! How can I help you?", 0.95, (10, 10, 200, 30), "dialogue")
    text2 = DetectedText("Yes", 0.98, (20, 60, 50, 80), "choice")
    text3 = DetectedText("No", 0.98, (60, 60, 90, 80), "choice")
    
    return VisualContext(
        screen_type="dialogue",
        detected_text=[text1, text2, text3],
        ui_elements=[],
        dominant_colors=[(255, 255, 255)],
        game_phase="dialogue_interaction",
        visual_summary="Test dialogue with choices"
    )

@pytest.fixture
def sample_choice_context():
    """Sample choice context for testing."""
    return ChoiceContext(
        dialogue_text="Hello! How can I help you?",
        screen_type="dialogue",
        npc_type="generic",
        current_objective=None,
        conversation_history=[],
        ui_layout="standard_dialogue"
    )

@pytest.fixture
def sample_recognized_choices():
    """Sample recognized choices for testing."""
    return [
        RecognizedChoice(
            text="Yes",
            choice_type=ChoiceType.YES_NO,
            position=ChoicePosition.TOP,
            action_mapping=["A"],
            confidence=0.9,
            priority=80,
            expected_outcome="accept_or_confirm",
            context_tags=["type_yes_no", "positive_response"],
            ui_coordinates=(20, 60, 50, 80)
        ),
        RecognizedChoice(
            text="No",
            choice_type=ChoiceType.YES_NO,
            position=ChoicePosition.BOTTOM,
            action_mapping=["B"],
            confidence=0.85,
            priority=60,
            expected_outcome="decline_or_cancel",
            context_tags=["type_yes_no", "negative_response"],
            ui_coordinates=(60, 60, 90, 80)
        )
    ]

@pytest.fixture
def pokemon_selection_texts():
    """Sample texts for Pokemon selection."""
    return [
        DetectedText("Choose your starter Pokemon!", 0.95, (10, 10, 200, 30), "dialogue"),
        DetectedText("Cyndaquil", 0.98, (20, 60, 90, 80), "choice"),
        DetectedText("Totodile", 0.97, (100, 60, 160, 80), "choice"),
        DetectedText("Chikorita", 0.96, (180, 60, 240, 80), "choice")
    ]

@pytest.fixture
def choice_scenarios():
    """Sample choice scenarios for testing."""
    return {
        "yes_no_simple": {
            "dialogue": "Do you want to continue?",
            "choices": ["Yes", "No"],
            "expected_types": [ChoiceType.YES_NO]
        },
        "pokemon_starter": {
            "dialogue": "Choose your starter Pokemon!",
            "choices": ["Cyndaquil", "Totodile", "Chikorita"],
            "expected_types": [ChoiceType.POKEMON_SELECTION]
        },
        "battle_menu": {
            "dialogue": "What will you do?",
            "choices": ["Fight", "Use Item", "Switch", "Run"],
            "expected_types": [ChoiceType.MENU_SELECTION]
        }
    }
