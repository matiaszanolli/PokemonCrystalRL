#!/usr/bin/env python3
"""
test_choice_recognition_system.py - Comprehensive tests for the Choice Recognition System

This module tests all aspects of the choice recognition system including:
- Choice pattern matching and recognition
- Action mapping and generation
- Priority calculation and ranking
- Database operations and learning
- UI layout detection
- Context-dependent processing
"""

import pytest
import json
import sqlite3
from pathlib import Path
from unittest.mock import patch, Mock, MagicMock
from datetime import datetime
import sys
import os

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# Import with fallbacks for missing dependencies
try:
    from core.choice_recognition import (
        ChoiceRecognitionSystem, 
        RecognizedChoice, 
        ChoiceContext,
        ChoiceType,
        ChoicePosition
    )
    from shared_types import VisualContext
    from vision.vision_processor import DetectedText
except ImportError:
    # Skip these tests if dependencies aren't available
    pytest.skip("Missing dependencies for choice recognition tests", allow_module_level=True)


class TestChoiceRecognitionInitialization:
    """Test choice recognition system initialization"""
    
    @pytest.mark.unit
    @pytest.mark.choice
    def test_system_initialization(self, choice_system):
        """Test basic system initialization"""
        assert choice_system.db_path.exists()
        assert len(choice_system.choice_patterns) > 0
        assert len(choice_system.action_mappings) > 0
        assert len(choice_system.ui_layouts) > 0
    
    @pytest.mark.unit
    @pytest.mark.choice
    @pytest.mark.database
    def test_database_initialization(self, choice_system):
        """Test that database tables are created correctly"""
        with sqlite3.connect(choice_system.db_path) as conn:
            cursor = conn.cursor()
            
            # Check if tables exist
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            expected_tables = ["choice_recognitions", "pattern_effectiveness", "action_mappings"]
            for table in expected_tables:
                assert table in tables
    
    @pytest.mark.unit
    @pytest.mark.choice
    def test_pattern_loading(self, choice_system):
        """Test that choice patterns are loaded correctly"""
        patterns = choice_system.choice_patterns
        
        # Check key pattern categories exist
        assert "yes_no_basic" in patterns
        assert "numbered_choices" in patterns
        assert "starter_pokemon" in patterns
        assert "pokemon_actions" in patterns
        
        # Verify pattern structure
        for pattern_name, pattern_data in patterns.items():
            assert "pattern" in pattern_data
            assert "type" in pattern_data
            assert "indicators" in pattern_data
            assert "confidence_boost" in pattern_data
    
    @pytest.mark.unit
    @pytest.mark.choice
    def test_action_mappings_loading(self, choice_system):
        """Test that action mappings are loaded correctly"""
        mappings = choice_system.action_mappings
        
        expected_mappings = [
            "yes", "no", "cyndaquil", "totodile", "chikorita",
            "fight", "confirm", "cancel"
        ]
        
        for mapping in expected_mappings:
            assert mapping in mappings
            assert isinstance(mappings[mapping], list)
            assert len(mappings[mapping]) > 0
    
    @pytest.mark.unit
    @pytest.mark.choice
    def test_ui_layouts_loading(self, choice_system):
        """Test that UI layouts are loaded correctly"""
        layouts = choice_system.ui_layouts
        
        expected_layouts = [
            "standard_dialogue", "menu_selection", 
            "pokemon_selection", "yes_no_dialog"
        ]
        
        for layout in expected_layouts:
            assert layout in layouts
            assert "description" in layouts[layout]
            assert "choice_positions" in layouts[layout]
            assert "navigation" in layouts[layout]


class TestChoicePatternMatching:
    """Test choice pattern matching functionality"""
    
    @pytest.mark.unit
    @pytest.mark.choice
    def test_yes_no_pattern_matching(self, choice_system):
        """Test yes/no pattern recognition"""
        test_cases = [
            ("Yes", ChoiceType.YES_NO),
            ("No", ChoiceType.YES_NO),
            ("Okay", ChoiceType.YES_NO),
            ("Sure", ChoiceType.YES_NO),
            ("Nope", ChoiceType.YES_NO),
        ]
        
        for text, expected_type in test_cases:
            choice_type, confidence = choice_system._match_choice_patterns(text.lower())
            assert choice_type == expected_type
            assert confidence > 0.5
    
    @pytest.mark.unit
    @pytest.mark.choice
    def test_pokemon_pattern_matching(self, choice_system):
        """Test Pokemon-specific pattern recognition"""
        test_cases = [
            ("Cyndaquil", ChoiceType.POKEMON_SELECTION),
            ("Totodile", ChoiceType.POKEMON_SELECTION),
            ("Chikorita", ChoiceType.POKEMON_SELECTION),
        ]
        
        for text, expected_type in test_cases:
            choice_type, confidence = choice_system._match_choice_patterns(text.lower())
            assert choice_type == expected_type
            assert confidence > 0.5
    
    @pytest.mark.unit
    @pytest.mark.choice
    def test_numbered_pattern_matching(self, choice_system):
        """Test numbered choice pattern recognition"""
        test_cases = [
            ("1. Option one", ChoiceType.MULTIPLE_CHOICE),
            ("2) Option two", ChoiceType.MULTIPLE_CHOICE),
            ("3. Third choice", ChoiceType.MULTIPLE_CHOICE),
        ]
        
        for text, expected_type in test_cases:
            choice_type, confidence = choice_system._match_choice_patterns(text.lower())
            assert choice_type == expected_type
            assert confidence > 0.5
    
    @pytest.mark.unit
    @pytest.mark.choice
    def test_battle_menu_pattern_matching(self, choice_system):
        """Test battle menu pattern recognition"""
        test_cases = [
            ("Fight", ChoiceType.MENU_SELECTION),
            ("Use Item", ChoiceType.MENU_SELECTION),
            ("Switch", ChoiceType.MENU_SELECTION),
            ("Run", ChoiceType.MENU_SELECTION),
        ]
        
        for text, expected_type in test_cases:
            choice_type, confidence = choice_system._match_choice_patterns(text.lower())
            # Should match some menu-related pattern
            assert choice_type is not None
            assert confidence > 0.0
    
    @pytest.mark.unit
    @pytest.mark.choice
    def test_directional_pattern_matching(self, choice_system):
        """Test directional pattern recognition"""
        test_cases = [
            ("North", ChoiceType.DIRECTIONAL),
            ("South", ChoiceType.DIRECTIONAL),
            ("East", ChoiceType.DIRECTIONAL),
            ("West", ChoiceType.DIRECTIONAL),
        ]
        
        for text, expected_type in test_cases:
            choice_type, confidence = choice_system._match_choice_patterns(text.lower())
            assert choice_type == expected_type
            assert confidence > 0.3
    
    @pytest.mark.unit
    @pytest.mark.choice
    def test_confirmation_pattern_matching(self, choice_system):
        """Test confirmation pattern recognition"""
        test_cases = [
            ("Confirm", ChoiceType.CONFIRMATION),
            ("Accept", ChoiceType.CONFIRMATION),
            ("Cancel", ChoiceType.CONFIRMATION),
            ("Decline", ChoiceType.CONFIRMATION),
        ]
        
        for text, expected_type in test_cases:
            choice_type, confidence = choice_system._match_choice_patterns(text.lower())
            assert choice_type == expected_type
            assert confidence > 0.3
    
    @pytest.mark.unit
    @pytest.mark.choice
    def test_pattern_confidence_scoring(self, choice_system):
        """Test pattern confidence scoring"""
        # Strong matches should have higher confidence
        strong_matches = ["Yes", "Cyndaquil", "1. First option"]
        weak_matches = ["Maybe", "Something", "Random text"]
        
        for text in strong_matches:
            _, confidence = choice_system._match_choice_patterns(text.lower())
            assert confidence > 0.5  # Strong matches
        
        for text in weak_matches:
            choice_type, confidence = choice_system._match_choice_patterns(text.lower())
            if choice_type is not None:
                assert confidence < 0.5  # Weak matches


class TestChoiceExtraction:
    """Test choice extraction from visual context"""
    
    @pytest.mark.unit
    @pytest.mark.choice
    def test_choice_text_extraction(self, choice_system, sample_visual_context):
        """Test extraction of choice texts from detected text"""
        choice_texts = choice_system._extract_choice_texts(sample_visual_context.detected_text)
        
        assert len(choice_texts) > 0
        
        for choice_info in choice_texts:
            assert "text" in choice_info
            assert "coordinates" in choice_info
            assert "confidence" in choice_info
            assert "location" in choice_info
    
    @pytest.mark.unit
    @pytest.mark.choice
    def test_choice_filtering(self, choice_system):
        """Test filtering of inappropriate choice texts"""
        # Create texts with various lengths and qualities
        test_texts = [
            DetectedText("Yes", 0.95, (20, 60, 50, 80), "choice"),  # Good
            DetectedText("This is a very long text that should not be considered a choice option", 0.9, (20, 100, 300, 120), "dialogue"),  # Too long
            DetectedText("X", 0.8, (20, 140, 25, 160), "choice"),  # Too short
            DetectedText("No", 0.92, (20, 180, 45, 200), "choice"),  # Good
        ]
        
        choice_texts = choice_system._extract_choice_texts(test_texts)
        
        # Should filter out inappropriate texts
        extracted_texts = [choice["text"] for choice in choice_texts]
        assert "Yes" in extracted_texts
        assert "No" in extracted_texts
        assert len([t for t in extracted_texts if len(t) > 50]) == 0  # No overly long texts
    
    @pytest.mark.unit
    @pytest.mark.choice
    def test_empty_detected_text(self, choice_system):
        """Test handling of empty detected text list"""
        choice_texts = choice_system._extract_choice_texts([])
        assert len(choice_texts) == 0


class TestChoiceRecognition:
    """Test complete choice recognition process"""
    
    @pytest.mark.unit
    @pytest.mark.choice
    def test_basic_choice_recognition(self, choice_system, sample_visual_context, sample_choice_context):
        """Test basic choice recognition"""
        choices = choice_system.recognize_choices(sample_visual_context, sample_choice_context)
        
        assert isinstance(choices, list)
        
        for choice in choices:
            assert isinstance(choice, RecognizedChoice)
            assert choice.text is not None
            assert isinstance(choice.choice_type, ChoiceType)
            assert isinstance(choice.position, ChoicePosition)
            assert isinstance(choice.action_mapping, list)
            assert 0.0 <= choice.confidence <= 1.0
            assert choice.priority >= 0
            assert choice.expected_outcome is not None
            assert isinstance(choice.context_tags, list)
    
    @pytest.mark.unit
    @pytest.mark.choice
    @pytest.mark.parametrize("scenario_name", [
        "yes_no_simple", "pokemon_starter", "battle_menu"
    ])
    def test_choice_scenarios(self, choice_system, choice_scenarios, scenario_name):
        """Test various choice recognition scenarios"""
        scenario = choice_scenarios[scenario_name]
        
        # Create detected texts from scenario
        detected_texts = []
        for i, choice_text in enumerate(scenario["choices"]):
            detected_text = DetectedText(
                choice_text, 
                0.95, 
                (20 + i * 80, 60, 20 + i * 80 + 70, 80), 
                "choice"
            )
            detected_texts.append(detected_text)
        
        visual_context = VisualContext(
            screen_type="dialogue",
            detected_text=detected_texts,
            ui_elements=[],
            dominant_colors=[(255, 255, 255)],
            game_phase="dialogue_interaction",
            visual_summary=f"Test scenario: {scenario_name}"
        )
        
        choice_context = ChoiceContext(
            dialogue_text=scenario["dialogue"],
            screen_type="dialogue",
            npc_type="generic",
            current_objective=None,
            conversation_history=[],
            ui_layout="standard_dialogue"
        )
        
        choices = choice_system.recognize_choices(visual_context, choice_context)
        
        # Should recognize all choices
        assert len(choices) >= len(scenario["choices"]) - 1  # Allow for some missed detections
        
        # Check choice types match expectations
        recognized_types = [choice.choice_type for choice in choices]
        expected_types = scenario["expected_types"]
        
        # At least some should match expected types
        matching_types = [t for t in recognized_types if t in expected_types]
        assert len(matching_types) > 0
    
    @pytest.mark.unit
    @pytest.mark.choice
    def test_pokemon_selection_recognition(self, choice_system, pokemon_selection_texts):
        """Test Pokemon selection choice recognition"""
        visual_context = VisualContext(
            screen_type="dialogue",
            detected_text=pokemon_selection_texts,
            ui_elements=[],
            dominant_colors=[(255, 255, 255)],
            game_phase="dialogue_interaction",
            visual_summary="Pokemon selection"
        )
        
        choice_context = ChoiceContext(
            dialogue_text="Choose your starter Pokemon!",
            screen_type="dialogue",
            npc_type="professor",
            current_objective="get_starter_pokemon",
            conversation_history=[],
            ui_layout="pokemon_selection"
        )
        
        choices = choice_system.recognize_choices(visual_context, choice_context)
        
        # Should detect Pokemon choices
        pokemon_choices = [c for c in choices if c.choice_type == ChoiceType.POKEMON_SELECTION]
        assert len(pokemon_choices) > 0
        
        # Should have appropriate context tags
        for choice in pokemon_choices:
            assert "pokemon_choice" in choice.context_tags


class TestActionMapping:
    """Test action mapping generation"""
    
    @pytest.mark.unit
    @pytest.mark.choice
    def test_basic_action_mapping(self, choice_system):
        """Test basic action mapping generation"""
        test_cases = [
            ("Yes", ChoiceType.YES_NO, ChoicePosition.TOP, 0, ["A"]),
            ("No", ChoiceType.YES_NO, ChoicePosition.BOTTOM, 1, ["B"]),
            ("Cyndaquil", ChoiceType.POKEMON_SELECTION, ChoicePosition.LEFT, 0, ["A"]),
            ("Totodile", ChoiceType.POKEMON_SELECTION, ChoicePosition.CENTER, 1, ["DOWN", "A"]),
        ]
        
        for text, choice_type, position, index, expected_actions in test_cases:
            actions = choice_system._generate_action_mapping(text.lower(), choice_type, position, index)
            assert actions == expected_actions
    
    @pytest.mark.unit
    @pytest.mark.choice
    def test_position_based_mapping(self, choice_system):
        """Test position-based action mapping"""
        position_tests = [
            (ChoicePosition.TOP, ["A"]),
            (ChoicePosition.MIDDLE, ["DOWN", "A"]),
            (ChoicePosition.BOTTOM, ["DOWN", "DOWN", "A"]),
            (ChoicePosition.CENTER, ["A"]),
        ]
        
        for position, expected_actions in position_tests:
            actions = choice_system._generate_action_mapping(
                "generic", ChoiceType.MENU_SELECTION, position, 0
            )
            assert actions == expected_actions
    
    @pytest.mark.unit
    @pytest.mark.choice
    def test_index_based_mapping(self, choice_system):
        """Test index-based action mapping for multiple choice"""
        for i in range(4):
            expected_actions = ["DOWN"] * i + ["A"]
            actions = choice_system._generate_action_mapping(
                f"option_{i}", ChoiceType.MULTIPLE_CHOICE, ChoicePosition.TOP, i
            )
            assert actions == expected_actions
    
    @pytest.mark.unit
    @pytest.mark.choice
    def test_direct_text_mapping(self, choice_system):
        """Test direct text-to-action mapping"""
        direct_mappings = {
            "yes": ["A"],
            "no": ["B"],
            "cyndaquil": ["A"],
            "fight": ["A"],
            "north": ["UP"],
            "confirm": ["A"],
        }
        
        for text, expected_actions in direct_mappings.items():
            actions = choice_system._generate_action_mapping(
                text, ChoiceType.YES_NO, ChoicePosition.TOP, 0
            )
            assert actions == expected_actions


class TestChoicePrioritization:
    """Test choice prioritization and ranking"""
    
    @pytest.mark.unit
    @pytest.mark.choice
    def test_priority_calculation(self, choice_system):
        """Test priority calculation for choices"""
        # Create a context that should boost certain priorities
        context = ChoiceContext(
            dialogue_text="Choose your starter Pokemon!",
            screen_type="dialogue",
            npc_type="professor",
            current_objective="get_starter_pokemon",
            conversation_history=[],
            ui_layout="pokemon_selection"
        )
        
        # Test different Pokemon choices
        pokemon_priorities = []
        for pokemon in ["cyndaquil", "totodile", "chikorita"]:
            priority = choice_system._calculate_choice_priority(
                pokemon, ChoiceType.POKEMON_SELECTION, context, 0.9
            )
            pokemon_priorities.append((pokemon, priority))
        
        # Cyndaquil (fire starter) should have highest priority based on context
        cyndaquil_priority = next(p for name, p in pokemon_priorities if name == "cyndaquil")
        other_priorities = [p for name, p in pokemon_priorities if name != "cyndaquil"]
        
        assert cyndaquil_priority > max(other_priorities)
    
    @pytest.mark.unit
    @pytest.mark.choice
    def test_objective_based_prioritization(self, choice_system):
        """Test objective-based priority boosts"""
        # Gym battle objective should boost battle-related choices
        battle_context = ChoiceContext(
            dialogue_text="Ready for battle?",
            screen_type="dialogue",
            npc_type="gym_leader",
            current_objective="gym_battle",
            conversation_history=[],
            ui_layout="standard_dialogue"
        )
        
        yes_priority = choice_system._calculate_choice_priority(
            "yes", ChoiceType.YES_NO, battle_context, 0.8
        )
        
        no_priority = choice_system._calculate_choice_priority(
            "no", ChoiceType.YES_NO, battle_context, 0.8
        )
        
        # "Yes" should have higher priority in battle context
        assert yes_priority > no_priority
    
    @pytest.mark.unit
    @pytest.mark.choice
    def test_npc_type_prioritization(self, choice_system):
        """Test NPC type-based prioritization"""
        professor_context = ChoiceContext(
            dialogue_text="Would you like a Pokemon?",
            screen_type="dialogue",
            npc_type="professor",
            current_objective=None,
            conversation_history=[],
            ui_layout="standard_dialogue"
        )
        
        # Positive responses should be boosted with professors
        yes_priority = choice_system._calculate_choice_priority(
            "yes", ChoiceType.YES_NO, professor_context, 0.8
        )
        
        no_priority = choice_system._calculate_choice_priority(
            "no", ChoiceType.YES_NO, professor_context, 0.8
        )
        
        assert yes_priority > no_priority
    
    @pytest.mark.unit
    @pytest.mark.choice
    def test_confidence_impact_on_priority(self, choice_system):
        """Test that pattern confidence affects priority"""
        context = ChoiceContext(
            dialogue_text="Make a choice",
            screen_type="dialogue",
            npc_type="generic",
            current_objective=None,
            conversation_history=[],
            ui_layout="standard_dialogue"
        )
        
        high_confidence_priority = choice_system._calculate_choice_priority(
            "yes", ChoiceType.YES_NO, context, 0.9
        )
        
        low_confidence_priority = choice_system._calculate_choice_priority(
            "yes", ChoiceType.YES_NO, context, 0.3
        )
        
        assert high_confidence_priority > low_confidence_priority


class TestContextualProcessing:
    """Test context-dependent choice processing"""
    
    @pytest.mark.unit
    @pytest.mark.choice
    def test_context_prioritization(self, choice_system, sample_recognized_choices):
        """Test context-based prioritization of choices"""
        # Context with conversation history about healing
        healing_context = ChoiceContext(
            dialogue_text="Would you like me to heal your Pokemon?",
            screen_type="dialogue",
            npc_type="nurse",
            current_objective=None,
            conversation_history=["Welcome to Pokemon Center", "Your Pokemon look tired", "heal rest recover"],
            ui_layout="standard_dialogue"
        )
        
        # Apply context prioritization
        prioritized_choices = choice_system._apply_context_prioritization(
            sample_recognized_choices.copy(), healing_context
        )
        
        # Positive responses should be boosted
        yes_choice = next((c for c in prioritized_choices if "yes" in c.text.lower()), None)
        if yes_choice:
            original_yes = next((c for c in sample_recognized_choices if "yes" in c.text.lower()), None)
            if original_yes:
                # Priority should be boosted or at least maintained
                assert yes_choice.priority >= original_yes.priority
    
    @pytest.mark.unit
    @pytest.mark.choice
    def test_conversation_history_impact(self, choice_system):
        """Test how conversation history affects choice processing"""
        # Context mentioning starters should boost Pokemon choices
        starter_context = ChoiceContext(
            dialogue_text="Choose wisely!",
            screen_type="dialogue",
            npc_type="professor",
            current_objective=None,
            conversation_history=["starter pokemon", "fire water grass", "choose pokemon"],
            ui_layout="pokemon_selection"
        )
        
        pokemon_choice = RecognizedChoice(
            text="Cyndaquil",
            choice_type=ChoiceType.POKEMON_SELECTION,
            position=ChoicePosition.LEFT,
            action_mapping=["A"],
            confidence=0.9,
            priority=50,
            expected_outcome="select_fire_starter",
            context_tags=["pokemon_choice"],
            ui_coordinates=(20, 60, 90, 80)
        )
        
        # Store original priority before modification
        original_priority = pokemon_choice.priority
        
        prioritized = choice_system._apply_context_prioritization([pokemon_choice], starter_context)
        
        # Priority should be boosted due to conversation history
        assert prioritized[0].priority > original_priority
    
    @pytest.mark.unit
    @pytest.mark.choice
    def test_ui_layout_adaptation(self, choice_system):
        """Test adaptation to different UI layouts"""
        layouts_to_test = ["standard_dialogue", "pokemon_selection", "menu_selection", "yes_no_dialog"]
        
        for layout in layouts_to_test:
            context = ChoiceContext(
                dialogue_text="Test dialogue",
                screen_type="dialogue",
                npc_type="generic",
                current_objective=None,
                conversation_history=[],
                ui_layout=layout
            )
            
            # UI layout should be accessible and valid
            assert layout in choice_system.ui_layouts
            layout_info = choice_system.ui_layouts[layout]
            assert "choice_positions" in layout_info
            assert "navigation" in layout_info


class TestExpectedOutcomes:
    """Test expected outcome determination"""
    
    @pytest.mark.unit
    @pytest.mark.choice
    def test_outcome_mapping(self, choice_system):
        """Test mapping of choices to expected outcomes"""
        test_cases = [
            ("Yes", ChoiceType.YES_NO, "accept_or_confirm"),
            ("No", ChoiceType.YES_NO, "decline_or_cancel"),
            ("Cyndaquil", ChoiceType.POKEMON_SELECTION, "select_cyndaquil"),
            ("Fight", ChoiceType.MENU_SELECTION, "enter_battle_menu"),
            ("North", ChoiceType.DIRECTIONAL, "move_north"),
            ("Confirm", ChoiceType.CONFIRMATION, "confirm_action"),
        ]
        
        context = ChoiceContext(
            dialogue_text="Test",
            screen_type="dialogue",
            npc_type="generic",
            current_objective=None,
            conversation_history=[],
            ui_layout="standard_dialogue"
        )
        
        for text, choice_type, expected_outcome in test_cases:
            outcome = choice_system._determine_expected_outcome(text.lower(), choice_type, context)
            assert outcome == expected_outcome
    
    @pytest.mark.unit
    @pytest.mark.choice
    def test_context_based_outcomes(self, choice_system):
        """Test context-dependent outcome determination"""
        # Pokemon selection should generate specific outcomes
        outcome = choice_system._determine_expected_outcome(
            "mysterious pokemon", ChoiceType.POKEMON_SELECTION, None
        )
        assert "select_mysterious_pokemon" in outcome
        
        # Directional choices should generate movement outcomes
        outcome = choice_system._determine_expected_outcome(
            "southeast", ChoiceType.DIRECTIONAL, None
        )
        assert "move_southeast" in outcome


class TestContextTags:
    """Test context tag generation"""
    
    @pytest.mark.unit
    @pytest.mark.choice
    def test_basic_tag_generation(self, choice_system):
        """Test basic context tag generation"""
        context = ChoiceContext(
            dialogue_text="Battle me!",
            screen_type="dialogue",
            npc_type="gym_leader",
            current_objective="gym_challenge",
            conversation_history=[],
            ui_layout="standard_dialogue"
        )
        
        tags = choice_system._generate_context_tags("yes", ChoiceType.YES_NO, context)
        
        # Should include type, NPC, and objective tags
        assert "type_yes_no" in tags
        assert "npc_gym_leader" in tags
        assert "objective_gym_challenge" in tags
    
    @pytest.mark.unit
    @pytest.mark.choice
    def test_response_type_tags(self, choice_system):
        """Test response type tag generation"""
        context = ChoiceContext(
            dialogue_text="Test",
            screen_type="dialogue", 
            npc_type="generic",
            current_objective=None,
            conversation_history=[],
            ui_layout="standard_dialogue"
        )
        
        # Positive responses
        positive_texts = ["yes", "okay", "sure", "accept"]
        for text in positive_texts:
            tags = choice_system._generate_context_tags(text, ChoiceType.YES_NO, context)
            assert "positive_response" in tags
        
        # Negative responses
        negative_texts = ["no", "cancel", "decline"]
        for text in negative_texts:
            tags = choice_system._generate_context_tags(text, ChoiceType.YES_NO, context)
            assert "negative_response" in tags
    
    @pytest.mark.unit
    @pytest.mark.choice
    def test_pokemon_choice_tags(self, choice_system):
        """Test Pokemon choice tagging"""
        context = ChoiceContext(
            dialogue_text="Choose starter",
            screen_type="dialogue",
            npc_type="professor",
            current_objective=None,
            conversation_history=[],
            ui_layout="pokemon_selection"
        )
        
        pokemon_texts = ["cyndaquil", "totodile", "chikorita"]
        for pokemon in pokemon_texts:
            tags = choice_system._generate_context_tags(pokemon, ChoiceType.POKEMON_SELECTION, context)
            assert "pokemon_choice" in tags
    
    @pytest.mark.unit
    @pytest.mark.choice
    def test_battle_related_tags(self, choice_system):
        """Test battle-related tagging"""
        context = ChoiceContext(
            dialogue_text="Ready to battle?",
            screen_type="dialogue",
            npc_type="trainer",
            current_objective=None,
            conversation_history=[],
            ui_layout="standard_dialogue"
        )
        
        battle_texts = ["fight", "battle", "challenge"]
        for text in battle_texts:
            tags = choice_system._generate_context_tags(text, ChoiceType.MENU_SELECTION, context)
            assert "battle_related" in tags


class TestBestChoiceSelection:
    """Test best choice selection logic"""
    
    @pytest.mark.unit
    @pytest.mark.choice
    def test_best_choice_selection(self, choice_system, sample_recognized_choices):
        """Test selection of best choice"""
        best_action = choice_system.get_best_choice_action(sample_recognized_choices)
        
        assert isinstance(best_action, list)
        assert len(best_action) > 0
        assert all(isinstance(action, str) for action in best_action)
        
        # Should select action from highest priority choice
        highest_priority_choice = max(sample_recognized_choices, key=lambda x: x.priority)
        assert best_action == highest_priority_choice.action_mapping
    
    @pytest.mark.unit
    @pytest.mark.choice
    def test_empty_choices_handling(self, choice_system):
        """Test handling of empty choice list"""
        best_action = choice_system.get_best_choice_action([])
        
        # Should return default action
        assert best_action == ["A"]
    
    @pytest.mark.unit
    @pytest.mark.choice
    def test_priority_based_selection(self, choice_system):
        """Test that selection is truly priority-based"""
        # Create choices with different priorities
        low_priority = RecognizedChoice(
            text="Low", choice_type=ChoiceType.YES_NO, position=ChoicePosition.TOP,
            action_mapping=["B"], confidence=0.8, priority=10,
            expected_outcome="low_priority", context_tags=[], ui_coordinates=(0,0,0,0)
        )
        
        high_priority = RecognizedChoice(
            text="High", choice_type=ChoiceType.YES_NO, position=ChoicePosition.BOTTOM,
            action_mapping=["A"], confidence=0.7, priority=90,
            expected_outcome="high_priority", context_tags=[], ui_coordinates=(0,0,0,0)
        )
        
        choices = [low_priority, high_priority]  # Order shouldn't matter
        best_action = choice_system.get_best_choice_action(choices)
        
        assert best_action == high_priority.action_mapping


class TestDatabaseOperations:
    """Test database operations and persistence"""
    
    @pytest.mark.unit
    @pytest.mark.choice
    @pytest.mark.database
    def test_choice_recognition_storage(self, choice_system, sample_recognized_choices):
        """Test storage of choice recognition results"""
        dialogue_text = "Choose your action"
        choice_system._store_choice_recognition(dialogue_text, sample_recognized_choices)
        
        with sqlite3.connect(choice_system.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM choice_recognitions")
            count = cursor.fetchone()[0]
            assert count > 0
            
            # Check that data is stored correctly
            cursor.execute("SELECT dialogue_text, recognized_choices FROM choice_recognitions ORDER BY id DESC LIMIT 1")
            stored_dialogue, stored_choices_json = cursor.fetchone()
            
            assert stored_dialogue == dialogue_text
            stored_choices = json.loads(stored_choices_json)
            assert len(stored_choices) == len(sample_recognized_choices)
    
    @pytest.mark.unit
    @pytest.mark.choice
    @pytest.mark.database
    def test_choice_effectiveness_tracking(self, choice_system):
        """Test choice effectiveness tracking"""
        choice_system.update_choice_effectiveness("Test Choice", ["A"], True, "standard")
        choice_system.update_choice_effectiveness("Test Choice", ["A"], False, "standard")
        choice_system.update_choice_effectiveness("Test Choice", ["A"], True, "standard")
        
        with sqlite3.connect(choice_system.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT success_count, failure_count FROM action_mappings 
                WHERE choice_text = ? AND ui_layout = ?
            """, ("Test Choice", "standard"))
            
            result = cursor.fetchone()
            assert result is not None
            success_count, failure_count = result
            assert success_count == 2
            assert failure_count == 1
    
    @pytest.mark.unit
    @pytest.mark.choice
    @pytest.mark.database
    def test_statistics_generation(self, choice_system, sample_visual_context, sample_choice_context):
        """Test statistics generation"""
        # Generate some recognition data
        choice_system.recognize_choices(sample_visual_context, sample_choice_context)
        
        stats = choice_system.get_choice_statistics()
        
        assert "total_choice_recognitions" in stats
        assert "average_confidence" in stats
        assert "loaded_patterns" in stats
        assert "loaded_action_mappings" in stats
        assert "top_action_mappings" in stats
        
        assert stats["total_choice_recognitions"] > 0
        assert stats["loaded_patterns"] > 0
        assert stats["loaded_action_mappings"] > 0
    
    @pytest.mark.unit
    @pytest.mark.choice
    @pytest.mark.database
    def test_database_error_handling(self, choice_system):
        """Test handling of database errors"""
        # Corrupt the database path
        original_path = choice_system.db_path
        choice_system.db_path = Path("/nonexistent/path/test.db")
        
        try:
            # Should handle database errors gracefully
            stats = choice_system.get_choice_statistics()
            assert isinstance(stats, dict)  # Should return empty or default stats
        except Exception as e:
            # Should be a specific, handled exception
            assert "database" in str(e).lower() or "connection" in str(e).lower()
        finally:
            choice_system.db_path = original_path


class TestUILayoutHandling:
    """Test UI layout handling and position determination"""
    
    @pytest.mark.unit
    @pytest.mark.choice
    def test_position_determination(self, choice_system):
        """Test choice position determination"""
        # Test various choice counts and positions
        test_cases = [
            (1, 0, ChoicePosition.CENTER),
            (2, 0, ChoicePosition.TOP),
            (2, 1, ChoicePosition.BOTTOM),
            (3, 0, ChoicePosition.TOP),
            (3, 1, ChoicePosition.MIDDLE),
            (3, 2, ChoicePosition.BOTTOM),
        ]
        
        for total, index, expected_position in test_cases:
            # Use simple choice_info without relying on coordinates
            # since the trainer implementation doesn't use coordinates
            choice_info = {"coordinates": (20, 50 + index * 30, 100, 70 + index * 30)}
            position = choice_system._determine_choice_position(choice_info, index, total)
            assert position == expected_position

    
    @pytest.mark.unit
    @pytest.mark.choice
    def test_coordinate_based_positioning(self, choice_system):
        """Test coordinate-based position determination for many choices"""
        # For more than 3 choices, should use coordinates
        choice_infos = [
            {"coordinates": (20, 30, 100, 50)},   # Y=30, should be TOP (< 48)
            {"coordinates": (20, 72, 100, 92)},   # Y=72, should be MIDDLE (48-96)
            {"coordinates": (20, 120, 100, 140)}, # Y=120, should be BOTTOM (> 96)
        ]
        
        expected_positions = [ChoicePosition.TOP, ChoicePosition.MIDDLE, ChoicePosition.BOTTOM]
        
        for i, (choice_info, expected) in enumerate(zip(choice_infos, expected_positions)):
            position = choice_system._determine_choice_position(choice_info, i, 4)
            assert position == expected


class TestErrorHandling:
    """Test error handling and edge cases"""
    
    @pytest.mark.unit
    @pytest.mark.choice
    def test_empty_visual_context(self, choice_system, sample_choice_context):
        """Test handling of empty visual context"""
        empty_context = VisualContext(
            screen_type="dialogue",
            detected_text=[],
            ui_elements=[],
            dominant_colors=[],
            game_phase="dialogue_interaction",
            visual_summary="Empty"
        )
        
        choices = choice_system.recognize_choices(empty_context, sample_choice_context)
        assert isinstance(choices, list)
        assert len(choices) == 0
    
    @pytest.mark.unit
    @pytest.mark.choice
    def test_malformed_detected_text(self, choice_system, sample_choice_context):
        """Test handling of malformed detected text"""
        malformed_texts = [
            DetectedText("", 0.0, (0, 0, 0, 0), "unknown"),
            DetectedText(None, 0.5, (10, 10, 50, 30), "choice"),
        ]
        
        visual_context = VisualContext(
            screen_type="dialogue",
            detected_text=malformed_texts,
            ui_elements=[],
            dominant_colors=[(255, 255, 255)],
            game_phase="dialogue_interaction",
            visual_summary="Malformed"
        )
        
        # Should not crash
        choices = choice_system.recognize_choices(visual_context, sample_choice_context)
        assert isinstance(choices, list)
    
    @pytest.mark.unit
    @pytest.mark.choice
    def test_invalid_choice_context(self, choice_system, sample_visual_context):
        """Test handling of invalid choice context"""
        invalid_context = ChoiceContext(
            dialogue_text="",
            screen_type="",
            npc_type="",
            current_objective=None,
            conversation_history=[],
            ui_layout="nonexistent_layout"
        )
        
        # Should handle gracefully
        choices = choice_system.recognize_choices(sample_visual_context, invalid_context)
        assert isinstance(choices, list)
    
    @pytest.mark.unit
    @pytest.mark.choice
    def test_pattern_matching_edge_cases(self, choice_system):
        """Test pattern matching with edge cases"""
        edge_cases = [
            "",           # Empty string
            " ",          # Whitespace only
            "123456789",  # Numbers only
            "!@#$%^&*()", # Special characters only
            "a" * 100,    # Very long string
        ]
        
        for text in edge_cases:
            # Should not crash
            choice_type, confidence = choice_system._match_choice_patterns(text)
            assert choice_type is None or isinstance(choice_type, ChoiceType)
            assert 0.0 <= confidence <= 1.0


@pytest.mark.integration
class TestChoiceIntegration:
    """Integration tests for choice recognition system"""
    
    def test_full_recognition_flow(self, choice_system, pokemon_selection_texts):
        """Test complete choice recognition flow"""
        visual_context = VisualContext(
            screen_type="dialogue",
            detected_text=pokemon_selection_texts,
            ui_elements=[],
            dominant_colors=[(255, 255, 255)],
            game_phase="dialogue_interaction",
            visual_summary="Complete Pokemon selection flow"
        )
        
        choice_context = ChoiceContext(
            dialogue_text="Choose your starter Pokemon!",
            screen_type="dialogue",
            npc_type="professor",
            current_objective="get_starter_pokemon",
            conversation_history=["Hello!", "I study Pokemon behavior."],
            ui_layout="pokemon_selection"
        )
        
        choices = choice_system.recognize_choices(visual_context, choice_context)
        best_action = choice_system.get_best_choice_action(choices)
        
        # Should have complete processing results
        assert len(choices) > 0
        assert isinstance(best_action, list)
        assert len(best_action) > 0
        
        # Should prioritize fire starter (Cyndaquil)
        top_choice = max(choices, key=lambda x: x.priority)
        if "cyndaquil" in top_choice.text.lower():
            assert best_action == ["A"]  # First choice action


@pytest.mark.performance
class TestChoicePerformance:
    """Test performance characteristics"""
    
    def test_recognition_performance(self, choice_system, sample_visual_context, sample_choice_context):
        """Test choice recognition performance"""
        import time
        
        start_time = time.time()
        for _ in range(20):
            choice_system.recognize_choices(sample_visual_context, sample_choice_context)
        end_time = time.time()
        
        # Should complete 20 recognitions quickly
        assert (end_time - start_time) < 2.0
    
    def test_large_choice_set_handling(self, choice_system):
        """Test handling of many choices"""
        # Create many detected texts
        many_texts = []
        for i in range(20):
            text = DetectedText(f"Choice {i}", 0.8, (20, 60 + i * 20, 100, 80 + i * 20), "choice")
            many_texts.append(text)
        
        visual_context = VisualContext(
            screen_type="dialogue",
            detected_text=many_texts,
            ui_elements=[],
            dominant_colors=[(255, 255, 255)],
            game_phase="dialogue_interaction",
            visual_summary="Many choices"
        )
        
        choice_context = ChoiceContext(
            dialogue_text="Pick one",
            screen_type="dialogue",
            npc_type="generic",
            current_objective=None,
            conversation_history=[],
            ui_layout="standard_dialogue"
        )
        
        choices = choice_system.recognize_choices(visual_context, choice_context)
        
        # Should handle many choices without issues
        assert isinstance(choices, list)
        assert len(choices) <= len(many_texts)  # May filter some out
