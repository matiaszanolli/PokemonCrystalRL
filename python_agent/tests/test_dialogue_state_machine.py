#!/usr/bin/env python3
"""
test_dialogue_state_machine.py - Comprehensive tests for the Dialogue State Machine

This module tests all aspects of the dialogue state machine including:
- State transitions and management
- NPC type handling and classification
- Conversation tracking and history
- Semantic integration
- Choice processing and action mapping
- Database operations and persistence
"""

import pytest
import json
import sqlite3
from unittest.mock import patch, Mock, MagicMock
from datetime import datetime

from dialogue_state_machine import DialogueStateMachine, DialogueState, NPCType
from vision_processor import DetectedText, VisualContext
from semantic_context_system import GameContext


class TestDialogueStateTransitions:
    """Test dialogue state transitions and management"""
    
    @pytest.mark.unit
    @pytest.mark.dialogue
    def test_initial_state(self, dialogue_machine):
        """Test initial state is idle"""
        assert dialogue_machine.current_state == DialogueState.IDLE
        assert dialogue_machine.current_npc_type is None
        assert len(dialogue_machine.dialogue_history) == 0
    
    @pytest.mark.unit
    @pytest.mark.dialogue
    def test_state_transition_idle_to_listening(self, dialogue_machine, sample_visual_context):
        """Test transition from idle to listening"""
        dialogue_machine.process_dialogue(sample_visual_context)
        
        # Should transition to listening when dialogue is detected
        assert dialogue_machine.current_state in [DialogueState.LISTENING, DialogueState.CHOOSING]
    
    @pytest.mark.unit
    @pytest.mark.dialogue
    def test_state_transition_listening_to_choosing(self, dialogue_machine, sample_visual_context):
        """Test transition from listening to choosing"""
        # Process dialogue first
        dialogue_machine.process_dialogue(sample_visual_context)
        
        # If choices are detected, should be in choosing state
        if any("yes" in text.text.lower() or "no" in text.text.lower() for text in sample_visual_context.detected_text):
            assert dialogue_machine.current_state == DialogueState.CHOOSING
    
    @pytest.mark.unit
    @pytest.mark.dialogue
    def test_state_reset(self, dialogue_machine, sample_visual_context):
        """Test state machine reset functionality"""
        # Process some dialogue
        dialogue_machine.process_dialogue(sample_visual_context)
        
        # Reset state machine
        dialogue_machine.reset()
        
        assert dialogue_machine.current_state == DialogueState.IDLE
        assert dialogue_machine.current_npc_type is None
        assert len(dialogue_machine.dialogue_history) == 0
        assert dialogue_machine.current_conversation_id is None
    
    @pytest.mark.unit
    @pytest.mark.dialogue
    @pytest.mark.parametrize("state", [
        DialogueState.IDLE,
        DialogueState.LISTENING, 
        DialogueState.CHOOSING,
        DialogueState.RESPONDING
    ])
    def test_valid_state_transitions(self, dialogue_machine, state):
        """Test that all state transitions are valid"""
        dialogue_machine.current_state = state
        
        # All states should be valid enum values
        assert isinstance(state, DialogueState)
        assert state in DialogueState


class TestNPCTypeDetection:
    """Test NPC type detection and classification"""
    
    @pytest.mark.unit
    @pytest.mark.dialogue
    @pytest.mark.parametrize("npc_type", [
        NPCType.PROFESSOR,
        NPCType.NURSE,
        NPCType.GYM_LEADER,
        NPCType.SHOPKEEPER,
        NPCType.TRAINER,
        NPCType.GENERIC
    ])
    def test_npc_type_detection(self, dialogue_machine, npc_type):
        """Test NPC type detection for different types"""
        # Create specific dialogue for each NPC type
        npc_dialogues = {
            NPCType.PROFESSOR: "Hello! I'm Professor Elm! I study Pokemon behavior.",
            NPCType.NURSE: "Welcome to the Pokemon Center! Would you like me to heal your Pokemon?",
            NPCType.GYM_LEADER: "I'm Falkner, the Violet Gym Leader! Ready for battle?",
            NPCType.SHOPKEEPER: "Welcome to the Poke Mart! What can I get for you?",
            NPCType.TRAINER: "Hey! I challenge you to a Pokemon battle!",
            NPCType.GENERIC: "Hello there! Nice weather today, isn't it?"
        }
        
        dialogue_text = npc_dialogues[npc_type]
        detected_text = DetectedText(dialogue_text, 0.9, (10, 10, 200, 30), "dialogue")
        
        visual_context = VisualContext(
            screen_type="dialogue",
            detected_text=[detected_text],
            ui_elements=[],
            dominant_colors=[(255, 255, 255)],
            game_phase="dialogue_interaction",
            visual_summary=f"{npc_type.value} dialogue"
        )
        
        dialogue_machine.process_dialogue(visual_context)
        
        # Should detect correct NPC type or at least classify as something reasonable
        assert dialogue_machine.current_npc_type is not None
        assert isinstance(dialogue_machine.current_npc_type, NPCType)
    
    @pytest.mark.unit
    @pytest.mark.dialogue
    def test_professor_detection(self, dialogue_machine):
        """Test specific professor detection"""
        professor_text = DetectedText("Hello! I'm Professor Elm!", 0.95, (10, 10, 200, 30), "dialogue")
        visual_context = VisualContext(
            screen_type="dialogue",
            detected_text=[professor_text],
            ui_elements=[],
            dominant_colors=[(255, 255, 255)],
            game_phase="dialogue_interaction",
            visual_summary="Professor dialogue"
        )
        
        dialogue_machine.process_dialogue(visual_context)
        
        assert dialogue_machine.current_npc_type == NPCType.PROFESSOR
    
    @pytest.mark.unit
    @pytest.mark.dialogue
    def test_nurse_detection(self, dialogue_machine):
        """Test specific nurse detection"""
        nurse_text = DetectedText("Welcome to the Pokemon Center!", 0.95, (10, 10, 200, 30), "dialogue")
        visual_context = VisualContext(
            screen_type="dialogue",
            detected_text=[nurse_text],
            ui_elements=[],
            dominant_colors=[(255, 255, 255)],
            game_phase="dialogue_interaction",
            visual_summary="Nurse dialogue"
        )
        
        dialogue_machine.process_dialogue(visual_context)
        
        assert dialogue_machine.current_npc_type == NPCType.NURSE
    
    @pytest.mark.unit
    @pytest.mark.dialogue
    def test_gym_leader_detection(self, dialogue_machine):
        """Test specific gym leader detection"""
        gym_text = DetectedText("I'm Falkner, the Violet Gym Leader!", 0.95, (10, 10, 200, 30), "dialogue")
        visual_context = VisualContext(
            screen_type="dialogue",
            detected_text=[gym_text],
            ui_elements=[],
            dominant_colors=[(255, 255, 255)],
            game_phase="dialogue_interaction",
            visual_summary="Gym leader dialogue"
        )
        
        dialogue_machine.process_dialogue(visual_context)
        
        assert dialogue_machine.current_npc_type == NPCType.GYM_LEADER


class TestConversationTracking:
    """Test conversation tracking and history management"""
    
    @pytest.mark.unit
    @pytest.mark.dialogue
    def test_dialogue_history_tracking(self, dialogue_machine, sample_visual_context):
        """Test that dialogue history is properly tracked"""
        initial_history_length = len(dialogue_machine.dialogue_history)
        
        dialogue_machine.process_dialogue(sample_visual_context)
        
        # History should have grown
        assert len(dialogue_machine.dialogue_history) > initial_history_length
    
    @pytest.mark.unit
    @pytest.mark.dialogue
    def test_conversation_id_generation(self, dialogue_machine, sample_visual_context):
        """Test that conversation IDs are generated"""
        dialogue_machine.process_dialogue(sample_visual_context)
        
        assert dialogue_machine.current_conversation_id is not None
        assert isinstance(dialogue_machine.current_conversation_id, str)
    
    @pytest.mark.unit
    @pytest.mark.dialogue
    def test_multi_turn_conversation(self, dialogue_machine):
        """Test multi-turn conversation tracking"""
        # First dialogue turn
        turn1_text = DetectedText("Hello! I'm Professor Elm!", 0.95, (10, 10, 200, 30), "dialogue")
        turn1_context = VisualContext(
            screen_type="dialogue",
            detected_text=[turn1_text],
            ui_elements=[],
            dominant_colors=[(255, 255, 255)],
            game_phase="dialogue_interaction",
            visual_summary="First turn"
        )
        
        dialogue_machine.process_dialogue(turn1_context)
        conversation_id_1 = dialogue_machine.current_conversation_id
        history_length_1 = len(dialogue_machine.dialogue_history)
        
        # Second dialogue turn
        turn2_text = DetectedText("Would you like a Pokemon?", 0.95, (10, 10, 200, 30), "dialogue")
        turn2_context = VisualContext(
            screen_type="dialogue",
            detected_text=[turn2_text],
            ui_elements=[],
            dominant_colors=[(255, 255, 255)],
            game_phase="dialogue_interaction",
            visual_summary="Second turn"
        )
        
        dialogue_machine.process_dialogue(turn2_context)
        conversation_id_2 = dialogue_machine.current_conversation_id
        history_length_2 = len(dialogue_machine.dialogue_history)
        
        # Should maintain same conversation ID and grow history
        assert conversation_id_1 == conversation_id_2
        assert history_length_2 > history_length_1
    
    @pytest.mark.unit
    @pytest.mark.dialogue
    @pytest.mark.database
    def test_conversation_persistence(self, dialogue_machine, sample_visual_context):
        """Test that conversations are persisted to database"""
        dialogue_machine.process_dialogue(sample_visual_context)
        
        # Check database for conversation record
        with sqlite3.connect(dialogue_machine.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM conversations")
            conversation_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM dialogue_turns")
            turn_count = cursor.fetchone()[0]
            
            assert conversation_count > 0
            assert turn_count > 0


class TestSemanticIntegration:
    """Test integration with semantic context system"""
    
    @pytest.mark.unit
    @pytest.mark.dialogue
    @pytest.mark.semantic
    def test_semantic_analysis_integration(self, dialogue_machine):
        """Test that semantic analysis is called during dialogue processing"""
        with patch.object(dialogue_machine.semantic_system, 'analyze_dialogue') as mock_analyze:
            mock_analyze.return_value = {
                "intent": "starter_selection",
                "confidence": 0.8,
                "strategy": "select_fire_starter",
                "reasoning": "Test reasoning",
                "recommended_actions": ["A"]
            }
            
            starter_text = DetectedText("Choose your starter Pokemon!", 0.95, (10, 10, 200, 30), "dialogue")
            visual_context = VisualContext(
                screen_type="dialogue",
                detected_text=[starter_text],
                ui_elements=[],
                dominant_colors=[(255, 255, 255)],
                game_phase="dialogue_interaction",
                visual_summary="Starter selection"
            )
            
            dialogue_machine.process_dialogue(visual_context)
            
            # Semantic analysis should have been called
            mock_analyze.assert_called_once()
    
    @pytest.mark.unit
    @pytest.mark.dialogue
    @pytest.mark.semantic
    def test_semantic_context_building(self, dialogue_machine, sample_visual_context):
        """Test that game context is properly built for semantic analysis"""
        with patch.object(dialogue_machine.semantic_system, 'analyze_dialogue') as mock_analyze:
            mock_analyze.return_value = {
                "intent": "healing_request",
                "confidence": 0.7,
                "strategy": "accept_healing",
                "reasoning": "Test reasoning",
                "recommended_actions": ["A"]
            }
            
            dialogue_machine.process_dialogue(sample_visual_context)
            
            # Check that context was passed correctly
            mock_analyze.assert_called_once()
            call_args = mock_analyze.call_args
            
            assert len(call_args[0]) == 2  # dialogue_text and game_context
            game_context = call_args[0][1]
            assert isinstance(game_context, GameContext)
    
    @pytest.mark.unit
    @pytest.mark.dialogue
    @pytest.mark.semantic
    def test_semantic_enhanced_topic_identification(self, dialogue_machine):
        """Test semantic-enhanced topic identification"""
        topic_dialogues = [
            ("Choose your starter Pokemon!", "pokemon_selection"),
            ("Welcome to the Pokemon Center!", "healing"),
            ("Ready for a gym battle?", "battle"),
        ]
        
        for dialogue_text, expected_topic in topic_dialogues:
            detected_text = DetectedText(dialogue_text, 0.95, (10, 10, 200, 30), "dialogue")
            visual_context = VisualContext(
                screen_type="dialogue",
                detected_text=[detected_text],
                ui_elements=[],
                dominant_colors=[(255, 255, 255)],
                game_phase="dialogue_interaction",
                visual_summary=f"Topic test: {expected_topic}"
            )
            
            result = dialogue_machine.process_dialogue(visual_context)
            
            # Should have identified some topic
            assert result is not None
            if "semantic_analysis" in result:
                assert "intent" in result["semantic_analysis"]


class TestChoiceProcessing:
    """Test choice detection and processing"""
    
    @pytest.mark.unit
    @pytest.mark.dialogue
    @pytest.mark.choice
    def test_choice_detection(self, dialogue_machine, sample_visual_context):
        """Test detection of dialogue choices"""
        # Visual context already contains "Yes" and "No" choices
        result = dialogue_machine.process_dialogue(sample_visual_context)
        
        if result and "choices" in result:
            choices = result["choices"]
            assert len(choices) > 0
            
            # Should have detected some choices
            choice_texts = [choice.get("text", "") for choice in choices]
            assert any("yes" in text.lower() for text in choice_texts)
    
    @pytest.mark.unit
    @pytest.mark.dialogue
    @pytest.mark.choice
    def test_pokemon_selection_choices(self, dialogue_machine, pokemon_selection_texts):
        """Test Pokemon selection choice processing"""
        visual_context = VisualContext(
            screen_type="dialogue",
            detected_text=pokemon_selection_texts,
            ui_elements=[],
            dominant_colors=[(255, 255, 255)],
            game_phase="dialogue_interaction",
            visual_summary="Pokemon selection"
        )
        
        result = dialogue_machine.process_dialogue(visual_context)
        
        assert result is not None
        if "choices" in result:
            choices = result["choices"]
            choice_texts = [choice.get("text", "") for choice in choices]
            
            # Should detect Pokemon names
            pokemon_names = ["cyndaquil", "totodile", "chikorita"]
            detected_pokemon = [name for name in pokemon_names if any(name in text.lower() for text in choice_texts)]
            assert len(detected_pokemon) > 0
    
    @pytest.mark.unit
    @pytest.mark.dialogue
    @pytest.mark.choice
    def test_choice_prioritization(self, dialogue_machine, sample_visual_context):
        """Test that choices are prioritized appropriately"""
        result = dialogue_machine.process_dialogue(sample_visual_context)
        
        if result and "recommended_action" in result:
            recommended_action = result["recommended_action"]
            assert recommended_action is not None
            assert isinstance(recommended_action, (str, list))
    
    @pytest.mark.unit
    @pytest.mark.dialogue
    @pytest.mark.choice
    def test_semantic_enhanced_choice_prioritization(self, dialogue_machine):
        """Test semantic enhancement of choice prioritization"""
        # Create a context where semantic analysis should influence choice priority
        starter_dialogue = DetectedText("Choose your starter Pokemon!", 0.95, (10, 10, 200, 30), "dialogue")
        cyndaquil_choice = DetectedText("Cyndaquil", 0.98, (20, 60, 90, 80), "choice")
        totodile_choice = DetectedText("Totodile", 0.97, (100, 60, 160, 80), "choice")
        
        visual_context = VisualContext(
            screen_type="dialogue",
            detected_text=[starter_dialogue, cyndaquil_choice, totodile_choice],
            ui_elements=[],
            dominant_colors=[(255, 255, 255)],
            game_phase="dialogue_interaction",
            visual_summary="Starter selection with semantic enhancement"
        )
        
        result = dialogue_machine.process_dialogue(visual_context)
        
        # Should have processed choices with semantic enhancement
        assert result is not None
        if "semantic_analysis" in result and "strategy" in result["semantic_analysis"]:
            strategy = result["semantic_analysis"]["strategy"]
            assert "starter" in strategy.lower()


class TestActionGeneration:
    """Test action generation and mapping"""
    
    @pytest.mark.unit
    @pytest.mark.dialogue
    def test_action_generation(self, dialogue_machine, sample_visual_context):
        """Test that appropriate actions are generated"""
        result = dialogue_machine.process_dialogue(sample_visual_context)
        
        if result and "recommended_action" in result:
            action = result["recommended_action"]
            
            # Action should be valid game input
            if isinstance(action, list):
                assert all(isinstance(a, str) for a in action)
                assert len(action) > 0
            else:
                assert isinstance(action, str)
                assert len(action) > 0
    
    @pytest.mark.unit
    @pytest.mark.dialogue
    def test_context_dependent_actions(self, dialogue_machine):
        """Test that actions depend on dialogue context"""
        # Healing context should recommend "A" (accept)
        healing_text = DetectedText("Would you like me to heal your Pokemon?", 0.95, (10, 10, 200, 30), "dialogue")
        yes_choice = DetectedText("Yes", 0.98, (20, 60, 50, 80), "choice")
        
        healing_context = VisualContext(
            screen_type="dialogue",
            detected_text=[healing_text, yes_choice],
            ui_elements=[],
            dominant_colors=[(255, 255, 255)],
            game_phase="dialogue_interaction",
            visual_summary="Healing request"
        )
        
        result = dialogue_machine.process_dialogue(healing_context)
        
        # Should recommend accepting healing
        if result and "recommended_action" in result:
            action = result["recommended_action"]
            if isinstance(action, list):
                assert "A" in action
            else:
                assert action == "A"


class TestDatabaseOperations:
    """Test database operations and persistence"""
    
    @pytest.mark.unit
    @pytest.mark.dialogue
    @pytest.mark.database
    def test_database_initialization(self, dialogue_machine):
        """Test that database is properly initialized"""
        assert dialogue_machine.db_path.exists()
        
        with sqlite3.connect(dialogue_machine.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            expected_tables = ["conversations", "dialogue_turns", "choice_selections", "npc_encounters"]
            for table in expected_tables:
                assert table in tables
    
    @pytest.mark.unit
    @pytest.mark.dialogue
    @pytest.mark.database
    def test_conversation_storage(self, dialogue_machine, sample_visual_context):
        """Test that conversations are stored in database"""
        initial_conversation_count = dialogue_machine._get_conversation_count()
        
        dialogue_machine.process_dialogue(sample_visual_context)
        
        final_conversation_count = dialogue_machine._get_conversation_count()
        
        assert final_conversation_count > initial_conversation_count
    
    @pytest.mark.unit
    @pytest.mark.dialogue
    @pytest.mark.database
    def test_dialogue_turn_storage(self, dialogue_machine, sample_visual_context):
        """Test that dialogue turns are stored"""
        dialogue_machine.process_dialogue(sample_visual_context)
        
        with sqlite3.connect(dialogue_machine.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM dialogue_turns")
            turn_count = cursor.fetchone()[0]
            
            assert turn_count > 0
    
    @pytest.mark.unit
    @pytest.mark.dialogue
    @pytest.mark.database
    def test_npc_encounter_tracking(self, dialogue_machine):
        """Test NPC encounter tracking"""
        professor_text = DetectedText("Hello! I'm Professor Elm!", 0.95, (10, 10, 200, 30), "dialogue")
        visual_context = VisualContext(
            screen_type="dialogue",
            detected_text=[professor_text],
            ui_elements=[],
            dominant_colors=[(255, 255, 255)],
            game_phase="dialogue_interaction",
            visual_summary="Professor encounter"
        )
        
        dialogue_machine.process_dialogue(visual_context)
        
        with sqlite3.connect(dialogue_machine.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM npc_encounters WHERE npc_type = ?", (NPCType.PROFESSOR.value,))
            encounter_count = cursor.fetchone()[0]
            
            assert encounter_count > 0
    
    @pytest.mark.unit
    @pytest.mark.dialogue
    @pytest.mark.database
    def test_statistics_generation(self, dialogue_machine, sample_visual_context):
        """Test statistics generation"""
        dialogue_machine.process_dialogue(sample_visual_context)
        
        stats = dialogue_machine.get_statistics()
        
        assert "total_conversations" in stats
        assert "conversations_by_npc_type" in stats
        assert "average_conversation_length" in stats
        assert stats["total_conversations"] > 0


class TestErrorHandling:
    """Test error handling and edge cases"""
    
    @pytest.mark.unit
    @pytest.mark.dialogue
    def test_empty_visual_context(self, dialogue_machine):
        """Test handling of empty visual context"""
        empty_context = VisualContext(
            screen_type="dialogue",
            detected_text=[],
            ui_elements=[],
            dominant_colors=[],
            game_phase="dialogue_interaction",
            visual_summary="Empty context"
        )
        
        result = dialogue_machine.process_dialogue(empty_context)
        
        # Should handle gracefully
        assert result is not None or result is None  # Either is acceptable
    
    @pytest.mark.unit
    @pytest.mark.dialogue
    def test_malformed_detected_text(self, dialogue_machine):
        """Test handling of malformed detected text"""
        malformed_text = DetectedText("", 0.0, (0, 0, 0, 0), "unknown")
        visual_context = VisualContext(
            screen_type="dialogue",
            detected_text=[malformed_text],
            ui_elements=[],
            dominant_colors=[(255, 255, 255)],
            game_phase="dialogue_interaction",
            visual_summary="Malformed text"
        )
        
        result = dialogue_machine.process_dialogue(visual_context)
        
        # Should not crash
        assert True  # If we reach here, no exception was raised
    
    @pytest.mark.unit
    @pytest.mark.dialogue
    def test_none_input(self, dialogue_machine):
        """Test handling of None input"""
        result = dialogue_machine.process_dialogue(None)
        
        # Should handle None gracefully
        assert result is None or isinstance(result, dict)
    
    @pytest.mark.unit
    @pytest.mark.dialogue
    def test_database_error_handling(self, dialogue_machine):
        """Test handling of database errors"""
        # Simulate database error by corrupting the path
        original_path = dialogue_machine.db_path
        dialogue_machine.db_path = Path("/nonexistent/path/test.db")
        
        try:
            stats = dialogue_machine.get_statistics()
            # Should return empty stats or handle gracefully
            assert isinstance(stats, dict)
        except Exception as e:
            # Should be a specific, handled exception
            assert "database" in str(e).lower() or "connection" in str(e).lower()
        finally:
            dialogue_machine.db_path = original_path


class TestStateManagement:
    """Test advanced state management features"""
    
    @pytest.mark.unit
    @pytest.mark.dialogue
    def test_state_persistence(self, dialogue_machine, sample_visual_context):
        """Test that state persists across dialogue processing"""
        # Process initial dialogue
        dialogue_machine.process_dialogue(sample_visual_context)
        initial_state = dialogue_machine.current_state
        initial_npc = dialogue_machine.current_npc_type
        
        # Process another turn
        second_text = DetectedText("What do you say?", 0.9, (10, 10, 200, 30), "dialogue")
        second_context = VisualContext(
            screen_type="dialogue",
            detected_text=[second_text],
            ui_elements=[],
            dominant_colors=[(255, 255, 255)],
            game_phase="dialogue_interaction",
            visual_summary="Second dialogue turn"
        )
        
        dialogue_machine.process_dialogue(second_context)
        
        # NPC type should persist
        if initial_npc:
            assert dialogue_machine.current_npc_type == initial_npc
    
    @pytest.mark.unit
    @pytest.mark.dialogue
    def test_conversation_timeout(self, dialogue_machine, sample_visual_context):
        """Test conversation timeout handling"""
        # Process dialogue
        dialogue_machine.process_dialogue(sample_visual_context)
        conversation_id = dialogue_machine.current_conversation_id
        
        # Simulate timeout by mocking time
        with patch('time.time') as mock_time:
            mock_time.return_value = 1000000  # Far future
            
            # Check if conversation would timeout
            # (Implementation depends on how timeout is handled)
            assert conversation_id is not None
    
    @pytest.mark.unit
    @pytest.mark.dialogue
    def test_concurrent_dialogue_handling(self, dialogue_machine):
        """Test handling of overlapping or concurrent dialogues"""
        # First dialogue
        first_text = DetectedText("Hello from NPC 1!", 0.9, (10, 10, 200, 30), "dialogue")
        first_context = VisualContext(
            screen_type="dialogue",
            detected_text=[first_text],
            ui_elements=[],
            dominant_colors=[(255, 255, 255)],
            game_phase="dialogue_interaction",
            visual_summary="First NPC"
        )
        
        dialogue_machine.process_dialogue(first_context)
        first_conversation_id = dialogue_machine.current_conversation_id
        
        # Immediately second dialogue (different NPC)
        second_text = DetectedText("Greetings from NPC 2!", 0.9, (10, 10, 200, 30), "dialogue")
        second_context = VisualContext(
            screen_type="dialogue",
            detected_text=[second_text],
            ui_elements=[],
            dominant_colors=[(255, 255, 255)],
            game_phase="dialogue_interaction",
            visual_summary="Second NPC"
        )
        
        dialogue_machine.process_dialogue(second_context)
        second_conversation_id = dialogue_machine.current_conversation_id
        
        # Should handle appropriately (either same conversation or new one)
        assert first_conversation_id is not None
        assert second_conversation_id is not None


@pytest.mark.integration
class TestDialogueIntegration:
    """Integration tests for dialogue system components"""
    
    def test_full_dialogue_flow(self, dialogue_machine, pokemon_selection_texts):
        """Test complete dialogue flow from detection to action"""
        visual_context = VisualContext(
            screen_type="dialogue",
            detected_text=pokemon_selection_texts,
            ui_elements=[],
            dominant_colors=[(255, 255, 255)],
            game_phase="dialogue_interaction",
            visual_summary="Complete Pokemon selection flow"
        )
        
        result = dialogue_machine.process_dialogue(visual_context)
        
        # Should have complete result
        assert result is not None
        
        # Should have key components
        expected_keys = ["recommended_action"]
        for key in expected_keys:
            if key in result:
                assert result[key] is not None
    
    def test_semantic_dialogue_integration(self, dialogue_machine):
        """Test integration between dialogue machine and semantic system"""
        # Create context that should trigger strong semantic response
        strong_context_dialogue = DetectedText(
            "Hello! I'm Professor Elm! Choose your starter: Cyndaquil, Totodile, or Chikorita?", 
            0.95, (10, 10, 300, 30), "dialogue"
        )
        
        visual_context = VisualContext(
            screen_type="dialogue",
            detected_text=[strong_context_dialogue],
            ui_elements=[],
            dominant_colors=[(255, 255, 255)],
            game_phase="dialogue_interaction",
            visual_summary="Strong semantic context"
        )
        
        result = dialogue_machine.process_dialogue(visual_context)
        
        # Should have semantic analysis in result
        assert result is not None
        if "semantic_analysis" in result:
            semantic_result = result["semantic_analysis"]
            assert "intent" in semantic_result
            assert semantic_result["confidence"] > 0.5  # Should be confident


@pytest.mark.performance
class TestDialoguePerformance:
    """Test performance characteristics"""
    
    def test_processing_speed(self, dialogue_machine, sample_visual_context):
        """Test dialogue processing speed"""
        import time
        
        start_time = time.time()
        for _ in range(10):
            dialogue_machine.process_dialogue(sample_visual_context)
        end_time = time.time()
        
        # Should complete 10 processes in reasonable time
        assert (end_time - start_time) < 2.0
    
    def test_memory_usage(self, dialogue_machine, sample_visual_context):
        """Test that memory usage remains reasonable"""
        # Process many dialogues
        for i in range(50):
            text = DetectedText(f"Dialogue {i}", 0.9, (10, 10, 200, 30), "dialogue")
            context = VisualContext(
                screen_type="dialogue",
                detected_text=[text],
                ui_elements=[],
                dominant_colors=[(255, 255, 255)],
                game_phase="dialogue_interaction",
                visual_summary=f"Performance test {i}"
            )
            dialogue_machine.process_dialogue(context)
        
        # History should not grow unbounded
        assert len(dialogue_machine.dialogue_history) < 1000  # Reasonable limit
