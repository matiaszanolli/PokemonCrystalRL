#!/usr/bin/env python3
"""
test_integration_and_performance.py - Integration and Performance Tests

This module tests:
- Full pipeline integration from vision input to action output
- Cross-system communication and data flow
- Performance characteristics under various loads
- Edge cases and error handling
- Memory usage and resource management
- Concurrent processing scenarios
"""

import pytest
import json
import sqlite3
import time
import threading
import psutil
import os
import sys
from unittest.mock import patch, Mock, MagicMock
from datetime import datetime
from pathlib import Path

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# Import with fallbacks for missing dependencies
try:
    from pokemon_crystal_rl.core.vision_processor import DetectedText, VisualContext
    from pokemon_crystal_rl.core.semantic_context_system import SemanticContextSystem, GameContext
    from pokemon_crystal_rl.core.dialogue_state_machine import DialogueStateMachine, DialogueState, NPCType
    from pokemon_crystal_rl.core.choice_recognition_system import ChoiceRecognitionSystem, ChoiceContext, ChoiceType
except ImportError:
    # Skip these tests if dependencies aren't available
    pytest.skip("Missing dependencies for integration and performance tests", allow_module_level=True)


@pytest.mark.integration
class TestFullPipelineIntegration:
    """Test complete pipeline integration from vision to action"""
    
    def test_complete_pokemon_selection_flow(self, temp_db, temp_choice_db):
        """Test complete Pokemon selection flow through all systems"""
        # Initialize all systems
        semantic_system = SemanticContextSystem(db_path=temp_db)
        dialogue_machine = DialogueStateMachine(db_path=temp_db)
        choice_system = ChoiceRecognitionSystem(db_path=temp_choice_db)
        
        # Create comprehensive visual context for Pokemon selection
        detected_texts = [
            DetectedText("Hello! I'm Professor Elm!", 0.95, (10, 10, 200, 30), "dialogue"),
            DetectedText("So, you want to be a Pokemon trainer?", 0.92, (10, 40, 250, 60), "dialogue"),
            DetectedText("Choose your starter Pokemon!", 0.95, (10, 70, 220, 90), "dialogue"),
            DetectedText("Cyndaquil", 0.98, (20, 110, 90, 130), "choice"),
            DetectedText("Totodile", 0.97, (100, 110, 160, 130), "choice"),
            DetectedText("Chikorita", 0.96, (170, 110, 240, 130), "choice"),
        ]
        
        visual_context = VisualContext(
            screen_type="dialogue",
            detected_text=detected_texts,
            ui_elements=[],
            dominant_colors=[(255, 255, 255), (0, 0, 0)],
            game_phase="dialogue_interaction",
            visual_summary="Professor Elm offering starter Pokemon"
        )
        
        # Process through dialogue state machine
        dialogue_result = dialogue_machine.process_dialogue(visual_context)
        
        # Verify dialogue processing
        assert dialogue_result is not None
        assert dialogue_machine.current_npc_type == NPCType.PROFESSOR
        assert dialogue_machine.current_state in [DialogueState.CHOOSING, DialogueState.LISTENING]
        
        # Process through choice recognition
        choice_context = ChoiceContext(
            dialogue_text="Choose your starter Pokemon!",
            screen_type="dialogue",
            npc_type="professor",
            current_objective="get_starter_pokemon",
            conversation_history=["Hello! I'm Professor Elm!", "So, you want to be a Pokemon trainer?"],
            ui_layout="pokemon_selection"
        )
        
        choices = choice_system.recognize_choices(visual_context, choice_context)
        best_action = choice_system.get_best_choice_action(choices)
        
        # Verify choice processing
        assert len(choices) >= 3  # Should detect at least the three Pokemon
        assert isinstance(best_action, list)
        assert len(best_action) > 0
        
        # Verify semantic analysis integration
        if "semantic_analysis" in dialogue_result:
            semantic_analysis = dialogue_result["semantic_analysis"]
            assert semantic_analysis["intent"] == "starter_selection"
            assert semantic_analysis["confidence"] > 0.6
            assert "starter" in semantic_analysis["strategy"]
        
        # Verify cross-system consistency
        pokemon_choices = [c for c in choices if c.choice_type == ChoiceType.POKEMON_SELECTION]
        assert len(pokemon_choices) > 0
        
        # Fire starter should be prioritized
        top_choice = max(choices, key=lambda x: x.priority)
        if "cyndaquil" in top_choice.text.lower():
            assert best_action == ["A"]
        
        print(f"✅ Complete flow processed successfully:")
        print(f"   - Detected {len(choices)} choices")
        print(f"   - Best action: {best_action}")
        print(f"   - Current state: {dialogue_machine.current_state}")
        print(f"   - NPC type: {dialogue_machine.current_npc_type}")
    
    def test_healing_sequence_integration(self, temp_db, temp_choice_db):
        """Test Pokemon Center healing sequence integration"""
        # Initialize systems
        semantic_system = SemanticContextSystem(db_path=temp_db)
        dialogue_machine = DialogueStateMachine(db_path=temp_db)
        choice_system = ChoiceRecognitionSystem(db_path=temp_choice_db)
        
        # Create healing dialogue context
        detected_texts = [
            DetectedText("Welcome to the Pokemon Center!", 0.95, (10, 10, 220, 30), "dialogue"),
            DetectedText("Would you like me to heal your Pokemon?", 0.93, (10, 40, 280, 60), "dialogue"),
            DetectedText("Yes", 0.98, (20, 80, 50, 100), "choice"),
            DetectedText("No", 0.97, (70, 80, 90, 100), "choice"),
        ]
        
        visual_context = VisualContext(
            screen_type="dialogue",
            detected_text=detected_texts,
            ui_elements=[],
            dominant_colors=[(255, 255, 255), (255, 192, 203)],  # Pink for Pokemon Center
            game_phase="dialogue_interaction",
            visual_summary="Pokemon Center healing request"
        )
        
        # Process through pipeline
        dialogue_result = dialogue_machine.process_dialogue(visual_context)
        
        choice_context = ChoiceContext(
            dialogue_text="Would you like me to heal your Pokemon?",
            screen_type="dialogue",
            npc_type="nurse",
            current_objective="heal_pokemon",
            conversation_history=["Welcome to the Pokemon Center!"],
            ui_layout="yes_no_dialog"
        )
        
        choices = choice_system.recognize_choices(visual_context, choice_context)
        best_action = choice_system.get_best_choice_action(choices)
        
        # Verify integration results
        assert dialogue_machine.current_npc_type == NPCType.NURSE
        assert len(choices) >= 2  # Yes and No
        assert best_action == ["A"]  # Should choose to heal
        
        # Verify semantic understanding
        if "semantic_analysis" in dialogue_result:
            semantic_analysis = dialogue_result["semantic_analysis"]
            assert semantic_analysis["intent"] == "healing_request"
            assert semantic_analysis["strategy"] == "accept_healing"
    
    def test_gym_battle_integration(self, temp_db, temp_choice_db):
        """Test gym battle challenge integration"""
        # Initialize systems
        semantic_system = SemanticContextSystem(db_path=temp_db)
        dialogue_machine = DialogueStateMachine(db_path=temp_db)
        choice_system = ChoiceRecognitionSystem(db_path=temp_choice_db)
        
        # Create gym battle context
        detected_texts = [
            DetectedText("I'm Falkner, the Violet Gym Leader!", 0.95, (10, 10, 250, 30), "dialogue"),
            DetectedText("I'm the best trainer of flying Pokemon!", 0.92, (10, 40, 270, 60), "dialogue"),
            DetectedText("Are you ready for a Pokemon battle?", 0.94, (10, 70, 240, 90), "dialogue"),
            DetectedText("Bring it on!", 0.96, (20, 110, 110, 130), "choice"),
            DetectedText("Maybe later", 0.94, (130, 110, 200, 130), "choice"),
        ]
        
        visual_context = VisualContext(
            screen_type="dialogue",
            detected_text=detected_texts,
            ui_elements=[],
            dominant_colors=[(135, 206, 235), (255, 255, 255)],  # Sky blue for flying gym
            game_phase="dialogue_interaction",
            visual_summary="Gym Leader Falkner battle challenge"
        )
        
        # Process through systems
        dialogue_result = dialogue_machine.process_dialogue(visual_context)
        
        choice_context = ChoiceContext(
            dialogue_text="Are you ready for a Pokemon battle?",
            screen_type="dialogue",
            npc_type="gym_leader",
            current_objective="gym_challenge",
            conversation_history=["I'm Falkner, the Violet Gym Leader!", "I'm the best trainer of flying Pokemon!"],
            ui_layout="standard_dialogue"
        )
        
        choices = choice_system.recognize_choices(visual_context, choice_context)
        best_action = choice_system.get_best_choice_action(choices)
        
        # Verify gym challenge processing
        assert dialogue_machine.current_npc_type == NPCType.GYM_LEADER
        assert len(choices) >= 2
        
        # Should choose to battle
        top_choice = max(choices, key=lambda x: x.priority)
        assert "bring" in top_choice.text.lower() or any("yes" in c.text.lower() for c in choices)
        
        # Verify semantic understanding
        if "semantic_analysis" in dialogue_result:
            semantic_analysis = dialogue_result["semantic_analysis"]
            assert semantic_analysis["intent"] == "gym_challenge"
            assert semantic_analysis["strategy"] == "accept_challenge"
    
    def test_multi_turn_conversation_integration(self, temp_db, temp_choice_db):
        """Test multi-turn conversation handling across systems"""
        # Initialize systems
        dialogue_machine = DialogueStateMachine(db_path=temp_db)
        choice_system = ChoiceRecognitionSystem(db_path=temp_choice_db)
        
        # First turn
        turn1_texts = [
            DetectedText("Hello! I'm Professor Oak!", 0.95, (10, 10, 200, 30), "dialogue"),
            DetectedText("Are you interested in Pokemon?", 0.93, (10, 40, 220, 60), "dialogue"),
            DetectedText("Yes", 0.98, (20, 80, 50, 100), "choice"),
            DetectedText("No", 0.97, (70, 80, 90, 100), "choice"),
        ]
        
        turn1_context = VisualContext(
            screen_type="dialogue",
            detected_text=turn1_texts,
            ui_elements=[],
            dominant_colors=[(255, 255, 255)],
            game_phase="dialogue_interaction",
            visual_summary="Professor Oak introduction"
        )
        
        # Process first turn
        result1 = dialogue_machine.process_dialogue(turn1_context)
        conversation_id_1 = dialogue_machine.current_conversation_id
        
        # Second turn
        turn2_texts = [
            DetectedText("Great! Pokemon are wonderful creatures!", 0.94, (10, 10, 280, 30), "dialogue"),
            DetectedText("Would you like to learn more?", 0.92, (10, 40, 200, 60), "dialogue"),
            DetectedText("Tell me more", 0.96, (20, 80, 110, 100), "choice"),
            DetectedText("That's enough", 0.94, (130, 80, 220, 100), "choice"),
        ]
        
        turn2_context = VisualContext(
            screen_type="dialogue",
            detected_text=turn2_texts,
            ui_elements=[],
            dominant_colors=[(255, 255, 255)],
            game_phase="dialogue_interaction",
            visual_summary="Professor Oak continued conversation"
        )
        
        # Process second turn
        result2 = dialogue_machine.process_dialogue(turn2_context)
        conversation_id_2 = dialogue_machine.current_conversation_id
        
        # Verify conversation continuity
        assert conversation_id_1 == conversation_id_2
        assert len(dialogue_machine.dialogue_history) > 2
        assert dialogue_machine.current_npc_type == NPCType.PROFESSOR
        
        # Verify history influences choice prioritization
        choice_context = ChoiceContext(
            dialogue_text="Would you like to learn more?",
            screen_type="dialogue",
            npc_type="professor",
            current_objective=None,
            conversation_history=dialogue_machine.dialogue_history[-3:],
            ui_layout="standard_dialogue"
        )
        
        choices = choice_system.recognize_choices(turn2_context, choice_context)
        
        # Should prioritize learning more based on positive first response
        learning_choice = next((c for c in choices if "more" in c.text.lower()), None)
        if learning_choice:
            assert learning_choice.priority > 50  # Should be prioritized


@pytest.mark.integration
class TestCrossSystemDataFlow:
    """Test data flow and consistency across systems"""
    
    def test_game_context_propagation(self, temp_db):
        """Test that game context is properly propagated through systems"""
        semantic_system = SemanticContextSystem(db_path=temp_db)
        dialogue_machine = DialogueStateMachine(db_path=temp_db, semantic_system=semantic_system)
        
        # Mock game context building
        with patch.object(dialogue_machine, '_build_game_context') as mock_context:
            test_context = GameContext(
                current_objective="continue_journey",
                player_progress={"badges": 1, "pokemon_count": 3},
                location_info={"current_map": "violet_city", "region": "johto"},
                recent_events=["beat_falkner", "got_zephyr_badge"],
                active_quests=["go_to_azalea_town"]
            )
            mock_context.return_value = test_context
            
            # Create dialogue that should use context
            detected_texts = [
                DetectedText("Congratulations on beating Falkner!", 0.95, (10, 10, 250, 30), "dialogue"),
                DetectedText("Ready for your next challenge?", 0.93, (10, 40, 220, 60), "dialogue"),
            ]
            
            visual_context = VisualContext(
                screen_type="dialogue",
                detected_text=detected_texts,
                ui_elements=[],
                dominant_colors=[(255, 255, 255)],
                game_phase="dialogue_interaction",
                visual_summary="Post-gym victory dialogue"
            )
            
            # Process dialogue
            with patch.object(semantic_system, 'analyze_dialogue') as mock_analyze:
                mock_analyze.return_value = {
                    "intent": "story_progression",
                    "confidence": 0.8,
                    "strategy": "continue_journey",
                    "reasoning": "Player has progressed, ready for next challenge",
                    "recommended_actions": ["A"]
                }
                
                result = dialogue_machine.process_dialogue(visual_context)
                
                # Verify context was passed to semantic system
                mock_analyze.assert_called_once()
                call_args = mock_analyze.call_args[0]
                passed_context = call_args[1]
                
                assert isinstance(passed_context, GameContext)
                assert passed_context.player_progress["badges"] == 1
                assert "beat_falkner" in passed_context.recent_events
    
    def test_database_consistency(self, temp_db, temp_choice_db):
        """Test database consistency across systems"""
        # Initialize systems with shared data
        dialogue_machine = DialogueStateMachine(db_path=temp_db)
        choice_system = ChoiceRecognitionSystem(db_path=temp_choice_db)
        
        # Process some interactions
        detected_texts = [
            DetectedText("Hello trainer!", 0.95, (10, 10, 120, 30), "dialogue"),
            DetectedText("Battle me?", 0.93, (10, 40, 90, 60), "dialogue"),
            DetectedText("Yes", 0.98, (20, 80, 50, 100), "choice"),
        ]
        
        visual_context = VisualContext(
            screen_type="dialogue",
            detected_text=detected_texts,
            ui_elements=[],
            dominant_colors=[(255, 255, 255)],
            game_phase="dialogue_interaction",
            visual_summary="Trainer battle request"
        )
        
        choice_context = ChoiceContext(
            dialogue_text="Battle me?",
            screen_type="dialogue",
            npc_type="trainer",
            current_objective=None,
            conversation_history=["Hello trainer!"],
            ui_layout="yes_no_dialog"
        )
        
        # Process through both systems
        dialogue_result = dialogue_machine.process_dialogue(visual_context)
        choices = choice_system.recognize_choices(visual_context, choice_context)
        
        # Verify database entries exist
        with sqlite3.connect(temp_db) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM conversations")
            conversation_count = cursor.fetchone()[0]
            assert conversation_count > 0
        
        with sqlite3.connect(temp_choice_db) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM choice_recognitions")
            choice_count = cursor.fetchone()[0]
            assert choice_count > 0
    
    def test_state_synchronization(self, temp_db, temp_choice_db):
        """Test state synchronization across systems"""
        dialogue_machine = DialogueStateMachine(db_path=temp_db)
        choice_system = ChoiceRecognitionSystem(db_path=temp_choice_db)
        
        # Process dialogue that should change state
        detected_texts = [
            DetectedText("Choose your Pokemon!", 0.95, (10, 10, 180, 30), "dialogue"),
            DetectedText("Pikachu", 0.98, (20, 50, 80, 70), "choice"),
            DetectedText("Charmander", 0.97, (100, 50, 180, 70), "choice"),
        ]
        
        visual_context = VisualContext(
            screen_type="dialogue",
            detected_text=detected_texts,
            ui_elements=[],
            dominant_colors=[(255, 255, 255)],
            game_phase="dialogue_interaction",
            visual_summary="Pokemon choice"
        )
        
        # Process and verify state changes
        dialogue_machine.process_dialogue(visual_context)
        
        assert dialogue_machine.current_state in [DialogueState.CHOOSING, DialogueState.RESPONDING]
        assert dialogue_machine.current_conversation_id is not None
        
        # Choice system should work with current dialogue state
        choice_context = ChoiceContext(
            dialogue_text="Choose your Pokemon!",
            screen_type="dialogue",
            npc_type="generic",
            current_objective=None,
            conversation_history=[],
            ui_layout="pokemon_selection"
        )
        
        choices = choice_system.recognize_choices(visual_context, choice_context)
        assert len(choices) >= 2


@pytest.mark.performance
class TestPerformanceCharacteristics:
    """Test performance under various conditions"""
    
    def test_processing_speed_benchmarks(self, temp_db, temp_choice_db):
        """Test processing speed benchmarks"""
        # Initialize systems
        semantic_system = SemanticContextSystem(db_path=temp_db)
        dialogue_machine = DialogueStateMachine(db_path=temp_db)
        choice_system = ChoiceRecognitionSystem(db_path=temp_choice_db)
        
        # Create test data
        detected_texts = [
            DetectedText("Test dialogue", 0.9, (10, 10, 100, 30), "dialogue"),
            DetectedText("Choice 1", 0.95, (20, 50, 80, 70), "choice"),
            DetectedText("Choice 2", 0.93, (100, 50, 160, 70), "choice"),
        ]
        
        visual_context = VisualContext(
            screen_type="dialogue",
            detected_text=detected_texts,
            ui_elements=[],
            dominant_colors=[(255, 255, 255)],
            game_phase="dialogue_interaction",
            visual_summary="Performance test"
        )
        
        # Test dialogue processing speed
        iterations = 100
        start_time = time.time()
        
        for _ in range(iterations):
            dialogue_machine.process_dialogue(visual_context)
        
        dialogue_time = time.time() - start_time
        dialogue_avg = dialogue_time / iterations
        
        # Test choice recognition speed
        choice_context = ChoiceContext(
            dialogue_text="Test dialogue",
            screen_type="dialogue",
            npc_type="generic",
            current_objective=None,
            conversation_history=[],
            ui_layout="standard_dialogue"
        )
        
        start_time = time.time()
        
        for _ in range(iterations):
            choice_system.recognize_choices(visual_context, choice_context)
        
        choice_time = time.time() - start_time
        choice_avg = choice_time / iterations
        
        # Test semantic analysis speed
        game_context = GameContext(
            current_objective="test_processing",
            player_progress={"badges": 0},
            location_info={"current_map": "test"},
            recent_events=[],
            active_quests=[]
        )
        
        start_time = time.time()
        
        for _ in range(iterations):
            semantic_system.analyze_dialogue("Test dialogue", game_context)
        
        semantic_time = time.time() - start_time
        semantic_avg = semantic_time / iterations
        
        # Assert performance requirements
        assert dialogue_avg < 0.01, f"Dialogue processing too slow: {dialogue_avg:.4f}s"
        assert choice_avg < 0.01, f"Choice recognition too slow: {choice_avg:.4f}s"
        assert semantic_avg < 0.005, f"Semantic analysis too slow: {semantic_avg:.4f}s"
        
        print(f"📊 Performance Benchmarks:")
        print(f"   - Dialogue processing: {dialogue_avg:.4f}s avg")
        print(f"   - Choice recognition: {choice_avg:.4f}s avg")
        print(f"   - Semantic analysis: {semantic_avg:.4f}s avg")
    
    def test_memory_usage_patterns(self, temp_db):
        """Test memory usage patterns during extended use"""
        semantic_system = SemanticContextSystem(db_path=temp_db)
        dialogue_machine = DialogueStateMachine(db_path=temp_db)
        
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Process many dialogues
        for i in range(200):
            detected_texts = [
                DetectedText(f"Dialogue {i}", 0.9, (10, 10, 100, 30), "dialogue"),
                DetectedText(f"Response {i}", 0.9, (20, 50, 120, 70), "choice"),
            ]
            
            visual_context = VisualContext(
                screen_type="dialogue",
                detected_text=detected_texts,
                ui_elements=[],
                dominant_colors=[(255, 255, 255)],
                game_phase="dialogue_interaction",
                visual_summary=f"Memory test {i}"
            )
            
            dialogue_machine.process_dialogue(visual_context)
            
            # Reset periodically to simulate normal usage
            if i % 50 == 0:
                dialogue_machine.reset()
        
        # Check final memory usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_growth = final_memory - initial_memory
        
        # Memory growth should be reasonable (less than 100MB for this test)
        assert memory_growth < 100, f"Excessive memory growth: {memory_growth:.1f}MB"
        
        print(f"🧠 Memory Usage:")
        print(f"   - Initial: {initial_memory:.1f}MB")
        print(f"   - Final: {final_memory:.1f}MB")
        print(f"   - Growth: {memory_growth:.1f}MB")
    
    def test_concurrent_processing(self, temp_db, temp_choice_db):
        """Test concurrent processing capabilities"""
        dialogue_machine = DialogueStateMachine(db_path=temp_db)
        choice_system = ChoiceRecognitionSystem(db_path=temp_choice_db)
        
        results = []
        errors = []
        
        def process_dialogue(thread_id):
            """Process dialogue in separate thread"""
            try:
                for i in range(20):
                    detected_texts = [
                        DetectedText(f"Thread {thread_id} dialogue {i}", 0.9, (10, 10, 200, 30), "dialogue"),
                        DetectedText("Choice", 0.9, (20, 50, 80, 70), "choice"),
                    ]
                    
                    visual_context = VisualContext(
                        screen_type="dialogue",
                        detected_text=detected_texts,
                        ui_elements=[],
                        dominant_colors=[(255, 255, 255)],
                        game_phase="dialogue_interaction",
                        visual_summary=f"Concurrent test {thread_id}-{i}"
                    )
                    
                    result = dialogue_machine.process_dialogue(visual_context)
                    results.append((thread_id, i, result))
                    time.sleep(0.001)  # Small delay to simulate realistic timing
            except Exception as e:
                errors.append((thread_id, str(e)))
        
        # Start multiple threads
        threads = []
        num_threads = 5
        
        start_time = time.time()
        
        for thread_id in range(num_threads):
            thread = threading.Thread(target=process_dialogue, args=(thread_id,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        end_time = time.time()
        
        # Verify results
        assert len(errors) == 0, f"Errors in concurrent processing: {errors}"
        assert len(results) == num_threads * 20, f"Missing results: got {len(results)}, expected {num_threads * 20}"
        
        total_time = end_time - start_time
        assert total_time < 10, f"Concurrent processing too slow: {total_time:.2f}s"
        
        print(f"🔄 Concurrent Processing:")
        print(f"   - Threads: {num_threads}")
        print(f"   - Operations per thread: 20")
        print(f"   - Total time: {total_time:.2f}s")
        print(f"   - Errors: {len(errors)}")
    
    def test_large_dataset_handling(self, temp_db, temp_choice_db):
        """Test handling of large datasets"""
        choice_system = ChoiceRecognitionSystem(db_path=temp_choice_db)
        
        # Create large number of detected texts
        many_texts = []
        for i in range(100):
            text = DetectedText(f"Choice {i:03d}", 0.8 + (i % 20) / 100, (20, 50 + i * 15, 120, 70 + i * 15), "choice")
            many_texts.append(text)
        
        visual_context = VisualContext(
            screen_type="dialogue",
            detected_text=many_texts,
            ui_elements=[],
            dominant_colors=[(255, 255, 255)],
            game_phase="dialogue_interaction",
            visual_summary="Large dataset test"
        )
        
        choice_context = ChoiceContext(
            dialogue_text="Choose from many options",
            screen_type="dialogue",
            npc_type="generic",
            current_objective=None,
            conversation_history=[],
            ui_layout="standard_dialogue"
        )
        
        # Test processing time with large dataset
        start_time = time.time()
        choices = choice_system.recognize_choices(visual_context, choice_context)
        processing_time = time.time() - start_time
        
        # Should handle large dataset reasonably fast
        assert processing_time < 2.0, f"Large dataset processing too slow: {processing_time:.2f}s"
        assert isinstance(choices, list)
        
        # Should still produce reasonable results
        if choices:
            assert all(isinstance(c.confidence, float) for c in choices)
            assert all(0.0 <= c.confidence <= 1.0 for c in choices)
        
        print(f"📊 Large Dataset Handling:")
        print(f"   - Input texts: {len(many_texts)}")
        print(f"   - Processed choices: {len(choices)}")
        print(f"   - Processing time: {processing_time:.3f}s")


@pytest.mark.integration 
class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling across systems"""
    
    def test_corrupted_input_handling(self, temp_db, temp_choice_db):
        """Test handling of corrupted or malformed inputs"""
        dialogue_machine = DialogueStateMachine(db_path=temp_db)
        choice_system = ChoiceRecognitionSystem(db_path=temp_choice_db)
        
        # Test with corrupted detected texts
        corrupted_texts = [
            DetectedText("", 0.0, (0, 0, 0, 0), "unknown"),
            DetectedText(None, 0.5, (10, 10, 50, 30), "dialogue"),
            DetectedText("Test", -0.5, (-10, -10, 0, 0), "choice"),  # Invalid confidence and coords
            DetectedText("A" * 1000, 1.5, (10, 10, 50, 30), "dialogue"),  # Very long text, invalid confidence
        ]
        
        visual_context = VisualContext(
            screen_type="dialogue",
            detected_text=corrupted_texts,
            ui_elements=[],
            dominant_colors=[],
            game_phase="dialogue_interaction",
            visual_summary="Corrupted input test"
        )
        
        # Systems should handle corrupted input gracefully
        dialogue_result = dialogue_machine.process_dialogue(visual_context)
        
        choice_context = ChoiceContext(
            dialogue_text="",
            screen_type="",
            npc_type="",
            current_objective=None,
            conversation_history=[],
            ui_layout="standard_dialogue"
        )
        
        choices = choice_system.recognize_choices(visual_context, choice_context)
        
        # Should not crash and return reasonable defaults
        assert dialogue_result is None or isinstance(dialogue_result, dict)
        assert isinstance(choices, list)
    
    def test_database_corruption_recovery(self, temp_db):
        """Test recovery from database corruption"""
        dialogue_machine = DialogueStateMachine(db_path=temp_db)
        
        # Corrupt the database file
        with open(temp_db, 'w') as f:
            f.write("CORRUPTED DATABASE CONTENT")
        
        # System should handle database corruption gracefully
        try:
            stats = dialogue_machine.get_statistics()
            assert isinstance(stats, dict)
        except Exception as e:
            # Should be a handled database exception
            assert any(word in str(e).lower() for word in ["database", "corruption", "connection", "sqlite"])
    
    def test_extreme_input_values(self, temp_db, temp_choice_db):
        """Test handling of extreme input values"""
        semantic_system = SemanticContextSystem(db_path=temp_db)
        
        # Test with extreme game context
        extreme_context = GameContext(
            current_objective="test_extreme_values",
            player_progress={str(i): i for i in range(10000)},  # Very large dict
            location_info={"map": "A" * 10000},  # Very long strings
            recent_events=["event"] * 1000,  # Very long list
            active_quests=[]
        )
        
        # Test with very long dialogue
        long_dialogue = "This is a very long dialogue. " * 1000
        
        # Should handle extreme values without crashing
        result = semantic_system.analyze_dialogue(long_dialogue, extreme_context)
        
        assert isinstance(result, dict)
        assert "intent" in result
        assert "confidence" in result
        assert 0.0 <= result["confidence"] <= 1.0
    
    def test_resource_cleanup(self, temp_db, temp_choice_db):
        """Test proper resource cleanup"""
        # Create and process with systems
        dialogue_machine = DialogueStateMachine(db_path=temp_db)
        choice_system = ChoiceRecognitionSystem(db_path=temp_choice_db)
        
        detected_texts = [DetectedText("Test", 0.9, (10, 10, 50, 30), "dialogue")]
        visual_context = VisualContext("dialogue", detected_texts, [], [(255, 255, 255)], "test", "Test")
        
        # Process some data
        dialogue_machine.process_dialogue(visual_context)
        
        choice_context = ChoiceContext("Test", "dialogue", "generic", None, [], "standard_dialogue")
        choice_system.recognize_choices(visual_context, choice_context)
        
        # Verify database connections are properly managed
        # (This is implicit - if connections aren't closed properly, we'd get errors)
        
        # Reset systems
        dialogue_machine.reset()
        
        # Should still be functional after reset
        result = dialogue_machine.process_dialogue(visual_context)
        assert result is None or isinstance(result, dict)
    
    def test_unicode_and_special_characters(self, temp_db):
        """Test handling of Unicode and special characters"""
        semantic_system = SemanticContextSystem(db_path=temp_db)
        
        # Test with various special characters
        test_dialogues = [
            "Pokémon Center! ñoble créature!",  # Accented characters
            "Hello! 😊 Welcome to the world of Pokémon! 🐉",  # Emoji
            "Test with symbols: @#$%^&*()_+-={}[]|\\:;\"'<>?,./",  # Special symbols
            "Test with numbers: 1234567890",  # Numbers
            "Mixed: Pokémon #001 is 'Bulbasaur'! (草 type)",  # Mixed Unicode
            "",  # Empty string
            " ",  # Whitespace only
        ]
        
        game_context = GameContext(
            current_objective="test_unicode",
            player_progress={},
            location_info={},
            recent_events=[],
            active_quests=[]
        )
        
        for dialogue in test_dialogues:
            result = semantic_system.analyze_dialogue(dialogue, game_context)
            
            # Should handle all character types gracefully
            assert isinstance(result, dict)
            assert "intent" in result
            assert "confidence" in result
            assert 0.0 <= result["confidence"] <= 1.0


@pytest.mark.integration
class TestSystemInteroperability:
    """Test interoperability between different system versions"""
    
    def test_backwards_compatibility(self, temp_db):
        """Test backwards compatibility with older data formats"""
        # This would test compatibility with older database schemas
        # For now, we'll test basic schema flexibility
        
        dialogue_machine = DialogueStateMachine(db_path=temp_db)
        
        # Manually insert data in "old" format (simulated)
        with sqlite3.connect(temp_db) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO dialogue_sessions (session_start, npc_type, location_map, total_exchanges, choices_made)
                VALUES ('2023-01-01T00:00:00', 'professor', 0, 1, 0)
            """)
            conn.commit()
        
        # System should handle existing data gracefully
        stats = dialogue_machine.get_statistics()
        assert stats["total_conversations"] >= 1
    
    def test_cross_platform_compatibility(self, temp_db):
        """Test cross-platform file path and database handling"""
        dialogue_machine = DialogueStateMachine(db_path=temp_db)
        
        # Test path handling
        assert dialogue_machine.db_path.exists()
        assert dialogue_machine.db_path.is_file()
        
        # Test database operations work across platforms
        detected_texts = [DetectedText("Cross-platform test", 0.9, (10, 10, 150, 30), "dialogue")]
        visual_context = VisualContext("dialogue", detected_texts, [], [(255, 255, 255)], "test", "Cross-platform")
        
        result = dialogue_machine.process_dialogue(visual_context)
        assert result is None or isinstance(result, dict)


@pytest.mark.performance
class TestScalabilityLimits:
    """Test system scalability limits"""
    
    def test_maximum_concurrent_users(self, temp_db, temp_choice_db):
        """Test maximum concurrent users (simulated)"""
        # This would test how many concurrent "users" the system can handle
        # We'll simulate this with rapid-fire requests
        
        dialogue_machine = DialogueStateMachine(db_path=temp_db)
        choice_system = ChoiceRecognitionSystem(db_path=temp_choice_db)
        
        num_simulated_users = 50
        requests_per_user = 10
        
        start_time = time.time()
        
        for user_id in range(num_simulated_users):
            for request_id in range(requests_per_user):
                detected_texts = [
                    DetectedText(f"User {user_id} request {request_id}", 0.9, (10, 10, 200, 30), "dialogue")
                ]
                visual_context = VisualContext("dialogue", detected_texts, [], [(255, 255, 255)], "test", f"User {user_id}")
                
                dialogue_machine.process_dialogue(visual_context)
        
        end_time = time.time()
        total_time = end_time - start_time
        total_requests = num_simulated_users * requests_per_user
        requests_per_second = total_requests / total_time
        
        # Should handle reasonable load
        assert requests_per_second > 100, f"Too slow: {requests_per_second:.1f} req/s"
        
        print(f"🚀 Scalability Test:")
        print(f"   - Total requests: {total_requests}")
        print(f"   - Total time: {total_time:.2f}s")
        print(f"   - Requests per second: {requests_per_second:.1f}")
    
    def test_database_size_limits(self, temp_db):
        """Test behavior with large databases"""
        dialogue_machine = DialogueStateMachine(db_path=temp_db)
        
        # Insert a large amount of test data
        with sqlite3.connect(temp_db) as conn:
            cursor = conn.cursor()
            
            # Insert many conversations (without specifying ID since it's auto-increment)
            for i in range(1000):
                cursor.execute("""
                    INSERT INTO dialogue_sessions (session_start, npc_type, location_map, total_exchanges, choices_made)
                    VALUES ('2023-01-01T00:00:00', 'generic', 1, 3, 1)
                """)
            
            conn.commit()
        
        # System should still function with large database
        stats = dialogue_machine.get_statistics()
        assert stats["total_conversations"] >= 1000
        
        # Performance should still be reasonable
        start_time = time.time()
        for _ in range(10):
            dialogue_machine.get_statistics()
        end_time = time.time()
        
        avg_query_time = (end_time - start_time) / 10
        assert avg_query_time < 0.1, f"Database queries too slow: {avg_query_time:.3f}s"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
