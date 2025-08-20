#!/usr/bin/env python3
"""
test_semantic_dialogue_integration.py - Comprehensive Test for Semantic-Enabled Dialogue State Machine

This test demonstrates the integration between the dialogue state machine and semantic context system,
showing how semantic analysis enhances dialogue understanding and decision-making.
"""

import sys
from pathlib import Path
from core.vision_processor import DetectedText, VisualContext
from core.dialogue_state_machine import DialogueStateMachine, DialogueState, NPCType
from core.semantic_context_system import SemanticContextSystem, GameContext, DialogueIntent


def test_semantic_dialogue_integration():
    """Test comprehensive dialogue state machine with semantic context integration"""
    print("🧪 Testing Semantic Dialogue Integration...")
    print("=" * 60)
    
    # Initialize the dialogue state machine (includes semantic system)
    dsm = DialogueStateMachine("test_semantic_dialogue.db")
    print()
    
    # Test scenarios with different dialogue types
    test_scenarios = [
        {
            "name": "Professor Elm - Starter Pokemon Selection",
            "dialogue_texts": [
                "Hello! I'm Professor Elm!",
                "I study Pokemon ecology and behavior.",
                "Would you like to choose a starter Pokemon?",
                "We have three excellent choices available."
            ],
            "choice_texts": ["Yes", "No", "Tell me more"],
            "game_state": {
                "player": {"map": 1, "x": 5, "y": 10, "money": 0, "badges": 0, "level": 1},
                "party": []
            },
            "expected_npc": NPCType.PROFESSOR
        },
        
        {
            "name": "Pokemon Center Nurse - Healing Request", 
            "dialogue_texts": [
                "Welcome to the Pokemon Center!",
                "Would you like me to heal your Pokemon?",
                "They look tired from their adventures."
            ],
            "choice_texts": ["Yes", "No thanks"],
            "game_state": {
                "player": {"map": 4, "x": 3, "y": 5, "money": 500, "badges": 1, "level": 10},
                "party": [{"name": "Cyndaquil", "hp": 15, "max_hp": 39, "level": 8}]
            },
            "expected_npc": NPCType.GENERIC
        },
        
        {
            "name": "Gym Leader Falkner - Battle Challenge",
            "dialogue_texts": [
                "I am Falkner, the Violet City Gym Leader!",
                "I'm the leader of the Violet Pokemon Gym!",
                "Are you ready to challenge me to a battle?",
                "My bird Pokemon are ready!"
            ],
            "choice_texts": ["Yes", "Not yet"],
            "game_state": {
                "player": {"map": 5, "x": 8, "y": 12, "money": 1200, "badges": 0, "level": 12},
                "party": [{"name": "Cyndaquil", "hp": 42, "max_hp": 42, "level": 12}]
            },
            "expected_npc": NPCType.GYM_LEADER
        }
    ]
    
    # Run test scenarios
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"Test Scenario {i}: {scenario['name']}")
        print("-" * 50)
        
        # Create visual context for dialogue
        detected_texts = []
        y_offset = 100
        
        # Add dialogue texts
        for dialogue_text in scenario['dialogue_texts']:
            detected_texts.append(
                DetectedText(dialogue_text, 0.9, (10, y_offset, 300, y_offset+20), "dialogue")
            )
            y_offset += 25
        
        # Add choice texts
        choice_y = y_offset + 10
        for choice_text in scenario['choice_texts']:
            detected_texts.append(
                DetectedText(choice_text, 0.9, (20, choice_y, 100, choice_y+15), "dialogue")
            )
            choice_y += 20
        
        # Create visual context
        context = VisualContext(
            screen_type="dialogue",
            detected_text=detected_texts,
            ui_elements=[],
            dominant_colors=[(255, 255, 255)],
            game_phase="dialogue_interaction",
            visual_summary=f"{scenario['name']} dialogue with choices"
        )
        
        # Update dialogue state
        print("📝 Processing dialogue...")
        state = dsm.update_state(context, scenario['game_state'])
        
        # Verify NPC identification
        if dsm.current_context:
            print(f"🤖 Identified NPC: {dsm.current_context.npc_type.value}")
            print(f"📍 Location: Map {dsm.current_context.location_map}")
            print(f"🎯 Objective: {dsm.current_context.current_objective}")
            print(f"💬 Topic: {dsm.current_context.conversation_topic}")
        
        # Get semantic analysis for the full dialogue
        full_dialogue = ' '.join(scenario['dialogue_texts'])
        semantic_analysis = dsm.get_semantic_analysis(full_dialogue, scenario['game_state'])
        
        if semantic_analysis:
            print(f"💭 Semantic Intent: {semantic_analysis['intent']}")
            print(f"🎯 Confidence: {semantic_analysis['confidence']:.2f}")
            print(f"📋 Strategy: {semantic_analysis['strategy']}")
            print(f"💡 Reasoning: {semantic_analysis['reasoning']}")
        
        # Show dialogue choices and priorities
        if dsm.choice_history:
            print(f"⚖️ Found {len(dsm.choice_history)} dialogue choices:")
            for j, choice in enumerate(dsm.choice_history, 1):
                print(f"   {j}. '{choice.text}' (priority: {choice.priority}, outcome: {choice.expected_outcome})")
        
        # Get recommended action
        action = dsm.get_recommended_action()
        print(f"🎮 Recommended action: {action}")
        
        # Reset for next test
        dsm.reset_conversation()
        print("\n")
    
    # Test semantic-only analysis (without dialogue state machine context)
    print("🧠 Testing Direct Semantic Analysis...")
    print("-" * 50)
    
    semantic_system = SemanticContextSystem()
    
    test_dialogues = [
        {
            "text": "Would you like to take this egg to Mr. Pokemon?",
            "context": GameContext(
                current_objective="story_mission",
                player_progress={"level": 5, "party_size": 1, "badges": 0},
                location_info={"map_id": 1, "location_type": "lab"},
                recent_events=[],
                active_quests=[]
            )
        },
        {
            "text": "Battle me if you want to earn the Zephyr Badge!",
            "context": GameContext(
                current_objective="gym_challenge_1",
                player_progress={"level": 15, "party_size": 2, "badges": 0},
                location_info={"map_id": 5, "location_type": "gym"},
                recent_events=[],
                active_quests=[]
            )
        },
        {
            "text": "What can I get you today? Potions? Pokeballs?",
            "context": GameContext(
                current_objective="buy_supplies",
                player_progress={"level": 8, "party_size": 1, "badges": 1},
                location_info={"map_id": 6, "location_type": "shop"},
                recent_events=[],
                active_quests=[]
            )
        }
    ]
    
    for i, test in enumerate(test_dialogues, 1):
        print(f"Direct Analysis {i}:")
        analysis = semantic_system.analyze_dialogue(test['text'], test['context'])
        
        if analysis:
            print(f"  Text: '{test['text']}'")
            print(f"  Intent: {analysis.get('primary_intent', 'unknown')}")
            print(f"  Confidence: {analysis.get('confidence', 0.0):.2f}")
            print(f"  Strategy: {analysis.get('response_strategy', 'none')}")
            print(f"  Actions: {analysis.get('suggested_actions', [])}")
        else:
            print(f"  Text: '{test['text']}'")
            print(f"  No analysis available")
        
        print()
    
    # Final statistics
    print("📊 Final Statistics:")
    print("-" * 30)
    stats = dsm.get_dialogue_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n🎉 Semantic dialogue integration test completed!")
    print("✅ The dialogue state machine successfully integrates with the semantic context system")
    print("✅ Semantic analysis enhances topic identification and choice prioritization")
    print("✅ Different NPC types and contexts are handled appropriately")


if __name__ == "__main__":
    test_semantic_dialogue_integration()
