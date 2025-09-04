#!/usr/bin/env python3
"""
Test script to verify the dialogue state machine database fix
"""

import sys
import os
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent))

from core.dialogue_state_machine import DialogueStateMachine
from vision.vision_processor import VisualContext, DetectedText

def test_dialogue_state_machine():
    """Test the dialogue state machine with the database fix"""
    
    # Create a test database path
    test_db_path = "test_dialogue_fix.db"
    
    # Clean up any existing test database
    if os.path.exists(test_db_path):
        os.remove(test_db_path)
    
    try:
        # Initialize the dialogue state machine
        print("Initializing DialogueStateMachine...")
        dsm = DialogueStateMachine(db_path=test_db_path)
        print("✓ DialogueStateMachine initialized successfully")
        
        # Create test visual context
        detected_texts = [
            DetectedText(
                text="Hello! I'm Professor Elm!",
                confidence=0.9,
                bbox=(10, 10, 200, 50),
                location="dialogue"
            )
        ]
        
        visual_context = VisualContext(
            screen_type='dialogue',
            detected_text=detected_texts,
            ui_elements=[],
            dominant_colors=[(255, 255, 255), (0, 0, 0)],
            game_phase='dialogue_interaction',
            visual_summary='Professor Elm - Starter Pokemon Selection dialogue with choices'
        )
        
        # Test game state
        game_state = {
            'party': [],
            'player': {'badges': 0, 'level': 1, 'map': 1, 'money': 0},
            'location': 1,
            'objective': 'Get starter Pokemon'
        }
        
        # Test the process_dialogue method
        print("Testing process_dialogue method...")
        result = dsm.process_dialogue(visual_context, game_state)
        print("✓ process_dialogue completed successfully")
        
        if result:
            print(f"✓ Dialogue processed: {result['dialogue']}")
            print(f"✓ NPC type identified: {result['npc_type']}")
            print(f"✓ Session ID: {result['session_id']}")
        
        # Test the update_state method
        print("Testing update_state method...")
        state_result = dsm.update_state(visual_context, game_state)
        print("✓ update_state completed successfully")
        
        print("\n🎉 All tests passed! The database fix is working correctly.")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Clean up test database
        if os.path.exists(test_db_path):
            os.remove(test_db_path)
    
    return True

if __name__ == "__main__":
    success = test_dialogue_state_machine()
    sys.exit(0 if success else 1)