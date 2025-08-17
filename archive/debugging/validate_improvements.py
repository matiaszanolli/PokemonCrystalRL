#!/usr/bin/env python3
"""
Quick validation script to test the improvements to game state detection
"""

import os
import sys
import time

# Add paths for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from trainer.trainer import UnifiedPokemonTrainer
from trainer.config import TrainingConfig, TrainingMode

def validate_improvements():
    """Test the improved game state detection and intro skip logic"""
    print("🧪 Validating Game State Detection Improvements")
    print("=" * 55)
    
    config = TrainingConfig(
        mode=TrainingMode.FAST_MONITORED,
        rom_path="../roms/pokemon_crystal.gbc",
        max_actions=150,
        frames_per_action=6,  # Slightly slower for better observation
        headless=True,
        debug_mode=True,
        log_level="INFO"
    )
    
    trainer = UnifiedPokemonTrainer(config)
    
    print("🎮 Running validation...")
    
    state_history = []
    intro_detections = 0
    dialogue_detections = 0
    title_screen_detections = 0
    
    try:
        for step in range(150):
            # Get current screenshot and detect state
            screenshot = trainer.pyboy.screen.ndarray
            current_state = trainer.game_state_detector.detect_game_state(screenshot)
            
            # Track states
            state_history.append(current_state)
            
            if current_state == "intro_sequence":
                intro_detections += 1
            elif current_state == "dialogue":
                dialogue_detections += 1
            elif current_state == "title_screen":
                title_screen_detections += 1
            
            # Take a simple action to progress
            if current_state == "intro_sequence":
                trainer.strategy_manager.execute_action(7)  # START to skip
                if step % 20 == 0:
                    print(f"   Step {step:3d}: {current_state:15s} → Skipping intro")
            elif current_state == "title_screen":
                trainer.strategy_manager.execute_action(7)  # START to enter
                if step % 20 == 0:
                    print(f"   Step {step:3d}: {current_state:15s} → Entering game")
            elif current_state == "dialogue":
                trainer.strategy_manager.execute_action(5)  # A to advance
                if step % 5 == 0:
                    print(f"   Step {step:3d}: {current_state:15s} → Advancing dialogue")
            else:
                trainer.strategy_manager.execute_action(5)  # Default A button
                if step % 20 == 0:
                    print(f"   Step {step:3d}: {current_state:15s} → Default action")
            
            time.sleep(0.05)  # Small delay for stability
        
        # Analysis
        print(f"\n📊 VALIDATION RESULTS:")
        print(f"   Total steps: 150")
        print(f"   Intro sequence detections: {intro_detections}")
        print(f"   Dialogue detections: {dialogue_detections}")  
        print(f"   Title screen detections: {title_screen_detections}")
        
        # Count unique states
        unique_states = list(set(state_history))
        print(f"   Unique states detected: {len(unique_states)}")
        print(f"   States: {unique_states}")
        
        # Check for successful progression
        has_dialogue = dialogue_detections > 0
        has_intro = intro_detections > 0
        has_variety = len(unique_states) >= 3
        
        print(f"\n✅ SUCCESS METRICS:")
        print(f"   🎬 Intro detection: {'✅' if has_intro else '❌'}")
        print(f"   💬 Dialogue detection: {'✅' if has_dialogue else '❌'}")
        print(f"   🎯 State variety: {'✅' if has_variety else '❌'}")
        
        success_score = sum([has_intro, has_dialogue, has_variety])
        print(f"   🏆 Overall score: {success_score}/3")
        
        if success_score >= 2:
            print(f"\n🎉 IMPROVEMENTS VALIDATED SUCCESSFULLY!")
            print(f"   The enhanced game state detection is working correctly.")
        else:
            print(f"\n⚠️ VALIDATION INCOMPLETE")
            print(f"   Some aspects may need additional tuning.")
            
        return {
            'success': success_score >= 2,
            'intro_detections': intro_detections,
            'dialogue_detections': dialogue_detections,
            'unique_states': len(unique_states),
            'state_history': state_history[-10:],  # Last 10 states
        }
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return {'success': False, 'error': str(e)}
    
    finally:
        print(f"\n🧹 Cleaning up...")
        trainer._finalize_training()

if __name__ == "__main__":
    try:
        result = validate_improvements()
        if result['success']:
            print(f"\n✅ Validation completed successfully!")
        else:
            print(f"\n❌ Validation encountered issues")
    except KeyboardInterrupt:
        print(f"\n⏸️ Validation interrupted by user")
    except Exception as e:
        print(f"\n💥 Validation crashed: {e}")
