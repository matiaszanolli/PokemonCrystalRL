#!/usr/bin/env python3
"""
Comprehensive Test for Dialogue Detection Improvements

This script specifically tests whether the improved dialogue detection logic
correctly distinguishes between actual dialogue and title screen false positives.
"""

import os
import sys
import time
import numpy as np
from collections import Counter
from typing import Dict, List, Tuple

# Add paths for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from trainer.trainer import UnifiedPokemonTrainer
from trainer.config import TrainingConfig, TrainingMode
from trainer.game_state import GameStateDetector

def create_test_screenshots() -> Dict[str, np.ndarray]:
    """Create synthetic test screenshots for different game states"""
    screenshots = {}
    
    # Title screen with "NEW GAME / OPTIONS" (should NOT trigger dialogue detection)
    title_screen = np.full((144, 160, 4), 106, dtype=np.uint8)  # Base brightness ~106
    # Add high variance for title screen elements
    title_screen[120:140, :] = 220  # Bottom section bright (menu area)
    # But make it uniform (low variance) - typical of menu backgrounds
    screenshots["title_screen_menu"] = title_screen
    
    # Actual dialogue box (should trigger dialogue detection)
    dialogue_screen = np.full((144, 160, 4), 80, dtype=np.uint8)  # Darker base
    dialogue_screen[115:140, 10:150] = 240  # Bright dialogue box at bottom
    # Add text-like variance inside dialogue box
    for i in range(118, 137, 2):  # Text lines
        dialogue_screen[i, 15:145, :] = np.random.randint(50, 200, (130, 4))  # Text variance
    screenshots["actual_dialogue"] = dialogue_screen
    
    # Intro sequence (bright, uniform)
    intro_screen = np.full((144, 160, 4), 250, dtype=np.uint8)
    screenshots["intro_sequence"] = intro_screen
    
    # Loading screen (very dark)
    loading_screen = np.full((144, 160, 4), 5, dtype=np.uint8)
    screenshots["loading"] = loading_screen
    
    # Overworld (high variance, complex scene)
    overworld_screen = np.random.randint(50, 150, (144, 160, 4), dtype=np.uint8)
    screenshots["overworld"] = overworld_screen
    
    return screenshots

def test_dialogue_detection_accuracy():
    """Test the accuracy of dialogue detection with synthetic data"""
    print("🧪 Testing Dialogue Detection Accuracy")
    print("=" * 50)
    
    detector = GameStateDetector(debug_mode=True)
    test_screenshots = create_test_screenshots()
    
    expected_results = {
        "title_screen_menu": "title_screen",  # Should NOT be dialogue
        "actual_dialogue": "dialogue",       # Should BE dialogue  
        "intro_sequence": "intro_sequence",
        "loading": "loading",
        "overworld": "overworld"
    }
    
    results = {}
    correct_predictions = 0
    
    print("📊 Running detection tests...")
    for test_name, screenshot in test_screenshots.items():
        detected_state = detector.detect_game_state(screenshot)
        expected_state = expected_results[test_name]
        
        is_correct = detected_state == expected_state
        if is_correct:
            correct_predictions += 1
            
        results[test_name] = {
            "detected": detected_state,
            "expected": expected_state,
            "correct": is_correct,
            "screenshot_stats": {
                "mean": float(np.mean(screenshot)),
                "variance": float(np.var(screenshot)),
                "bottom_mean": float(np.mean(screenshot[int(screenshot.shape[0] * 0.8):])),
                "bottom_variance": float(np.var(screenshot[int(screenshot.shape[0] * 0.8):]))
            }
        }
        
        status = "✅" if is_correct else "❌"
        print(f"  {status} {test_name:20s}: {detected_state:15s} (expected: {expected_state})")
    
    accuracy = correct_predictions / len(test_screenshots) * 100
    print(f"\n📈 Overall Accuracy: {accuracy:.1f}% ({correct_predictions}/{len(test_screenshots)})")
    
    # Detailed analysis for dialogue detection specifically
    dialogue_test = results["actual_dialogue"]
    title_test = results["title_screen_menu"]
    
    print(f"\n🔍 Dialogue Detection Analysis:")
    print(f"  Actual dialogue detected as: {dialogue_test['detected']} ({'✅' if dialogue_test['correct'] else '❌'})")
    print(f"  Title screen detected as: {title_test['detected']} ({'✅' if title_test['correct'] else '❌'})")
    
    if dialogue_test['correct'] and title_test['correct']:
        print(f"  🎉 Dialogue detection is working correctly!")
    else:
        print(f"  ⚠️ Dialogue detection needs adjustment")
        
    return results

def test_real_game_progression():
    """Test dialogue detection during actual game progression"""
    print("\n🎮 Testing Real Game Progression")
    print("=" * 40)
    
    config = TrainingConfig(
        mode=TrainingMode.FAST_MONITORED,
        rom_path="../roms/pokemon_crystal.gbc",
        max_actions=200,
        frames_per_action=8,  # Slower for better observation
        headless=True,
        debug_mode=True,
        log_level="INFO"
    )
    
    trainer = UnifiedPokemonTrainer(config)
    
    state_sequence = []
    dialogue_transitions = []
    false_positive_count = 0
    
    print("📊 Monitoring state transitions...")
    
    try:
        last_state = None
        consecutive_title_dialogue = 0
        
        for step in range(200):
            screenshot = trainer.pyboy.screen.ndarray
            current_state = trainer.game_state_detector.detect_game_state(screenshot)
            
            # Record state sequence
            state_sequence.append(current_state)
            
            # Check for state transitions
            if current_state != last_state:
                transition = f"{last_state} → {current_state}"
                
                # Look for suspicious transitions
                if last_state == "title_screen" and current_state == "dialogue":
                    consecutive_title_dialogue += 1
                    print(f"  ⚠️  Step {step:3d}: Suspicious transition: {transition}")
                    
                    # This might be a false positive - investigate
                    screen_stats = {
                        "mean": float(np.mean(screenshot)),
                        "variance": float(np.var(screenshot)),
                        "bottom_mean": float(np.mean(screenshot[int(screenshot.shape[0] * 0.8):])),
                        "bottom_variance": float(np.var(screenshot[int(screenshot.shape[0] * 0.8):]))
                    }
                    
                    dialogue_transitions.append({
                        "step": step,
                        "transition": transition,
                        "stats": screen_stats,
                        "suspicious": True
                    })
                    
                elif current_state == "dialogue":
                    # Non-suspicious dialogue detection
                    dialogue_transitions.append({
                        "step": step,
                        "transition": transition,
                        "stats": None,
                        "suspicious": False
                    })
                    print(f"  ✅  Step {step:3d}: Valid dialogue: {transition}")
                
                last_state = current_state
            
            # Take action to progress the game
            if current_state == "intro_sequence":
                trainer.strategy_manager.execute_action(7)  # START to skip
            elif current_state == "title_screen":
                trainer.strategy_manager.execute_action(7)  # START to enter
            elif current_state == "dialogue":
                trainer.strategy_manager.execute_action(5)  # A to advance
            else:
                trainer.strategy_manager.execute_action(5)  # Default action
            
            time.sleep(0.05)
    
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return None
    
    finally:
        trainer._finalize_training()
    
    # Analysis
    state_counts = Counter(state_sequence)
    unique_states = len(set(state_sequence))
    
    print(f"\n📈 Game Progression Analysis:")
    print(f"  Total steps: {len(state_sequence)}")
    print(f"  Unique states encountered: {unique_states}")
    print(f"  State distribution:")
    for state, count in state_counts.most_common():
        percentage = count / len(state_sequence) * 100
        print(f"    {state:15s}: {count:3d} ({percentage:5.1f}%)")
    
    print(f"\n💬 Dialogue Detection Results:")
    print(f"  Total dialogue transitions: {len(dialogue_transitions)}")
    print(f"  Suspicious title→dialogue: {consecutive_title_dialogue}")
    
    if consecutive_title_dialogue == 0:
        print(f"  🎉 No false positives detected!")
        detection_success = True
    elif consecutive_title_dialogue <= 2:
        print(f"  ⚠️  Few suspicious transitions - needs investigation")
        detection_success = False
    else:
        print(f"  ❌ Multiple false positives detected")
        detection_success = False
    
    return {
        "state_sequence": state_sequence,
        "state_counts": dict(state_counts),
        "dialogue_transitions": dialogue_transitions,
        "false_positives": consecutive_title_dialogue,
        "detection_success": detection_success,
        "unique_states": unique_states
    }

def run_comprehensive_test():
    """Run all dialogue detection tests"""
    print("🔬 COMPREHENSIVE DIALOGUE DETECTION TEST")
    print("=" * 60)
    
    # Test 1: Synthetic data accuracy
    synthetic_results = test_dialogue_detection_accuracy()
    
    # Test 2: Real game progression 
    progression_results = test_real_game_progression()
    
    if progression_results is None:
        print(f"\n❌ TESTING FAILED")
        return
    
    # Overall assessment
    print(f"\n🏆 FINAL ASSESSMENT")
    print("=" * 30)
    
    synthetic_success = all(result["correct"] for result in synthetic_results.values())
    progression_success = progression_results["detection_success"]
    
    print(f"  Synthetic data tests: {'✅ PASS' if synthetic_success else '❌ FAIL'}")
    print(f"  Real progression test: {'✅ PASS' if progression_success else '❌ FAIL'}")
    
    if synthetic_success and progression_success:
        print(f"\n🎉 DIALOGUE DETECTION IMPROVEMENTS ARE WORKING CORRECTLY!")
        print(f"   • No false positives on title screens")
        print(f"   • Proper detection of actual dialogue")
        print(f"   • Stable state transitions")
    elif synthetic_success:
        print(f"\n⚠️  MIXED RESULTS - Synthetic tests pass but real-world issues detected")
        print(f"   • Logic is correct but may need fine-tuning")
        print(f"   • Check for edge cases in real gameplay")
    else:
        print(f"\n❌ DIALOGUE DETECTION NEEDS MORE WORK")
        print(f"   • Basic logic has issues")
        print(f"   • Review variance thresholds and detection criteria")

if __name__ == "__main__":
    try:
        run_comprehensive_test()
    except KeyboardInterrupt:
        print(f"\n⏸️ Testing interrupted by user")
    except Exception as e:
        print(f"\n💥 Testing crashed: {e}")
        import traceback
        traceback.print_exc()
