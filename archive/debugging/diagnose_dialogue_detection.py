#!/usr/bin/env python3
"""
Dialogue Detection Diagnostic Tool

This tool captures and analyzes the specific screens that cause false positive
dialogue detection to understand exactly what's happening.
"""

import os
import sys
import time
import numpy as np
from PIL import Image
from typing import Dict, List, Tuple, Optional

# Add paths for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from trainer.trainer import UnifiedPokemonTrainer
from trainer.config import TrainingConfig, TrainingMode
from trainer.game_state import GameStateDetector

def analyze_screen_thoroughly(screenshot: np.ndarray, detector: GameStateDetector) -> Dict:
    """Thoroughly analyze a screenshot and the detection logic"""
    if screenshot is None or screenshot.size == 0:
        return {"error": "Invalid screenshot"}
    
    # Basic stats
    mean_brightness = np.mean(screenshot)
    color_variance = np.var(screenshot)
    
    # Region analysis matching the detection logic
    height, width = screenshot.shape[:2]
    bottom_section = screenshot[int(height * 0.8)::2, ::2]  # Same as in detector
    
    bottom_stats = {}
    if bottom_section.size > 0:
        bottom_stats = {
            "mean": float(np.mean(bottom_section)),
            "variance": float(np.var(bottom_section)),
            "max": float(np.max(bottom_section)),
            "min": float(np.min(bottom_section)),
            "shape": bottom_section.shape,
            "bright_pixel_ratio": float(np.sum(bottom_section > 200) / bottom_section.size)
        }
    
    # Use moderate sampling like the detector
    sample_screenshot = screenshot[::4, ::4]
    sample_mean = np.mean(sample_screenshot)
    sample_variance = np.var(sample_screenshot)
    
    # Check each condition in the detection logic
    conditions = {
        "loading_check": mean_brightness < 10,
        "intro_check": mean_brightness > 240,
        "dialogue_bottom_bright": bottom_stats.get("mean", 0) > 200,
        "dialogue_bottom_variance": bottom_stats.get("variance", 0) > 100,
        "title_screen_bright_lowvar": sample_mean >= 200 and sample_variance < 100,
        "title_screen_high_variance": color_variance > 2000,
        "battle_brightness_180": abs(mean_brightness - 180.0) < 1.0,
        "battle_brightness_100": 95 <= mean_brightness <= 105,
        "menu_brightness_range": 120 <= mean_brightness <= 180,
        "overworld_high_variance": color_variance > 800 and 50 < mean_brightness < 200,
        "overworld_medium_variance": color_variance > 400 and 60 < mean_brightness < 180,
        "overworld_low_variance": color_variance > 200 and 40 < mean_brightness < 220
    }
    
    # Run actual detection
    detected_state = detector.detect_game_state(screenshot)
    
    # Determine which path was taken
    detection_path = []
    if conditions["loading_check"]:
        detection_path.append("loading")
    elif conditions["intro_check"]:
        detection_path.append("intro_sequence")
    elif conditions["dialogue_bottom_bright"] and conditions["dialogue_bottom_variance"]:
        detection_path.append("dialogue")
    elif conditions["title_screen_bright_lowvar"]:
        detection_path.append("title_screen (bright+lowvar)")
    elif conditions["title_screen_high_variance"]:
        detection_path.append("title_screen (high_variance)")
    elif conditions["battle_brightness_180"]:
        detection_path.append("battle (180)")
    elif conditions["battle_brightness_100"]:
        detection_path.append("battle (100)")
    elif conditions["menu_brightness_range"]:
        detection_path.append("menu")
    elif conditions["overworld_high_variance"]:
        detection_path.append("overworld (high_var)")
    elif conditions["overworld_medium_variance"]:
        detection_path.append("overworld (medium_var)")
    elif conditions["overworld_low_variance"]:
        detection_path.append("overworld (low_var)")
    else:
        detection_path.append("unknown")
    
    return {
        "detected_state": detected_state,
        "detection_path": detection_path,
        "overall_stats": {
            "mean_brightness": float(mean_brightness),
            "color_variance": float(color_variance),
            "sample_mean": float(sample_mean),
            "sample_variance": float(sample_variance),
            "shape": screenshot.shape
        },
        "bottom_section": bottom_stats,
        "conditions": conditions
    }

def capture_and_diagnose_false_positives():
    """Capture screens and diagnose false positives in real-time"""
    print("🔍 DIALOGUE DETECTION DIAGNOSTIC TOOL")
    print("=" * 50)
    
    config = TrainingConfig(
        mode=TrainingMode.FAST_MONITORED,
        rom_path="../roms/pokemon_crystal.gbc",
        max_actions=300,
        frames_per_action=6,
        headless=True,
        debug_mode=True,
        log_level="INFO"
    )
    
    trainer = UnifiedPokemonTrainer(config)
    detector = GameStateDetector(debug_mode=True)
    
    false_positives = []
    screenshot_count = 0
    last_state = None
    
    print("🎮 Monitoring for false positive dialogue detection...")
    print("💾 Will save screenshots of problematic transitions")
    
    # Create output directory
    os.makedirs("diagnostic_screenshots", exist_ok=True)
    
    try:
        for step in range(300):
            screenshot = trainer.pyboy.screen.ndarray
            current_state = detector.detect_game_state(screenshot)
            
            # Check for suspicious transitions
            if (last_state == "title_screen" and current_state == "dialogue") or \
               (last_state in ["loading", "intro_sequence"] and current_state == "dialogue"):
                
                print(f"\n🚨 FALSE POSITIVE DETECTED at step {step}")
                print(f"   Transition: {last_state} → {current_state}")
                
                # Detailed analysis
                analysis = analyze_screen_thoroughly(screenshot, detector)
                
                # Save screenshot
                screenshot_filename = f"diagnostic_screenshots/false_positive_{step}_{last_state}_to_{current_state}.png"
                img = Image.fromarray(screenshot)
                img.save(screenshot_filename)
                screenshot_count += 1
                
                # Log detailed analysis
                print(f"   📊 Analysis:")
                print(f"      Overall brightness: {analysis['overall_stats']['mean_brightness']:.1f}")
                print(f"      Overall variance: {analysis['overall_stats']['color_variance']:.1f}")
                print(f"      Bottom mean: {analysis['bottom_section'].get('mean', 'N/A'):.1f}")
                print(f"      Bottom variance: {analysis['bottom_section'].get('variance', 'N/A'):.1f}")
                print(f"      Detection path: {' → '.join(analysis['detection_path'])}")
                print(f"      Screenshot saved: {screenshot_filename}")
                
                # Store for later analysis
                false_positives.append({
                    "step": step,
                    "transition": f"{last_state} → {current_state}",
                    "analysis": analysis,
                    "screenshot_file": screenshot_filename
                })
                
            # Also capture some normal transitions for comparison
            elif current_state != last_state and len(false_positives) < 3:
                if current_state in ["title_screen", "dialogue", "overworld", "intro_sequence"]:
                    print(f"   Normal transition at step {step}: {last_state} → {current_state}")
                    
                    # Save comparison screenshot
                    screenshot_filename = f"diagnostic_screenshots/normal_{step}_{last_state}_to_{current_state}.png"
                    img = Image.fromarray(screenshot)
                    img.save(screenshot_filename)
                    
                    analysis = analyze_screen_thoroughly(screenshot, detector)
                    print(f"      Brightness: {analysis['overall_stats']['mean_brightness']:.1f}, "
                          f"Variance: {analysis['overall_stats']['color_variance']:.1f}")
            
            # Take action to progress
            if current_state == "intro_sequence":
                trainer.strategy_manager.execute_action(7)  # START
            elif current_state == "title_screen":
                trainer.strategy_manager.execute_action(7)  # START
            elif current_state == "dialogue":
                trainer.strategy_manager.execute_action(5)  # A
            else:
                trainer.strategy_manager.execute_action(5)  # A
            
            last_state = current_state
            time.sleep(0.1)  # Slower for observation
    
    except Exception as e:
        print(f"❌ Diagnostic failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        trainer._finalize_training()
    
    # Summary
    print(f"\n📋 DIAGNOSTIC SUMMARY")
    print(f"   False positives detected: {len(false_positives)}")
    print(f"   Screenshots saved: {screenshot_count}")
    
    if false_positives:
        print(f"\n🔍 FALSE POSITIVE ANALYSIS:")
        for i, fp in enumerate(false_positives):
            print(f"   #{i+1}: {fp['transition']} at step {fp['step']}")
            analysis = fp['analysis']
            conditions = analysis['conditions']
            
            print(f"      Why detected as dialogue:")
            if conditions['dialogue_bottom_bright']:
                print(f"        ✓ Bottom section bright ({analysis['bottom_section']['mean']:.1f} > 200)")
            if conditions['dialogue_bottom_variance']:
                print(f"        ✓ Bottom section has variance ({analysis['bottom_section']['variance']:.1f} > 100)")
            
            print(f"      Should have been detected as:")
            if conditions['title_screen_bright_lowvar']:
                print(f"        → title_screen (bright+lowvar: {analysis['overall_stats']['sample_mean']:.1f} >= 200, {analysis['overall_stats']['sample_variance']:.1f} < 100)")
            elif conditions['overworld_high_variance']:
                print(f"        → overworld (high_var: {analysis['overall_stats']['color_variance']:.1f} > 800)")
    
    return false_positives

if __name__ == "__main__":
    try:
        false_positives = capture_and_diagnose_false_positives()
        
        if false_positives:
            print(f"\n⚠️  ISSUES FOUND - Need to adjust detection thresholds")
        else:
            print(f"\n✅ NO FALSE POSITIVES DETECTED")
            
    except KeyboardInterrupt:
        print(f"\n⏸️ Diagnostic interrupted by user")
    except Exception as e:
        print(f"\n💥 Diagnostic crashed: {e}")
        import traceback
        traceback.print_exc()
