#!/usr/bin/env python3
"""
Pokemon Crystal Intro Screen Debugger

This tool helps us understand exactly what's happening in the Pokemon Crystal
intro sequence so we can improve the intro skip logic.
"""

import os
import sys
import time
import numpy as np
import cv2
from PIL import Image

# Add paths for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from trainer.trainer import UnifiedPokemonTrainer
from trainer.config import TrainingConfig, TrainingMode, LLMBackend


def save_screenshot(screenshot, filename, description=""):
    """Save a screenshot with timestamp for analysis"""
    if screenshot is not None and screenshot.size > 0:
        # Convert to PIL Image and save
        img = Image.fromarray(screenshot)
        timestamp = int(time.time() * 1000)
        full_filename = f"debug_screenshots/{timestamp}_{filename}.png"
        
        # Create directory if it doesn't exist
        os.makedirs("debug_screenshots", exist_ok=True)
        
        img.save(full_filename)
        print(f"   📸 Saved: {full_filename} - {description}")
        return full_filename
    return None


def analyze_screen_details(screenshot):
    """Analyze screen in detail to understand its characteristics"""
    if screenshot is None or screenshot.size == 0:
        return {"error": "Invalid screenshot"}
    
    # Basic statistics
    mean = np.mean(screenshot)
    variance = np.var(screenshot)
    std = np.std(screenshot)
    
    # Color analysis
    if len(screenshot.shape) == 3:
        # RGB analysis
        r_mean = np.mean(screenshot[:, :, 0])
        g_mean = np.mean(screenshot[:, :, 1])
        b_mean = np.mean(screenshot[:, :, 2])
        color_info = f"RGB({r_mean:.1f}, {g_mean:.1f}, {b_mean:.1f})"
    else:
        color_info = "Grayscale"
    
    # Region analysis
    h, w = screenshot.shape[:2]
    regions = {
        "top": screenshot[:h//4, :],
        "middle": screenshot[h//4:3*h//4, :],
        "bottom": screenshot[3*h//4:, :],
        "left": screenshot[:, :w//4],
        "center": screenshot[:, w//4:3*w//4],
        "right": screenshot[:, 3*w//4:],
    }
    
    region_stats = {}
    for region_name, region in regions.items():
        region_stats[region_name] = {
            "mean": float(np.mean(region)),
            "var": float(np.var(region)),
        }
    
    return {
        "overall": {
            "mean": float(mean),
            "variance": float(variance),
            "std": float(std),
            "shape": screenshot.shape,
            "color_info": color_info,
        },
        "regions": region_stats
    }


def debug_pokemon_crystal_intro(max_attempts=150, save_screenshots=True):
    """
    Debug the Pokemon Crystal intro sequence with detailed analysis
    """
    print("🔍 Pokemon Crystal Intro Sequence Debugger")
    print("=" * 60)
    
    # Create minimal config for debugging
    config = TrainingConfig(
        mode=TrainingMode.FAST_MONITORED,
        rom_path="../roms/pokemon_crystal.gbc",
        max_actions=1000,
        frames_per_action=4,
        headless=True,
        debug_mode=True,
        log_level="INFO"
    )
    
    print("🏗️ Initializing trainer...")
    trainer = UnifiedPokemonTrainer(config)
    
    print("🎮 Starting Pokemon Crystal intro analysis...")
    
    # Validate ROM is loaded
    print(f"📄 ROM Info:")
    print(f"   ROM path: {config.rom_path}")
    print(f"   PyBoy initialized: {trainer.pyboy is not None}")
    # Note: Accessing PyBoy internals differently based on version
        
    # Force some frame advancement to get the ROM started
    print(f"🔄 Advancing initial frames...")
    for i in range(60):  # Advance 60 frames to get ROM started
        if trainer.pyboy:
            trainer.pyboy.tick()
        if i % 20 == 0:
            screen = trainer.pyboy.screen.ndarray if trainer.pyboy else None
            mean = np.mean(screen) if screen is not None and screen.size > 0 else 0
            print(f"   Frame {i}: mean = {mean:.1f}")
        time.sleep(0.01)
    
    last_state = None
    state_change_count = 0
    screenshots_saved = 0
    detailed_log = []
    
    # Action sequence to try
    actions = [
        (7, "START"),  # START button
        (5, "A"),      # A button  
        (6, "B"),      # B button
        (7, "START"),  # START again
        (5, "A"),      # A again
    ]
    action_index = 0
    
    for attempt in range(max_attempts):
        # Get current screen and analyze it
        screen = trainer.pyboy.screen.ndarray
        current_state = trainer.game_state_detector.detect_game_state(screen)
        screen_analysis = analyze_screen_details(screen)
        
        # Track state changes
        if current_state != last_state:
            state_change_count += 1
            print(f"\n🔄 STATE CHANGE #{state_change_count}: {last_state} → {current_state}")
            
            if save_screenshots and screenshots_saved < 20:  # Limit screenshots
                filename = f"state_{state_change_count}_{current_state}"
                saved_file = save_screenshot(screen, filename, f"Transition to {current_state}")
                if saved_file:
                    screenshots_saved += 1
            
            last_state = current_state
        
        # Log every 10 attempts
        if attempt % 10 == 0:
            print(f"   Attempt {attempt:3d}: {current_state:15s} | "
                  f"Mean: {screen_analysis['overall']['mean']:6.1f} | "
                  f"Var: {screen_analysis['overall']['variance']:8.1f} | "
                  f"Shape: {screen_analysis['overall']['shape']}")
        
        # Detailed analysis for key states
        if current_state in ["title_screen", "menu", "dialogue"] or attempt % 20 == 0:
            log_entry = {
                "attempt": attempt,
                "state": current_state,
                "analysis": screen_analysis,
                "action_taken": None
            }
            detailed_log.append(log_entry)
        
        # Take action based on current state
        if current_state == "loading":
            # Wait for loading to finish
            time.sleep(0.2)
        elif current_state == "intro_sequence":
            # Skip with START
            action_id, action_name = 7, "START"
            trainer.strategy_manager.execute_action(action_id)
            print(f"      → Executed: {action_name} (skip intro)")
        elif current_state == "title_screen":
            # Try different approaches for title screen
            if attempt < 20:
                action_id, action_name = 7, "START"  # START to enter
            elif attempt < 40:
                action_id, action_name = 5, "A"      # A button
            elif attempt < 60:
                # Wait longer before pressing
                time.sleep(0.5)
                action_id, action_name = 7, "START"
            else:
                # Cycle through different actions
                action_id, action_name = actions[action_index % len(actions)]
                action_index += 1
                
            trainer.strategy_manager.execute_action(action_id)
            print(f"      → Executed: {action_name} (title screen)")
            
            if detailed_log:
                detailed_log[-1]["action_taken"] = action_name
                
        elif current_state == "menu":
            print(f"      🎉 REACHED MENU! Attempt {attempt}")
            action_id, action_name = 5, "A"  # Select first option
            trainer.strategy_manager.execute_action(action_id)
            print(f"      → Executed: {action_name} (select menu item)")
        elif current_state == "dialogue":
            print(f"      💬 IN DIALOGUE! Attempt {attempt}")
            action_id, action_name = 5, "A"  # Advance dialogue
            trainer.strategy_manager.execute_action(action_id)
            print(f"      → Executed: {action_name} (advance dialogue)")
        elif current_state == "overworld":
            print(f"      🏆 SUCCESS! REACHED OVERWORLD at attempt {attempt}")
            if save_screenshots:
                save_screenshot(screen, "SUCCESS_overworld", "Successfully reached overworld!")
            break
        else:
            # Unknown state - try START
            action_id, action_name = 7, "START"
            trainer.strategy_manager.execute_action(action_id)
            print(f"      → Executed: {action_name} (unknown state)")
        
        # Add small delay
        time.sleep(0.1)
    
    # Final analysis
    print(f"\n📊 ANALYSIS COMPLETE")
    print(f"Total attempts: {attempt + 1}")
    print(f"State changes: {state_change_count}")
    print(f"Screenshots saved: {screenshots_saved}")
    print(f"Final state: {current_state}")
    
    # Print detailed log summary
    print(f"\n📋 STATE TRANSITION SUMMARY:")
    for i, entry in enumerate(detailed_log):
        if i < 10:  # Show first 10 entries
            print(f"  {entry['attempt']:3d}: {entry['state']:15s} | "
                  f"Mean: {entry['analysis']['overall']['mean']:6.1f} | "
                  f"Action: {entry['action_taken'] or 'None'}")
    
    # Cleanup
    print(f"\n🧹 Cleaning up...")
    trainer._finalize_training()
    
    return {
        "final_state": current_state,
        "attempts": attempt + 1,
        "state_changes": state_change_count,
        "success": current_state == "overworld",
        "detailed_log": detailed_log
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Debug Pokemon Crystal intro sequence")
    parser.add_argument("--max-attempts", type=int, default=150, help="Maximum attempts")
    parser.add_argument("--no-screenshots", action="store_true", help="Don't save screenshots")
    
    args = parser.parse_args()
    
    try:
        result = debug_pokemon_crystal_intro(
            max_attempts=args.max_attempts,
            save_screenshots=not args.no_screenshots
        )
        
        print(f"\n✅ Debug Complete!")
        print(f"   Success: {result['success']}")
        print(f"   Final State: {result['final_state']}")
        print(f"   Attempts: {result['attempts']}")
        
    except KeyboardInterrupt:
        print(f"\n⏸️ Debug interrupted by user")
    except Exception as e:
        print(f"\n❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()
