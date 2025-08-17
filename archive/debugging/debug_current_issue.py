#!/usr/bin/env python3
"""
Quick diagnostic to understand current streaming issue.
"""

import os
import sys
import time
import numpy as np
import base64
import io
import requests
from PIL import Image

# Add paths for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from trainer.trainer import UnifiedPokemonTrainer
from trainer.config import TrainingConfig, TrainingMode, LLMBackend


def analyze_current_issue():
    """Analyze what's happening with the streaming"""
    print("🔍 Diagnosing Current Streaming Issue")
    print("=" * 60)
    
    # Test 1: Check if web monitor is accessible
    print("\n1️⃣ Testing web monitor accessibility...")
    try:
        response = requests.get("http://127.0.0.1:5001/api/status", timeout=5)
        if response.status_code == 200:
            print("   ✅ Web monitor API accessible")
            status_data = response.json()
            print(f"   Status: {status_data}")
        else:
            print(f"   ⚠️ Web monitor returned status {response.status_code}")
    except Exception as e:
        print(f"   ❌ Cannot connect to web monitor: {e}")
        print("   💡 Make sure the training process is running")
        return False
    
    # Test 2: Create a fresh trainer and check its state
    print("\n2️⃣ Creating fresh trainer to test state...")
    try:
        config = TrainingConfig(
            mode=TrainingMode.FAST_MONITORED,
            rom_path="../roms/pokemon_crystal.gbc",
            capture_screens=True,
            headless=True,
            debug_mode=True
        )
        trainer = UnifiedPokemonTrainer(config)
        
        # Check initial PyBoy state
        screen = trainer.pyboy.screen.ndarray
        variance = np.var(screen.astype(np.float32))
        print(f"   Initial PyBoy variance: {variance:.1f}")
        
        if variance < 1.0:
            print("   ⚠️ PyBoy starting with blank screen")
            
            # Try advancing frames
            print("   🔄 Advancing frames...")
            for i in range(50):
                trainer.pyboy.tick()
                if i % 10 == 0:
                    screen = trainer.pyboy.screen.ndarray
                    variance = np.var(screen.astype(np.float32))
                    print(f"      Frame {i}: variance = {variance:.1f}")
                    
                    if variance > 100:
                        print(f"   ✅ Content found at frame {i}")
                        break
            
            # Try pressing START
            print("   🎮 Pressing START button...")
            for i in range(10):
                trainer.strategy_manager.execute_action(7)  # START
                screen = trainer.pyboy.screen.ndarray
                variance = np.var(screen.astype(np.float32))
                print(f"      START {i+1}: variance = {variance:.1f}")
                
                if variance > 1000:
                    print("   ✅ Reached game content")
                    break
                elif i == 9:
                    print("   ⚠️ Still stuck at menu/title screen")
        
        # Save current screen for inspection
        final_screen = trainer.pyboy.screen.ndarray
        final_variance = np.var(final_screen.astype(np.float32))
        print(f"   Final variance: {final_variance:.1f}")
        
        # Convert and save
        if len(final_screen.shape) == 3 and final_screen.shape[2] == 4:
            rgb_screen = final_screen[:, :, :3]
        else:
            rgb_screen = final_screen
            
        current_img = Image.fromarray(rgb_screen.astype(np.uint8))
        current_img.save("debug_current_state.png")
        print("   💾 Saved current state: debug_current_state.png")
        
        # Test trainer's processing pipeline
        print("\n3️⃣ Testing trainer processing pipeline...")
        trainer._start_screen_capture()
        time.sleep(3)
        
        if trainer.latest_screen:
            screen_data = trainer.latest_screen
            print(f"   ✅ Trainer captured screen: {screen_data.get('data_length', 0)} bytes")
            
            # Decode and analyze
            try:
                img_b64 = screen_data['image_b64']
                img_data = base64.b64decode(img_b64)
                processed_img = Image.open(io.BytesIO(img_data))
                processed_array = np.array(processed_img)
                processed_variance = np.var(processed_array.astype(np.float32))
                
                print(f"   Processed image size: {processed_img.size}")
                print(f"   Processed variance: {processed_variance:.1f}")
                
                processed_img.save("debug_processed_screen.jpg")
                print("   💾 Saved processed screen: debug_processed_screen.jpg")
                
                if processed_variance < 10:
                    print("   ❌ ISSUE: Processed screen is blank!")
                    print("   🔍 The trainer is capturing and processing blank frames")
                else:
                    print("   ✅ Processed screen has content")
                
            except Exception as e:
                print(f"   ❌ Failed to decode processed screen: {e}")
        else:
            print("   ❌ Trainer did not capture any screen data")
        
        trainer._finalize_training()
        
    except Exception as e:
        print(f"   ❌ Trainer creation failed: {e}")
        return False
    
    print("\n🎯 DIAGNOSIS SUMMARY:")
    print("-" * 40)
    
    if final_variance < 10:
        print("❌ ROOT CAUSE: Emulator stuck on blank/title screen")
        print("💡 SOLUTION: Need more aggressive game state advancement")
        print("📋 RECOMMENDATIONS:")
        print("   1. Use enhanced game start sequence")
        print("   2. Try different button combinations")
        print("   3. Wait longer for game to load")
        print("   4. Check if ROM file is correct")
    elif processed_variance < 10:
        print("❌ ROOT CAUSE: Processing pipeline creating blank images")
        print("💡 SOLUTION: Fix image processing/compression")
    else:
        print("✅ ISSUE: Streaming/bridge problem")
        print("💡 SOLUTION: Check bridge connection and web monitor")
    
    return True


def quick_game_advancement_test():
    """Test more aggressive game advancement"""
    print("\n🎮 Quick Game Advancement Test")
    print("-" * 50)
    
    config = TrainingConfig(
        mode=TrainingMode.FAST_MONITORED,
        rom_path="../roms/pokemon_crystal.gbc",
        headless=True,
        debug_mode=True
    )
    trainer = UnifiedPokemonTrainer(config)
    
    # More aggressive button pressing sequence
    button_actions = [
        (7, "START", 15),   # Press START many times
        (5, "A", 10),       # Press A many times  
        (1, "DOWN", 3),     # Navigate down
        (5, "A", 5),        # Select
        (3, "RIGHT", 3),    # Navigate right
        (5, "A", 5),        # Select
        (8, "B", 3),        # Back/cancel
        (7, "START", 5),    # START again
    ]
    
    print("🔄 Trying aggressive button sequence...")
    
    for action_code, button_name, presses in button_actions:
        print(f"   Pressing {button_name} {presses} times...")
        
        for i in range(presses):
            trainer.strategy_manager.execute_action(action_code)
            
            # Check screen every few presses
            if i % 3 == 0:
                screen = trainer.pyboy.screen.ndarray
                variance = np.var(screen.astype(np.float32))
                
                if variance > 2000:  # Rich gameplay content
                    print(f"   ✅ Found gameplay content! (variance: {variance:.1f})")
                    
                    # Save successful state
                    if len(screen.shape) == 3 and screen.shape[2] == 4:
                        rgb_screen = screen[:, :, :3]
                    else:
                        rgb_screen = screen
                    
                    gameplay_img = Image.fromarray(rgb_screen.astype(np.uint8))
                    gameplay_img.save("debug_gameplay_found.png")
                    print("   💾 Saved gameplay screen: debug_gameplay_found.png")
                    
                    trainer._finalize_training()
                    return True
        
        # Brief pause between button types
        time.sleep(0.2)
    
    # Final check
    final_screen = trainer.pyboy.screen.ndarray
    final_variance = np.var(final_screen.astype(np.float32))
    print(f"Final result: variance = {final_variance:.1f}")
    
    trainer._finalize_training()
    return final_variance > 1000


if __name__ == "__main__":
    print("🔧 Current Issue Diagnostic Tool")
    print("=" * 50)
    
    # Run main diagnosis
    if analyze_current_issue():
        print("\n" + "="*60)
        
        # Try advanced game advancement
        if quick_game_advancement_test():
            print("✅ Advanced game advancement worked!")
            print("💡 Use the enhanced streaming fix for better results")
        else:
            print("❌ Still having issues advancing game state")
            print("💡 May need to check ROM file or try different approach")
    
    print("\n🎯 Run enhanced_streaming_fix.py for the complete solution")
