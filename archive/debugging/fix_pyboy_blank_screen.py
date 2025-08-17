#!/usr/bin/env python3
"""
Fix for PyBoy blank screen - Advance the emulator to show game content.
"""

import os
import sys
import time
import numpy as np
from PIL import Image

# Add the parent directory to the Python path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from trainer.trainer import UnifiedPokemonTrainer
from trainer.config import TrainingConfig, TrainingMode
from monitoring.trainer_monitor_bridge_patched import TrainerWebMonitorBridge
from monitoring.web_monitor import PokemonRLWebMonitor


def fix_pyboy_blank_screen():
    """Fix PyBoy blank screen by advancing the emulator"""
    print("🔧 Fixing PyBoy Blank Screen Issue")
    print("=" * 60)
    
    # Create trainer
    config = TrainingConfig(
        mode=TrainingMode.FAST_MONITORED,
        rom_path="../roms/pokemon_crystal.gbc",
        capture_screens=True,
        headless=True,
        debug_mode=True
    )
    trainer = UnifiedPokemonTrainer(config)
    
    print("\n1️⃣ INITIAL PYBOY STATE")
    print("-" * 40)
    
    # Check initial state
    initial_screen = trainer.pyboy.screen.ndarray
    initial_variance = np.var(initial_screen.astype(np.float32))
    print(f"Initial screen variance: {initial_variance:.3f}")
    
    if initial_variance < 1.0:
        print("⚠️ PyBoy showing blank screen - need to advance emulator")
        
        print("\n2️⃣ ADVANCING PYBOY TO START GAME")
        print("-" * 40)
        
        # Advance PyBoy frames to get past initial loading
        print("⏳ Advancing frames...")
        for i in range(100):  # Advance 100 frames
            trainer.pyboy.tick()
            if i % 20 == 0:
                screen = trainer.pyboy.screen.ndarray
                variance = np.var(screen.astype(np.float32))
                print(f"   Frame {i}: variance = {variance:.3f}")
                
                if variance > 10:  # Found content
                    print(f"✅ Found game content at frame {i}")
                    break
        
        print("\n3️⃣ PRESSING START TO BEGIN GAME")
        print("-" * 40)
        
        # Press START button (action 7) to get past title screen
        for i in range(20):
            trainer.strategy_manager.execute_action(7)  # START button
            screen = trainer.pyboy.screen.ndarray
            variance = np.var(screen.astype(np.float32))
            print(f"   START press {i+1}: variance = {variance:.3f}")
            
            if variance > 100:  # Significant content change
                print(f"✅ Game started - screen has content")
                break
                
        print("\n4️⃣ FINAL SCREEN STATE")
        print("-" * 40)
        
        final_screen = trainer.pyboy.screen.ndarray
        final_variance = np.var(final_screen.astype(np.float32))
        print(f"Final screen variance: {final_variance:.3f}")
        
        if final_variance > 1.0:
            print("✅ PyBoy now showing game content!")
            
            # Convert and save the working screenshot
            rgb_screen = final_screen[:, :, :3] if final_screen.shape[2] == 4 else final_screen
            game_img = Image.fromarray(rgb_screen.astype(np.uint8))
            game_img.save("debug_game_content.png")
            print("💾 Saved working screenshot: debug_game_content.png")
            
        else:
            print("❌ Still showing blank screen")
            trainer._finalize_training()
            return False
    else:
        print("✅ PyBoy already showing content")
        final_variance = initial_variance
    
    print("\n5️⃣ TESTING WITH FIXED EMULATOR")
    print("-" * 40)
    
    # Now test the complete pipeline with working emulator
    trainer._start_screen_capture()
    time.sleep(2)
    
    # Check if trainer captures are working
    trainer_screen = trainer._simple_screenshot_capture()
    if trainer_screen is not None:
        trainer_variance = np.var(trainer_screen.astype(np.float32))
        print(f"Trainer capture variance: {trainer_variance:.3f}")
        
        if trainer_variance > 1.0:
            print("✅ Trainer captures are now working")
            
            # Test bridge transfer
            web_monitor = PokemonRLWebMonitor()
            bridge = TrainerWebMonitorBridge(trainer, web_monitor)
            bridge.start_bridge()
            
            print("⏳ Testing bridge with working screenshots...")
            time.sleep(5)
            
            bridge_stats = bridge.get_bridge_stats()
            print(f"Bridge transferred: {bridge_stats['screenshots_transferred']} screenshots")
            print(f"Bridge success rate: {bridge_stats['success_rate']:.1f}%")
            
            bridge.stop_bridge()
            
            if bridge_stats['screenshots_transferred'] > 0:
                print("✅ SUCCESS! Bridge is now transferring screenshots")
                trainer._finalize_training()
                return True
            else:
                print("❌ Bridge still not transferring")
        else:
            print("❌ Trainer captures still blank")
    else:
        print("❌ Trainer capture failed")
    
    trainer._finalize_training()
    return False


def create_final_fix():
    """Create the final comprehensive fix"""
    print("\n" + "=" * 70)
    print("🎯 CREATING FINAL FIX FOR SOCKET STREAMING")
    print("=" * 70)
    
    fix_code = '''
# FINAL FIX FOR SOCKET BLANK SCREEN ISSUE
# =====================================

def fix_blank_screen_streaming(trainer):
    """
    Complete fix for blank screen streaming issue.
    
    The problem was that PyBoy starts with a blank screen and needs
    to be advanced to show actual game content.
    """
    
    # Step 1: Advance PyBoy past initial blank frames
    print("🎮 Advancing PyBoy to show game content...")
    
    for i in range(100):
        trainer.pyboy.tick()
        if i % 20 == 0:
            screen = trainer.pyboy.screen.ndarray
            variance = np.var(screen.astype(np.float32))
            if variance > 10:  # Found content
                print(f"✅ Game content found at frame {i}")
                break
    
    # Step 2: Press START to begin game
    print("🕹️ Pressing START to begin game...")
    
    for i in range(20):
        trainer.strategy_manager.execute_action(7)  # START button
        screen = trainer.pyboy.screen.ndarray
        variance = np.var(screen.astype(np.float32))
        if variance > 100:
            print(f"✅ Game started successfully")
            break
    
    # Step 3: Start screen capture AFTER game has content
    print("📸 Starting screen capture with game content...")
    
    if trainer.config.capture_screens:
        trainer._start_screen_capture()
    
    print("✅ Fix applied - streaming should now work!")
    
# Usage:
# trainer = UnifiedPokemonTrainer(config)
# fix_blank_screen_streaming(trainer)
# # Now start bridge and web monitor
'''
    
    print(fix_code)
    
    # Save to file
    with open('socket_streaming_final_fix.py', 'w') as f:
        f.write(fix_code)
    
    print("\n💾 Final fix saved to: socket_streaming_final_fix.py")
    
    print("\n📋 SUMMARY:")
    print("-" * 50)
    print("❌ PROBLEM: PyBoy starts with blank screen")
    print("✅ SOLUTION: Advance emulator before starting capture")
    print("🔧 METHOD: Run frames + press START button")
    print("📊 RESULT: Game content → Working screenshots → Socket streaming")


if __name__ == "__main__":
    success = fix_pyboy_blank_screen()
    
    if success:
        create_final_fix()
    else:
        print("\n❌ Fix testing failed - may need further investigation")
