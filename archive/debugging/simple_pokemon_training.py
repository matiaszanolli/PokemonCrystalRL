#!/usr/bin/env python3
"""
Simplified, stable Pokemon Crystal training launcher.
"""

import os
import sys
import time
import numpy as np
import signal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from trainer.trainer import UnifiedPokemonTrainer
from trainer.config import TrainingConfig, TrainingMode, LLMBackend


def skip_to_pokemon_gameplay(trainer):
    """Simple, reliable Pokemon Crystal intro skip"""
    print("🎮 Skipping to Pokemon gameplay...")
    
    # Simple sequence - press START multiple times to get through intro
    for i in range(30):
        trainer.strategy_manager.execute_action(7)  # START
        
        if i % 5 == 0:
            screen = trainer.pyboy.screen.ndarray
            variance = np.var(screen.astype(np.float32))
            print(f"   Frame {i}: variance = {variance:.1f}")
            
            # Check if we have rich content (gameplay)
            if variance > 5000:
                print(f"✅ Found gameplay content at frame {i}")
                
                # Test movement to confirm controllable gameplay
                for direction in [0, 1, 2, 3]:  # Test all directions
                    trainer.strategy_manager.execute_action(direction)
                    time.sleep(0.05)
                
                return True
    
    # If START sequence didn't work, try A button spam
    print("   Trying A button sequence...")
    for i in range(20):
        trainer.strategy_manager.execute_action(5)  # A
        
        if i % 5 == 0:
            screen = trainer.pyboy.screen.ndarray
            variance = np.var(screen.astype(np.float32))
            print(f"   A button {i}: variance = {variance:.1f}")
            
            if variance > 5000:
                print(f"✅ Found gameplay content with A button")
                return True
    
    # Final check
    final_screen = trainer.pyboy.screen.ndarray
    final_variance = np.var(final_screen.astype(np.float32))
    print(f"Final result: variance = {final_variance:.1f}")
    
    return final_variance > 3000


def launch_simple_pokemon_training():
    """Launch simple, stable Pokemon training"""
    print("🚀 Simple Pokemon Crystal Training")
    print("=" * 50)
    
    # Simple, stable configuration
    config = TrainingConfig(
        mode=TrainingMode.FAST_MONITORED,
        rom_path="../roms/pokemon_crystal.gbc",
        
        # Basic settings
        max_actions=5000,
        llm_backend=LLMBackend.NONE,
        frames_per_action=4,
        
        # No web monitoring for now - focus on getting the training stable
        enable_web=False,
        capture_screens=False,  # Disable to avoid conflicts
        
        # Stable settings
        headless=True,
        debug_mode=True,
        log_level="INFO"
    )
    
    print(f"📋 Configuration:")
    print(f"   Max actions: {config.max_actions}")
    print(f"   Web monitoring: {config.enable_web}")
    print(f"   Screen capture: {config.capture_screens}")
    print()
    
    # Create trainer
    print("🏗️ Creating trainer...")
    trainer = UnifiedPokemonTrainer(config)
    
    # Skip to gameplay
    if not skip_to_pokemon_gameplay(trainer):
        print("❌ Failed to reach Pokemon gameplay")
        trainer._finalize_training()
        return False
    
    print("✅ Ready for Pokemon training!")
    
    # Simple graceful shutdown
    def signal_handler(sig, frame):
        print("\n⏸️ Stopping training...")
        trainer._finalize_training()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    print("🎯 Starting Pokemon Crystal training...")
    print("🔄 Press Ctrl+C to stop")
    print()
    
    try:
        # Start training
        trainer.start_training()
    except KeyboardInterrupt:
        signal_handler(signal.SIGINT, None)
    except Exception as e:
        print(f"❌ Training error: {e}")
        trainer._finalize_training()
        return False
    
    return True


def test_pokemon_gameplay_only():
    """Test just the Pokemon gameplay skip without training"""
    print("🧪 Testing Pokemon Gameplay Skip Only")
    print("=" * 45)
    
    config = TrainingConfig(
        mode=TrainingMode.FAST_MONITORED,
        rom_path="../roms/pokemon_crystal.gbc",
        headless=True,
        debug_mode=True,
        enable_web=False,
        capture_screens=False
    )
    
    trainer = UnifiedPokemonTrainer(config)
    
    success = skip_to_pokemon_gameplay(trainer)
    
    if success:
        # Save screenshot for verification
        final_screen = trainer.pyboy.screen.ndarray
        if len(final_screen.shape) == 3 and final_screen.shape[2] == 4:
            rgb_screen = final_screen[:, :, :3]
        else:
            rgb_screen = final_screen
            
        from PIL import Image
        gameplay_img = Image.fromarray(rgb_screen.astype(np.uint8))
        gameplay_img.save("simple_pokemon_gameplay.png")
        print("💾 Saved: simple_pokemon_gameplay.png")
        
        # Test a few more actions to show it's working
        print("🎮 Testing extended gameplay...")
        for i in range(20):
            action = i % 8  # Cycle through all actions
            trainer.strategy_manager.execute_action(action)
            
            if i % 5 == 0:
                screen = trainer.pyboy.screen.ndarray
                variance = np.var(screen.astype(np.float32))
                print(f"   Action {i}: variance = {variance:.1f}")
    
    trainer._finalize_training()
    return success


if __name__ == "__main__":
    print("🎮 Simple Pokemon Crystal Training")
    print("=" * 50)
    
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        # Test gameplay skip only
        if test_pokemon_gameplay_only():
            print("\n✅ Pokemon gameplay test PASSED")
            print("💡 Run without --test for full training")
        else:
            print("\n❌ Pokemon gameplay test FAILED")
    else:
        # Full training
        success = launch_simple_pokemon_training()
        if success:
            print("\n✅ Pokemon training completed successfully")
        else:
            print("\n❌ Pokemon training failed")
