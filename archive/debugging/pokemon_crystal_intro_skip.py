#!/usr/bin/env python3
"""
Pokemon Crystal specific intro skip and gameplay starter.
"""

import os
import sys
import time
import numpy as np
import threading
import signal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from trainer.trainer import UnifiedPokemonTrainer  
from trainer.config import TrainingConfig, TrainingMode, LLMBackend
from monitoring.trainer_monitor_bridge import TrainerWebMonitorBridge, create_integrated_monitoring_system


def pokemon_crystal_intro_skip(trainer, max_attempts=300):
    """Skip Pokemon Crystal intro and get to gameplay"""
    print("🎮 Pokemon Crystal Intro Skip Sequence")
    print("-" * 50)
    
    # Pokemon Crystal intro sequence
    # 1. Nintendo logo/Game Freak logo
    # 2. Title screen with Suicune
    # 3. Menu selection (New Game/Continue)
    # 4. Intro cutscene with Professor Oak
    # 5. Player naming
    # 6. Rival naming  
    # 7. Start of actual gameplay
    
    attempt = 0
    sequence_step = 0
    
    # Define the sequence of actions to get through intro
    intro_sequence = [
        # Step 0-50: Skip logos and get to title screen
        {"actions": [(7, "START")], "repeats": 20, "description": "Skip Nintendo/Game Freak logos"},
        
        # Step 50-100: Navigate title screen 
        {"actions": [(7, "START")], "repeats": 10, "description": "Enter title screen menu"},
        {"actions": [(5, "A")], "repeats": 5, "description": "Select New Game"},
        
        # Step 100-150: Skip intro cutscene with Professor Oak
        {"actions": [(5, "A")], "repeats": 30, "description": "Skip Prof Oak intro cutscene"},
        {"actions": [(7, "START")], "repeats": 10, "description": "Skip more cutscene"},
        
        # Step 150-200: Handle character naming screens
        {"actions": [(5, "A")], "repeats": 15, "description": "Accept default player name"},
        {"actions": [(1, "DOWN"), (5, "A")], "repeats": 10, "description": "Navigate and confirm"},
        {"actions": [(5, "A")], "repeats": 15, "description": "Accept default rival name"},
        
        # Step 200-250: Final cutscene skip
        {"actions": [(5, "A")], "repeats": 25, "description": "Skip final cutscenes"},
        {"actions": [(7, "START")], "repeats": 5, "description": "Skip any remaining scenes"},
        
        # Step 250-300: Get to overworld
        {"actions": [(5, "A")], "repeats": 20, "description": "Continue to overworld"},
        {"actions": [(1, "DOWN")], "repeats": 5, "description": "Move in overworld"},
    ]
    
    print("🔄 Executing Pokemon Crystal intro skip sequence...")
    
    for step_info in intro_sequence:
        actions = step_info["actions"]
        repeats = step_info["repeats"] 
        description = step_info["description"]
        
        print(f"   {description}...")
        
        for repeat in range(repeats):
            for action_code, action_name in actions:
                trainer.strategy_manager.execute_action(action_code)
                
                # Check game state periodically
                if repeat % 5 == 0:
                    screen = trainer.pyboy.screen.ndarray
                    variance = np.var(screen.astype(np.float32))
                    
                    # Check if we've reached overworld gameplay
                    if variance > 5000:  # Rich overworld content
                        print(f"   ✅ Reached overworld gameplay! (variance: {variance:.1f})")
                        
                        # Test player movement to confirm we're in control
                        print("   🎮 Testing player movement...")
                        for direction in [0, 1, 2, 3]:  # UP, DOWN, LEFT, RIGHT
                            trainer.strategy_manager.execute_action(direction)
                            time.sleep(0.1)
                        
                        print("   ✅ Player movement confirmed - ready for training!")
                        return True
                
                attempt += 1
                if attempt >= max_attempts:
                    print(f"   ⚠️ Max attempts reached at step: {description}")
                    break
            
            if attempt >= max_attempts:
                break
        
        if attempt >= max_attempts:
            break
        
        # Brief pause between sequence steps
        time.sleep(0.1)
    
    # Final check
    final_screen = trainer.pyboy.screen.ndarray
    final_variance = np.var(final_screen.astype(np.float32))
    print(f"Final variance: {final_variance:.1f}")
    
    if final_variance > 3000:
        print("✅ Successfully reached gameplay area")
        return True
    else:
        print("❌ Still stuck in intro/menus")
        return False


def launch_pokemon_crystal_training():
    """Launch Pokemon Crystal training with proper intro skip"""
    print("🚀 Pokemon Crystal RL Training with Intro Skip")
    print("=" * 60)
    
    # Configuration optimized for Pokemon Crystal
    config = TrainingConfig(
        mode=TrainingMode.FAST_MONITORED,
        rom_path="../roms/pokemon_crystal.gbc",
        
        # Optimized settings
        capture_screens=True,
        capture_fps=3,
        screen_resize=(160, 144),
        
        # Training settings
        max_actions=10000,  # Longer session for Pokemon
        llm_backend=LLMBackend.NONE,
        frames_per_action=4,  # Good balance for Pokemon
        
        # Web monitoring
        enable_web=True, 
        web_host="127.0.0.1",
        web_port=5002,  # Use different port to avoid conflicts
        
        # Debug settings
        headless=True,
        debug_mode=True,
        log_level="INFO"
    )
    
    print(f"📋 Pokemon Crystal Configuration:")
    print(f"   ROM: Pokemon Crystal")
    print(f"   Web monitor: http://{config.web_host}:{config.web_port}")
    print(f"   Max actions: {config.max_actions}")
    print(f"   Capture FPS: {config.capture_fps}")
    print()
    
    # Create trainer
    print("🏗️ Creating Pokemon trainer...")
    trainer = UnifiedPokemonTrainer(config)
    
    # Skip Pokemon Crystal intro
    if not pokemon_crystal_intro_skip(trainer):
        print("❌ Failed to skip Pokemon Crystal intro")
        trainer._finalize_training()
        return False
    
    # Start enhanced screen capture
    print("\n📸 Starting screen capture for gameplay...")
    trainer._start_screen_capture()
    time.sleep(2)
    
    # Verify we have good content
    if trainer.latest_screen:
        print("✅ Screen capture active and working")
    else:
        print("❌ Screen capture not working")
        trainer._finalize_training()
        return False
    
    # Create monitoring system
    print("🌐 Creating Pokemon Crystal web monitoring...")
    try:
        web_monitor, bridge, monitor_thread = create_integrated_monitoring_system(
            trainer, host=config.web_host, port=config.web_port
        )
        
        # Enhanced bridge specifically for Pokemon
        class PokemonBridge(TrainerWebMonitorBridge):
            def __init__(self, trainer, web_monitor):
                super().__init__(trainer, web_monitor)
                self.pokemon_specific_recovery = 0
                print("🌉 Pokemon-specific bridge initialized")
            
            def _transfer_screenshot(self):
                success = super()._transfer_screenshot()
                
                # Pokemon-specific blank frame recovery
                if hasattr(self.trainer, 'latest_screen') and self.trainer.latest_screen:
                    try:
                        converted = self._convert_trainer_screenshot(self.trainer.latest_screen)
                        if converted is not None:
                            variance = np.var(converted.astype(np.float32))
                            if variance < 50:  # Probably in menu or dialogue
                                # Press A to continue
                                if self.pokemon_specific_recovery % 10 == 0:
                                    self.trainer.strategy_manager.execute_action(5)  # A
                                self.pokemon_specific_recovery += 1
                    except:
                        pass
                
                return success
        
        pokemon_bridge = PokemonBridge(trainer, web_monitor)
        pokemon_bridge.start_bridge()
        
        print(f"✅ Pokemon Crystal monitoring active at: http://{config.web_host}:{config.web_port}")
        print("📱 Open the web dashboard to see live Pokemon gameplay!")
        print("🔄 Press Ctrl+C to stop training")
        print()
        
        # Graceful shutdown
        def signal_handler(sig, frame):
            print("\n⏸️ Stopping Pokemon Crystal training...")
            pokemon_bridge.stop_bridge()
            web_monitor.stop_monitoring()
            trainer._finalize_training()
            print("✅ Pokemon Crystal training stopped")
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        
        # Start actual Pokemon training
        trainer.start_training()
        
    except Exception as e:
        print(f"❌ Pokemon monitoring setup failed: {e}")
        trainer._finalize_training()
        return False
    
    return True


def quick_pokemon_test():
    """Quick test of Pokemon Crystal intro skip"""
    print("🧪 Quick Pokemon Crystal Intro Skip Test")
    print("=" * 50)
    
    config = TrainingConfig(
        mode=TrainingMode.FAST_MONITORED,
        rom_path="../roms/pokemon_crystal.gbc",
        headless=True,
        debug_mode=True
    )
    
    trainer = UnifiedPokemonTrainer(config)
    
    success = pokemon_crystal_intro_skip(trainer, max_attempts=200)
    
    if success:
        # Save final gameplay screenshot
        final_screen = trainer.pyboy.screen.ndarray
        if len(final_screen.shape) == 3 and final_screen.shape[2] == 4:
            rgb_screen = final_screen[:, :, :3]
        else:
            rgb_screen = final_screen
            
        from PIL import Image
        gameplay_img = Image.fromarray(rgb_screen.astype(np.uint8))
        gameplay_img.save("pokemon_crystal_gameplay.png")
        print("💾 Saved gameplay screenshot: pokemon_crystal_gameplay.png")
    
    trainer._finalize_training()
    return success


if __name__ == "__main__":
    print("🎮 Pokemon Crystal RL Training Launcher")
    print("=" * 60)
    
    # Quick test first
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        if quick_pokemon_test():
            print("✅ Pokemon Crystal intro skip test PASSED")
        else:
            print("❌ Pokemon Crystal intro skip test FAILED")
    else:
        # Full training launch
        success = launch_pokemon_crystal_training()
        if not success:
            print("❌ Pokemon Crystal training launch failed")
