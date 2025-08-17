#!/usr/bin/env python3
"""
Pokemon Crystal RL Training Launcher

Production-ready launcher with integrated fixes for:
- Pokemon Crystal intro skip
- Socket streaming optimization
- Web monitoring stability
"""

import os
import sys
import time
import numpy as np
import signal
import argparse

# Add paths for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from trainer.trainer import UnifiedPokemonTrainer
from trainer.config import TrainingConfig, TrainingMode, LLMBackend
from monitoring.trainer_monitor_bridge import TrainerWebMonitorBridge, create_integrated_monitoring_system
import shutil


def backup_save_files(rom_path):
    """
    Temporarily backup save files to allow fresh start.
    
    Args:
        rom_path: Path to the ROM file
        
    Returns:
        list: List of backed up files for later restoration
    """
    save_extensions = ['.ram', '.rtc', '.state']
    backed_up_files = []
    
    for ext in save_extensions:
        save_file = rom_path + ext
        backup_file = save_file + '.backup'
        
        if os.path.exists(save_file):
            try:
                shutil.move(save_file, backup_file)
                backed_up_files.append((save_file, backup_file))
                print(f"   📁 Backed up: {os.path.basename(save_file)}")
            except Exception as e:
                print(f"   ⚠️ Failed to backup {save_file}: {e}")
    
    return backed_up_files


def restore_save_files(backed_up_files):
    """
    Restore previously backed up save files.
    
    Args:
        backed_up_files: List of (original, backup) file paths
    """
    for original_file, backup_file in backed_up_files:
        if os.path.exists(backup_file):
            try:
                shutil.move(backup_file, original_file)
                print(f"   📁 Restored: {os.path.basename(original_file)}")
            except Exception as e:
                print(f"   ⚠️ Failed to restore {original_file}: {e}")


def skip_pokemon_crystal_intro(trainer, max_attempts=100, allow_continue=True, fresh_start=False):
    """
    Skip Pokemon Crystal intro sequence to reach controllable gameplay.
    
    This function handles the Pokemon Crystal startup sequence:
    1. Nintendo/Game Freak logos (START to skip)
    2. Title screen with Suicune animation (START to enter)
    3. Main menu: NEW GAME / CONTINUE (A to select NEW GAME)
    4. Professor Oak intro cutscene (START/A to skip)
    5. Character selection (Boy/Girl) (A to select)
    6. Name input screen (START/A to accept default)
    7. Final intro cutscenes (START/A to skip)
    8. Overworld gameplay begins
    
    Args:
        trainer: UnifiedPokemonTrainer instance
        max_attempts: Maximum attempts to reach overworld
        
    Returns:
        bool: True if overworld gameplay was reached, False otherwise
    """
    print("🎮 Skipping Pokemon Crystal intro sequence...")
    
    # Track state progression to avoid getting stuck
    last_state = None
    state_stuck_counter = 0
    title_screen_attempts = 0
    
    for attempt in range(max_attempts):
        screen = trainer.pyboy.screen.ndarray
        current_state = trainer.game_state_detector.detect_game_state(screen)
        variance = np.var(screen.astype(np.float32))
        
        # Track if we're stuck in the same state
        if current_state == last_state:
            state_stuck_counter += 1
        else:
            state_stuck_counter = 0
            last_state = current_state
        
        # Log progress every 5 attempts
        if attempt % 5 == 0:
            print(f"   Attempt {attempt}: state = {current_state}, variance = {variance:.1f}")
        
        # SUCCESS: Reached overworld gameplay
        if current_state == "overworld":
            print(f"✅ Reached Pokemon Crystal overworld gameplay!")
            
            # Brief test of player controls to confirm
            print("   🎮 Testing player controls...")
            test_actions = [0, 1, 2, 3]  # UP, DOWN, LEFT, RIGHT
            for test_action in test_actions:
                trainer.strategy_manager.execute_action(test_action)
                time.sleep(0.05)
            
            return True
        
        # PHASE 1: Intro sequence (logos, startup)
        elif current_state == "intro_sequence":
            # Skip logos and startup screens with START
            trainer.strategy_manager.execute_action(7)  # START
            time.sleep(0.1)
        
        # PHASE 2: Title screen (Suicune animation)
        elif current_state == "title_screen":
            title_screen_attempts += 1
            
            if title_screen_attempts < 5:
                # First few attempts: press START to enter main menu
                trainer.strategy_manager.execute_action(7)  # START
            elif title_screen_attempts < 15:
                # If START isn't working, try A button
                trainer.strategy_manager.execute_action(5)  # A
            else:
                # Title screen usually needs specific timing, try both buttons
                trainer.strategy_manager.execute_action(7)  # START
                time.sleep(0.1)
                trainer.strategy_manager.execute_action(5)  # A
            
            time.sleep(0.15)  # Title screen needs more time
        
        # PHASE 3: Menu screens (NEW GAME/CONTINUE selection)
        elif current_state == "menu":
            if allow_continue and state_stuck_counter < 3:
                print(f"   📋 In menu - trying CONTINUE first...")
                # If we have a save file, CONTINUE might be faster
                trainer.strategy_manager.execute_action(5)  # A to select (CONTINUE is usually first)
            else:
                print(f"   📋 In menu - selecting NEW GAME...")
                # Navigate to NEW GAME: DOWN then A
                trainer.strategy_manager.execute_action(1)  # DOWN to NEW GAME
                time.sleep(0.05)
                trainer.strategy_manager.execute_action(5)  # A to select NEW GAME
            
            time.sleep(0.1)
        
        # PHASE 4: Dialogue/cutscenes (Prof Oak, character selection, etc.)
        elif current_state == "dialogue":
            print(f"   💬 In dialogue/cutscene - advancing...")
            
            # Advance dialogue and cutscenes rapidly
            if attempt % 2 == 0:
                trainer.strategy_manager.execute_action(5)  # A to advance text
            else:
                trainer.strategy_manager.execute_action(7)  # START to skip
            
            time.sleep(0.08)
        
        # PHASE 5: Battle screens (if any early tutorial battles)
        elif current_state == "battle":
            print(f"   ⚔️ In battle - trying to escape/end quickly...")
            
            # Try to end battle quickly
            trainer.strategy_manager.execute_action(6)  # B (back/run)
            time.sleep(0.1)
            trainer.strategy_manager.execute_action(5)  # A (confirm)
            time.sleep(0.1)
        
        # FALLBACK: Unknown state or stuck
        else:
            # If stuck in unknown state for too long, try multiple approaches
            if state_stuck_counter > 10:
                print(f"   ❓ Stuck in {current_state}, trying recovery actions...")
                
                # Recovery sequence: try multiple buttons
                recovery_actions = [7, 5, 6, 5, 7]  # START, A, B, A, START
                for recovery_action in recovery_actions:
                    trainer.strategy_manager.execute_action(recovery_action)
                    time.sleep(0.05)
            else:
                # Default: try START and A buttons
                if attempt % 2 == 0:
                    trainer.strategy_manager.execute_action(7)  # START
                else:
                    trainer.strategy_manager.execute_action(5)  # A
                
                time.sleep(0.1)
    
    # Final verification
    final_screen = trainer.pyboy.screen.ndarray
    final_state = trainer.game_state_detector.detect_game_state(final_screen)
    final_variance = np.var(final_screen.astype(np.float32))
    
    print(f"   Final check: state = {final_state}, variance = {final_variance:.1f}")
    
    if final_state == "overworld":
        print("✅ Pokemon Crystal intro skip successful!")
        return True
    else:
        print(f"❌ Pokemon Crystal intro skip incomplete - stuck at {final_state}")
        print(f"   💡 Managed to progress through: intro → title → {final_state}")
        return False


def launch_pokemon_training(args):
    """Launch Pokemon Crystal training with specified configuration"""
    
    print("🎮 Pokemon Crystal RL Training Launcher")
    print("=" * 60)
    
    # Configure training based on arguments
    config = TrainingConfig(
        mode=TrainingMode.FAST_MONITORED,
        rom_path=args.rom_path,
        
        # Training settings
        max_actions=args.max_actions,
        llm_backend=LLMBackend.NONE if not args.enable_llm else LLMBackend.OPENAI,
        llm_interval=args.llm_interval,
        frames_per_action=args.frames_per_action,
        
        # Web monitoring settings
        enable_web=args.enable_web,
        web_host=args.web_host,
        web_port=args.web_port,
        
        # Screen capture settings
        capture_screens=args.enable_web,  # Only capture if web monitoring enabled
        capture_fps=args.capture_fps,
        screen_resize=(160, 144),  # Game Boy resolution
        
        # Performance settings
        headless=True,
        debug_mode=args.debug,
        log_level="INFO" if args.debug else "WARNING"
    )
    
    # Display configuration
    print(f"📋 Training Configuration:")
    print(f"   ROM: {config.rom_path}")
    print(f"   Mode: {config.mode.value}")
    print(f"   Max actions: {config.max_actions}")
    print(f"   LLM backend: {config.llm_backend.value}")
    print(f"   Web monitoring: {config.enable_web}")
    if config.enable_web:
        print(f"   Web URL: http://{config.web_host}:{config.web_port}")
    print(f"   Screen capture: {config.capture_screens}")
    print()
    
    # Create trainer
    print("🏗️ Initializing Pokemon Crystal trainer...")
    trainer = UnifiedPokemonTrainer(config)
    
    # Skip Pokemon Crystal intro sequence
    print("\n🎯 Preparing Pokemon Crystal for training...")
    
    # Check if we should try a fresh start (no save files)
    backed_up_files = []
    if hasattr(args, 'fresh_start') and args.fresh_start:
        print("🔄 Starting fresh - backing up save files...")
        backed_up_files = backup_save_files(config.rom_path)
    
    try:
        if not skip_pokemon_crystal_intro(trainer):
            # If first attempt failed and we haven't tried fresh start yet, try it
            if not backed_up_files and not (hasattr(args, 'fresh_start') and args.fresh_start):
                print("🔄 Intro skip failed - trying fresh start without save files...")
                backed_up_files = backup_save_files(config.rom_path)
                
                # Restart trainer to pick up the fresh state
                trainer._finalize_training()
                trainer = UnifiedPokemonTrainer(config)
                
                if not skip_pokemon_crystal_intro(trainer):
                    print("❌ Failed to skip Pokemon Crystal intro even with fresh start")
                    print("💡 This may happen if the ROM is not Pokemon Crystal or is corrupted")
                    return False
            else:
                print("❌ Failed to skip Pokemon Crystal intro")
                print("💡 This may happen if the ROM is not Pokemon Crystal or is corrupted")
                return False
    
    finally:
        # Restore save files if we backed them up
        if backed_up_files:
            print("🔄 Restoring save files...")
            restore_save_files(backed_up_files)
    
    # Set up web monitoring if enabled
    web_monitor = None
    bridge = None
    if config.enable_web:
        print(f"\n🌐 Setting up web monitoring...")
        try:
            web_monitor, bridge, monitor_thread = create_integrated_monitoring_system(
                trainer, host=config.web_host, port=config.web_port
            )
            print(f"✅ Web monitoring active: http://{config.web_host}:{config.web_port}")
        except Exception as e:
            print(f"⚠️ Web monitoring setup failed: {e}")
            print("   Continuing with training only...")
    
    # Set up graceful shutdown
    def signal_handler(sig, frame):
        print("\n⏸️ Shutting down Pokemon Crystal training...")
        
        if bridge:
            print("   Stopping bridge...")
            bridge.stop_bridge()
        
        if web_monitor:
            print("   Stopping web monitor...")
            web_monitor.stop_monitoring()
        
        print("   Finalizing trainer...")
        trainer._finalize_training()
        
        print("✅ Shutdown complete")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start training
    print(f"\n🚀 Starting Pokemon Crystal training...")
    if config.enable_web:
        print(f"📱 Open http://{config.web_host}:{config.web_port} to monitor live gameplay")
    print("🔄 Press Ctrl+C to stop training gracefully")
    print("=" * 60)
    print()
    
    try:
        trainer.start_training()
        
    except KeyboardInterrupt:
        signal_handler(signal.SIGINT, None)
        
    except Exception as e:
        print(f"❌ Training error: {e}")
        
        # Cleanup
        if bridge:
            bridge.stop_bridge()
        if trainer:
            trainer._finalize_training()
        
        return False
    
    return True


def main():
    """Main entry point with argument parsing"""
    
    parser = argparse.ArgumentParser(
        description="Pokemon Crystal RL Training Launcher",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Core training arguments
    parser.add_argument(
        "--rom-path", 
        default="../roms/pokemon_crystal.gbc",
        help="Path to Pokemon Crystal ROM file"
    )
    parser.add_argument(
        "--max-actions", 
        type=int, 
        default=10000,
        help="Maximum number of actions to perform"
    )
    parser.add_argument(
        "--frames-per-action", 
        type=int, 
        default=4,
        help="Number of emulator frames per action"
    )
    
    # LLM arguments
    parser.add_argument(
        "--enable-llm", 
        action="store_true",
        help="Enable LLM-based decision making (requires OpenAI API key)"
    )
    parser.add_argument(
        "--llm-interval", 
        type=int, 
        default=20,
        help="Interval between LLM decisions (in actions)"
    )
    
    # Web monitoring arguments
    parser.add_argument(
        "--enable-web", 
        action="store_true",
        help="Enable web-based monitoring dashboard"
    )
    parser.add_argument(
        "--web-host", 
        default="127.0.0.1",
        help="Web monitor host address"
    )
    parser.add_argument(
        "--web-port", 
        type=int, 
        default=5001,
        help="Web monitor port"
    )
    parser.add_argument(
        "--capture-fps", 
        type=int, 
        default=2,
        help="Screen capture frames per second (for web streaming)"
    )
    
    # Other arguments
    parser.add_argument(
        "--debug", 
        action="store_true",
        help="Enable debug logging"
    )
    
    args = parser.parse_args()
    
    # Validate ROM path
    if not os.path.exists(args.rom_path):
        print(f"❌ ROM file not found: {args.rom_path}")
        print("💡 Please ensure Pokemon Crystal ROM is available at the specified path")
        return False
    
    # Launch training
    success = launch_pokemon_training(args)
    
    if success:
        print("\n✅ Pokemon Crystal training completed successfully!")
    else:
        print("\n❌ Pokemon Crystal training failed!")
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
