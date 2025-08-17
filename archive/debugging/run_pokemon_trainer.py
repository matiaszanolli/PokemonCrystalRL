#!/usr/bin/env python3
"""
Pokemon Crystal RL Trainer v2.0 - Production Edition

Unified entry point for the comprehensive Pokemon Crystal RL training system.
This wrapper provides access to the full production-grade trainer with:

- Multiple training modes (ultra-fast, synchronized, curriculum)
- Local LLM integration with zero API costs
- Real-time web monitoring with 10x optimized streaming
- Professional logging and crash-free operation
- Production-grade error handling and recovery

Usage Examples:
  # Quick production demo
  python run_pokemon_trainer.py --rom crystal.gbc --quick
  
  # Full production training
  python run_pokemon_trainer.py --rom crystal.gbc --production
  
  # Content creation mode
  python run_pokemon_trainer.py --rom crystal.gbc --windowed --web --hq
  
  # Ultra-fast training
  python run_pokemon_trainer.py --rom crystal.gbc --mode ultra_fast --no-llm --actions 10000
"""

import os
import sys
import time
import signal
import argparse
import numpy as np
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Tuple, Dict, Any

from pokemon_crystal_rl.trainer.trainer import UnifiedPokemonTrainer
from pokemon_crystal_rl.trainer.config import TrainingConfig, TrainingMode, LLMBackend
from pokemon_crystal_rl.monitoring.trainer_monitor_bridge import create_integrated_monitoring_system


def backup_save_files(rom_path: str) -> List[Tuple[str, str]]:
    """Backup save files to allow fresh start.
    
    Args:
        rom_path: Path to the ROM file
        
    Returns:
        List of (original_path, backup_path) tuples
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


def restore_save_files(backed_up_files: List[Tuple[str, str]]):
    """Restore previously backed up save files.
    
    Args:
        backed_up_files: List of (original_path, backup_path) tuples
    """
    for original_file, backup_file in backed_up_files:
        if os.path.exists(backup_file):
            try:
                shutil.move(backup_file, original_file)
                print(f"   📁 Restored: {os.path.basename(original_file)}")
            except Exception as e:
                print(f"   ⚠️ Failed to restore {original_file}: {e}")


def skip_pokemon_crystal_intro(trainer: UnifiedPokemonTrainer, max_attempts: int = 100,
                           allow_continue: bool = True, fresh_start: bool = False) -> bool:
    """Skip Pokemon Crystal intro sequence to reach controllable gameplay.
    
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
        allow_continue: Whether to try CONTINUE option first
        fresh_start: Whether this is a fresh start attempt
        
    Returns:
        bool: True if overworld gameplay was reached
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


def create_argument_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description='Pokemon Crystal RL Trainer v2.0 - Production Edition',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
🎯 TRAINING MODES:
  fast_monitored - Frame-synchronized with comprehensive monitoring (default)
  curriculum     - Progressive skill-based learning with adaptive difficulty
  ultra_fast     - Rule-based maximum speed training (800+ actions/sec)
  custom         - User-defined configuration

🤖 LLM MODELS:
  smollm2:1.7b   - Ultra-fast, optimized for real-time (default)
  llama3.2:1b    - Fastest Llama model
  llama3.2:3b    - Balanced speed/quality
  qwen2.5:3b     - Alternative high-quality option

🚀 PRODUCTION EXAMPLES:
  # Quick production demo (500 actions, rule-based)
  python run_pokemon_trainer.py --rom crystal.gbc --production --quick
  
  # Full production run with LLM and monitoring
  python run_pokemon_trainer.py --rom crystal.gbc --production --web --actions 5000
  
  # Ultra-fast training for rapid iteration
  python run_pokemon_trainer.py --rom crystal.gbc --mode ultra_fast --no-llm --actions 10000
  
  # Content creation mode with high-quality streaming
  python run_pokemon_trainer.py --rom crystal.gbc --windowed --web --hq --actions 2000
  
  # Research mode with curriculum learning
  python run_pokemon_trainer.py --rom crystal.gbc --mode curriculum --episodes 20 --debug
        """
    )
    
    # === CORE ARGUMENTS ===
    parser.add_argument('--rom', required=True, help='Path to Pokemon Crystal ROM file')
    
    # Production mode shortcuts
    parser.add_argument('--production', action='store_true', 
                       help='Enable production mode with optimal settings')
    parser.add_argument('--quick', action='store_true',
                       help='Quick demo run (500 actions, ~3 minutes)')
    
    # === TRAINING CONFIGURATION ===
    parser.add_argument('--mode', choices=[m.value for m in TrainingMode], 
                       default='fast_monitored', help='Training mode (default: fast_monitored)')
    
    # Training parameters
    parser.add_argument('--actions', type=int, default=1000, 
                       help='Maximum actions to execute (default: 1000)')
    parser.add_argument('--episodes', type=int, default=10, 
                       help='Maximum episodes to run (default: 10)')
    parser.add_argument('--frames-per-action', type=int, default=24, 
                       help='Frames per action: 24=standard, 16=faster, 8=legacy (default: 24)')
    
    # === LLM CONFIGURATION ===
    parser.add_argument('--model', choices=[m.value for m in LLMBackend if m.value], 
                       default='smollm2:1.7b', help='LLM model (default: smollm2:1.7b)')
    parser.add_argument('--no-llm', action='store_true', 
                       help='Disable LLM, use rule-based decisions only')
    parser.add_argument('--llm-interval', type=int, default=10, 
                       help='Actions between LLM calls (default: 10)')
    
    # === INTERFACE OPTIONS ===
    parser.add_argument('--web', action='store_true', 
                       help='Enable real-time web monitoring interface')
    parser.add_argument('--port', type=int, default=8080, 
                       help='Web interface port (default: 8080)')
    parser.add_argument('--windowed', action='store_true', 
                       help='Show game window (default: headless)')
    
    # === CAPTURE & STREAMING ===
    parser.add_argument('--no-capture', action='store_true', 
                       help='Disable screen capture (faster but no monitoring)')
    parser.add_argument('--capture-fps', type=int, default=10,
                       help='Screen capture frame rate (default: 10 FPS)')
    parser.add_argument('--hq', '--high-quality', action='store_true', dest='high_quality',
                       help='High-quality streaming (320x288, good for content creation)')
    parser.add_argument('--low-quality', action='store_true',
                       help='Low-quality streaming (240x216, faster performance)')
    
    # === PERFORMANCE & DEBUGGING ===
    parser.add_argument('--save-state', help='Save state file to load (auto-detects ROM.state)')
    parser.add_argument('--debug', action='store_true', 
                       help='Enable debug mode with detailed logging')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run performance benchmarks during training')
    
    # === OUTPUT & STATISTICS ===
    parser.add_argument('--no-save-stats', action='store_true', 
                       help='Disable saving training statistics')
    parser.add_argument('--stats-file', 
                       default=f'training_stats_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json',
                       help='Statistics output file')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                       default='INFO', help='Logging verbosity (default: INFO)')
    
    # === ADVANCED OPTIONS ===
    parser.add_argument('--curriculum-stages', type=int, default=5,
                       help='Number of curriculum stages (curriculum mode only)')
    parser.add_argument('--auto-save', action='store_true',
                       help='Automatically save progress every 1000 actions')
    parser.add_argument('--validate', action='store_true',
                       help='Run system validation before training')
    
    return parser


def launch_pokemon_training(args: argparse.Namespace) -> bool:
    """Launch Pokemon Crystal training with specified configuration.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        bool: True if training completed successfully
    """
    print("🎮 POKEMON CRYSTAL RL TRAINING LAUNCHER")
    print("=" * 60)
    
    # Apply production mode defaults
    if args.production:
        if not args.quick:
            args.actions = 5000  # Extended session
            args.episodes = 3
        args.web = True
        args.llm_interval = 15
        
    # Apply quick mode settings
    if args.quick:
        args.actions = 500  # Quick demo
        args.episodes = 1
        if not args.web:
            args.web = True  # Enable monitoring for demos
    
    # Auto-enable web for windowed mode (content creation)
    if args.windowed and not args.no_capture:
        args.web = True
    
    # High quality settings for content creation
    if args.high_quality:
        args.capture_fps = 15
        args.web = True
    
    # Create production-optimized configuration
    config = TrainingConfig(
        # Core settings
        rom_path=args.rom,
        mode=TrainingMode(args.mode),
        llm_backend=None if args.no_llm else LLMBackend(args.model),
        
        # Training parameters
        max_actions=args.actions,
        max_episodes=args.episodes,
        llm_interval=args.llm_interval,
        frames_per_action=args.frames_per_action,
        
        # Performance settings
        headless=not args.windowed,
        debug_mode=args.debug,
        save_state_path=args.save_state,
        
        # Web interface with optimizations
        enable_web=args.web,
        web_port=args.port,
        web_host="localhost",
        
        # Screen capture with high quality for monitoring
        capture_screens=not args.no_capture,
        capture_fps=args.capture_fps,
        screen_resize=(320, 288) if not args.low_quality else (240, 216),
        
        # Stats and logging
        save_stats=not args.no_save_stats,
        stats_file=args.stats_file,
        log_level=args.log_level
    )
    
    # Display configuration
    print(f"📋 Training Configuration:")
    print(f"   ROM: {config.rom_path}")
    print(f"   Mode: {config.mode.value}")
    print(f"   Max actions: {config.max_actions}")
    print(f"   LLM backend: {config.llm_backend.value if config.llm_backend else 'None (rule-based)'}")
    print(f"   Web monitoring: {config.enable_web}")
    if config.enable_web:
        print(f"   Web URL: http://{config.web_host}:{config.web_port}")
    print(f"   Screen capture: {config.capture_screens}")
    print()
    
    # Create trainer instance
    trainer = UnifiedPokemonTrainer(config)
    
    # Skip Pokemon Crystal intro sequence
    print("\n🎯 Preparing Pokemon Crystal for training...")
    
    # Check if we should try a fresh start (no save files)
    backed_up_files = []
    if args.validate:
        print("🔄 Starting fresh - backing up save files...")
        backed_up_files = backup_save_files(config.rom_path)
    
    try:
        if not skip_pokemon_crystal_intro(trainer):
            # If first attempt failed and we haven't tried fresh start yet, try it
            if not backed_up_files and not args.validate:
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
        return True
        
    except KeyboardInterrupt:
        signal_handler(signal.SIGINT, None)
        return True
        
    except Exception as e:
        print(f"❌ Training error: {e}")
        if config.debug_mode:
            import traceback
            traceback.print_exc()
        
        # Cleanup
        if bridge:
            bridge.stop_bridge()
        if web_monitor:
            web_monitor.stop_monitoring()
        if trainer:
            trainer._finalize_training()
        
        return False


def main() -> int:
    """Main entry point with argument parsing.
    
    Returns:
        int: Exit code (0 for success, 1 for failure)
    """
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Validate ROM path
    if not os.path.exists(args.rom):
        print(f"❌ ROM file not found: {args.rom}")
        print("💡 Please ensure Pokemon Crystal ROM is available at the specified path")
        return 1
    
    # Launch training
    success = launch_pokemon_training(args)
    
    if success:
        print("\n✅ Pokemon Crystal training completed successfully!")
        return 0
    else:
        print("\n❌ Pokemon Crystal training failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
