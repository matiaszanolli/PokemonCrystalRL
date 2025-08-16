#!/usr/bin/env python3
"""
Pokemon Trainer v2.0 - Modular Entry Point

A clean, modular entry point for the refactored Pokemon Crystal RL Trainer.
"""

import argparse
import sys
import os

# Add the parent directory to Python path to find the trainer module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trainer import TrainingConfig, TrainingMode, LLMBackend, UnifiedPokemonTrainer


def main():
    """Main CLI interface for the modular trainer"""
    
    parser = argparse.ArgumentParser(
        description='Pokemon Crystal RL Trainer v2.0 - Modular Edition',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Training Modes:
  fast_monitored - Fast training with comprehensive monitoring (default)
  curriculum     - Progressive skill-based learning (5 stages)
  ultra_fast     - Rule-based maximum speed training
  custom         - User-defined configuration

LLM Models:
  smollm2:1.7b   - Ultra-fast, optimized (default)
  llama3.2:1b    - Fastest Llama
  llama3.2:3b    - Balanced speed/quality
  qwen2.5:3b     - Alternative fast option

Examples:
  # Fast monitored training with web interface
  python pokemon_trainer_v2.py --rom game.gbc --web --actions 2000
  
  # Ultra-fast rule-based training (no LLM)
  python pokemon_trainer_v2.py --rom game.gbc --mode ultra_fast --no-llm --actions 5000
  
  # Curriculum learning with different LLM
  python pokemon_trainer_v2.py --rom game.gbc --mode curriculum --model llama3.2:3b --episodes 20
  
  # Debug mode with detailed logging
  python pokemon_trainer_v2.py --rom game.gbc --debug --windowed --web
        """
    )
    
    # Required arguments
    parser.add_argument('--rom', required=True, help='Path to Pokemon Crystal ROM')
    
    # Training mode
    parser.add_argument('--mode', choices=[m.value for m in TrainingMode], 
                       default='fast_monitored', help='Training mode (default: fast_monitored)')
    
    # LLM settings
    parser.add_argument('--model', choices=[m.value for m in LLMBackend if m.value], 
                       default='smollm2:1.7b', help='LLM model to use (default: smollm2:1.7b)')
    parser.add_argument('--no-llm', action='store_true', help='Use rule-based training only')
    
    # Training parameters
    parser.add_argument('--actions', type=int, default=1000, help='Maximum actions (default: 1000)')
    parser.add_argument('--episodes', type=int, default=10, help='Maximum episodes (default: 10)')
    parser.add_argument('--llm-interval', type=int, default=10, help='Actions between LLM calls (default: 10)')
    parser.add_argument('--frames-per-action', type=int, default=24, 
                       help='Frames per action - 24=standard, 16=faster, 8=legacy (default: 24)')
    
    # Interface options
    parser.add_argument('--web', action='store_true', help='Enable web monitoring interface')
    parser.add_argument('--port', type=int, default=8080, help='Web interface port (default: 8080)')
    parser.add_argument('--no-capture', action='store_true', help='Disable screen capture')
    
    # Performance options
    parser.add_argument('--save-state', help='Save state file to load from')
    parser.add_argument('--windowed', action='store_true', help='Show game window (default: headless)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode with detailed logging')
    
    # Output options
    parser.add_argument('--no-save-stats', action='store_true', help='Disable saving statistics')
    parser.add_argument('--stats-file', default='training_stats.json', help='Statistics file name')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                       default='INFO', help='Logging level (default: INFO)')
    
    args = parser.parse_args()
    
    # Validate ROM file exists
    import os
    if not os.path.exists(args.rom):
        print(f"❌ ROM file not found: {args.rom}")
        print("Please check the path and ensure the ROM file exists.")
        return 1
    
    # Create configuration
    config = TrainingConfig(
        rom_path=args.rom,
        mode=TrainingMode(args.mode),
        llm_backend=None if args.no_llm else LLMBackend(args.model),
        max_actions=args.actions,
        max_episodes=args.episodes,
        llm_interval=args.llm_interval,
        frames_per_action=args.frames_per_action,
        headless=not args.windowed,
        debug_mode=args.debug,
        save_state_path=args.save_state,
        enable_web=args.web,
        web_port=args.port,
        capture_screens=not args.no_capture,
        save_stats=not args.no_save_stats,
        stats_file=args.stats_file,
        log_level=args.log_level
    )
    
    # Print startup banner
    print("🎮 Pokemon Crystal RL Trainer v2.0 - Modular Edition")
    print("=" * 60)
    print(f"📁 ROM: {args.rom}")
    print(f"🎯 Mode: {config.mode.value}")
    print(f"🤖 LLM: {config.llm_backend.value if config.llm_backend else 'None (rule-based)'}")
    print(f"⚡ Target: {config.max_actions} actions / {config.max_episodes} episodes")
    
    if config.enable_web:
        print(f"🌐 Web UI will be available at http://localhost:{config.web_port}")
    
    print()
    
    # Create and start trainer
    try:
        trainer = UnifiedPokemonTrainer(config)
        trainer.start_training()
        return 0
    except KeyboardInterrupt:
        print("\n⏸️ Training interrupted by user")
        return 0
    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
        return 1
    except Exception as e:
        print(f"❌ Training error: {e}")
        if config.debug_mode:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
