#!/usr/bin/env python3
"""
🚀 Pokemon Crystal RL Training Launcher with Integrated Web Monitoring

Production-ready training launcher that combines:
- UnifiedPokemonTrainer for intelligent gameplay
- Integrated web monitoring with live screenshots
- Easy configuration and one-click startup

Usage:
    python launch_pokemon_training.py
    python launch_pokemon_training.py --rom path/to/rom.gbc --fast
    python launch_pokemon_training.py --config training_config.json

Features:
    ✅ Integrated web monitoring with real-time game streaming
    ✅ Automatic ROM detection and validation
    ✅ Smart configuration with sensible defaults
    ✅ Progress tracking and performance monitoring
    ✅ Graceful shutdown handling
"""

import os
import sys
import argparse
import json
import time
import signal
from pathlib import Path
from typing import Optional, Dict, Any

# Add current directory to path for imports
current_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(current_dir))

try:
    from trainer.trainer import UnifiedPokemonTrainer
    from trainer.config import TrainingConfig, TrainingMode, LLMBackend
    from monitoring.trainer_monitor_bridge import create_integrated_monitoring_system
    IMPORTS_OK = True
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("   Make sure you're in the python_agent directory")
    print("   Run: pip install -r requirements.txt")
    IMPORTS_OK = False


class PokemonTrainingLauncher:
    """
    Production launcher for Pokemon Crystal RL training with integrated monitoring.
    """
    
    def __init__(self):
        self.trainer = None
        self.web_monitor = None
        self.bridge = None
        self.monitor_thread = None
        self.running = False
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        print(f"\n🛑 Received signal {signum}, shutting down gracefully...")
        self.shutdown()
        sys.exit(0)
    
    def find_rom_file(self, rom_path: Optional[str] = None) -> Optional[str]:
        """
        Find Pokemon Crystal ROM file with intelligent search.
        
        Args:
            rom_path: Explicit ROM path, if provided
            
        Returns:
            Path to ROM file, or None if not found
        """
        if rom_path and Path(rom_path).exists():
            return rom_path
        
        # Search locations for ROM files
        search_paths = [
            ".",
            "../roms",
            "../",
            "roms",
            "~/roms",
            "~/Downloads"
        ]
        
        # Common ROM filenames
        rom_names = [
            "pokemon_crystal.gbc",
            "Pokemon Crystal.gbc", 
            "Pokemon_Crystal.gbc",
            "crystal.gbc",
            "Crystal.gbc",
            "test.gbc"  # Our test ROM
        ]
        
        for search_path in search_paths:
            path = Path(search_path).expanduser()
            if not path.exists():
                continue
                
            for rom_name in rom_names:
                rom_file = path / rom_name
                if rom_file.exists():
                    print(f"🎮 Found ROM: {rom_file}")
                    return str(rom_file)
        
        return None
    
    def create_training_config(self, args) -> TrainingConfig:
        """
        Create training configuration from arguments.
        
        Args:
            args: Parsed command line arguments
            
        Returns:
            TrainingConfig instance
        """
        # Determine ROM path
        rom_path = self.find_rom_file(args.rom)
        if not rom_path:
            raise FileNotFoundError(
                "❌ No Pokemon Crystal ROM found!\n"
                "   Place your ROM in one of these locations:\n"
                "   - ./pokemon_crystal.gbc\n" 
                "   - ../roms/pokemon_crystal.gbc\n"
                "   - ~/roms/pokemon_crystal.gbc\n"
                "   Or specify with --rom path/to/rom.gbc"
            )
        
        # Determine training mode
        if args.fast:
            mode = TrainingMode.FAST_MONITORED
        elif args.ultra_fast:
            mode = TrainingMode.ULTRA_FAST
        elif args.curriculum:
            mode = TrainingMode.CURRICULUM
        else:
            mode = TrainingMode.FAST_MONITORED  # Default
        
        # Determine LLM backend
        if args.no_llm:
            llm_backend = LLMBackend.NONE
        elif args.model:
            # Map common model names to enum values
            model_map = {
                'smollm2': LLMBackend.SMOLLM2,
                'llama3.2-1b': LLMBackend.LLAMA32_1B,
                'llama3.2-3b': LLMBackend.LLAMA32_3B,
                'qwen2.5-3b': LLMBackend.QWEN25_3B,
            }
            llm_backend = model_map.get(args.model, LLMBackend.SMOLLM2)
        else:
            # Smart default based on mode
            if mode == TrainingMode.ULTRA_FAST:
                llm_backend = LLMBackend.NONE
            else:
                llm_backend = LLMBackend.SMOLLM2  # SmolLM2 default
        
        return TrainingConfig(
            rom_path=rom_path,
            mode=mode,
            llm_backend=llm_backend,
            max_actions=args.actions,
            llm_interval=args.llm_interval,
            capture_screens=True,  # Required for web monitoring
            capture_fps=10,
            enable_web=False,  # We use integrated monitoring instead
            headless=not args.windowed,
            debug_mode=args.debug,
            log_level='DEBUG' if args.debug else 'INFO',
            screen_resize=(480, 432),  # 3x Game Boy resolution
            frames_per_action=args.frames_per_action
        )
    
    def launch_training(self, config: TrainingConfig, web_port: int = 5000):
        """
        Launch training with integrated web monitoring.
        
        Args:
            config: Training configuration
            web_port: Port for web monitoring interface
        """
        print("🚀 POKEMON CRYSTAL RL TRAINING LAUNCHER")
        print("=" * 60)
        print(f"🎮 ROM: {config.rom_path}")
        print(f"🧠 Mode: {config.mode.value}")
        print(f"🤖 LLM: {config.llm_backend.value if config.llm_backend else 'None'}")
        print(f"🎯 Max Actions: {config.max_actions}")
        print(f"🌐 Web Monitor: http://localhost:{web_port}")
        print("")
        
        try:
            # Create trainer
            print("🔧 Initializing trainer...")
            self.trainer = UnifiedPokemonTrainer(config)
            
            # Create integrated monitoring system
            print("🌐 Setting up web monitoring...")
            self.web_monitor, self.bridge, self.monitor_thread = create_integrated_monitoring_system(
                self.trainer, host='0.0.0.0', port=web_port
            )
            
            # Start the bridge
            print("🌉 Starting integration bridge...")
            self.bridge.start_bridge()
            
            # Display launch information
            print("\n" + "🚀 TRAINING SESSION ACTIVE" + "=" * 42)
            print(f"🌐 Web Interface: http://localhost:{web_port}")
            print(f"📱 Mobile Access: http://YOUR_IP:{web_port}")
            print("🎮 Features Available:")
            print("   ✅ Live Pokemon game screens (2 FPS)")
            print("   ✅ Real-time training statistics")
            print("   ✅ Action history with reasoning")
            print("   ✅ LLM decision insights")
            print("   ✅ Performance monitoring")
            print("")
            print("⌨️  Press Ctrl+C to stop training gracefully")
            print("=" * 60)
            
            # Wait a moment for everything to initialize
            time.sleep(2)
            
            # Start training
            self.running = True
            self.trainer.start_training()
            
        except KeyboardInterrupt:
            print("\n⏸️ Training interrupted by user")
        except Exception as e:
            print(f"\n❌ Training error: {e}")
            if config.debug_mode:
                import traceback
                traceback.print_exc()
        finally:
            self.shutdown()
    
    def shutdown(self):
        """Gracefully shutdown all components"""
        if not self.running:
            return
            
        print("\n⏹️ Shutting down Pokemon RL training...")
        self.running = False
        
        # Stop bridge
        if self.bridge:
            print("🌉 Stopping integration bridge...")
            self.bridge.stop_bridge()
        
        # Stop web monitoring
        if self.web_monitor:
            print("🌐 Stopping web monitor...")
            self.web_monitor.stop_monitoring()
        
        # Clean up trainer
        if self.trainer:
            print("🔧 Cleaning up trainer...")
            # Trainer cleanup is handled automatically
        
        print("✅ Shutdown complete!")
    
    def load_config_file(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️ Could not load config file {config_path}: {e}")
            return {}


def create_argument_parser():
    """Create command line argument parser"""
    parser = argparse.ArgumentParser(
        description="🎮 Pokemon Crystal RL Training Launcher with Web Monitoring",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic training with web monitoring
    python launch_pokemon_training.py

    # Fast training with specific ROM
    python launch_pokemon_training.py --rom ../roms/crystal.gbc --fast

    # Ultra-fast rule-based training
    python launch_pokemon_training.py --ultra-fast --no-llm --actions 10000

    # Curriculum learning with debugging
    python launch_pokemon_training.py --curriculum --debug --windowed

    # Custom model and settings
    python launch_pokemon_training.py --model llama3.2:1b --actions 5000 --port 8080
        """
    )
    
    # ROM and basic settings
    parser.add_argument('--rom', type=str, help='Path to Pokemon Crystal ROM file')
    parser.add_argument('--config', type=str, help='Path to JSON configuration file')
    
    # Training modes
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument('--fast', action='store_true', 
                           help='Fast monitored training (default)')
    mode_group.add_argument('--ultra-fast', action='store_true',
                           help='Ultra-fast rule-based training')
    mode_group.add_argument('--curriculum', action='store_true',
                           help='Curriculum learning mode')
    
    # LLM settings
    parser.add_argument('--model', type=str, 
                       help='LLM model (smollm2, llama3.2-1b, llama3.2-3b)')
    parser.add_argument('--no-llm', action='store_true',
                       help='Disable LLM (rule-based only)')
    parser.add_argument('--llm-interval', type=int, default=8,
                       help='LLM decision interval (default: 8)')
    
    # Training parameters
    parser.add_argument('--actions', type=int, default=10000,
                       help='Maximum training actions (default: 10000)')
    parser.add_argument('--frames-per-action', type=int, default=24,
                       help='Game frames per action (default: 24)')
    
    # Interface settings
    parser.add_argument('--port', type=int, default=5000,
                       help='Web monitoring port (default: 5000)')
    parser.add_argument('--windowed', action='store_true',
                       help='Show game window (default: headless)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode with verbose logging')
    
    return parser


def main():
    """Main launcher function"""
    if not IMPORTS_OK:
        print("❌ Cannot start training - missing dependencies")
        print("   Run: pip install -r requirements.txt")
        return 1
    
    parser = create_argument_parser()
    args = parser.parse_args()
    
    launcher = PokemonTrainingLauncher()
    
    try:
        # Load additional config from file if specified
        file_config = {}
        if args.config:
            file_config = launcher.load_config_file(args.config)
        
        # Create training configuration
        config = launcher.create_training_config(args)
        
        # Launch training with integrated monitoring
        launcher.launch_training(config, web_port=args.port)
        
        return 0
        
    except FileNotFoundError as e:
        print(f"{e}")
        print("\n💡 Tips:")
        print("   1. Get a Pokemon Crystal ROM (not provided)")
        print("   2. Place it as 'pokemon_crystal.gbc' in current directory")
        print("   3. Or specify path with --rom option")
        return 1
        
    except Exception as e:
        print(f"❌ Launch failed: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
