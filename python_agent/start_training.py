#!/usr/bin/env python3
"""
start_training.py - Pokemon Crystal RL Training Starter Script

This script helps set up and start training for the Pokemon Crystal RL agent.
It checks dependencies, verifies ROM availability, and provides multiple training options.
"""

import os
import sys
import argparse
import time
from pathlib import Path
from typing import Optional

def check_dependencies():
    """Check if all required dependencies are available"""
    print("🔍 Checking dependencies...")
    
    missing_deps = []
    available_deps = []
    
    # Check PyBoy
    try:
        from pyboy import PyBoy
        available_deps.append("✅ PyBoy: Available")
    except ImportError:
        missing_deps.append("❌ PyBoy: Not available (pip install pyboy)")
    
    # Check Ollama
    try:
        import ollama
        available_deps.append("✅ Ollama: Available")
    except ImportError:
        missing_deps.append("❌ Ollama: Not available (pip install ollama)")
    
    # Check Gymnasium
    try:
        import gymnasium as gym
        available_deps.append(f"✅ Gymnasium: Available (v{gym.__version__})")
    except ImportError:
        try:
            import gym
            available_deps.append(f"✅ OpenAI Gym: Available (v{gym.__version__})")
        except ImportError:
            missing_deps.append("❌ Gym/Gymnasium: Not available (pip install gymnasium)")
    
    # Check Stable Baselines3
    try:
        from stable_baselines3 import PPO
        available_deps.append("✅ Stable Baselines3: Available")
    except ImportError:
        missing_deps.append("❌ Stable Baselines3: Not available (pip install stable-baselines3[extra])")
    
    # Check NumPy
    try:
        import numpy as np
        available_deps.append(f"✅ NumPy: Available (v{np.__version__})")
    except ImportError:
        missing_deps.append("❌ NumPy: Not available (pip install numpy)")
    
    # Print results
    for dep in available_deps:
        print(f"  {dep}")
    
    if missing_deps:
        print("\n🚨 Missing dependencies:")
        for dep in missing_deps:
            print(f"  {dep}")
        print("\n💡 Install missing dependencies and try again.")
        return False
    
    print("✅ All dependencies available!")
    return True

def check_ollama_service():
    """Check if Ollama service is running and show available models"""
    print("\n🤖 Checking Ollama service...")
    
    try:
        import ollama
        models = ollama.list()
        model_list = models.get('models', [])
        
        if not model_list:
            print("⚠️ Ollama service is running but no models found")
            print("💡 Install a recommended model:")
            print("   ollama pull smollm2:1.7b      # Fast, efficient model")
            print("   ollama pull llama3.2:1b      # Alternative fast model")
            print("   ollama pull llama3.2:3b      # Better quality, slower")
            return False
        
        print(f"✅ Ollama service running with {len(model_list)} models:")
        
        # Check for recommended models
        recommended_models = ['smollm2:1.7b', 'llama3.2:1b', 'llama3.2:3b']
        available_recommended = []
        
        for model in model_list:
            name = model.get('name', 'unknown')
            size = model.get('size', 0)
            print(f"   • {name} ({size//1000000}MB)")
            
            if name in recommended_models:
                available_recommended.append(name)
        
        if available_recommended:
            print(f"✅ Recommended models available: {', '.join(available_recommended)}")
            return True
        else:
            print("⚠️ No recommended training models found")
            print("💡 Install a recommended model for better performance:")
            print("   ollama pull smollm2:1.7b")
            return True  # Still return True since Ollama is working
    
    except Exception as e:
        print(f"❌ Ollama service error: {e}")
        print("💡 Start Ollama service with: ollama serve")
        return False

def find_rom_files():
    """Search for Pokemon Crystal ROM files in common locations"""
    print("\n🔍 Searching for Pokemon Crystal ROM files...")
    
    # Common ROM file extensions and names
    rom_extensions = ['.gbc', '.gb']
    rom_names = ['pokemon_crystal', 'pokecrystal', 'crystal', 'pokémon_crystal']
    
    # Search locations
    search_paths = [
        '.',
        '..',
        '../..',
        './roms',
        '../roms',
        '../../roms',
        './ROMs',
        '../ROMs',
        '../../ROMs'
    ]
    
    found_roms = []
    
    for search_path in search_paths:
        if not os.path.exists(search_path):
            continue
            
        try:
            for root, dirs, files in os.walk(search_path):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in rom_extensions):
                        # Check if it's likely a Pokemon Crystal ROM
                        file_lower = file.lower()
                        if any(name in file_lower for name in rom_names):
                            full_path = os.path.join(root, file)
                            found_roms.append(full_path)
                        elif file_lower.endswith('.gbc') or file_lower.endswith('.gb'):
                            # Add any Game Boy ROM for user consideration
                            full_path = os.path.join(root, file)
                            found_roms.append(full_path)
        except PermissionError:
            continue
    
    if found_roms:
        print(f"✅ Found {len(found_roms)} potential ROM file(s):")
        for i, rom in enumerate(found_roms):
            print(f"   {i+1}. {rom}")
        return found_roms
    else:
        print("❌ No ROM files found in common locations")
        print("💡 Pokemon Crystal ROM file needed for training")
        print("   - ROM should be named something like 'pokemon_crystal.gbc'")
        print("   - Place in current directory, ../roms/, or specify path manually")
        return []

def find_save_states():
    """Search for existing save state files"""
    print("\n💾 Checking for save state files...")
    
    save_extensions = ['.state', '.ss1', '.sav']
    search_paths = ['.', '..', '../..']
    
    found_saves = []
    
    for search_path in search_paths:
        if not os.path.exists(search_path):
            continue
            
        try:
            for root, dirs, files in os.walk(search_path):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in save_extensions):
                        full_path = os.path.join(root, file)
                        found_saves.append(full_path)
        except PermissionError:
            continue
    
    if found_saves:
        print(f"✅ Found {len(found_saves)} save state file(s):")
        for save in found_saves:
            print(f"   • {save}")
        return found_saves
    else:
        print("ℹ️ No save state files found (will start from beginning)")
        return []

def show_training_options():
    """Display available training modes and options"""
    print("\n🎮 Available Training Modes:")
    print("="*50)
    
    training_modes = [
        ("fast_local", "Optimized local training with real-time capture", "Best for: Quick testing and development"),
        ("curriculum", "Progressive skill-based learning (5 stages)", "Best for: Systematic learning approach"),
        ("ultra_fast", "Rule-based maximum speed training", "Best for: Data collection and stress testing"),
        ("monitored", "Full analysis and monitoring with web UI", "Best for: Detailed analysis and debugging"),
    ]
    
    for mode, desc, best_for in training_modes:
        print(f"🔸 {mode.upper()}")
        print(f"   {desc}")
        print(f"   {best_for}")
        print()
    
    print("🤖 Available LLM Models:")
    try:
        import ollama
        models = ollama.list()
        model_list = models.get('models', [])
        
        for model in model_list:
            name = model.get('name', 'unknown')
            size = model.get('size', 0)
            print(f"   • {name} ({size//1000000}MB)")
    except:
        print("   ❌ Could not list models")
    
    print("\n🚀 Quick Start Commands:")
    print("="*50)
    print("# Fast training with SmolLM (recommended for testing)")
    print("python start_training.py --mode fast_local --model smollm2:1.7b --actions 1000 --web")
    print()
    print("# Curriculum learning with monitoring")
    print("python start_training.py --mode curriculum --model llama3.2:3b --episodes 20 --web")
    print()
    print("# Ultra-fast rule-based training (no LLM)")
    print("python start_training.py --mode ultra_fast --no-llm --actions 5000")
    print()

def create_training_command(args):
    """Create the training command based on arguments"""
    if not args.rom:
        print("❌ ROM file is required for training")
        return None
    
    cmd_parts = [
        "python",
        "run_pokemon_trainer.py",
        f"--rom {args.rom}",
        f"--mode {args.mode}",
        f"--actions {args.actions}",
        f"--episodes {args.episodes}"
    ]
    
    if args.model and not args.no_llm:
        cmd_parts.append(f"--model {args.model}")
    elif args.no_llm:
        cmd_parts.append("--no-llm")
    
    if args.web:
        cmd_parts.append("--web")
        if args.port != 8080:
            cmd_parts.append(f"--port {args.port}")
    
    if args.save_state:
        cmd_parts.append(f"--save-state {args.save_state}")
    
    if args.windowed:
        cmd_parts.append("--windowed")
    
    if args.debug:
        cmd_parts.append("--debug")
    
    return " ".join(cmd_parts)

def main():
    parser = argparse.ArgumentParser(
        description="Pokemon Crystal RL Training Starter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python start_training.py                           # Check dependencies and show options
  python start_training.py --setup                  # Interactive setup
  python start_training.py --rom ../crystal.gbc --mode fast_local --web
  python start_training.py --rom crystal.gbc --mode curriculum --episodes 50
        """
    )
    
    # Setup and check options
    parser.add_argument('--setup', action='store_true', 
                       help='Interactive setup mode')
    parser.add_argument('--check-only', action='store_true',
                       help='Only check dependencies and ROM availability')
    
    # Training options
    parser.add_argument('--rom', type=str,
                       help='Path to Pokemon Crystal ROM file')
    parser.add_argument('--mode', choices=['fast_local', 'curriculum', 'ultra_fast', 'monitored'],
                       default='fast_local', help='Training mode')
    parser.add_argument('--model', type=str, default='smollm2:1.7b',
                       help='LLM model to use')
    parser.add_argument('--no-llm', action='store_true',
                       help='Use rule-based training only')
    
    # Training parameters
    parser.add_argument('--actions', type=int, default=1000,
                       help='Maximum actions per episode')
    parser.add_argument('--episodes', type=int, default=10,
                       help='Maximum episodes')
    
    # Interface options
    parser.add_argument('--web', action='store_true',
                       help='Enable web monitoring interface')
    parser.add_argument('--port', type=int, default=8080,
                       help='Web interface port')
    parser.add_argument('--windowed', action='store_true',
                       help='Show emulator window')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode')
    
    # Other options
    parser.add_argument('--save-state', type=str,
                       help='Save state file to load from')
    
    args = parser.parse_args()
    
    print("🎮 Pokemon Crystal RL Training Starter")
    print("="*50)
    
    # Check dependencies
    if not check_dependencies():
        return 1
    
    # Check Ollama
    ollama_ok = check_ollama_service()
    
    # Find ROM files
    found_roms = find_rom_files()
    
    # Find save states
    found_saves = find_save_states()
    
    # If only checking, exit here
    if args.check_only:
        print("\n✅ Dependency check completed!")
        return 0
    
    # If no ROM specified and none found, show options and exit
    if not args.rom and not found_roms:
        print("\n⚠️ No ROM file available for training")
        print("To start training, you need a Pokemon Crystal ROM file.")
        print("Please obtain a ROM file and either:")
        print("1. Place it in the current directory")
        print("2. Specify its path with --rom /path/to/rom.gbc")
        print("\nOnce you have a ROM file, run:")
        print("python start_training.py --rom your_rom.gbc --mode fast_local --web")
        show_training_options()
        return 1
    
    # Interactive setup mode
    if args.setup:
        print("\n🛠️ Interactive Setup Mode")
        print("=" * 30)
        
        # ROM selection
        if not args.rom:
            if found_roms:
                print("Select ROM file:")
                for i, rom in enumerate(found_roms):
                    print(f"  {i+1}. {rom}")
                
                try:
                    choice = int(input("Enter choice (1-{}): ".format(len(found_roms))))
                    if 1 <= choice <= len(found_roms):
                        args.rom = found_roms[choice-1]
                    else:
                        print("Invalid choice")
                        return 1
                except ValueError:
                    print("Invalid input")
                    return 1
            else:
                args.rom = input("Enter ROM file path: ").strip()
        
        # Training mode selection
        print("\nSelect training mode:")
        modes = ['fast_local', 'curriculum', 'ultra_fast', 'monitored']
        for i, mode in enumerate(modes):
            print(f"  {i+1}. {mode}")
        
        try:
            choice = int(input("Enter choice (1-4): "))
            if 1 <= choice <= 4:
                args.mode = modes[choice-1]
        except ValueError:
            print("Using default: fast_local")
        
        # Web interface
        web_choice = input("Enable web interface? (y/n): ").strip().lower()
        args.web = web_choice in ['y', 'yes']
        
        print(f"\n✅ Setup complete! ROM: {args.rom}, Mode: {args.mode}, Web: {args.web}")
    
    # Auto-select ROM if not specified but found
    if not args.rom and found_roms:
        args.rom = found_roms[0]  # Use first found ROM
        print(f"🎯 Using ROM: {args.rom}")
    
    # Auto-select save state if available
    if not args.save_state and found_saves:
        # Prefer save states with "intro" or "start" in name
        preferred_saves = [s for s in found_saves if any(word in s.lower() for word in ['intro', 'start', 'begin'])]
        if preferred_saves:
            args.save_state = preferred_saves[0]
        else:
            args.save_state = found_saves[0]
        print(f"💾 Using save state: {args.save_state}")
    
    # Create and show training command
    if args.rom:
        cmd = create_training_command(args)
        if cmd:
            print(f"\n🚀 Starting training with command:")
            print(f"   {cmd}")
            print("\n" + "="*50)
            
            # Import and run the trainer
            try:
                from scripts.pokemon_trainer import UnifiedPokemonTrainer, TrainingConfig, TrainingMode, LLMBackend
                
                # Create config from arguments
                config = TrainingConfig(
                    rom_path=args.rom,
                    mode=TrainingMode(args.mode),
                    llm_backend=None if args.no_llm else LLMBackend(args.model),
                    max_actions=args.actions,
                    max_episodes=args.episodes,
                    headless=not args.windowed,
                    debug_mode=args.debug,
                    save_state_path=args.save_state,
                    enable_web=args.web,
                    web_port=args.port,
                )
                
                # Create and start trainer
                trainer = UnifiedPokemonTrainer(config)
                trainer.start_training()
                
            except KeyboardInterrupt:
                print("\n⏸️ Training interrupted by user")
            except Exception as e:
                print(f"\n❌ Training error: {e}")
                print("💡 Try running the command manually or check the ROM file")
                return 1
    else:
        show_training_options()
        print("\n💡 Specify a ROM file to start training!")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
