#!/usr/bin/env python3
"""
LLM-Enhanced Pokemon Crystal RL Training Script

An advanced training script that combines:
- Local LLM integration for intelligent decision making
- Sophisticated reward function based on Pokemon game progress
- Memory map integration for game state analysis
- Web monitoring with LLM decision tracking
- DQN hybrid learning capabilities
"""

import sys
import os
import argparse
import time
import signal
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from trainer.pokemon_trainer import LLMPokemonTrainer
import base64
from io import BytesIO
from PIL import Image
import numpy as np


def validate_environment():
    """Validate that all required dependencies are available"""
    try:
        import pyboy
        import numpy as np
        import requests
        from PIL import Image
        print("✅ All dependencies available")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please install required dependencies:")
        print("  pip install pyboy numpy requests pillow")
        return False


def validate_rom(rom_path):
    """Validate ROM file exists and is readable"""
    if not os.path.exists(rom_path):
        print(f"❌ ROM file not found: {rom_path}")
        return False
    
    if not os.path.isfile(rom_path):
        print(f"❌ ROM path is not a file: {rom_path}")
        return False
    
    if not rom_path.lower().endswith(('.gbc', '.gb')):
        print(f"⚠️  Warning: ROM file doesn't have .gbc or .gb extension: {rom_path}")
    
    return True


def test_llm_connection(base_url="http://localhost:11434"):
    """Test if local LLM is available"""
    try:
        import requests
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        if response.status_code == 200:
            print("✅ Local LLM (Ollama) is available")
            return True
        else:
            print(f"⚠️  LLM server responded with status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("⚠️  Local LLM (Ollama) not available - will use fallback logic")
        print("   To enable LLM: Install Ollama and run: ollama pull smollm2:1.7b")
        return False
    except Exception as e:
        print(f"⚠️  Could not test LLM connection: {e}")
        return False


def run_training_loop(trainer, quiet=False):
    """Main training loop that coordinates all trainer components"""
    try:
        print("🎯 Starting main training loop...")
        
        # Initialize progress tracking
        last_progress_update = time.time()
        progress_interval = 30  # Print progress every 30 seconds
        last_reward_update = 0
        
        # Action mapping for PyBoy
        action_map = {
            'up': ['up'],
            'down': ['down'], 
            'left': ['left'],
            'right': ['right'],
            'a': ['a'],
            'b': ['b'],
            'start': ['start'],
            'select': ['select']
        }
        
        while trainer.running and trainer.actions_taken < trainer.max_actions:
            try:
                # Get current game state for decision making
                current_game_state = trainer.get_game_state()
                screen_analysis = trainer.analyze_screen()
                
                # Get next action from hybrid LLM/DQN system
                action, reasoning = trainer.get_next_action()
                
                # Execute the action
                if action in action_map:
                    for button in action_map[action]:
                        trainer.pyboy.button_press(button)
                        trainer.pyboy.tick()
                        trainer.pyboy.button_release(button)
                        trainer.pyboy.tick()
                    
                    # Additional ticks for action to take effect
                    for _ in range(8):
                        trainer.pyboy.tick()
                else:
                    # Unknown action, just tick the emulator
                    for _ in range(10):
                        trainer.pyboy.tick()
                
                # Get new game state after action
                new_game_state = trainer.get_game_state()
                
                # Calculate reward
                reward, breakdown = trainer.reward_calculator.calculate_reward(
                    trainer.previous_game_state, new_game_state
                )
                
                # Track reward and experience for DQN training
                if trainer.enable_dqn and trainer.dqn_agent:
                    # Store experience for DQN learning
                    trainer.dqn_agent.remember(
                        state=trainer.previous_game_state,
                        screen_analysis=screen_analysis,
                        action=action,
                        reward=reward,
                        next_state=new_game_state,
                        done=False  # Pokemon games don't really "end"
                    )
                    
                    # Train DQN periodically
                    if trainer.actions_taken % trainer.dqn_training_frequency == 0:
                        trainer.dqn_agent.train()
                    
                    # Save DQN model periodically
                    if trainer.actions_taken % trainer.dqn_save_frequency == 0:
                        dqn_path = os.path.join(trainer.data_dir, f'dqn_checkpoint_{trainer.actions_taken}.pt')
                        os.makedirs(trainer.data_dir, exist_ok=True)
                        trainer.dqn_agent.save_model(dqn_path)
                
                # Update trainer state
                trainer.actions_taken += 1
                trainer.total_reward += reward
                trainer.previous_game_state = new_game_state
                
                # Track recent actions for decision making
                trainer.recent_actions.append({
                    'action': action,
                    'reasoning': reasoning,
                    'reward': reward,
                    'timestamp': time.time()
                })
                
                # Keep only recent actions (last 10)
                if len(trainer.recent_actions) > 10:
                    trainer.recent_actions.pop(0)
                
                # Update statistics
                trainer.stats.update({
                    'actions_taken': trainer.actions_taken,
                    'training_time': time.time() - trainer.start_time,
                    'actions_per_second': trainer.actions_taken / (time.time() - trainer.start_time),
                    'total_reward': trainer.total_reward,
                    'player_level': new_game_state.get('player_level', 0),
                    'badges_total': new_game_state.get('badges_total', 0),
                    'last_reward_breakdown': breakdown
                })
                
                # Update web server with current screenshot and stats
                if trainer.enable_web and trainer.web_server:
                    try:
                        # Get current screen as base64 for web display
                        screen_image = Image.fromarray(trainer.pyboy.screen.ndarray)
                        buffer = BytesIO()
                        screen_image.save(buffer, format='PNG')
                        screenshot_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
                        
                        trainer.web_server.screenshot_data = screenshot_data
                        trainer.web_server.trainer_stats = trainer.stats.copy()
                    except Exception as e:
                        if not quiet:
                            print(f"⚠️ Web update failed: {e}")
                
                # Reward tracking
                if reward > 0:
                    trainer.last_positive_reward_action = trainer.actions_taken
                    trainer.actions_without_reward = 0
                    last_reward_update = trainer.actions_taken
                else:
                    trainer.actions_without_reward += 1
                
                # Anti-stuck mechanism
                if trainer.actions_without_reward > trainer.stuck_threshold:
                    if not quiet:
                        print(f"⚠️ Stuck detected! {trainer.actions_without_reward} actions without reward")
                        print(f"   Current location: ({new_game_state.get('player_x', 0)}, {new_game_state.get('player_y', 0)})")
                        print(f"   Map: {new_game_state.get('map_id', 0)}, Menu: {new_game_state.get('menu_state', 0)}")
                    
                    # Reset stuck counter and force exploration
                    trainer.actions_without_reward = 0
                
                # Progress updates
                current_time = time.time()
                if not quiet and (current_time - last_progress_update) >= progress_interval:
                    elapsed = current_time - trainer.start_time
                    actions_per_sec = trainer.actions_taken / elapsed if elapsed > 0 else 0
                    
                    print(f"\n📊 Progress Update (Action {trainer.actions_taken:,}/{trainer.max_actions:,})")
                    print(f"   ⏱️  Time: {elapsed:.1f}s ({actions_per_sec:.1f} actions/sec)")
                    print(f"   🎯 Total Reward: {trainer.total_reward:.2f}")
                    print(f"   🎮 Level: {new_game_state.get('player_level', 0)}, Badges: {new_game_state.get('badges_total', 0)}")
                    print(f"   📍 Position: ({new_game_state.get('player_x', 0)}, {new_game_state.get('player_y', 0)}) Map: {new_game_state.get('map_id', 0)}")
                    print(f"   🤖 LLM Decisions: {trainer.stats['llm_decision_count']}")
                    print(f"   💰 Money: {new_game_state.get('money', 0):,}")
                    
                    if trainer.enable_dqn and trainer.dqn_agent:
                        dqn_info = f"DQN ε={trainer.dqn_agent.epsilon:.3f}, Memory: {len(trainer.dqn_agent.memory)}"
                        print(f"   🧠 {dqn_info}")
                    
                    if reasoning and len(reasoning) > 0:
                        print(f"   💭 Last Decision: {action} - {reasoning[:60]}...")
                    
                    if reward != 0 or trainer.actions_taken - last_reward_update < 5:
                        print(f"   ⭐ Last Reward: {reward:.2f} - {breakdown}")
                    
                    last_progress_update = current_time
                    print()
                
                # Small delay to prevent overwhelming the system
                time.sleep(0.01)
                
            except KeyboardInterrupt:
                print("\n🛑 Training interrupted by user")
                trainer.running = False
                break
            except Exception as e:
                if not quiet:
                    print(f"⚠️ Error in training loop: {e}")
                    import traceback
                    traceback.print_exc()
                # Continue training despite errors
                continue
        
        # Training completed
        elapsed_time = time.time() - trainer.start_time
        actions_per_sec = trainer.actions_taken / elapsed_time if elapsed_time > 0 else 0
        
        print(f"\n🏁 Training Completed!")
        print(f"   Actions: {trainer.actions_taken:,}/{trainer.max_actions:,}")
        print(f"   Time: {elapsed_time:.1f}s ({actions_per_sec:.2f} actions/sec)")
        print(f"   Total Reward: {trainer.total_reward:.2f}")
        print(f"   LLM Decisions: {trainer.stats['llm_decision_count']}")
        
        final_state = trainer.get_game_state()
        print(f"   Final State: Level {final_state.get('player_level', 0)}, {final_state.get('badges_total', 0)} badges")
        
        # Save all training data
        print("💾 Saving training data...")
        trainer.save_training_data()
        print("✅ Training data saved")
        
        return 0
        
    except Exception as e:
        print(f"❌ Training loop failed: {e}")
        if not quiet:
            import traceback
            traceback.print_exc()
        return 1
    finally:
        # Ensure cleanup
        if trainer.pyboy:
            trainer.pyboy.stop()


def main():
    """Main entry point for LLM-enhanced Pokemon Crystal training"""
    parser = argparse.ArgumentParser(
        description="LLM-Enhanced Pokemon Crystal RL Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python llm_trainer.py --rom roms/pokemon_crystal.gbc
  python llm_trainer.py --rom roms/crystal.gbc --actions 10000 --llm-interval 30
  python llm_trainer.py --rom roms/crystal.gbc --no-web --no-dqn
  python llm_trainer.py --rom roms/crystal.gbc --llm-model "smollm2:3b"

Features:
  • Local LLM integration for intelligent decision making
  • Sophisticated reward system with anti-glitch validation
  • Real-time web monitoring dashboard
  • DQN hybrid learning capabilities
  • Memory map integration for game state analysis
        """
    )
    
    # Required arguments
    parser.add_argument("--rom", required=True, help="Path to Pokemon Crystal ROM file")
    
    # Training parameters
    parser.add_argument("--actions", type=int, default=5000, 
                       help="Maximum number of actions to execute (default: 5000)")
    parser.add_argument("--llm-interval", type=int, default=20,
                       help="Actions between LLM decisions (default: 20)")
    parser.add_argument("--llm-model", default="smollm2:1.7b",
                       help="LLM model name (default: smollm2:1.7b)")
    
    # Web monitoring
    parser.add_argument("--web-port", type=int, default=8080,
                       help="Web monitoring port (default: 8080)")
    parser.add_argument("--no-web", action="store_true",
                       help="Disable web monitoring")
    
    # DQN options
    parser.add_argument("--no-dqn", action="store_true",
                       help="Disable DQN hybrid learning")
    parser.add_argument("--dqn-model", help="Path to existing DQN model to load")
    
    # Advanced options
    parser.add_argument("--llm-base-url", default="http://localhost:11434",
                       help="LLM API base URL (default: http://localhost:11434)")
    parser.add_argument("--quiet", action="store_true",
                       help="Reduce output verbosity")
    
    args = parser.parse_args()
    
    if not args.quiet:
        print("🤖 LLM-Enhanced Pokemon Crystal RL Training")
        print("=" * 50)
    
    # Validate environment
    if not validate_environment():
        return 1
    
    # Validate ROM file
    if not validate_rom(args.rom):
        return 1
    
    # Test LLM connection (non-fatal)
    if not args.quiet:
        test_llm_connection(args.llm_base_url)
    
    # Print configuration
    if not args.quiet:
        print(f"\n📋 Training Configuration:")
        print(f"   ROM: {args.rom}")
        print(f"   Max Actions: {args.actions:,}")
        print(f"   LLM Model: {args.llm_model}")
        print(f"   LLM Interval: {args.llm_interval}")
        print(f"   Web Monitoring: {'Disabled' if args.no_web else f'http://localhost:{args.web_port}'}")
        print(f"   DQN Hybrid: {'Disabled' if args.no_dqn else 'Enabled'}")
        if args.dqn_model:
            print(f"   DQN Model: {args.dqn_model}")
        print()
    
    try:
        # Create trainer with configuration
        trainer = LLMPokemonTrainer(
            rom_path=args.rom,
            max_actions=args.actions,
            llm_model=args.llm_model,
            llm_interval=args.llm_interval,
            enable_web=not args.no_web,
            web_port=args.web_port,
            enable_dqn=not args.no_dqn,
            dqn_model_path=args.dqn_model
        )
        
        if not args.quiet:
            print("🚀 Starting training...")
            if not args.no_web:
                print(f"🌐 Web monitor: http://localhost:{args.web_port}")
            print("   Press Ctrl+C to stop training safely")
            print()
        
        # Start training
        trainer.initialize_pyboy()
        trainer.setup_web_server()
        
        # Run the main training loop
        return run_training_loop(trainer, args.quiet)
        
        return 0
        
    except KeyboardInterrupt:
        print("\n🛑 Training interrupted by user")
        return 0
    except Exception as e:
        print(f"❌ Training failed: {e}")
        if not args.quiet:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
