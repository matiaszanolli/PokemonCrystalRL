#!/usr/bin/env python3
"""
enhanced_monitored_training.py - Pokemon Crystal RL training with Advanced Web Monitoring

This script provides comprehensive Pokemon Crystal RL training with the new advanced monitoring system:
- Real-time web dashboard with charts and analytics
- Full PyBoy environment integration with monitoring
- Enhanced LLM agent with visual analysis
- Comprehensive performance tracking
- Screenshot streaming and analysis
- Text detection and frequency analysis
- System performance monitoring
"""

import time
import threading
import signal
import sys
import numpy as np
import argparse
import os
from collections import deque
from typing import Dict, List, Any, Optional
import json

from pyboy_env import PyBoyPokemonCrystalEnv
from enhanced_llm_agent import EnhancedLLMPokemonAgent
from monitoring_client import MonitoredTraining
from vision_processor import PokemonVisionProcessor


class AdvancedTrainingSession:
    """
    Advanced Pokemon Crystal training session with comprehensive monitoring
    """
    
    def __init__(self, 
                 rom_path: str,
                 save_state_path: str = None,
                 episodes: int = 100,
                 max_steps_per_episode: int = 2000,
                 llm_interval: int = 10,        # LLM decision every N steps
                 visual_interval: int = 20,     # Visual analysis every N steps
                 screenshot_interval: int = 5,  # Screenshot update every N steps
                 enable_vision: bool = True,
                 enable_monitoring: bool = True,
                 monitor_server_url: str = "http://localhost:5000"):
        
        self.rom_path = rom_path
        self.save_state_path = save_state_path
        self.episodes = episodes
        self.max_steps_per_episode = max_steps_per_episode
        self.llm_interval = llm_interval
        self.visual_interval = visual_interval
        self.screenshot_interval = screenshot_interval
        self.enable_vision = enable_vision
        self.enable_monitoring = enable_monitoring
        self.monitor_server_url = monitor_server_url
        
        # Validate ROM path
        if not os.path.exists(rom_path):
            raise FileNotFoundError(f"ROM file not found: {rom_path}")
        
        # Initialize environment with monitoring
        print("🎮 Initializing PyBoy environment with monitoring...")
        self.env = PyBoyPokemonCrystalEnv(
            rom_path=rom_path,
            save_state_path=save_state_path,
            max_steps=max_steps_per_episode,
            headless=True,
            debug_mode=False,
            enable_monitoring=enable_monitoring,
            monitor_server_url=monitor_server_url
        )
        
        # Initialize enhanced LLM agent with monitoring
        print("🤖 Initializing Enhanced LLM Agent with monitoring...")
        self.agent = EnhancedLLMPokemonAgent(
            use_vision=enable_vision,
            enable_monitoring=enable_monitoring,
            monitor_server_url=monitor_server_url
        )
        
        # Initialize vision processor if enabled
        self.vision_processor = None
        if enable_vision:
            print("👁️ Initializing vision processor...")
            try:
                self.vision_processor = PokemonVisionProcessor()
                print("✓ Vision processor initialized")
            except Exception as e:
                print(f"⚠️ Vision processor failed to initialize: {e}")
                self.enable_vision = False
        
        # Training state
        self.training_active = False
        self.current_episode = 0
        self.start_time = None
        
        # Performance tracking
        self.episode_rewards = []
        self.episode_steps = []
        self.total_steps = 0
        self.llm_decisions = 0
        self.visual_analyses = 0
        
        # Action history for context
        self.action_history = deque(maxlen=50)
        
        # Signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        print("✅ Advanced training session initialized!")
        
        # Check monitoring availability
        if enable_monitoring:
            if self.env.is_monitoring_available():
                print("📊 Monitoring system is available")
            else:
                print("⚠️ Monitoring system is not available - running without web monitoring")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        print(f"\n🛑 Received signal {signum}, shutting down gracefully...")
        self.stop()
        sys.exit(0)
    
    def start_training(self):
        """Start the training session"""
        print(f"\n🚀 Starting Advanced Pokemon Crystal RL Training")
        print(f"📦 ROM: {self.rom_path}")
        print(f"💾 Save state: {self.save_state_path or 'None (starting from beginning)'}")
        print(f"🎯 Episodes: {self.episodes}")
        print(f"📊 Max steps per episode: {self.max_steps_per_episode}")
        print(f"🤖 LLM decision interval: {self.llm_interval} steps")
        print(f"👁️ Visual analysis: {'Enabled' if self.enable_vision else 'Disabled'}")
        print(f"📡 Web monitoring: {'Enabled' if self.enable_monitoring else 'Disabled'}")
        
        if self.enable_monitoring and self.env.is_monitoring_available():
            print(f"🌐 Monitor dashboard: {self.monitor_server_url}")
        
        self.training_active = True
        self.start_time = time.time()
        
        try:
            self._run_training_loop()
        except KeyboardInterrupt:
            print("\n🛑 Training interrupted by user")
        except Exception as e:
            print(f"\n❌ Training failed: {e}")
            raise
        finally:
            self.stop()
    
    def _run_training_loop(self):
        """Main training loop"""
        
        for episode in range(1, self.episodes + 1):
            self.current_episode = episode
            print(f"\n🎮 Episode {episode}/{self.episodes}")
            
            # Reset environment and get initial state
            obs, info = self.env.reset()
            episode_reward = 0
            episode_steps = 0
            done = False
            
            # Episode timing
            episode_start_time = time.time()
            
            while not done and episode_steps < self.max_steps_per_episode:
                step_start_time = time.time()
                
                # Get game state for decision making
                game_state = self.env.get_game_state()
                
                # Decide next action
                action = self._decide_action(episode_steps, game_state)
                
                # Execute action in environment
                obs, reward, terminated, truncated, info = self.env.step(action)
                
                # Update tracking
                episode_reward += reward
                episode_steps += 1
                self.total_steps += 1
                
                # Track action history
                action_str = self.env.action_map.get(action, str(action))
                self.action_history.append(action_str)
                
                # Handle visual analysis and text detection
                if self.enable_vision and episode_steps % self.visual_interval == 0:
                    self._perform_visual_analysis(game_state)
                
                # Update monitoring with screenshot
                if self.enable_monitoring and episode_steps % self.screenshot_interval == 0:
                    screenshot = self.env.get_screenshot()
                    if screenshot is not None and screenshot.size > 0:
                        if hasattr(self.env, 'monitor') and self.env.monitor:
                            self.env.monitor.update_screenshot(screenshot)
                
                # Check if episode is done
                done = terminated or truncated
                
                # Performance logging
                step_time = time.time() - step_start_time
                if episode_steps % 100 == 0:  # Log every 100 steps
                    actions_per_sec = 1.0 / step_time if step_time > 0 else 0
                    print(f"  Step {episode_steps}: Reward={reward:.2f}, "
                          f"Action={action_str}, Speed={actions_per_sec:.1f} actions/sec")
            
            # Episode completed
            episode_time = time.time() - episode_start_time
            self.episode_rewards.append(episode_reward)
            self.episode_steps.append(episode_steps)
            
            # Send episode end notification
            self.env.send_episode_end(success=(episode_reward > 0))
            
            # Episode summary
            avg_reward = sum(self.episode_rewards) / len(self.episode_rewards)
            avg_steps = sum(self.episode_steps) / len(self.episode_steps)
            
            print(f"📊 Episode {episode} complete:")
            print(f"   Reward: {episode_reward:.2f} (avg: {avg_reward:.2f})")
            print(f"   Steps: {episode_steps} (avg: {avg_steps:.1f})")
            print(f"   Duration: {episode_time:.1f}s")
            print(f"   LLM Decisions: {self.llm_decisions}")
            print(f"   Visual Analyses: {self.visual_analyses}")
            
            # Brief pause between episodes
            time.sleep(0.5)
        
        # Training completed
        self._print_final_summary()
    
    def _decide_action(self, step: int, game_state: Dict[str, Any]) -> int:
        """Decide next action using LLM at intervals or fallback"""
        
        # Use LLM for decision making at specified intervals
        if step % self.llm_interval == 0 or step == 0:
            return self._get_llm_action(game_state, step)
        else:
            return self._get_fallback_action(game_state)
    
    def _get_llm_action(self, game_state: Dict[str, Any], step: int) -> int:
        """Get action from LLM with full context"""
        try:
            start_time = time.time()
            
            # Get screenshot for visual context
            screenshot = None
            if self.enable_vision and self.vision_processor:
                screenshot = self.env.get_screenshot()
            
            # Get LLM decision
            action = self.agent.decide_next_action(
                state=game_state,
                screenshot=screenshot,
                recent_history=list(self.action_history)
            )
            
            decision_time = time.time() - start_time
            self.llm_decisions += 1
            
            if step % 50 == 0:  # Periodic logging
                print(f"🤖 LLM decision: {self.env.action_map.get(action, action)} "
                      f"(took {decision_time:.2f}s)")
            
            return action
            
        except Exception as e:
            print(f"⚠️ LLM decision failed: {e}")
            return self._get_fallback_action(game_state)
    
    def _get_fallback_action(self, game_state: Dict[str, Any]) -> int:
        """Simple fallback action when LLM is not used"""
        # Simple exploration strategy: favor movement and interaction
        fallback_actions = [1, 2, 3, 4, 5]  # UP, DOWN, LEFT, RIGHT, A
        weights = [0.2, 0.2, 0.2, 0.2, 0.2]  # Equal probability
        
        return np.random.choice(fallback_actions, p=weights)
    
    def _perform_visual_analysis(self, game_state: Dict[str, Any]):
        """Perform visual analysis and text detection"""
        if not self.vision_processor:
            return
        
        try:
            screenshot = self.env.get_screenshot()
            if screenshot is not None and screenshot.size > 0:
                visual_context = self.vision_processor.process_screenshot(screenshot)
                self.visual_analyses += 1
                
                # Send text updates to monitoring
                if visual_context.detected_text and self.enable_monitoring:
                    for text_obj in visual_context.detected_text[:3]:  # First 3 text objects
                        self.env.send_text_update(text_obj.text, "dialogue")
                
                # Log interesting visual findings
                if len(visual_context.detected_text) > 0:
                    text_samples = [t.text for t in visual_context.detected_text[:2]]
                    print(f"👁️ Visual: {visual_context.screen_type}, text: {text_samples}")
                
        except Exception as e:
            print(f"⚠️ Visual analysis failed: {e}")
    
    def _print_final_summary(self):
        """Print final training summary"""
        total_time = time.time() - self.start_time
        
        print(f"\n🏁 Training Complete!")
        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"📊 Episodes completed: {len(self.episode_rewards)}")
        print(f"⏱️ Total time: {total_time:.1f} seconds ({total_time/3600:.2f} hours)")
        print(f"🎯 Total steps: {self.total_steps}")
        print(f"⚡ Average speed: {self.total_steps/total_time:.1f} steps/second")
        
        if self.episode_rewards:
            avg_reward = sum(self.episode_rewards) / len(self.episode_rewards)
            max_reward = max(self.episode_rewards)
            avg_steps = sum(self.episode_steps) / len(self.episode_steps)
            
            print(f"💰 Average reward: {avg_reward:.2f}")
            print(f"🏆 Best reward: {max_reward:.2f}")
            print(f"👣 Average steps per episode: {avg_steps:.1f}")
        
        print(f"🤖 LLM decisions made: {self.llm_decisions}")
        print(f"👁️ Visual analyses: {self.visual_analyses}")
        
        if self.enable_monitoring:
            print(f"📊 Monitoring dashboard: {self.monitor_server_url}")
    
    def stop(self):
        """Stop training and cleanup"""
        self.training_active = False
        
        if hasattr(self, 'env') and self.env:
            print("🔄 Closing environment...")
            self.env.close()
        
        print("✅ Training session stopped")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Pokemon Crystal RL Training with Advanced Monitoring')
    
    parser.add_argument('--rom', type=str, default='pokecrystal.gbc',
                       help='Path to Pokemon Crystal ROM file')
    parser.add_argument('--save-state', type=str, default=None,
                       help='Path to save state file (optional)')
    parser.add_argument('--episodes', type=int, default=50,
                       help='Number of episodes to run')
    parser.add_argument('--max-steps', type=int, default=2000,
                       help='Maximum steps per episode')
    parser.add_argument('--llm-interval', type=int, default=10,
                       help='LLM decision interval (steps)')
    parser.add_argument('--visual-interval', type=int, default=20,
                       help='Visual analysis interval (steps)')
    parser.add_argument('--screenshot-interval', type=int, default=5,
                       help='Screenshot update interval (steps)')
    parser.add_argument('--no-vision', action='store_true',
                       help='Disable computer vision')
    parser.add_argument('--no-monitoring', action='store_true',
                       help='Disable web monitoring')
    parser.add_argument('--monitor-url', type=str, default='http://localhost:5000',
                       help='Monitoring server URL')
    
    args = parser.parse_args()
    
    # Validate ROM file
    if not os.path.exists(args.rom):
        print(f"❌ ROM file not found: {args.rom}")
        print("Please provide a valid Pokemon Crystal ROM file path")
        return 1
    
    # Validate save state if provided
    if args.save_state and not os.path.exists(args.save_state):
        print(f"❌ Save state file not found: {args.save_state}")
        print("Continuing without save state (starting from beginning)")
        args.save_state = None
    
    try:
        # Create training session
        session = AdvancedTrainingSession(
            rom_path=args.rom,
            save_state_path=args.save_state,
            episodes=args.episodes,
            max_steps_per_episode=args.max_steps,
            llm_interval=args.llm_interval,
            visual_interval=args.visual_interval,
            screenshot_interval=args.screenshot_interval,
            enable_vision=not args.no_vision,
            enable_monitoring=not args.no_monitoring,
            monitor_server_url=args.monitor_url
        )
        
        # Start training
        session.start_training()
        
        return 0
        
    except KeyboardInterrupt:
        print("\n🛑 Training interrupted by user")
        return 0
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
