#!/usr/bin/env python3
"""
monitored_training.py - Speed-optimized Pokemon Crystal RL training with full web monitoring

This version maintains all web monitoring capabilities while implementing key speed optimizations:
- Action caching to reduce LLM calls
- Efficient web updates (not real-time but frequent enough)
- Smart visual analysis
- Optimized but comprehensive monitoring

The goal is practical training speed (20-50 actions/sec) with full observability.
"""

import time
import threading
import signal
import sys
import numpy as np
from collections import deque
from typing import Dict, List, Any, Optional
import json

from pyboy_env import PyBoyPokemonCrystalEnv
from enhanced_llm_agent import EnhancedLLMPokemonAgent
from web_monitor import PokemonRLWebMonitor, create_dashboard_templates
from vision_processor import PokemonVisionProcessor


class MonitoredTrainingSession:
    """
    Speed-optimized training with comprehensive web monitoring
    """
    
    def __init__(self, 
                 rom_path: str,
                 save_state_path: str = None,
                 llm_interval: int = 15,        # LLM decision every N steps
                 visual_interval: int = 30,     # Visual analysis every N steps
                 web_update_interval: int = 5,  # Web updates every N steps
                 enable_vision: bool = True):
        
        self.rom_path = rom_path
        self.save_state_path = save_state_path
        self.llm_interval = llm_interval
        self.visual_interval = visual_interval
        self.web_update_interval = web_update_interval
        self.enable_vision = enable_vision
        
        # Initialize environment 
        print("🎮 Initializing PyBoy environment...")
        self.env = PyBoyPokemonCrystalEnv(
            rom_path=rom_path,
            save_state_path=save_state_path,
            headless=True,  # Headless for speed but web UI for monitoring
            debug_mode=False
        )
        
        # Initialize enhanced agent
        print("🤖 Initializing Enhanced LLM Agent...")
        self.agent = EnhancedLLMPokemonAgent(
            use_vision=enable_vision
        )
        
        # Initialize vision processor if enabled
        self.vision_processor = None
        if enable_vision:
            print("👁️ Initializing vision processor...")
            self.vision_processor = PokemonVisionProcessor()
        
        # Initialize web monitor (always enabled)
        print("🌐 Initializing web monitor...")
        self.web_monitor = PokemonRLWebMonitor()
        
        # Training metrics
        self.training_stats = {
            "episodes": 0,
            "total_steps": 0,
            "llm_calls": 0,
            "visual_analyses": 0,
            "decisions_made": 0,
            "actions_per_second": 0,
            "start_time": None
        }
        
        # Action caching for speed
        self.action_sequence = deque()
        self.last_llm_decision_step = 0
        
        # Action history for context
        self.action_history = deque(maxlen=20)
        
        # Screenshot history for visual analysis
        self.screenshot_history = deque(maxlen=10)
        
        # Web server thread
        self.web_server_thread = None
        self.training_active = False
        
        # Signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        print("✅ Monitored training session initialized!")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        print(f"\n🛑 Received signal {signum}, shutting down gracefully...")
        self.stop()
        sys.exit(0)
    
    def start_web_server(self, host='127.0.0.1', port=5000):
        """Start the web monitoring server"""
        print(f"🚀 Starting web server on {host}:{port}")
        
        # Create templates
        create_dashboard_templates()
        
        # Start web server in separate thread
        def run_web_server():
            self.web_monitor.run(host=host, port=port, debug=False)
        
        self.web_server_thread = threading.Thread(target=run_web_server, daemon=True)
        self.web_server_thread.start()
        
        # Give the server a moment to start
        time.sleep(2)
        
        print(f"📊 Web monitoring dashboard available at: http://{host}:{port}")
        return f"http://{host}:{port}"
    
    def smart_action_decision(self, step: int, game_state: Dict[str, Any]) -> int:
        """
        Smart action decision with caching for speed but full monitoring
        """
        # Use LLM at intervals or when cache is empty
        if (step % self.llm_interval == 0 or 
            len(self.action_sequence) == 0 or
            step - self.last_llm_decision_step > self.llm_interval * 2):
            
            return self._get_llm_action_sequence(game_state, step)
        
        # Use cached action if available
        if self.action_sequence:
            return self.action_sequence.popleft()
        
        # Fallback: simple exploration
        return self._get_fallback_action(game_state)
    
    def _get_llm_action_sequence(self, game_state: Dict[str, Any], step: int) -> int:
        """Get action sequence from LLM with full context"""
        try:
            # Get visual context if enabled and due for analysis
            visual_context = None
            if (self.enable_vision and self.vision_processor and 
                step % self.visual_interval == 0):
                
                screenshot = self.env.get_screenshot()
                if screenshot is not None:
                    visual_context = self.vision_processor.process_screenshot(screenshot)
                    self.training_stats["visual_analyses"] += 1
                    self.screenshot_history.append(screenshot.copy())
            
            # Get LLM decision with full context
            start_time = time.time()
            action = self.agent.decide_next_action(
                state=game_state,
                screenshot=self.screenshot_history[-1] if self.screenshot_history else None,
                recent_history=list(self.action_history)
            )
            llm_time = time.time() - start_time
            
            self.training_stats["llm_calls"] += 1
            self.last_llm_decision_step = step
            
            # For speed, generate a few follow-up actions based on the LLM decision
            follow_up_actions = self._generate_follow_up_actions(action, game_state)
            self.action_sequence.extend(follow_up_actions)
            
            # Create decision data for web monitoring
            decision_data = {
                'decision': f"Action: {self.agent.action_map.get(action, action)}",
                'reasoning': f"LLM decision after {llm_time:.2f}s analysis",
                'confidence': 0.8,
                'visual_context': {
                    'screen_type': visual_context.screen_type if visual_context else 'unknown',
                    'has_visual_data': visual_context is not None
                }
            }
            self.web_monitor.update_decision(decision_data)
            
            if step % 20 == 0:  # Occasional performance logging
                print(f"⚡ LLM call took {llm_time:.2f}s (step {step})")
            
            return action
            
        except Exception as e:
            print(f"⚠️ LLM error at step {step}: {e}")
            return self._get_fallback_action(game_state)
    
    def _generate_follow_up_actions(self, primary_action: int, game_state: Dict[str, Any]) -> List[int]:
        """Generate logical follow-up actions based on primary LLM decision"""
        follow_ups = []
        
        # Simple heuristics for follow-up actions
        if primary_action == 5:  # A button (interact)
            follow_ups = [5, 5]  # Continue pressing A for dialogue
        elif primary_action in [1, 2, 3, 4]:  # Movement
            # Continue in same direction briefly
            follow_ups = [primary_action, primary_action]
        elif primary_action == 7:  # START (menu)
            follow_ups = [2, 5]  # DOWN, A (navigate menu)
        else:
            # Default exploration sequence
            follow_ups = [1, 2, 3, 4][:(np.random.randint(1, 3))]  # 1-2 movement actions
        
        return follow_ups
    
    def _get_fallback_action(self, game_state: Dict[str, Any]) -> int:
        """Fast fallback action when LLM unavailable"""
        # Simple exploration pattern
        exploration_actions = [1, 2, 3, 4]  # UP, DOWN, LEFT, RIGHT
        return np.random.choice(exploration_actions)
    
    def run_monitored_episode(self, episode_num: int, max_steps: int = 2000) -> Dict[str, Any]:
        """Run a single episode with full monitoring"""
        print(f"🚀 Monitored Episode {episode_num} (max {max_steps} steps)")
        
        # Reset environment
        obs = self.env.reset()
        done = False
        step = 0
        episode_reward = 0
        episode_start = time.time()
        
        # Clear caches for new episode
        self.action_sequence.clear()
        self.last_llm_decision_step = 0
        
        while not done and step < max_steps:
            step_start = time.time()
            
            # Get current game state
            game_state = self.env.get_game_state()
            
            # Smart action decision (with caching)
            action = self.smart_action_decision(step, game_state)
            
            # Execute action
            step_result = self.env.step(action)
            if len(step_result) == 5:
                obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                obs, reward, done, info = step_result
            
            episode_reward += reward
            step += 1
            self.training_stats["total_steps"] += 1
            self.training_stats["decisions_made"] += 1
            
            # Update action history
            action_name = self.agent.action_map.get(action, f"Action_{action}")
            self.action_history.append(action_name)
            
            # Web monitoring updates (at regular intervals)
            if step % self.web_update_interval == 0:
                self._update_web_monitoring(game_state, action, step)
            
            # Progress logging
            if step % 100 == 0:
                player = game_state.get('player', {})
                print(f"  📊 Step {step}: Map {player.get('map', 0)} "
                      f"(${player.get('money', 0)}, {player.get('badges', 0)} badges) "
                      f"- {action_name}")
            
            # Track performance
            step_time = time.time() - step_start
            if step % 50 == 0:  # Update speed tracking
                self.training_stats["actions_per_second"] = 1.0 / step_time if step_time > 0 else 0
        
        episode_duration = time.time() - episode_start
        avg_speed = step / episode_duration if episode_duration > 0 else 0
        
        self.training_stats["episodes"] += 1
        
        print(f"📋 Episode {episode_num} completed:")
        print(f"   Steps: {step}, Reward: {episode_reward:.2f}")
        print(f"   Duration: {episode_duration:.1f}s")
        print(f"   Speed: {avg_speed:.1f} actions/second")
        print(f"   LLM calls: {self.training_stats['llm_calls']} total")
        print(f"   Visual analyses: {self.training_stats['visual_analyses']} total")
        
        return {
            "episode": episode_num,
            "steps": step,
            "reward": episode_reward,
            "duration": episode_duration,
            "actions_per_second": avg_speed,
            "llm_calls": self.training_stats["llm_calls"],
            "visual_analyses": self.training_stats["visual_analyses"]
        }
    
    def _update_web_monitoring(self, game_state: Dict[str, Any], action: int, step: int):
        """Update web monitoring with current data"""
        try:
            # Update screenshot
            screenshot = self.env.get_screenshot()
            if screenshot is not None:
                self.web_monitor.update_screenshot(screenshot)
            
            # Update action
            action_name = self.agent.action_map.get(action, f"Action_{action}")
            reasoning = f"Step {step} - Cached action" if len(self.action_sequence) > 0 else f"Step {step} - LLM decision"
            self.web_monitor.update_action(action_name, reasoning)
            
            # Performance metrics
            self.web_monitor.add_performance_metric('steps_per_second', self.training_stats.get('actions_per_second', 0))
            self.web_monitor.add_performance_metric('llm_calls', self.training_stats['llm_calls'])
            
        except Exception as e:
            # Don't let web monitoring errors break training
            print(f"⚠️ Web monitoring error: {e}")
    
    def run_monitored_training(self, num_episodes: int = 10):
        """Run multiple episodes with full monitoring"""
        print(f"🎮 Starting Monitored Training: {num_episodes} episodes")
        print("=" * 60)
        print("🌐 Web monitoring: ENABLED")
        print(f"⚡ LLM interval: every {self.llm_interval} steps")
        print(f"👁️ Visual analysis: every {self.visual_interval} steps")
        print(f"📊 Web updates: every {self.web_update_interval} steps")
        print("=" * 60)
        
        self.training_stats["start_time"] = time.time()
        self.training_active = True
        
        # Start web monitoring
        self.web_monitor.start_monitoring()
        
        episode_results = []
        
        for episode in range(1, num_episodes + 1):
            try:
                result = self.run_monitored_episode(episode)
                episode_results.append(result)
                
                # Progress summary
                if episode % 3 == 0:
                    elapsed = time.time() - self.training_stats["start_time"]
                    total_steps = self.training_stats["total_steps"]
                    avg_speed = total_steps / elapsed if elapsed > 0 else 0
                    
                    print(f"\n📈 Training Progress (Episode {episode}):")
                    print(f"   Total steps: {total_steps}")
                    print(f"   Average speed: {avg_speed:.1f} actions/second")
                    print(f"   LLM efficiency: {self.training_stats['llm_calls']} calls / {total_steps} steps")
                    print(f"   Visual analyses: {self.training_stats['visual_analyses']}")
                    print()
                
            except KeyboardInterrupt:
                print("\n⏸️ Training interrupted by user")
                break
            except Exception as e:
                print(f"❌ Episode {episode} error: {e}")
                continue
        
        # Final summary
        total_duration = time.time() - self.training_stats["start_time"]
        total_steps = self.training_stats["total_steps"]
        overall_speed = total_steps / total_duration if total_duration > 0 else 0
        
        print(f"\n🏁 Monitored Training completed!")
        print(f"   Episodes: {self.training_stats['episodes']}")
        print(f"   Total steps: {total_steps}")
        print(f"   Duration: {total_duration:.1f}s")
        print(f"   Overall speed: {overall_speed:.1f} actions/second")
        print(f"   LLM calls: {self.training_stats['llm_calls']}")
        print(f"   Visual analyses: {self.training_stats['visual_analyses']}")
        print(f"   Web monitoring: Active throughout training")
        
        return episode_results
    
    def stop(self):
        """Stop training and monitoring"""
        print("🛑 Stopping training and monitoring...")
        
        self.training_active = False
        
        if self.web_monitor:
            self.web_monitor.stop_monitoring()
        
        if self.env:
            self.env.close()
        
        print("✅ Training and monitoring stopped")
    
    def run_interactive(self):
        """Run in interactive mode with web monitoring"""
        print("🎮 Pokemon Crystal RL - Monitored Training")
        print("=" * 50)
        
        # Start web server
        dashboard_url = self.start_web_server()
        
        print("🌐 Web monitoring dashboard started!")
        print()
        print("📋 Instructions:")
        print(f"1. Open {dashboard_url} in your browser")
        print("2. Monitor real-time training progress")
        print("3. Training will start automatically with live updates")
        print()
        print("Features enabled:")
        print("  ✅ Real-time web monitoring")
        print("  ✅ LLM-powered decisions (cached for speed)")
        print("  ✅ Visual analysis")
        print("  ✅ Performance optimization")
        print()
        print("Press Ctrl+C at any time to stop training gracefully")
        print("=" * 50)
        print()
        
        # Wait a moment for user to open browser
        print("⏳ Waiting 5 seconds for you to open the web dashboard...")
        time.sleep(5)
        
        # Start training with monitoring
        return self.run_monitored_training()


def main():
    """Run monitored training with command line arguments"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Pokemon Crystal RL with Full Monitoring")
    parser.add_argument("--rom", default="../roms/pokemon_crystal.gbc", help="ROM path")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes")
    parser.add_argument("--llm-interval", type=int, default=15, help="LLM decision interval")
    parser.add_argument("--visual-interval", type=int, default=30, help="Visual analysis interval")
    parser.add_argument("--web-interval", type=int, default=5, help="Web update interval")
    parser.add_argument("--no-vision", action="store_true", help="Disable vision processing")
    parser.add_argument("--host", default="127.0.0.1", help="Web server host")
    parser.add_argument("--port", type=int, default=5000, help="Web server port")
    
    args = parser.parse_args()
    
    # Initialize monitored training
    session = MonitoredTrainingSession(
        rom_path=args.rom,
        llm_interval=args.llm_interval,
        visual_interval=args.visual_interval,
        web_update_interval=args.web_interval,
        enable_vision=not args.no_vision
    )
    
    # Start web server
    session.start_web_server(host=args.host, port=args.port)
    print(f"🌐 Web monitoring available at http://{args.host}:{args.port}")
    time.sleep(2)
    
    # Run training
    session.run_monitored_training(args.episodes)


if __name__ == "__main__":
    main()
