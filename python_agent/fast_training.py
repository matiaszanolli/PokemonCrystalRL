#!/usr/bin/env python3
"""
fast_training.py - High-performance Pokemon Crystal RL training with speed optimizations

Major optimizations:
1. Reduced LLM calls (every N steps instead of every step)
2. Cached action sequences between LLM decisions
3. Minimal visual analysis (only when needed)
4. Batch database operations
5. Reduced web monitoring overhead
6. No artificial delays
7. Simple decision fallbacks
"""

import time
import numpy as np
from collections import deque
from typing import Dict, List, Any, Optional
import threading
import queue

from pyboy_env import PyBoyPokemonCrystalEnv
from enhanced_llm_agent import EnhancedLLMPokemonAgent
from web_monitor import PokemonRLWebMonitor, create_dashboard_templates


class FastTrainingSession:
    """
    Speed-optimized training session for Pokemon Crystal RL
    """
    
    def __init__(self, 
                 rom_path: str,
                 save_state_path: str = None,
                 llm_interval: int = 20,  # LLM decision every N steps
                 visual_interval: int = 50,  # Visual analysis every N steps  
                 web_update_interval: int = 10):  # Web updates every N steps
        
        self.rom_path = rom_path
        self.save_state_path = save_state_path
        self.llm_interval = llm_interval
        self.visual_interval = visual_interval
        self.web_update_interval = web_update_interval
        
        # Initialize environment (headless for speed)
        print("🚀 Initializing fast training environment...")
        self.env = PyBoyPokemonCrystalEnv(
            rom_path=rom_path,
            save_state_path=save_state_path,
            headless=True,  # No GUI for maximum speed
            debug_mode=False  # Disable debug prints
        )
        
        # Initialize agent
        print("🤖 Initializing speed-optimized agent...")
        self.agent = EnhancedLLMPokemonAgent(
            use_vision=False  # Disable vision initially for speed
        )
        
        # Performance tracking
        self.training_stats = {
            "episodes": 0,
            "total_steps": 0,
            "llm_calls": 0,
            "visual_analyses": 0,
            "actions_per_second": 0,
            "start_time": None
        }
        
        # Action caching for speed
        self.action_sequence = deque()  # Cached actions from last LLM call
        self.last_llm_decision_step = 0
        
        # Simple fallback actions when LLM not consulted
        self.exploration_actions = [1, 2, 3, 4]  # UP, DOWN, LEFT, RIGHT
        self.action_weights = [0.3, 0.3, 0.2, 0.2]  # Slightly prefer UP/DOWN
        
        # Batch operations
        self.pending_db_operations = []
        self.batch_size = 10
        
        # Web monitoring (optional)
        self.web_monitor = None
        self.enable_web_monitoring = False
        
        print("✅ Fast training session initialized!")
    
    def enable_web_monitoring(self, host='127.0.0.1', port=5000):
        """Enable web monitoring with reduced overhead"""
        print("🌐 Setting up lightweight web monitoring...")
        self.web_monitor = PokemonRLWebMonitor()
        self.enable_web_monitoring = True
        
        # Start web server in background
        def run_web_server():
            create_dashboard_templates()
            self.web_monitor.run(host=host, port=port, debug=False)
        
        web_thread = threading.Thread(target=run_web_server, daemon=True)
        web_thread.start()
        time.sleep(1)  # Brief pause for server startup
        
        print(f"📊 Web monitoring available at http://{host}:{port}")
    
    def fast_action_decision(self, step: int, game_state: Dict[str, Any]) -> int:
        """
        Fast action decision with minimal LLM overhead
        """
        # Use LLM only at intervals or when cache is empty
        if (step % self.llm_interval == 0 or 
            len(self.action_sequence) == 0 or
            self._detect_state_change(game_state)):
            
            return self._get_llm_action_sequence(game_state, step)
        
        # Use cached action if available
        if self.action_sequence:
            return self.action_sequence.popleft()
        
        # Fast fallback: simple exploration
        return self._get_exploration_action()
    
    def _get_llm_action_sequence(self, game_state: Dict[str, Any], step: int) -> int:
        """Get a sequence of actions from LLM to cache"""
        try:
            # Simplified state analysis for speed
            analysis = self._quick_state_analysis(game_state)
            
            # Get LLM decision for next few actions
            prompt = self._create_fast_prompt(game_state, analysis)
            
            start_time = time.time()
            response = self.agent._query_local_llm(prompt)
            llm_time = time.time() - start_time
            
            self.training_stats["llm_calls"] += 1
            
            # Parse response for action sequence
            actions = self._parse_action_sequence(response)
            
            if actions:
                # Cache actions for next few steps
                self.action_sequence.extend(actions[1:])  # Cache remaining
                return actions[0]  # Return first action
            
            print(f"⚡ LLM call took {llm_time:.2f}s")
            
        except Exception as e:
            print(f"⚠️ LLM error: {e}")
        
        # Fallback to exploration
        return self._get_exploration_action()
    
    def _quick_state_analysis(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Ultra-fast state analysis without heavy processing"""
        player = state.get('player', {})
        party = state.get('party', [])
        
        # Simple categorization
        if len(party) == 0:
            phase = "starter_selection"
            goal = "get_starter"
        elif player.get('badges', 0) == 0:
            phase = "early_game"
            goal = "explore_train"
        else:
            phase = "progression"
            goal = "advance_story"
        
        return {
            "phase": phase,
            "primary_goal": goal,
            "party_size": len(party),
            "badges": player.get('badges', 0),
            "location": (player.get('x', 0), player.get('y', 0))
        }
    
    def _create_fast_prompt(self, state: Dict[str, Any], analysis: Dict[str, Any]) -> str:
        """Create a concise prompt for fast LLM processing"""
        player = state.get('player', {})
        
        prompt = f"""Pokemon Crystal Agent - Quick Decision:

CURRENT STATE:
- Location: Map {player.get('map', 0)} at ({player.get('x', 0)}, {player.get('y', 0)})
- Money: ${player.get('money', 0)}
- Badges: {player.get('badges', 0)}
- Party size: {analysis['party_size']}
- Phase: {analysis['phase']}

GOAL: {analysis['primary_goal']}

Give me 3-5 quick actions to take. Format: "UP,A,DOWN,RIGHT,A"
Available: UP,DOWN,LEFT,RIGHT,A,B,START,SELECT
"""
        return prompt
    
    def _parse_action_sequence(self, response: str) -> List[int]:
        """Parse LLM response into sequence of action integers"""
        actions = []
        action_map = {
            'UP': 1, 'DOWN': 2, 'LEFT': 3, 'RIGHT': 4,
            'A': 5, 'B': 6, 'START': 7, 'SELECT': 8
        }
        
        # Look for comma-separated actions or individual words
        response_upper = response.upper()
        
        # Try comma-separated format first
        if ',' in response_upper:
            parts = [p.strip() for p in response_upper.split(',')]
            for part in parts[:5]:  # Max 5 actions
                if part in action_map:
                    actions.append(action_map[part])
        
        # Fallback: scan for individual action words
        if not actions:
            for action_name, action_id in action_map.items():
                if action_name in response_upper:
                    actions.append(action_id)
                    if len(actions) >= 3:  # Limit for speed
                        break
        
        return actions if actions else [self._get_exploration_action()]
    
    def _get_exploration_action(self) -> int:
        """Fast random exploration action"""
        return np.random.choice(self.exploration_actions, p=self.action_weights)
    
    def _detect_state_change(self, game_state: Dict[str, Any]) -> bool:
        """Detect if we need a new LLM decision due to state change"""
        # Simple heuristics for when to consult LLM again
        player = game_state.get('player', {})
        
        # If we moved significantly or other major changes
        # This is a simplified check - in practice, you'd want more sophisticated detection
        return False  # For now, rely on interval-based decisions
    
    def run_fast_episode(self, episode_num: int, max_steps: int = 2000) -> Dict[str, Any]:
        """Run a single episode optimized for speed"""
        print(f"⚡ Fast Episode {episode_num} (max {max_steps} steps)")
        
        # Reset environment
        obs = self.env.reset()
        done = False
        step = 0
        episode_reward = 0
        
        episode_start = time.time()
        step_times = []
        
        while not done and step < max_steps:
            step_start = time.time()
            
            # Get game state (minimal processing)
            game_state = self.env.get_game_state()
            
            # Fast action decision
            action = self.fast_action_decision(step, game_state)
            
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
            
            # Minimal logging (only significant steps)
            if step % 200 == 0:
                player = game_state.get('player', {})
                print(f"  ⚡ Step {step}: Map {player.get('map', 0)} "
                      f"(${player.get('money', 0)}, {player.get('badges', 0)} badges)")
            
            # Web monitoring update (reduced frequency)
            if (self.enable_web_monitoring and self.web_monitor and 
                step % self.web_update_interval == 0):
                self._fast_web_update(game_state, action)
            
            # Track performance
            step_time = time.time() - step_start
            step_times.append(step_time)
            
            # NO artificial delays for maximum speed
        
        episode_duration = time.time() - episode_start
        avg_step_time = np.mean(step_times)
        steps_per_second = 1.0 / avg_step_time if avg_step_time > 0 else 0
        
        self.training_stats["episodes"] += 1
        self.training_stats["actions_per_second"] = steps_per_second
        
        print(f"⚡ Episode {episode_num} completed in {episode_duration:.1f}s")
        print(f"   Steps: {step}, Reward: {episode_reward:.2f}")
        print(f"   Speed: {steps_per_second:.1f} actions/second")
        print(f"   LLM calls: {self.training_stats['llm_calls']}")
        
        return {
            "episode": episode_num,
            "steps": step,
            "reward": episode_reward,
            "duration": episode_duration,
            "actions_per_second": steps_per_second,
            "llm_calls": self.training_stats["llm_calls"]
        }
    
    def _fast_web_update(self, game_state: Dict[str, Any], action: int):
        """Minimal web monitoring update"""
        try:
            # Only send essential data
            if hasattr(self.web_monitor, 'update_action'):
                action_name = self.agent.action_map.get(action, f"Action_{action}")
                self.web_monitor.update_action(action_name, "Fast mode")
            
        except Exception as e:
            # Silently ignore web monitoring errors in fast mode
            pass
    
    def run_fast_training(self, num_episodes: int = 10):
        """Run multiple episodes optimized for maximum speed"""
        print(f"🚀 Starting FAST training: {num_episodes} episodes")
        print("=" * 50)
        
        self.training_stats["start_time"] = time.time()
        
        for episode in range(1, num_episodes + 1):
            try:
                episode_result = self.run_fast_episode(episode)
                
                # Brief summary every few episodes
                if episode % 5 == 0:
                    elapsed = time.time() - self.training_stats["start_time"]
                    total_steps = self.training_stats["total_steps"]
                    avg_speed = total_steps / elapsed if elapsed > 0 else 0
                    
                    print(f"\n📊 Progress Update (Episode {episode}):")
                    print(f"   Total steps: {total_steps}")
                    print(f"   Average speed: {avg_speed:.1f} actions/second")
                    print(f"   LLM efficiency: {self.training_stats['llm_calls']} calls for {total_steps} steps")
                    print()
                
            except KeyboardInterrupt:
                print("\n⏸️ Training interrupted by user")
                break
            except Exception as e:
                print(f"❌ Episode {episode} error: {e}")
                continue
        
        # Final summary
        total_duration = time.time() - self.training_stats["start_time"]
        print(f"\n🏁 FAST Training completed!")
        print(f"   Episodes: {self.training_stats['episodes']}")
        print(f"   Total steps: {self.training_stats['total_steps']}")
        print(f"   Duration: {total_duration:.1f}s")
        print(f"   Overall speed: {self.training_stats['total_steps'] / total_duration:.1f} actions/second")
        print(f"   LLM calls: {self.training_stats['llm_calls']}")
        
        # Cleanup
        self.env.close()


def main():
    """Run fast training with command line arguments"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Fast Pokemon Crystal RL Training")
    parser.add_argument("--rom", default="../roms/pokemon_crystal.gbc", help="ROM path")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes")
    parser.add_argument("--llm-interval", type=int, default=20, help="LLM decision interval")
    parser.add_argument("--web", action="store_true", help="Enable web monitoring")
    
    args = parser.parse_args()
    
    # Initialize fast training
    session = FastTrainingSession(
        rom_path=args.rom,
        llm_interval=args.llm_interval
    )
    
    # Enable web monitoring if requested
    if args.web:
        session.enable_web_monitoring()
        print("🌐 Web monitoring enabled - visit http://127.0.0.1:5000")
        time.sleep(2)
    
    # Run training
    session.run_fast_training(args.episodes)


if __name__ == "__main__":
    main()
