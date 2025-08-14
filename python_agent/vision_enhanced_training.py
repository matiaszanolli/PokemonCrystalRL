"""
vision_enhanced_training.py - Complete training pipeline with vision-enhanced LLM agent

This script integrates:
- PyBoy Pokemon Crystal environment
- Computer vision for screenshot analysis
- Local LLM agent with visual context understanding
- Training loop with episodic memory and learning
"""

import numpy as np
import time
import json
import cv2
from typing import Dict, List, Any, Tuple
from collections import deque
import matplotlib.pyplot as plt
from datetime import datetime
import os

# Import our modules
from pyboy_env import PyBoyPokemonCrystalEnv
from enhanced_llm_agent import EnhancedLLMPokemonAgent
from vision_processor import PokemonVisionProcessor, VisualContext


class VisionEnhancedTrainingSession:
    """
    Complete training session manager with vision-enhanced LLM agent
    """
    
    def __init__(self, 
                 rom_path: str,
                 save_state_path: str = None,
                 model_name: str = "llama3.2:3b",
                 max_steps_per_episode: int = 5000,
                 log_interval: int = 100,
                 screenshot_interval: int = 10):
        """
        Initialize training session
        
        Args:
            rom_path: Path to Pokemon Crystal ROM
            save_state_path: Path to save state file
            model_name: Ollama model name
            max_steps_per_episode: Maximum steps per episode
            log_interval: Steps between progress logs
            screenshot_interval: Steps between visual analysis
        """
        self.rom_path = rom_path
        self.save_state_path = save_state_path
        self.max_steps_per_episode = max_steps_per_episode
        self.log_interval = log_interval
        self.screenshot_interval = screenshot_interval
        
        # Initialize environment
        print("🎮 Initializing PyBoy environment...")
        self.env = PyBoyPokemonCrystalEnv(
            rom_path=rom_path,
            save_state_path=save_state_path,
            debug_mode=True
        )
        
        # Initialize enhanced agent
        print("🤖 Initializing Enhanced LLM Agent...")
        self.agent = EnhancedLLMPokemonAgent(
            model_name=model_name,
            use_vision=True
        )
        
        # Training metrics
        self.training_stats = {
            "episodes": 0,
            "total_steps": 0,
            "decisions_made": 0,
            "visual_analyses": 0,
            "last_progress": {
                "badges": 0,
                "money": 0,
                "party_size": 0,
                "location": (0, 0, 0)
            }
        }
        
        # Action history for LLM context
        self.action_history = deque(maxlen=20)
        
        # Screenshots for analysis
        self.screenshot_history = deque(maxlen=50)
        
        print("✅ Training session initialized")
    
    def run_training_episode(self, episode_num: int) -> Dict[str, Any]:
        """Run a single training episode"""
        print(f"\n🚀 Starting Episode {episode_num}")
        
        # Reset environment
        obs = self.env.reset()
        done = False
        step = 0
        episode_reward = 0
        episode_actions = []
        visual_analyses_count = 0
        
        episode_start_time = time.time()
        
        while not done and step < self.max_steps_per_episode:
            # Get current game state
            game_state = self.env.get_game_state()
            
            # Get screenshot for visual analysis
            screenshot = None
            if step % self.screenshot_interval == 0:
                screenshot = self.env.get_screenshot()
                self.screenshot_history.append(screenshot.copy())
                visual_analyses_count += 1
            
            # Decide action using enhanced LLM agent
            action = self.agent.decide_next_action(
                state=game_state,
                screenshot=screenshot,
                recent_history=list(self.action_history)
            )
            
            # Execute action
            step_result = self.env.step(action)
            if len(step_result) == 5:
                # New Gymnasium API: (obs, reward, terminated, truncated, info)
                obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                # Old Gym API: (obs, reward, done, info)
                obs, reward, done, info = step_result
            
            # Update metrics
            episode_reward += reward
            episode_actions.append(self.agent.action_map[action])
            self.action_history.append(self.agent.action_map[action])
            step += 1
            self.training_stats["total_steps"] += 1
            self.training_stats["decisions_made"] += 1
            
            # Log progress
            if step % self.log_interval == 0:
                current_state = self.env.get_game_state()
                player = current_state.get('player', {})
                progress = {
                    'step': step,
                    'location': f"Map {player.get('map', 0)} ({player.get('x', 0)}, {player.get('y', 0)})",
                    'money': player.get('money', 0),
                    'badges': player.get('badges', 0),
                    'party_size': len(current_state.get('party', [])),
                    'last_action': self.agent.action_map[action]
                }
                print(f"  📊 Step {step}: {progress}")
            
            # Check for significant progress
            self._check_for_progress(game_state)
            
            # Small delay to avoid overwhelming the system
            time.sleep(0.1)
        
        episode_duration = time.time() - episode_start_time
        
        # Episode summary
        episode_summary = {
            "episode": episode_num,
            "steps": step,
            "reward": episode_reward,
            "duration": episode_duration,
            "actions_taken": len(set(episode_actions)),
            "visual_analyses": visual_analyses_count,
            "final_state": self.env.get_game_state(),
            "action_distribution": {action: episode_actions.count(action) for action in set(episode_actions)}
        }
        
        self.training_stats["episodes"] += 1
        self.training_stats["visual_analyses"] += visual_analyses_count
        
        print(f"📋 Episode {episode_num} completed:")
        print(f"   Steps: {step}, Reward: {episode_reward:.2f}")
        print(f"   Duration: {episode_duration:.1f}s, Visual analyses: {visual_analyses_count}")
        print(f"   Unique actions: {len(set(episode_actions))}")
        
        return episode_summary
    
    def _check_for_progress(self, game_state: Dict[str, Any]):
        """Check for significant game progress and log it"""
        player = game_state.get('player', {})
        party = game_state.get('party', [])
        
        current_progress = {
            "badges": player.get('badges', 0),
            "money": player.get('money', 0),
            "party_size": len(party),
            "location": (player.get('x', 0), player.get('y', 0), player.get('map', 0))
        }
        
        last_progress = self.training_stats["last_progress"]
        
        # Check for badge progress
        if current_progress["badges"] > last_progress["badges"]:
            print(f"🏆 PROGRESS: Earned badge! Now have {current_progress['badges']} badges")
        
        # Check for new Pokemon
        if current_progress["party_size"] > last_progress["party_size"]:
            print(f"🐾 PROGRESS: Caught new Pokemon! Party size: {current_progress['party_size']}")
        
        # Check for significant money gain (with safeguards against false positives)
        money_diff = current_progress["money"] - last_progress["money"]
        current_money = current_progress["money"]
        last_money = last_progress["money"]
        
        # Only consider it significant progress if:
        # 1. Money increased by more than 1000
        # 2. Current money is greater than 2000 (to avoid early game noise)
        # 3. The increase is more than 20% of previous money (proportional check)
        if (money_diff > 1000 and 
            current_money > 2000 and 
            last_money > 0 and 
            money_diff > (last_money * 0.2)):
            print(f"💰 PROGRESS: Significant money gain of ${money_diff}! Total: ${current_money}")
        
        # Check for location change (new map)
        if current_progress["location"][2] != last_progress["location"][2]:
            print(f"🗺️ PROGRESS: Moved to new map {current_progress['location'][2]}!")
        
        self.training_stats["last_progress"] = current_progress
    
    def run_training_session(self, num_episodes: int = 10):
        """Run a complete training session"""
        print(f"🎯 Starting training session: {num_episodes} episodes")
        print(f"📊 Max steps per episode: {self.max_steps_per_episode}")
        print(f"👁️ Visual analysis every {self.screenshot_interval} steps")
        
        session_start_time = time.time()
        episode_summaries = []
        
        try:
            for episode in range(1, num_episodes + 1):
                episode_summary = self.run_training_episode(episode)
                episode_summaries.append(episode_summary)
                
                # Show agent memory summary every few episodes
                if episode % 3 == 0:
                    memory_summary = self.agent.get_enhanced_memory_summary()
                    print(f"\n💭 Agent Memory Summary (Episode {episode}):")
                    print(f"   Decisions stored: {memory_summary['decisions_stored']}")
                    print(f"   Visual analyses: {memory_summary['visual_analyses']}")
                    print(f"   Screen types seen: {memory_summary['screen_type_breakdown']}")
                    if memory_summary['latest_progress']:
                        latest = memory_summary['latest_progress']
                        print(f"   Latest progress: {latest['badges']} badges, ${latest['money']}, {latest['party_size']} Pokemon")
                        print(f"   Last screen: {latest['last_screen_type']}")
                    print()
        
        except KeyboardInterrupt:
            print("\n⏹️ Training interrupted by user")
        except Exception as e:
            print(f"\n❌ Training error: {e}")
        
        finally:
            session_duration = time.time() - session_start_time
            self._generate_training_report(episode_summaries, session_duration)
            self.env.close()
    
    def _generate_training_report(self, episode_summaries: List[Dict], session_duration: float):
        """Generate comprehensive training report"""
        print(f"\n📈 TRAINING SESSION REPORT")
        print(f"=" * 50)
        print(f"Duration: {session_duration:.1f} seconds ({session_duration/60:.1f} minutes)")
        print(f"Episodes completed: {len(episode_summaries)}")
        print(f"Total steps: {self.training_stats['total_steps']}")
        print(f"Total decisions: {self.training_stats['decisions_made']}")
        print(f"Visual analyses: {self.training_stats['visual_analyses']}")
        print(f"Average steps per episode: {np.mean([ep['steps'] for ep in episode_summaries]):.1f}")
        print(f"Average episode duration: {np.mean([ep['duration'] for ep in episode_summaries]):.1f}s")
        
        # Action analysis
        all_actions = []
        for episode in episode_summaries:
            all_actions.extend(episode['action_distribution'].keys())
        
        unique_actions_used = len(set(all_actions))
        print(f"Unique actions used: {unique_actions_used}/9")
        
        # Progress analysis
        final_progress = self.training_stats["last_progress"]
        print(f"\nFINAL PROGRESS:")
        print(f"  Badges: {final_progress['badges']}")
        print(f"  Money: ${final_progress['money']}")
        print(f"  Party size: {final_progress['party_size']}")
        print(f"  Location: Map {final_progress['location'][2]} at ({final_progress['location'][0]}, {final_progress['location'][1]})")
        
        # Agent memory analysis
        memory_summary = self.agent.get_enhanced_memory_summary()
        print(f"\nAGENT MEMORY ANALYSIS:")
        print(f"  Total decisions stored: {memory_summary['decisions_stored']}")
        print(f"  Game states recorded: {memory_summary['states_recorded']}")
        print(f"  Visual analyses: {memory_summary['visual_analyses']}")
        print(f"  Screen types encountered: {list(memory_summary['screen_type_breakdown'].keys())}")
        
        # Save detailed report
        os.makedirs("outputs", exist_ok=True)
        report_filename = f"outputs/training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report_data = {
            "session_info": {
                "duration": session_duration,
                "episodes": len(episode_summaries),
                "total_steps": self.training_stats['total_steps'],
                "model_used": self.agent.model_name
            },
            "episode_summaries": episode_summaries,
            "training_stats": self.training_stats,
            "memory_summary": memory_summary,
            "final_progress": final_progress
        }
        
        with open(report_filename, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"\n📄 Detailed report saved to: {report_filename}")
        print(f"🎉 Training session complete!")
    
    def save_screenshot_analysis(self, save_dir: str = "outputs/screenshot_analysis"):
        """Save recent screenshots with visual analysis"""
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        print(f"📸 Saving screenshot analysis to {save_dir}...")
        
        for i, screenshot in enumerate(list(self.screenshot_history)[-10:]):  # Last 10 screenshots
            # Process screenshot
            visual_context = self.agent.vision_processor.process_screenshot(screenshot)
            
            # Save screenshot
            screenshot_path = os.path.join(save_dir, f"screenshot_{i:03d}.png")
            cv2.imwrite(screenshot_path, cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR))
            
            # Save analysis
            analysis_path = os.path.join(save_dir, f"analysis_{i:03d}.json")
            analysis_data = {
                "screen_type": visual_context.screen_type,
                "game_phase": visual_context.game_phase,
                "visual_summary": visual_context.visual_summary,
                "detected_text": [{"text": t.text, "location": t.location, "confidence": t.confidence} 
                                for t in visual_context.detected_text],
                "ui_elements": [{"type": e.element_type, "confidence": e.confidence} 
                              for e in visual_context.ui_elements],
                "dominant_colors": visual_context.dominant_colors
            }
            
            with open(analysis_path, 'w') as f:
                json.dump(analysis_data, f, indent=2)
        
        print(f"✅ Saved {len(list(self.screenshot_history)[-10:])} screenshots with analysis")


def main():
    """Main training function"""
    # Configuration
    ROM_PATH = "/mnt/data/src/pokemon_crystal_rl/roms/pokemon_crystal.gbc"
    SAVE_STATE_PATH = "/mnt/data/src/pokemon_crystal_rl/save_states/crystal_intro_done.state"
    
    # Check if files exist
    if not os.path.exists(ROM_PATH):
        print(f"❌ ROM file not found: {ROM_PATH}")
        print("Please ensure you have a Pokemon Crystal ROM file")
        return
    
    if SAVE_STATE_PATH and not os.path.exists(SAVE_STATE_PATH):
        print(f"⚠️ Save state not found: {SAVE_STATE_PATH}")
        print("Will start from beginning of game")
        SAVE_STATE_PATH = None
    
    # Initialize training session
    try:
        training_session = VisionEnhancedTrainingSession(
            rom_path=ROM_PATH,
            save_state_path=SAVE_STATE_PATH,
            model_name="llama3.2:3b",
            max_steps_per_episode=3000,  # Shorter episodes for testing
            screenshot_interval=15  # Visual analysis every 15 steps
        )
        
        # Run training
        training_session.run_training_session(num_episodes=5)
        
        # Save screenshot analysis
        training_session.save_screenshot_analysis()
        
    except Exception as e:
        print(f"❌ Training session failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
