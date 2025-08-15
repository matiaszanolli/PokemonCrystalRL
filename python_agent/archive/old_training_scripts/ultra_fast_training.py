#!/usr/bin/env python3
"""
ultra_fast_training.py - Maximum speed Pokemon Crystal training with minimal overhead

Extreme optimizations:
1. No LLM calls during training loop (pre-generated action patterns)
2. No visual analysis 
3. No database operations
4. No web monitoring during training
5. Minimal memory allocations
6. Simple rule-based actions
7. Batch environment operations
"""

import time
import numpy as np
from typing import Dict, Any

from pyboy_env import PyBoyPokemonCrystalEnv


class UltraFastTraining:
    """
    Maximum speed training with rule-based agent
    """
    
    def __init__(self, rom_path: str, save_state_path: str = None):
        self.rom_path = rom_path
        self.save_state_path = save_state_path
        
        # Initialize environment with maximum speed settings
        print("⚡ Initializing ULTRA-FAST environment...")
        self.env = PyBoyPokemonCrystalEnv(
            rom_path=rom_path,
            save_state_path=save_state_path,
            headless=True,
            debug_mode=False
        )
        
        # Rule-based action patterns for different scenarios
        self.action_patterns = {
            "exploration": [1, 1, 2, 2, 3, 4, 1, 2],  # Move around
            "menu_navigation": [5, 5, 2, 5],  # A, A, DOWN, A
            "dialogue": [5, 5, 5, 5],  # Keep pressing A
            "battle": [5, 2, 5, 2, 5],  # A, DOWN, A, DOWN, A
        }
        
        self.current_pattern = "exploration"
        self.pattern_index = 0
        self.steps_in_pattern = 0
        self.max_steps_per_pattern = 10
        
        # Performance tracking
        self.stats = {
            "episodes": 0,
            "total_steps": 0,
            "start_time": None
        }
        
        print("⚡ Ultra-fast training ready!")
    
    def get_ultra_fast_action(self, step: int) -> int:
        """
        Ultra-fast action selection using simple rules
        """
        # Switch patterns based on step count
        if self.steps_in_pattern >= self.max_steps_per_pattern:
            self.steps_in_pattern = 0
            self.pattern_index = 0
            
            # Simple pattern switching
            if step % 50 < 30:
                self.current_pattern = "exploration"
            elif step % 50 < 40:
                self.current_pattern = "dialogue"
            else:
                self.current_pattern = "menu_navigation"
        
        # Get action from current pattern
        pattern = self.action_patterns[self.current_pattern]
        action = pattern[self.pattern_index % len(pattern)]
        
        self.pattern_index += 1
        self.steps_in_pattern += 1
        
        return action
    
    def run_ultra_fast_episode(self, episode_num: int, max_steps: int = 3000) -> Dict[str, Any]:
        """
        Run episode with maximum speed optimizations
        """
        print(f"⚡⚡ ULTRA Episode {episode_num}")
        
        # Reset with minimal operations
        obs = self.env.reset()
        done = False
        step = 0
        episode_reward = 0
        
        episode_start = time.time()
        
        # Pre-allocate for speed
        actions_taken = np.zeros(max_steps, dtype=np.int8)
        
        while not done and step < max_steps:
            # Ultra-fast action decision (no state analysis)
            action = self.get_ultra_fast_action(step)
            actions_taken[step] = action
            
            # Execute action with minimal overhead
            step_result = self.env.step(action)
            if len(step_result) == 5:
                obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                obs, reward, done, info = step_result
            
            episode_reward += reward
            step += 1
            
            # Absolutely minimal logging
            if step % 500 == 0:
                print(f"  ⚡ {step} steps")
        
        episode_duration = time.time() - episode_start
        steps_per_second = step / episode_duration if episode_duration > 0 else 0
        
        self.stats["episodes"] += 1
        self.stats["total_steps"] += step
        
        print(f"⚡⚡ Episode {episode_num}: {step} steps in {episode_duration:.1f}s")
        print(f"    Speed: {steps_per_second:.1f} actions/second")
        
        return {
            "episode": episode_num,
            "steps": step,
            "reward": episode_reward,
            "duration": episode_duration,
            "actions_per_second": steps_per_second
        }
    
    def run_speed_test(self, num_episodes: int = 5):
        """
        Run speed test to measure maximum possible performance
        """
        print(f"⚡⚡⚡ ULTRA-FAST SPEED TEST: {num_episodes} episodes")
        print("=" * 60)
        print("Optimizations enabled:")
        print("  ❌ No LLM calls")
        print("  ❌ No visual analysis") 
        print("  ❌ No database operations")
        print("  ❌ No web monitoring")
        print("  ❌ No artificial delays")
        print("  ✅ Rule-based actions only")
        print("  ✅ Minimal memory allocation")
        print("  ✅ Headless mode")
        print("=" * 60)
        
        self.stats["start_time"] = time.time()
        episode_results = []
        
        for episode in range(1, num_episodes + 1):
            try:
                result = self.run_ultra_fast_episode(episode)
                episode_results.append(result)
                
            except KeyboardInterrupt:
                print("\n⏸️ Speed test interrupted")
                break
            except Exception as e:
                print(f"❌ Episode {episode} error: {e}")
                continue
        
        # Final performance analysis
        total_duration = time.time() - self.stats["start_time"]
        total_steps = self.stats["total_steps"]
        avg_speed = total_steps / total_duration if total_duration > 0 else 0
        
        speeds = [r["actions_per_second"] for r in episode_results]
        max_speed = max(speeds) if speeds else 0
        min_speed = min(speeds) if speeds else 0
        
        print(f"\n⚡⚡⚡ ULTRA-FAST SPEED TEST RESULTS:")
        print(f"Episodes completed: {len(episode_results)}")
        print(f"Total steps: {total_steps}")
        print(f"Total duration: {total_duration:.1f}s")
        print(f"Average speed: {avg_speed:.1f} actions/second")
        print(f"Max speed: {max_speed:.1f} actions/second")
        print(f"Min speed: {min_speed:.1f} actions/second")
        
        if avg_speed > 100:
            print(f"🚀 EXCELLENT: {avg_speed:.0f} actions/sec")
        elif avg_speed > 50:
            print(f"✅ GOOD: {avg_speed:.0f} actions/sec")
        elif avg_speed > 20:
            print(f"⚠️ MODERATE: {avg_speed:.0f} actions/sec")
        else:
            print(f"🐌 SLOW: {avg_speed:.0f} actions/sec")
        
        # Performance recommendations
        print(f"\n💡 Performance Analysis:")
        if avg_speed < 50:
            print("  - Consider reducing emulator frame processing")
            print("  - Check for memory allocations in environment")
            print("  - Verify headless mode is working")
        if max_speed - min_speed > 20:
            print("  - Performance varies significantly between episodes")
            print("  - May indicate GC or memory pressure issues")
        
        # Cleanup
        self.env.close()
        return episode_results


def benchmark_action_speed():
    """
    Benchmark pure action execution speed
    """
    print("🔥 Benchmarking raw action execution speed...")
    
    env = PyBoyPokemonCrystalEnv(
        rom_path="../roms/pokemon_crystal.gbc",
        headless=True,
        debug_mode=False
    )
    
    env.reset()
    
    # Test raw action speed
    num_actions = 1000
    actions = np.random.randint(0, 9, num_actions)
    
    start_time = time.time()
    
    for i, action in enumerate(actions):
        env.step(action)
        if i % 100 == 0:
            print(f"  {i} actions...")
    
    duration = time.time() - start_time
    actions_per_second = num_actions / duration
    
    print(f"🔥 RAW SPEED: {actions_per_second:.1f} actions/second")
    print(f"   ({duration:.2f}s for {num_actions} actions)")
    
    env.close()
    return actions_per_second


def main():
    """Run ultra-fast training or benchmark"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Ultra-Fast Pokemon Crystal Training")
    parser.add_argument("--rom", default="../roms/pokemon_crystal.gbc", help="ROM path")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes") 
    parser.add_argument("--benchmark", action="store_true", help="Run raw speed benchmark")
    
    args = parser.parse_args()
    
    if args.benchmark:
        benchmark_action_speed()
        return
    
    # Run ultra-fast training
    trainer = UltraFastTraining(rom_path=args.rom)
    trainer.run_speed_test(args.episodes)


if __name__ == "__main__":
    main()
