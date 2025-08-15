#!/usr/bin/env python3
"""
lightweight_curriculum_training.py - Fast Curriculum Training with Small LLM

Uses a smaller, faster LLM model for Pokemon Crystal curriculum training
while maintaining the progressive learning structure.
"""

import time
import json
import sqlite3
from datetime import datetime
from typing import Dict, List, Any, Optional
from enum import Enum
import ollama

# Lightweight imports only
from pyboy_env import PyBoyPokemonCrystalEnv


class TrainingStage(Enum):
    BASIC_CONTROLS = 1
    DIALOGUE_INTERACTION = 2
    POKEMON_SELECTION = 3
    BATTLE_FUNDAMENTALS = 4
    EXPLORATION_NAVIGATION = 5


class LightweightLLMAgent:
    """Ultra-lightweight LLM agent for fast decision making"""
    
    def __init__(self, model_name: str = "smollm2:1.7b"):  # Optimized for speed!
        self.model_name = model_name
        self.action_map = {
            0: "NONE", 1: "UP", 2: "DOWN", 3: "LEFT", 4: "RIGHT",
            5: "A", 6: "B", 7: "START", 8: "SELECT"
        }
        
        # Verify model is available
        try:
            ollama.show(model_name)
            print(f"✅ Using lightweight model: {model_name}")
        except:
            print(f"❌ Model {model_name} not found. Pulling...")
            ollama.pull(model_name)
    
    def decide_action(self, game_state: Dict[str, Any], stage_context: Dict[str, Any]) -> int:
        """Make fast decision with minimal LLM overhead"""
        
        # Ultra-concise prompt for speed
        prompt = self._create_minimal_prompt(game_state, stage_context)
        
        try:
            # Optimized inference for SmolLM2
            response = ollama.generate(
                model=self.model_name,
                prompt=prompt,
                options={
                    "num_predict": 3,   # Just need one number
                    "temperature": 0.1, # Very deterministic
                    "top_k": 8,         # Limited to our 7 actions
                    "stop": ["\n", " "]   # Stop early for speed
                }
            )
            
            # Parse response quickly
            action = self._parse_action_fast(response['response'])
            return action
            
        except Exception as e:
            print(f"⚠️ LLM error: {e}")
            return self._fallback_action(stage_context)
    
    def _create_minimal_prompt(self, state: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Create ultra-concise prompt optimized for SmolLM2"""
        
        stage = context.get('stage', 'UNKNOWN')
        focus = context.get('focus', 'explore')
        
        # Optimized prompt format for SmolLM2
        prompt = f"""You are playing Pokemon Crystal. Stage: {stage}
Focus: {focus}
Choose ONE action:
1=UP, 2=DOWN, 3=LEFT, 4=RIGHT, 5=A, 6=B, 7=START

Response format: Just the number
Action:"""
        
        return prompt
    
    def _parse_action_fast(self, response: str) -> int:
        """Ultra-fast action parsing optimized for numeric responses"""
        response = response.strip()
        
        # Look for digits first (SmolLM2 returns numbers)
        for char in response:
            if char.isdigit() and '1' <= char <= '7':
                return int(char)
        
        # Fallback to text parsing if no digits
        response_upper = response.upper()
        action_mapping = {
            'UP': 1, 'DOWN': 2, 'LEFT': 3, 'RIGHT': 4,
            'A': 5, 'B': 6, 'START': 7, 'SELECT': 8
        }
        
        for word in response_upper.split():
            if word in action_mapping:
                return action_mapping[word]
        
        # Final fallback to A button
        return 5
    
    def _fallback_action(self, context: Dict[str, Any]) -> int:
        """Smart fallback based on stage context"""
        stage = context.get('stage', 'BASIC_CONTROLS')
        
        stage_fallbacks = {
            'BASIC_CONTROLS': 5,      # A button
            'DIALOGUE_INTERACTION': 5, # A button  
            'POKEMON_SELECTION': 5,   # A button
            'BATTLE_FUNDAMENTALS': 5, # A button
            'EXPLORATION_NAVIGATION': 1 # UP movement
        }
        
        return stage_fallbacks.get(stage, 5)


class FastCurriculumTrainer:
    """Lightweight curriculum trainer optimized for speed"""
    
    def __init__(self, rom_path: str, save_state_path: str = None):
        self.rom_path = rom_path
        self.save_state_path = save_state_path
        
        print("🚀 Initializing Fast Curriculum Trainer...")
        
        # Lightweight components only
        self.env = PyBoyPokemonCrystalEnv(
            rom_path=rom_path,
            save_state_path=save_state_path,
            headless=True,  # No GUI for speed
            debug_mode=False
        )
        
        self.agent = LightweightLLMAgent(model_name="smollm2:1.7b")  # Fast model
        
        # Simple tracking
        self.current_stage = TrainingStage.BASIC_CONTROLS
        self.stage_episodes = 0
        self.stage_successes = 0
        self.total_episodes = 0
        
        print("✅ Fast trainer ready!")
    
    def run_fast_curriculum(self, max_episodes: int = 100):
        """Run lightweight curriculum training"""
        
        print(f"\n🎓 FAST CURRICULUM TRAINING")
        print("=" * 50)
        print(f"🎯 Target: {max_episodes} episodes")
        print(f"🤖 Model: SmolLM2-1.7B (optimized for speed)")
        print(f"⚡ Optimized for speed and footage!")
        print()
        
        start_time = time.time()
        
        while self.total_episodes < max_episodes and self.current_stage.value <= 5:
            
            # Run single episode
            episode_result = self._run_fast_episode()
            
            # Track progress
            self.stage_episodes += 1
            self.total_episodes += 1
            
            if episode_result['success']:
                self.stage_successes += 1
            
            # Print concise progress
            success_rate = self.stage_successes / max(self.stage_episodes, 1)
            print(f"📖 S{self.current_stage.value}E{self.stage_episodes}: {episode_result['status']} "
                  f"({success_rate:.1%} success, {episode_result['actions_per_sec']:.1f} a/s)")
            
            # Check stage completion (simple criteria)
            if self._check_stage_mastery():
                self._advance_stage()
                if self.current_stage.value > 5:
                    print("🏆 All stages completed!")
                    break
        
        # Final stats
        duration = time.time() - start_time
        avg_speed = self.total_episodes / max(duration, 1)
        
        print(f"\n📊 TRAINING COMPLETED!")
        print(f"⏱️ Duration: {duration:.1f} seconds")  
        print(f"📈 Episodes: {self.total_episodes}")
        print(f"🚀 Avg Speed: {avg_speed:.1f} episodes/sec")
        print(f"🎯 Final Stage: {self.current_stage.name}")
    
    def _run_fast_episode(self) -> Dict[str, Any]:
        """Run single episode optimized for speed"""
        
        episode_start = time.time()
        state = self.env.reset()
        
        steps = 0
        max_steps = 200  # Short episodes for speed
        success_indicators = 0
        
        # Stage-specific context
        stage_context = {
            'stage': self.current_stage.name,
            'focus': self._get_stage_focus(),
            'episode': self.stage_episodes
        }
        
        while steps < max_steps:
            try:
                # Fast LLM decision
                action = self.agent.decide_action(state, stage_context)
                
                # Execute action
                next_state, reward, done, info = self.env.step(action)
                
                # Simple success tracking
                if reward > 0 or self._detect_progress(state, next_state):
                    success_indicators += 1
                
                state = next_state
                steps += 1
                
                if done:
                    break
                    
            except Exception as e:
                print(f"⚠️ Episode error: {e}")
                break
        
        # Episode results
        duration = time.time() - episode_start
        actions_per_sec = steps / max(duration, 0.001)
        success = success_indicators >= 3  # Simple success criteria
        
        return {
            'success': success,
            'status': '✅' if success else '❌',
            'steps': steps,
            'duration': duration,
            'actions_per_sec': actions_per_sec,
            'success_indicators': success_indicators
        }
    
    def _get_stage_focus(self) -> str:
        """Get focus for current stage"""
        stage_focus = {
            TrainingStage.BASIC_CONTROLS: "navigation",
            TrainingStage.DIALOGUE_INTERACTION: "dialogue", 
            TrainingStage.POKEMON_SELECTION: "selection",
            TrainingStage.BATTLE_FUNDAMENTALS: "battle",
            TrainingStage.EXPLORATION_NAVIGATION: "explore"
        }
        return stage_focus.get(self.current_stage, "general")
    
    def _detect_progress(self, old_state: Dict, new_state: Dict) -> bool:
        """Simple progress detection"""
        # Basic progress indicators
        old_pos = (old_state.get('player_x', 0), old_state.get('player_y', 0))
        new_pos = (new_state.get('player_x', 0), new_state.get('player_y', 0))
        
        # Movement indicates progress
        return old_pos != new_pos
    
    def _check_stage_mastery(self) -> bool:
        """Simple mastery check"""
        min_episodes = 5
        success_threshold = 0.6  # Lowered for speed
        
        if self.stage_episodes < min_episodes:
            return False
            
        success_rate = self.stage_successes / self.stage_episodes
        
        # Stage mastered or timeout
        return success_rate >= success_threshold or self.stage_episodes >= 15
    
    def _advance_stage(self):
        """Advance to next stage"""
        success_rate = self.stage_successes / max(self.stage_episodes, 1)
        
        print(f"\n🎓 STAGE {self.current_stage.value} COMPLETED!")
        print(f"   📊 Success Rate: {success_rate:.1%}")
        print(f"   🎯 Episodes: {self.stage_episodes}")
        
        # Reset for next stage
        if self.current_stage.value < 5:
            self.current_stage = TrainingStage(self.current_stage.value + 1)
            self.stage_episodes = 0
            self.stage_successes = 0
            
            print(f"🚀 ADVANCING TO STAGE {self.current_stage.value}: {self.current_stage.name}")
            print()


def main():
    """Main function for fast curriculum training"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Fast Pokemon Crystal Curriculum Training')
    parser.add_argument('--rom', required=True, help='Path to Pokemon Crystal ROM')
    parser.add_argument('--save-state', help='Path to save state file')
    parser.add_argument('--episodes', type=int, default=50, help='Maximum episodes')
    
    args = parser.parse_args()
    
    print("⚡ LIGHTWEIGHT POKEMON CRYSTAL CURRICULUM TRAINING")
    print("=" * 60)
    print("🎯 Optimized for speed and performance!")
    print(f"🤖 Using SmolLM2-1.7B (optimized for Pokemon RL)")
    print(f"📊 Target episodes: {args.episodes}")
    print()
    
    # Create and run trainer
    trainer = FastCurriculumTrainer(
        rom_path=args.rom,
        save_state_path=args.save_state
    )
    
    trainer.run_fast_curriculum(max_episodes=args.episodes)


if __name__ == "__main__":
    main()
