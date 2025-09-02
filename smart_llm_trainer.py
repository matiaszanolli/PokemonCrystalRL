#!/usr/bin/env python3
"""
Smart LLM Trainer for Pokemon Crystal RL

This trainer uses advanced strategic context analysis to make intelligent 
decisions based on comprehensive understanding of game state variables.
"""

import time
import numpy as np
from pyboy import PyBoy
import json
import signal
import sys
import os
import threading
from datetime import datetime
import requests
from typing import Dict, List, Tuple, Optional, Any
import argparse

# Import our enhanced systems
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core.game_state_analyzer import GameStateAnalyzer, GameStateAnalysis
from core.strategic_context_builder import StrategicContextBuilder, DecisionContext

class SmartLLMAgent:
    """Enhanced LLM agent with strategic decision-making capabilities"""
    
    def __init__(self, model_name="smollm2:1.7b", base_url="http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
        self.context_builder = StrategicContextBuilder()
        
        # Test LLM connection
        self.llm_available = self._test_llm_connection()
        if not self.llm_available:
            print("⚠️ LLM not available - will use rule-based fallbacks")
    
    def _test_llm_connection(self) -> bool:
        """Test if LLM is available"""
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": "Test",
                    "stream": False,
                    "options": {"num_predict": 1}
                },
                timeout=5
            )
            return response.status_code == 200
        except:
            return False
    
    def decide(self, raw_game_state: Dict, last_action: str = None, 
              last_reward: float = None) -> Tuple[str, str, DecisionContext]:
        """
        Make an intelligent decision based on comprehensive state analysis
        
        Returns:
            Tuple of (action, reasoning, decision_context)
        """
        # Build comprehensive decision context
        context = self.context_builder.build_context(
            raw_game_state, last_action, last_reward
        )
        
        # Use LLM if available, otherwise fall back to rule-based
        if self.llm_available:
            action, reasoning = self._llm_decision(context)
        else:
            action, reasoning = self._rule_based_decision(context)
        
        return action, reasoning, context
    
    def _llm_decision(self, context: DecisionContext) -> Tuple[str, str]:
        """Get decision from LLM using enhanced context"""
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": context.complete_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,  # Lower temperature for more consistent decisions
                        "num_predict": 100,
                        "stop": ["\n\n", "Reasoning:"]
                    }
                },
                timeout=15
            )
            
            if response.status_code == 200:
                llm_response = response.json().get('response', '').strip()
                action, reasoning = self._parse_llm_response(llm_response)
                return action, reasoning
            
        except Exception as e:
            print(f"LLM Error: {e}")
        
        # Fallback to rule-based decision
        return self._rule_based_decision(context)
    
    def _parse_llm_response(self, response: str) -> Tuple[str, str]:
        """Parse LLM response to extract action and reasoning"""
        action_words = {
            'up', 'down', 'left', 'right', 'a', 'b', 'start', 'select',
            'north', 'south', 'east', 'west', 'interact', 'confirm', 
            'cancel', 'back', 'menu', 'flee', 'run', 'attack'
        }
        
        response_lower = response.lower()
        
        # Look for explicit action mentions
        for word in action_words:
            if word in response_lower:
                if word in ['up', 'north']:
                    return 'up', response
                elif word in ['down', 'south']:
                    return 'down', response
                elif word in ['left', 'west']:
                    return 'left', response
                elif word in ['right', 'east']:
                    return 'right', response
                elif word in ['a', 'interact', 'confirm', 'attack']:
                    return 'a', response
                elif word in ['b', 'cancel', 'back', 'flee', 'run']:
                    return 'b', response
                elif word in ['start', 'menu']:
                    return 'start', response
                elif word == 'select':
                    return 'select', response
        
        # Default to 'a' if no clear action found
        return 'a', f"Defaulted to A button. LLM response: {response}"
    
    def _rule_based_decision(self, context: DecisionContext) -> Tuple[str, str]:
        """Rule-based decision making when LLM unavailable"""
        analysis = context.current_analysis
        
        # Emergency situations
        if context.emergency_actions:
            action = context.emergency_actions[0]
            return action, f"Emergency action: {action} to address critical situation"
        
        # Stuck pattern breaking
        recent_actions = context.recent_actions[-5:]
        if len(recent_actions) >= 3 and len(set(recent_actions)) <= 2:
            # Try an action not recently used
            all_actions = ['up', 'down', 'left', 'right', 'a', 'b']
            unused_actions = [a for a in all_actions if a not in recent_actions[-3:]]
            if unused_actions:
                action = unused_actions[0]
                return action, f"Breaking stuck pattern by trying {action}"
        
        # Phase-specific logic
        if analysis.phase.value == 'early_game':
            # Need to get to Prof. Elm's lab
            preferred_actions = ['a', 'up', 'right', 'down']  # Interaction and movement
        elif analysis.phase.value == 'starter_phase':
            # Focus on training and exploration
            preferred_actions = ['a', 'up', 'down', 'left', 'right']
        else:
            # General exploration
            preferred_actions = ['a', 'up', 'down', 'left', 'right', 'start']
        
        # Choose first preferred action not recently used
        for action in preferred_actions:
            if action not in recent_actions[-2:]:  # Not used in last 2 actions
                return action, f"Rule-based choice: {action} for {analysis.phase.value}"
        
        return 'a', "Default rule-based action"

class SmartPokemonTrainer:
    """Main trainer class with enhanced strategic intelligence"""
    
    def __init__(self, rom_path: str, max_actions: int = 2000, 
                 llm_model: str = "smollm2:1.7b", llm_interval: int = 10):
        self.rom_path = rom_path
        self.max_actions = max_actions
        self.llm_interval = llm_interval
        
        self.pyboy = None
        self.agent = SmartLLMAgent(llm_model)
        
        # Training state
        self.actions_taken = 0
        self.total_reward = 0.0
        self.start_time = time.time()
        self.running = True
        
        # Enhanced logging
        self.decision_log = []
        self.performance_log = []
        
        # Setup graceful shutdown
        signal.signal(signal.SIGINT, self.shutdown)
        signal.signal(signal.SIGTERM, self.shutdown)
    
    def shutdown(self, signum, frame):
        """Graceful shutdown"""
        print(f"\\n⏸️ Shutting down smart trainer...")
        self.running = False
        if self.pyboy:
            self.pyboy.stop()
        self._save_results()
        sys.exit(0)
    
    def train(self):
        """Run the enhanced training loop"""
        print("🚀 Starting Smart Pokemon Crystal RL Training")
        print("=" * 60)
        print(f"🧠 LLM Model: {self.agent.model_name}")
        print(f"🎯 Max Actions: {self.max_actions}")
        print(f"⚡ LLM Decision Interval: Every {self.llm_interval} actions")
        print("=" * 60)
        
        # Initialize PyBoy
        self._initialize_pyboy()
        
        last_action = None
        last_reward = 0.0
        
        try:
            while self.running and self.actions_taken < self.max_actions:
                # Get current game state
                game_state = self._get_game_state()
                
                # Make decision (LLM every N actions, rule-based otherwise)
                use_llm = (self.actions_taken % self.llm_interval == 0) or self.actions_taken == 0
                
                if use_llm and self.agent.llm_available:
                    action, reasoning, context = self.agent.decide(
                        game_state, last_action, last_reward
                    )
                    decision_type = "LLM"
                    
                    # Log detailed decision
                    self.decision_log.append({
                        'step': self.actions_taken,
                        'action': action,
                        'reasoning': reasoning,
                        'context_summary': context.current_analysis.situation_summary,
                        'criticality': context.current_analysis.criticality.value,
                        'phase': context.current_analysis.phase.value,
                        'health': context.current_analysis.health_percentage,
                        'threats': context.current_analysis.immediate_threats,
                        'opportunities': context.current_analysis.opportunities
                    })
                else:
                    # Simple rule-based decision for intermediate steps
                    action, reasoning, context = self.agent.decide(
                        game_state, last_action, last_reward
                    )
                    decision_type = "Rule"
                
                # Execute action
                reward = self._execute_action(action)
                
                # Update state
                last_action = action
                last_reward = reward
                self.total_reward += reward
                self.actions_taken += 1
                
                # Log performance
                if self.actions_taken % 50 == 0:
                    self._log_performance(context)
                
                # Display progress
                if self.actions_taken % 10 == 0 or use_llm:
                    self._display_progress(action, reasoning, decision_type, 
                                         context, reward)
                
                time.sleep(0.1)  # Small delay to prevent overwhelming
        
        except Exception as e:
            print(f"Training error: {e}")
        finally:
            self._save_results()
    
    def _initialize_pyboy(self):
        """Initialize PyBoy emulator"""
        print("🎮 Initializing PyBoy...")
        
        try:
            self.pyboy = PyBoy(self.rom_path, window_type="headless")
            
            # Try to load save state
            save_state_path = self.rom_path + ".state"
            if os.path.exists(save_state_path):
                with open(save_state_path, 'rb') as f:
                    self.pyboy.load_state(f)
                print(f"💾 Loaded save state: {save_state_path}")
            
            print("✅ PyBoy initialized successfully")
        except Exception as e:
            print(f"❌ PyBoy initialization failed: {e}")
            sys.exit(1)
    
    def _get_game_state(self) -> Dict:
        """Extract comprehensive game state from memory"""
        if not self.pyboy:
            return {}
        
        memory = self.pyboy.memory
        
        # Build comprehensive state using memory addresses from llm_trainer.py
        try:
            # Party data
            party = []
            party_count = memory[0xD163] if memory[0xD163] <= 6 else 0
            
            for i in range(min(party_count, 6)):
                base = 0xD163 + i * 44
                try:
                    species = memory[base]
                    hp = memory[base + 4] + (memory[base + 5] << 8)
                    max_hp = memory[base + 6] + (memory[base + 7] << 8)
                    level = memory[base + 8]
                    
                    party.append({
                        "species": species,
                        "hp": hp,
                        "max_hp": max_hp,
                        "level": level
                    })
                except:
                    break
            
            # Location and coordinates
            player_x = memory[0xDCB8]
            player_y = memory[0xDCB9] 
            player_map = memory[0xDCBA]
            
            # Money (3-byte little-endian)
            money = memory[0xD347] + (memory[0xD348] << 8) + (memory[0xD349] << 16)
            
            # Badges
            badges = memory[0xD359]
            badge_count = bin(badges).count('1')
            
            # Battle state
            in_battle = bool(memory[0xD057])
            
            return {
                "party": party,
                "party_count": party_count,
                "player_hp": party[0]["hp"] if party else 0,
                "player_max_hp": party[0]["max_hp"] if party else 0,
                "player_level": party[0]["level"] if party else 0,
                "player_x": player_x,
                "player_y": player_y,
                "player_map": player_map,
                "money": money,
                "badges": badges,
                "badge_count": badge_count,
                "in_battle": in_battle
            }
        
        except Exception as e:
            print(f"Error reading game state: {e}")
            return {}
    
    def _execute_action(self, action: str) -> float:
        """Execute action and return reward"""
        if not self.pyboy:
            return 0.0
        
        # Map action to button names
        valid_actions = ['up', 'down', 'left', 'right', 'a', 'b', 'start', 'select']
        
        # Execute action using PyBoy's button press/release methods
        if action in valid_actions:
            # Press button
            self.pyboy.button_press(action)
            
            # Hold for a couple frames
            for _ in range(2):
                self.pyboy.tick()
            
            # Release button
            self.pyboy.button_release(action)
            
            # Wait a few more frames for processing
            for _ in range(4):
                self.pyboy.tick()
        
        # Simple reward calculation (can be enhanced)
        return self._calculate_reward()
    
    def _calculate_reward(self) -> float:
        """Calculate reward for current state (simplified version)"""
        # This is a simplified version - the full reward system
        # from the original trainer could be integrated here
        return -0.01  # Small time penalty
    
    def _log_performance(self, context: DecisionContext):
        """Log performance metrics"""
        elapsed = time.time() - self.start_time
        aps = self.actions_taken / elapsed if elapsed > 0 else 0
        
        self.performance_log.append({
            'step': self.actions_taken,
            'elapsed_time': elapsed,
            'actions_per_second': aps,
            'total_reward': self.total_reward,
            'avg_reward': self.total_reward / max(self.actions_taken, 1),
            'phase': context.current_analysis.phase.value,
            'criticality': context.current_analysis.criticality.value,
            'health': context.current_analysis.health_percentage,
            'progression': context.current_analysis.progression_score
        })
    
    def _display_progress(self, action: str, reasoning: str, decision_type: str,
                         context: DecisionContext, reward: float):
        """Display training progress"""
        elapsed = time.time() - self.start_time
        aps = self.actions_taken / elapsed if elapsed > 0 else 0
        
        print(f"\\n⚡ Step {self.actions_taken}/{self.max_actions} | "
              f"{action.upper()} ({decision_type})")
        print(f"   📊 {aps:.1f} a/s | "
              f"Phase: {context.current_analysis.phase.value} | "
              f"Health: {context.current_analysis.health_percentage:.0f}%")
        print(f"   💰 Reward: {reward:.3f} (Total: {self.total_reward:.2f})")
        
        if context.current_analysis.immediate_threats:
            threats_to_show = context.current_analysis.immediate_threats[:2] if len(context.current_analysis.immediate_threats) > 1 else context.current_analysis.immediate_threats
            print(f"   ⚠️ Threats: {', '.join(threats_to_show)}")
        
        if context.current_analysis.opportunities:
            opps_to_show = context.current_analysis.opportunities[:2] if len(context.current_analysis.opportunities) > 1 else context.current_analysis.opportunities
            print(f"   🎯 Opportunities: {', '.join(opps_to_show)}")
        
        if decision_type == "LLM":
            print(f"   🧠 Reasoning: {reasoning[:100]}...")
    
    def _save_results(self):
        """Save training results and analysis"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save decision log
        decision_file = f"logs/smart_decisions_{timestamp}.json"
        os.makedirs("logs", exist_ok=True)
        
        with open(decision_file, 'w') as f:
            json.dump(self.decision_log, f, indent=2)
        
        # Save performance log
        performance_file = f"logs/smart_performance_{timestamp}.json"
        with open(performance_file, 'w') as f:
            json.dump(self.performance_log, f, indent=2)
        
        # Save summary
        summary = {
            'total_actions': self.actions_taken,
            'total_reward': self.total_reward,
            'training_time': time.time() - self.start_time,
            'avg_reward_per_action': self.total_reward / max(self.actions_taken, 1),
            'llm_decisions_made': len(self.decision_log),
            'final_performance': self.performance_log[-1] if self.performance_log else {}
        }
        
        summary_file = f"logs/smart_summary_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\\n📊 Results saved:")
        print(f"   - Decisions: {decision_file}")
        print(f"   - Performance: {performance_file}")
        print(f"   - Summary: {summary_file}")

def main():
    parser = argparse.ArgumentParser(description="Smart LLM Pokemon Crystal RL Trainer")
    parser.add_argument("--rom", default="roms/pokemon_crystal.gbc",
                       help="ROM file path")
    parser.add_argument("--actions", type=int, default=2000,
                       help="Number of actions to execute")
    parser.add_argument("--llm-model", default="smollm2:1.7b",
                       choices=["smollm2:1.7b", "llama3.2:1b", "llama3.2:3b", "deepseek-coder:latest"],
                       help="LLM model to use")
    parser.add_argument("--llm-interval", type=int, default=10,
                       help="Actions between LLM decisions")
    
    args = parser.parse_args()
    
    # Verify ROM exists
    if not os.path.exists(args.rom):
        print(f"❌ ROM file not found: {args.rom}")
        sys.exit(1)
    
    # Create trainer and start training
    trainer = SmartPokemonTrainer(
        rom_path=args.rom,
        max_actions=args.actions,
        llm_model=args.llm_model,
        llm_interval=args.llm_interval
    )
    
    trainer.train()

if __name__ == "__main__":
    main()
