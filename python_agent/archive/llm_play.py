#!/usr/bin/env python3
"""
llm_play.py - LLM-guided Pokemon Crystal gameplay using local Ollama model

This script runs the local LLM agent to play Pokemon Crystal intelligently,
replacing expensive OpenAI API calls with fast local inference.
"""

import argparse
import time
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any

from pyboy_env import PyBoyPokemonCrystalEnv
from local_llm_agent import LocalLLMPokemonAgent


class LLMPokemonPlayer:
    """
    Pokemon Crystal player powered by local LLM
    """
    
    def __init__(self, 
                 rom_path: str = "../pokecrystal.gbc",
                 save_state_path: str = None,
                 model_name: str = "llama3.2:3b",
                 headless: bool = True):
        """Initialize LLM-powered Pokemon player"""
        self.rom_path = rom_path
        self.save_state_path = save_state_path
        self.headless = headless
        
        # Initialize environment
        self.env = PyBoyPokemonCrystalEnv(
            rom_path=rom_path,
            save_state_path=save_state_path,
            headless=headless
        )
        
        # Initialize LLM agent
        self.agent = LocalLLMPokemonAgent(model_name=model_name)
        
        # Gameplay tracking
        self.action_history = []
        self.state_history = []
        self.step_count = 0
        self.start_time = None
        
        print(f"🎮 LLM Pokemon Player initialized")
        print(f"   ROM: {rom_path}")
        print(f"   Save state: {save_state_path if save_state_path else 'None (new game)'}")
        print(f"   LLM Model: {model_name}")
        print(f"   Headless: {headless}")
    
    def play_session(self, max_steps: int = 1000, step_delay: float = 0.1):
        """Run an LLM-guided gameplay session"""
        print(f"\n🚀 Starting LLM gameplay session...")
        print(f"   Max steps: {max_steps}")
        print(f"   Step delay: {step_delay}s")
        
        self.start_time = datetime.now()
        
        try:
            # Reset environment and get initial state
            obs, info = self.env.reset()
            current_state = info.get('raw_state', {})
            
            print(f"\n📍 Initial state:")
            self._print_game_status(current_state)
            
            for step in range(max_steps):
                self.step_count = step
                
                # Get LLM decision
                action = self.agent.decide_next_action(
                    current_state, 
                    self.action_history[-5:]  # Last 5 actions as context
                )
                
                action_name = self.agent.action_map[action]
                
                # Execute action in environment
                obs, reward, terminated, truncated, info = self.env.step(action)
                current_state = info.get('raw_state', {})
                
                # Track action and state
                self.action_history.append(action_name)
                self.state_history.append(current_state)
                
                # Periodic status updates
                if step % 50 == 0:
                    elapsed = datetime.now() - self.start_time
                    print(f"\n⏱️  Step {step} (Elapsed: {elapsed})")
                    print(f"   Action: {action_name}")
                    self._print_game_status(current_state)
                    
                    # Show recent actions
                    recent_actions = ', '.join(self.action_history[-5:])
                    print(f"   Recent actions: {recent_actions}")
                
                # Check for episode end
                if terminated or truncated:
                    print(f"\n🏁 Episode ended at step {step}")
                    print(f"   Terminated: {terminated}")
                    print(f"   Truncated: {truncated}")
                    break
                
                # Add delay to make it observable
                if step_delay > 0:
                    time.sleep(step_delay)
            
            # Session summary
            self._session_summary()
            
        except KeyboardInterrupt:
            print(f"\n⏹️  Gameplay interrupted by user at step {self.step_count}")
            self._session_summary()
        
        except Exception as e:
            print(f"\n❌ Error during gameplay: {e}")
            self._session_summary()
        
        finally:
            self.env.close()
    
    def _print_game_status(self, state: Dict[str, Any]):
        """Print current game status in a readable format"""
        if not state or 'player' not in state:
            print("   Status: No valid game state")
            return
        
        player = state['player']
        party = state.get('party', [])
        
        print(f"   Position: ({player.get('x', 0)}, {player.get('y', 0)}) on Map {player.get('map', 0)}")
        print(f"   Money: ${player.get('money', 0)} | Badges: {player.get('badges', 0)}")
        print(f"   Party: {len(party)} Pokemon")
        
        if party:
            for i, pokemon in enumerate(party[:2]):  # Show first 2
                species = pokemon.get('species', 0)
                species_name = self.agent.pokemon_context['starter_pokemon'].get(species, f"Pokemon #{species}")
                hp = pokemon.get('hp', 0)
                max_hp = pokemon.get('max_hp', 1)
                hp_percent = (hp / max_hp * 100) if max_hp > 0 else 0
                print(f"     #{i+1}: {species_name} L{pokemon.get('level', '?')} ({hp}/{max_hp} HP - {hp_percent:.0f}%)")
    
    def _session_summary(self):
        """Print session summary"""
        if not self.start_time:
            return
        
        elapsed = datetime.now() - self.start_time
        
        print(f"\n📈 === SESSION SUMMARY ===")
        print(f"⏱️  Duration: {elapsed}")
        print(f"🎯 Steps completed: {self.step_count}")
        print(f"⚡ Average steps/second: {self.step_count / elapsed.total_seconds():.2f}")
        
        # Show final memory summary
        memory = self.agent.get_memory_summary()
        print(f"\n💭 Final memory state:")
        print(f"   Decisions stored: {memory['decisions_stored']}")
        print(f"   Game states recorded: {memory['states_recorded']}")
        
        if memory['latest_progress']:
            latest = memory['latest_progress']
            print(f"   Latest progress: {latest['badges']} badges, ${latest['money']}, {latest['party_size']} Pokemon")
        
        print("=" * 50)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='LLM-guided Pokemon Crystal gameplay')
    
    parser.add_argument('--rom-path', type=str, default='../pokecrystal.gbc',
                       help='Path to Pokemon Crystal ROM')
    
    parser.add_argument('--save-state-path', type=str, default='../pokemon_crystal_intro.state',
                       help='Path to save state file')
    
    parser.add_argument('--model', type=str, default='llama3.2:3b',
                       help='Ollama model name to use')
    
    parser.add_argument('--max-steps', type=int, default=1000,
                       help='Maximum steps to run')
    
    parser.add_argument('--step-delay', type=float, default=0.2,
                       help='Delay between actions in seconds')
    
    parser.add_argument('--no-headless', action='store_true',
                       help='Show emulator GUI (not recommended for long runs)')
    
    parser.add_argument('--fast', action='store_true',
                       help='Fast mode - minimal delay and fewer status updates')
    
    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_args()
    
    # Adjust settings for fast mode
    if args.fast:
        args.step_delay = 0.01
        args.max_steps = 10000
    
    print("🧠 Local LLM Pokemon Crystal Player")
    print("=" * 50)
    
    # Initialize player
    player = LLMPokemonPlayer(
        rom_path=args.rom_path,
        save_state_path=args.save_state_path,
        model_name=args.model,
        headless=not args.no_headless
    )
    
    # Start gameplay session
    player.play_session(
        max_steps=args.max_steps,
        step_delay=args.step_delay
    )
    
    print("\n👋 Gameplay session completed!")


if __name__ == "__main__":
    main()
