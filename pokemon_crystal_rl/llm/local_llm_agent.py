"""
local_llm_agent.py - Local LLM-based Pokemon Crystal agent using Ollama

This replaces expensive OpenAI API calls with a local Llama model
optimized for Pokemon gameplay strategy.
"""

import json
import sqlite3
import os
from typing import Dict, List, Any, Optional
from datetime import datetime
import ollama
import numpy as np
from ...core.pyboy_env import PyBoyPokemonCrystalEnv


class LocalLLMPokemonAgent:
    """
    Pokemon Crystal AI agent powered by local Llama model via Ollama
    """
    
    def __init__(self, 
                 model_name: str = "llama3.2:3b",
                 memory_db: str = "../data/pokemon_agent_memory.db"):
        """
        Initialize the local LLM Pokemon agent
        
        Args:
            model_name: Ollama model name to use
            memory_db: SQLite database for episodic memory
        """
        self.model_name = model_name
        self.memory_db = memory_db
        
        # Initialize memory database
        self._init_memory_db()
        
        # Game knowledge context
        self.pokemon_context = {
            "starter_pokemon": {
                155: "Cyndaquil (Fire)",
                158: "Totodile (Water)", 
                152: "Chikorita (Grass)"
            },
            "key_locations": {
                "New Bark Town": "Starting town with Prof. Elm's lab",
                "Route 29": "First route, catch early Pokemon",
                "Cherrygrove City": "First real city, Pokemon Center",
                "Route 30": "Trainer battles, Mr. Pokemon's house",
                "Violet City": "First gym - Falkner (Flying type)",
            },
            "early_pokemon": {
                16: "Pidgey (Normal/Flying)",
                19: "Rattata (Normal)", 
                10: "Caterpie (Bug)",
                13: "Weedle (Bug)",
                129: "Magikarp (Water)"
            }
        }
        
        # Action mapping for game controls
        self.action_map = {
            0: "NONE",    # No action
            1: "UP",      # Move up
            2: "DOWN",    # Move down
            3: "LEFT",    # Move left
            4: "RIGHT",   # Move right
            5: "A",       # Interact/confirm
            6: "B",       # Cancel/back
            7: "START",   # Menu
            8: "SELECT"   # Select button
        }
        
        print(f"🤖 Local LLM Pokemon Agent initialized with {model_name}")
    
    def _init_memory_db(self):
        """Initialize SQLite database for episodic memory"""
        conn = sqlite3.connect(self.memory_db)
        cursor = conn.cursor()
        
        # Create tables for different types of memories
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS game_states (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                player_x INTEGER,
                player_y INTEGER,
                player_map INTEGER,
                party_size INTEGER,
                money INTEGER,
                badges INTEGER,
                game_progress TEXT,
                notes TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS strategic_decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                situation TEXT,
                decision TEXT,
                reasoning TEXT,
                outcome TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS pokemon_encounters (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                species INTEGER,
                level INTEGER,
                location TEXT,
                caught BOOLEAN,
                notes TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _query_local_llm(self, prompt: str, system_prompt: str = None) -> str:
        """Query the local Llama model via Ollama"""
        try:
            messages = []
            
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            
            messages.append({"role": "user", "content": prompt})
            
            response = ollama.chat(
                model=self.model_name,
                messages=messages,
                options={
                    "temperature": 0.3,  # Low temperature for consistent strategic decisions
                    "top_p": 0.9,
                    "num_predict": 200,  # Limit response length for speed
                }
            )
            
            return response['message']['content'].strip()
            
        except Exception as e:
            print(f"Error querying local LLM: {e}")
            return "ERROR: Could not get LLM response"
    
    def analyze_game_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze current game state and determine strategic context"""
        analysis = {
            "phase": "unknown",
            "immediate_goals": [],
            "threats": [],
            "opportunities": [],
            "next_actions": []
        }
        
        if not state or 'player' not in state:
            return analysis
        
        player = state['player']
        party = state.get('party', [])
        
        # Determine game phase
        if len(party) == 0:
            analysis["phase"] = "intro/starter_selection"
            analysis["immediate_goals"] = ["Get starter Pokemon", "Leave Professor's lab"]
        elif len(party) == 1 and player.get('badges', 0) == 0:
            analysis["phase"] = "early_game"
            analysis["immediate_goals"] = ["Catch more Pokemon", "Train team", "Head to first gym"]
        elif player.get('badges', 0) > 0:
            analysis["phase"] = "gym_progression"
            analysis["immediate_goals"] = [f"Progress with {player.get('badges', 0)} badges"]
        
        # Check party status
        if party:
            low_hp_pokemon = [p for p in party if p.get('hp', 0) < p.get('max_hp', 1) * 0.3]
            if low_hp_pokemon:
                analysis["threats"].append("Pokemon with low HP need healing")
        
        # Money situation
        money = player.get('money', 0)
        if money < 1000:
            analysis["threats"].append("Low on money")
        elif money > 10000:
            analysis["opportunities"].append("Good money for items/pokeballs")
        
        return analysis
    
    def decide_next_action(self, state: Dict[str, Any], recent_history: List[str] = None) -> int:
        """Use local LLM to decide next action based on game state"""
        
        # Analyze current situation
        analysis = self.analyze_game_state(state)
        
        # Build context for LLM
        context_prompt = self._build_strategy_prompt(state, analysis, recent_history)
        
        system_prompt = """You are a Pokemon Crystal gameplay expert. You help players make strategic decisions.
        
IMPORTANT RULES:
1. Be concise - give short, direct answers
2. Focus on immediate next action, not long-term strategy  
3. Consider the current game state and respond appropriately
4. Your response should end with ONE of these actions: UP, DOWN, LEFT, RIGHT, A, B, START, SELECT, NONE

Actions explained:
- Movement: UP/DOWN/LEFT/RIGHT to navigate
- A: Interact with objects/people, confirm in menus, attack in battle
- B: Cancel, go back, run from wild Pokemon
- START: Open main menu
- SELECT: Usually unused
- NONE: Wait/do nothing"""

        # Query local LLM
        llm_response = self._query_local_llm(context_prompt, system_prompt)
        
        # Parse response to extract action
        action = self._parse_action_from_response(llm_response)
        
        # Store decision for learning
        self._store_decision(state, llm_response, action, analysis)
        
        return action
    
    def _build_strategy_prompt(self, state: Dict[str, Any], analysis: Dict[str, Any], recent_history: List[str] = None) -> str:
        """Build a focused prompt for the local LLM"""
        
        player = state.get('player', {})
        party = state.get('party', [])
        
        prompt = f"""Pokemon Crystal - Current Situation:

LOCATION: Map {player.get('map', 0)}, Position ({player.get('x', 0)}, {player.get('y', 0)})
MONEY: ${player.get('money', 0)}
BADGES: {player.get('badges', 0)}

TEAM: {len(party)} Pokemon"""
        
        if party:
            for i, pokemon in enumerate(party[:3]):  # Show first 3
                species = pokemon.get('species', 0)
                species_name = self.pokemon_context['starter_pokemon'].get(species, f"Pokemon #{species}")
                prompt += f"\n  {i+1}. {species_name} - Level {pokemon.get('level', '?')} (HP: {pokemon.get('hp', 0)}/{pokemon.get('max_hp', 0)})"
        
        prompt += f"\n\nGAME PHASE: {analysis['phase']}"
        prompt += f"\nGOALS: {', '.join(analysis['immediate_goals'])}"
        
        if analysis['threats']:
            prompt += f"\nTHREATS: {', '.join(analysis['threats'])}"
        
        if analysis['opportunities']:
            prompt += f"\nOPPORTUNITIES: {', '.join(analysis['opportunities'])}"
        
        if recent_history:
            prompt += f"\nRECENT ACTIONS: {', '.join(recent_history[-3:])}"
        
        prompt += "\n\nWhat should I do next? Give me ONE action to take right now."
        
        return prompt
    
    def _parse_action_from_response(self, response: str) -> int:
        """Parse LLM response to extract action command"""
        response_upper = response.upper()
        
        # Look for action keywords in the response
        if "UP" in response_upper:
            return 1
        elif "DOWN" in response_upper:
            return 2
        elif "LEFT" in response_upper:
            return 3
        elif "RIGHT" in response_upper:
            return 4
        elif " A " in response_upper or response_upper.endswith(" A") or "PRESS A" in response_upper:
            return 5
        elif " B " in response_upper or response_upper.endswith(" B") or "PRESS B" in response_upper:
            return 6
        elif "START" in response_upper or "MENU" in response_upper:
            return 7
        elif "SELECT" in response_upper:
            return 8
        else:
            # Default to no action if unclear
            return 0
    
    def _store_decision(self, state: Dict[str, Any], llm_response: str, action: int, analysis: Dict[str, Any]):
        """Store decision in memory for future learning"""
        conn = sqlite3.connect(self.memory_db)
        cursor = conn.cursor()
        
        # Store strategic decision
        cursor.execute('''
            INSERT INTO strategic_decisions (timestamp, situation, decision, reasoning, outcome)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            json.dumps(analysis),
            self.action_map[action],
            llm_response,
            "pending"  # Will be updated later based on results
        ))
        
        # Store game state snapshot
        player = state.get('player', {})
        cursor.execute('''
            INSERT INTO game_states (timestamp, player_x, player_y, player_map, 
                                   party_size, money, badges, game_progress, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            player.get('x', 0),
            player.get('y', 0),
            player.get('map', 0),
            len(state.get('party', [])),
            player.get('money', 0),
            player.get('badges', 0),
            analysis['phase'],
            f"Action decided: {self.action_map[action]}"
        ))
        
        conn.commit()
        conn.close()
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """Get summary of stored memories for analysis"""
        conn = sqlite3.connect(self.memory_db)
        cursor = conn.cursor()
        
        # Count different types of memories
        cursor.execute("SELECT COUNT(*) FROM strategic_decisions")
        decision_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM game_states")
        state_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM pokemon_encounters")
        encounter_count = cursor.fetchone()[0]
        
        # Get recent progress
        cursor.execute("""
            SELECT badges, money, party_size 
            FROM game_states 
            ORDER BY timestamp DESC 
            LIMIT 1
        """)
        latest = cursor.fetchone()
        
        conn.close()
        
        return {
            "decisions_stored": decision_count,
            "states_recorded": state_count,
            "pokemon_encounters": encounter_count,
            "latest_progress": {
                "badges": latest[0] if latest else 0,
                "money": latest[1] if latest else 0,
                "party_size": latest[2] if latest else 0
            } if latest else None
        }


def test_local_llm_agent():
    """Test the local LLM agent with a mock game state"""
    print("🧪 Testing Local LLM Pokemon Agent...")
    
    agent = LocalLLMPokemonAgent()
    
    # Mock game state - early game scenario
    mock_state = {
        "player": {
            "x": 5,
            "y": 10,
            "map": 1,
            "money": 3000,
            "badges": 0
        },
        "party": [
            {
                "species": 155,  # Cyndaquil
                "level": 8,
                "hp": 25,
                "max_hp": 30,
                "status": 0
            }
        ]
    }
    
    print("📊 Mock game state:", json.dumps(mock_state, indent=2))
    
    # Get strategic analysis
    analysis = agent.analyze_game_state(mock_state)
    print("🔍 Analysis:", json.dumps(analysis, indent=2))
    
    # Get next action decision
    action = agent.decide_next_action(mock_state, ["UP", "UP", "A"])
    print(f"🎯 Decided action: {action} ({agent.action_map[action]})")
    
    # Show memory summary
    memory = agent.get_memory_summary()
    print("💭 Memory summary:", json.dumps(memory, indent=2))
    
    print("\n✅ Local LLM agent test completed!")


class LLMManager:
    """Manager class for interfacing with local LLM"""
    
    def __init__(self, model: str = None, interval: int = 10):
        """Initialize the LLM manager.
        
        Args:
            model: Name of the local LLM model to use
            interval: How often to query the LLM
        """
        self.model = model
        self.interval = interval
        self.agent = LocalLLMPokemonAgent(model_name=model)
        self.response_times = []
        self.window_size = 10
    
    def get_action(self) -> int:
        """Get the next action from the LLM."""
        # Mock action for now - this will be implemented later
        return 5  # Default to A button


if __name__ == "__main__":
    test_local_llm_agent()
