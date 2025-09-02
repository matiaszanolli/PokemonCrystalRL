#!/usr/bin/env python3
"""
LLM-Enhanced Pokemon Crystal RL Training Script

An advanced training script that combines:
- Local LLM integration for intelligent decision making
- Sophisticated reward function based on Pokemon game progress
- Memory map integration for game state analysis
- Web monitoring with LLM decision tracking
"""

import time
import numpy as np
from pyboy import PyBoy
import json
import signal
import sys
import os
import threading
import io
from datetime import datetime
from PIL import Image
from http.server import HTTPServer, BaseHTTPRequestHandler
import requests
from typing import Dict, List, Tuple, Optional

# Import our accurate game state system
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from accurate_game_state import AccurateGameState

# Memory address mappings for Pokemon Crystal (VALIDATED addresses from your analysis)
MEMORY_ADDRESSES = {
    # Party and Pokemon data - Based on your party structure analysis
    'party_count': 0xD163,      # Number of Pokemon in party
    'player_species': 0xD163,   # Species of first Pokemon (party slot 0 + 0)
    'player_held_item': 0xD164, # Held item of first Pokemon (party slot 0 + 1)
    'player_hp': 0xD167,        # Current HP of first Pokemon (party slot 0 + 4, low byte)
    'player_hp_high': 0xD168,   # Current HP of first Pokemon (party slot 0 + 4, high byte)
    'player_max_hp': 0xD169,    # Max HP of first Pokemon (party slot 0 + 6, low byte)
    'player_max_hp_high': 0xD16A, # Max HP of first Pokemon (party slot 0 + 6, high byte)
    'player_level': 0xD16B,     # Level of first Pokemon (party slot 0 + 8)
    'player_status': 0xD16C,    # Status condition of first Pokemon (party slot 0 + 9)
    
    # Location and movement - From your coordinate analysis
    'player_map': 0xD35D,       # Current map ID
    'player_x': 0xD361,         # Player X coordinate
    'player_y': 0xD362,         # Player Y coordinate
    'player_direction': 0xD363, # Direction player is facing (0=down,1=up,2=left,3=right)
    
    # Resources and progress - From your money/badge analysis
    'money_low': 0xD347,        # Money (low byte, 3 bytes little-endian)
    'money_mid': 0xD348,        # Money (mid byte)
    'money_high': 0xD349,       # Money (high byte)
    'badges': 0xD359,           # Badge flags (bit flags for 8 Johto badges)
    
    # Battle state - From your battle analysis
    'in_battle': 0xD057,        # Battle active flag (0=overworld, 1=battle)
    'battle_turn': 0xD068,      # Turn counter in battle
    'enemy_species': 0xD0A5,    # Opponent Pokemon species
    'enemy_hp_low': 0xD0A8,     # Opponent HP (low byte, 2 bytes)
    'enemy_hp_high': 0xD0A9,    # Opponent HP (high byte)
    'enemy_level': 0xD0AA,      # Opponent Pokemon level
    'player_active_slot': 0xD05E, # Player active Pokemon slot (0-5)
    'move_selected': 0xD05F,    # Move selected (0-3)
    
    # Misc useful - From your misc analysis
    'step_counter': 0xD164,     # Step counter for movement tracking
    'game_time_hours': 0xD3E1,  # Time played (hours)
}

# Function to build complete observation using your validated structure
def build_observation(memory) -> Dict:
    """Build complete game state observation using validated memory addresses"""
    
    # Party data - using your party structure (44 bytes per Pokemon)
    party = []
    party_count = memory[0xD163] if memory[0xD163] <= 6 else 0  # Validate party count
    
    for i in range(6):  # Always check all 6 slots
        base = 0xD163 + i * 44
        try:
            species = memory[base] if i < party_count else 0
            held_item = memory[base + 1] if i < party_count else 0
            hp = memory[base + 4] + (memory[base + 5] << 8) if i < party_count else 0
            max_hp = memory[base + 6] + (memory[base + 7] << 8) if i < party_count else 0
            level = memory[base + 8] if i < party_count else 0
            status = memory[base + 9] if i < party_count else 0
            moves = [memory[base + 10 + j] for j in range(4)] if i < party_count else [0, 0, 0, 0]
            pp = [memory[base + 14 + j] for j in range(4)] if i < party_count else [0, 0, 0, 0]
            
            party.append({
                "species": species,
                "held_item": held_item,
                "hp": hp,
                "max_hp": max_hp,
                "level": level,
                "status": status,
                "moves": moves,
                "pp": pp
            })
        except:
            # Fallback for any memory read errors
            party.append({
                "species": 0, "held_item": 0, "hp": 0, "max_hp": 0,
                "level": 0, "status": 0, "moves": [0, 0, 0, 0], "pp": [0, 0, 0, 0]
            })
    
    # Money - using your 3-byte little-endian structure
    try:
        money = memory[0xD347] + (memory[0xD348] << 8) + (memory[0xD349] << 16)
    except:
        money = 0
    
    # Location and coordinates - using your validated addresses
    try:
        map_id = memory[0xD35D]
        player_x = memory[0xD361]
        player_y = memory[0xD362]
        facing = memory[0xD363]
    except:
        map_id = player_x = player_y = facing = 0
    
    # Battle state - using your battle structure
    try:
        battle_flag = memory[0xD057]
        turn_count = memory[0xD068] if battle_flag else 0
        enemy_species = memory[0xD0A5] if battle_flag else 0
        enemy_hp = memory[0xD0A8] + (memory[0xD0A9] << 8) if battle_flag else 0
        enemy_level = memory[0xD0AA] if battle_flag else 0
    except:
        battle_flag = turn_count = enemy_species = enemy_hp = enemy_level = 0
    
    # Badge and progression - using your badge structure
    try:
        badges = memory[0xD359]
        badges_count = bin(badges).count('1')
    except:
        badges = badges_count = 0
    
    # Step counter for exploration
    try:
        step_counter = memory[0xD164]
    except:
        step_counter = 0
    
    # Compile complete state
    return {
        "party": party,
        "party_count": party_count,
        "money": money,
        "badges": badges,
        "badges_count": badges_count,
        "badges_total": badges_count,  # For compatibility
        "map_id": map_id,
        "player_map": map_id,  # For compatibility
        "coords": (player_x, player_y),
        "player_x": player_x,
        "player_y": player_y,
        "facing": facing,
        "player_direction": facing,
        "in_battle": bool(battle_flag),
        "battle_turn": turn_count,
        "enemy_species": enemy_species,
        "enemy_hp": enemy_hp,
        "enemy_level": enemy_level,
        "step_counter": step_counter,
        
        # Derived values for first Pokemon (main player stats)
        "player_species": party[0]["species"] if party_count > 0 else 0,
        "player_hp": party[0]["hp"] if party_count > 0 else 0,
        "player_max_hp": party[0]["max_hp"] if party_count > 0 else 0,
        "player_level": party[0]["level"] if party_count > 0 else 0,
        "player_status": party[0]["status"] if party_count > 0 else 0,
        "has_pokemon": party_count > 0,
        "health_percentage": (party[0]["hp"] / max(party[0]["max_hp"], 1)) * 100 if party_count > 0 else 0,
    }

# Derived values calculated from memory addresses
DERIVED_VALUES = {
    'badges_total': lambda state: state.get('badges_count', 0),
    'health_percentage': lambda state: (state.get('player_hp', 0) / max(state.get('player_max_hp', 1), 1)) * 100,
    'has_pokemon': lambda state: state.get('party_count', 0) > 0,
    'location_key': lambda state: f"{state.get('player_map', 0)}_{state.get('player_x', 0)}_{state.get('player_y', 0)}",
}

# Important locations in the game
IMPORTANT_LOCATIONS = {
    24: "Player's Bedroom",
    25: "Player's House", 
    26: "New Bark Town",
    27: "Prof. Elm's Lab",
    28: "Route 29",
    29: "Route 30",
    30: "Cherrygrove City",
}

# Pokemon species IDs (partial list for important ones)
POKEMON_SPECIES = {
    0: "None",
    152: "Chikorita",
    155: "Cyndaquil", 
    158: "Totodile",
    16: "Pidgey",
    19: "Rattata",
    129: "Magikarp",
}

# Status conditions
STATUS_CONDITIONS = {
    0: "Healthy",
    1: "Sleep",
    2: "Poison",
    3: "Burn",
    4: "Freeze",
    5: "Paralysis",
}

# Badge masks for checking individual badges
BADGE_MASKS = {
    'johto': {
        'zephyr': 0x01, 'hive': 0x02, 'plain': 0x04, 'fog': 0x08,
        'storm': 0x10, 'mineral': 0x20, 'glacier': 0x40, 'rising': 0x80
    }
}

def get_badges_earned(badges_byte: int) -> list:
    """Get list of badges earned from badge byte"""
    earned = []
    
    # Check Johto badges using your bit flag structure
    for badge_name, mask in BADGE_MASKS['johto'].items():
        if badges_byte & mask:
            earned.append(f"johto_{badge_name}")
    
    return earned

# Import game intelligence system
from core.game_intelligence import GameIntelligence, GameContext, ActionPlan
from core.experience_memory import ExperienceMemory

# Import DQN agent
from core.dqn_agent import DQNAgent, HybridAgent

class LLMAgent:
    """Local LLM agent for Pokemon Crystal decision making"""
    
    def __init__(self, model_name="smollm2:1.7b", base_url="http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
        self.decision_history = []
        self.last_decision_time = 0
        
        # Initialize game intelligence system
        self.game_intelligence = GameIntelligence()
        
        # Initialize experience memory system
        self.experience_memory = ExperienceMemory()
        
        # Test LLM availability
        self.available = self._test_llm_connection()
        
    def _test_llm_connection(self) -> bool:
        """Test if LLM is available"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def get_decision(self, game_state: Dict, screen_analysis: Dict, recent_actions: List[str]) -> Tuple[str, str]:
        """Get LLM decision based on game state"""
        if not self.available:
            return self._fallback_decision(game_state), "LLM unavailable - using fallback"
        
        try:
            prompt = self._build_prompt(game_state, screen_analysis, recent_actions)
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "num_predict": 50
                    }
                },
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                action = self._parse_llm_response(result.get('response', ''))
                reasoning = result.get('response', '').strip()
                
                # Track decision
                self.decision_history.append({
                    'timestamp': time.time(),
                    'action': action,
                    'reasoning': reasoning,
                    'game_state': game_state.copy()
                })
                
                return action, reasoning
            else:
                return self._fallback_decision(game_state), "LLM request failed"
                
        except Exception as e:
            return self._fallback_decision(game_state), f"LLM error: {str(e)}"
    
    def _build_prompt(self, game_state: Dict, screen_analysis: Dict, recent_actions: List[str]) -> str:
        """Build context-aware prompt for LLM with game intelligence"""
        
        # Check if failsafe intervention is active
        failsafe_context = getattr(self, 'failsafe_context', {})
        is_failsafe_active = failsafe_context.get('stuck_detected', False)
        
        # Use game intelligence to analyze context
        game_context = self.game_intelligence.analyze_game_context(game_state, screen_analysis)
        action_plans = self.game_intelligence.get_action_plan(game_context, game_state)
        contextual_advice = self.game_intelligence.get_contextual_advice(game_context, recent_actions)
        
        # Check for learned experiences
        situation_hash = self.experience_memory.get_situation_hash(game_state, screen_analysis, {
            'phase': game_context.phase.name,
            'location_type': game_context.location_type.name
        })
        learned_actions = self.experience_memory.get_recommended_actions(situation_hash, {
            'phase': game_context.phase.name,
            'location_type': game_context.location_type.name
        })
        
        # Game state summary
        party_count = game_state.get('party_count', 0)
        if party_count > 0:
            player_info = f"Player: Level {game_state.get('player_level', '?')}, HP {game_state.get('player_hp', 0)}/{game_state.get('player_max_hp', 1)}"
        else:
            player_info = f"Player: NO POKEMON YET (HP display shows 0/0 but this is normal)"
        
        # Enhanced location context with coordinates
        current_map = game_state.get('player_map', 0)
        current_x = game_state.get('player_x', 0)
        current_y = game_state.get('player_y', 0)
        location_info = f"Location: {game_context.location_name} (Map {current_map}, Position {current_x},{current_y})"
        
        badges_info = f"Badges: {game_state.get('badges_total', 0)}/16"
        money_info = f"Money: ¥{game_state.get('money', 0)}"
        party_info = f"Party: {party_count} Pokemon" + (" - YOU NEED TO GET YOUR FIRST POKEMON!" if party_count == 0 else "")
        
        # Screen analysis
        screen_state = screen_analysis.get('state', 'unknown')
        screen_variance = screen_analysis.get('variance', 0)
        
        # Recent actions context
        recent = " → ".join(recent_actions[-5:]) if recent_actions else "None"
        
        # Game phase and progress information
        phase_info = f"Game Phase: {game_context.phase.name}"
        
        # Health and urgency context
        health_info = f"Health Status: {game_context.health_status} (Urgency: {game_context.urgency_level}/5)"
        
        # Battle context
        battle_context = ""
        if game_state.get('in_battle', 0) == 1:
            enemy_level = game_state.get('enemy_level', 0)
            enemy_species = game_state.get('enemy_species', 0)
            battle_context = f"\n🔥 IN BATTLE: Enemy Level {enemy_level} (Species {enemy_species})"
        
        # Build failsafe-specific guidance
        failsafe_guidance = ""
        if is_failsafe_active:
            stuck_location = failsafe_context.get('stuck_location', (0, 0, 0))
            actions_without_reward = failsafe_context.get('actions_without_reward', 0)
            
            failsafe_guidance = f"""\n🚨 FAILSAFE INTERVENTION ACTIVE! 🚨
You are STUCK at Map {stuck_location[0]}, Position ({stuck_location[1]},{stuck_location[2]})
Actions without progress: {actions_without_reward}

SPECIFIC MOVEMENT INSTRUCTIONS:
- If you're in the bedroom (Map 24) at positions (0,0), (1,0), or (2,0):
  * The door is DOWN and to the RIGHT from the starting position
  * Try: DOWN → DOWN → RIGHT → DOWN to exit the room
  * Avoid pressing 'A' near the radio (top-left area)
- If stuck repeating same movements: try the OPPOSITE direction
- If coordinates aren't changing: you might be hitting walls - try a different direction
- Priority: GET OUT OF THIS ROOM by moving to new map coordinates

CONCRETE ACTION PLAN:
1. If at (0,0) or (1,0): Move DOWN or RIGHT
2. If at (2,0): Move DOWN then RIGHT
3. Look for doorways and transitions to new areas
4. Press 'A' only when you see NPCs or objects to interact with

FAILSAFE OVERRIDE: Ignore vague goals. Focus ONLY on changing your coordinates!"""
        
        # Build recommended actions list
        recommended_actions_text = "\n".join([f"- {action}" for action in game_context.recommended_actions])
        
        # Format immediate goals - make them more specific if failsafe is active
        if is_failsafe_active and party_count == 0:
            immediate_goals_text = "\n".join([
                "- MOVE to different coordinates (change position numbers)",
                "- EXIT the current room/area",
                "- Find NPCs or doorways by exploring systematically",
                "- Get to a NEW map ID (currently Map 24 = bedroom)"
            ])
        else:
            immediate_goals_text = "\n".join([f"- {goal}" for goal in game_context.immediate_goals])
        
        # Format action plans if available
        action_plan_text = ""
        if action_plans and not is_failsafe_active:  # Skip complex plans during failsafe
            top_plan = action_plans[0]  # Get highest priority plan
            action_plan_text = f"\n\nCURRENT PLAN: {top_plan.goal}\nSteps:\n"
            action_plan_text += "\n".join([f"{i+1}. {step}" for i, step in enumerate(top_plan.steps)])
        
        # Add learned experience if available
        experience_text = ""
        if learned_actions:
            experience_text = f"\n\nLEARNED EXPERIENCE: In similar situations, these actions worked well: {' → '.join(learned_actions)}"
            memory_stats = self.experience_memory.get_memory_stats()
            experience_text += f"\n(Based on {memory_stats['total_experiences']} past experiences)"
        
        # Modify guidelines based on failsafe state
        movement_guidelines = ""
        if is_failsafe_active:
            movement_guidelines = """\nFAILSAFE MOVEMENT STRATEGY:
🎯 PRIMARY GOAL: Change your position coordinates!
- Current coordinates: ({current_x},{current_y}) on Map {current_map}
- Target: ANY different coordinates or map
- Method: Try each direction (up/down/left/right) systematically
- If hitting walls: coordinates won't change, try different direction
- If coordinates change: GOOD! Continue in that direction
- Look for map transitions (screen changes, loading)""".format(current_x=current_x, current_y=current_y, current_map=current_map)
        else:
            movement_guidelines = """\nSTRATEGY PRIORITIES:
1. If stuck in settings_menu: Press 'b' immediately
2. If in battle: Use 'a' to attack
3. If in dialogue: Use 'a' to progress
4. If in unwanted menu: Use 'b' to exit
5. If in overworld: Follow IMMEDIATE GOALS and RECOMMENDED ACTIONS"""
        
        prompt = f"""You are an AI playing Pokemon Crystal. Make the best action choice based on the current situation.

CURRENT STATUS:
{player_info}
{location_info}
{badges_info}
{money_info}
{party_info}
{phase_info}
{health_info}
Screen State: {screen_state} (variance: {screen_variance:.1f})
Recent Actions: {recent}
{battle_context}
{failsafe_guidance}

GAME CONTEXT:
{contextual_advice}

IMMEDIATE GOALS:
{immediate_goals_text}

RECOMMENDED ACTIONS:
{recommended_actions_text}
{action_plan_text}
{experience_text}

AVAILABLE ACTIONS:
up, down, left, right - Movement
a - Interact/Confirm/Attack
b - Cancel/Back/Exit Menu
start - Open Menu (FORBIDDEN until you have Pokemon!)
select - Select button (FORBIDDEN until you have Pokemon!)

CRITICAL SCREEN STATE RULES:
🔥 BATTLE: Use 'a' to attack
💬 DIALOGUE: Use 'a' to progress text
⚙️ SETTINGS_MENU: Use 'b' to exit (you're stuck in settings!)
📋 MENU: Use 'b' to exit unless you have a specific goal
🌍 OVERWORLD: Explore with movement + 'a' to interact
⏳ LOADING: Wait (any action is fine)

IMPORTANT: NO POKEMON = NO HEALING NEEDED!
- If you have 0 Pokemon, HP shows 0/0 but this is NORMAL
- Do NOT try to heal when you have no Pokemon
- Focus on getting your first Pokemon instead

IMPORTANT GUIDELINES:
- If screen_state is 'settings_menu': ALWAYS use 'b' to escape
- If screen_state is 'menu' and you didn't intend to open it: use 'b'
- If screen_state is 'dialogue': ALWAYS use 'a' to progress
- If recent actions show 'START' but you're in 'settings_menu': use 'b' to exit
- Only use 'start' when you specifically need to access menu for healing/items
- 'b' is your escape key - use it liberally to exit unwanted screens
{movement_guidelines}

Choose ONE action and briefly explain why. Focus on CONCRETE movement if coordinates aren't changing!
Format: ACTION: [action]
Reasoning: [brief explanation]

Your choice:"""
        
        # Clear failsafe context after use
        if hasattr(self, 'failsafe_context'):
            self.failsafe_context = {}
            
        return prompt

        return prompt
    
    def _parse_llm_response(self, response: str) -> str:
        """Parse LLM response to extract action"""
        response = response.lower().strip()
        
        # Look for ACTION: pattern
        if "action:" in response:
            action_part = response.split("action:")[1].split("\n")[0].strip()
            # Extract first word that looks like an action
            words = action_part.split()
            for word in words:
                clean_word = word.strip('.,!?()[]{}').lower()
                if clean_word in ['up', 'down', 'left', 'right', 'a', 'b', 'start', 'select']:
                    return clean_word
        
        # Look for common action words in the response
        valid_actions = ['up', 'down', 'left', 'right', 'a', 'b', 'start', 'select']
        for action in valid_actions:
            if action in response:
                return action
        
        # Fallback to 'a' if nothing found
        return 'a'
    
    def _fallback_decision(self, game_state: Dict) -> str:
        """Fallback decision when LLM is unavailable"""
        # Smart fallback based on game state
        if game_state.get('in_battle', 0) == 1:
            return 'a'  # Attack in battle
        elif game_state.get('menu_state', 0) != 0:
            return 'a'  # Navigate menus
        else:
            # Exploration pattern
            actions = ['up', 'right', 'down', 'left', 'a']
            return actions[int(time.time()) % len(actions)]

class PokemonRewardCalculator:
    """Sophisticated reward calculation for Pokemon Crystal"""
    
    def __init__(self):
        self.previous_state = {}
        self.exploration_bonus = {}
        self.last_reward_time = time.time()
        # Track visited locations to prevent reward farming
        self.visited_locations = set()  # Will store (map_id, x, y) tuples
        
        # Track repeated blocked movements for escalating penalties
        self.blocked_movement_tracker = {}  # (map, x, y, direction) -> consecutive_count
        self.max_blocked_penalty = -0.1  # Maximum penalty for being very stuck
        
    def calculate_reward(self, current_state: Dict, previous_state: Dict) -> Tuple[float, Dict[str, float]]:
        """Calculate comprehensive reward based on game progress"""
        rewards = {}
        total_reward = 0.0
        
        # 1. Health and survival rewards
        hp_reward = self._calculate_hp_reward(current_state, previous_state)
        rewards['health'] = hp_reward
        total_reward += hp_reward
        
        # 2. Level progression rewards
        level_reward = self._calculate_level_reward(current_state, previous_state)
        rewards['level'] = level_reward
        total_reward += level_reward
        
        # 3. Badge progression rewards (major milestone)
        badge_reward = self._calculate_badge_reward(current_state, previous_state)
        rewards['badges'] = badge_reward
        total_reward += badge_reward
        
        # 4. Money and item rewards
        money_reward = self._calculate_money_reward(current_state, previous_state)
        rewards['money'] = money_reward
        total_reward += money_reward
        
        # 5. Coordinate movement rewards (small rewards for any position change)
        movement_reward = self._calculate_movement_reward(current_state, previous_state)
        rewards['movement'] = movement_reward
        total_reward += movement_reward
        
        # 6. Exploration rewards (larger rewards for completely new areas)
        exploration_reward = self._calculate_exploration_reward(current_state, previous_state)
        rewards['exploration'] = exploration_reward
        total_reward += exploration_reward
        
        # 7. Battle performance rewards
        battle_reward = self._calculate_battle_reward(current_state, previous_state)
        rewards['battle'] = battle_reward
        total_reward += battle_reward
        
        # 8. Progress and efficiency penalties
        efficiency_penalty = self._calculate_efficiency_penalty(current_state)
        rewards['efficiency'] = efficiency_penalty
        total_reward += efficiency_penalty
        
        # 9. Early game progression rewards (getting first Pokemon, etc.)
        progression_reward = self._calculate_progression_reward(current_state, previous_state)
        rewards['progression'] = progression_reward
        total_reward += progression_reward
        
        # 10. Dialogue and interaction rewards (to guide toward first Pokemon)
        dialogue_reward = self._calculate_dialogue_reward(current_state, previous_state)
        rewards['dialogue'] = dialogue_reward
        total_reward += dialogue_reward
        
        # 11. Blocked movement penalty (escalating for repeated attempts)
        blocked_penalty = self._calculate_blocked_movement_penalty(current_state, previous_state)
        rewards['blocked_movement'] = blocked_penalty
        total_reward += blocked_penalty
        
        # 12. Time-based small negative reward to encourage efficiency
        time_penalty = -0.01  # Small penalty each step
        rewards['time'] = time_penalty
        total_reward += time_penalty
        
        return total_reward, rewards
    
    def _calculate_hp_reward(self, current: Dict, previous: Dict) -> float:
        """Reward for maintaining/improving health"""
        # Only calculate health rewards if player has Pokemon
        party_count = current.get('party_count', 0)
        if party_count == 0:
            return 0.0  # No Pokemon = no health rewards/penalties
            
        curr_hp = current.get('player_hp', 0)
        curr_max_hp = current.get('player_max_hp', 1)
        prev_hp = previous.get('player_hp', curr_hp)
        prev_max_hp = previous.get('player_max_hp', curr_max_hp)
        
        # Skip if no valid HP data
        if curr_max_hp == 0 or prev_max_hp == 0:
            return 0.0
        
        curr_hp_pct = curr_hp / curr_max_hp
        prev_hp_pct = prev_hp / prev_max_hp
        
        # Reward health improvement, penalize health loss
        hp_change = curr_hp_pct - prev_hp_pct
        
        if hp_change > 0:
            return hp_change * 5.0  # Reward healing
        elif hp_change < 0:
            return hp_change * 10.0  # Penalty for taking damage
        
        # Small bonus for staying healthy
        if curr_hp_pct > 0.8:
            return 0.1
        elif curr_hp_pct < 0.2:
            return -0.5  # Penalty for being low on health
            
        return 0.0
    
    def _calculate_level_reward(self, current: Dict, previous: Dict) -> float:
        """Reward for leveling up Pokemon with anti-glitch guards"""
        curr_level = current.get('player_level', 0)
        prev_level = previous.get('player_level', curr_level)
        
        # Get screen state and party count for validation
        curr_screen_state = getattr(self, 'last_screen_state', 'unknown')
        prev_screen_state = getattr(self, 'prev_screen_state', curr_screen_state)
        curr_party_count = current.get('party_count', 0)
        prev_party_count = previous.get('party_count', 0)
        
        # CRITICAL FIX: Only award level rewards if we have Pokemon
        # Level changes without Pokemon are memory glitches
        if curr_party_count == 0 or prev_party_count == 0:
            return 0.0  # No Pokemon = no level rewards possible
        
        # CRITICAL FIX: Only award level rewards in overworld state
        # This prevents menu operations from triggering false level changes
        if curr_screen_state != 'overworld' or prev_screen_state != 'overworld':
            return 0.0
        
        # Guard against impossible level spikes (>100 or huge jumps)
        if curr_level > 100 or prev_level > 100:
            return 0.0
        
        # Additional validation: levels must be reasonable (1-100)
        if not (1 <= curr_level <= 100 and 1 <= prev_level <= 100):
            return 0.0
        
        if curr_level > prev_level:
            level_gain = curr_level - prev_level
            # Cap level gain to prevent huge memory spike rewards
            level_gain = min(level_gain, 5)  # Max 5 levels per step
            
            # Additional validation: require HP values to be reasonable for this level
            curr_hp = current.get('player_hp', 0)
            curr_max_hp = current.get('player_max_hp', 0)
            if curr_max_hp < 10 or curr_hp > curr_max_hp:
                return 0.0  # Suspicious HP values, likely memory glitch
            
            return level_gain * 50.0  # Big reward for leveling up
            
        return 0.0
    
    def _calculate_badge_reward(self, current: Dict, previous: Dict) -> float:
        """Huge reward for earning badges (major milestones), with much stricter anti-glitch guards."""
        # Get screen state to prevent menu-state false rewards
        curr_screen_state = getattr(self, 'last_screen_state', 'unknown')
        prev_screen_state = getattr(self, 'prev_screen_state', curr_screen_state)
        
        # CRITICAL FIX: Only award badge rewards in consistent overworld states
        # This prevents menu operations from triggering false badge changes
        if curr_screen_state != 'overworld' or prev_screen_state != 'overworld':
            return 0.0
        
        curr_badges = current.get('badges_total', 0)
        prev_badges = previous.get('badges_total', curr_badges)
        
        # Get badge raw values (bitmasks)
        curr_raw = (current.get('badges', 0), current.get('kanto_badges', 0))
        prev_raw = (previous.get('badges', curr_raw[0]), previous.get('kanto_badges', curr_raw[1]))
        
        # Additional validation: avoid early game memory spikes
        early_game = current.get('party_count', 0) == 0 and current.get('player_level', 0) == 0
        if early_game and (0xFF in curr_raw or 0xFF in prev_raw):
            return 0.0

        # Additional validation: badges shouldn't change without actual progression
        # Must have at least one Pokemon to earn badges
        if current.get('party_count', 0) == 0:
            return 0.0
            
        # Only reward if the total is within plausible range AND actually increased
        if 0 <= curr_badges <= 16 and 0 <= prev_badges <= 16 and curr_badges > prev_badges:
            # Create milestone key to prevent repeat rewards for the same badge
            # This is similar to the progression milestone system
            milestone_key = f"badge_{curr_badges}_{curr_raw[0]}_{curr_raw[1]}"
            
            if not hasattr(self, 'badge_milestones'):
                self.badge_milestones = set()
                
            # Only reward each badge milestone once
            if milestone_key not in self.badge_milestones:
                self.badge_milestones.add(milestone_key)
                
                # Cap to 1 badge per step to prevent jumps awarding huge rewards
                badge_gain = min(curr_badges - prev_badges, 1)
                
                # Debug logging to track badge rewards
                print(f"🎖️ BADGE REWARD: {badge_gain * 500.0:.2f} | Raw: {curr_raw} | Total: {curr_badges}")
                
                return badge_gain * 500.0  # Huge reward for badge progress!
            else:
                # Already rewarded this badge milestone
                return 0.0

        return 0.0
    
    def _calculate_money_reward(self, current: Dict, previous: Dict) -> float:
        """Reward for earning money - with ULTRA strict validation to prevent SELECT button spam"""
        # Get screen state to prevent menu-state false rewards
        curr_screen_state = getattr(self, 'last_screen_state', 'unknown')
        prev_screen_state = getattr(self, 'prev_screen_state', curr_screen_state)
        
        # Get the last action to prevent SELECT button spam
        last_action = getattr(self, 'last_action', 'unknown')
        
        # ULTRA FIX: NEVER give money rewards - they are too unreliable
        # Money changes are often caused by memory glitches, BCD parsing issues,
        # and SELECT button spam. Disable all money rewards completely.
        return 0.0
        
        # The rest of this code is commented out as money rewards are disabled
        # 
        # # CRITICAL FIX: Only award money rewards in overworld or battle states
        # # This prevents SELECT button in menus from causing BCD parsing fluctuations
        # if curr_screen_state not in ['overworld', 'battle']:
        #     return 0.0
        # if prev_screen_state not in ['overworld', 'battle']:
        #     return 0.0
        # 
        # # ADDITIONAL FIX: Never reward money changes caused by SELECT button
        # # SELECT button can cause memory read issues leading to false money changes
        # if last_action and last_action.lower() == 'select':
        #     return 0.0
        # 
        # curr_money = current.get('money', 0)
        # prev_money = previous.get('money', curr_money)
        # 
        # # Additional validation: money values must be reasonable (0 to 999999)
        # if not (0 <= curr_money <= 999999 and 0 <= prev_money <= 999999):
        #     return 0.0
        # 
        # money_change = curr_money - prev_money
        # 
        # # Additional validation: money changes should be reasonable
        # # No single action should give more than 50 (very conservative)
        # if abs(money_change) > 50:
        #     return 0.0  # Suspicious money change, likely memory glitch
        # 
        # if money_change > 0:
        #     # Only reward genuine money gains (winning battles, finding items)
        #     # Must be in overworld with reasonable change and not SELECT action
        #     return min(money_change * 0.01, 0.5)  # Very small cap on money rewards
        # elif money_change < 0:
        #     # Small penalty for spending money (buying items)
        #     return max(money_change * 0.005, -0.2)  # Very small penalty for spending
        #     
        # return 0.0
    
    def _calculate_exploration_reward(self, current: Dict, previous: Dict) -> float:
        """Reward for exploring new areas - ONLY in overworld state"""
        # CRITICAL FIX: Only reward exploration when actually in overworld
        # This prevents menu state coordinate fluctuations from giving false rewards
        curr_screen_state = getattr(self, 'last_screen_state', 'unknown')
        prev_screen_state = getattr(self, 'prev_screen_state', curr_screen_state)
        
        # Only consider exploration rewards in overworld state
        if curr_screen_state != 'overworld':
            return 0.0
        
        # Also require previous state to be overworld to prevent menu->overworld transitions
        # from giving false rewards due to coordinate resets
        if prev_screen_state != 'overworld':
            return 0.0
        
        curr_map = current.get('player_map', 0)
        curr_x = current.get('player_x', 0)
        curr_y = current.get('player_y', 0)
        
        prev_map = previous.get('player_map', curr_map)
        prev_x = previous.get('player_x', curr_x)
        prev_y = previous.get('player_y', curr_y)
        
        # Additional validation: coordinates must be reasonable (0-255 range)
        if not (0 <= curr_x <= 255 and 0 <= curr_y <= 255 and 0 <= curr_map <= 255):
            return 0.0
        if not (0 <= prev_x <= 255 and 0 <= prev_y <= 255 and 0 <= prev_map <= 255):
            return 0.0
        
        # Current location tuple
        current_location = (curr_map, curr_x, curr_y)
        previous_location = (prev_map, prev_x, prev_y)
        
        # Skip if coordinates are exactly the same (no movement)
        if current_location == previous_location:
            return 0.0
        
        # New map reward - but only if it's a reasonable map change (adjacent maps)
        if curr_map != prev_map:
            map_diff = abs(curr_map - prev_map)
            # Only reward reasonable map transitions (not huge jumps that indicate glitches)
            if map_diff <= 10:  # Adjacent or nearby maps
                self.visited_locations.add(current_location)
                return 10.0  # Reward for entering new area
            else:
                # Suspicious map jump - likely a glitch, no reward
                return 0.0
        
        # Check if this location has been visited before
        if current_location not in self.visited_locations:
            # Validate this is actual movement (not coordinate glitch)
            coord_diff = abs(curr_x - prev_x) + abs(curr_y - prev_y)
            if 1 <= coord_diff <= 5:  # Reasonable movement distance
                # New unvisited location! Add to visited set and give reward
                self.visited_locations.add(current_location)
                return 0.1  # Small reward for discovering new tile
        
        # No reward for revisiting locations or suspicious movements
        return 0.0
    
    def _calculate_movement_reward(self, current: Dict, previous: Dict) -> float:
        """Small reward for any coordinate movement - helps break stuck patterns"""
        # CRITICAL: Only reward movement when actually in overworld
        curr_screen_state = getattr(self, 'last_screen_state', 'unknown')
        prev_screen_state = getattr(self, 'prev_screen_state', curr_screen_state)
        
        # Only consider movement rewards in overworld state
        if curr_screen_state != 'overworld':
            return 0.0
        
        # Also require previous state to be overworld to prevent false rewards
        if prev_screen_state != 'overworld':
            return 0.0
        
        curr_map = current.get('player_map', 0)
        curr_x = current.get('player_x', 0)
        curr_y = current.get('player_y', 0)
        
        prev_map = previous.get('player_map', curr_map)
        prev_x = previous.get('player_x', curr_x)
        prev_y = previous.get('player_y', curr_y)
        
        # Validate coordinates are reasonable
        if not (0 <= curr_x <= 255 and 0 <= curr_y <= 255 and 0 <= curr_map <= 255):
            return 0.0
        if not (0 <= prev_x <= 255 and 0 <= prev_y <= 255 and 0 <= prev_map <= 255):
            return 0.0
        
        # Check if coordinates changed
        position_changed = (curr_x != prev_x) or (curr_y != prev_y)
        map_changed = (curr_map != prev_map)
        
        # Reward any position change
        if position_changed or map_changed:
            # Validate this is reasonable movement (not huge coordinate jumps)
            if map_changed:
                map_diff = abs(curr_map - prev_map)
                if map_diff <= 10:  # Reasonable map transition
                    return 0.05  # Small reward for changing maps
                else:
                    return 0.0  # Skip suspicious map jumps
            else:
                # Same map, different coordinates
                coord_diff = abs(curr_x - prev_x) + abs(curr_y - prev_y)
                if 1 <= coord_diff <= 3:  # Reasonable single-step movement
                    return 0.02  # Small reward for moving within same map
                else:
                    return 0.0  # Skip suspicious coordinate jumps
        
        return 0.0  # No movement, no reward
    
    def _calculate_battle_reward(self, current: Dict, previous: Dict) -> float:
        """Reward for battle performance"""
        curr_in_battle = current.get('in_battle', 0)
        prev_in_battle = previous.get('in_battle', 0)
        
        # Entered battle
        if curr_in_battle == 1 and prev_in_battle == 0:
            return 2.0  # Small reward for engaging in battle
            
        # Exited battle (assuming victory if health didn't drop significantly)
        if curr_in_battle == 0 and prev_in_battle == 1:
            curr_hp_pct = current.get('player_hp', 0) / max(current.get('player_max_hp', 1), 1)
            if curr_hp_pct > 0.5:  # Likely won the battle
                return 20.0  # Good reward for winning battle
            else:
                return -5.0  # Penalty for losing/fleeing
                
        return 0.0
    
    def _calculate_efficiency_penalty(self, current: Dict) -> float:
        """Penalty for inefficient actions"""
        # Penalty for being stuck (no progress)
        # This could be enhanced with more sophisticated stuck detection
        return 0.0  # Placeholder for now
    
    def _calculate_progression_reward(self, current: Dict, previous: Dict) -> float:
        """Reward for early game progression milestones - with strict validation"""
        curr_party_count = current.get('party_count', 0)
        prev_party_count = previous.get('party_count', 0)
        
        # Get screen state to prevent menu-state false rewards
        curr_screen_state = getattr(self, 'last_screen_state', 'unknown')
        prev_screen_state = getattr(self, 'prev_screen_state', curr_screen_state)
        
        # CRITICAL FIX: Only award progression rewards in consistent overworld states
        # This prevents menu operations from triggering false party_count changes
        if curr_screen_state != 'overworld' or prev_screen_state != 'overworld':
            return 0.0
        
        # Additional validation: party count must be reasonable and stable
        if not (0 <= curr_party_count <= 6 and 0 <= prev_party_count <= 6):
            return 0.0
        
        # Require multiple corroborating signals for genuine Pokemon acquisition
        if curr_party_count > prev_party_count:
            # Check if we have corroborating evidence of genuine progression
            curr_level = current.get('player_level', 0)
            prev_level = previous.get('player_level', 0)
            curr_hp = current.get('player_hp', 0)
            curr_max_hp = current.get('player_max_hp', 0)
            
            # First Pokemon: require level > 0 and reasonable HP values
            if prev_party_count == 0 and curr_party_count == 1:
                if curr_level > 0 and curr_max_hp > 0 and curr_hp <= curr_max_hp:
                    # Track this as a major milestone to prevent repeated rewards
                    milestone_key = f"first_pokemon_{curr_level}_{curr_max_hp}"
                    if not hasattr(self, 'progression_milestones'):
                        self.progression_milestones = set()
                    
                    if milestone_key not in self.progression_milestones:
                        self.progression_milestones.add(milestone_key)
                        return 100.0  # First Pokemon is a huge milestone!
                    else:
                        return 0.0  # Already rewarded this milestone
                else:
                    return 0.0  # Invalid Pokemon data, likely memory glitch
            
            # Additional Pokemon: require reasonable level progression
            elif prev_party_count > 0 and curr_party_count <= 6:
                # Require stable level values (not memory glitches)
                if 1 <= curr_level <= 100 and abs(curr_level - prev_level) <= 5:
                    party_milestone_key = f"party_{curr_party_count}_{curr_level}"
                    if not hasattr(self, 'progression_milestones'):
                        self.progression_milestones = set()
                    
                    if party_milestone_key not in self.progression_milestones:
                        self.progression_milestones.add(party_milestone_key)
                        return 25.0  # Additional Pokemon rewarded
                    else:
                        return 0.0  # Already rewarded this milestone
        
        return 0.0
    
    def _calculate_dialogue_reward(self, current: Dict, previous: Dict) -> float:
        """Reward for dialogue progression to guide toward first Pokemon"""
        # Get screen state to determine if we're in dialogue
        curr_screen_state = getattr(self, 'last_screen_state', 'unknown')
        prev_screen_state = getattr(self, 'prev_screen_state', curr_screen_state)
        
        # Only reward dialogue progression before getting first Pokemon
        party_count = current.get('party_count', 0)
        if party_count > 0:
            return 0.0  # No dialogue rewards after getting first Pokemon
        
        # Small reward for being in dialogue state (encourages talking to NPCs)
        if curr_screen_state == 'dialogue':
            return 0.05  # Small positive reward for dialogue engagement
        
        # Small reward for transitioning into dialogue (finding NPCs to talk to)
        if prev_screen_state == 'overworld' and curr_screen_state == 'dialogue':
            return 0.1  # Reward for initiating dialogue
        
        # Small reward for progressing through dialogue sequences
        if prev_screen_state == 'dialogue' and curr_screen_state == 'dialogue':
            return 0.02  # Small reward for progressing dialogue
        
        return 0.0
    
    def _calculate_blocked_movement_penalty(self, current: Dict, previous: Dict) -> float:
        """Escalating penalty for repeatedly trying blocked movements at same location"""
        # Only apply this penalty in overworld state
        curr_screen_state = getattr(self, 'last_screen_state', 'unknown')
        if curr_screen_state != 'overworld':
            return 0.0
        
        # Get current and previous positions
        curr_map = current.get('player_map', 0)
        curr_x = current.get('player_x', 0)
        curr_y = current.get('player_y', 0)
        
        prev_map = previous.get('player_map', curr_map)
        prev_x = previous.get('player_x', curr_x)
        prev_y = previous.get('player_y', curr_y)
        
        # Get the last action attempted (stored by the trainer)
        last_action = getattr(self, 'last_action', 'unknown')
        
        # Only track directional movements
        if last_action not in ['up', 'down', 'left', 'right']:
            return 0.0
        
        # Check if position didn't change (blocked movement)
        position_unchanged = (
            curr_map == prev_map and 
            curr_x == prev_x and 
            curr_y == prev_y
        )
        
        if position_unchanged:
            # Create tracking key: (map, x, y, direction)
            blocked_key = (curr_map, curr_x, curr_y, last_action)
            
            # Increment consecutive blocked attempts
            if blocked_key not in self.blocked_movement_tracker:
                self.blocked_movement_tracker[blocked_key] = 0
            self.blocked_movement_tracker[blocked_key] += 1
            
            consecutive_blocks = self.blocked_movement_tracker[blocked_key]
            
            # Calculate small escalating penalty: -0.005, -0.01, -0.015, -0.02, ..., capped at max
            base_penalty = -0.005
            escalation_factor = consecutive_blocks  # Linear escalation
            penalty = base_penalty * escalation_factor
            penalty = max(penalty, self.max_blocked_penalty)  # Cap at maximum penalty
            
            # Clean up old blocked movement tracking (prevent memory bloat)
            if len(self.blocked_movement_tracker) > 100:
                # Remove entries with low consecutive counts
                self.blocked_movement_tracker = {
                    k: v for k, v in self.blocked_movement_tracker.items() 
                    if v >= 2
                }
            
            return penalty
        else:
            # Position changed - clear any blocked movement tracking for this location
            # This rewards successfully getting unstuck
            keys_to_clear = [
                key for key in self.blocked_movement_tracker.keys()
                if key[0] == prev_map and key[1] == prev_x and key[2] == prev_y
            ]
            for key in keys_to_clear:
                del self.blocked_movement_tracker[key]
            
            return 0.0
    
    def get_reward_summary(self, rewards: Dict[str, float]) -> str:
        """Get human-readable reward summary"""
        summary_parts = []
        for category, value in rewards.items():
            if abs(value) > 0.01:  # Only show significant rewards
                summary_parts.append(f"{category}: {value:+.2f}")
        
        return " | ".join(summary_parts) if summary_parts else "no rewards"

class WebMonitor(BaseHTTPRequestHandler):
    """Enhanced web server with LLM decision tracking"""
    
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            
            html = """
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Pokemon Crystal LLM RL Training Dashboard</title>
                <style>
                    * { margin: 0; padding: 0; box-sizing: border-box; }
                    
                    body {
                        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                        background: linear-gradient(135deg, #0c0c0c 0%, #1a1a2e 100%);
                        color: #fff;
                        min-height: 100vh;
                        overflow-x: hidden;
                    }
                    
                    .header {
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        padding: 20px;
                        margin-bottom: 20px;
                        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
                        position: sticky;
                        top: 0;
                        z-index: 100;
                    }
                    
                    .header h1 {
                        font-size: 28px;
                        font-weight: 700;
                        margin-bottom: 8px;
                        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
                    }
                    
                    .header p {
                        opacity: 0.9;
                        font-size: 16px;
                    }
                    
                    .status-indicator {
                        display: inline-block;
                        width: 12px;
                        height: 12px;
                        border-radius: 50%;
                        background: #00ff88;
                        margin-left: 10px;
                        animation: pulse 2s infinite;
                    }
                    
                    @keyframes pulse {
                        0% { opacity: 1; }
                        50% { opacity: 0.5; }
                        100% { opacity: 1; }
                    }
                    
                    .container {
                        max-width: 1400px;
                        margin: 0 auto;
                        padding: 0 20px;
                        display: grid;
                        grid-template-columns: 2fr 1fr;
                        gap: 20px;
                    }
                    
                    .main-panel {
                        display: flex;
                        flex-direction: column;
                        gap: 20px;
                    }
                    
                    .side-panel {
                        display: flex;
                        flex-direction: column;
                        gap: 20px;
                    }
                    
                    .stats-grid {
                        display: grid;
                        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
                        gap: 15px;
                        margin-bottom: 20px;
                    }
                    
                    .stat-card {
                        background: linear-gradient(135deg, #1e1e2f 0%, #2a2a3a 100%);
                        border-radius: 12px;
                        padding: 20px;
                        border: 1px solid #333;
                        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
                        transition: transform 0.2s ease, box-shadow 0.2s ease;
                        position: relative;
                        overflow: hidden;
                    }
                    
                    .stat-card:hover {
                        transform: translateY(-2px);
                        box-shadow: 0 6px 25px rgba(0,0,0,0.3);
                    }
                    
                    .stat-card::before {
                        content: '';
                        position: absolute;
                        top: 0;
                        left: 0;
                        width: 100%;
                        height: 3px;
                        background: linear-gradient(90deg, #00ff88, #00d4ff);
                    }
                    
                    .stat-value {
                        font-size: 32px;
                        font-weight: 800;
                        color: #00ff88;
                        margin-bottom: 5px;
                        text-shadow: 0 0 10px rgba(0, 255, 136, 0.3);
                    }
                    
                    .stat-label {
                        font-size: 14px;
                        color: #aaa;
                        text-transform: uppercase;
                        letter-spacing: 0.5px;
                        font-weight: 500;
                    }
                    
                    .stat-change {
                        font-size: 12px;
                        margin-top: 5px;
                        display: flex;
                        align-items: center;
                        gap: 4px;
                    }
                    
                    .stat-change.positive { color: #00ff88; }
                    .stat-change.negative { color: #ff4757; }
                    
                    .game-screen-container {
                        background: linear-gradient(135deg, #1e1e2f 0%, #2a2a3a 100%);
                        border-radius: 12px;
                        padding: 20px;
                        border: 1px solid #333;
                        text-align: center;
                        position: relative;
                        overflow: hidden;
                    }
                    
                    .game-screen-container h3 {
                        margin-bottom: 15px;
                        font-size: 20px;
                        color: #fff;
                    }
                    
                    .game-screen {
                        border-radius: 8px;
                        border: 2px solid #00ff88;
                        box-shadow: 0 0 20px rgba(0, 255, 136, 0.2);
                        max-width: 100%;
                        height: auto;
                        image-rendering: pixelated;
                        image-rendering: crisp-edges;
                    }
                    
                    .screen-info {
                        display: flex;
                        justify-content: space-between;
                        margin-top: 15px;
                        font-size: 12px;
                        color: #aaa;
                    }
                    
                    .panel {
                        background: linear-gradient(135deg, #1e1e2f 0%, #2a2a3a 100%);
                        border-radius: 12px;
                        padding: 20px;
                        border: 1px solid #333;
                        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
                    }
                    
                    .panel h3 {
                        font-size: 18px;
                        margin-bottom: 15px;
                        color: #fff;
                        display: flex;
                        align-items: center;
                        gap: 8px;
                    }
                    
                    .reward-bars {
                        display: flex;
                        flex-direction: column;
                        gap: 10px;
                    }
                    
                    .reward-bar {
                        background: rgba(255,255,255,0.1);
                        border-radius: 6px;
                        padding: 8px 12px;
                        display: flex;
                        justify-content: between;
                        align-items: center;
                        font-size: 14px;
                    }
                    
                    .reward-bar.positive {
                        background: linear-gradient(90deg, rgba(0,255,136,0.2), rgba(0,255,136,0.1));
                        border-left: 3px solid #00ff88;
                    }
                    
                    .reward-bar.negative {
                        background: linear-gradient(90deg, rgba(255,71,87,0.2), rgba(255,71,87,0.1));
                        border-left: 3px solid #ff4757;
                    }
                    
                    .llm-decision {
                        background: rgba(255,255,255,0.05);
                        border-radius: 8px;
                        padding: 12px;
                        margin-bottom: 10px;
                        border-left: 4px solid #00ff88;
                        position: relative;
                    }
                    
                    .llm-decision::before {
                        content: '🧠';
                        position: absolute;
                        top: 12px;
                        right: 12px;
                        font-size: 16px;
                    }
                    
                    .llm-action {
                        font-weight: 700;
                        color: #00ff88;
                        font-size: 16px;
                        margin-bottom: 5px;
                    }
                    
                    .llm-reasoning {
                        color: #ccc;
                        font-size: 13px;
                        line-height: 1.4;
                    }
                    
                    .llm-timestamp {
                        color: #666;
                        font-size: 11px;
                        margin-top: 5px;
                    }
                    
                    .progress-bar {
                        background: rgba(255,255,255,0.1);
                        border-radius: 10px;
                        height: 8px;
                        overflow: hidden;
                        margin-top: 8px;
                    }
                    
                    .progress-fill {
                        height: 100%;
                        background: linear-gradient(90deg, #00ff88, #00d4ff);
                        transition: width 0.3s ease;
                        border-radius: 10px;
                    }
                    
                    .loading-spinner {
                        display: inline-block;
                        width: 16px;
                        height: 16px;
                        border: 2px solid #333;
                        border-radius: 50%;
                        border-top: 2px solid #00ff88;
                        animation: spin 1s linear infinite;
                    }
                    
                    @keyframes spin {
                        0% { transform: rotate(0deg); }
                        100% { transform: rotate(360deg); }
                    }
                    
                    .error-state {
                        color: #ff4757;
                        text-align: center;
                        padding: 20px;
                    }
                    
                    @media (max-width: 1024px) {
                        .container {
                            grid-template-columns: 1fr;
                            max-width: 800px;
                        }
                        
                        .stats-grid {
                            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
                        }
                    }
                    
                    .memory-section {
                        margin-bottom: 15px;
                    }
                    
                    .memory-section h4 {
                        font-size: 14px;
                        color: #00d4ff;
                        margin-bottom: 10px;
                        font-weight: 600;
                        text-transform: uppercase;
                        letter-spacing: 0.5px;
                    }
                    
                    .memory-grid {
                        display: flex;
                        flex-direction: column;
                        gap: 6px;
                    }
                    
                    .memory-row {
                        display: grid;
                        grid-template-columns: auto 1fr auto;
                        gap: 8px;
                        align-items: center;
                        padding: 4px 8px;
                        background: rgba(255,255,255,0.05);
                        border-radius: 4px;
                        font-size: 12px;
                        font-family: 'Courier New', monospace;
                    }
                    
                    .memory-row .addr {
                        color: #ffb347;
                        font-weight: 700;
                        min-width: 70px;
                    }
                    
                    .memory-row .label {
                        color: #ccc;
                        font-size: 11px;
                        text-transform: uppercase;
                    }
                    
                    .memory-row .value {
                        color: #00ff88;
                        font-weight: 700;
                        text-align: right;
                        min-width: 40px;
                        font-family: 'Courier New', monospace;
                    }
                    
                    .memory-row .value.hex {
                        color: #ff6b6b;
                    }
                    
                    .memory-row .value.changed {
                        animation: memoryFlash 0.5s ease;
                        background: rgba(0, 255, 136, 0.2);
                        border-radius: 3px;
                        padding: 2px 4px;
                    }
                    
                    @keyframes memoryFlash {
                        0% { background: rgba(255, 255, 255, 0.3); }
                        100% { background: rgba(0, 255, 136, 0.2); }
                    }
                    
                    @media (max-width: 768px) {
                        .stats-grid {
                            grid-template-columns: repeat(2, 1fr);
                        }
                        
                        .header h1 {
                            font-size: 24px;
                        }
                        
                        .container {
                            padding: 0 15px;
                        }
                    }
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>🤖 Pokemon Crystal LLM RL Training Dashboard<span class="status-indicator" id="status-indicator"></span></h1>
                    <p>Advanced reinforcement learning with local LLM decision making and real-time monitoring</p>
                </div>
                
                <div class="container">
                    <div class="main-panel">
                        <!-- Stats Grid -->
                        <div class="stats-grid">
                            <div class="stat-card">
                                <div class="stat-value" id="actions">-</div>
                                <div class="stat-label">Total Actions</div>
                                <div class="stat-change" id="actions-change"></div>
                            </div>
                            <div class="stat-card">
                                <div class="stat-value" id="aps">-</div>
                                <div class="stat-label">Actions/Second</div>
                                <div class="stat-change" id="aps-change"></div>
                            </div>
                            <div class="stat-card">
                                <div class="stat-value" id="llm-decisions">-</div>
                                <div class="stat-label">LLM Decisions</div>
                                <div class="stat-change" id="llm-change"></div>
                            </div>
                            <div class="stat-card">
                                <div class="stat-value" id="total-reward">-</div>
                                <div class="stat-label">Total Reward</div>
                                <div class="stat-change" id="reward-change"></div>
                            </div>
                            <div class="stat-card">
                                <div class="stat-value" id="level">-</div>
                                <div class="stat-label">Player Level</div>
                                <div class="stat-change" id="level-change"></div>
                            </div>
                            <div class="stat-card">
                                <div class="stat-value" id="badges">-</div>
                                <div class="stat-label">Badges Earned</div>
                                <div class="progress-bar"><div class="progress-fill" id="badge-progress"></div></div>
                            </div>
                        </div>
                        
                        <!-- Game Screen -->
                        <div class="game-screen-container">
                            <h3>🖼️ Live Game Screen</h3>
                            <img id="gameScreen" class="game-screen" src="/screenshot?t=0" width="480" height="432" alt="Game Screen">
                            <div class="screen-info">
                                <span>Screen State: <span id="screen-state">-</span></span>
                                <span>Refresh Rate: <span id="refresh-rate">500ms</span></span>
                                <span>Last Updated: <span id="last-update">-</span></span>
                            </div>
                        </div>
                    </div>
                    
                    <div class="side-panel">
                        <!-- Game State Info -->
                        <div class="panel">
                            <h3>🎮 Game State</h3>
                            <div class="reward-bars" id="game-state-info">
                                <div class="reward-bar">Map: <span id="player-map">-</span></div>
                                <div class="reward-bar">Position: <span id="player-position">-</span></div>
                                <div class="reward-bar">Money: ¥<span id="player-money">-</span></div>
                                <div class="reward-bar">Party: <span id="party-count">-</span> Pokemon</div>
                                <div class="reward-bar">HP: <span id="player-hp">-</span></div>
                            </div>
                        </div>
                        
                        <!-- Reward Breakdown -->
                        <div class="panel">
                            <h3>💰 Reward Analysis</h3>
                            <div class="reward-bars" id="reward-breakdown">
                                <div class="loading-spinner"></div> Loading...
                            </div>
                        </div>
                        
                        <!-- Memory Debugger -->
                        <div class="panel">
                            <h3>🔧 Live Memory Debug</h3>
                            <div class="memory-section">
                                <h4>🎯 Core Addresses</h4>
                                <div class="memory-grid">
                                    <div class="memory-row">
                                        <span class="addr">0xD163:</span>
                                        <span class="label">Party Count</span>
                                        <span class="value" id="mem-party-count">-</span>
                                    </div>
                                    <div class="memory-row">
                                        <span class="addr">0xD35D:</span>
                                        <span class="label">Map ID</span>
                                        <span class="value" id="mem-map-id">-</span>
                                    </div>
                                    <div class="memory-row">
                                        <span class="addr">0xD361:</span>
                                        <span class="label">Player X</span>
                                        <span class="value" id="mem-player-x">-</span>
                                    </div>
                                    <div class="memory-row">
                                        <span class="addr">0xD362:</span>
                                        <span class="label">Player Y</span>
                                        <span class="value" id="mem-player-y">-</span>
                                    </div>
                                    <div class="memory-row">
                                        <span class="addr">0xD347-49:</span>
                                        <span class="label">Money (3B)</span>
                                        <span class="value" id="mem-money">-</span>
                                    </div>
                                    <div class="memory-row">
                                        <span class="addr">0xD359:</span>
                                        <span class="label">Badge Flags</span>
                                        <span class="value hex" id="mem-badges">-</span>
                                    </div>
                                    <div class="memory-row">
                                        <span class="addr">0xD057:</span>
                                        <span class="label">In Battle</span>
                                        <span class="value" id="mem-battle">-</span>
                                    </div>
                                </div>
                            </div>
                            <div class="memory-section">
                                <h4>🎮 First Pokemon Stats</h4>
                                <div class="memory-grid">
                                    <div class="memory-row">
                                        <span class="addr">0xD163:</span>
                                        <span class="label">Species</span>
                                        <span class="value" id="mem-species">-</span>
                                    </div>
                                    <div class="memory-row">
                                        <span class="addr">0xD16B:</span>
                                        <span class="label">Level</span>
                                        <span class="value" id="mem-level">-</span>
                                    </div>
                                    <div class="memory-row">
                                        <span class="addr">0xD167-68:</span>
                                        <span class="label">HP (2B)</span>
                                        <span class="value" id="mem-hp">-</span>
                                    </div>
                                    <div class="memory-row">
                                        <span class="addr">0xD169-6A:</span>
                                        <span class="label">Max HP (2B)</span>
                                        <span class="value" id="mem-max-hp">-</span>
                                    </div>
                                </div>
                            </div>
                        </div>
                        
                        <!-- LLM Decisions -->
                        <div class="panel">
                            <h3>🧠 Recent LLM Decisions</h3>
                            <div id="llm-decisions-list">
                                <div class="loading-spinner"></div> Loading decisions...
                            </div>
                        </div>
                    </div>
                </div>
                
                <script>
                    let lastStats = {};
                    let updateCount = 0;
                    let startTime = Date.now();
                    
                    function formatNumber(num) {
                        if (num >= 1000000) return (num / 1000000).toFixed(1) + 'M';
                        if (num >= 1000) return (num / 1000).toFixed(1) + 'K';
                        return num.toString();
                    }
                    
                    function formatChange(current, previous) {
                        if (previous === undefined) return '';
                        const change = current - previous;
                        if (change === 0) return '';
                        const sign = change > 0 ? '+' : '';
                        const className = change > 0 ? 'positive' : 'negative';
                        return `<span class="${className}">${sign}${change.toFixed(2)}</span>`;
                    }
                    
                    function updateGameScreen() {
                        const screen = document.getElementById('gameScreen');
                        const timestamp = Date.now();
                        screen.src = `/screenshot?t=${timestamp}`;
                        document.getElementById('last-update').textContent = new Date().toLocaleTimeString();
                    }
                    
                    async function updateStats() {
                        try {
                            updateCount++;
                            const response = await fetch('/stats');
                            const stats = await response.json();
                            
                            // Update status indicator
                            document.getElementById('status-indicator').style.background = '#00ff88';
                            
                            // Update main stats with change indicators
                            document.getElementById('actions').textContent = formatNumber(stats.actions_taken || 0);
                            document.getElementById('aps').textContent = (stats.actions_per_second || 0).toFixed(1);
                            document.getElementById('llm-decisions').textContent = formatNumber(stats.llm_decision_count || 0);
                            document.getElementById('total-reward').textContent = (stats.total_reward || 0).toFixed(2);
                            document.getElementById('level').textContent = stats.player_level || 0;
                            document.getElementById('badges').textContent = `${stats.badges_total || 0}/16`;
                            
                            // Update change indicators
                            document.getElementById('actions-change').innerHTML = formatChange(stats.actions_taken, lastStats.actions_taken);
                            document.getElementById('aps-change').innerHTML = formatChange(stats.actions_per_second, lastStats.actions_per_second);
                            document.getElementById('llm-change').innerHTML = formatChange(stats.llm_decision_count, lastStats.llm_decision_count);
                            document.getElementById('reward-change').innerHTML = formatChange(stats.total_reward, lastStats.total_reward);
                            document.getElementById('level-change').innerHTML = formatChange(stats.player_level, lastStats.player_level);
                            
                            // Update badge progress
                            const badgeProgress = ((stats.badges_total || 0) / 16) * 100;
                            document.getElementById('badge-progress').style.width = badgeProgress + '%';
                            
                            // Update game state info
                            document.getElementById('player-map').textContent = stats.final_game_state?.player_map || '-';
                            document.getElementById('player-position').textContent = 
                                stats.final_game_state ? 
                                `(${stats.final_game_state.player_x}, ${stats.final_game_state.player_y})` : '-';
                            document.getElementById('player-money').textContent = formatNumber(stats.final_game_state?.money || 0);
                            document.getElementById('party-count').textContent = stats.final_game_state?.party_count || 0;
                            document.getElementById('player-hp').textContent = 
                                stats.final_game_state ? 
                                `${stats.final_game_state.player_hp}/${stats.final_game_state.player_max_hp}` : '-';
                            
                            // Update screen state
                            document.getElementById('screen-state').textContent = stats.screen_state || 'Unknown';
                            
                            // Update reward breakdown with bars
                            const rewardDiv = document.getElementById('reward-breakdown');
                            if (stats.last_reward_breakdown && stats.last_reward_breakdown !== 'no rewards') {
                                const rewards = stats.last_reward_breakdown.split(' | ');
                                rewardDiv.innerHTML = rewards.map(reward => {
                                    const [category, value] = reward.split(': ');
                                    const numValue = parseFloat(value);
                                    const className = numValue > 0 ? 'positive' : 'negative';
                                    return `<div class="reward-bar ${className}">${category}: <strong>${value}</strong></div>`;
                                }).join('');
                            } else {
                                rewardDiv.innerHTML = '<div class="reward-bar">No active rewards</div>';
                            }
                            
                            // Update LLM decisions with enhanced formatting
                            const decisionsDiv = document.getElementById('llm-decisions-list');
                            if (stats.recent_llm_decisions && stats.recent_llm_decisions.length > 0) {
                                decisionsDiv.innerHTML = stats.recent_llm_decisions.map((d, index) => {
                                    const timestamp = new Date(d.timestamp * 1000);
                                    return `<div class="llm-decision">
                                        <div class="llm-action">Action: ${d.action.toUpperCase()}</div>
                                        <div class="llm-reasoning">${d.reasoning.substring(0, 150)}${d.reasoning.length > 150 ? '...' : ''}</div>
                                        <div class="llm-timestamp">${timestamp.toLocaleTimeString()}</div>
                                    </div>`;
                                }).join('');
                            } else {
                                decisionsDiv.innerHTML = '<div class="llm-decision">No LLM decisions yet...</div>';
                            }
                            
                            lastStats = {...stats};
                            
                        } catch (e) {
                            console.error('Failed to update stats:', e);
                            document.getElementById('status-indicator').style.background = '#ff4757';
                        }
                    }
                    
                    // Memory debugging variables
                    let lastMemoryValues = {};
                    
                    function updateMemoryDebugger() {
                        fetch('/memory')
                            .then(response => response.json())
                            .then(memData => {
                                // Update core addresses
                                updateMemoryValue('mem-party-count', memData.party_count, lastMemoryValues.party_count);
                                updateMemoryValue('mem-map-id', memData.player_map, lastMemoryValues.player_map);
                                updateMemoryValue('mem-player-x', memData.player_x, lastMemoryValues.player_x);
                                updateMemoryValue('mem-player-y', memData.player_y, lastMemoryValues.player_y);
                                updateMemoryValue('mem-money', `¥${memData.money || 0}`, lastMemoryValues.money);
                                updateMemoryValue('mem-badges', `0x${(memData.badges || 0).toString(16).toUpperCase().padStart(2, '0')}`, lastMemoryValues.badges, true);
                                updateMemoryValue('mem-battle', memData.in_battle ? '1' : '0', lastMemoryValues.in_battle);
                                
                                // Update first Pokemon stats
                                updateMemoryValue('mem-species', memData.player_species || 0, lastMemoryValues.player_species);
                                updateMemoryValue('mem-level', memData.player_level || 0, lastMemoryValues.player_level);
                                updateMemoryValue('mem-hp', `${memData.player_hp || 0}`, lastMemoryValues.player_hp);
                                updateMemoryValue('mem-max-hp', `${memData.player_max_hp || 0}`, lastMemoryValues.player_max_hp);
                                
                                // Store current values as previous for next comparison
                                lastMemoryValues = {
                                    party_count: memData.party_count,
                                    player_map: memData.player_map,
                                    player_x: memData.player_x,
                                    player_y: memData.player_y,
                                    money: memData.money,
                                    badges: memData.badges,
                                    in_battle: memData.in_battle,
                                    player_species: memData.player_species,
                                    player_level: memData.player_level,
                                    player_hp: memData.player_hp,
                                    player_max_hp: memData.player_max_hp
                                };
                            })
                            .catch(e => {
                                console.warn('Memory update failed:', e);
                            });
                    }
                    
                    function updateMemoryValue(elementId, newValue, oldValue, isHex = false) {
                        const element = document.getElementById(elementId);
                        if (!element) return;
                        
                        const displayValue = newValue !== undefined ? newValue : '-';
                        element.textContent = displayValue;
                        
                        // Add flash animation if value changed
                        if (oldValue !== undefined && newValue !== oldValue && newValue !== undefined) {
                            element.classList.add('changed');
                            setTimeout(() => {
                                element.classList.remove('changed');
                            }, 500);
                        }
                    }
                    
                    // Update stats every 400ms for better responsiveness
                    setInterval(updateStats, 400);
                    
                    // Update memory debugger every 150ms for real-time feel
                    setInterval(updateMemoryDebugger, 150);
                    
                    // Update game screen every 200ms for smooth visuals
                    setInterval(updateGameScreen, 200);
                    
                    // Update refresh rate display
                    document.getElementById('refresh-rate').textContent = '200ms';
                    
                    // Initial load
                    updateStats();
                    updateMemoryDebugger();
                    updateGameScreen();
                </script>
            </body>
            </html>
            """
            self.wfile.write(html.encode())
            
        elif self.path == '/stats':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            stats = getattr(self.server, 'trainer_stats', {})
            self.wfile.write(json.dumps(stats).encode())
            
        elif self.path == '/memory':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Expires', '0')
            self.end_headers()
            
            # Get live memory data directly from the trainer
            memory_data = getattr(self.server, 'live_memory_data', {})
            self.wfile.write(json.dumps(memory_data).encode())
            
        elif self.path.startswith('/screenshot'):
            self.send_response(200)
            self.send_header('Content-type', 'image/png')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Expires', '0')
            self.end_headers()
            
            screenshot_data = getattr(self.server, 'screenshot_data', None)
            if screenshot_data:
                self.wfile.write(screenshot_data)
            else:
                # Send empty 1x1 PNG if no screenshot
                empty_img = Image.new('RGB', (1, 1), (0, 0, 0))
                buf = io.BytesIO()
                empty_img.save(buf, format='PNG')
                self.wfile.write(buf.getvalue())
        
        else:
            self.send_response(404)
            self.end_headers()
    
    def log_message(self, format, *args):
        pass  # Suppress HTTP server logs

class LLMPokemonTrainer:
    """Advanced Pokemon Crystal trainer with LLM integration and reward system"""
    
    def __init__(self, rom_path, max_actions=5000, llm_model="smollm2:1.7b", 
                 llm_interval=20, enable_web=True, web_port=8080, enable_dqn=True, 
                 dqn_model_path=None):
        self.rom_path = rom_path
        self.max_actions = max_actions
        self.llm_interval = llm_interval
        self.enable_web = enable_web
        self.web_port = web_port
        self.enable_dqn = enable_dqn
        
        # Core components
        self.pyboy = None
        self.llm_agent = LLMAgent(llm_model)
        self.reward_calculator = PokemonRewardCalculator()
        
        # DQN components
        self.dqn_agent = None
        self.hybrid_agent = None
        self.dqn_training_frequency = 4  # Train DQN every N actions
        self.dqn_save_frequency = 500  # Save DQN model every N actions
        
        if self.enable_dqn:
            # Initialize DQN agent
            self.dqn_agent = DQNAgent(
                state_size=32,
                action_size=8,
                learning_rate=1e-4,
                gamma=0.99,
                epsilon_start=0.9,
                epsilon_end=0.05,
                epsilon_decay=0.995,
                memory_size=50000,
                batch_size=32,
                target_update=1000
            )
            
            # Load existing model if provided
            if dqn_model_path and os.path.exists(dqn_model_path):
                self.dqn_agent.load_model(dqn_model_path)
            
            # Create hybrid agent combining LLM and DQN
            self.hybrid_agent = HybridAgent(
                dqn_agent=self.dqn_agent,
                llm_agent=self.llm_agent,
                dqn_weight=0.2,  # Start with low DQN influence
                exploration_bonus=0.1
            )
            
            print(f"🧠 DQN Agent initialized with {self.dqn_agent.device}")
        
        # Experience tracking
        self.recent_situation_hashes = []
        self.recent_action_sequences = []
        self.experience_window = 10  # Track last N actions for experience recording
        
        # Training state
        self.actions_taken = 0
        self.start_time = time.time()
        self.previous_game_state = {}
        self.total_reward = 0.0
        self.last_llm_decision_action = 0
        
        # Failsafe mechanism for stuck detection
        self.last_positive_reward_action = 0
        self.actions_without_reward = 0
        self.stuck_threshold = 100  # Actions without reward before intervention
        self.location_stuck_tracker = {}  # Track how long we've been in same location
        
        # Statistics
        self.stats = {
            'actions_taken': 0,
            'training_time': 0,
            'actions_per_second': 0,
            'llm_decision_count': 0,
            'total_reward': 0.0,
            'player_level': 0,
            'badges_total': 0,
            'last_reward_breakdown': '',
            'recent_llm_decisions': [],
            'start_time': datetime.now().isoformat()
        }
        
        self.running = True
        self.recent_actions = []
        
        # Web server setup
        self.web_server = None
        self.web_thread = None
        
        # Setup graceful shutdown
        signal.signal(signal.SIGINT, self.shutdown)
        signal.signal(signal.SIGTERM, self.shutdown)
        
        print(f"🤖 LLM Agent: {llm_model} {'✅' if self.llm_agent.available else '❌'}")
        
    def shutdown(self, signum, frame):
        """Graceful shutdown handler"""
        print(f"\n⏸️ Shutting down LLM training...")
        self.running = False
        
        if self.web_server:
            print("Stopping web server...")
            try:
                self.web_server.shutdown()
            except:
                pass
            
        if self.pyboy:
            self.pyboy.stop()
            
        self.save_training_data()
        print("✅ Training stopped cleanly")
        sys.exit(0)
    
    def setup_web_server(self):
        """Setup enhanced web monitoring server"""
        if not self.enable_web:
            return
            
        try:
            self.web_server = HTTPServer(('localhost', self.web_port), WebMonitor)
            self.web_server.trainer_stats = self.stats
            self.web_server.screenshot_data = None
            
            self.web_thread = threading.Thread(target=self.web_server.serve_forever, daemon=True)
            self.web_thread.start()
            
            print(f"🌐 Enhanced web monitor: http://localhost:{self.web_port}")
            
        except Exception as e:
            print(f"⚠️ Failed to start web server: {e}")
            self.enable_web = False
    
    def initialize_pyboy(self):
        """Initialize PyBoy emulator"""
        print(f"🎮 Initializing PyBoy with {self.rom_path}")
        self.pyboy = PyBoy(self.rom_path, window="null", debug=False)
        
        # Load save state if available
        save_state_path = self.rom_path + '.state'
        if os.path.exists(save_state_path):
            print(f"💾 Loading save state: {save_state_path}")
            with open(save_state_path, 'rb') as f:
                self.pyboy.load_state(f)
            print("✅ Save state loaded - starting from saved position")
        
        print("✅ PyBoy initialized successfully")
        
        # Get initial game state
        self.previous_game_state = self.get_game_state()
        
    def get_game_state(self) -> Dict:
        """Extract comprehensive game state from memory using validated addresses"""
        if not self.pyboy:
            return {}
        
        # Use the build_observation function with validated memory structure
        return build_observation(self.pyboy.memory)
    
    def analyze_screen(self) -> Dict:
        """Analyze current screen state with improved detection"""
        if not self.pyboy:
            return {'state': 'unknown', 'variance': 0, 'colors': 0}
            
        screen = self.pyboy.screen.ndarray
        variance = float(np.var(screen.astype(np.float32)))
        
        # Get additional screen analysis metrics
        unique_colors = len(np.unique(screen.reshape(-1, screen.shape[-1]), axis=0))
        brightness = float(np.mean(screen.astype(np.float32)))
        
        # CRITICAL: Very few colors (2-3) almost always means menu/battle/evolution
        # This prevents false positives where menus are classified as overworld
        if unique_colors <= 3:
            # Very few colors - definitely not overworld
            if variance < 50:
                state = "loading"  # Solid colors or very simple screen
            elif brightness > 200:
                state = "dialogue"  # High brightness with few colors = dialogue box
            else:
                state = "menu"  # Low brightness with few colors = menu/battle/evolution
        # Very low variance = loading/transition screen
        elif variance < 50:
            state = "loading"
        # Very high variance with many colors = battle screen (lots of sprites/effects)
        elif variance > 20000 and unique_colors > 8:
            state = "battle"
        # Medium-high variance with many colors = overworld
        elif variance > 3000 and unique_colors > 10:
            state = "overworld"
        # Low variance patterns
        elif variance < 3000:
            # Further distinguish between menu and dialogue
            if brightness > 200 and unique_colors < 8:
                # Very bright with few colors = likely dialogue box
                state = "dialogue"
            elif unique_colors < 6:
                # Few colors = menu system
                state = "menu"
            elif variance > 500 and unique_colors >= 8:
                # Some variance with multiple colors = likely settings/menu
                state = "settings_menu"
            else:
                # Default to menu for low variance screens
                state = "menu"
        else:
            # Medium variance with reasonable colors - could be overworld
            if unique_colors > 8:
                state = "overworld"
            else:
                state = "menu"  # Conservative: few colors = likely menu
            
        return {
            'state': state,
            'variance': variance,
            'colors': unique_colors,
            'brightness': brightness
        }
    
    def get_next_action(self) -> Tuple[str, str]:
        """Get next action using hybrid DQN+LLM or fallback logic"""
        game_state = self.get_game_state()
        screen_analysis = self.analyze_screen()
        
        # CRITICAL: Forbid START and SELECT until first Pokemon is obtained
        party_count = game_state.get('party_count', 0)
        forbidden_actions = set()
        if party_count == 0:
            forbidden_actions.add('start')
            forbidden_actions.add('select')
        
        # If DQN is enabled and available, use hybrid approach
        if self.enable_dqn and self.hybrid_agent:
            # Use hybrid agent that combines LLM reasoning with DQN experience
            use_llm = (self.actions_taken - self.last_llm_decision_action) >= self.llm_interval
            
            if use_llm and self.llm_agent.available:
                # Get hybrid decision combining LLM and DQN
                action, reasoning = self.hybrid_agent.get_hybrid_action(
                    game_state, screen_analysis, self.recent_actions
                )
                
                # Override forbidden actions
                if action in forbidden_actions:
                    action = self._get_allowed_alternative_action(action, game_state, screen_analysis)
                    reasoning = f"Forbidden {action} -> using allowed alternative"
                
                self.last_llm_decision_action = self.actions_taken
                self.stats['llm_decision_count'] += 1
                
                # Track recent decisions for web display
                self.stats['recent_llm_decisions'].append({
                    'action': action,
                    'reasoning': reasoning,
                    'timestamp': time.time()
                })
                
                # Keep only last 5 decisions
                if len(self.stats['recent_llm_decisions']) > 5:
                    self.stats['recent_llm_decisions'].pop(0)
                
                return action, f"Hybrid: {reasoning[:50]}..."
            else:
                # Use DQN-only action selection between LLM decisions
                action, q_value = self.dqn_agent.get_action(game_state, screen_analysis, training=True)
                
                # Override forbidden actions
                if action in forbidden_actions:
                    action = self._get_allowed_alternative_action(action, game_state, screen_analysis)
                
                dqn_info = self.hybrid_agent.get_info()
                return action, f"DQN (Q={q_value:.3f}) - {dqn_info}"
        
        # Fallback to original LLM-only logic
        elif self.llm_agent.available:
            # Use LLM every N actions
            use_llm = (self.actions_taken - self.last_llm_decision_action) >= self.llm_interval
            
            if use_llm:
                action, reasoning = self.llm_agent.get_decision(game_state, screen_analysis, self.recent_actions)
                
                # Override forbidden actions
                if action in forbidden_actions:
                    action = self._get_allowed_alternative_action(action, game_state, screen_analysis)
                    reasoning = f"Forbidden action -> using {action}"
                
                self.last_llm_decision_action = self.actions_taken
                self.stats['llm_decision_count'] += 1
                
                # Track recent LLM decisions for web display
                self.stats['recent_llm_decisions'].append({
                    'action': action,
                    'reasoning': reasoning,
                    'timestamp': time.time()
                })
                
                # Keep only last 5 decisions
                if len(self.stats['recent_llm_decisions']) > 5:
                    self.stats['recent_llm_decisions'].pop(0)
                
                return action, f"LLM: {reasoning[:50]}..."
            else:
                # Fallback rule-based action
                action = self._get_rule_based_action(game_state, screen_analysis)
                
                # Override forbidden actions
                if action in forbidden_actions:
                    action = self._get_allowed_alternative_action(action, game_state, screen_analysis)
                
                return action, "Rule-based fallback"
        else:
            # No LLM available, use rule-based fallback
            action = self._get_rule_based_action(game_state, screen_analysis)
            
            # Override forbidden actions
            if action in forbidden_actions:
                action = self._get_allowed_alternative_action(action, game_state, screen_analysis)
            
            return action, "Rule-based fallback"
    
    def _get_allowed_alternative_action(self, forbidden_action: str, game_state: Dict, screen_analysis: Dict) -> str:
        """Get an allowed alternative when an action is forbidden (START/SELECT before first Pokemon)"""
        state_type = screen_analysis.get('state', 'unknown')
        
        # Context-aware alternatives based on screen state
        if game_state.get('in_battle', 0) == 1:
            return 'a'  # Always attack in battle
        elif state_type == 'dialogue':
            return 'a'  # Progress dialogue
        elif state_type == 'menu':
            return 'b'  # Exit menus when we can't use START
        elif state_type == 'loading':
            return 'a'  # Wait during loading
        else:
            # In overworld - focus on exploration and interaction
            # Priority order: interact with objects/NPCs, then explore
            exploration_priority = ['a', 'up', 'down', 'left', 'right']
            return exploration_priority[self.actions_taken % len(exploration_priority)]
    
    def _get_rule_based_action(self, game_state: Dict, screen_analysis: Dict) -> str:
        """Rule-based fallback action with improved screen state handling"""
        state_type = screen_analysis.get('state', 'unknown')
        
        if game_state.get('in_battle', 0) == 1:
            return 'a'  # Attack in battle
        elif state_type == 'dialogue':
            return 'a'  # Progress dialogue
        elif state_type == 'settings_menu':
            return 'b'  # Always exit settings menu immediately
        elif state_type == 'menu':
            # Smart menu handling - check recent actions
            recent_actions_str = ' '.join(self.recent_actions[-3:]) if self.recent_actions else ''
            if 'START' in recent_actions_str:
                return 'b'  # Exit menu if we recently opened one
            else:
                return 'b'  # Default to exiting menus unless we have a specific goal
        elif state_type == 'loading':
            return 'a'  # Wait during loading screens
        else:
            # Exploration pattern for overworld (avoid START button spam)
            exploration_actions = ['up', 'up', 'a', 'right', 'right', 'a', 'down', 'down', 'a', 'left', 'left', 'a']
            return exploration_actions[self.actions_taken % len(exploration_actions)]
    
    def _update_stuck_detection(self, current_state: Dict, reward: float):
        """Update stuck detection mechanism and trigger failsafe if needed"""
        # Track positive rewards
        if reward > 0.05:  # Any positive reward (excluding tiny dialogue rewards)
            self.last_positive_reward_action = self.actions_taken
            self.actions_without_reward = 0
        else:
            self.actions_without_reward += 1
        
        # Track location-based stuck detection
        current_location = (
            current_state.get('player_map', 0),
            current_state.get('player_x', 0),
            current_state.get('player_y', 0)
        )
        
        if current_location not in self.location_stuck_tracker:
            self.location_stuck_tracker[current_location] = 0
        self.location_stuck_tracker[current_location] += 1
        
        # Clean up old location tracking (keep only recent 50 actions worth)
        if len(self.location_stuck_tracker) > 50:
            # Remove locations with low visit counts
            min_visits = min(self.location_stuck_tracker.values())
            self.location_stuck_tracker = {
                loc: count for loc, count in self.location_stuck_tracker.items() 
                if count > min_visits
            }
        
        # Check if we're stuck and need intervention
        self._check_failsafe_intervention(current_state, current_location)
    
    def _check_failsafe_intervention(self, current_state: Dict, current_location: Tuple):
        """Check if failsafe intervention is needed and modify LLM prompts accordingly"""
        # Detect stuck conditions
        stuck_too_long = self.actions_without_reward >= self.stuck_threshold
        stuck_at_location = self.location_stuck_tracker.get(current_location, 0) > 20
        
        if stuck_too_long or stuck_at_location:
            # Modify LLM behavior to break out of stuck pattern
            self._trigger_failsafe_intervention(current_state, current_location, stuck_too_long, stuck_at_location)
    
    def _trigger_failsafe_intervention(self, current_state: Dict, current_location: Tuple, 
                                     stuck_too_long: bool, stuck_at_location: bool):
        """Trigger failsafe intervention to break stuck patterns"""
        
        # Force next LLM decision with explicit stuck-breaking prompt
        self.last_llm_decision_action = self.actions_taken - self.llm_interval + 1
        
        # Add failsafe context to the LLM agent for next decision
        if not hasattr(self.llm_agent, 'failsafe_context'):
            self.llm_agent.failsafe_context = {}
        
        self.llm_agent.failsafe_context = {
            'stuck_detected': True,
            'stuck_location': current_location,
            'actions_without_reward': self.actions_without_reward,
            'stuck_reason': 'no_reward' if stuck_too_long else 'same_location',
            'intervention_action': self.actions_taken
        }
        
        print(f"🚨 FAILSAFE: Detected stuck behavior!")
        print(f"   📍 Location: Map {current_location[0]}, Pos ({current_location[1]},{current_location[2]})")
        print(f"   ⏱️ Actions without reward: {self.actions_without_reward}")
        print(f"   🎯 INTERVENTION: Providing concrete movement instructions...")
        
        # Reset some stuck tracking to prevent immediate re-triggering
        self.actions_without_reward = max(0, self.actions_without_reward - 20)
        if current_location in self.location_stuck_tracker:
            self.location_stuck_tracker[current_location] = max(1, 
                self.location_stuck_tracker[current_location] - 10)
    
    def execute_action(self, action: str):
        """Execute action and calculate rewards with smart movement handling"""
        if not self.running or not self.pyboy:
            return
            
        # Store previous state for reward calculation and DQN training
        previous_state = self.previous_game_state.copy()
        previous_screen_analysis = self.analyze_screen()
        
        # Check if this is a directional movement in overworld
        is_directional = action.lower() in ['up', 'down', 'left', 'right']
        screen_state = previous_screen_analysis.get('state', 'unknown')
        is_overworld = screen_state == 'overworld'
        
        if is_directional and is_overworld:
            # Smart directional movement: try twice if first attempt doesn't move
            current_state = self._execute_smart_movement(action, previous_state)
            current_screen_analysis = self.analyze_screen()
        else:
            # Normal action execution for non-directional actions or non-overworld
            self._execute_single_action(action)
            current_state = self.get_game_state()
            current_screen_analysis = self.analyze_screen()
        
        # Pass screen state info to reward calculator for BOTH exploration AND progression reward filtering
        self.reward_calculator.last_screen_state = current_screen_analysis.get('state', 'unknown')
        self.reward_calculator.prev_screen_state = previous_screen_analysis.get('state', 'unknown')
        
        # Pass action info to prevent SELECT button false rewards
        self.reward_calculator.last_action = action
        
        reward, reward_breakdown = self.reward_calculator.calculate_reward(current_state, previous_state)
        
        # DEBUG: Print all reward components if there's a significant reward
        if abs(reward) > 1.0:
            print(f"🔍 DEBUG REWARD: Total={reward:+.2f} | Action: {action}")
            for category, value in reward_breakdown.items():
                if abs(value) > 0.01:
                    print(f"   {category}: {value:+.2f}")
            print(f"   Party count: {previous_state.get('party_count', 0)} -> {current_state.get('party_count', 0)}")
            print(f"   Level: {previous_state.get('player_level', 0)} -> {current_state.get('player_level', 0)}")
            print(f"   Screen: {getattr(self.reward_calculator, 'prev_screen_state', '?')} -> {getattr(self.reward_calculator, 'last_screen_state', '?')}")
        
        # DQN experience storage and training
        if self.enable_dqn and self.dqn_agent:
            # Store experience in DQN replay buffer
            done = False  # We don't have episode termination in continuous play
            self.dqn_agent.store_experience(
                previous_state, previous_screen_analysis, action,
                reward, current_state, current_screen_analysis, done
            )
            
            # Train DQN periodically
            if self.actions_taken % self.dqn_training_frequency == 0:
                loss = self.dqn_agent.train_step()
                if loss > 0 and self.actions_taken % (self.dqn_training_frequency * 10) == 0:
                    print(f"🧠 DQN training: loss={loss:.4f}, ε={self.dqn_agent.epsilon:.3f}")
            
            # Record performance for hybrid agent adaptation
            if self.hybrid_agent:
                self.hybrid_agent.record_performance(reward)
            
            # Save DQN model periodically
            if self.actions_taken % self.dqn_save_frequency == 0:
                model_path = os.path.join("logs", f"dqn_model_{self.actions_taken}.pth")
                self.dqn_agent.save_model(model_path)
                print(f"🔄 DQN model saved at action {self.actions_taken}")
        
        # Track experience for learning
        self._track_experience(action, previous_state, current_state, reward)
        
        # Update tracking
        self.actions_taken += 1
        self.total_reward += reward
        self.previous_game_state = current_state
        
        # Update recent actions
        self.recent_actions.append(action.upper())
        if len(self.recent_actions) > 10:
            self.recent_actions.pop(0)
        
        # FAILSAFE: Track stuck detection
        self._update_stuck_detection(current_state, reward)
        
        # Update stats
        self.stats['actions_taken'] = self.actions_taken
        self.stats['total_reward'] = float(self.total_reward)
        self.stats['player_level'] = current_state.get('player_level', 0)
        self.stats['badges_total'] = current_state.get('badges_total', 0)
        self.stats['last_reward_breakdown'] = self.reward_calculator.get_reward_summary(reward_breakdown)
        
        # Add DQN stats if enabled
        if self.enable_dqn and self.dqn_agent:
            dqn_stats = self.dqn_agent.get_training_stats()
            self.stats['dqn_steps'] = dqn_stats['steps_trained']
            self.stats['dqn_epsilon'] = dqn_stats['epsilon']
            self.stats['dqn_memory_size'] = dqn_stats['memory_size']
        
        return reward, reward_breakdown
    
    def _execute_single_action(self, action: str):
        """Execute a single action press"""
        self.pyboy.button_press(action)
        
        # Hold button for several frames
        for _ in range(8):
            if not self.running:
                break
            self.pyboy.tick()
        
        self.pyboy.button_release(action)
        
        # Wait a few more frames
        for _ in range(4):
            if not self.running:
                break
            self.pyboy.tick()
    
    def _execute_smart_movement(self, direction: str, previous_state: Dict) -> Dict:
        """Execute directional movement with automatic retry logic"""
        # Get initial position
        prev_x = previous_state.get('player_x', 0)
        prev_y = previous_state.get('player_y', 0)
        prev_map = previous_state.get('player_map', 0)
        
        # First attempt: press direction key
        self._execute_single_action(direction)
        
        # Check if we moved
        intermediate_state = self.get_game_state()
        curr_x = intermediate_state.get('player_x', 0)
        curr_y = intermediate_state.get('player_y', 0)
        curr_map = intermediate_state.get('player_map', 0)
        
        # Check if position changed (either coordinates or map)
        position_changed = (curr_x != prev_x) or (curr_y != prev_y) or (curr_map != prev_map)
        
        if not position_changed:
            # First press likely just changed facing direction, try again
            self._execute_single_action(direction)
            
            # Get final state after second attempt
            final_state = self.get_game_state()
            final_x = final_state.get('player_x', 0)
            final_y = final_state.get('player_y', 0)
            final_map = final_state.get('player_map', 0)
            
            # Check if second attempt moved us
            second_attempt_moved = (final_x != curr_x) or (final_y != curr_y) or (final_map != curr_map)
            
            if not second_attempt_moved:
                # Still didn't move - likely blocked by wall or edge
                print(f"🚧 Movement blocked: {direction.upper()} at ({curr_x},{curr_y}) on Map {curr_map}")
            else:
                print(f"↔️ Smart movement: {direction.upper()} → ({prev_x},{prev_y}) → ({final_x},{final_y})")
            
            return final_state
        else:
            # First press actually moved us (unusual but possible)
            print(f"⚡ Direct movement: {direction.upper()} → ({prev_x},{prev_y}) → ({curr_x},{curr_y})")
            return intermediate_state
    
    def _track_experience(self, action: str, previous_state: Dict, current_state: Dict, reward: float):
        """Track experience for learning system"""
        # Create screen analysis for previous state
        screen_analysis = self.analyze_screen()
        
        # Get game context
        game_context = self.llm_agent.game_intelligence.analyze_game_context(previous_state, screen_analysis)
        
        # Create situation hash
        situation_hash = self.llm_agent.experience_memory.get_situation_hash(
            previous_state, 
            screen_analysis, 
            {
                'phase': game_context.phase.name,
                'location_type': game_context.location_type.name
            }
        )
        
        # Add to recent tracking
        self.recent_situation_hashes.append(situation_hash)
        self.recent_action_sequences.append(action.upper())
        
        # Keep window size manageable
        if len(self.recent_action_sequences) > self.experience_window:
            self.recent_situation_hashes.pop(0)
            self.recent_action_sequences.pop(0)
        
        # Record significant experiences (positive rewards or major events)
        if reward > 0.1 or abs(reward) > 10.0:  # Significant positive or major event
            if len(self.recent_action_sequences) >= 3:  # Need some history
                self.llm_agent.experience_memory.record_experience(
                    situation_hash=situation_hash,
                    actions=self.recent_action_sequences[-5:],  # Last 5 actions
                    reward=reward,
                    context={
                        'phase': game_context.phase.name,
                        'location_type': game_context.location_type.name,
                        'timestamp': time.time()
                    }
                )
    
    def update_web_data(self):
        """Update data for web monitoring"""
        if not self.enable_web or not self.web_server:
            return
            
        # Update performance stats
        elapsed = time.time() - self.start_time
        self.stats['training_time'] = elapsed
        self.stats['actions_per_second'] = self.actions_taken / elapsed if elapsed > 0 else 0
        
        # Update current game state and screen analysis
        current_game_state = self.get_game_state()
        screen_analysis = self.analyze_screen()
        
        self.stats['final_game_state'] = current_game_state
        self.stats['screen_state'] = screen_analysis.get('state', 'unknown')
        
        # Update screenshot
        try:
            screen = self.pyboy.screen.ndarray
            if screen.shape[2] == 4:  # RGBA to RGB
                screen = screen[:, :, :3]
            
            img = Image.fromarray(screen)
            img = img.resize((480, 432), Image.NEAREST)  # Match the display size
            
            buf = io.BytesIO()
            img.save(buf, format='PNG', optimize=True)
            self.web_server.screenshot_data = buf.getvalue()
            
        except Exception as e:
            pass  # Ignore screenshot errors
        
        # Update live memory data for the web interface
        self.web_server.live_memory_data = current_game_state.copy()
        
        # Update server stats
        self.web_server.trainer_stats = self.stats.copy()
    
    def print_progress(self, action: str, decision_source: str, reward: float, reward_breakdown: Dict):
        """Print detailed training progress"""
        elapsed = time.time() - self.start_time
        aps = self.actions_taken / elapsed if elapsed > 0 else 0
        
        game_state = self.previous_game_state
        screen_analysis = self.analyze_screen()
        
        reward_summary = self.reward_calculator.get_reward_summary(reward_breakdown)
        
        # Enhanced screen analysis display
        screen_info = f"{screen_analysis['state']} (v:{screen_analysis['variance']:.0f}, b:{screen_analysis['brightness']:.0f}, c:{screen_analysis['colors']})"
        
        print(f"⚡ Action {self.actions_taken}/{self.max_actions} | {action.upper()} ({decision_source})")
        print(f"   📊 {aps:.1f} a/s | Screen: {screen_info} | Level: {game_state.get('player_level', 0)} | Badges: {game_state.get('badges_total', 0)}")
        print(f"   💰 Reward: {reward:+.2f} (Total: {self.total_reward:.2f}) | {reward_summary}")
        print()
    
    def save_training_data(self):
        """Save comprehensive training data"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Ensure logs directory exists
        logs_dir = "logs"
        os.makedirs(logs_dir, exist_ok=True)
        
        # Save main stats
        stats_file = os.path.join(logs_dir, f"llm_training_stats_{timestamp}.json")
        final_stats = self.stats.copy()
        final_stats['final_game_state'] = self.get_game_state()
        final_stats['llm_decisions'] = len(self.llm_agent.decision_history)
        
        with open(stats_file, 'w') as f:
            json.dump(final_stats, f, indent=2)
        
        # Save detailed LLM decisions
        if self.llm_agent.decision_history:
            llm_file = os.path.join(logs_dir, f"llm_decisions_{timestamp}.json")
            with open(llm_file, 'w') as f:
                json.dump(self.llm_agent.decision_history, f, indent=2)
            print(f"🧠 LLM decisions saved to {llm_file}")
        
        # Save final DQN model if enabled
        if self.enable_dqn and self.dqn_agent:
            final_model_path = os.path.join(logs_dir, f"dqn_final_model_{timestamp}.pth")
            self.dqn_agent.save_model(final_model_path)
            print(f"🧠 Final DQN model saved to {final_model_path}")
        
        # Save experience memory
        self.llm_agent.experience_memory.save_memory()
        memory_stats = self.llm_agent.experience_memory.get_memory_stats()
        print(f"📚 Experience memory saved: {memory_stats['total_experiences']} experiences, {memory_stats['total_patterns']} patterns")
        
        print(f"📊 Training stats saved to {stats_file}")
    
    def start_training(self):
        """Start the LLM-enhanced training process"""
        print("🚀 Starting LLM-Enhanced Pokemon Crystal RL Training")
        print("=" * 80)
        print(f"🤖 LLM Model: {self.llm_agent.model_name}")
        print(f"🧠 LLM Decision Interval: Every {self.llm_interval} actions")
        print(f"💰 Reward System: Multi-factor Pokemon progress tracking")
        print()
        
        try:
            # Setup components
            self.setup_web_server()
            self.initialize_pyboy()
            
            print(f"🎯 Starting training loop ({self.max_actions} actions)")
            if self.enable_web:
                print(f"🌐 Monitor at: http://localhost:{self.web_port}")
            print("🔄 Press Ctrl+C to stop training gracefully")
            print("=" * 80)
            print()
            
            # Main training loop
            while self.running and self.actions_taken < self.max_actions:
                # Get next action
                action, decision_source = self.get_next_action()
                
                # Execute action and get reward
                reward, reward_breakdown = self.execute_action(action)
                
                # Update web monitoring
                if self.actions_taken % 5 == 0:
                    self.update_web_data()
                
                # Print progress every 50 actions or on LLM decisions
                if (self.actions_taken % 50 == 0 or "LLM" in decision_source):
                    self.print_progress(action, decision_source, reward, reward_breakdown)
            
            # Training completed
            if self.actions_taken >= self.max_actions:
                print(f"✅ LLM training completed! {self.actions_taken} actions executed")
                print(f"🏆 Final reward: {self.total_reward:.2f}")
                print(f"🧠 LLM decisions made: {self.stats['llm_decision_count']}")
                
            return True
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            import traceback
            traceback.print_exc()
            return False
            
        finally:
            # Cleanup
            if self.web_server:
                try:
                    self.web_server.shutdown()
                except:
                    pass
                    
            if self.pyboy:
                self.pyboy.stop()
            
            self.save_training_data()

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Hybrid LLM+DQN Pokemon Crystal RL Training")
    parser.add_argument("--rom", default="roms/pokemon_crystal.gbc", help="ROM file path")
    parser.add_argument("--actions", type=int, default=2000, help="Number of actions to execute")
    parser.add_argument("--llm-model", default="smollm2:1.7b", 
                       choices=["smollm2:1.7b", "llama3.2:1b", "llama3.2:3b", "deepseek-coder:latest"],
                       help="LLM model to use")
    parser.add_argument("--llm-interval", type=int, default=20, 
                       help="Actions between LLM decisions")
    parser.add_argument("--web-port", type=int, default=8080, help="Web monitoring port")
    parser.add_argument("--no-web", action="store_true", help="Disable web monitoring")
    parser.add_argument("--no-dqn", action="store_true", help="Disable DQN agent (LLM-only mode)")
    parser.add_argument("--dqn-model", type=str, help="Path to pre-trained DQN model to load")
    
    args = parser.parse_args()
    
    # Validate ROM file
    if not os.path.exists(args.rom):
        print(f"❌ ROM file not found: {args.rom}")
        return 1
    
    # Validate DQN model path if provided
    if args.dqn_model and not os.path.exists(args.dqn_model):
        print(f"⚠️ DQN model file not found: {args.dqn_model}")
        print("Starting with fresh DQN model...")
        args.dqn_model = None
    
    # Create and start trainer
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
    
    success = trainer.start_training()
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
