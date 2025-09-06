"""
LLM Agent for Pokemon Crystal RL

This module contains the LLMAgent class, which provides AI decision-making
capabilities for Pokemon Crystal using Large Language Models. The agent
handles strategic decision-making, maintains decision history, and includes
fallback mechanisms for when the LLM is unavailable.

Features:
- LLM-based decision making with strategic context
- Decision history tracking
- Rule-based fallbacks when LLM is unavailable
- Enhanced prompt building with game intelligence
- Experience memory integration
- Failsafe intervention support
"""

import time
import requests
import logging
from typing import Dict, List, Tuple

from utils.action_parser import parse_action_response
from config.constants import TRAINING_PARAMS

from core.game_intelligence import GameIntelligence
from core.experience_memory import ExperienceMemory
from core.strategic_context_builder import StrategicContextBuilder, DecisionContext


class LLMAgent:
    """Enhanced LLM agent with strategic decision-making capabilities"""
    
    def __init__(self, model_name="smollm2:1.7b", base_url="http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
        self.decision_history = []
        self.last_decision_time = 0
        
        # Initialize enhanced systems
        self.game_intelligence = GameIntelligence()
        self.experience_memory = ExperienceMemory()
        self.context_builder = StrategicContextBuilder()
        
        # Test LLM availability
        self.available = self._test_llm_connection()
        if not self.available:
            print("⚠️ LLM not available - will use rule-based fallbacks")
        
    def _test_llm_connection(self) -> bool:
        """Test if LLM is available"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def get_decision(self, game_state: Dict, screen_analysis: Dict, recent_actions: List[str]) -> Tuple[str, str]:
        """Get LLM decision based on comprehensive state analysis"""
        if not self.available:
            return self._fallback_decision(game_state), "LLM unavailable - using fallback"
        
        try:
            # Build comprehensive decision context
            context = self.context_builder.build_context(
                game_state, recent_actions[-1] if recent_actions else None, None
            )
            
            # Get prompt using enhanced context
            prompt = self._build_prompt(game_state, screen_analysis, recent_actions, context)
            
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
    
    def _build_prompt(self, game_state: Dict, screen_analysis: Dict, recent_actions: List[str], 
                       decision_context: DecisionContext) -> str:
        """Build enhanced prompt with strategic context and intelligence"""
        
        # Check if failsafe intervention is active
        failsafe_context = getattr(self, 'failsafe_context', {})
        is_failsafe_active = failsafe_context.get('stuck_detected', False)
        
        # Use game intelligence and decision context
        game_context = self.game_intelligence.analyze_game_context(game_state, screen_analysis)
        action_plans = self.game_intelligence.get_action_plan(game_context, game_state)
        contextual_advice = self.game_intelligence.get_contextual_advice(game_context, recent_actions)
        strategic_context = decision_context.current_analysis
        
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
        
# Add strategic analysis if available
        strategic_text = ""
        if strategic_context:
            strategic_text = f"""

STRATEGIC ANALYSIS:
- Phase: {strategic_context.phase.name}
- Criticality: {strategic_context.criticality.value}/5
- Progress: {strategic_context.progression_score}%
- Threats: {', '.join(strategic_context.immediate_threats) if strategic_context.immediate_threats else 'None'}
- Opportunities: {', '.join(strategic_context.opportunities) if strategic_context.opportunities else 'None'}"""

        # Add learned experience if available
        experience_text = ""
        if learned_actions:
            experience_text = f"\n\nLEARNED EXPERIENCE: In similar situations, these actions worked well: {' → '.join(learned_actions)}"
            memory_stats = self.experience_memory.get_memory_stats()
            experience_text += f"\n(Based on {memory_stats['total_experiences']} past experiences)"
        
        # Modify guidelines based on failsafe state
        movement_guidelines = ""
        if is_failsafe_active:
            movement_guidelines = f"""\nFAILSAFE MOVEMENT STRATEGY:
🎯 PRIMARY GOAL: Change your position coordinates!
- Current coordinates: ({current_x},{current_y}) on Map {current_map}
- Target: ANY different coordinates or map
- Method: Try each direction (up/down/left/right) systematically
- If hitting walls: coordinates won't change, try different direction
- If coordinates change: GOOD! Continue in that direction
- Look for map transitions (screen changes, loading)"""
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
        """Parse LLM response to extract action with enhanced synonym recognition"""
        response_lower = response.lower().strip()
        
        # Enhanced action word mapping with synonyms
        action_mappings = {
            # Basic directions
            'up': ['up', 'north', 'forward'],
            'down': ['down', 'south', 'backward'],
            'left': ['left', 'west'],
            'right': ['right', 'east'],
            # Buttons
            'a': ['a', 'interact', 'confirm', 'attack', 'select_pokemon', 'use'],
            'b': ['b', 'cancel', 'back', 'flee', 'run', 'escape'],
            'start': ['start', 'menu', 'pause'],
            'select': ['select']
        }
        
        # Look for ACTION: pattern first
        if "action:" in response_lower:
            action_part = response_lower.split("action:")[1].split("\n")[0].strip()
            words = action_part.split()
            for word in words:
                clean_word = word.strip('.,!?()[]{}').lower()
                for action, synonyms in action_mappings.items():
                    if clean_word in synonyms:
                        return action
        
        # Look for any action word or synonym in the response
        for action, synonyms in action_mappings.items():
            for synonym in synonyms:
                if synonym in response_lower:
                    return action
        
        # Fallback to 'a' if nothing found
        return 'a'
    
    def _fallback_decision(self, game_state: Dict) -> str:
        """Enhanced fallback decision with stuck pattern detection"""
        # Check for stuck patterns in recent actions
        if hasattr(self, 'recent_actions') and len(self.recent_actions) >= 5:
            recent_5 = self.recent_actions[-5:]
            if len(set(recent_5)) <= 2:  # Only 1-2 unique actions in last 5
                # Try an action not recently used
                all_actions = ['up', 'down', 'left', 'right', 'a', 'b']
                unused_actions = [a for a in all_actions if a not in recent_5[-3:]]
                if unused_actions:
                    return unused_actions[0]  # Use first unused action
        
        # Smart fallback based on game state
        if game_state.get('in_battle', 0) == 1:
            return 'a'  # Attack in battle
        elif game_state.get('menu_state', 0) != 0:
            return 'a'  # Navigate menus
        else:
            # Phase-aware exploration pattern
            phase = game_state.get('phase', 'exploration')
            if phase == 'early_game':
                preferred_actions = ['a', 'up', 'right', 'down']  # Interaction-focused
            elif phase == 'starter_phase':
                preferred_actions = ['a', 'up', 'down', 'left', 'right']  # Training-focused
            else:
                preferred_actions = ['up', 'right', 'down', 'left', 'a', 'start']  # General exploration
            
            # Choose action not recently used
            if hasattr(self, 'recent_actions') and len(self.recent_actions) >= 2:
                for action in preferred_actions:
                    if action not in self.recent_actions[-2:]:
                        return action
            
            # Fallback to time-based selection
            return preferred_actions[int(time.time()) % len(preferred_actions)]