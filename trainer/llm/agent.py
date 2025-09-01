"""
LLM-based decision making agent for Pokemon Crystal.
"""

import time
import requests
from typing import Dict, List, Tuple

from core.game_intelligence import GameIntelligence
from core.experience_memory import ExperienceMemory

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
            blocked_dirs = failsafe_context.get('blocked_directions', [])
            available_dirs = failsafe_context.get('available_directions', ['up', 'down', 'left', 'right'])
            
            blocked_text = f"Blocked directions: {', '.join(blocked_dirs)}" if blocked_dirs else "No directions blocked yet"
            available_text = f"Available directions: {', '.join(available_dirs)}"
            
            failsafe_guidance = f"""\n🚨 FAILSAFE INTERVENTION ACTIVE! 🚨
You are STUCK at Map {stuck_location[0]}, Position ({stuck_location[1]},{stuck_location[2]})
Actions without progress: {actions_without_reward}
{blocked_text}
{available_text}

INTELLIGENT MOVEMENT STRATEGY:
- PRIORITIZE available directions: {', '.join(available_dirs)}
- AVOID blocked directions: {', '.join(blocked_dirs) if blocked_dirs else 'none blocked yet'}
- If all directions blocked, try 'A' to interact with objects/NPCs
- Focus on changing coordinates to escape stuck location

SPECIFIC MOVEMENT INSTRUCTIONS:
- If you're in the bedroom (Map 24) at positions (0,0), (1,0), or (2,0):
  * The door is DOWN and to the RIGHT from the starting position
  * Try: DOWN → DOWN → RIGHT → DOWN to exit the room
  * Avoid pressing 'A' near the radio (top-left area)
- If stuck repeating same movements: try the OPPOSITE direction
- If coordinates aren't changing: you might be hitting walls - try a different direction
- Priority: GET OUT OF THIS ROOM by moving to new map coordinates

CONCRETE ACTION PLAN:
1. If at (0,0) or (1,0): Move DOWN or RIGHT (if available)
2. If at (2,0): Move DOWN then RIGHT (if available)
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
