"""
enhanced_llm_agent.py - Local LLM Pokemon Agent with Computer Vision

Enhanced version of the local LLM agent that integrates computer vision
for better game state understanding and strategic decision making.
"""

import json
import sqlite3
import os
from typing import Dict, List, Any, Optional
from datetime import datetime
import ollama
import numpy as np

# Import our custom modules
from pyboy_env import PyBoyPokemonCrystalEnv
from vision_processor import PokemonVisionProcessor, VisualContext


class EnhancedLLMPokemonAgent:
    """
    Enhanced Pokemon Crystal AI agent with local LLM and computer vision
    """
    
    def __init__(self, 
                 model_name: str = "llama3.2:3b",
                 memory_db: str = None,
                 use_vision: bool = True):
        """
        Initialize the enhanced LLM Pokemon agent
        
        Args:
            model_name: Ollama model name to use
            memory_db: SQLite database for episodic memory
            use_vision: Whether to enable computer vision processing
        """
        self.model_name = model_name
        # Ensure outputs directory exists and set default database path
        os.makedirs("outputs", exist_ok=True)
        self.memory_db = memory_db or "outputs/pokemon_agent_memory.db"
        self.use_vision = use_vision
        
        # Initialize memory database
        self._init_memory_db()
        
        # Initialize vision processor if enabled
        if self.use_vision:
            try:
                self.vision_processor = PokemonVisionProcessor()
                print("👁️ Computer vision enabled")
            except Exception as e:
                print(f"⚠️ Vision processor failed to initialize: {e}")
                print("📄 Falling back to text-only mode")
                self.use_vision = False
                self.vision_processor = None
        else:
            self.vision_processor = None
        
        # Map discovery and state tracking
        self.discovered_maps = set()  # Track visited areas
        self.position_history = []    # Track recent positions to detect being stuck
        self.last_position = None
        self.stuck_counter = 0
        
        # Enhanced game knowledge and tracking
        self.gameplay_knowledge = {
            "escape_strategies": [
                "If stuck in dialogue/menu and A isn't working, try B to go back",
                "If stuck in same screen, try different movement directions",
                "If completely stuck, try START menu then B to exit",
                "B button usually cancels or goes back in most situations",
                "In battles, B can run away from wild Pokemon"
            ],
            "navigation_tips": [
                "A button interacts with objects, people, and confirms menu choices",
                "UP/DOWN navigate menus, LEFT/RIGHT can change values in some menus", 
                "START opens the main menu in overworld",
                "In dialogue, A continues text, B might skip or go back",
                "Movement in overworld: UP/DOWN/LEFT/RIGHT"
            ],
            "game_progression": {
                "intro": "Get starter Pokemon from Professor Elm, learn basic controls",
                "early_exploration": "Explore Route 29, catch wild Pokemon, visit Cherrygrove City",
                "gym_preparation": "Train Pokemon, heal at Pokemon Centers, challenge Violet City gym",
                "post_gym": "Continue story, explore new areas, catch stronger Pokemon"
            }
        }
        
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
                "Route 31": "Dark Cave entrance, Violet City approach",
                "Violet City": "First gym - Falkner (Flying type)",
                "Route 32": "Ruins of Alph, Union Cave",
                "Azalea Town": "Second gym - Bugsy (Bug type)"
            },
            "early_pokemon": {
                16: "Pidgey (Normal/Flying)",
                19: "Rattata (Normal)", 
                10: "Caterpie (Bug)",
                13: "Weedle (Bug)",
                129: "Magikarp (Water)",
                21: "Spearow (Normal/Flying)",
                161: "Sentret (Normal)"
            },
            "type_effectiveness": {
                "Fire": {"weak_to": ["Water", "Ground", "Rock"], "strong_against": ["Grass", "Bug", "Ice"]},
                "Water": {"weak_to": ["Electric", "Grass"], "strong_against": ["Fire", "Ground", "Rock"]},
                "Grass": {"weak_to": ["Fire", "Ice", "Poison", "Flying", "Bug"], "strong_against": ["Water", "Ground", "Rock"]},
                "Flying": {"weak_to": ["Electric", "Ice", "Rock"], "strong_against": ["Grass", "Fighting", "Bug"]},
                "Normal": {"weak_to": ["Fighting"], "strong_against": []},
                "Bug": {"weak_to": ["Fire", "Flying", "Rock"], "strong_against": ["Grass", "Psychic"]}
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
        
        print(f"🤖 Enhanced LLM Pokemon Agent initialized with {model_name}")
    
    def _init_memory_db(self):
        """Initialize SQLite database for episodic memory"""
        conn = sqlite3.connect(self.memory_db)
        cursor = conn.cursor()
        
        # Create basic tables first
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
        
        # Add new columns to existing tables if they don't exist
        self._add_column_if_not_exists(cursor, 'game_states', 'visual_summary', 'TEXT')
        self._add_column_if_not_exists(cursor, 'game_states', 'screen_type', 'TEXT')
        self._add_column_if_not_exists(cursor, 'strategic_decisions', 'visual_context', 'TEXT')
        self._add_column_if_not_exists(cursor, 'strategic_decisions', 'confidence_score', 'REAL')
        self._add_column_if_not_exists(cursor, 'pokemon_encounters', 'visual_detected', 'BOOLEAN')
        
        # New table for visual analysis results
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS visual_analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                screen_type TEXT,
                game_phase TEXT,
                detected_text TEXT,
                ui_elements TEXT,
                dominant_colors TEXT,
                visual_summary TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _add_column_if_not_exists(self, cursor, table_name: str, column_name: str, column_type: str):
        """Add a column to a table if it doesn't already exist"""
        try:
            cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}")
        except sqlite3.OperationalError as e:
            if "duplicate column name" not in str(e).lower():
                # Re-raise if it's not a duplicate column error
                raise
    
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
    
    def analyze_game_state(self, state: Dict[str, Any], 
                          visual_context: Optional[VisualContext] = None) -> Dict[str, Any]:
        """Analyze current game state and determine strategic context"""
        analysis = {
            "phase": "unknown",
            "immediate_goals": [],
            "threats": [],
            "opportunities": [],
            "next_actions": [],
            "visual_insights": [],
            "phase_strategy": "",
            "location_context": ""
        }
        
        if not state or 'player' not in state:
            return analysis
        
        player = state['player']
        party = state.get('party', [])
        current_map = player.get('map', 0)
        money = player.get('money', 0)
        badges = player.get('badges', 0)
        
        # Enhanced game phase determination with location awareness
        analysis["location_context"] = self._get_location_context(current_map)
        
        if len(party) == 0:
            analysis["phase"] = "intro/starter_selection"
            analysis["immediate_goals"] = ["Get starter Pokemon", "Leave Professor's lab"]
            analysis["phase_strategy"] = "Focus on dialogue and menu navigation. Use A to proceed through conversations."
            analysis["next_actions"] = ["A", "UP", "DOWN"]  # Dialogue and menu navigation
            
        elif len(party) == 1 and badges == 0:
            if current_map <= 2:  # Still in starting area
                analysis["phase"] = "tutorial/early_exploration"
                analysis["immediate_goals"] = ["Learn basic controls", "Explore starting area", "Talk to NPCs"]
                analysis["phase_strategy"] = "Explore systematically, talk to everyone, learn game mechanics."
                analysis["next_actions"] = ["UP", "DOWN", "LEFT", "RIGHT", "A"]
            else:
                analysis["phase"] = "early_game_exploration"
                analysis["immediate_goals"] = ["Catch wild Pokemon", "Train starter", "Find Pokemon Center"]
                analysis["phase_strategy"] = "Catch 2-3 Pokemon, train to level 10+, head to first gym."
                analysis["next_actions"] = ["UP", "DOWN", "LEFT", "RIGHT", "A"]
                
        elif len(party) > 1 and badges == 0:
            avg_level = sum(p.get('level', 1) for p in party) / len(party)
            if avg_level < 8:
                analysis["phase"] = "team_building"
                analysis["immediate_goals"] = ["Train Pokemon team", "Battle wild Pokemon", "Level up"]
                analysis["phase_strategy"] = "Focus on training. Battle wild Pokemon to gain experience."
            else:
                analysis["phase"] = "gym_preparation"
                analysis["immediate_goals"] = ["Head to Violet City", "Prepare for Flying-type gym", "Stock up on items"]
                analysis["phase_strategy"] = "Team is ready for first gym. Head to Violet City."
                
        elif badges == 1:
            analysis["phase"] = "post_first_gym"
            analysis["immediate_goals"] = ["Explore new areas", "Head to Azalea Town", "Catch stronger Pokemon"]
            analysis["phase_strategy"] = "First gym complete! Explore south to Union Cave and Azalea Town."
            
        elif badges >= 2:
            analysis["phase"] = "mid_game_progression"
            analysis["immediate_goals"] = [f"Continue with {badges} badges", "Next gym challenge", "Story progression"]
            analysis["phase_strategy"] = "Experienced trainer. Focus on story progression and gym challenges."
            
        # Money-based strategy adjustments
        if money < 500:
            analysis["threats"].append("Very low on money - battle trainers for cash")
            analysis["immediate_goals"].append("Fight trainers for money")
        elif money < 2000:
            analysis["threats"].append("Low money - be careful with purchases")
        elif money > 5000:
            analysis["opportunities"].append("Good money - can buy items/pokeballs freely")
        
        # Enhanced analysis with visual context
        if visual_context:
            # Screen-specific insights
            if visual_context.screen_type == 'battle':
                analysis["phase"] = "battle"
                analysis["immediate_goals"] = ["Win the battle", "Use effective moves"]
                if visual_context.ui_elements:
                    health_bars = [e for e in visual_context.ui_elements if e.element_type == 'healthbar']
                    if health_bars:
                        analysis["visual_insights"].append("Health bars detected - monitor HP carefully")
            
            elif visual_context.screen_type == 'dialogue':
                analysis["immediate_goals"] = ["Read dialogue", "Continue conversation"]
                if visual_context.detected_text:
                    important_keywords = ['gym', 'badge', 'pokemon', 'center', 'heal']
                    dialogue_text = ' '.join([t.text.lower() for t in visual_context.detected_text])
                    for keyword in important_keywords:
                        if keyword in dialogue_text:
                            analysis["visual_insights"].append(f"Dialogue mentions '{keyword}' - important info")
            
            elif visual_context.screen_type == 'menu':
                analysis["immediate_goals"] = ["Navigate menu", "Select appropriate option"]
                if visual_context.detected_text:
                    menu_options = [t.text for t in visual_context.detected_text if t.location == 'menu']
                    if menu_options:
                        analysis["visual_insights"].append(f"Menu options detected: {', '.join(menu_options[:3])}")
            
            elif visual_context.screen_type == 'overworld':
                analysis["immediate_goals"] = ["Explore area", "Look for interactions"]
                # Check dominant colors for location hints
                for r, g, b in visual_context.dominant_colors[:2]:
                    if g > 150 and r < 100 and b < 100:  # Green dominant
                        analysis["visual_insights"].append("Green area - likely outdoor/grass route")
                    elif b > 150 and r < 100 and g < 100:  # Blue dominant
                        analysis["visual_insights"].append("Blue area - possibly water or indoor")
        
        # Check party status
        if party:
            low_hp_pokemon = [p for p in party if p.get('hp', 0) < p.get('max_hp', 1) * 0.3]
            if low_hp_pokemon:
                analysis["threats"].append(f"{len(low_hp_pokemon)} Pokemon with low HP need healing")
            
            # Level analysis
            avg_level = sum(p.get('level', 1) for p in party) / len(party)
            if avg_level < 10:
                analysis["opportunities"].append("Low level team - focus on training")
            elif avg_level > 20:
                analysis["opportunities"].append("Strong team - ready for gym challenges")
        
        # Money situation
        money = player.get('money', 0)
        if money < 1000:
            analysis["threats"].append("Low on money")
        elif money > 10000:
            analysis["opportunities"].append("Good money for items/pokeballs")
        
        return analysis
    
    def _detect_stuck_state(self, state: Dict[str, Any], recent_history: List[str] = None) -> Dict[str, Any]:
        """Detect if agent is stuck and suggest escape strategies"""
        stuck_info = {
            "is_stuck": False,
            "stuck_type": None,
            "escape_strategy": None
        }
        
        if not state or 'player' not in state:
            return stuck_info
            
        player = state.get('player', {})
        current_pos = (player.get('x', 0), player.get('y', 0), player.get('map', 0))
        
        # Track position history
        self.position_history.append(current_pos)
        if len(self.position_history) > 10:  # Keep last 10 positions
            self.position_history.pop(0)
            
        # Check if stuck in same position
        if len(self.position_history) >= 5:
            if all(pos == current_pos for pos in self.position_history[-5:]):
                stuck_info["is_stuck"] = True
                stuck_info["stuck_type"] = "same_position"
                stuck_info["escape_strategy"] = "Try B to exit menus, or different movement directions"
        
        # Check for action repetition
        if recent_history and len(recent_history) >= 4:
            last_four = recent_history[-4:]
            if len(set(last_four)) <= 2:  # Only 1-2 unique actions
                stuck_info["is_stuck"] = True
                stuck_info["stuck_type"] = "action_loop"
                stuck_info["escape_strategy"] = "Break the pattern with B button or try opposite direction"
                
        return stuck_info
    
    def _get_location_context(self, current_map: int) -> str:
        """Get contextual information about current location"""
        location_map = {
            0: "New Bark Town - Starting town with Professor Elm's lab",
            1: "Route 29 - First route, catch early Pokemon like Pidgey and Sentret", 
            2: "Cherrygrove City - First real city, Pokemon Center for healing",
            3: "Route 30 - Trainer battles, Mr. Pokemon's house to the north",
            4: "Route 31 - Dark Cave entrance, path to Violet City",
            5: "Violet City - First gym town, Falkner specializes in Flying-types",
            6: "Route 32 - Ruins of Alph nearby, Union Cave entrance",
            7: "Azalea Town - Second gym town, Bugsy specializes in Bug-types",
            8: "Ilex Forest - Cut required, shrine in the center",
            9: "Route 34 - Daycare center, breeding location",
            10: "Goldenrod City - Big city, department store, radio tower"
        }
        
        return location_map.get(current_map, f"Unknown area (Map {current_map})")
    
    def _update_map_discovery(self, player_map: int):
        """Track discovered maps for exploration progress"""
        if player_map not in self.discovered_maps:
            self.discovered_maps.add(player_map)
            print(f"🗺️ Discovered new area: Map {player_map}")
    
    def decide_next_action(self, state: Dict[str, Any], 
                          screenshot: Optional[np.ndarray] = None,
                          recent_history: List[str] = None) -> int:
        """Use local LLM to decide next action based on game state and visual context"""
        
        # Update map discovery
        if state and 'player' in state:
            self._update_map_discovery(state['player'].get('map', 0))
        
        # Detect stuck state
        stuck_info = self._detect_stuck_state(state, recent_history)
        
        # Process visual context if screenshot provided
        visual_context = None
        if screenshot is not None and self.use_vision and self.vision_processor:
            try:
                visual_context = self.vision_processor.process_screenshot(screenshot)
                print(f"👁️ Visual: {visual_context.visual_summary}")
                
                # Store visual analysis
                self._store_visual_analysis(visual_context)
                
            except Exception as e:
                print(f"⚠️ Vision processing failed: {e}")
        
        # Analyze current situation
        analysis = self.analyze_game_state(state, visual_context)
        
        # Build context for LLM
        context_prompt = self._build_enhanced_strategy_prompt(
            state, analysis, visual_context, recent_history, stuck_info
        )
        
        system_prompt = """You are a Pokemon Crystal gameplay expert with visual analysis capabilities.
        
IMPORTANT RULES:
1. Be concise - give short, direct answers
2. Focus on immediate next action, not long-term strategy  
3. Consider both game state data AND visual information when available
4. VARY your actions - don't repeat the same action unless specifically needed
5. Your response should be ONLY ONE of these actions: UP, DOWN, LEFT, RIGHT, A, B, START, SELECT, NONE

Actions explained:
- Movement: UP/DOWN/LEFT/RIGHT to navigate and explore
- A: Interact with objects/people, confirm in menus, attack in battle, continue dialogue
- B: Cancel, go back, run from wild Pokemon  
- START: Open main menu
- SELECT: Usually unused
- NONE: Wait/do nothing (use sparingly)

Action Context Guidelines:
- DIALOGUE SCREENS: Use A to continue, but if you've pressed A multiple times, try B to back out or movement to explore
- MENU SCREENS: Use UP/DOWN to navigate options, A to select, B to cancel
- OVERWORLD: Use movement (UP/DOWN/LEFT/RIGHT) to explore, A to interact with objects/NPCs
- TITLE/INTRO: Use A to proceed through initial screens
- BATTLE: Use A for attacks, or UP/DOWN to navigate battle menus
- When stuck or repeating actions: Try different movement directions or B to escape

IMPORTANT: If recent actions show repetitive behavior, choose a DIFFERENT action to break the pattern!"""

        # Query local LLM
        llm_response = self._query_local_llm(context_prompt, system_prompt)
        
        # Parse response to extract action
        action = self._parse_action_from_response(llm_response)
        
        # Store decision for learning
        self._store_enhanced_decision(state, llm_response, action, analysis, visual_context)
        
        return action
    
    def _build_enhanced_strategy_prompt(self, state: Dict[str, Any], 
                                       analysis: Dict[str, Any], 
                                       visual_context: Optional[VisualContext] = None,
                                       recent_history: List[str] = None,
                                       stuck_info: Dict[str, Any] = None) -> str:
        """Build a comprehensive prompt including visual context"""
        
        player = state.get('player', {})
        party = state.get('party', [])
        
        prompt = f"""Pokemon Crystal - Enhanced Situation Analysis:

LOCATION: Map {player.get('map', 0)}, Position ({player.get('x', 0)}, {player.get('y', 0)})
MONEY: ${player.get('money', 0)}
BADGES: {player.get('badges', 0)}

TEAM: {len(party)} Pokemon"""
        
        if party:
            for i, pokemon in enumerate(party[:3]):  # Show first 3
                species = pokemon.get('species', 0)
                species_name = self.pokemon_context['starter_pokemon'].get(
                    species, self.pokemon_context['early_pokemon'].get(species, f"Pokemon #{species}")
                )
                hp_status = "CRITICAL" if pokemon.get('hp', 0) < pokemon.get('max_hp', 1) * 0.2 else "OK"
                prompt += f"\n  {i+1}. {species_name} - Level {pokemon.get('level', '?')} (HP: {pokemon.get('hp', 0)}/{pokemon.get('max_hp', 0)} - {hp_status})"
        
        # Add visual context analysis
        if visual_context:
            prompt += f"\n\nVISUAL ANALYSIS:"
            prompt += f"\n- Screen Type: {visual_context.screen_type.upper()}"
            prompt += f"\n- Game Phase: {visual_context.game_phase}"
            prompt += f"\n- Summary: {visual_context.visual_summary}"
            
            if visual_context.detected_text:
                prompt += f"\n- Text on Screen:"
                for text_obj in visual_context.detected_text[:3]:
                    prompt += f"\n  * '{text_obj.text}' ({text_obj.location})"
            
            if visual_context.ui_elements:
                ui_types = [elem.element_type for elem in visual_context.ui_elements]
                prompt += f"\n- UI Elements: {', '.join(set(ui_types))}"
        
        prompt += f"\n\nGAME PHASE: {analysis['phase']}"
        prompt += f"\nLOCATION CONTEXT: {analysis['location_context']}"
        prompt += f"\nPHASE STRATEGY: {analysis['phase_strategy']}"
        prompt += f"\nGOALS: {', '.join(analysis['immediate_goals'])}"
        
        if analysis['threats']:
            prompt += f"\nTHREATS: {', '.join(analysis['threats'])}"
        
        if analysis['opportunities']:
            prompt += f"\nOPPORTUNITIES: {', '.join(analysis['opportunities'])}"
            
        # Add contextual action recommendations
        if analysis['next_actions']:
            prompt += f"\nRECOMMENDED ACTIONS: {', '.join(analysis['next_actions'])}"
        
        if analysis['visual_insights']:
            prompt += f"\nVISUAL INSIGHTS: {', '.join(analysis['visual_insights'])}"
        
        if recent_history:
            prompt += f"\nRECENT ACTIONS: {', '.join(recent_history[-5:])}"
            
            # Detect repetitive behavior
            if len(recent_history) >= 3:
                last_actions = recent_history[-3:]
                if len(set(last_actions)) == 1:  # All same action
                    prompt += f"\n⚠️ WARNING: You've been repeating the same action ({last_actions[0]}) - try something different!"
                elif len(set(last_actions)) == 2 and len(recent_history) >= 4:
                    # Check for alternating pattern
                    last_four = recent_history[-4:]
                    if last_four[0] == last_four[2] and last_four[1] == last_four[3]:
                        prompt += f"\n⚠️ WARNING: You're stuck in a pattern ({', '.join(last_four)}) - break the cycle!"
        
        # Add stuck detection information
        if stuck_info and stuck_info.get('is_stuck'):
            prompt += f"\n🚨 STUCK DETECTED: {stuck_info['stuck_type']} - {stuck_info['escape_strategy']}"
        
        # Add exploration progress
        prompt += f"\n\nEXPLORATION: Discovered {len(self.discovered_maps)} areas"
        
        # Add gameplay knowledge hints
        prompt += "\n\nGAMEPLAY KNOWLEDGE:"
        prompt += "\n- If stuck in dialogue/menu: try B to go back"
        prompt += "\n- If repeating same actions: try different directions or B"
        prompt += "\n- A interacts/confirms, B cancels/backs out"
        prompt += "\n- START opens menu, movement keys navigate"
        
        # Add screen-specific guidance
        if visual_context:
            if visual_context.screen_type == 'battle':
                prompt += "\n\nBATTLE MODE: Choose attack moves wisely. Consider type effectiveness."
            elif visual_context.screen_type == 'dialogue':
                prompt += "\n\nDIALOGUE MODE: Continue reading important information."
            elif visual_context.screen_type == 'menu':
                prompt += "\n\nMENU MODE: Navigate to appropriate option."
            elif visual_context.screen_type == 'overworld':
                prompt += "\n\nOVERWORLD MODE: Explore systematically, interact with objects/NPCs."
        
        prompt += "\n\nWhat should I do next? Give me ONE action to take right now."
        
        return prompt
    
    def _parse_action_from_response(self, response: str) -> int:
        """Parse LLM response to extract action command with improved fallback logic"""
        response_upper = response.upper()
        
        # Enhanced action parsing with more flexible patterns
        if "UP" in response_upper or "NORTH" in response_upper:
            return 1
        elif "DOWN" in response_upper or "SOUTH" in response_upper:
            return 2
        elif "LEFT" in response_upper or "WEST" in response_upper:
            return 3
        elif "RIGHT" in response_upper or "EAST" in response_upper:
            return 4
        elif (" A " in response_upper or response_upper.endswith(" A") or 
              "PRESS A" in response_upper or "BUTTON A" in response_upper or
              "HIT A" in response_upper or "USE A" in response_upper):
            return 5
        elif (" B " in response_upper or response_upper.endswith(" B") or 
              "PRESS B" in response_upper or "BUTTON B" in response_upper or
              "HIT B" in response_upper or "USE B" in response_upper or
              "CANCEL" in response_upper or "BACK" in response_upper or "EXIT" in response_upper):
            return 6
        elif "START" in response_upper or "MENU" in response_upper:
            return 7
        elif "SELECT" in response_upper and "BUTTON" in response_upper:
            return 8
        
        # Context-aware action selection based on content
        elif ("INTERACT" in response_upper or "TALK" in response_upper or 
              "CONFIRM" in response_upper or "ENTER" in response_upper or
              "CONTINUE" in response_upper or "NEXT" in response_upper or "PROCEED" in response_upper):
            return 5
        elif "MOVE" in response_upper or "WALK" in response_upper or "GO" in response_upper:
            # If movement is mentioned but direction unclear, choose random movement
            return np.random.choice([1, 2, 3, 4])  # UP, DOWN, LEFT, RIGHT
        elif "EXPLORE" in response_upper or "LOOK" in response_upper:
            return np.random.choice([1, 2, 3, 4])  # Random exploration movement
        elif "WAIT" in response_upper or "PAUSE" in response_upper or "NOTHING" in response_upper:
            return 0  # NONE - only when explicitly requested
        else:
            # Last resort: more balanced fallback
            # Check if response suggests any direction even without exact keywords
            if any(word in response_upper for word in ["FORWARD", "AHEAD", "ADVANCE"]):
                return 1
            elif any(word in response_upper for word in ["BACKWARD", "RETREAT", "RETURN"]):
                return 2
            else:
                # Truly random fallback with more balanced distribution
                fallback_actions = [1, 2, 3, 4, 5, 0]  # Include NONE occasionally
                weights = [0.25, 0.25, 0.2, 0.2, 0.08, 0.02]  # Favor movement, some A, rare NONE
                return np.random.choice(fallback_actions, p=weights)
    
    def _store_enhanced_decision(self, state: Dict[str, Any], llm_response: str, 
                               action: int, analysis: Dict[str, Any],
                               visual_context: Optional[VisualContext] = None):
        """Store enhanced decision in memory for future learning"""
        conn = sqlite3.connect(self.memory_db)
        cursor = conn.cursor()
        
        # Store strategic decision with visual context
        cursor.execute('''
            INSERT INTO strategic_decisions (timestamp, situation, decision, reasoning, 
                                           outcome, visual_context, confidence_score)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            json.dumps(analysis),
            self.action_map[action],
            llm_response,
            "pending",  # Will be updated later based on results
            json.dumps({
                "screen_type": visual_context.screen_type if visual_context else None,
                "visual_summary": visual_context.visual_summary if visual_context else None,
                "detected_text_count": len(visual_context.detected_text) if visual_context else 0,
                "ui_elements_count": len(visual_context.ui_elements) if visual_context else 0
            }),
            0.8  # Default confidence
        ))
        
        # Store game state snapshot with visual info
        player = state.get('player', {})
        cursor.execute('''
            INSERT INTO game_states (timestamp, player_x, player_y, player_map, 
                                   party_size, money, badges, game_progress, notes,
                                   visual_summary, screen_type)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            player.get('x', 0),
            player.get('y', 0),
            player.get('map', 0),
            len(state.get('party', [])),
            player.get('money', 0),
            player.get('badges', 0),
            analysis['phase'],
            f"Action decided: {self.action_map[action]}",
            visual_context.visual_summary if visual_context else None,
            visual_context.screen_type if visual_context else None
        ))
        
        conn.commit()
        conn.close()
    
    def _store_visual_analysis(self, visual_context: VisualContext):
        """Store visual analysis results for pattern learning"""
        conn = sqlite3.connect(self.memory_db)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO visual_analysis (timestamp, screen_type, game_phase, detected_text,
                                       ui_elements, dominant_colors, visual_summary)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            visual_context.screen_type,
            visual_context.game_phase,
            json.dumps([{"text": t.text, "location": t.location, "confidence": t.confidence} 
                       for t in visual_context.detected_text]),
            json.dumps([{"type": e.element_type, "confidence": e.confidence} 
                       for e in visual_context.ui_elements]),
            json.dumps(visual_context.dominant_colors),
            visual_context.visual_summary
        ))
        
        conn.commit()
        conn.close()
    
    def get_enhanced_memory_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of stored memories and visual data"""
        conn = sqlite3.connect(self.memory_db)
        cursor = conn.cursor()
        
        # Count different types of memories
        cursor.execute("SELECT COUNT(*) FROM strategic_decisions")
        decision_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM game_states")
        state_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM pokemon_encounters")
        encounter_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM visual_analysis")
        visual_count = cursor.fetchone()[0]
        
        # Get recent progress
        cursor.execute("""
            SELECT badges, money, party_size, visual_summary, screen_type
            FROM game_states 
            ORDER BY timestamp DESC 
            LIMIT 1
        """)
        latest = cursor.fetchone()
        
        # Get visual analysis breakdown
        cursor.execute("""
            SELECT screen_type, COUNT(*) 
            FROM visual_analysis 
            GROUP BY screen_type
        """)
        screen_type_breakdown = dict(cursor.fetchall())
        
        conn.close()
        
        return {
            "decisions_stored": decision_count,
            "states_recorded": state_count,
            "pokemon_encounters": encounter_count,
            "visual_analyses": visual_count,
            "screen_type_breakdown": screen_type_breakdown,
            "latest_progress": {
                "badges": latest[0] if latest else 0,
                "money": latest[1] if latest else 0,
                "party_size": latest[2] if latest else 0,
                "last_visual_summary": latest[3] if latest and latest[3] else "No visual data",
                "last_screen_type": latest[4] if latest and latest[4] else "Unknown"
            } if latest else None
        }


def test_enhanced_llm_agent():
    """Test the enhanced LLM agent with vision capabilities"""
    print("🧪 Testing Enhanced LLM Pokemon Agent...")
    
    agent = EnhancedLLMPokemonAgent()
    
    # Mock game state - battle scenario
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
                "level": 12,
                "hp": 28,
                "max_hp": 35,
                "status": 0
            }
        ]
    }
    
    # Create mock screenshot for testing
    mock_screenshot = np.zeros((144, 160, 3), dtype=np.uint8)
    mock_screenshot.fill(255)  # White background
    mock_screenshot[10:30, 120:150] = [255, 0, 0]    # Red health bar (low HP)
    mock_screenshot[100:140, 10:150] = [200, 200, 255]  # Light blue dialogue area
    
    print("📊 Mock game state:", json.dumps(mock_state, indent=2))
    
    # Get enhanced strategic analysis with visual context
    print("\n🔍 Testing with visual context...")
    action = agent.decide_next_action(mock_state, mock_screenshot, ["UP", "A", "DOWN"])
    print(f"🎯 Decided action with vision: {action} ({agent.action_map[action]})")
    
    # Test without visual context
    print("\n🔍 Testing without visual context...")
    action_no_vision = agent.decide_next_action(mock_state, None, ["UP", "A", "DOWN"])
    print(f"🎯 Decided action without vision: {action_no_vision} ({agent.action_map[action_no_vision]})")
    
    # Show enhanced memory summary
    memory = agent.get_enhanced_memory_summary()
    print("\n💭 Enhanced memory summary:", json.dumps(memory, indent=2))
    
    print("\n✅ Enhanced LLM agent test completed!")


if __name__ == "__main__":
    test_enhanced_llm_agent()
