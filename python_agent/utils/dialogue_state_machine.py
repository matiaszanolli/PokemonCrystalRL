"""
dialogue_state_machine.py - Dialogue State Management for Pokemon Crystal Agent

This module tracks conversation states, dialogue choices, and provides
intelligent dialogue navigation based on current objectives and context.
"""

import json
import sqlite3
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Set, Any
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import hashlib

try:
    from ..vision.vision_processor import DetectedText, VisualContext
except ImportError:
    from vision.vision_processor import DetectedText, VisualContext

# Import semantic context system
try:
    from .semantic_context_system import SemanticContextSystem, GameContext, DialogueIntent
    SEMANTIC_CONTEXT_AVAILABLE = True
except ImportError:
    try:
        from semantic_context_system import SemanticContextSystem, GameContext, DialogueIntent
        SEMANTIC_CONTEXT_AVAILABLE = True
    except ImportError:
        # Create stub classes if semantic context system is not available
        SEMANTIC_CONTEXT_AVAILABLE = False
        
        class GameContext:
            def __init__(self, current_objective, player_progress, location_info, recent_events, active_quests):
                self.current_objective = current_objective
                self.player_progress = player_progress
                self.location_info = location_info
                self.recent_events = recent_events
                self.active_quests = active_quests
        
        class DialogueIntent:
            pass
        
        class SemanticContextSystem:
            def __init__(self, *args, **kwargs):
                pass
        
        print("⚠️ Semantic context system not available - using stubs")


class DialogueState(Enum):
    """Possible dialogue states"""
    IDLE = "idle"                    # No dialogue active
    READING = "reading"              # Reading dialogue text
    CHOOSING = "choosing"            # Multiple choice selection
    WAITING_RESPONSE = "waiting"     # Waiting for dialogue to advance
    LISTENING = "listening"          # Listening to NPC (alias for reading)
    RESPONDING = "responding"        # Formulating response
    CONVERSATION_END = "ended"       # Dialogue finished
    INTERRUPTED = "interrupted"      # Dialogue interrupted by battle/event


class NPCType(Enum):
    """Types of NPCs for different conversation handling"""
    PROFESSOR = "professor"          # Professor Elm, Oak, etc.
    FAMILY = "family"               # Mom, relatives
    TRAINER = "trainer"             # Opposing trainers
    GYM_LEADER = "gym_leader"       # Gym leaders
    SHOPKEEPER = "shopkeeper"       # Store clerks
    NURSE = "nurse"                 # Pokemon Center nurse
    GENERIC = "generic"             # Regular NPCs
    STORY = "story"                 # Important story characters


@dataclass
class DialogueChoice:
    """Represents a dialogue choice option"""
    text: str
    choice_id: str
    confidence: float
    expected_outcome: str           # What we expect this choice to do
    priority: int                   # Higher priority = preferred choice


@dataclass
class ConversationContext:
    """Context information for a conversation"""
    npc_type: NPCType
    npc_name: Optional[str]
    location_map: int
    conversation_topic: Optional[str]
    current_objective: Optional[str]
    conversation_history: List[str]
    choices_made: List[str]
    dialogue_length: int


@dataclass
class DialogueMemory:
    """Memory of past dialogue interactions"""
    npc_identifier: str             # Location + NPC type combo
    last_conversation: str
    topics_discussed: Set[str]
    choices_made: Dict[str, str]    # choice_id -> selected option
    conversation_outcomes: List[str]
    last_interaction_time: str
    interaction_count: int


class DialogueStateMachine:
    """
    Manages dialogue states and provides intelligent dialogue navigation
    """
    
    def __init__(self, db_path: str = "dialogue_state.db", semantic_system=None):
        """Initialize the dialogue state machine"""
        self.db_path = Path(db_path)
        self.current_state = DialogueState.IDLE
        self.current_context: Optional[ConversationContext] = None
        self.dialogue_history: List[str] = []
        self.choice_history: List[DialogueChoice] = []
        
        # NPC memory system
        self.npc_memories: Dict[str, DialogueMemory] = {}
        
        # Dialogue patterns and keywords
        self._load_dialogue_patterns()
        
        # Initialize semantic context system
        if semantic_system is not None:
            self.semantic_system = semantic_system
            print("💭 Semantic context system provided")
        elif SEMANTIC_CONTEXT_AVAILABLE:
            self.semantic_system = SemanticContextSystem(db_path=db_path)
            print("💭 Semantic context system initialized")
        else:
            self.semantic_system = None
        
        # Initialize database
        self._init_database()
        
        print("💬 Dialogue state machine initialized")
    
    def _init_database(self):
        """Initialize SQLite database for dialogue state tracking"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Dialogue sessions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS dialogue_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_start TEXT NOT NULL,
                    session_end TEXT,
                    npc_type TEXT NOT NULL,
                    npc_name TEXT,
                    location_map INTEGER NOT NULL,
                    conversation_topic TEXT,
                    objective TEXT,
                    total_exchanges INTEGER DEFAULT 0,
                    choices_made INTEGER DEFAULT 0,
                    outcome TEXT
                )
            """)
            
            # Create conversations table as an alias/view for test compatibility
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_start TEXT NOT NULL,
                    session_end TEXT,
                    start_time TEXT NOT NULL DEFAULT (datetime('now')),
                    end_time TEXT,
                    npc_type TEXT NOT NULL,
                    npc_name TEXT,
                    location_map INTEGER NOT NULL,
                    conversation_topic TEXT,
                    objective TEXT,
                    total_exchanges INTEGER DEFAULT 0,
                    total_turns INTEGER DEFAULT 0,
                    choices_made INTEGER DEFAULT 0,
                    outcome TEXT
                )
            """)
            
            # Dialogue choices table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS dialogue_choices (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER,
                    timestamp TEXT NOT NULL,
                    choice_text TEXT NOT NULL,
                    choice_id TEXT,
                    selected BOOLEAN DEFAULT FALSE,
                    expected_outcome TEXT,
                    actual_outcome TEXT,
                    confidence REAL,
                    FOREIGN KEY (session_id) REFERENCES dialogue_sessions (id)
                )
            """)
            
            # NPC interactions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS npc_interactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    npc_identifier TEXT NOT NULL,
                    location_map INTEGER NOT NULL,
                    npc_type TEXT NOT NULL,
                    npc_name TEXT,
                    last_conversation TEXT,
                    topics_discussed TEXT,
                    interaction_count INTEGER DEFAULT 1,
                    last_interaction TEXT NOT NULL,
                    conversation_success BOOLEAN,
                    UNIQUE(npc_identifier)
                )
            """)
            
            conn.commit()
    
    def _load_dialogue_patterns(self):
        """Load common dialogue patterns and responses"""
        self.dialogue_patterns = {
            # Starter Pokemon selection
            "starter_selection": {
                "keywords": ["starter", "pokemon", "choose", "pick", "cyndaquil", "totodile", "chikorita"],
                "choices": {
                    "cyndaquil": {"priority": 3, "outcome": "fire_starter"},
                    "totodile": {"priority": 2, "outcome": "water_starter"},
                    "chikorita": {"priority": 1, "outcome": "grass_starter"}
                }
            },
            
            # Professor conversations
            "professor_elm": {
                "keywords": ["professor", "elm", "research", "egg", "mr. pokemon"],
                "responses": {
                    "yes": {"priority": 3, "outcome": "accept_task"},
                    "okay": {"priority": 3, "outcome": "accept_task"},
                    "no": {"priority": 1, "outcome": "decline_task"}
                }
            },
            
            # Mom conversations
            "mom_conversations": {
                "keywords": ["mom", "mother", "home", "money", "save"],
                "responses": {
                    "yes": {"priority": 2, "outcome": "accept_help"},
                    "okay": {"priority": 2, "outcome": "accept_help"},
                    "no": {"priority": 1, "outcome": "polite_decline"}
                }
            },
            
            # Gym battles
            "gym_challenge": {
                "keywords": ["gym", "leader", "badge", "challenge", "battle"],
                "responses": {
                    "yes": {"priority": 5, "outcome": "accept_battle"},
                    "challenge": {"priority": 5, "outcome": "accept_battle"},
                    "no": {"priority": 1, "outcome": "decline_battle"}
                }
            },
            
            # Pokemon Center
            "pokemon_center": {
                "keywords": ["heal", "pokemon center", "nurse", "restore", "rest"],
                "responses": {
                    "yes": {"priority": 4, "outcome": "heal_pokemon"},
                    "please": {"priority": 4, "outcome": "heal_pokemon"},
                    "no": {"priority": 1, "outcome": "skip_healing"}
                }
            }
        }
        
        # Common dialogue navigation phrases
        self.navigation_phrases = {
            "continue": ["okay", "yes", "i see", "continue", "next"],
            "accept": ["yes", "okay", "sure", "accept", "i will"],
            "decline": ["no", "maybe later", "not now", "decline"],
            "ask_more": ["tell me more", "explain", "what", "how"]
        }
    
    def update_state(self, visual_context: VisualContext, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update dialogue state based on visual context and game state
        
        Args:
            visual_context: Current visual context
            game_state: Current game state information
            
        Returns:
            Dictionary with dialogue analysis results
        """
        if visual_context is None:
            return {"state": self.current_state.value, "error": "No visual context provided"}
            
        previous_state = self.current_state
        result = {
            "state": self.current_state.value,
            "previous_state": previous_state.value,
            "choices": [],
            "recommended_action": None,
            "semantic_analysis": None
        }
        
        # Determine current state based on visual context
        if visual_context.screen_type == 'dialogue':
            dialogue_result = self._process_dialogue_screen(visual_context, game_state)
            if dialogue_result:
                result.update(dialogue_result)
        elif visual_context.screen_type == 'menu':
            menu_result = self._process_menu_screen(visual_context, game_state)
            if menu_result:
                result.update(menu_result)
        else:
            # Not in dialogue
            if self.current_state != DialogueState.IDLE:
                self._end_conversation("left_dialogue")
            self.current_state = DialogueState.IDLE
        
        # Log state changes
        if previous_state != self.current_state:
            print(f"💬 Dialogue state: {previous_state.value} → {self.current_state.value}")
        
        result["state"] = self.current_state.value
        return result
    
    def _process_dialogue_screen(self, visual_context: VisualContext, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """Process dialogue screen and update state accordingly"""
        # Filter out any None or invalid text values
        detected_text = [t.text for t in visual_context.detected_text 
                        if t.location == 'dialogue' and t.text is not None and isinstance(t.text, str)]
        
        if not detected_text:
            return {}
        
        current_text = ' '.join(detected_text).lower()
        result = {}
        
        # Check if there are explicit choices in the visual context
        choice_texts = [t for t in visual_context.detected_text if t.location == 'choice']
        
        # Determine if we're in a choice selection screen
        has_choices = len(choice_texts) >= 1 or self._detect_choices(current_text)
        
        if has_choices:
            # Start conversation if not already started
            if self.current_state == DialogueState.IDLE:
                self._start_conversation(visual_context, game_state)
            
            # Process each individual dialogue text to build history properly
            for individual_text in detected_text:
                self._process_dialogue_text(individual_text.lower(), visual_context, game_state)
                
            self.current_state = DialogueState.CHOOSING
            choice_result = self._process_dialogue_choices(current_text, visual_context, game_state)
            if choice_result:
                result.update(choice_result)
            
            # Semantic analysis is already included in choice_result
        else:
            # Regular dialogue reading
            if self.current_state == DialogueState.IDLE:
                self._start_conversation(visual_context, game_state)
            
            self.current_state = DialogueState.READING
            # Process each individual dialogue text to build history properly
            for individual_text in detected_text:
                self._process_dialogue_text(individual_text.lower(), visual_context, game_state)
            
            # Get semantic analysis for the dialogue
            semantic_result = self.get_semantic_analysis(current_text, game_state)
            if semantic_result:
                result["semantic_analysis"] = semantic_result
        
        # Always include recommended action
        result["recommended_action"] = self.get_recommended_action()
        
        return result
    
    def _process_menu_screen(self, visual_context: VisualContext, game_state: Dict[str, Any]):
        """Process menu screen that might be part of dialogue"""
        # Some dialogue choices appear as menu screens
        menu_text = [t.text for t in visual_context.detected_text if t.location == 'menu']
        
        if menu_text and self.current_state in [DialogueState.READING, DialogueState.CHOOSING]:
            menu_content = ' '.join(menu_text).lower()
            if self._detect_choices(menu_content):
                self.current_state = DialogueState.CHOOSING
                self._process_dialogue_choices(menu_content, visual_context, game_state)
    
    def _detect_choices(self, text: str) -> bool:
        """Detect if text contains dialogue choices"""
        choice_indicators = [
            "yes", "no", "okay", "cancel",
            "accept", "decline", "sure", "maybe",
            "▶", "►", "→", "*",  # Choice markers
            "1)", "2)", "a)", "b)"  # Numbered/lettered choices
        ]
        
        # Check if multiple choice indicators are present
        choice_count = sum(1 for indicator in choice_indicators if indicator in text)
        return choice_count >= 2
    
    def _start_conversation(self, visual_context: VisualContext, game_state: Dict[str, Any]):
        """Start a new conversation"""
        player = game_state.get('player', {})
        current_map = player.get('map', 0)
        
        # Determine NPC type and context
        npc_type = self._identify_npc_type(visual_context, game_state)
        npc_name = self._identify_npc_name(visual_context, game_state)
        objective = self._get_current_objective(game_state)
        
        self.current_context = ConversationContext(
            npc_type=npc_type,
            npc_name=npc_name,
            location_map=current_map,
            conversation_topic=None,
            current_objective=objective,
            conversation_history=[],
            choices_made=[],
            dialogue_length=0
        )
        
        # Start tracking in database
        self._start_database_session()
        
        # Immediately update NPC memory for tracking
        self._update_npc_memory()
        
        print(f"💬 Started conversation with {npc_type.value} at map {current_map}")
    
    def _process_dialogue_text(self, text: str, visual_context: VisualContext, game_state: Dict[str, Any]):
        """Process regular dialogue text"""
        if not self.current_context:
            return
        
        # Add to conversation history
        self.current_context.conversation_history.append(text)
        self.current_context.dialogue_length += len(text)
        
        # Use semantic analysis to enhance topic identification
        if not self.current_context.conversation_topic:
            topic = self._identify_topic_with_semantics(text, game_state)
            self.current_context.conversation_topic = topic
        
        # Update dialogue history
        self.dialogue_history.append(text)
        if len(self.dialogue_history) > 10:  # Keep last 10 exchanges
            self.dialogue_history.pop(0)
    
    def _process_dialogue_choices(self, text: str, visual_context: VisualContext, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """Process dialogue choices and determine best option"""
        if not self.current_context:
            return {}
        
        # Get semantic analysis once for use in choice enhancement
        semantic_analysis = self.get_semantic_analysis(text, game_state)
        
        # Extract choices from text and pass semantic analysis to avoid double calls
        choices = self._extract_choices(text, visual_context, semantic_analysis)
        self.choice_history = choices
        
        # Store choices in database
        self._store_choices(choices)
        
        print(f"💬 Found {len(choices)} dialogue choices")
        for i, choice in enumerate(choices):
            print(f"  {i+1}. {choice.text} (priority: {choice.priority})")
        
        # Return choices in the format expected by tests
        choice_list = [{"text": choice.text, "priority": choice.priority, "outcome": choice.expected_outcome} for choice in choices]
        return {"choices": choice_list, "semantic_analysis": semantic_analysis}
    
    def _extract_choices(self, text: str, visual_context: VisualContext, semantic_analysis: Optional[Dict[str, Any]] = None) -> List[DialogueChoice]:
        """Extract dialogue choices from text and visual context"""
        choices = []
        
        # Common dialogue choices
        common_choices = {
            "yes": {"priority": 3, "outcome": "positive_response"},
            "no": {"priority": 1, "outcome": "negative_response"},
            "okay": {"priority": 3, "outcome": "acceptance"},
            "sure": {"priority": 3, "outcome": "agreement"},
            "maybe": {"priority": 2, "outcome": "uncertain_response"},
            "cancel": {"priority": 1, "outcome": "cancel_action"}
        }
        
        # Battle menu choices - critical for battle recognition
        battle_choices = {
            "fight": {"priority": 4, "outcome": "battle_attack"},
            "pack": {"priority": 2, "outcome": "use_item"},
            "pokemon": {"priority": 2, "outcome": "switch_pokemon"},
            "run": {"priority": 1, "outcome": "escape_battle"}
        }
        
        # Gym battle specific choices
        gym_choices = {
            "bring it on": {"priority": 5, "outcome": "accept_battle"},
            "bring": {"priority": 4, "outcome": "accept_battle"},
            "challenge": {"priority": 5, "outcome": "accept_battle"},
            "battle": {"priority": 4, "outcome": "accept_battle"},
            "let's go": {"priority": 4, "outcome": "accept_battle"}
        }
        
        # Starter selection choices
        starter_choices = {
            "cyndaquil": {"priority": 3, "outcome": "fire_starter"},
            "totodile": {"priority": 2, "outcome": "water_starter"},
            "chikorita": {"priority": 1, "outcome": "grass_starter"}
        }
        
        # Combine all choice sets
        all_choices = {**common_choices, **battle_choices, **gym_choices, **starter_choices}
        
        # First, extract choices from explicit 'choice' text elements in visual context
        choice_texts = [t for t in visual_context.detected_text if t.location == 'choice']
        for choice_obj in choice_texts:
            choice_text = choice_obj.text.lower()
            # Find matching choice data if available
            choice_data = None
            for pattern, data in all_choices.items():
                if pattern in choice_text:
                    choice_data = data
                    break
            
            # If no specific match found, use generic values
            if not choice_data:
                choice_data = {"priority": 2, "outcome": "generic_choice"}
            
            choices.append(DialogueChoice(
                text=choice_obj.text,  # Keep original case
                choice_id=hashlib.md5(choice_obj.text.encode()).hexdigest()[:8],
                confidence=choice_obj.confidence,
                expected_outcome=choice_data["outcome"],
                priority=choice_data["priority"]
            ))
        
        # Then also look for choices in the dialogue text itself
        for choice_text, choice_data in all_choices.items():
            if choice_text in text.lower():
                # Check if this choice is already added from visual context
                if not any(choice.text.lower() == choice_text for choice in choices):
                    choices.append(DialogueChoice(
                        text=choice_text,
                        choice_id=hashlib.md5(choice_text.encode()).hexdigest()[:8],
                        confidence=0.8,
                        expected_outcome=choice_data["outcome"],
                        priority=choice_data["priority"]
                    ))
        
        # Enhanced choice extraction based on context
        if self.current_context:
            choices = self._enhance_choices_with_context(choices, semantic_analysis)
        
        return sorted(choices, key=lambda x: x.priority, reverse=True)
    
    def _enhance_choices_with_context(self, choices: List[DialogueChoice], semantic_analysis: Optional[Dict[str, Any]] = None) -> List[DialogueChoice]:
        """Enhance choices based on current context and objectives"""
        if not self.current_context:
            return choices
        
        # Use semantic analysis to enhance choice priorities
        if self.semantic_system:
            choices = self._enhance_choices_with_semantics(choices, semantic_analysis)
        
        # Adjust priorities based on NPC type and objective
        for choice in choices:
            # Boost priority for objective-relevant choices
            if self.current_context.current_objective:
                if "starter" in self.current_context.current_objective.lower():
                    if any(starter in choice.text.lower() for starter in ["cyndaquil", "fire"]):
                        choice.priority += 2  # Prefer fire starter
                elif "gym" in self.current_context.current_objective.lower():
                    if "yes" in choice.text.lower() or "challenge" in choice.text.lower():
                        choice.priority += 3  # Accept gym challenges
                elif "heal" in self.current_context.current_objective.lower():
                    if "yes" in choice.text.lower():
                        choice.priority += 2  # Accept healing
            
            # Adjust based on NPC type
            if self.current_context.npc_type == NPCType.PROFESSOR:
                if "yes" in choice.text.lower() or "okay" in choice.text.lower():
                    choice.priority += 1  # Be cooperative with professors
            elif self.current_context.npc_type == NPCType.GYM_LEADER:
                if "yes" in choice.text.lower() or "challenge" in choice.text.lower():
                    choice.priority += 2  # Accept gym challenges
        
        return choices
    
    def _enhance_choices_with_semantics(self, choices: List[DialogueChoice], semantic_analysis: Optional[Dict[str, Any]] = None) -> List[DialogueChoice]:
        """Use semantic analysis to enhance dialogue choice priorities"""
        if not self.semantic_system or not self.current_context:
            return choices
        
        try:
            # Use provided semantic analysis if available, otherwise get strategy
            strategy = None
            if semantic_analysis and "strategy" in semantic_analysis:
                # Create a mock strategy object from analysis
                class MockStrategy:
                    def __init__(self, suggested_action, confidence=0.8):
                        self.suggested_action = suggested_action
                        self.confidence = confidence
                        self.reasoning = "From semantic analysis"
                
                strategy = MockStrategy(semantic_analysis["strategy"])
            else:
                # Fall back to getting strategy from semantic system
                context = GameContext(
                    current_objective=self.current_context.current_objective or "explore",
                    player_progress={
                        "level": 1,  # Default since not in game state
                        "badges": 0, # Default assumption
                        "party_size": 1    # Default assumption
                    },
                    location_info={
                        "map_id": self.current_context.location_map,
                        "location_type": self._determine_location_type(self.current_context.location_map)
                    },
                    recent_events=[],
                    active_quests=[]
                )
                
                # Get conversation context for semantic analysis
                dialogue_text = ' '.join(self.current_context.conversation_history[-3:])  # Last 3 exchanges
                
                # Get response strategy from semantic system
                strategy = self.semantic_system.suggest_response_strategy(
                    dialogue_text, context, self.current_context.npc_type.value
                )
            
            if strategy:
                print(f"💭 Semantic strategy: {strategy.suggested_action} - {strategy.reasoning}")
                
                # Adjust choice priorities based on semantic strategy
                for choice in choices:
                    # If semantic system suggests accepting/agreeing, boost positive choices
                    if "accept" in strategy.suggested_action.lower() or "agree" in strategy.suggested_action.lower():
                        if "yes" in choice.text.lower() or "okay" in choice.text.lower() or "sure" in choice.text.lower():
                            choice.priority += int(strategy.confidence * 3)  # Boost based on confidence
                    
                    # If semantic system suggests declining, boost negative choices
                    elif "decline" in strategy.suggested_action.lower() or "refuse" in strategy.suggested_action.lower():
                        if "no" in choice.text.lower() or "cancel" in choice.text.lower():
                            choice.priority += int(strategy.confidence * 3)
                    
                    # If semantic system suggests specific action, try to match it
                    elif strategy.suggested_action.lower() in choice.text.lower():
                        choice.priority += int(strategy.confidence * 4)  # Strong boost for exact matches
                
        except Exception as e:
            print(f"⚠️ Semantic choice enhancement failed: {e}")
        
        return choices
    
    def get_recommended_action(self) -> str:
        """Get recommended action based on current dialogue state"""
        if self.current_state == DialogueState.IDLE:
            return "A"  # Try to interact
        elif self.current_state == DialogueState.READING:
            return "A"  # Continue reading
        elif self.current_state == DialogueState.CHOOSING:
            return self._get_best_choice_action()
        elif self.current_state == DialogueState.WAITING_RESPONSE:
            return "A"  # Advance dialogue
        elif self.current_state == DialogueState.CONVERSATION_END:
            return "B"  # Exit dialogue
        else:
            return "A"  # Default action
    
    def _get_best_choice_action(self) -> str:
        """Get the action for the best dialogue choice"""
        if not self.choice_history:
            return "A"  # Default to A if no choices detected
        
        # Return action for highest priority choice
        best_choice = self.choice_history[0]  # Already sorted by priority
        
        # Map choice to game action (simplified)
        if "yes" in best_choice.text.lower():
            return "A"  # Usually A selects "Yes"
        elif "no" in best_choice.text.lower():
            return "B"  # Usually B selects "No" or goes back
        else:
            return "A"  # Default selection
    
    def _identify_npc_type(self, visual_context: VisualContext, game_state: Dict[str, Any]) -> NPCType:
        """Identify the type of NPC based on context"""
        # Filter out None text and safely handle detected text
        valid_texts = []
        for t in visual_context.detected_text:
            if t.text is not None and isinstance(t.text, str):
                valid_texts.append(t.text.lower())
        
        all_text = ' '.join(valid_texts)
        
        # Check for professors first (highest priority for research context)
        professor_keywords = ["professor", "elm", "oak", "research", "laboratory", "lab"]
        if any(word in all_text for word in professor_keywords):
            return NPCType.PROFESSOR
        
        # More comprehensive family detection
        family_keywords = ["mom", "mother", "dear", "honey", "son", "daughter", "child", "sweetie", "sweetheart"]
        if any(word in all_text for word in family_keywords):
            return NPCType.FAMILY
        elif any(word in all_text for word in ["gym", "leader", "badge"]):
            return NPCType.GYM_LEADER
        elif any(word in all_text for word in ["shop", "mart", "buy", "sell"]):
            return NPCType.SHOPKEEPER
        elif any(word in all_text for word in ["trainer", "battle", "challenge"]):
            return NPCType.TRAINER
        # Check for Pokemon Center nurse (more specific keywords to avoid false positives)
        elif any(phrase in all_text for phrase in ["pokemon center", "welcome to the pokemon center", "heal your pokemon", "nurse joy"]) or \
             ("heal" in all_text and ("pokemon" in all_text or "tired" in all_text or "rest" in all_text)):
            return NPCType.NURSE
        else:
            return NPCType.GENERIC
    
    def _identify_npc_name(self, visual_context: VisualContext, game_state: Dict[str, Any]) -> Optional[str]:
        """Try to identify specific NPC name"""
        # Filter out None text and safely handle detected text
        valid_texts = []
        for t in visual_context.detected_text:
            if t.text is not None and isinstance(t.text, str):
                valid_texts.append(t.text.lower())
        
        all_text = ' '.join(valid_texts)
        
        npc_names = ["elm", "oak", "mom", "falkner", "bugsy", "whitney", "morty"]
        
        for name in npc_names:
            if name in all_text:
                return name.title()
        
        return None
    
    def _identify_topic(self, text: str) -> Optional[str]:
        """Identify conversation topic from text"""
        text_lower = text.lower()
        
        topics = {
            "starter_pokemon": ["starter", "pokemon", "choose", "pick", "cyndaquil", "totodile", "chikorita"],
            "gym_challenge": ["gym", "badge", "challenge", "battle", "leader"],
            "healing": ["heal", "pokemon center", "rest", "restore"],
            "story_mission": ["mr. pokemon", "egg", "research", "errand"],
            "shopping": ["buy", "sell", "item", "potion", "pokeball"]
        }
        
        for topic, keywords in topics.items():
            if any(keyword in text_lower for keyword in keywords):
                return topic
        
        return None
    
    def _identify_topic_with_semantics(self, text: str, game_state: Dict[str, Any]) -> Optional[str]:
        """Identify conversation topic using semantic analysis"""
        # First try basic topic identification
        basic_topic = self._identify_topic(text)
        
        # Skip semantic analysis here to avoid double calls
        # The semantic analysis is done separately in get_semantic_analysis
        return basic_topic
    
    def _determine_location_type(self, map_id: int) -> str:
        """Determine location type based on map ID"""
        # Simplified location type determination
        location_mapping = {
            0: "home",
            1: "lab",
            2: "route",
            3: "city",
            4: "pokemon_center",
            5: "gym",
            6: "shop"
        }
        return location_mapping.get(map_id % 7, "unknown")
    
    def get_semantic_analysis(self, text: str, game_state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get detailed semantic analysis of dialogue text"""
        if not self.semantic_system:
            return None
        
        try:
            # Use the _build_game_context method so it can be mocked in tests
            context = self._build_game_context(game_state)
            
            # Get intent analysis  
            analysis = self.semantic_system.analyze_dialogue(text, context)
            
            return {
                "intent": analysis.get('primary_intent') if analysis else None,
                "primary_intent": analysis.get('primary_intent') if analysis else None,  # Duplicate for compatibility
                "confidence": analysis.get('confidence', 0.0) if analysis else 0.0,
                "strategy": analysis.get('response_strategy') if analysis else None,
                "response_strategy": analysis.get('response_strategy') if analysis else None,  # Duplicate for compatibility
                "reasoning": "Based on dialogue pattern analysis" if analysis else None
            }
            
        except Exception as e:
            print(f"⚠️ Semantic analysis failed: {e}")
            return None
    
    def _get_current_objective(self, game_state: Dict[str, Any]) -> Optional[str]:
        """Determine current game objective"""
        player = game_state.get('player', {})
        party = game_state.get('party', [])
        badges = player.get('badges', 0)
        
        if len(party) == 0:
            return "get_starter_pokemon"
        elif badges == 0 and len(party) == 1:
            return "first_gym_challenge"
        elif player.get('money', 0) < 500:
            return "earn_money"
        else:
            return f"gym_challenge_{badges + 1}"
    
    def _end_conversation(self, reason: str):
        """End current conversation and update records"""
        if self.current_context:
            # Store conversation outcome in database
            self._end_database_session(reason)
            
            # Update NPC memory
            self._update_npc_memory()
            
            print(f"💬 Ended conversation: {reason}")
            self.current_context = None
        
        self.current_state = DialogueState.IDLE
        self.choice_history.clear()
    
    def _start_database_session(self):
        """Start tracking conversation in database"""
        if not self.current_context:
            return
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Insert into both dialogue_sessions and conversations (for test compatibility)
            cursor.execute("""
                INSERT INTO dialogue_sessions 
                (session_start, npc_type, npc_name, location_map, conversation_topic, objective)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().isoformat(),
                self.current_context.npc_type.value,
                self.current_context.npc_name,
                self.current_context.location_map,
                self.current_context.conversation_topic,
                self.current_context.current_objective
            ))
            
            self.current_session_id = cursor.lastrowid
            
            # Also insert into conversations table for test compatibility
            cursor.execute("""
                INSERT INTO conversations 
                (session_start, start_time, npc_type, npc_name, location_map, conversation_topic, objective)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().isoformat(),
                datetime.now().isoformat(),
                self.current_context.npc_type.value,
                self.current_context.npc_name,
                self.current_context.location_map,
                self.current_context.conversation_topic,
                self.current_context.current_objective
            ))
            
            conn.commit()
    
    def _store_choices(self, choices: List[DialogueChoice]):
        """Store dialogue choices in database"""
        if not hasattr(self, 'current_session_id'):
            return
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            for choice in choices:
                cursor.execute("""
                    INSERT INTO dialogue_choices 
                    (session_id, timestamp, choice_text, choice_id, expected_outcome, confidence)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    self.current_session_id,
                    datetime.now().isoformat(),
                    choice.text,
                    choice.choice_id,
                    choice.expected_outcome,
                    choice.confidence
                ))
            conn.commit()
    
    def _end_database_session(self, outcome: str):
        """End database session tracking"""
        if not hasattr(self, 'current_session_id') or not self.current_context:
            return
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE dialogue_sessions 
                SET session_end = ?, total_exchanges = ?, choices_made = ?, outcome = ?
                WHERE id = ?
            """, (
                datetime.now().isoformat(),
                len(self.current_context.conversation_history),
                len(self.current_context.choices_made),
                outcome,
                self.current_session_id
            ))
            conn.commit()
    
    def _update_npc_memory(self):
        """Update NPC memory with conversation results"""
        if not self.current_context:
            return
        
        npc_id = f"{self.current_context.location_map}_{self.current_context.npc_type.value}"
        if self.current_context.npc_name:
            npc_id += f"_{self.current_context.npc_name.lower()}"
        
        conversation_summary = ' '.join(self.current_context.conversation_history[-3:])  # Last 3 exchanges
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO npc_interactions 
                (npc_identifier, location_map, npc_type, npc_name, last_conversation, 
                 topics_discussed, interaction_count, last_interaction, conversation_success)
                VALUES (?, ?, ?, ?, ?, ?, 
                        COALESCE((SELECT interaction_count FROM npc_interactions WHERE npc_identifier = ?) + 1, 1),
                        ?, ?)
            """, (
                npc_id, self.current_context.location_map, self.current_context.npc_type.value,
                self.current_context.npc_name, conversation_summary,
                self.current_context.conversation_topic or "general",
                npc_id,  # For the COALESCE subquery
                datetime.now().isoformat(),
                len(self.current_context.choices_made) > 0  # Success if choices were made
            ))
            conn.commit()
    
    def get_dialogue_stats(self) -> Dict[str, Any]:
        """Get dialogue system statistics"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Total conversations
            cursor.execute("SELECT COUNT(*) FROM dialogue_sessions")
            total_conversations = cursor.fetchone()[0]
            
            # Conversations by NPC type
            cursor.execute("""
                SELECT npc_type, COUNT(*) FROM dialogue_sessions 
                GROUP BY npc_type ORDER BY COUNT(*) DESC
            """)
            conversations_by_type = dict(cursor.fetchall())
            
            # Average conversation length
            cursor.execute("SELECT AVG(total_exchanges) FROM dialogue_sessions WHERE total_exchanges > 0")
            avg_length = cursor.fetchone()[0] or 0
            
            # Choice statistics
            cursor.execute("SELECT COUNT(*) FROM dialogue_choices")
            total_choices = cursor.fetchone()[0]
            
            return {
                "total_conversations": total_conversations,
                "conversations_by_npc_type": conversations_by_type,
                "average_conversation_length": round(avg_length, 1),
                "total_choices_presented": total_choices,
                "current_state": self.current_state.value,
                "dialogue_history_length": len(self.dialogue_history)
            }
    
    def reset_conversation(self):
        """Reset current conversation state"""
        if self.current_state != DialogueState.IDLE:
            self._end_conversation("manual_reset")
        
        # Clear dialogue history
        self.dialogue_history.clear()
        
        print("💬 Dialogue state machine reset")
    
    # Legacy method for backwards compatibility
    def process_dialogue(self, visual_context: VisualContext) -> Dict[str, Any]:
        """Legacy method for processing dialogue - calls update_state"""
        game_state = {"player": {"map": 0}, "party": []}
        return self.update_state(visual_context, game_state)
    
    # Missing attributes that tests expect
    @property
    def current_npc_type(self) -> Optional[NPCType]:
        """Get current NPC type"""
        return self.current_context.npc_type if self.current_context else None
    
    @property
    def current_session_id(self) -> Optional[str]:
        """Get current session ID"""
        return getattr(self, '_current_session_id', None)
    
    @current_session_id.setter
    def current_session_id(self, value: Optional[str]):
        """Set current session ID"""
        self._current_session_id = value
    
    @property
    def current_conversation_id(self) -> Optional[str]:
        """Get current conversation ID (alias for session_id)"""
        return self.current_session_id
    
    def reset(self):
        """Reset the dialogue state machine"""
        self.reset_conversation()
    
    def _build_game_context(self, game_state: Dict[str, Any]) -> GameContext:
        """Build game context for semantic analysis"""
        player = game_state.get('player', {})
        party = game_state.get('party', [])
        
        return GameContext(
            current_objective=self._get_current_objective(game_state) or "explore",
            player_progress={
                "level": player.get('level', 1),
                "badges": player.get('badges', 0),
                "party_size": len(party)
            },
            location_info={
                "map_id": player.get('map', 0),
                "location_type": self._determine_location_type(player.get('map', 0))
            },
            recent_events=[],
            active_quests=[]
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Alias for get_dialogue_stats for compatibility"""
        return self.get_dialogue_stats()


def test_dialogue_state_machine():
    """Test the dialogue state machine"""
    from vision_processor import DetectedText, VisualContext
    
    print("🧪 Testing Dialogue State Machine...")
    
    # Create state machine
    dsm = DialogueStateMachine("test_dialogue.db")
    
    # Create test visual context (dialogue screen)
    test_texts = [
        DetectedText("Hello! I'm Professor Elm!", 0.9, (10, 100, 150, 130), "dialogue"),
        DetectedText("Would you like to choose a starter Pokemon?", 0.8, (10, 130, 150, 160), "dialogue"),
        DetectedText("Yes", 0.9, (20, 170, 40, 185), "dialogue"),
        DetectedText("No", 0.9, (60, 170, 80, 185), "dialogue")
    ]
    
    context = VisualContext(
        screen_type="dialogue",
        detected_text=test_texts,
        ui_elements=[],
        dominant_colors=[(255, 255, 255)],
        game_phase="dialogue_interaction",
        visual_summary="Professor dialogue with choices"
    )
    
    # Test game state
    game_state = {
        "player": {"map": 0, "x": 5, "y": 10, "money": 0, "badges": 0},
        "party": []
    }
    
    # Update state
    new_state = dsm.update_state(context, game_state)
    print(f"✅ Updated state: {new_state}")
    
    # Get recommended action
    action = dsm.get_recommended_action()
    print(f"✅ Recommended action: {action}")
    
    # Get stats
    stats = dsm.get_dialogue_stats()
    print(f"✅ Stats: {stats}")
    
    print("\n🎉 Dialogue state machine test completed!")


if __name__ == "__main__":
    test_dialogue_state_machine()
