#!/usr/bin/env python3
"""
choice_recognition_system.py - Enhanced Choice Recognition System for Pokemon Crystal Agent

This module provides sophisticated dialogue choice extraction, recognition, and mapping
to game actions. It handles various choice formats, UI layouts, and provides intelligent
action selection based on context and objectives.
"""

import re
import json
import sqlite3
from typing import Dict, List, Optional, Tuple, Set, Any
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from datetime import datetime
import hashlib

from vision_processor import DetectedText, VisualContext


class ChoiceType(Enum):
    """Types of dialogue choices"""
    YES_NO = "yes_no"                    # Simple yes/no choice
    MULTIPLE_CHOICE = "multiple_choice"  # Multiple numbered/lettered options
    MENU_SELECTION = "menu_selection"    # Menu-style choices
    POKEMON_SELECTION = "pokemon_selection"  # Choosing Pokemon
    ITEM_SELECTION = "item_selection"    # Choosing items
    DIRECTIONAL = "directional"          # Direction-based choices
    CONFIRMATION = "confirmation"        # Confirm/cancel dialogs
    NUMERIC_INPUT = "numeric_input"      # Number selection


class ChoicePosition(Enum):
    """Position of choice in UI layout"""
    TOP = "top"
    MIDDLE = "middle" 
    BOTTOM = "bottom"
    LEFT = "left"
    RIGHT = "right"
    CENTER = "center"


@dataclass
class RecognizedChoice:
    """A recognized dialogue choice with all metadata"""
    text: str                           # The choice text
    choice_type: ChoiceType            # Type of choice
    position: ChoicePosition           # Position in UI
    action_mapping: List[str]          # Game actions to select this choice
    confidence: float                  # Recognition confidence (0.0-1.0)
    priority: int                      # Priority for selection (higher = better)
    expected_outcome: str              # What we expect this choice to do
    context_tags: List[str]           # Context tags for this choice
    ui_coordinates: Tuple[int, int, int, int]  # Bounding box coordinates


@dataclass
class ChoiceContext:
    """Context information for choice recognition"""
    dialogue_text: str                 # The dialogue text above choices
    screen_type: str                   # Type of screen (dialogue, menu, etc.)
    npc_type: str                     # Type of NPC we're talking to
    current_objective: Optional[str]   # Player's current objective
    conversation_history: List[str]    # Recent conversation history
    ui_layout: str                    # Detected UI layout pattern


class ChoiceRecognitionSystem:
    """
    Advanced system for recognizing and processing dialogue choices
    """
    
    def __init__(self, db_path: str = "choice_recognition.db"):
        """Initialize the choice recognition system"""
        self.db_path = Path(db_path)
        
        # Load choice patterns and mappings
        self.choice_patterns: Dict[str, Any] = {}
        self.action_mappings: Dict[str, List[str]] = {}
        self.ui_layouts: Dict[str, Dict[str, Any]] = {}
        
        # Initialize database and patterns
        self._init_database()
        self._load_choice_patterns()
        self._load_action_mappings()
        self._load_ui_layouts()
        
        print("🎯 Choice recognition system initialized")
    
    def _init_database(self):
        """Initialize SQLite database for choice recognition tracking"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Choice recognition history
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS choice_recognitions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    dialogue_text TEXT NOT NULL,
                    recognized_choices TEXT NOT NULL,
                    chosen_action TEXT,
                    success BOOLEAN,
                    confidence REAL,
                    context_data TEXT
                )
            """)
            
            # Choice pattern effectiveness
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS pattern_effectiveness (
                    pattern_id TEXT PRIMARY KEY,
                    usage_count INTEGER DEFAULT 0,
                    success_count INTEGER DEFAULT 0,
                    average_confidence REAL DEFAULT 0.5,
                    last_used TEXT
                )
            """)
            
            # Action mapping success rates
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS action_mappings (
                    choice_text TEXT,
                    ui_layout TEXT,
                    action_sequence TEXT,
                    success_count INTEGER DEFAULT 0,
                    failure_count INTEGER DEFAULT 0,
                    last_used TEXT,
                    PRIMARY KEY (choice_text, ui_layout, action_sequence)
                )
            """)
            
            conn.commit()
    
    def _load_choice_patterns(self):
        """Load choice recognition patterns"""
        self.choice_patterns = {
            # Yes/No patterns
            "yes_no_basic": {
                "pattern": r"\b(yes|no)\b",
                "type": ChoiceType.YES_NO,
                "indicators": ["yes", "no"],
                "confidence_boost": 0.3
            },
            
            "yes_no_alternatives": {
                "pattern": r"\b(okay|sure|nope|nah|alright|fine)\b",
                "type": ChoiceType.YES_NO,
                "indicators": ["okay", "sure", "nope", "nah", "alright", "fine"],
                "confidence_boost": 0.2
            },
            
            # Multiple choice patterns
            "numbered_choices": {
                "pattern": r"[1-9]\.\s*\w+|[1-9]\)\s*\w+",
                "type": ChoiceType.MULTIPLE_CHOICE,
                "indicators": ["1.", "2.", "3.", "1)", "2)", "3)"],
                "confidence_boost": 0.4
            },
            
            "lettered_choices": {
                "pattern": r"[a-d]\.\s*\w+|[a-d]\)\s*\w+",
                "type": ChoiceType.MULTIPLE_CHOICE,
                "indicators": ["a.", "b.", "c.", "d.", "a)", "b)", "c)", "d)"],
                "confidence_boost": 0.4
            },
            
            # Pokemon-specific patterns
            "starter_pokemon": {
                "pattern": r"\b(cyndaquil|totodile|chikorita)\b",
                "type": ChoiceType.POKEMON_SELECTION,
                "indicators": ["cyndaquil", "totodile", "chikorita"],
                "confidence_boost": 0.5
            },
            
            "pokemon_actions": {
                "pattern": r"\b(fight|use item|switch|run)\b",
                "type": ChoiceType.MENU_SELECTION,
                "indicators": ["fight", "use item", "switch", "run"],
                "confidence_boost": 0.4
            },
            
            # Menu patterns
            "menu_navigation": {
                "pattern": r"\b(back|cancel|exit|continue|next)\b",
                "type": ChoiceType.MENU_SELECTION,
                "indicators": ["back", "cancel", "exit", "continue", "next"],
                "confidence_boost": 0.3
            },
            
            # Confirmation patterns
            "confirmation": {
                "pattern": r"\b(confirm|cancel|accept|decline)\b",
                "type": ChoiceType.CONFIRMATION,
                "indicators": ["confirm", "cancel", "accept", "decline"],
                "confidence_boost": 0.3
            },
            
            # Directional patterns
            "directions": {
                "pattern": r"\b(north|south|east|west|up|down|left|right)\b",
                "type": ChoiceType.DIRECTIONAL,
                "indicators": ["north", "south", "east", "west", "up", "down", "left", "right"],
                "confidence_boost": 0.3
            }
        }
    
    def _load_action_mappings(self):
        """Load mappings from choices to game actions"""
        self.action_mappings = {
            # Basic navigation
            "yes": ["A"],
            "no": ["B"],
            "okay": ["A"], 
            "sure": ["A"],
            "nope": ["B"],
            "cancel": ["B"],
            "back": ["B"],
            
            # Menu selections - position based
            "top_choice": ["A"],
            "middle_choice": ["DOWN", "A"],
            "bottom_choice": ["DOWN", "DOWN", "A"],
            "second_choice": ["DOWN", "A"],
            "third_choice": ["DOWN", "DOWN", "A"],
            
            # Pokemon-specific
            "cyndaquil": ["A"],      # Usually first choice
            "totodile": ["DOWN", "A"],   # Usually second choice  
            "chikorita": ["DOWN", "DOWN", "A"],  # Usually third choice
            
            # Battle menu
            "fight": ["A"],
            "use item": ["DOWN", "A"],
            "switch": ["DOWN", "DOWN", "A"],
            "run": ["DOWN", "DOWN", "DOWN", "A"],
            
            # Directional
            "north": ["UP"],
            "south": ["DOWN"],
            "east": ["RIGHT"],
            "west": ["LEFT"],
            "up": ["UP"],
            "down": ["DOWN"],
            "left": ["LEFT"],
            "right": ["RIGHT"],
            
            # Numbered choices
            "1": ["A"],
            "2": ["DOWN", "A"],
            "3": ["DOWN", "DOWN", "A"],
            "4": ["DOWN", "DOWN", "DOWN", "A"],
            
            # Confirmation
            "confirm": ["A"],
            "accept": ["A"],
            "decline": ["B"]
        }
    
    def _load_ui_layouts(self):
        """Load UI layout patterns for different screen types"""
        self.ui_layouts = {
            "standard_dialogue": {
                "description": "Standard dialogue box with choices below",
                "choice_positions": {
                    2: [ChoicePosition.TOP, ChoicePosition.BOTTOM],
                    3: [ChoicePosition.TOP, ChoicePosition.MIDDLE, ChoicePosition.BOTTOM],
                    4: [ChoicePosition.TOP, ChoicePosition.TOP, ChoicePosition.BOTTOM, ChoicePosition.BOTTOM]
                },
                "navigation": "vertical"
            },
            
            "menu_selection": {
                "description": "Menu-style selection screen",
                "choice_positions": {
                    2: [ChoicePosition.LEFT, ChoicePosition.RIGHT],
                    3: [ChoicePosition.LEFT, ChoicePosition.CENTER, ChoicePosition.RIGHT],
                    4: [ChoicePosition.LEFT, ChoicePosition.LEFT, ChoicePosition.RIGHT, ChoicePosition.RIGHT]
                },
                "navigation": "grid"
            },
            
            "pokemon_selection": {
                "description": "Pokemon selection screen",
                "choice_positions": {
                    3: [ChoicePosition.LEFT, ChoicePosition.CENTER, ChoicePosition.RIGHT]
                },
                "navigation": "horizontal"
            },
            
            "yes_no_dialog": {
                "description": "Simple yes/no confirmation",
                "choice_positions": {
                    2: [ChoicePosition.LEFT, ChoicePosition.RIGHT]
                },
                "navigation": "horizontal"
            }
        }
    
    def recognize_choices(self, visual_context: VisualContext, choice_context: ChoiceContext) -> List[RecognizedChoice]:
        """
        Recognize dialogue choices from visual context
        
        Args:
            visual_context: Current visual analysis
            choice_context: Context information for choice recognition
            
        Returns:
            List of recognized choices with metadata
        """
        if not visual_context.detected_text:
            return []
        
        # Extract potential choice texts
        choice_texts = self._extract_choice_texts(visual_context.detected_text)
        
        if not choice_texts:
            return []
        
        # Recognize choice patterns
        recognized_choices = []
        for i, choice_info in enumerate(choice_texts):
            choice = self._analyze_choice(choice_info, choice_context, i, len(choice_texts))
            if choice:
                recognized_choices.append(choice)
        
        # Apply context-based prioritization
        recognized_choices = self._apply_context_prioritization(recognized_choices, choice_context)
        
        # Store recognition for learning
        self._store_choice_recognition(choice_context.dialogue_text, recognized_choices)
        
        return sorted(recognized_choices, key=lambda x: x.priority, reverse=True)
    
    def _extract_choice_texts(self, detected_texts: List[DetectedText]) -> List[Dict[str, Any]]:
        """Extract potential choice texts from detected text"""
        choice_texts = []
        
        for text_obj in detected_texts:
            # Handle None text gracefully
            if text_obj.text is None:
                continue
            text = text_obj.text.strip()
            
            # Skip if text is too long (probably not a choice)
            if len(text) > 50:
                continue
            
            # Skip if text is too short (probably not meaningful)
            if len(text) < 2:
                continue
            
            choice_info = {
                "text": text,
                "coordinates": text_obj.bbox,
                "confidence": text_obj.confidence,
                "location": text_obj.location
            }
            
            choice_texts.append(choice_info)
        
        return choice_texts
    
    def _analyze_choice(self, choice_info: Dict[str, Any], context: ChoiceContext, 
                      index: int, total_choices: int) -> Optional[RecognizedChoice]:
        """Analyze a single choice and create RecognizedChoice object"""
        text = choice_info["text"].lower()
        
        # Pattern matching
        choice_type, pattern_confidence = self._match_choice_patterns(text)
        
        if choice_type is None:
            return None
        
        # Determine position
        position = self._determine_choice_position(choice_info, index, total_choices)
        
        # Generate action mapping
        action_mapping = self._generate_action_mapping(text, choice_type, position, index)
        
        # Calculate priority
        priority = self._calculate_choice_priority(text, choice_type, context, pattern_confidence)
        
        # Determine expected outcome
        expected_outcome = self._determine_expected_outcome(text, choice_type, context)
        
        # Generate context tags
        context_tags = self._generate_context_tags(text, choice_type, context)
        
        return RecognizedChoice(
            text=choice_info["text"],
            choice_type=choice_type,
            position=position,
            action_mapping=action_mapping,
            confidence=pattern_confidence,
            priority=priority,
            expected_outcome=expected_outcome,
            context_tags=context_tags,
            ui_coordinates=choice_info["coordinates"]
        )
    
    def _match_choice_patterns(self, text: str) -> Tuple[Optional[ChoiceType], float]:
        """Match text against choice patterns"""
        best_match = None
        best_confidence = 0.0
        
        for pattern_id, pattern_data in self.choice_patterns.items():
            matches = re.findall(pattern_data["pattern"], text, re.IGNORECASE)
            
            if matches:
                # Calculate confidence based on match quality
                confidence = 0.5  # Base confidence
                
                # Boost for exact indicator matches
                for indicator in pattern_data["indicators"]:
                    if indicator.lower() in text.lower():
                        confidence += pattern_data["confidence_boost"] / len(pattern_data["indicators"])
                
                # Boost for pattern specificity
                confidence += len(matches) * 0.1
                confidence = min(confidence, 1.0)  # Cap at 1.0
                
                if confidence > best_confidence:
                    best_match = pattern_data["type"]
                    best_confidence = confidence
        
        return best_match, best_confidence
    
    def _determine_choice_position(self, choice_info: Dict[str, Any], index: int, total: int) -> ChoicePosition:
        """Determine the position of a choice in the UI"""
        if total == 1:
            return ChoicePosition.CENTER
        elif total == 2:
            return ChoicePosition.TOP if index == 0 else ChoicePosition.BOTTOM
        elif total == 3:
            if index == 0:
                return ChoicePosition.TOP
            elif index == 1:
                return ChoicePosition.MIDDLE
            else:
                return ChoicePosition.BOTTOM
        else:
            # For more than 3 choices, use coordinate-based positioning
            y_coord = choice_info["coordinates"][1]  # Y coordinate
            if y_coord < 100:
                return ChoicePosition.TOP
            elif y_coord > 200:
                return ChoicePosition.BOTTOM
            else:
                return ChoicePosition.MIDDLE
    
    def _generate_action_mapping(self, text: str, choice_type: ChoiceType, 
                               position: ChoicePosition, index: int) -> List[str]:
        """Generate action mapping for a choice"""
        text_lower = text.lower()
        
        # Direct text mapping
        if text_lower in self.action_mappings:
            return self.action_mappings[text_lower]
        
        # Position-based mapping
        if choice_type == ChoiceType.YES_NO:
            if "yes" in text_lower or "okay" in text_lower or "sure" in text_lower:
                return ["A"]
            else:
                return ["B"]
        
        # Index-based mapping for multiple choice
        if choice_type == ChoiceType.MULTIPLE_CHOICE:
            actions = []
            for _ in range(index):
                actions.append("DOWN")
            actions.append("A")
            return actions
        
        # Position-based mapping
        position_mapping = {
            ChoicePosition.TOP: ["A"],
            ChoicePosition.MIDDLE: ["DOWN", "A"],
            ChoicePosition.BOTTOM: ["DOWN", "DOWN", "A"],
            ChoicePosition.LEFT: ["LEFT", "A"],
            ChoicePosition.RIGHT: ["RIGHT", "A"],
            ChoicePosition.CENTER: ["A"]
        }
        
        return position_mapping.get(position, ["A"])
    
    def _calculate_choice_priority(self, text: str, choice_type: ChoiceType, 
                                 context: ChoiceContext, pattern_confidence: float) -> int:
        """Calculate priority score for a choice"""
        priority = int(pattern_confidence * 50)  # Base priority from pattern confidence
        
        text_lower = text.lower()
        
        # Objective-based priority boosts
        if context.current_objective:
            objective = context.current_objective.lower()
            
            if "starter" in objective and any(pokemon in text_lower for pokemon in ["cyndaquil", "totodile", "chikorita"]):
                if "cyndaquil" in text_lower:  # Prefer fire starter
                    priority += 30
                else:
                    priority += 10
            
            elif "gym" in objective or "battle" in objective:
                if any(word in text_lower for word in ["yes", "challenge", "battle", "fight"]):
                    priority += 25
            
            elif "heal" in objective:
                if any(word in text_lower for word in ["yes", "heal", "rest"]):
                    priority += 20
        
        # NPC-type based priority
        if context.npc_type == "professor":
            if any(word in text_lower for word in ["yes", "okay", "sure"]):
                priority += 15
        elif context.npc_type == "gym_leader":
            if any(word in text_lower for word in ["yes", "challenge", "battle"]):
                priority += 20
        
        # Choice type priority adjustments
        type_priorities = {
            ChoiceType.YES_NO: 10,
            ChoiceType.POKEMON_SELECTION: 15,
            ChoiceType.CONFIRMATION: 5,
            ChoiceType.MULTIPLE_CHOICE: 8,
            ChoiceType.MENU_SELECTION: 8
        }
        
        priority += type_priorities.get(choice_type, 0)
        
        return priority
    
    def _determine_expected_outcome(self, text: str, choice_type: ChoiceType, context: ChoiceContext) -> str:
        """Determine the expected outcome of selecting this choice"""
        text_lower = text.lower()
        
        # Direct outcome mappings
        outcome_mappings = {
            "yes": "accept_or_confirm",
            "no": "decline_or_cancel",
            "okay": "accept_or_confirm",
            "sure": "accept_or_confirm",
            "cancel": "cancel_action",
            "back": "go_back",
            "continue": "proceed_forward",
            "cyndaquil": "select_fire_starter",
            "totodile": "select_water_starter",
            "chikorita": "select_grass_starter",
            "fight": "enter_battle_menu",
            "run": "flee_battle",
            "confirm": "confirm_action",
            "accept": "accept_offer",
            "decline": "decline_offer",
            "north": "move_north"  # Add missing north mapping
        }
        
        # Sort by length (descending) to prioritize longer, more specific matches
        sorted_mappings = sorted(outcome_mappings.items(), key=lambda x: len(x[0]), reverse=True)
        for key, outcome in sorted_mappings:
            if key in text_lower:
                return outcome
        
        # Context-based outcomes
        if choice_type == ChoiceType.POKEMON_SELECTION:
            return f"select_{text_lower.replace(' ', '_')}"
        elif choice_type == ChoiceType.DIRECTIONAL:
            return f"move_{text_lower}"
        elif choice_type == ChoiceType.MENU_SELECTION:
            return f"select_{text_lower.replace(' ', '_')}"
        
        return "unknown_outcome"
    
    def _generate_context_tags(self, text: str, choice_type: ChoiceType, context: ChoiceContext) -> List[str]:
        """Generate context tags for a choice"""
        tags = []
        
        # Add choice type tag
        tags.append(f"type_{choice_type.value}")
        
        # Add NPC type tag
        if context.npc_type:
            tags.append(f"npc_{context.npc_type}")
        
        # Add objective tag
        if context.current_objective:
            tags.append(f"objective_{context.current_objective}")
        
        # Add text-based tags
        text_lower = text.lower()
        if any(word in text_lower for word in ["yes", "okay", "sure", "accept"]):
            tags.append("positive_response")
        elif any(word in text_lower for word in ["no", "cancel", "decline"]):
            tags.append("negative_response")
        
        if any(pokemon in text_lower for pokemon in ["cyndaquil", "totodile", "chikorita"]):
            tags.append("pokemon_choice")
        
        if any(word in text_lower for word in ["fight", "battle", "challenge"]):
            tags.append("battle_related")
        
        return tags
    
    def _apply_context_prioritization(self, choices: List[RecognizedChoice], 
                                    context: ChoiceContext) -> List[RecognizedChoice]:
        """Apply context-based prioritization to choices"""
        # Create new list with updated priorities to avoid modifying originals
        updated_choices = []
        for choice in choices:
            new_priority = choice.priority
            
            # Boost priority based on conversation history
            if context.conversation_history:
                recent_text = " ".join(context.conversation_history[-3:]).lower()
                
                # If recent conversation mentions specific topics, boost related choices
                if "starter" in recent_text and "pokemon_choice" in choice.context_tags:
                    new_priority += 15
                elif "battle" in recent_text and "battle_related" in choice.context_tags:
                    new_priority += 15
                elif "heal" in recent_text and "positive_response" in choice.context_tags:
                    new_priority += 10
            
            # Create updated choice with new priority
            updated_choice = RecognizedChoice(
                text=choice.text,
                choice_type=choice.choice_type,
                position=choice.position,
                action_mapping=choice.action_mapping,
                confidence=choice.confidence,
                priority=new_priority,
                expected_outcome=choice.expected_outcome,
                context_tags=choice.context_tags,
                ui_coordinates=choice.ui_coordinates
            )
            updated_choices.append(updated_choice)
        
        return updated_choices
    
    def _store_choice_recognition(self, dialogue_text: str, choices: List[RecognizedChoice]):
        """Store choice recognition for learning and analysis"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Convert choices to JSON-serializable format
            choices_data = []
            for choice in choices:
                choice_dict = asdict(choice)
                # Convert enum to string for JSON serialization
                choice_dict['choice_type'] = choice.choice_type.value
                choice_dict['position'] = choice.position.value
                choices_data.append(choice_dict)
            
            choices_json = json.dumps(choices_data)
            
            cursor.execute("""
                INSERT INTO choice_recognitions 
                (timestamp, dialogue_text, recognized_choices, confidence)
                VALUES (?, ?, ?, ?)
            """, (
                datetime.now().isoformat(),
                dialogue_text,
                choices_json,
                sum(choice.confidence for choice in choices) / len(choices) if choices else 0.0
            ))
            
            conn.commit()
    
    def get_best_choice_action(self, choices: List[RecognizedChoice]) -> List[str]:
        """Get the action sequence for the best choice"""
        if not choices:
            return ["A"]  # Default action
        
        best_choice = max(choices, key=lambda x: x.priority)
        return best_choice.action_mapping
    
    def update_choice_effectiveness(self, choice_text: str, action_sequence: List[str], 
                                   success: bool, ui_layout: str = "standard"):
        """Update the effectiveness of a choice-action mapping"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            action_str = json.dumps(action_sequence)
            
            # Get current stats
            cursor.execute("""
                SELECT success_count, failure_count FROM action_mappings 
                WHERE choice_text = ? AND ui_layout = ? AND action_sequence = ?
            """, (choice_text, ui_layout, action_str))
            
            result = cursor.fetchone()
            
            if result:
                success_count, failure_count = result
                if success:
                    success_count += 1
                else:
                    failure_count += 1
            else:
                success_count = 1 if success else 0
                failure_count = 0 if success else 1
            
            cursor.execute("""
                INSERT OR REPLACE INTO action_mappings 
                (choice_text, ui_layout, action_sequence, success_count, failure_count, last_used)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                choice_text, ui_layout, action_str, success_count, failure_count,
                datetime.now().isoformat()
            ))
            
            conn.commit()
    
    def get_choice_statistics(self) -> Dict[str, Any]:
        """Get choice recognition system statistics"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Total recognitions
            cursor.execute("SELECT COUNT(*) FROM choice_recognitions")
            total_recognitions = cursor.fetchone()[0]
            
            # Average confidence
            cursor.execute("SELECT AVG(confidence) FROM choice_recognitions WHERE confidence > 0")
            avg_confidence = cursor.fetchone()[0] or 0.0
            
            # Action mapping success rates
            cursor.execute("""
                SELECT choice_text, success_count, failure_count 
                FROM action_mappings 
                ORDER BY (success_count + failure_count) DESC 
                LIMIT 10
            """)
            top_mappings = cursor.fetchall()
            
            return {
                "total_choice_recognitions": total_recognitions,
                "average_confidence": round(avg_confidence, 3),
                "loaded_patterns": len(self.choice_patterns),
                "loaded_action_mappings": len(self.action_mappings),
                "top_action_mappings": [
                    {
                        "choice": mapping[0],
                        "success_rate": mapping[1] / (mapping[1] + mapping[2]) if (mapping[1] + mapping[2]) > 0 else 0
                    }
                    for mapping in top_mappings
                ]
            }


def test_choice_recognition_system():
    """Test the choice recognition system"""
    print("🧪 Testing Choice Recognition System...")
    
    # Create system
    system = ChoiceRecognitionSystem("test_choice_recognition.db")
    
    # Test dialogue scenarios
    test_scenarios = [
        {
            "name": "Yes/No Choice",
            "detected_texts": [
                DetectedText("Would you like to continue?", 0.9, (10, 100, 200, 120), "dialogue"),
                DetectedText("Yes", 0.95, (20, 140, 50, 160), "dialogue"),
                DetectedText("No", 0.95, (20, 170, 50, 190), "dialogue")
            ],
            "context": ChoiceContext(
                dialogue_text="Would you like to continue?",
                screen_type="dialogue",
                npc_type="generic",
                current_objective="progress_story",
                conversation_history=[],
                ui_layout="standard_dialogue"
            )
        },
        
        {
            "name": "Pokemon Selection",
            "detected_texts": [
                DetectedText("Choose your starter Pokemon!", 0.9, (10, 100, 200, 120), "dialogue"),
                DetectedText("Cyndaquil", 0.95, (20, 140, 80, 160), "dialogue"),
                DetectedText("Totodile", 0.95, (100, 140, 160, 160), "dialogue"),
                DetectedText("Chikorita", 0.95, (180, 140, 240, 160), "dialogue")
            ],
            "context": ChoiceContext(
                dialogue_text="Choose your starter Pokemon!",
                screen_type="dialogue",
                npc_type="professor",
                current_objective="get_starter_pokemon",
                conversation_history=["Hello! I'm Professor Elm!", "I study Pokemon behavior."],
                ui_layout="pokemon_selection"
            )
        }
    ]
    
    for scenario in test_scenarios:
        print(f"\n📝 Testing: {scenario['name']}")
        
        # Create visual context
        visual_context = VisualContext(
            screen_type="dialogue",
            detected_text=scenario["detected_texts"],
            ui_elements=[],
            dominant_colors=[(255, 255, 255)],
            game_phase="dialogue_interaction",
            visual_summary=scenario["name"]
        )
        
        # Recognize choices
        choices = system.recognize_choices(visual_context, scenario["context"])
        
        print(f"✅ Recognized {len(choices)} choices:")
        for i, choice in enumerate(choices, 1):
            print(f"  {i}. '{choice.text}' (type: {choice.choice_type.value}, "
                  f"priority: {choice.priority}, actions: {choice.action_mapping})")
        
        # Get best action
        best_action = system.get_best_choice_action(choices)
        print(f"🎮 Best action sequence: {best_action}")
    
    # Show statistics
    stats = system.get_choice_statistics()
    print(f"\n📊 System Statistics: {stats}")
    
    print("\n🎉 Choice recognition system test completed!")


if __name__ == "__main__":
    test_choice_recognition_system()
