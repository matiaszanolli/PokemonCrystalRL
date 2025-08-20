#!/usr/bin/env python3
"""
choice_recognition_system.py - Recognition system for game choices/options

This module provides the system for recognizing and handling different types of
choices presented in the game, including binary choices, menu selections,
Pokemon choices, etc.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
import numpy as np
from vision.vision_processor import VisualContext


class ChoiceType(Enum):
    """Types of choices that can be recognized"""
    YES_NO = "yes_no"
    MULTIPLE_CHOICE = "multiple_choice"
    POKEMON_SELECTION = "pokemon_selection"
    MENU_SELECTION = "menu_selection"
    DIRECTIONAL = "directional"
    CONFIRMATION = "confirmation"


class ChoicePosition(Enum):
    """Position of a choice in the UI"""
    TOP = "top"
    MIDDLE = "middle"
    BOTTOM = "bottom"
    LEFT = "left"
    CENTER = "center"
    RIGHT = "right"


@dataclass
class ChoiceContext:
    """Context for choice recognition and analysis"""
    dialogue_text: Optional[str] = None  # Current dialogue text if any
    screen_type: str = "unknown"  # Type of screen showing choices
    npc_type: Optional[str] = None  # Type of NPC if interacting
    current_objective: Optional[str] = None  # Current game objective
    conversation_history: List[str] = None  # Recent dialogue history
    ui_layout: str = "standard_dialogue"  # UI layout type
    battle_context: Optional[Dict] = None  # Battle context if in battle
    move_types: Optional[Dict[str, str]] = None  # Move types mapping
    semantic_context: Optional[Dict] = None  # Additional semantic info
    previous_moves: Optional[List[str]] = None  # Previously used moves

    def __post_init__(self):
        if self.conversation_history is None:
            self.conversation_history = []


@dataclass
class RecognizedChoice:
    """A recognized choice option with analysis"""
    text: str  # Choice text
    choice_type: ChoiceType  # Type of choice
    position: ChoicePosition  # Position in UI
    action_mapping: List[str]  # Actions to select this choice
    confidence: float  # Recognition confidence
    priority: float  # Priority for selection
    expected_outcome: str  # Expected outcome if selected
    context_tags: List[str]  # Contextual tags
    ui_coordinates: Tuple[int, int, int, int]  # Position in UI


class ChoiceRecognitionSystem:
    """System for recognizing and analyzing game choices"""
    
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or "data/choice_patterns.db"
        self.initialize_patterns()
        self.initialize_action_mappings()
        self.initialize_ui_layouts()

    def initialize_patterns(self):
        """Initialize choice recognition patterns"""
        self.choice_patterns = {
            "yes_no_basic": {
                "pattern": r"yes|no|okay|sure|nope",
                "type": ChoiceType.YES_NO,
                "indicators": ["affirmative", "negative"],
                "confidence_boost": 1.2
            },
            "pokemon_selection": {
                "pattern": r"cyndaquil|totodile|chikorita",
                "type": ChoiceType.POKEMON_SELECTION,
                "indicators": ["starter", "pokemon_name"],
                "confidence_boost": 1.1
            },
            "menu_selection": {
                "pattern": r"fight|use item|switch|run",
                "type": ChoiceType.MENU_SELECTION,
                "indicators": ["menu", "battle"],
                "confidence_boost": 1.0
            },
            "numbered_choices": {
                "pattern": r"\d+[.)] .+",
                "type": ChoiceType.MULTIPLE_CHOICE,
                "indicators": ["numbered", "list"],
                "confidence_boost": 1.0
            }
        }

    def initialize_action_mappings(self):
        """Initialize mappings from choices to game actions"""
        self.action_mappings = {
            "yes": ["A"],
            "no": ["B"],
            "cyndaquil": ["A"],
            "totodile": ["DOWN", "A"],
            "chikorita": ["DOWN", "DOWN", "A"],
            "fight": ["A"],
            "run": ["DOWN", "DOWN", "DOWN", "A"],
            "confirm": ["A"],
            "cancel": ["B"]
        }

    def initialize_ui_layouts(self):
        """Initialize UI layout definitions"""
        self.ui_layouts = {
            "standard_dialogue": {
                "description": "Standard dialogue layout with choices",
                "choice_positions": ["top", "bottom"],
                "navigation": "vertical"
            },
            "menu_selection": {
                "description": "Standard menu layout",
                "choice_positions": ["top", "middle", "bottom"],
                "navigation": "vertical"
            },
            "pokemon_selection": {
                "description": "Pokemon selection layout",
                "choice_positions": ["left", "center", "right"],
                "navigation": "horizontal"
            },
            "yes_no_dialog": {
                "description": "Yes/No dialogue prompt",
                "choice_positions": ["top", "bottom"],
                "navigation": "vertical"
            }
        }

    def recognize_choices(self, visual_context: VisualContext, 
                        choice_context: ChoiceContext) -> List[RecognizedChoice]:
        """Recognize and analyze choices in the current screen"""
        if not visual_context or not visual_context.detected_text:
            return []

        choices = []
        texts = self._extract_choice_texts(visual_context.detected_text)

        for i, text_info in enumerate(texts):
            text = text_info["text"]
            # Pattern match the choice
            choice_type, confidence = self._match_choice_patterns(text.lower())
            if not choice_type:
                continue

            # Determine position based on UI layout
            position = self._determine_choice_position(text_info, i, len(texts))

            # Generate action mapping
            action_mapping = self._generate_action_mapping(text.lower(), choice_type, position, i)

            # Calculate priority
            priority = self._calculate_choice_priority(text, choice_type, choice_context, confidence)

            # Determine expected outcome
            expected_outcome = self._determine_expected_outcome(text, choice_type, choice_context)

            # Generate context tags
            context_tags = self._generate_context_tags(text, choice_type, choice_context)

            # Create recognized choice
            choice = RecognizedChoice(
                text=text,
                choice_type=choice_type,
                position=position,
                action_mapping=action_mapping,
                confidence=confidence,
                priority=priority,
                expected_outcome=expected_outcome,
                context_tags=context_tags,
                ui_coordinates=text_info.get("coordinates", (0, 0, 0, 0))
            )
            choices.append(choice)

        return choices

    def get_best_choice_action(self, choices: List[RecognizedChoice]) -> List[str]:
        """Get the best action sequence based on available choices"""
        if not choices:
            return ["A"]  # Default to A button if no choices

        # Get choice with highest priority
        best_choice = max(choices, key=lambda x: x.priority)
        return best_choice.action_mapping

    def _extract_choice_texts(self, detected_texts: List) -> List[Dict]:
        """Extract potential choice texts from detected text"""
        choice_texts = []
        for text in detected_texts:
            if 2 <= len(text.text) <= 20:  # Reasonable length for a choice
                choice_texts.append({
                    "text": text.text,
                    "coordinates": text.bbox,
                    "confidence": text.confidence,
                    "location": text.location
                })
        return choice_texts

    def _match_choice_patterns(self, text: str) -> Tuple[Optional[ChoiceType], float]:
        """Match text against choice patterns"""
        if not text:
            return None, 0.0

        best_match = None
        best_confidence = 0.0

        # Simple pattern matching for demonstration
        text = text.lower()
        if any(x in text for x in ["yes", "okay", "sure"]):
            return ChoiceType.YES_NO, 0.9
        elif any(x in text for x in ["no", "nope", "cancel"]):
            return ChoiceType.YES_NO, 0.9
        elif any(x in text for x in ["cyndaquil", "totodile", "chikorita"]):
            return ChoiceType.POKEMON_SELECTION, 0.95
        elif text.startswith(("1.", "2.", "3.")):
            return ChoiceType.MULTIPLE_CHOICE, 0.8
        elif any(x in text for x in ["fight", "item", "pokemon", "run"]):
            return ChoiceType.MENU_SELECTION, 0.85
        elif any(x in text for x in ["north", "south", "east", "west"]):
            return ChoiceType.DIRECTIONAL, 0.8

        return None, 0.0

    def _determine_choice_position(self, choice_info: Dict, index: int, total: int) -> ChoicePosition:
        """Determine UI position of a choice"""
        if total == 1:
            return ChoicePosition.CENTER
        elif total == 2:
            return ChoicePosition.TOP if index == 0 else ChoicePosition.BOTTOM
        else:
            if index == 0:
                return ChoicePosition.TOP
            elif index == total - 1:
                return ChoicePosition.BOTTOM
            else:
                return ChoicePosition.MIDDLE

    def _generate_action_mapping(self, text: str, choice_type: ChoiceType, 
                               position: ChoicePosition, index: int) -> List[str]:
        """Generate action sequence to select this choice"""
        # Direct text mappings
        if text in self.action_mappings:
            return self.action_mappings[text]

        # Position-based mappings
        if position == ChoicePosition.TOP:
            return ["A"]
        elif position == ChoicePosition.MIDDLE:
            return ["DOWN", "A"]
        elif position == ChoicePosition.BOTTOM:
            return ["DOWN", "DOWN", "A"]

        return ["A"]  # Default to A button

    def _calculate_choice_priority(self, text: str, choice_type: ChoiceType, 
                                 context: ChoiceContext, confidence: float) -> float:
        """Calculate priority score for a choice"""
        priority = confidence * 50  # Base priority from confidence

        # Adjust based on NPC type
        if context.npc_type == "professor" and choice_type == ChoiceType.POKEMON_SELECTION:
            priority += 20

        # Adjust based on current objective
        if context.current_objective == "get_starter_pokemon" and "cyndaquil" in text.lower():
            priority += 30

        return min(priority, 100)  # Cap at 100

    def _determine_expected_outcome(self, text: str, choice_type: ChoiceType, 
                                  context: Optional[ChoiceContext]) -> str:
        """Determine expected outcome of selecting this choice"""
        text = text.lower()
        
        if choice_type == ChoiceType.YES_NO:
            return "accept_or_confirm" if text in ["yes", "okay", "sure"] else "decline_or_cancel"
        elif choice_type == ChoiceType.POKEMON_SELECTION:
            return f"select_{text}"
        elif choice_type == ChoiceType.MENU_SELECTION:
            return f"enter_{text}_menu"
        elif choice_type == ChoiceType.DIRECTIONAL:
            return f"move_{text}"
        else:
            return "unknown_outcome"

    def _generate_context_tags(self, text: str, choice_type: ChoiceType, 
                             context: ChoiceContext) -> List[str]:
        """Generate contextual tags for the choice"""
        tags = [f"type_{choice_type.name.lower()}"]

        if context.npc_type:
            tags.append(f"npc_{context.npc_type}")
        if context.current_objective:
            tags.append(f"objective_{context.current_objective}")

        # Add response type tags
        if choice_type == ChoiceType.YES_NO:
            if text.lower() in ["yes", "okay", "sure"]:
                tags.append("positive_response")
            else:
                tags.append("negative_response")
        elif choice_type == ChoiceType.POKEMON_SELECTION:
            tags.append("pokemon_choice")

        return tags

    def update_choice_effectiveness(self, choice_text: str, actions: List[str], 
                                  success: bool, ui_layout: str):
        """Update effectiveness metrics for choices"""
        # This would update the database with success/failure metrics
        pass

    def get_choice_statistics(self) -> Dict:
        """Get statistics about choice recognition performance"""
        return {
            "total_choice_recognitions": 0,
            "average_confidence": 0.0,
            "loaded_patterns": len(self.choice_patterns),
            "loaded_action_mappings": len(self.action_mappings),
            "top_action_mappings": []
        }
