#!/usr/bin/env python3
"""
choice_recognition_system.py - Choice Recognition System for Pokemon Crystal

This module handles detection and analysis of choice options in dialogues
and menus, integrating with the semantic context system for informed decisions.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple

class ChoiceType(Enum):
    """Types of choices available in the game"""
    UNKNOWN = "unknown"
    YES_NO = "yes_no"
    MENU_OPTION = "menu_option"
    POKEMON_SELECTION = "pokemon_selection"
    ITEM_SELECTION = "item_selection"
    MOVE_SELECTION = "move_selection"
    DIRECTIONAL = "directional"

@dataclass
class ChoiceContext:
    """Context information for choice recognition"""
    dialogue_text: str
    screen_type: str
    npc_type: str
    current_objective: Optional[str]
    conversation_history: List[str]
    ui_layout: str

@dataclass
class Choice:
    """Represents a recognized choice option"""
    text: str
    choice_type: ChoiceType
    bbox: Tuple[int, int, int, int]
    priority: float
    expected_outcome: Optional[str]

class ChoiceRecognitionSystem:
    """System for recognizing and analyzing choice options"""
    
    def __init__(self, db_path: Optional[str] = None):
        """Initialize choice recognition system"""
        self.db_path = db_path
        
    def recognize_choices(self, visual_context: 'VisualContext', choice_context: ChoiceContext) -> List[Choice]:
        """Recognize available choices from visual and semantic context"""
        if not visual_context or not visual_context.detected_text:
            return []
            
        choices = []
        
        # Extract choice text elements
        choice_texts = [
            text for text in visual_context.detected_text
            if text.location in ["choice", "menu"]
        ]
        
        if not choice_texts:
            return []
            
        # Special case for yes/no choices
        if len(choice_texts) == 2 and any("yes" in text.text.lower() for text in choice_texts):
            yes_choice = next(text for text in choice_texts if "yes" in text.text.lower())
            no_choice = next(text for text in choice_texts if "no" in text.text.lower())
            
            # Yes choice usually has higher priority based on context
            yes_priority = 0.8
            no_priority = 0.2
            
            choices.extend([
                Choice(
                    text=yes_choice.text,
                    choice_type=ChoiceType.YES_NO,
                    bbox=yes_choice.bbox,
                    priority=yes_priority,
                    expected_outcome="confirm"
                ),
                Choice(
                    text=no_choice.text,
                    choice_type=ChoiceType.YES_NO,
                    bbox=no_choice.bbox,
                    priority=no_priority,
                    expected_outcome="decline"
                )
            ])
            
        # Pokemon selection choices
        elif any("starter" in choice_context.dialogue_text.lower()):
            for text in choice_texts:
                # Assign different priorities based on Pokemon type
                if "cyndaquil" in text.text.lower():
                    priority = 0.9  # Fire type often preferred
                elif "totodile" in text.text.lower():
                    priority = 0.8  # Water type second choice
                else:
                    priority = 0.7  # Grass type
                    
                choices.append(Choice(
                    text=text.text,
                    choice_type=ChoiceType.POKEMON_SELECTION,
                    bbox=text.bbox,
                    priority=priority,
                    expected_outcome="select_pokemon"
                ))
                
        # Item selection choices
        elif "buy" in choice_context.dialogue_text.lower() or "shop" in choice_context.ui_layout.lower():
            for text in choice_texts:
                # Prioritize based on item type
                priority = 0.5  # Default priority
                if "potion" in text.text.lower():
                    priority = 0.9  # Healing items high priority
                elif "ball" in text.text.lower():
                    priority = 0.8  # Pokeballs also important
                    
                choices.append(Choice(
                    text=text.text,
                    choice_type=ChoiceType.ITEM_SELECTION,
                    bbox=text.bbox,
                    priority=priority,
                    expected_outcome="purchase_item"
                ))
                
        # Generic menu options
        else:
            for text in choice_texts:
                choices.append(Choice(
                    text=text.text,
                    choice_type=ChoiceType.MENU_OPTION,
                    bbox=text.bbox,
                    priority=0.5,  # Default priority for menu options
                    expected_outcome="select_option"
                ))
                
        return choices
        
    def get_best_choice_action(self, choices: List[Choice]) -> List[str]:
        """Get the recommended action sequence for the best choice"""
        if not choices:
            return []
            
        # Get highest priority choice
        best_choice = max(choices, key=lambda x: x.priority)
        
        # Map choice type to action sequence
        if best_choice.choice_type == ChoiceType.YES_NO:
            return ["A"]  # Simple selection
            
        elif best_choice.choice_type == ChoiceType.POKEMON_SELECTION:
            return ["A"]  # Pokemon selection
            
        elif best_choice.choice_type == ChoiceType.ITEM_SELECTION:
            return ["A", "A"]  # Select and confirm
            
        elif best_choice.choice_type == ChoiceType.MENU_OPTION:
            return ["A"]  # Menu selection
            
        return []  # Default empty action sequence
        
    def filter_choices(self, choices: List[Choice], criteria: Dict[str, Any]) -> List[Choice]:
        """Filter choices based on given criteria"""
        filtered = choices.copy()
        
        if "type" in criteria:
            filtered = [c for c in filtered if c.choice_type == criteria["type"]]
            
        if "min_priority" in criteria:
            filtered = [c for c in filtered if c.priority >= criteria["min_priority"]]
            
        if "text_contains" in criteria:
            filtered = [c for c in filtered if criteria["text_contains"].lower() in c.text.lower()]
            
        if "outcome" in criteria:
            filtered = [c for c in filtered if c.expected_outcome == criteria["outcome"]]
            
        return filtered
