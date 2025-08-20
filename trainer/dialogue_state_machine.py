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

from vision.vision_processor import DetectedText, VisualContext
from core.semantic_context_system import SemanticContextSystem, GameContext, DialogueIntent


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
