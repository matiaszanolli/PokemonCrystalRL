#!/usr/bin/env python3
"""
Core Game Intelligence Module for Pokemon Crystal

This module provides high-level game understanding capabilities:
- Location context analysis
- Progress tracking and goal setting
- Battle strategy
- Multi-step action planning

The goal is to give the AI deeper understanding of Pokemon mechanics
beyond just reacting to current screen state.
"""

from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum, auto
import logging

from .memory_map import (
    IMPORTANT_LOCATIONS, 
    POKEMON_SPECIES, 
    BADGE_MASKS, 
    STATUS_CONDITIONS,
    get_badges_earned
)

class IntelligenceGamePhase(Enum):
    """High-level game progression phases for intelligence module"""
    TUTORIAL = auto()           # Getting first Pokemon
    EARLY_EXPLORATION = auto()  # First routes and towns
    GYM_CHALLENGE = auto()      # Preparing for/challenging gyms
    MID_GAME = auto()          # Multiple badges, exploring Johto
    LATE_GAME = auto()         # Elite Four preparation
    POST_GAME = auto()         # Kanto region

class LocationType(Enum):
    """Types of locations with different strategic contexts"""
    TOWN = auto()              # Safe areas with services
    ROUTE = auto()             # Wild Pokemon and trainers
    GYM = auto()              # Gym leader challenges
    POKEMON_CENTER = auto()    # Healing and PC access
    POKEMON_LAB = auto()       # Research and starter Pokemon
    CAVE = auto()             # Underground exploration
    FOREST = auto()           # Special wild Pokemon areas
    UNKNOWN = auto()          # Unrecognized location

@dataclass
class IntelligenceGameContext:
    """Rich context about current game situation for intelligence module"""
    phase: IntelligenceGamePhase
    location_type: LocationType
    location_name: str
    immediate_goals: List[str]
    strategic_goals: List[str]
    health_status: str
    party_status: str
    recommended_actions: List[str]
    urgency_level: int  # 1-5, higher means more urgent action needed

@dataclass
class ActionPlan:
    """Multi-step action plan"""
    goal: str
    steps: List[str]
    priority: int  # Higher = more important
    estimated_actions: int

class LocationAnalyzer:
    """Analyzes location context and determines appropriate strategies"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Location type mappings (these should be refined based on actual map IDs)
        self.location_types = {
            1: (LocationType.TOWN, "New Bark Town"),
            2: (LocationType.ROUTE, "Route 29"),
            3: (LocationType.TOWN, "Cherrygrove City"),
            4: (LocationType.ROUTE, "Route 30"),
            5: (LocationType.ROUTE, "Route 31"),
            6: (LocationType.TOWN, "Violet City"),
            7: (LocationType.GYM, "Sprout Tower"),
            8: (LocationType.ROUTE, "Route 32"),
            9: (LocationType.CAVE, "Ruins of Alph"),
            10: (LocationType.CAVE, "Union Cave"),
            11: (LocationType.ROUTE, "Route 33"),
            12: (LocationType.TOWN, "Azalea Town"),
            13: (LocationType.CAVE, "Slowpoke Well"),
            14: (LocationType.FOREST, "Ilex Forest"),
            15: (LocationType.ROUTE, "Route 34"),
            16: (LocationType.TOWN, "Goldenrod City"),
            17: (LocationType.ROUTE, "National Park"),
            18: (LocationType.ROUTE, "Route 35"),
            19: (LocationType.ROUTE, "Route 36"),
            20: (LocationType.ROUTE, "Route 37"),
            21: (LocationType.TOWN, "Ecruteak City"),
        }
        
        # Pokemon Centers are typically in towns, but let's identify them specifically
        self.pokemon_center_maps = {1, 3, 6, 12, 16, 21}  # Towns with Pokemon Centers
        
        # Gym locations
        self.gym_locations = {
            6: "Violet Gym",      # Falkner (Flying)
            12: "Azalea Gym",     # Bugsy (Bug)  
            16: "Goldenrod Gym",  # Whitney (Normal)
            21: "Ecruteak Gym",   # Morty (Ghost)
            # Add more as needed
        }
    
    def analyze_location(self, game_state: Dict) -> Tuple[LocationType, str]:
        """Analyze current location and return type and name"""
        map_id = game_state.get('player_map', 0)
        
        if map_id in self.location_types:
            return self.location_types[map_id]
        else:
            return LocationType.UNKNOWN, f"Unknown Location {map_id}"
    
    def get_location_strategy(self, location_type: LocationType, game_state: Dict) -> List[str]:
        """Get recommended strategies for this location type"""
        strategies = []
        
        if location_type == LocationType.POKEMON_CENTER:
            if game_state.get('player_hp', 0) < game_state.get('player_max_hp', 1):
                strategies.append("Heal Pokemon at counter")
            strategies.append("Check PC for stored Pokemon")
        
        elif location_type == LocationType.TOWN:
            if game_state.get('player_hp', 0) < game_state.get('player_max_hp', 1) * 0.5:
                strategies.append("Find Pokemon Center to heal")
            strategies.append("Explore for items and NPCs")
            strategies.append("Look for gym if badges < expected")
            
        elif location_type == LocationType.ROUTE:
            strategies.append("Battle wild Pokemon for experience")
            strategies.append("Battle trainers for money and experience")
            strategies.append("Explore grass and hidden areas")
            
        elif location_type == LocationType.GYM:
            if game_state.get('player_hp', 0) > game_state.get('player_max_hp', 1) * 0.8:
                strategies.append("Challenge gym leader")
            else:
                strategies.append("Heal before gym challenge")
                
        elif location_type == LocationType.CAVE:
            strategies.append("Explore for rare Pokemon")
            strategies.append("Watch for items and hidden passages")
            
        return strategies

class ProgressTracker:
    """Tracks game progress and determines next objectives"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def get_game_phase(self, game_state: Dict) -> GamePhase:
        """Determine current game progression phase"""
        party_count = game_state.get('party_count', 0)
        badges_total = game_state.get('badges_total', 0)
        
        if party_count == 0:
            return GamePhase.TUTORIAL
        elif badges_total == 0 and party_count > 0:
            return GamePhase.EARLY_EXPLORATION
        elif badges_total < 4:
            return GamePhase.GYM_CHALLENGE
        elif badges_total < 8:
            return GamePhase.MID_GAME
        elif badges_total < 16:
            return GamePhase.LATE_GAME
        else:
            return GamePhase.POST_GAME
    
    def get_immediate_goals(self, game_state: Dict, location_type: LocationType) -> List[str]:
        """Get immediate, actionable goals"""
        goals = []
        phase = self.get_game_phase(game_state)
        party_count = game_state.get('party_count', 0)
        
        # Health is only a priority if we actually have Pokemon
        if party_count > 0:
            hp_ratio = game_state.get('player_hp', 0) / max(game_state.get('player_max_hp', 1), 1)
            if hp_ratio < 0.3:
                goals.append("URGENT: Heal Pokemon immediately")
            elif hp_ratio < 0.6:
                goals.append("Find healing when convenient")
        
        # Phase-specific goals
        if phase == GamePhase.TUTORIAL:
            goals.append("Get your first Pokemon from Professor Elm")
            
        elif phase == GamePhase.EARLY_EXPLORATION:
            goals.append("Level up your Pokemon to ~10")
            goals.append("Explore routes and catch more Pokemon")
            goals.append("Head to Violet City for first gym")
            
        elif phase == GamePhase.GYM_CHALLENGE:
            badges = game_state.get('badges_total', 0)
            goals.append(f"Prepare for gym #{badges + 1}")
            goals.append("Level Pokemon to ~15-20")
            
        return goals
    
    def get_strategic_goals(self, game_state: Dict) -> List[str]:
        """Get longer-term strategic goals"""
        goals = []
        phase = self.get_game_phase(game_state)
        
        if phase == GamePhase.TUTORIAL:
            goals.append("Complete Professor Elm's tasks")
            goals.append("Learn basic game mechanics")
            
        elif phase == GamePhase.EARLY_EXPLORATION:
            goals.append("Build a balanced party")
            goals.append("Earn first gym badge")
            
        elif phase == GamePhase.GYM_CHALLENGE:
            goals.append("Earn all 8 Johto badges")
            goals.append("Prepare for Elite Four")
            
        return goals

class BattleStrategy:
    """Intelligent battle decision making"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Type effectiveness (simplified - could be expanded)
        self.type_effectiveness = {
            # Format: (attacker_type, defender_type): multiplier
            ("WATER", "FIRE"): 2.0,
            ("FIRE", "GRASS"): 2.0,
            ("GRASS", "WATER"): 2.0,
            ("ELECTRIC", "WATER"): 2.0,
            ("ELECTRIC", "FLYING"): 2.0,
            # Add more type matchups as needed
        }
    
    def get_battle_strategy(self, game_state: Dict) -> str:
        """Get recommended battle strategy"""
        if not game_state.get('in_battle', 0):
            return "Not in battle"
        
        player_hp_ratio = game_state.get('player_hp', 0) / max(game_state.get('player_max_hp', 1), 1)
        enemy_level = game_state.get('enemy_level', 0)
        player_level = game_state.get('player_level', 0)
        
        # Critical health
        if player_hp_ratio < 0.2:
            return "CRITICAL: Use healing item or switch Pokemon"
        
        # Level disadvantage
        if enemy_level > player_level + 5:
            return "Consider switching Pokemon or using items"
        
        # Standard attack
        return "Attack with your best move"

class GameIntelligence:
    """Main game intelligence coordinator"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.location_analyzer = LocationAnalyzer()
        self.progress_tracker = ProgressTracker()
        self.battle_strategy = BattleStrategy()
    
    def analyze_game_context(self, game_state: Dict, screen_analysis: Dict) -> GameContext:
        """Perform comprehensive game analysis"""
        
        # Location analysis
        location_type, location_name = self.location_analyzer.analyze_location(game_state)
        
        # Game phase
        phase = self.progress_tracker.get_game_phase(game_state)
        
        # Goals
        immediate_goals = self.progress_tracker.get_immediate_goals(game_state, location_type)
        strategic_goals = self.progress_tracker.get_strategic_goals(game_state)
        
        # Health and party status
        party_count = game_state.get('party_count', 0)
        
        if party_count > 0:
            hp_ratio = game_state.get('player_hp', 0) / max(game_state.get('player_max_hp', 1), 1)
            health_status = "Critical" if hp_ratio < 0.3 else "Low" if hp_ratio < 0.6 else "Good"
        else:
            hp_ratio = 1.0  # No Pokemon = no health concerns
            health_status = "No Pokemon"
            
        party_status = f"{party_count} Pokemon" + (" (need more!)" if party_count < 2 else "")
        
        # Recommended actions based on context
        recommended_actions = self.location_analyzer.get_location_strategy(location_type, game_state)
        
        # Battle strategy if in battle
        if game_state.get('in_battle', 0):
            battle_advice = self.battle_strategy.get_battle_strategy(game_state)
            recommended_actions.insert(0, f"Battle: {battle_advice}")
        
        # Urgency level
        urgency = 1
        if party_count == 0:
            urgency = 4  # High priority: get first Pokemon
        elif party_count > 0 and hp_ratio < 0.2:
            urgency = 5  # Critical: heal immediately
        elif party_count > 0 and hp_ratio < 0.4:
            urgency = 3  # Medium: consider healing
        
        return GameContext(
            phase=phase,
            location_type=location_type,
            location_name=location_name,
            immediate_goals=immediate_goals,
            strategic_goals=strategic_goals,
            health_status=health_status,
            party_status=party_status,
            recommended_actions=recommended_actions,
            urgency_level=urgency
        )
    
    def get_action_plan(self, game_context: GameContext, game_state: Dict) -> List[ActionPlan]:
        """Generate multi-step action plans"""
        plans = []
        
        # Emergency healing plan (only if we have Pokemon)
        party_count = game_state.get('party_count', 0)
        if game_context.urgency_level >= 4 and party_count > 0:
            if game_context.location_type == LocationType.POKEMON_CENTER:
                plans.append(ActionPlan(
                    goal="Emergency heal at Pokemon Center",
                    steps=["Walk to counter", "Interact with nurse", "Confirm healing"],
                    priority=10,
                    estimated_actions=5
                ))
            else:
                plans.append(ActionPlan(
                    goal="Find Pokemon Center for emergency healing",
                    steps=["Open map", "Navigate to nearest town", "Find Pokemon Center"],
                    priority=9,
                    estimated_actions=20
                ))
        
        # Tutorial plan
        if game_context.phase == GamePhase.TUTORIAL:
            plans.append(ActionPlan(
                goal="Get starter Pokemon from Professor Elm",
                steps=["Navigate to lab", "Talk to Professor Elm", "Choose starter"],
                priority=8,
                estimated_actions=15
            ))
        
        # Gym challenge plan
        elif game_context.phase == GamePhase.GYM_CHALLENGE:
            if game_context.location_type == LocationType.GYM:
                plans.append(ActionPlan(
                    goal="Challenge gym leader",
                    steps=["Navigate through gym", "Battle gym trainers", "Challenge leader"],
                    priority=7,
                    estimated_actions=30
                ))
        
        return sorted(plans, key=lambda x: x.priority, reverse=True)
    
    def get_contextual_advice(self, game_context: GameContext, recent_actions: List[str]) -> str:
        """Get human-readable advice for current situation"""
        advice_parts = []
        
        # Phase context
        advice_parts.append(f"Phase: {game_context.phase.name}")
        advice_parts.append(f"Location: {game_context.location_name} ({game_context.location_type.name})")
        
        # Urgent matters
        if game_context.urgency_level >= 4:
            advice_parts.append(f"⚠️ URGENT ({game_context.urgency_level}/5): {game_context.immediate_goals[0]}")
        
        # Immediate goals
        if game_context.immediate_goals:
            advice_parts.append(f"Next: {game_context.immediate_goals[0]}")
        
        # Strategic context
        if game_context.strategic_goals:
            advice_parts.append(f"Goal: {game_context.strategic_goals[0]}")
        
        # Recent action analysis
        if recent_actions:
            recent_str = " → ".join(recent_actions[-3:])
            if "START" in recent_actions[-2:] and "b" not in recent_actions[-1:]:
                advice_parts.append("⚠️ Recently opened menu - consider exiting with 'b'")
            elif recent_actions[-1:] == recent_actions[-2:-1]:
                advice_parts.append("⚠️ Repeated action detected - may be stuck")
        
        return " | ".join(advice_parts)
