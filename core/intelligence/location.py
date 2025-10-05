"""
Location Intelligence Module

Provides location analysis and strategic recommendations for Pokemon Crystal.
"""

from typing import Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum, auto
import logging

from environments.state.analyzer import GamePhase


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
    phase: GamePhase
    location_type: LocationType
    location_name: str
    immediate_goals: List[str]
    strategic_goals: List[str]
    health_status: str
    party_status: str
    recommended_actions: List[str]
    urgency_level: int  # 1-5, higher means more urgent action needed


# Backward-compatibility alias expected by other modules/tests
GameContext = IntelligenceGameContext


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
