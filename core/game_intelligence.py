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

from environments.state.analyzer import GamePhase, SituationCriticality
from environments.state.memory_map import (
    IMPORTANT_LOCATIONS, 
    POKEMON_SPECIES, 
    BADGE_MASKS, 
    STATUS_CONDITIONS,
    get_badges_earned
)

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
            return GamePhase.EARLY_GAME
        elif badges_total < 4:
            return GamePhase.GYM_BATTLES
        elif badges_total < 8:
            return GamePhase.LATE_GAME
        elif badges_total < 16:
            return GamePhase.POST_GAME
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
            
        elif phase == GamePhase.EARLY_GAME:
            goals.append("Level up your Pokemon to ~10")
            goals.append("Explore routes and catch more Pokemon")
            goals.append("Head to Violet City for first gym")
            
        elif phase == GamePhase.GYM_BATTLES:
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
            
        elif phase == GamePhase.EARLY_GAME:
            goals.append("Build a balanced party")
            goals.append("Earn first gym badge")
            
        elif phase == GamePhase.GYM_BATTLES:
            goals.append("Earn all 8 Johto badges")
            goals.append("Prepare for Elite Four")
            
        return goals

class BattleStrategy:
    """Intelligent battle decision making with comprehensive type effectiveness and move selection"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Comprehensive type effectiveness chart
        self.type_effectiveness = {
            # Water type matchups
            ("WATER", "FIRE"): 2.0, ("WATER", "GROUND"): 2.0, ("WATER", "ROCK"): 2.0,
            ("WATER", "WATER"): 0.5, ("WATER", "GRASS"): 0.5, ("WATER", "DRAGON"): 0.5,

            # Fire type matchups
            ("FIRE", "GRASS"): 2.0, ("FIRE", "ICE"): 2.0, ("FIRE", "BUG"): 2.0, ("FIRE", "STEEL"): 2.0,
            ("FIRE", "FIRE"): 0.5, ("FIRE", "WATER"): 0.5, ("FIRE", "ROCK"): 0.5, ("FIRE", "DRAGON"): 0.5,

            # Grass type matchups
            ("GRASS", "WATER"): 2.0, ("GRASS", "GROUND"): 2.0, ("GRASS", "ROCK"): 2.0,
            ("GRASS", "FIRE"): 0.5, ("GRASS", "GRASS"): 0.5, ("GRASS", "POISON"): 0.5,
            ("GRASS", "FLYING"): 0.5, ("GRASS", "BUG"): 0.5, ("GRASS", "DRAGON"): 0.5, ("GRASS", "STEEL"): 0.5,

            # Electric type matchups
            ("ELECTRIC", "WATER"): 2.0, ("ELECTRIC", "FLYING"): 2.0,
            ("ELECTRIC", "ELECTRIC"): 0.5, ("ELECTRIC", "GRASS"): 0.5, ("ELECTRIC", "DRAGON"): 0.5,
            ("ELECTRIC", "GROUND"): 0.0,  # No effect

            # Psychic type matchups
            ("PSYCHIC", "FIGHTING"): 2.0, ("PSYCHIC", "POISON"): 2.0,
            ("PSYCHIC", "PSYCHIC"): 0.5, ("PSYCHIC", "STEEL"): 0.5,
            ("PSYCHIC", "DARK"): 0.0,  # No effect

            # Fighting type matchups
            ("FIGHTING", "NORMAL"): 2.0, ("FIGHTING", "ICE"): 2.0, ("FIGHTING", "ROCK"): 2.0,
            ("FIGHTING", "DARK"): 2.0, ("FIGHTING", "STEEL"): 2.0,
            ("FIGHTING", "POISON"): 0.5, ("FIGHTING", "FLYING"): 0.5, ("FIGHTING", "PSYCHIC"): 0.5,
            ("FIGHTING", "BUG"): 0.5, ("FIGHTING", "GHOST"): 0.0,  # No effect

            # Flying type matchups
            ("FLYING", "ELECTRIC"): 0.5, ("FLYING", "ROCK"): 0.5, ("FLYING", "STEEL"): 0.5,
            ("FLYING", "GRASS"): 2.0, ("FLYING", "FIGHTING"): 2.0, ("FLYING", "BUG"): 2.0,

            # Add more comprehensive type matchups as needed
        }

        # Status condition priorities
        self.status_conditions = {
            'sleep': {'priority': 3, 'action': 'Wake up or switch'},
            'poison': {'priority': 2, 'action': 'Use antidote or heal'},
            'burn': {'priority': 2, 'action': 'Use burn heal'},
            'freeze': {'priority': 3, 'action': 'Use fire move or switch'},
            'paralysis': {'priority': 1, 'action': 'Use paralyze heal if needed'}
        }

        # Move categories and priorities
        self.move_categories = {
            'attack': {'priority': 3, 'description': 'Direct damage moves'},
            'status': {'priority': 1, 'description': 'Status-affecting moves'},
            'stat_boost': {'priority': 2, 'description': 'Stat-boosting moves'},
            'healing': {'priority': 4, 'description': 'HP recovery moves'}
        }

    def get_type_effectiveness(self, attacker_type: str, defender_type: str) -> float:
        """Get type effectiveness multiplier"""
        return self.type_effectiveness.get((attacker_type.upper(), defender_type.upper()), 1.0)

    def analyze_battle_situation(self, game_state: Dict) -> Dict[str, Any]:
        """Comprehensive battle situation analysis"""
        if not game_state.get('in_battle', 0):
            return {'in_battle': False}

        player_hp = game_state.get('player_hp', 0)
        player_max_hp = game_state.get('player_max_hp', 1)
        player_hp_ratio = player_hp / max(player_max_hp, 1)

        enemy_level = game_state.get('enemy_level', 0)
        player_level = game_state.get('player_level', 0)
        level_difference = enemy_level - player_level

        # Determine battle phase
        if player_hp_ratio > 0.7:
            battle_phase = "aggressive"
        elif player_hp_ratio > 0.3:
            battle_phase = "cautious"
        else:
            battle_phase = "defensive"

        # Calculate strategic metrics
        level_advantage = "enemy" if level_difference > 3 else "player" if level_difference < -3 else "even"

        return {
            'in_battle': True,
            'player_hp_ratio': player_hp_ratio,
            'level_difference': level_difference,
            'level_advantage': level_advantage,
            'battle_phase': battle_phase,
            'player_species': game_state.get('player_species', 0),
            'enemy_species': game_state.get('enemy_species', 0),
            'recommended_priority': self._get_action_priority(player_hp_ratio, level_difference)
        }

    def _get_action_priority(self, hp_ratio: float, level_diff: int) -> str:
        """Determine action priority based on battle state"""
        if hp_ratio < 0.15:
            return "emergency_heal"
        elif hp_ratio < 0.3 and level_diff > 5:
            return "switch_or_heal"
        elif level_diff > 8:
            return "consider_flee"
        elif hp_ratio > 0.8 and level_diff < -2:
            return "aggressive_attack"
        else:
            return "standard_attack"

    def get_battle_strategy(self, game_state: Dict) -> str:
        """Get intelligent battle strategy with move selection"""
        analysis = self.analyze_battle_situation(game_state)

        if not analysis['in_battle']:
            return "Not in battle"

        hp_ratio = analysis['player_hp_ratio']
        priority = analysis['recommended_priority']
        battle_phase = analysis['battle_phase']
        level_advantage = analysis['level_advantage']

        # Emergency situations
        if priority == "emergency_heal":
            return "EMERGENCY: Use healing item immediately or switch Pokemon"

        if priority == "consider_flee":
            return "RETREAT: Enemy too strong - consider fleeing or switching"

        # Status condition handling
        # Note: Status condition detection would require additional memory reading
        # For now, we'll focus on HP and level-based strategy

        # Strategic recommendations based on battle phase
        strategies = []

        if battle_phase == "aggressive":
            if level_advantage == "player":
                strategies.append("Use strongest attack move")
                strategies.append("Consider stat-boosting moves for setup")
            else:
                strategies.append("Use super-effective moves if available")
                strategies.append("Focus on consistent damage")

        elif battle_phase == "cautious":
            strategies.append("Use reliable moves to finish the battle")
            if level_advantage == "enemy":
                strategies.append("Consider defensive moves or healing")
            else:
                strategies.append("Maintain pressure with attacks")

        elif battle_phase == "defensive":
            strategies.append("PRIORITY: Heal or use defensive moves")
            strategies.append("Consider switching to a healthier Pokemon")
            if game_state.get('party_count', 1) > 1:
                strategies.append("Switch Pokemon if available")

        # Add type effectiveness advice (requires knowing Pokemon types)
        player_species = analysis.get('player_species', 0)
        enemy_species = analysis.get('enemy_species', 0)

        if player_species and enemy_species:
            # This would require a Pokemon species -> type mapping
            # For now, provide general type advice
            strategies.append("Check move types for effectiveness")

        return f"Battle Phase: {battle_phase.title()} | " + " | ".join(strategies[:2])

    def recommend_move_selection(self, game_state: Dict, available_moves: List[str] = None) -> Dict[str, Any]:
        """Recommend specific move selection (if move data available)"""
        analysis = self.analyze_battle_situation(game_state)

        if not analysis['in_battle']:
            return {'recommendation': 'Not in battle'}

        recommendations = {
            'primary_strategy': analysis['recommended_priority'],
            'battle_phase': analysis['battle_phase'],
            'suggested_move_types': [],
            'avoid_moves': []
        }

        # Move type recommendations based on situation
        if analysis['battle_phase'] == "aggressive":
            recommendations['suggested_move_types'] = ['attack', 'stat_boost']
        elif analysis['battle_phase'] == "cautious":
            recommendations['suggested_move_types'] = ['attack']
        else:  # defensive
            recommendations['suggested_move_types'] = ['healing', 'status']
            recommendations['avoid_moves'] = ['risky_attack', 'stat_boost']

        return recommendations

class InventoryManager:
    """Intelligent inventory and item management for Pokemon Crystal"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Item categories and their strategic value
        self.item_categories = {
            'healing': {
                'potion': {'hp_restore': 20, 'priority': 3, 'use_threshold': 0.5},
                'super_potion': {'hp_restore': 50, 'priority': 4, 'use_threshold': 0.4},
                'hyper_potion': {'hp_restore': 200, 'priority': 5, 'use_threshold': 0.3},
                'full_heal': {'status_cure': 'all', 'priority': 4, 'use_condition': 'status_ailment'}
            },
            'pokeballs': {
                'pokeball': {'catch_rate': 1.0, 'priority': 2, 'use_condition': 'wild_encounter'},
                'great_ball': {'catch_rate': 1.5, 'priority': 3, 'use_condition': 'wild_encounter'},
                'ultra_ball': {'catch_rate': 2.0, 'priority': 4, 'use_condition': 'wild_encounter'}
            },
            'battle_items': {
                'x_attack': {'stat_boost': 'attack', 'priority': 2, 'use_condition': 'tough_battle'},
                'x_defend': {'stat_boost': 'defense', 'priority': 2, 'use_condition': 'tough_battle'},
                'x_speed': {'stat_boost': 'speed', 'priority': 2, 'use_condition': 'tough_battle'}
            },
            'key_items': {
                'bicycle': {'functionality': 'fast_travel', 'priority': 5},
                'surf_hm': {'functionality': 'water_travel', 'priority': 5},
                'cut_hm': {'functionality': 'obstacle_removal', 'priority': 4}
            }
        }

        # Item usage strategies based on game state
        self.usage_strategies = {
            'battle': {
                'hp_critical': 'Use strongest healing item immediately',
                'hp_low': 'Use appropriate healing item',
                'status_ailment': 'Use status cure item',
                'tough_opponent': 'Consider battle enhancement items'
            },
            'exploration': {
                'wild_encounter': 'Use pokeball if Pokemon is valuable',
                'low_health': 'Heal before entering dangerous areas',
                'obstacle': 'Use appropriate HM or key item'
            }
        }

    def analyze_inventory_needs(self, game_state: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze current inventory needs based on game state and context"""
        analysis = {
            'immediate_needs': [],
            'recommended_items': [],
            'item_usage_advice': [],
            'inventory_priorities': []
        }

        # Determine current situation
        in_battle = game_state.get('in_battle', False)
        party_count = game_state.get('party_count', 0)

        if party_count > 0:
            hp_ratio = game_state.get('player_hp', 0) / max(game_state.get('player_max_hp', 1), 1)
        else:
            hp_ratio = 1.0

        # Health-based recommendations
        if party_count > 0:
            if hp_ratio < 0.2:
                analysis['immediate_needs'].append('emergency_healing')
                analysis['item_usage_advice'].append('Use strongest healing item immediately')
            elif hp_ratio < 0.5:
                analysis['recommended_items'].append('healing_item')
                analysis['item_usage_advice'].append('Consider using healing item')

        # Battle-specific item needs
        if in_battle:
            enemy_level = game_state.get('enemy_level', 0)
            player_level = game_state.get('player_level', 0)

            if enemy_level > player_level + 5:
                analysis['recommended_items'].append('battle_enhancement')
                analysis['item_usage_advice'].append('Consider using stat-boosting items')

        # Exploration needs
        current_state = context.get('detected_state', 'unknown')
        if current_state == 'overworld':
            badges_count = game_state.get('badges_total', 0)

            # Early game priorities
            if badges_count < 2:
                analysis['inventory_priorities'] = [
                    'Stock up on pokeballs for catching Pokemon',
                    'Carry healing items for long routes',
                    'Get key items from NPCs'
                ]
            else:
                analysis['inventory_priorities'] = [
                    'Maintain healing item supply',
                    'Carry varied pokeball types',
                    'Collect HMs for navigation'
                ]

        return analysis

    def recommend_item_usage(self, game_state: Dict[str, Any], held_item: int = None) -> Dict[str, Any]:
        """Recommend specific item usage based on current situation"""
        recommendations = {
            'should_use_item': False,
            'item_type': None,
            'urgency': 'normal',
            'reasoning': ''
        }

        party_count = game_state.get('party_count', 0)
        if party_count == 0:
            return recommendations

        hp_ratio = game_state.get('player_hp', 0) / max(game_state.get('player_max_hp', 1), 1)
        in_battle = game_state.get('in_battle', False)

        # Critical health situation
        if hp_ratio < 0.15:
            recommendations.update({
                'should_use_item': True,
                'item_type': 'healing',
                'urgency': 'critical',
                'reasoning': 'Pokemon health critically low - immediate healing required'
            })

        # Low health in battle
        elif in_battle and hp_ratio < 0.3:
            recommendations.update({
                'should_use_item': True,
                'item_type': 'healing',
                'urgency': 'high',
                'reasoning': 'Low health in battle - heal to continue fighting effectively'
            })

        # Preventive healing before tough encounters
        elif not in_battle and hp_ratio < 0.6:
            enemy_level = game_state.get('enemy_level', 0)
            player_level = game_state.get('player_level', 0)

            if enemy_level > player_level + 3:
                recommendations.update({
                    'should_use_item': True,
                    'item_type': 'healing',
                    'urgency': 'normal',
                    'reasoning': 'Heal before tough encounter to maximize chances'
                })

        return recommendations

    def get_optimal_pokeball(self, game_state: Dict[str, Any], wild_pokemon_info: Dict[str, Any] = None) -> str:
        """Recommend optimal pokeball type for wild encounters"""
        if not wild_pokemon_info:
            return "Use standard pokeball"

        enemy_level = game_state.get('enemy_level', 0)
        player_level = game_state.get('player_level', 0)
        enemy_hp_ratio = wild_pokemon_info.get('hp_ratio', 1.0)

        # Base recommendation on pokemon strength and rarity
        if enemy_level > player_level + 10:
            return "Use ultra ball - strong pokemon"
        elif enemy_level > player_level + 5 or enemy_hp_ratio > 0.7:
            return "Use great ball - moderately strong pokemon"
        else:
            return "Use pokeball - standard catch attempt"

    def evaluate_held_item_strategy(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate held item strategy for Pokemon"""
        held_item = game_state.get('player_held_item', 0)
        player_level = game_state.get('player_level', 0)

        strategy = {
            'current_item': held_item,
            'recommendation': 'keep',
            'alternative_items': [],
            'reasoning': ''
        }

        # Early game: prioritize healing items
        if player_level < 15:
            strategy.update({
                'recommendation': 'equip_berry',
                'alternative_items': ['oran_berry', 'pecha_berry'],
                'reasoning': 'Early game benefits from healing/status cure items'
            })

        # Mid game: consider stat-boosting items
        elif player_level < 40:
            strategy.update({
                'recommendation': 'consider_stat_items',
                'alternative_items': ['choice_band', 'leftovers'],
                'reasoning': 'Mid game can leverage stat-boosting held items'
            })

        return strategy

class GameIntelligence:
    """Main game intelligence coordinator"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.location_analyzer = LocationAnalyzer()
        self.progress_tracker = ProgressTracker()
        self.battle_strategy = BattleStrategy()
        self.inventory_manager = InventoryManager()
    
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

        # Inventory and item recommendations
        inventory_analysis = self.inventory_manager.analyze_inventory_needs(game_state, screen_analysis)
        if inventory_analysis['immediate_needs']:
            for need in inventory_analysis['immediate_needs']:
                recommended_actions.insert(0 if need == 'emergency_healing' else 1, f"Item: {need}")

        # Add item usage advice to recommendations
        item_recommendation = self.inventory_manager.recommend_item_usage(game_state)
        if item_recommendation['should_use_item']:
            urgency_text = f"({item_recommendation['urgency']})" if item_recommendation['urgency'] != 'normal' else ''
            recommended_actions.insert(0, f"Use {item_recommendation['item_type']} item {urgency_text}")

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
        elif game_context.phase == GamePhase.GYM_BATTLES:
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
