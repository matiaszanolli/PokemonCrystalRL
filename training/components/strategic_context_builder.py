"""
Strategic Context Builder - Enhanced context generation for intelligent LLM decisions

Provides rich strategic information to help the LLM make better informed decisions
based on game progress, objectives, and current situation.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from collections import deque, defaultdict
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class GameObjective:
    """Represents a current game objective."""
    name: str
    description: str
    priority: int  # 1-10, higher = more important
    progress: float  # 0.0-1.0


@dataclass
class QuestTracker:
    """Tracks quest progress and objectives"""
    quest_id: str
    name: str
    description: str
    location: str
    prerequisites: List[str]
    rewards: List[str]
    status: str  # 'available', 'active', 'completed', 'failed'
    progress_markers: Dict[str, bool]
    estimated_actions: int


@dataclass
class LocationInfo:
    """Enhanced location information with strategic context"""
    map_id: int
    name: str
    location_type: str  # 'town', 'route', 'gym', 'center', 'cave', 'special'
    strategic_value: int  # 1-10, importance for progression
    available_services: List[str]  # 'healing', 'shopping', 'gym', 'pc'
    key_npcs: List[str]
    quests_available: List[str]
    pokemon_encounters: List[str]
    items_available: List[str]
    required_hms: List[str]  # HMs needed to access
    connections: List[int]  # Connected map IDs


class StrategicContextBuilder:
    """Builds enhanced strategic context for LLM decision making."""

    def __init__(self):
        self.logger = logging.getLogger("StrategicContextBuilder")

        # Memory for strategic analysis
        self.position_history = deque(maxlen=50)  # Last 50 positions
        self.action_history = deque(maxlen=20)   # Last 20 actions
        self.reward_history = deque(maxlen=100)  # Last 100 rewards

        # Enhanced map knowledge and quest tracking
        self.visited_locations = set()
        self.location_database = self._initialize_location_database()
        self.quest_tracker = self._initialize_quest_system()
        self.navigation_history = deque(maxlen=100)  # Track movement patterns
        self.location_visit_count = defaultdict(int)
        self.discovered_connections = defaultdict(list)

        # Strategic mapping
        self.strategic_routes = self._initialize_strategic_routes()
        self.progression_checkpoints = self._initialize_progression_checkpoints()

        # Progress tracking
        self.last_progress_check = {
            'badges': 0,
            'level': 0,
            'money': 0,
            'party_count': 0
        }

        self.logger.info("Strategic context builder initialized")

    def build_enhanced_context(self,
                             game_state: Dict[str, Any],
                             action_count: int,
                             recent_rewards: List[float] = None) -> Dict[str, Any]:
        """Build comprehensive strategic context for LLM.

        Args:
            game_state: Current game state
            action_count: Current action count
            recent_rewards: Recent reward history

        Returns:
            Dict: Enhanced context with strategic information
        """
        try:
            # Update our tracking
            self._update_tracking(game_state, action_count, recent_rewards)

            # Build comprehensive context
            context = {
                # Basic game state
                'current_state': self._determine_game_situation(game_state),
                'position': self._get_position_context(game_state),
                'progress': self._get_progress_context(game_state),

                # Strategic information
                'current_objective': self._get_current_objective(game_state),
                'situation_analysis': self._analyze_current_situation(game_state),
                'recommended_actions': self._get_recommended_actions(game_state),

                # Performance insights
                'performance': self._get_performance_insights(),
                'stuck_analysis': self._analyze_stuck_patterns(),

                # Meta information
                'action_count': action_count,
                'confidence_factors': self._get_confidence_factors(game_state)
            }

            return context

        except Exception as e:
            self.logger.error(f"Failed to build enhanced context: {e}")
            return self._get_fallback_context(game_state, action_count)

    def _update_tracking(self, game_state: Dict[str, Any], action_count: int, recent_rewards: List[float]):
        """Update internal tracking for analysis."""
        # Track position
        position = (game_state.get('player_x', 0), game_state.get('player_y', 0), game_state.get('player_map', 0))
        self.position_history.append(position)

        # Track visited locations
        map_id = game_state.get('player_map', 0)
        if map_id > 0:
            self.visited_locations.add(map_id)

        # Track rewards
        if recent_rewards:
            self.reward_history.extend(recent_rewards[-10:])  # Last 10 rewards

    def _determine_game_situation(self, game_state: Dict[str, Any]) -> str:
        """Determine current game situation."""
        if game_state.get('in_battle', False):
            return 'battle'
        elif game_state.get('party_count', 0) == 0:
            return 'no_pokemon'
        elif game_state.get('badges', 0) == 0:
            return 'early_game'
        elif game_state.get('badges', 0) < 3:
            return 'beginning_journey'
        else:
            return 'progressing'

    def _get_position_context(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """Get position and location context."""
        map_id = game_state.get('player_map', 0)
        x, y = game_state.get('player_x', 0), game_state.get('player_y', 0)

        return {
            'map_id': map_id,
            'coordinates': (x, y),
            'location_name': self.important_locations.get(map_id, f'unknown_map_{map_id}'),
            'is_new_location': map_id not in self.visited_locations,
            'visited_maps_count': len(self.visited_locations)
        }

    def _get_progress_context(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """Get progress context and recent changes."""
        current_progress = {
            'badges': game_state.get('badges_count', game_state.get('badges', 0)),
            'level': game_state.get('player_level', 0),
            'money': game_state.get('money', 0),
            'party_count': game_state.get('party_count', 0),
            'hp_percentage': game_state.get('health_percentage', 0)
        }

        # Detect recent progress
        progress_changes = {}
        for key, current_value in current_progress.items():
            if key in self.last_progress_check:
                change = current_value - self.last_progress_check[key]
                if change != 0:
                    progress_changes[f'{key}_change'] = change

        # Update tracking
        self.last_progress_check.update(current_progress)

        return {
            **current_progress,
            'recent_changes': progress_changes,
            'has_pokemon': current_progress['party_count'] > 0,
            'health_status': self._get_health_status(current_progress['hp_percentage'])
        }

    def _get_health_status(self, hp_percentage: float) -> str:
        """Get health status description."""
        if hp_percentage <= 0:
            return 'no_pokemon_or_fainted'
        elif hp_percentage < 25:
            return 'critical'
        elif hp_percentage < 50:
            return 'low'
        elif hp_percentage < 75:
            return 'moderate'
        else:
            return 'healthy'

    def _get_current_objective(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """Determine current strategic objective."""
        badges = game_state.get('badges_count', game_state.get('badges', 0))
        party_count = game_state.get('party_count', 0)
        hp_percentage = game_state.get('health_percentage', 0)
        money = game_state.get('money', 0)
        map_id = game_state.get('player_map', 0)

        # Priority objectives based on game state
        if party_count == 0:
            return {
                'primary': 'get_starter_pokemon',
                'description': 'Visit Professor Elm to get your first Pokemon',
                'urgency': 'critical',
                'suggested_actions': ['navigate to new_bark_town', 'enter elm_lab', 'talk to elm']
            }

        elif hp_percentage < 25 and party_count > 0:
            return {
                'primary': 'heal_pokemon',
                'description': 'Find Pokemon Center to heal your Pokemon',
                'urgency': 'high',
                'suggested_actions': ['find pokemon_center', 'use healing machine']
            }

        elif badges == 0:
            return {
                'primary': 'reach_first_gym',
                'description': 'Travel to Violet City and challenge Falkner',
                'urgency': 'medium',
                'suggested_actions': ['explore routes', 'level up pokemon', 'find violet_city']
            }

        elif badges < 8:
            return {
                'primary': 'continue_gym_challenge',
                'description': f'Challenge the next gym leader ({badges + 1}/8)',
                'urgency': 'medium',
                'suggested_actions': ['prepare team', 'stock items', 'find next gym']
            }

        else:
            return {
                'primary': 'elite_four_preparation',
                'description': 'Prepare for Elite Four challenge',
                'urgency': 'low',
                'suggested_actions': ['train team', 'stock items', 'reach indigo_plateau']
            }

    def _analyze_current_situation(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze current situation for strategic insights."""
        analysis = {
            'exploration_opportunity': False,
            'combat_readiness': 'unknown',
            'resource_status': 'adequate',
            'immediate_concerns': []
        }

        # Check for exploration opportunities
        map_id = game_state.get('player_map', 0)
        if map_id not in self.visited_locations:
            analysis['exploration_opportunity'] = True

        # Assess combat readiness
        party_count = game_state.get('party_count', 0)
        hp_percentage = game_state.get('health_percentage', 0)

        if party_count == 0:
            analysis['combat_readiness'] = 'no_pokemon'
        elif hp_percentage < 25:
            analysis['combat_readiness'] = 'poor'
        elif hp_percentage < 75:
            analysis['combat_readiness'] = 'moderate'
        else:
            analysis['combat_readiness'] = 'good'

        # Identify immediate concerns
        if party_count == 0:
            analysis['immediate_concerns'].append('no_pokemon')
        if hp_percentage < 25:
            analysis['immediate_concerns'].append('low_health')
        if game_state.get('money', 0) == 0:
            analysis['immediate_concerns'].append('no_money')

        return analysis

    def _get_recommended_actions(self, game_state: Dict[str, Any]) -> List[str]:
        """Get recommended actions based on current situation."""
        recommendations = []

        # Get current objective
        objective = self._get_current_objective(game_state)

        # Add objective-based recommendations
        recommendations.extend(objective.get('suggested_actions', []))

        # Add situation-specific recommendations
        if self._is_stuck():
            recommendations.append('try_different_direction')
            recommendations.append('use_random_movement')

        return recommendations[:5]  # Top 5 recommendations

    def _get_performance_insights(self) -> Dict[str, Any]:
        """Get performance insights from recent activity."""
        insights = {
            'recent_reward_trend': 'stable',
            'exploration_rate': 0.0,
            'action_diversity': 0.0
        }

        # Analyze recent rewards
        if len(self.reward_history) > 10:
            recent_rewards = list(self.reward_history)[-10:]
            avg_reward = sum(recent_rewards) / len(recent_rewards)

            if avg_reward > 0.1:
                insights['recent_reward_trend'] = 'improving'
            elif avg_reward < -0.1:
                insights['recent_reward_trend'] = 'declining'
            else:
                insights['recent_reward_trend'] = 'stable'

        # Calculate exploration rate (new locations visited)
        if len(self.position_history) > 1:
            unique_positions = len(set(self.position_history))
            insights['exploration_rate'] = unique_positions / len(self.position_history)

        return insights

    def _analyze_stuck_patterns(self) -> Dict[str, Any]:
        """Analyze if agent appears to be stuck."""
        stuck_analysis = {
            'is_stuck': False,
            'stuck_type': 'none',
            'stuck_duration': 0,
            'recommended_recovery': []
        }

        if self._is_stuck():
            stuck_analysis['is_stuck'] = True
            stuck_analysis['stuck_type'] = self._get_stuck_type()
            stuck_analysis['recommended_recovery'] = self._get_stuck_recovery_actions()

        return stuck_analysis

    def _is_stuck(self) -> bool:
        """Check if agent appears to be stuck."""
        if len(self.position_history) < 10:
            return False

        # Check if positions are too similar
        recent_positions = list(self.position_history)[-10:]
        unique_positions = len(set(recent_positions))

        # If less than 3 unique positions in last 10 actions, consider stuck
        return unique_positions < 3

    def _get_stuck_type(self) -> str:
        """Determine type of stuck pattern."""
        if len(self.action_history) < 5:
            return 'unknown'

        recent_actions = list(self.action_history)[-5:]

        # Check for action repetition
        if len(set(recent_actions)) == 1:
            return 'action_loop'
        elif len(set(recent_actions)) == 2:
            return 'alternating_actions'
        else:
            return 'position_stuck'

    def _get_stuck_recovery_actions(self) -> List[str]:
        """Get actions to recover from stuck state."""
        return [
            'try_random_direction',
            'press_b_to_cancel',
            'try_menu_actions',
            'explore_different_area'
        ]

    def _get_confidence_factors(self, game_state: Dict[str, Any]) -> Dict[str, float]:
        """Get confidence factors for decision making."""
        factors = {
            'data_quality': 1.0,
            'situation_clarity': 1.0,
            'objective_clarity': 1.0
        }

        # Reduce confidence if we have insufficient data
        if len(self.position_history) < 5:
            factors['data_quality'] *= 0.7

        # Reduce confidence if situation is unclear
        if game_state.get('player_map', 0) == 0:
            factors['situation_clarity'] *= 0.5

        return factors

    def _get_fallback_context(self, game_state: Dict[str, Any], action_count: int) -> Dict[str, Any]:
        """Get minimal fallback context if enhanced context fails."""
        return {
            'current_state': 'overworld',
            'step': action_count,
            'stuck_counter': 0,
            'player_map': game_state.get('player_map', 0),
            'player_x': game_state.get('player_x', 0),
            'player_y': game_state.get('player_y', 0),
            'badges': game_state.get('badges', 0),
            'party_count': game_state.get('party_count', 0),
            'in_battle': game_state.get('in_battle', 0),
            'fallback_mode': True
        }

    def _initialize_location_database(self) -> Dict[int, LocationInfo]:
        """Initialize comprehensive location database with strategic information"""
        return {
            # New Bark Town area - Starting region
            1: LocationInfo(
                map_id=1,
                name="New Bark Town",
                location_type="town",
                strategic_value=9,  # Very important - starting town
                available_services=["pokemon_center", "professor_lab"],
                key_npcs=["professor_elm", "mom"],
                quests_available=["get_first_pokemon", "deliver_mystery_egg"],
                pokemon_encounters=[],
                items_available=["pokegear", "town_map"],
                required_hms=[],
                connections=[2]  # Route 29
            ),

            2: LocationInfo(
                map_id=2,
                name="Route 29",
                location_type="route",
                strategic_value=7,
                available_services=[],
                key_npcs=["trainer_1", "berry_person"],
                quests_available=["catch_wild_pokemon"],
                pokemon_encounters=["pidgey", "sentret"],
                items_available=["potion", "apricorn"],
                required_hms=[],
                connections=[1, 5]  # New Bark Town, Cherrygrove City
            ),

            5: LocationInfo(
                map_id=5,
                name="Cherrygrove City",
                location_type="town",
                strategic_value=8,
                available_services=["pokemon_center", "pokemart"],
                key_npcs=["guide_npc", "nurse_joy"],
                quests_available=["get_running_shoes"],
                pokemon_encounters=[],
                items_available=["running_shoes", "berries"],
                required_hms=[],
                connections=[2, 6]  # Route 29, Route 30
            ),

            8: LocationInfo(
                map_id=8,
                name="Violet City",
                location_type="town",
                strategic_value=10,
                available_services=["pokemon_center", "pokemart", "gym"],
                key_npcs=["falkner", "earl", "nurse_joy"],
                quests_available=["challenge_falkner", "learn_about_pokemon"],
                pokemon_encounters=[],
                items_available=["hm_flash", "miracle_seed"],
                required_hms=[],
                connections=[3, 32]  # Route 31, Sprout Tower
            ),

            32: LocationInfo(
                map_id=32,
                name="Sprout Tower",
                location_type="special",
                strategic_value=6,
                available_services=[],
                key_npcs=["sage_masters", "monks"],
                quests_available=["complete_sprout_tower"],
                pokemon_encounters=["rattata", "gastly"],
                items_available=["hm_flash", "escape_rope"],
                required_hms=[],
                connections=[8]  # Violet City
            ),

            # Add more locations as needed for progression tracking
        }

    def _initialize_quest_system(self) -> Dict[str, QuestTracker]:
        """Initialize quest tracking system with major story quests"""
        return {
            "get_first_pokemon": QuestTracker(
                quest_id="get_first_pokemon",
                name="Choose Your First Pokemon",
                description="Visit Professor Elm and receive your starter Pokemon",
                location="new_bark_town",
                prerequisites=[],
                rewards=["starter_pokemon", "pokedex"],
                status="available",
                progress_markers={"visited_elm": False, "received_pokemon": False},
                estimated_actions=20
            ),

            "deliver_mystery_egg": QuestTracker(
                quest_id="deliver_mystery_egg",
                name="Deliver the Mystery Egg",
                description="Deliver the mysterious egg to Mr. Pokemon",
                location="route_30",
                prerequisites=["get_first_pokemon"],
                rewards=["mystery_egg", "quest_progress"],
                status="available",
                progress_markers={"received_task": False, "found_mr_pokemon": False, "delivered_egg": False},
                estimated_actions=100
            ),

            "challenge_falkner": QuestTracker(
                quest_id="challenge_falkner",
                name="Violet Gym Challenge",
                description="Challenge Falkner for the Violet City Gym Badge",
                location="violet_city",
                prerequisites=["deliver_mystery_egg"],
                rewards=["zephyr_badge", "tm_roost"],
                status="available",
                progress_markers={"arrived_violet": False, "entered_gym": False, "defeated_falkner": False},
                estimated_actions=80
            ),

            "complete_sprout_tower": QuestTracker(
                quest_id="complete_sprout_tower",
                name="Sprout Tower Trial",
                description="Complete the Sprout Tower challenge to prove your worth",
                location="sprout_tower",
                prerequisites=["get_first_pokemon"],
                rewards=["hm_flash", "experience"],
                status="available",
                progress_markers={"entered_tower": False, "reached_top": False, "completed_trial": False},
                estimated_actions=60
            )
        }

    def _initialize_strategic_routes(self) -> Dict[str, List[int]]:
        """Initialize optimal routes between key locations"""
        return {
            "starter_to_first_gym": [1, 2, 5, 6, 7, 8],  # New Bark → Violet City
            "gym_circuit_johto": [8, 12, 16, 21, 25, 28, 31, 35],  # Johto gym order
            "elite_four_preparation": [35, 36, 37, 38],  # Route to Elite Four
            "kanto_access": [38, 39, 40, 41],  # Post-Elite Four Kanto access
        }

    def _initialize_progression_checkpoints(self) -> Dict[str, Dict[str, Any]]:
        """Initialize key progression milestones"""
        return {
            "tutorial_complete": {
                "requirements": {"party_count": 1, "badges": 0},
                "description": "First Pokemon obtained",
                "next_goal": "first_gym_preparation"
            },
            "first_gym_preparation": {
                "requirements": {"party_count": 1, "player_level": 10},
                "description": "Ready for first gym challenge",
                "next_goal": "first_badge"
            },
            "first_badge": {
                "requirements": {"badges": 1},
                "description": "Defeated Falkner at Violet Gym",
                "next_goal": "second_gym_preparation"
            },
            "johto_champion": {
                "requirements": {"badges": 8},
                "description": "Defeated all Johto gym leaders",
                "next_goal": "elite_four_preparation"
            },
            "kanto_access": {
                "requirements": {"badges": 8, "elite_four_beaten": True},
                "description": "Access to Kanto region",
                "next_goal": "kanto_gyms"
            }
        }

    def get_current_quest_objectives(self, game_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get current active quest objectives based on game state"""
        objectives = []
        badges = game_state.get('badges_total', 0)
        party_count = game_state.get('party_count', 0)
        current_map = game_state.get('player_map', 0)

        # Update quest statuses based on game state
        self._update_quest_statuses(game_state)

        # Collect active and available quests
        for quest_id, quest in self.quest_tracker.items():
            if quest.status in ['active', 'available']:
                # Check if prerequisites are met
                if self._are_prerequisites_met(quest, game_state):
                    objective = {
                        'quest_id': quest_id,
                        'name': quest.name,
                        'description': quest.description,
                        'priority': self._calculate_quest_priority(quest, game_state),
                        'location': quest.location,
                        'estimated_actions': quest.estimated_actions,
                        'progress': self._calculate_quest_progress(quest)
                    }
                    objectives.append(objective)

        # Sort by priority
        return sorted(objectives, key=lambda x: x['priority'], reverse=True)

    def _update_quest_statuses(self, game_state: Dict[str, Any]):
        """Update quest statuses based on current game state"""
        party_count = game_state.get('party_count', 0)
        badges = game_state.get('badges_total', 0)

        # Update quest progress markers
        if party_count > 0:
            self.quest_tracker["get_first_pokemon"].progress_markers["received_pokemon"] = True
            if self.quest_tracker["get_first_pokemon"].status == "available":
                self.quest_tracker["get_first_pokemon"].status = "completed"

        if badges >= 1:
            self.quest_tracker["challenge_falkner"].status = "completed"

    def _are_prerequisites_met(self, quest: QuestTracker, game_state: Dict[str, Any]) -> bool:
        """Check if quest prerequisites are satisfied"""
        for prereq in quest.prerequisites:
            if prereq in self.quest_tracker:
                if self.quest_tracker[prereq].status != "completed":
                    return False
        return True

    def _calculate_quest_priority(self, quest: QuestTracker, game_state: Dict[str, Any]) -> int:
        """Calculate dynamic quest priority based on current situation"""
        base_priority = 5
        party_count = game_state.get('party_count', 0)
        badges = game_state.get('badges_total', 0)

        # High priority for essential progression
        if quest.quest_id == "get_first_pokemon" and party_count == 0:
            return 10

        if quest.quest_id == "challenge_falkner" and badges == 0 and party_count > 0:
            return 9

        # Medium priority for side content
        if quest.quest_id == "complete_sprout_tower":
            return 6

        return base_priority

    def _calculate_quest_progress(self, quest: QuestTracker) -> float:
        """Calculate quest completion percentage"""
        if not quest.progress_markers:
            return 0.0

        completed_markers = sum(1 for completed in quest.progress_markers.values() if completed)
        total_markers = len(quest.progress_markers)

        return completed_markers / total_markers if total_markers > 0 else 0.0

    def get_navigation_advice(self, game_state: Dict[str, Any], target_location: str = None) -> Dict[str, Any]:
        """Get intelligent navigation recommendations"""
        current_map = game_state.get('player_map', 0)
        current_location = self.location_database.get(current_map)

        advice = {
            'current_location': current_location.name if current_location else f"Unknown (Map {current_map})",
            'strategic_value': current_location.strategic_value if current_location else 1,
            'available_services': current_location.available_services if current_location else [],
            'recommended_actions': [],
            'navigation_priority': 'medium'
        }

        if target_location:
            advice['target_location'] = target_location
            advice['route_suggestions'] = self._get_route_suggestions(current_map, target_location)

        # Add location-specific recommendations
        if current_location:
            if 'pokemon_center' in current_location.available_services:
                party_count = game_state.get('party_count', 0)
                if party_count > 0:
                    hp_ratio = game_state.get('player_hp', 0) / max(game_state.get('player_max_hp', 1), 1)
                    if hp_ratio < 0.5:
                        advice['recommended_actions'].append("Visit Pokemon Center for healing")

            if 'gym' in current_location.available_services:
                badges = game_state.get('badges_total', 0)
                if current_map == 8 and badges == 0:  # Violet City Gym
                    advice['recommended_actions'].append("Challenge Falkner for first badge")

        return advice

    def _get_route_suggestions(self, from_map: int, to_location: str) -> List[str]:
        """Get route suggestions for navigation"""
        # This would implement pathfinding logic
        # For now, return basic suggestions
        route_suggestions = []

        if to_location == "violet_city":
            route_suggestions = [
                "Head north through Route 29",
                "Pass through Cherrygrove City",
                "Continue to Route 30 and 31",
                "Arrive at Violet City"
            ]

        return route_suggestions