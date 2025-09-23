"""
Strategic Context Builder - Enhanced context generation for intelligent LLM decisions

Provides rich strategic information to help the LLM make better informed decisions
based on game progress, objectives, and current situation.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from collections import deque
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class GameObjective:
    """Represents a current game objective."""
    name: str
    description: str
    priority: int  # 1-10, higher = more important
    progress: float  # 0.0-1.0


class StrategicContextBuilder:
    """Builds enhanced strategic context for LLM decision making."""

    def __init__(self):
        self.logger = logging.getLogger("StrategicContextBuilder")

        # Memory for strategic analysis
        self.position_history = deque(maxlen=50)  # Last 50 positions
        self.action_history = deque(maxlen=20)   # Last 20 actions
        self.reward_history = deque(maxlen=100)  # Last 100 rewards

        # Map knowledge
        self.visited_locations = set()
        self.important_locations = {
            # New Bark Town area
            1: "new_bark_town",
            2: "route_29",
            3: "route_30",
            4: "route_31",
            # Cherrygrove and beyond
            5: "cherrygrove_city",
            6: "route_32",
            7: "route_33",
            # Violet City area
            8: "violet_city",
            32: "sprout_tower",
            # More locations can be added as discovered
        }

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