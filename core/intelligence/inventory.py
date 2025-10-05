"""
Inventory Management Module

Provides intelligent inventory and item management for Pokemon Crystal.
"""

from typing import Dict, Any
import logging


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
