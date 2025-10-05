"""
Battle Strategy Module

Provides intelligent battle decision making with type effectiveness and move selection.
"""

from typing import Dict, List, Any
import logging


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
