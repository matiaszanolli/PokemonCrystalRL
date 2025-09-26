"""
Official Battle Strategy Plugins

This module contains official battle strategy plugins that demonstrate
different approaches to Pokemon combat optimization.
"""

import time
from typing import Dict, Any, List
from core.plugin_system import BattleStrategyPlugin, PluginMetadata, PluginType


class AggressiveBattleStrategy(BattleStrategyPlugin):
    """Aggressive battle strategy focused on high damage output"""

    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="aggressive_battle",
            version="1.0.0",
            description="Aggressive battle strategy prioritizing high damage attacks",
            author="Pokemon Crystal RL Team",
            plugin_type=PluginType.BATTLE_STRATEGY,
            hot_swappable=True,
            priority=7,
            tags=["battle", "aggressive", "damage", "official"]
        )

    def initialize(self) -> bool:
        self.logger.info("Initializing Aggressive Battle Strategy")
        self.damage_multipliers = {
            'super_effective': 2.0,
            'not_very_effective': 0.5,
            'normal': 1.0
        }
        self.aggression_factor = self.config.get('aggression_factor', 0.8)
        return True

    def shutdown(self) -> bool:
        self.logger.info("Shutting down Aggressive Battle Strategy")
        return True

    def analyze_battle_situation(self, game_state: Dict[str, Any], battle_context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze battle with focus on damage potential"""
        start_time = time.time()

        try:
            player_hp = game_state.get('player_hp', 0)
            player_max_hp = game_state.get('player_max_hp', 1)
            enemy_hp = battle_context.get('enemy_hp', 100)
            enemy_max_hp = battle_context.get('enemy_max_hp', 100)

            player_hp_ratio = player_hp / max(player_max_hp, 1)
            enemy_hp_ratio = enemy_hp / max(enemy_max_hp, 1)

            analysis = {
                'strategy_type': 'aggressive',
                'player_hp_ratio': player_hp_ratio,
                'enemy_hp_ratio': enemy_hp_ratio,
                'damage_focus': True,
                'risk_tolerance': 'high',
                'recommended_approach': self._get_aggressive_approach(player_hp_ratio, enemy_hp_ratio),
                'urgency_level': self._calculate_urgency(player_hp_ratio, enemy_hp_ratio)
            }

            return analysis

        finally:
            self._track_performance("analyze_battle_situation", start_time)

    def recommend_move(self, game_state: Dict[str, Any], available_moves: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Recommend most aggressive move available"""
        start_time = time.time()

        try:
            if not available_moves:
                return {
                    'action': 5,  # A button - default attack
                    'confidence': 0.3,
                    'reasoning': "No moves available, using default attack",
                    'move_type': 'default'
                }

            # Score moves by damage potential
            best_move = None
            best_score = 0

            for move in available_moves:
                score = self._score_aggressive_move(move, game_state)
                if score > best_score:
                    best_score = score
                    best_move = move

            if best_move:
                return {
                    'action': best_move.get('action', 5),
                    'confidence': min(0.9, best_score / 100.0),
                    'reasoning': f"Aggressive strategy: {best_move.get('name', 'unknown move')} (score: {best_score:.1f})",
                    'move_type': 'aggressive_attack',
                    'expected_damage': best_move.get('expected_damage', 0),
                    'move_details': best_move
                }

            return {
                'action': 5,  # A button - default attack
                'confidence': 0.5,
                'reasoning': "Using aggressive default attack",
                'move_type': 'default_aggressive'
            }

        finally:
            self._track_performance("recommend_move", start_time)

    def assess_switch_opportunity(self, game_state: Dict[str, Any], party: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess switching with aggressive criteria (rarely switch)"""
        start_time = time.time()

        try:
            player_hp = game_state.get('player_hp', 0)
            player_max_hp = game_state.get('player_max_hp', 1)
            hp_ratio = player_hp / max(player_max_hp, 1)

            # Aggressive strategy rarely switches - only in critical situations
            if hp_ratio < 0.15:  # Very low HP
                for i, pokemon in enumerate(party):
                    if pokemon.get('hp', 0) > 0 and pokemon.get('can_battle', False):
                        return {
                            'should_switch': True,
                            'target_pokemon': i,
                            'confidence': 0.8,
                            'reasoning': f"Critical HP switch to {pokemon.get('name', 'unknown')}",
                            'switch_type': 'emergency'
                        }

            return {
                'should_switch': False,
                'confidence': 0.9,
                'reasoning': "Aggressive strategy: stay and fight",
                'switch_type': 'none'
            }

        finally:
            self._track_performance("assess_switch_opportunity", start_time)

    def _get_aggressive_approach(self, player_hp_ratio: float, enemy_hp_ratio: float) -> str:
        """Determine aggressive approach based on HP ratios"""
        if enemy_hp_ratio < 0.3:
            return "finish_off"
        elif player_hp_ratio > 0.7:
            return "full_assault"
        elif player_hp_ratio > 0.4:
            return "controlled_aggression"
        else:
            return "desperate_attack"

    def _calculate_urgency(self, player_hp_ratio: float, enemy_hp_ratio: float) -> int:
        """Calculate battle urgency level (1-5)"""
        if player_hp_ratio < 0.2:
            return 5  # Critical
        elif enemy_hp_ratio < 0.3:
            return 4  # High - finish them
        elif player_hp_ratio < 0.5:
            return 3  # Medium
        else:
            return 2  # Low

    def _score_aggressive_move(self, move: Dict[str, Any], game_state: Dict[str, Any]) -> float:
        """Score a move based on aggressive criteria"""
        score = 0

        # Base damage scoring
        power = move.get('power', 50)
        score += power * self.aggression_factor

        # Type effectiveness bonus
        effectiveness = move.get('type_effectiveness', 'normal')
        if effectiveness == 'super_effective':
            score *= 2.0
        elif effectiveness == 'not_very_effective':
            score *= 0.5

        # Accuracy consideration (aggressive but not reckless)
        accuracy = move.get('accuracy', 100)
        score *= (accuracy / 100.0)

        # Priority moves get bonus
        if move.get('priority', 0) > 0:
            score += 20

        # Critical hit chance bonus
        crit_chance = move.get('crit_chance', 0)
        score += crit_chance * 10

        return score


class DefensiveBattleStrategy(BattleStrategyPlugin):
    """Defensive battle strategy focused on survival and status effects"""

    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="defensive_battle",
            version="1.0.0",
            description="Defensive battle strategy prioritizing survival and status effects",
            author="Pokemon Crystal RL Team",
            plugin_type=PluginType.BATTLE_STRATEGY,
            hot_swappable=True,
            priority=5,
            tags=["battle", "defensive", "survival", "official"]
        )

    def initialize(self) -> bool:
        self.logger.info("Initializing Defensive Battle Strategy")
        self.defense_threshold = self.config.get('defense_threshold', 0.5)
        self.healing_priority = self.config.get('healing_priority', True)
        return True

    def shutdown(self) -> bool:
        self.logger.info("Shutting down Defensive Battle Strategy")
        return True

    def analyze_battle_situation(self, game_state: Dict[str, Any], battle_context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze battle with focus on defensive considerations"""
        start_time = time.time()

        try:
            player_hp = game_state.get('player_hp', 0)
            player_max_hp = game_state.get('player_max_hp', 1)
            enemy_hp = battle_context.get('enemy_hp', 100)

            player_hp_ratio = player_hp / max(player_max_hp, 1)

            analysis = {
                'strategy_type': 'defensive',
                'player_hp_ratio': player_hp_ratio,
                'damage_focus': False,
                'risk_tolerance': 'low',
                'recommended_approach': self._get_defensive_approach(player_hp_ratio),
                'healing_needed': player_hp_ratio < self.defense_threshold,
                'status_opportunity': self._assess_status_opportunity(battle_context)
            }

            return analysis

        finally:
            self._track_performance("analyze_battle_situation", start_time)

    def recommend_move(self, game_state: Dict[str, Any], available_moves: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Recommend most defensive move available"""
        start_time = time.time()

        try:
            if not available_moves:
                return {
                    'action': 5,  # A button
                    'confidence': 0.3,
                    'reasoning': "No moves available, using default",
                    'move_type': 'default'
                }

            player_hp = game_state.get('player_hp', 0)
            player_max_hp = game_state.get('player_max_hp', 1)
            hp_ratio = player_hp / max(player_max_hp, 1)

            # Prioritize healing if HP is low
            if hp_ratio < self.defense_threshold and self.healing_priority:
                healing_move = self._find_healing_move(available_moves)
                if healing_move:
                    return {
                        'action': healing_move.get('action', 5),
                        'confidence': 0.9,
                        'reasoning': f"Defensive healing: {healing_move.get('name', 'heal')}",
                        'move_type': 'healing',
                        'move_details': healing_move
                    }

            # Look for status moves
            status_move = self._find_status_move(available_moves)
            if status_move and hp_ratio > 0.3:  # Only use status when not critical
                return {
                    'action': status_move.get('action', 5),
                    'confidence': 0.7,
                    'reasoning': f"Defensive status: {status_move.get('name', 'status')}",
                    'move_type': 'status',
                    'move_details': status_move
                }

            # Default to safest attack
            safe_move = self._find_safest_attack(available_moves)
            return {
                'action': safe_move.get('action', 5),
                'confidence': 0.6,
                'reasoning': f"Defensive attack: {safe_move.get('name', 'safe attack')}",
                'move_type': 'safe_attack',
                'move_details': safe_move
            }

        finally:
            self._track_performance("recommend_move", start_time)

    def assess_switch_opportunity(self, game_state: Dict[str, Any], party: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess switching with defensive criteria (switch early to preserve Pokemon)"""
        start_time = time.time()

        try:
            player_hp = game_state.get('player_hp', 0)
            player_max_hp = game_state.get('player_max_hp', 1)
            hp_ratio = player_hp / max(player_max_hp, 1)

            # Defensive strategy switches earlier to preserve Pokemon
            if hp_ratio < 0.4:  # Switch when moderately low
                for i, pokemon in enumerate(party):
                    if pokemon.get('hp', 0) > player_hp and pokemon.get('can_battle', False):
                        return {
                            'should_switch': True,
                            'target_pokemon': i,
                            'confidence': 0.8,
                            'reasoning': f"Defensive switch to preserve current Pokemon",
                            'switch_type': 'preservation'
                        }

            return {
                'should_switch': False,
                'confidence': 0.7,
                'reasoning': "Current Pokemon still viable for defensive strategy",
                'switch_type': 'none'
            }

        finally:
            self._track_performance("assess_switch_opportunity", start_time)

    def _get_defensive_approach(self, player_hp_ratio: float) -> str:
        """Determine defensive approach based on HP ratio"""
        if player_hp_ratio < 0.2:
            return "emergency_healing"
        elif player_hp_ratio < 0.5:
            return "cautious_defense"
        else:
            return "status_control"

    def _assess_status_opportunity(self, battle_context: Dict[str, Any]) -> bool:
        """Assess if this is a good time for status moves"""
        enemy_status = battle_context.get('enemy_status', 'none')
        return enemy_status == 'none'  # Apply status if enemy has none

    def _find_healing_move(self, moves: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Find a healing move in available moves"""
        for move in moves:
            if move.get('category') == 'healing' or 'heal' in move.get('name', '').lower():
                return move
        return None

    def _find_status_move(self, moves: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Find a status move in available moves"""
        for move in moves:
            if move.get('category') == 'status' or move.get('status_effect'):
                return move
        return None

    def _find_safest_attack(self, moves: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Find the safest attacking move"""
        best_move = None
        best_safety_score = 0

        for move in moves:
            if move.get('category') == 'physical' or move.get('category') == 'special':
                # Safety score based on accuracy and low risk
                accuracy = move.get('accuracy', 100)
                power = move.get('power', 50)
                safety_score = (accuracy / 100.0) * (1.0 + power / 200.0)

                if safety_score > best_safety_score:
                    best_safety_score = safety_score
                    best_move = move

        return best_move or moves[0] if moves else {'action': 5}


class BalancedBattleStrategy(BattleStrategyPlugin):
    """Balanced battle strategy adapting to situation"""

    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="balanced_battle",
            version="1.0.0",
            description="Balanced battle strategy that adapts to current situation",
            author="Pokemon Crystal RL Team",
            plugin_type=PluginType.BATTLE_STRATEGY,
            hot_swappable=True,
            priority=6,
            tags=["battle", "balanced", "adaptive", "official"]
        )

    def initialize(self) -> bool:
        self.logger.info("Initializing Balanced Battle Strategy")
        self.aggressive_threshold = self.config.get('aggressive_threshold', 0.7)
        self.defensive_threshold = self.config.get('defensive_threshold', 0.3)
        return True

    def shutdown(self) -> bool:
        self.logger.info("Shutting down Balanced Battle Strategy")
        return True

    def analyze_battle_situation(self, game_state: Dict[str, Any], battle_context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze battle and determine balanced approach"""
        start_time = time.time()

        try:
            player_hp = game_state.get('player_hp', 0)
            player_max_hp = game_state.get('player_max_hp', 1)
            enemy_hp = battle_context.get('enemy_hp', 100)
            enemy_max_hp = battle_context.get('enemy_max_hp', 100)

            player_hp_ratio = player_hp / max(player_max_hp, 1)
            enemy_hp_ratio = enemy_hp / max(enemy_max_hp, 1)

            # Determine strategy mode based on situation
            if player_hp_ratio > self.aggressive_threshold and enemy_hp_ratio > 0.5:
                mode = "aggressive"
                risk_tolerance = "medium-high"
            elif player_hp_ratio < self.defensive_threshold:
                mode = "defensive"
                risk_tolerance = "low"
            else:
                mode = "balanced"
                risk_tolerance = "medium"

            analysis = {
                'strategy_type': 'balanced',
                'current_mode': mode,
                'player_hp_ratio': player_hp_ratio,
                'enemy_hp_ratio': enemy_hp_ratio,
                'risk_tolerance': risk_tolerance,
                'recommended_approach': self._get_balanced_approach(player_hp_ratio, enemy_hp_ratio),
                'situation_assessment': self._assess_battle_state(player_hp_ratio, enemy_hp_ratio)
            }

            return analysis

        finally:
            self._track_performance("analyze_battle_situation", start_time)

    def recommend_move(self, game_state: Dict[str, Any], available_moves: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Recommend move based on balanced strategy"""
        start_time = time.time()

        try:
            if not available_moves:
                return {
                    'action': 5,
                    'confidence': 0.3,
                    'reasoning': "No moves available",
                    'move_type': 'default'
                }

            # Analyze current situation
            analysis = self.analyze_battle_situation(game_state, {})
            mode = analysis['current_mode']

            if mode == "aggressive":
                return self._recommend_aggressive_move(available_moves, game_state)
            elif mode == "defensive":
                return self._recommend_defensive_move(available_moves, game_state)
            else:
                return self._recommend_balanced_move(available_moves, game_state)

        finally:
            self._track_performance("recommend_move", start_time)

    def assess_switch_opportunity(self, game_state: Dict[str, Any], party: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Balanced switching assessment"""
        start_time = time.time()

        try:
            player_hp = game_state.get('player_hp', 0)
            player_max_hp = game_state.get('player_max_hp', 1)
            hp_ratio = player_hp / max(player_max_hp, 1)

            # Balanced switching - not too early, not too late
            if hp_ratio < 0.25:  # Quarter health
                for i, pokemon in enumerate(party):
                    if pokemon.get('hp', 0) > player_hp and pokemon.get('can_battle', False):
                        return {
                            'should_switch': True,
                            'target_pokemon': i,
                            'confidence': 0.8,
                            'reasoning': "Balanced strategy: switch at quarter health",
                            'switch_type': 'tactical'
                        }

            return {
                'should_switch': False,
                'confidence': 0.7,
                'reasoning': "Balanced strategy: continue current battle",
                'switch_type': 'none'
            }

        finally:
            self._track_performance("assess_switch_opportunity", start_time)

    def _get_balanced_approach(self, player_hp_ratio: float, enemy_hp_ratio: float) -> str:
        """Determine balanced approach based on HP ratios"""
        if enemy_hp_ratio < 0.25:
            return "finish_enemy"
        elif player_hp_ratio < 0.3:
            return "damage_control"
        elif player_hp_ratio > 0.7 and enemy_hp_ratio > 0.7:
            return "establish_advantage"
        else:
            return "maintain_pressure"

    def _assess_battle_state(self, player_hp_ratio: float, enemy_hp_ratio: float) -> str:
        """Assess overall battle state"""
        if player_hp_ratio > enemy_hp_ratio + 0.3:
            return "winning"
        elif enemy_hp_ratio > player_hp_ratio + 0.3:
            return "losing"
        else:
            return "even"

    def _recommend_aggressive_move(self, moves: List[Dict[str, Any]], game_state: Dict[str, Any]) -> Dict[str, Any]:
        """Recommend aggressive move in aggressive mode"""
        best_move = max(moves, key=lambda m: m.get('power', 0) * m.get('accuracy', 100) / 100)
        return {
            'action': best_move.get('action', 5),
            'confidence': 0.8,
            'reasoning': "Balanced-aggressive: high damage move",
            'move_type': 'aggressive',
            'move_details': best_move
        }

    def _recommend_defensive_move(self, moves: List[Dict[str, Any]], game_state: Dict[str, Any]) -> Dict[str, Any]:
        """Recommend defensive move in defensive mode"""
        # Look for healing first
        healing_move = next((m for m in moves if 'heal' in m.get('name', '').lower()), None)
        if healing_move:
            return {
                'action': healing_move.get('action', 5),
                'confidence': 0.9,
                'reasoning': "Balanced-defensive: healing move",
                'move_type': 'healing',
                'move_details': healing_move
            }

        # Otherwise safest attack
        safe_move = max(moves, key=lambda m: m.get('accuracy', 100))
        return {
            'action': safe_move.get('action', 5),
            'confidence': 0.7,
            'reasoning': "Balanced-defensive: safe attack",
            'move_type': 'safe_attack',
            'move_details': safe_move
        }

    def _recommend_balanced_move(self, moves: List[Dict[str, Any]], game_state: Dict[str, Any]) -> Dict[str, Any]:
        """Recommend balanced move"""
        # Score moves by balanced criteria
        best_move = None
        best_score = 0

        for move in moves:
            power = move.get('power', 50)
            accuracy = move.get('accuracy', 100)
            # Balanced score = damage potential * reliability
            score = (power / 100.0) * (accuracy / 100.0) * 100

            if score > best_score:
                best_score = score
                best_move = move

        return {
            'action': best_move.get('action', 5) if best_move else 5,
            'confidence': 0.75,
            'reasoning': "Balanced strategy: optimal damage/accuracy ratio",
            'move_type': 'balanced',
            'move_details': best_move
        }