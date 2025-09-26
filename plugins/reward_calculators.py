"""
Official Reward Calculator Plugins

This module contains official reward calculation plugins that demonstrate
different approaches to reward system design.
"""

import time
from typing import Dict, Any
from core.plugin_system import RewardCalculatorPlugin, PluginMetadata, PluginType


class ProgressionFocusedReward(RewardCalculatorPlugin):
    """Reward calculator focused on game progression milestones"""

    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="progression_focused",
            version="1.0.0",
            description="Reward calculator emphasizing progression milestones",
            author="Pokemon Crystal RL Team",
            plugin_type=PluginType.REWARD_CALCULATOR,
            hot_swappable=True,
            priority=8,
            tags=["reward", "progression", "milestones", "official"]
        )

    def initialize(self) -> bool:
        self.logger.info("Initializing Progression Focused Reward Calculator")

        # Reward weights from config
        self.level_reward_weight = self.config.get('level_reward_weight', 100.0)
        self.badge_reward_weight = self.config.get('badge_reward_weight', 1000.0)
        self.money_reward_weight = self.config.get('money_reward_weight', 0.01)
        self.action_penalty = self.config.get('action_penalty', -0.1)
        self.exploration_bonus = self.config.get('exploration_bonus', 5.0)

        return True

    def shutdown(self) -> bool:
        self.logger.info("Shutting down Progression Focused Reward Calculator")
        return True

    def calculate_reward(self, old_state: Dict[str, Any], new_state: Dict[str, Any], action: int) -> float:
        """Calculate reward focused on progression"""
        start_time = time.time()

        try:
            total_reward = 0.0

            # Level progression rewards
            old_level = old_state.get('player_level', 0)
            new_level = new_state.get('player_level', 0)
            if new_level > old_level:
                level_gain = new_level - old_level
                total_reward += level_gain * self.level_reward_weight
                self.logger.info(f"Level up reward: {level_gain * self.level_reward_weight}")

            # Badge progression rewards
            old_badges = old_state.get('badges_total', 0)
            new_badges = new_state.get('badges_total', 0)
            if new_badges > old_badges:
                badge_gain = new_badges - old_badges
                total_reward += badge_gain * self.badge_reward_weight
                self.logger.info(f"Badge reward: {badge_gain * self.badge_reward_weight}")

            # Money progression (small bonus)
            old_money = old_state.get('player_money', 0)
            new_money = new_state.get('player_money', 0)
            if new_money > old_money:
                money_gain = new_money - old_money
                total_reward += money_gain * self.money_reward_weight

            # Map exploration bonus
            old_map = old_state.get('player_map', 0)
            new_map = new_state.get('player_map', 0)
            if new_map != old_map and new_map > 0:
                total_reward += self.exploration_bonus

            # Action penalty (encourage efficiency)
            total_reward += self.action_penalty

            return total_reward

        finally:
            self._track_performance("calculate_reward", start_time)

    def get_reward_breakdown(self, old_state: Dict[str, Any], new_state: Dict[str, Any]) -> Dict[str, float]:
        """Get detailed breakdown of reward components"""
        breakdown = {}

        # Level rewards
        old_level = old_state.get('player_level', 0)
        new_level = new_state.get('player_level', 0)
        if new_level > old_level:
            breakdown['level_progression'] = (new_level - old_level) * self.level_reward_weight

        # Badge rewards
        old_badges = old_state.get('badges_total', 0)
        new_badges = new_state.get('badges_total', 0)
        if new_badges > old_badges:
            breakdown['badge_progression'] = (new_badges - old_badges) * self.badge_reward_weight

        # Money rewards
        old_money = old_state.get('player_money', 0)
        new_money = new_state.get('player_money', 0)
        if new_money > old_money:
            breakdown['money_gain'] = (new_money - old_money) * self.money_reward_weight

        # Exploration rewards
        old_map = old_state.get('player_map', 0)
        new_map = new_state.get('player_map', 0)
        if new_map != old_map and new_map > 0:
            breakdown['exploration'] = self.exploration_bonus

        # Action penalty
        breakdown['action_penalty'] = self.action_penalty

        return breakdown


class BattleFocusedReward(RewardCalculatorPlugin):
    """Reward calculator focused on battle performance"""

    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="battle_focused",
            version="1.0.0",
            description="Reward calculator emphasizing battle performance",
            author="Pokemon Crystal RL Team",
            plugin_type=PluginType.REWARD_CALCULATOR,
            hot_swappable=True,
            priority=6,
            tags=["reward", "battle", "combat", "official"]
        )

    def initialize(self) -> bool:
        self.logger.info("Initializing Battle Focused Reward Calculator")

        # Battle-specific reward weights
        self.hp_preservation_weight = self.config.get('hp_preservation_weight', 10.0)
        self.enemy_damage_weight = self.config.get('enemy_damage_weight', 5.0)
        self.battle_victory_bonus = self.config.get('battle_victory_bonus', 50.0)
        self.battle_loss_penalty = self.config.get('battle_loss_penalty', -25.0)
        self.critical_hp_penalty = self.config.get('critical_hp_penalty', -20.0)

        return True

    def shutdown(self) -> bool:
        self.logger.info("Shutting down Battle Focused Reward Calculator")
        return True

    def calculate_reward(self, old_state: Dict[str, Any], new_state: Dict[str, Any], action: int) -> float:
        """Calculate reward focused on battle performance"""
        start_time = time.time()

        try:
            total_reward = 0.0

            # Check if in battle
            in_battle = new_state.get('in_battle', False)
            was_in_battle = old_state.get('in_battle', False)

            if in_battle or was_in_battle:
                # HP preservation rewards
                old_hp = old_state.get('player_hp', 0)
                new_hp = new_state.get('player_hp', 0)
                old_max_hp = old_state.get('player_max_hp', 1)
                new_max_hp = new_state.get('player_max_hp', 1)

                # Reward for maintaining HP
                if new_hp > 0 and new_max_hp > 0:
                    hp_ratio = new_hp / new_max_hp
                    if hp_ratio > 0.5:
                        total_reward += self.hp_preservation_weight * hp_ratio
                    elif hp_ratio < 0.2:
                        total_reward += self.critical_hp_penalty

                # Reward for dealing damage (if enemy HP info available)
                old_enemy_hp = old_state.get('enemy_hp', 100)
                new_enemy_hp = new_state.get('enemy_hp', 100)
                if new_enemy_hp < old_enemy_hp:
                    damage_dealt = old_enemy_hp - new_enemy_hp
                    total_reward += damage_dealt * self.enemy_damage_weight

                # Battle outcome rewards
                if was_in_battle and not in_battle:
                    # Battle ended
                    if new_hp > 0:  # Victory (player still has HP)
                        total_reward += self.battle_victory_bonus
                    else:  # Defeat
                        total_reward += self.battle_loss_penalty

            return total_reward

        finally:
            self._track_performance("calculate_reward", start_time)

    def get_reward_breakdown(self, old_state: Dict[str, Any], new_state: Dict[str, Any]) -> Dict[str, float]:
        """Get detailed breakdown of battle reward components"""
        breakdown = {}

        in_battle = new_state.get('in_battle', False)
        was_in_battle = old_state.get('in_battle', False)

        if in_battle or was_in_battle:
            # HP preservation
            new_hp = new_state.get('player_hp', 0)
            new_max_hp = new_state.get('player_max_hp', 1)
            if new_hp > 0 and new_max_hp > 0:
                hp_ratio = new_hp / new_max_hp
                if hp_ratio > 0.5:
                    breakdown['hp_preservation'] = self.hp_preservation_weight * hp_ratio
                elif hp_ratio < 0.2:
                    breakdown['critical_hp_penalty'] = self.critical_hp_penalty

            # Damage dealing
            old_enemy_hp = old_state.get('enemy_hp', 100)
            new_enemy_hp = new_state.get('enemy_hp', 100)
            if new_enemy_hp < old_enemy_hp:
                damage_dealt = old_enemy_hp - new_enemy_hp
                breakdown['damage_dealt'] = damage_dealt * self.enemy_damage_weight

            # Battle outcome
            if was_in_battle and not in_battle:
                if new_hp > 0:
                    breakdown['battle_victory'] = self.battle_victory_bonus
                else:
                    breakdown['battle_defeat'] = self.battle_loss_penalty

        return breakdown


class ExplorationFocusedReward(RewardCalculatorPlugin):
    """Reward calculator focused on exploration and discovery"""

    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="exploration_focused",
            version="1.0.0",
            description="Reward calculator emphasizing exploration and discovery",
            author="Pokemon Crystal RL Team",
            plugin_type=PluginType.REWARD_CALCULATOR,
            hot_swappable=True,
            priority=7,
            tags=["reward", "exploration", "discovery", "official"]
        )

    def initialize(self) -> bool:
        self.logger.info("Initializing Exploration Focused Reward Calculator")

        # Exploration-specific weights
        self.new_area_bonus = self.config.get('new_area_bonus', 20.0)
        self.movement_reward = self.config.get('movement_reward', 1.0)
        self.backtrack_penalty = self.config.get('backtrack_penalty', -0.5)
        self.stuck_penalty = self.config.get('stuck_penalty', -2.0)

        # Track visited positions for backtracking detection
        self.visited_positions = set()
        self.last_position = None

        return True

    def shutdown(self) -> bool:
        self.logger.info("Shutting down Exploration Focused Reward Calculator")
        return True

    def calculate_reward(self, old_state: Dict[str, Any], new_state: Dict[str, Any], action: int) -> float:
        """Calculate reward focused on exploration"""
        start_time = time.time()

        try:
            total_reward = 0.0

            # Current position
            current_pos = (new_state.get('player_x', 0), new_state.get('player_y', 0))
            old_pos = (old_state.get('player_x', 0), old_state.get('player_y', 0))

            # New area discovery
            old_map = old_state.get('player_map', 0)
            new_map = new_state.get('player_map', 0)
            if new_map != old_map and new_map > 0:
                total_reward += self.new_area_bonus

            # Movement rewards
            if current_pos != old_pos:
                # Reward for movement
                total_reward += self.movement_reward

                # Check for backtracking
                if current_pos in self.visited_positions:
                    total_reward += self.backtrack_penalty
                else:
                    # Bonus for visiting new position
                    total_reward += self.movement_reward * 0.5

                # Update visited positions
                self.visited_positions.add(current_pos)
            else:
                # Penalty for being stuck
                total_reward += self.stuck_penalty

            # Update last position
            self.last_position = current_pos

            return total_reward

        finally:
            self._track_performance("calculate_reward", start_time)

    def get_reward_breakdown(self, old_state: Dict[str, Any], new_state: Dict[str, Any]) -> Dict[str, float]:
        """Get detailed breakdown of exploration reward components"""
        breakdown = {}

        # Current position
        current_pos = (new_state.get('player_x', 0), new_state.get('player_y', 0))
        old_pos = (old_state.get('player_x', 0), old_state.get('player_y', 0))

        # New area discovery
        old_map = old_state.get('player_map', 0)
        new_map = new_state.get('player_map', 0)
        if new_map != old_map and new_map > 0:
            breakdown['new_area_discovery'] = self.new_area_bonus

        # Movement analysis
        if current_pos != old_pos:
            breakdown['movement'] = self.movement_reward

            if current_pos in self.visited_positions:
                breakdown['backtrack_penalty'] = self.backtrack_penalty
            else:
                breakdown['new_position_bonus'] = self.movement_reward * 0.5
        else:
            breakdown['stuck_penalty'] = self.stuck_penalty

        return breakdown


class BalancedReward(RewardCalculatorPlugin):
    """Balanced reward calculator combining multiple aspects"""

    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="balanced_reward",
            version="1.0.0",
            description="Balanced reward calculator combining progression, battle, and exploration",
            author="Pokemon Crystal RL Team",
            plugin_type=PluginType.REWARD_CALCULATOR,
            hot_swappable=True,
            priority=9,
            tags=["reward", "balanced", "comprehensive", "official"]
        )

    def initialize(self) -> bool:
        self.logger.info("Initializing Balanced Reward Calculator")

        # Balanced weights for different aspects
        self.progression_weight = self.config.get('progression_weight', 0.4)
        self.battle_weight = self.config.get('battle_weight', 0.3)
        self.exploration_weight = self.config.get('exploration_weight', 0.3)

        # Initialize sub-calculators
        self.progression_calc = ProgressionFocusedReward()
        self.battle_calc = BattleFocusedReward()
        self.exploration_calc = ExplorationFocusedReward()

        # Initialize all sub-calculators
        self.progression_calc.initialize()
        self.battle_calc.initialize()
        self.exploration_calc.initialize()

        return True

    def shutdown(self) -> bool:
        self.logger.info("Shutting down Balanced Reward Calculator")

        # Shutdown sub-calculators
        self.progression_calc.shutdown()
        self.battle_calc.shutdown()
        self.exploration_calc.shutdown()

        return True

    def calculate_reward(self, old_state: Dict[str, Any], new_state: Dict[str, Any], action: int) -> float:
        """Calculate balanced reward from all aspects"""
        start_time = time.time()

        try:
            # Get rewards from each aspect
            progression_reward = self.progression_calc.calculate_reward(old_state, new_state, action)
            battle_reward = self.battle_calc.calculate_reward(old_state, new_state, action)
            exploration_reward = self.exploration_calc.calculate_reward(old_state, new_state, action)

            # Combine with weights
            total_reward = (
                progression_reward * self.progression_weight +
                battle_reward * self.battle_weight +
                exploration_reward * self.exploration_weight
            )

            return total_reward

        finally:
            self._track_performance("calculate_reward", start_time)

    def get_reward_breakdown(self, old_state: Dict[str, Any], new_state: Dict[str, Any]) -> Dict[str, float]:
        """Get detailed breakdown of all reward components"""
        breakdown = {}

        # Get breakdowns from each sub-calculator
        progression_breakdown = self.progression_calc.get_reward_breakdown(old_state, new_state)
        battle_breakdown = self.battle_calc.get_reward_breakdown(old_state, new_state)
        exploration_breakdown = self.exploration_calc.get_reward_breakdown(old_state, new_state)

        # Add weighted components
        for key, value in progression_breakdown.items():
            breakdown[f'progression_{key}'] = value * self.progression_weight

        for key, value in battle_breakdown.items():
            breakdown[f'battle_{key}'] = value * self.battle_weight

        for key, value in exploration_breakdown.items():
            breakdown[f'exploration_{key}'] = value * self.exploration_weight

        # Add summary totals
        breakdown['total_progression'] = sum(progression_breakdown.values()) * self.progression_weight
        breakdown['total_battle'] = sum(battle_breakdown.values()) * self.battle_weight
        breakdown['total_exploration'] = sum(exploration_breakdown.values()) * self.exploration_weight

        return breakdown