"""
Battle and Interaction Reward Components

This module provides reward components for:
- Battle engagement and outcomes
- NPC dialogue interactions
- Money rewards
- Early game progression
"""

from typing import Dict, Tuple, Set

from ..component import RewardComponent, StateValidation

class BattleRewardComponent(RewardComponent):
    """Rewards for battle performance and outcomes."""
    
    def __init__(self):
        super().__init__("battle")
        
    def get_validation_rules(self) -> StateValidation:
        return StateValidation(
            required_fields={'in_battle'},
            value_ranges={
                'in_battle': (0, 1),
                'player_hp': (0, 999),
                'player_max_hp': (1, 999)
            }
        )
        
    def calculate(self, current_state: Dict, previous_state: Dict) -> Tuple[float, Dict[str, float]]:
        if not (self.validate_state(current_state) and self.validate_state(previous_state)):
            return 0.0, {}
            
        curr_in_battle = current_state.get('in_battle', 0)
        prev_in_battle = previous_state.get('in_battle', 0)
        
        rewards = {}
        total_reward = 0.0
        
        # Entered battle
        if curr_in_battle == 1 and prev_in_battle == 0:
            rewards['battle_start'] = 2.0
            total_reward += 2.0
            
        # Exited battle
        elif curr_in_battle == 0 and prev_in_battle == 1:
            curr_hp_pct = current_state.get('player_hp', 0) / max(current_state.get('player_max_hp', 1), 1)
            if curr_hp_pct > 0.5:  # Likely won
                rewards['battle_victory'] = 20.0
                total_reward += 20.0
            else:  # Lost or fled
                rewards['battle_loss'] = -5.0
                total_reward += -5.0
                
        return total_reward, rewards


class DialogueRewardComponent(RewardComponent):
    """Rewards for NPC dialogue interactions."""
    
    def __init__(self):
        super().__init__("dialogue")
        
    def get_validation_rules(self) -> StateValidation:
        return StateValidation(
            required_fields={'party_count'},
            value_ranges={'party_count': (0, 6)},
            require_screen_state=True,
            allowed_screen_states={'dialogue', 'overworld'}
        )
        
    def calculate(self, current_state: Dict, previous_state: Dict) -> Tuple[float, Dict[str, float]]:
        if not (self.validate_state(current_state) and self.validate_state(previous_state)):
            return 0.0, {}
            
        # Only reward dialogue before getting first Pokemon
        party_count = current_state.get('party_count', 0)
        if party_count > 0:
            return 0.0, {}
            
        rewards = {}
        total_reward = 0.0
        
        curr_screen = self.last_screen_state
        prev_screen = self.prev_screen_state
        
        # Being in dialogue
        if curr_screen == 'dialogue':
            rewards['in_dialogue'] = 0.05
            total_reward += 0.05
            
        # Starting dialogue
        if prev_screen == 'overworld' and curr_screen == 'dialogue':
            rewards['start_dialogue'] = 0.1
            total_reward += 0.1
            
        # Progressing dialogue
        if prev_screen == 'dialogue' and curr_screen == 'dialogue':
            rewards['progress_dialogue'] = 0.02
            total_reward += 0.02
            
        return total_reward, rewards


class MoneyRewardComponent(RewardComponent):
    """Rewards for earning and managing money."""
    
    def __init__(self):
        super().__init__("money")
        
    def get_validation_rules(self) -> StateValidation:
        return StateValidation(
            required_fields={'money'},
            value_ranges={'money': (0, 999999)},
            require_screen_state=True,
            allowed_screen_states={'overworld', 'battle'}
        )
        
    def calculate(self, current_state: Dict, previous_state: Dict) -> Tuple[float, Dict[str, float]]:
        if not (self.validate_state(current_state) and self.validate_state(previous_state)):
            return 0.0, {}
            
        # Never reward money changes from SELECT button
        if self.last_action and self.last_action.lower() == 'select':
            return 0.0, {}
            
        curr_money = current_state.get('money', 0)
        prev_money = previous_state.get('money', curr_money)
        
        money_change = curr_money - prev_money
        
        # Ignore large suspicious changes
        if abs(money_change) > 500:
            return 0.0, {}
            
        rewards = {}
        if money_change > 0:
            # Reward money gains
            reward = min(money_change * 0.01, 1.0)
            rewards['money_gain'] = reward
            return reward, rewards
        elif money_change < 0:
            # Small penalty for spending
            penalty = max(money_change * 0.005, -0.5)
            rewards['money_spent'] = penalty
            return penalty, rewards
            
        return 0.0, {}


class ProgressionRewardComponent(RewardComponent):
    """Rewards for early game progression milestones."""
    
    def __init__(self):
        super().__init__("progression")
        self.progression_milestones: Set[str] = set()
        
    def get_validation_rules(self) -> StateValidation:
        return StateValidation(
            required_fields={'party_count', 'player_level'},
            value_ranges={
                'party_count': (0, 6),
                'player_level': (0, 100),
                'player_hp': (0, 999),
                'player_max_hp': (1, 999)
            },
            require_screen_state=True,
            allowed_screen_states={'overworld'}
        )
        
    def calculate(self, current_state: Dict, previous_state: Dict) -> Tuple[float, Dict[str, float]]:
        if not (self.validate_state(current_state) and self.validate_state(previous_state)):
            return 0.0, {}
            
        curr_party_count = current_state.get('party_count', 0)
        prev_party_count = previous_state.get('party_count', 0)
        
        # No change in party
        if curr_party_count <= prev_party_count:
            return 0.0, {}
            
        curr_level = current_state.get('player_level', 0)
        prev_level = previous_state.get('player_level', 0)
        curr_hp = current_state.get('player_hp', 0)
        curr_max_hp = current_state.get('player_max_hp', 0)
        
        rewards = {}
        
        # First Pokemon
        if prev_party_count == 0 and curr_party_count == 1:
            if curr_level > 0 and curr_max_hp > 0 and curr_hp <= curr_max_hp:
                milestone_key = f"first_pokemon_{curr_level}_{curr_max_hp}"
                if milestone_key not in self.progression_milestones:
                    self.progression_milestones.add(milestone_key)
                    rewards['first_pokemon'] = 100.0
                    return 100.0, rewards
                    
        # Additional Pokemon
        elif prev_party_count > 0 and curr_party_count <= 6:
            if 1 <= curr_level <= 100 and abs(curr_level - prev_level) <= 5:
                milestone_key = f"party_{curr_party_count}_{curr_level}"
                if milestone_key not in self.progression_milestones:
                    self.progression_milestones.add(milestone_key)
                    rewards['new_pokemon'] = 25.0
                    return 25.0, rewards
                    
        return 0.0, {}
