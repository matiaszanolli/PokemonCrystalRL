"""
Core Progress Reward Components

This module provides reward components related to core game progression metrics:
- Health status and changes
- Level progression
- Badge collection
"""

from typing import Dict, Tuple, Set

from ..component import RewardComponent, StateValidation

class HealthRewardComponent(RewardComponent):
    """Rewards for maintaining and improving Pokemon health."""
    
    def __init__(self):
        super().__init__("health")
        
    def get_validation_rules(self) -> StateValidation:
        return StateValidation(
            required_fields={'party_count'},
            value_ranges={
                'player_hp': (0, 999),
                'player_max_hp': (1, 999),
                'party_count': (0, 6)
            }
        )
        
    def calculate(self, current_state: Dict, previous_state: Dict) -> Tuple[float, Dict[str, float]]:
        if not (self.validate_state(current_state) and self.validate_state(previous_state)):
            return 0.0, {}
            
        # Only calculate health rewards if player has Pokemon
        party_count = current_state.get('party_count', 0)
        if party_count == 0:
            return 0.0, {}
            
        curr_hp = current_state.get('player_hp', 0)
        curr_max_hp = current_state.get('player_max_hp', 1)
        prev_hp = previous_state.get('player_hp', curr_hp)
        prev_max_hp = previous_state.get('player_max_hp', curr_max_hp)
        
        # Skip if no valid HP data
        if curr_max_hp <= 0 or prev_max_hp <= 0:
            return 0.0, {}
        
        curr_hp_pct = curr_hp / curr_max_hp
        prev_hp_pct = prev_hp / prev_max_hp
        
        rewards = {}
        total_reward = 0.0
        
        # Reward health improvement, penalize health loss
        hp_change = curr_hp_pct - prev_hp_pct
        if hp_change > 0:
            rewards['healing'] = hp_change * 5.0
            total_reward += rewards['healing']
        elif hp_change < 0:
            rewards['damage'] = hp_change * 10.0
            total_reward += rewards['damage']
        
        # Small bonus for staying healthy
        if curr_hp_pct > 0.8:
            rewards['healthy_bonus'] = 0.1
            total_reward += 0.1
        elif curr_hp_pct < 0.2:
            rewards['low_health_penalty'] = -0.5
            total_reward += -0.5
            
        return total_reward, rewards


class LevelRewardComponent(RewardComponent):
    """Rewards for Pokemon level progression."""
    
    def __init__(self):
        super().__init__("level")
        
    def get_validation_rules(self) -> StateValidation:
        return StateValidation(
            required_fields={'player_level'},
            value_ranges={
                'player_level': (1, 100),
                'party_count': (0, 6)
            },
            require_screen_state=True,
            allowed_screen_states={'overworld'}
        )
        
    def calculate(self, current_state: Dict, previous_state: Dict) -> Tuple[float, Dict[str, float]]:
        if not (self.validate_state(current_state) and self.validate_state(previous_state)):
            return 0.0, {}
            
        curr_level = current_state.get('player_level', 0)
        prev_level = previous_state.get('player_level', curr_level)
        
        # Guard against impossible level spikes
        if curr_level > 100 or prev_level > 100:
            return 0.0, {}
        
        if curr_level > prev_level:
            level_gain = curr_level - prev_level
            # Cap level gain to prevent huge memory spike rewards
            level_gain = min(level_gain, 5)  # Max 5 levels per step
            
            # Additional validation: require HP values to be reasonable for this level
            if 'player_hp' in current_state and 'player_max_hp' in current_state:
                curr_hp = current_state.get('player_hp', 0)
                curr_max_hp = current_state.get('player_max_hp', 0)
                if curr_max_hp <= 0 or curr_hp > curr_max_hp or curr_max_hp < 10:
                    return 0.0, {}  # Suspicious HP values, likely memory glitch
            
            reward = level_gain * 50.0
            return reward, {'level_up': reward}
            
        return 0.0, {}


class BadgeRewardComponent(RewardComponent):
    """Rewards for earning gym badges."""
    
    def __init__(self):
        super().__init__("badges")
        self.badge_milestones: Set[str] = set()
        
    def get_validation_rules(self) -> StateValidation:
        return StateValidation(
            required_fields={'badges_total', 'badges'},
            value_ranges={
                'badges_total': (0, 16),
                'badges': (0, 255),  # Raw badge bitmask
                'party_count': (0, 6)
            },
            require_screen_state=True,
            allowed_screen_states={'overworld'}
        )
        
    def calculate(self, current_state: Dict, previous_state: Dict) -> Tuple[float, Dict[str, float]]:
        if not (self.validate_state(current_state) and self.validate_state(previous_state)):
            return 0.0, {}
            
        curr_badges = current_state.get('badges_total', 0)
        prev_badges = previous_state.get('badges_total', curr_badges)
        
        # Get badge raw values (bitmasks)
        curr_raw = (current_state.get('badges', 0), current_state.get('kanto_badges', 0))
        prev_raw = (previous_state.get('badges', curr_raw[0]), previous_state.get('kanto_badges', curr_raw[1]))
        
        # Additional validation: avoid early game memory spikes
        if ('party_count' in current_state and 'player_level' in current_state):
            early_game = current_state.get('party_count', 0) == 0 and current_state.get('player_level', 0) == 0
            if early_game and (0xFF in curr_raw or 0xFF in prev_raw):
                return 0.0, {}
        
        # Additional validation: must have at least one Pokemon to earn badges
        if 'party_count' in current_state and current_state.get('party_count', 0) == 0:
            return 0.0, {}
            
        # Only reward if the total is within plausible range AND actually increased
        if 0 <= curr_badges <= 16 and 0 <= prev_badges <= 16 and curr_badges > prev_badges:
            # Create milestone key to prevent repeat rewards
            milestone_key = f"badge_{curr_badges}_{curr_raw[0]}_{curr_raw[1]}"
            
            # Only reward each badge milestone once
            if milestone_key not in self.badge_milestones:
                self.badge_milestones.add(milestone_key)
                
                # Cap to 1 badge per step to prevent jumps
                badge_gain = min(curr_badges - prev_badges, 1)
                reward = badge_gain * 500.0
                
                return reward, {'badge_earned': reward}
                
        return 0.0, {}
