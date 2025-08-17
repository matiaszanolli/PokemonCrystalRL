"""Reward calculation and shaping utilities."""

from typing import Dict, Any
from ..core.memory_map import MEMORY_ADDRESSES, DERIVED_VALUES, IMPORTANT_LOCATIONS

def calculate_reward(current_state: Dict[str, Any], 
                    previous_state: Dict[str, Any]) -> float:
    """Calculate reward based on state changes."""
    reward = 0.0
    
    # Base survival reward
    if current_state.get('player_hp', 0) > 0:
        reward += 0.1
    
    # Level progression reward
    curr_level = current_state.get('player_level', 1)
    prev_level = previous_state.get('player_level', 1)
    if curr_level > prev_level:
        reward += 10.0 * (curr_level - prev_level)
    
    # Experience gain reward
    curr_exp = current_state.get('player_exp', 0)
    prev_exp = previous_state.get('player_exp', 0)
    if curr_exp > prev_exp:
        reward += 0.01 * (curr_exp - prev_exp)
    
    # Badge acquisition reward
    curr_badges = DERIVED_VALUES['badges_total'](current_state)
    prev_badges = DERIVED_VALUES['badges_total'](previous_state)
    if curr_badges > prev_badges:
        reward += 100.0 * (curr_badges - prev_badges)
    
    # Money reward
    curr_money = current_state.get('money', 0)
    prev_money = previous_state.get('money', 0)
    if curr_money > prev_money:
        reward += 0.001 * (curr_money - prev_money)
    
    # HP loss penalty
    curr_hp = current_state.get('player_hp', 0)
    prev_hp = previous_state.get('player_hp', 0)
    if curr_hp < prev_hp:
        reward -= 0.5 * (prev_hp - curr_hp)
    
    # Death penalty
    if curr_hp <= 0:
        reward -= 50.0
    
    # Movement exploration bonus
    curr_map = current_state.get('player_map', 0)
    prev_map = previous_state.get('player_map', 0)
    if curr_map != prev_map:
        reward += 5.0
    
    # Position change bonus
    curr_x = current_state.get('player_x', 0)
    curr_y = current_state.get('player_y', 0)
    prev_x = previous_state.get('player_x', 0)
    prev_y = previous_state.get('player_y', 0)
    
    distance_moved = abs(curr_x - prev_x) + abs(curr_y - prev_y)
    if distance_moved > 0:
        reward += 0.01 * min(distance_moved, 5)  # Cap movement reward
    
    # Battle engagement reward
    curr_battle = current_state.get('in_battle', 0)
    prev_battle = previous_state.get('in_battle', 0)
    if curr_battle and not prev_battle:
        reward += 2.0  # Reward for entering battle
    elif not curr_battle and prev_battle:
        # Battle ended - bonus if we won
        if curr_hp > 0:
            reward += 5.0
    
    # Party growth reward
    curr_party = current_state.get('party_count', 0)
    prev_party = previous_state.get('party_count', 0)
    if curr_party > prev_party:
        reward += 20.0 * (curr_party - prev_party)
    
    # Location bonuses
    if curr_map in IMPORTANT_LOCATIONS.values():
        if curr_map != prev_map:
            reward += 10.0
    
    # Stuck detection penalty
    consecutive_same_screens = current_state.get('consecutive_same_screens', 0)
    if consecutive_same_screens > 10:
        stuck_penalty = -0.1 * (consecutive_same_screens - 10)
        reward += stuck_penalty
        
        if consecutive_same_screens > 25:
            reward -= 2.0
    
    # Menu and state transition rewards
    curr_game_state = current_state.get('game_state', 'unknown')
    prev_game_state = previous_state.get('game_state', 'unknown')
    
    # Menu navigation rewards
    if curr_game_state == 'menu' and prev_game_state != 'menu':
        reward += 1.0  # Reward for entering menu
    elif curr_game_state != 'menu' and prev_game_state == 'menu':
        reward += 2.0  # Reward for exiting menu
    
    # State transition bonuses
    state_transitions = {
        ('title_screen', 'new_game_menu'): 5.0,
        ('title_screen', 'intro_sequence'): 3.0,
        ('intro_sequence', 'new_game_menu'): 3.0,
        ('new_game_menu', 'overworld'): 10.0,
        ('dialogue', 'overworld'): 1.0,
        ('menu', 'overworld'): 1.5,
        ('overworld', 'battle'): 2.0,
        ('battle', 'overworld'): 3.0,
        ('loading', 'overworld'): 1.0,
        ('unknown', 'overworld'): 2.0,
    }
    
    transition = (prev_game_state, curr_game_state)
    if transition in state_transitions:
        reward += state_transitions[transition]
    
    # Action diversity reward
    recent_actions = current_state.get('recent_actions', [])
    if len(recent_actions) >= 5:
        unique_actions = len(set(recent_actions[-5:]))
        if unique_actions >= 3:
            reward += 0.05
        elif unique_actions == 1:
            reward -= 0.02
    
    # Progress momentum reward
    progress_indicators = [
        curr_level > prev_level,
        curr_exp > prev_exp,
        curr_money > prev_money,
        curr_map != prev_map,
        distance_moved > 0,
        curr_party > prev_party
    ]
    
    progress_count = sum(progress_indicators)
    if progress_count >= 2:
        reward += 0.1 * progress_count
    
    # Time penalty
    reward -= 0.002
    
    return reward


def calculate_shaped_reward(current_state: Dict[str, Any], 
                          previous_state: Dict[str, Any],
                          action_taken: int) -> float:
    """Calculate shaped reward with action consideration."""
    base_reward = calculate_reward(current_state, previous_state)
    shaped_reward = base_reward
    
    # Action-specific shaping
    if action_taken == 5:  # A button
        if current_state.get('in_battle', 0):
            shaped_reward += 0.1  # Battle actions bonus
        elif current_state.get('text_box_state', 0):
            shaped_reward += 0.05  # Text interactions bonus
    
    # Discourage no-ops
    if action_taken == 0:
        shaped_reward -= 0.01
    
    # Encourage movement
    if action_taken in [1, 2, 3, 4] and not current_state.get('menu_state', 0):
        shaped_reward += 0.005
    
    return shaped_reward
