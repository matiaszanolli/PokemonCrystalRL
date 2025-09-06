"""Utility functions for Pokemon Crystal RL trainer."""

from typing import Dict, Optional
from core.state.machine import PyBoyGameState, STATE_TRANSITION_REWARDS


def calculate_reward(current_state: Dict, previous_state: Optional[Dict] = None) -> float:
    """Calculate reward based on state transitions and progress."""
    if previous_state is None:
        return 0.0
    
    reward = 0.0
    
    # Base survival reward (small positive reward for staying alive)
    player_hp = current_state.get('player_hp', 0)
    if player_hp > 0:
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
        reward += min(0.01 * (curr_exp - prev_exp), 5.0)  # Cap at 5.0
    
    # Badge rewards
    if 'badges' in current_state and 'badges' in previous_state:
        badges_gained = current_state['badges'] - previous_state['badges']
        if badges_gained > 0:
            reward += 100.0 * badges_gained  # Major reward for badge acquisition
    
    # Money rewards
    curr_money = current_state.get('money', 0)
    prev_money = previous_state.get('money', 0)
    money_diff = curr_money - prev_money
    if money_diff > 0:
        reward += min(0.001 * money_diff, 0.5)  # Cap at 0.5
    elif money_diff < 0:
        reward -= 0.2  # Small penalty for losing money

    # HP loss/gain penalties/rewards
    prev_hp = previous_state.get('player_hp', 0)
    if player_hp < prev_hp:
        reward -= 0.5 * (prev_hp - player_hp)  # HP loss penalty
    elif player_hp <= 0:
        reward -= 50.0  # Major penalty for fainting
    
    # Movement and exploration rewards
    curr_map = current_state.get('player_map', 0)
    prev_map = previous_state.get('player_map', 0)
    curr_x = current_state.get('player_x', 0)
    curr_y = current_state.get('player_y', 0)
    prev_x = previous_state.get('player_x', 0)
    prev_y = previous_state.get('player_y', 0)
    
    # Calculate distance moved
    distance_moved = abs(curr_x - prev_x) + abs(curr_y - prev_y)
    
    # Apply movement rewards
    if curr_map != prev_map:
        reward += 5.0  # Major reward for discovering new areas
    elif distance_moved > 0:
        reward += 0.1  # Small reward for movement
    
    # Battle rewards
    curr_battle = current_state.get('in_battle', 0) or current_state.get('game_state') == 'battle'
    prev_battle = previous_state.get('in_battle', 0) or previous_state.get('game_state') == 'battle'
    if curr_battle and not prev_battle:
        reward += 2.0  # Reward for entering battle
    elif not curr_battle and prev_battle:
        if player_hp > 0:
            reward += 5.0  # Major reward for winning battle
    elif curr_battle and ('enemy_hp' in current_state and 'enemy_hp' in previous_state and
          current_state['enemy_hp'] < previous_state['enemy_hp']):
        reward += 0.3  # Reward for dealing damage
    
    # Party growth rewards
    curr_party_count = current_state.get('party_count', 0)
    prev_party_count = previous_state.get('party_count', 0)
    if curr_party_count > prev_party_count:
        reward += 20.0 * (curr_party_count - prev_party_count)  # Major reward for catching Pokemon
    
    # Stuck detection penalties (escalating)
    consecutive_same_screens = current_state.get('consecutive_same_screens', 0)
    if consecutive_same_screens > 10:
        stuck_penalty = -0.1 * (consecutive_same_screens - 10)
        reward += stuck_penalty
        
        if consecutive_same_screens >= 25:
            reward -= 2.0  # Severe penalty for being very stuck
    
    # Menu and state transition rewards
    curr_game_state = current_state.get('game_state', 'unknown')
    prev_game_state = previous_state.get('game_state', 'unknown')
    
    # Define state transition rewards
    state_transitions = {
        ('title_screen', 'new_game_menu'): 5.0,
        ('title_screen', 'intro_sequence'): 3.0,
        ('intro_sequence', 'new_game_menu'): 3.0,
        ('new_game_menu', 'overworld'): 10.0,
        ('dialogue', 'overworld'): 1.0,
        ('menu', 'overworld'): 2.0,  # Apply both this and menu exit bonus
        ('overworld', 'battle'): 2.0,
        ('battle', 'overworld'): 3.0,
        ('loading', 'overworld'): 1.0,
        ('unknown', 'overworld'): 2.0
    }
    
    # Apply state transition rewards
    transition = (prev_game_state, curr_game_state)
    if transition in state_transitions:
        reward += state_transitions[transition]
    
    # Menu interaction rewards
    if curr_game_state == 'menu' and prev_game_state != 'menu':
        reward += 1.0  # Reward for entering menu
    elif curr_game_state != 'menu' and prev_game_state == 'menu':
        reward += 2.0  # Larger reward for successfully exiting menu
    
    # Action diversity rewards/penalties
    recent_actions = current_state.get('recent_actions', [])
    if len(recent_actions) >= 5:
        unique_actions = len(set(recent_actions[-5:]))
        if unique_actions >= 3:
            reward += 0.05  # Bonus for action variety
        elif unique_actions == 1:
            reward -= 0.02  # Penalty for repeating actions
    
    # Progress momentum rewards
    progress_indicators = [
        curr_level > prev_level,  # Level up
        curr_exp > prev_exp,      # Exp gain
        curr_money > prev_money,   # Money gain
        curr_map != prev_map,      # Map change
        distance_moved > 0,        # Movement
        curr_party_count > prev_party_count,  # Party growth
        curr_battle and not prev_battle,  # Battle entry
        not curr_battle and prev_battle and player_hp > 0  # Battle victory
    ]
    
    progress_count = sum(progress_indicators)
    if progress_count >= 2:
        reward += 0.1 * progress_count  # Bonus for compound progress
    
    # Small time penalty to encourage efficiency
    reward -= 0.002
    
    return reward
