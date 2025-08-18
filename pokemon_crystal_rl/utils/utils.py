"""Utility functions for Pokemon Crystal RL trainer."""

from typing import Dict, Optional
from pokemon_crystal_rl.core.game_states import PyBoyGameState, STATE_TRANSITION_REWARDS


def calculate_reward(current_state: Dict, previous_state: Dict) -> float:
    """Calculate reward based on state transitions and progress."""
    reward = 0.0
    
    # Base survival reward
    reward += 0.1
    
    # Time penalty (small penalty to encourage efficiency)
    reward -= 0.002
    
    # Add stuck penalty if applicable
    if 'consecutive_same_screens' in current_state:
        stuck_count = current_state['consecutive_same_screens']
        
        if stuck_count >= 25:
            reward -= 2.0  # Severe stuck penalty
        elif stuck_count >= 15:
            reward -= 1.0  # Moderate stuck penalty
        elif stuck_count >= 10:
            reward -= 0.5  # Light stuck penalty
        
        # Additional escalating penalty
        if stuck_count >= 20:
            reward -= stuck_count * 0.1  # Scales with stuck duration
    
    # Add state transition rewards if applicable
    if 'game_state' in current_state and 'game_state' in previous_state:
        try:
            from_state = PyBoyGameState(previous_state['game_state'])
            to_state = PyBoyGameState(current_state['game_state'])
            
            # Check if this transition has a defined reward
            transition = (from_state, to_state)
            if transition in STATE_TRANSITION_REWARDS:
                reward += STATE_TRANSITION_REWARDS[transition]
        except ValueError:
            # Invalid state name, ignore transition reward
            pass
    
    # Add rewards for progress indicators
    if 'player_level' in current_state and 'player_level' in previous_state:
        if current_state['player_level'] > previous_state['player_level']:
            reward += 10.0  # Level up bonus
            
    if 'player_exp' in current_state and 'player_exp' in previous_state:
        exp_gain = current_state['player_exp'] - previous_state['player_exp']
        if exp_gain > 0:
            reward += min(exp_gain * 0.01, 5.0)  # Cap at 5.0
            
    if 'money' in current_state and 'money' in previous_state:
        money_gain = current_state['money'] - previous_state['money']
        if money_gain > 0:
            reward += min(money_gain * 0.001, 0.5)  # Cap at 0.5
            
    if all(key in current_state and key in previous_state 
           for key in ['player_map', 'player_x', 'player_y']):
        # Map change bonus
        if current_state['player_map'] != previous_state['player_map']:
            reward += 5.0
        # Position change within same map
        elif (current_state['player_x'] != previous_state['player_x'] or
              current_state['player_y'] != previous_state['player_y']):
            reward += 0.1
    
    # Compound progress bonus - if multiple progress indicators changed
    progress_changes = sum([
        1 if 'player_level' in current_state and 'player_level' in previous_state and
        current_state['player_level'] > previous_state['player_level'] else 0,
        1 if 'player_exp' in current_state and 'player_exp' in previous_state and
        current_state['player_exp'] > previous_state['player_exp'] else 0,
        1 if 'money' in current_state and 'money' in previous_state and
        current_state['money'] > previous_state['money'] else 0,
        1 if 'player_map' in current_state and 'player_map' in previous_state and
        current_state['player_map'] != previous_state['player_map'] else 0
    ])
    
    if progress_changes >= 2:
        reward += progress_changes * 1.0  # Bonus for multiple progress types
    
    # Action diversity rewards/penalties
    if 'recent_actions' in current_state and len(current_state['recent_actions']) >= 5:
        unique_actions = len(set(current_state['recent_actions'][-5:]))
        if unique_actions >= 3:
            reward += 0.05  # Bonus for action variety
        elif unique_actions == 1:
            reward -= 0.02  # Penalty for repetitive actions
    
    return reward
