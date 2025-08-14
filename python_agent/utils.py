"""
utils.py - Helper functions for reward shaping, preprocessing, and other utilities
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, List, Tuple
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from gym import spaces

from memory_map import MEMORY_ADDRESSES, DERIVED_VALUES, IMPORTANT_LOCATIONS, BADGE_MASKS


def normalize_value(value: float, min_val: float, max_val: float) -> float:
    """
    Normalize a value to range [0, 1]
    
    Args:
        value: Value to normalize
        min_val: Minimum possible value
        max_val: Maximum possible value
    
    Returns:
        Normalized value in [0, 1]
    """
    if max_val == min_val:
        return 0.0
    return max(0.0, min(1.0, (value - min_val) / (max_val - min_val)))


def preprocess_state(raw_state: Dict[str, Any]) -> np.ndarray:
    """
    Convert raw game state to normalized observation vector
    
    Args:
        raw_state: Raw game state dictionary from Lua script
    
    Returns:
        Normalized observation vector
    """
    features = []
    
    # Position features (normalized to map size, assuming 255x255 max)
    player_x = normalize_value(raw_state.get('player_x', 0), 0, 255)
    player_y = normalize_value(raw_state.get('player_y', 0), 0, 255)
    features.extend([player_x, player_y])
    
    # Map ID (normalized - assuming max 100 different maps)
    map_id = normalize_value(raw_state.get('player_map', 0), 0, 100)
    features.append(map_id)
    
    # HP ratio
    hp_ratio = DERIVED_VALUES['hp_percentage'](raw_state)
    features.append(hp_ratio)
    
    # Level (normalized - assuming max level 100)
    level = normalize_value(raw_state.get('player_level', 1), 1, 100)
    features.append(level)
    
    # Experience (normalized - rough estimate)
    exp = normalize_value(raw_state.get('player_exp', 0), 0, 1000000)
    features.append(exp)
    
    # Money (normalized - assuming max 999,999)
    money = normalize_value(raw_state.get('money', 0), 0, 999999)
    features.append(money)
    
    # Badge count (normalized - max 16 badges total)
    badge_count = DERIVED_VALUES['badges_total'](raw_state)
    badge_ratio = normalize_value(badge_count, 0, 16)
    features.append(badge_ratio)
    
    # Party information
    party_count = normalize_value(raw_state.get('party_count', 0), 0, 6)
    features.append(party_count)
    
    # Battle state
    in_battle = float(raw_state.get('in_battle', 0))
    features.append(in_battle)
    
    # Menu/UI state
    menu_state = normalize_value(raw_state.get('menu_state', 0), 0, 10)
    features.append(menu_state)
    
    # Time of day
    time_of_day = normalize_value(raw_state.get('time_of_day', 1), 1, 4)
    features.append(time_of_day)
    
    # Movement capabilities
    can_move = float(raw_state.get('can_move', 1))
    surf_state = float(raw_state.get('surf_state', 0))
    bike_state = float(raw_state.get('bike_state', 0))
    features.extend([can_move, surf_state, bike_state])
    
    # Enemy information (if in battle)
    if raw_state.get('in_battle', 0):
        enemy_hp_ratio = (
            raw_state.get('enemy_hp', 0) / 
            max(raw_state.get('enemy_max_hp', 1), 1)
        )
        enemy_level = normalize_value(raw_state.get('enemy_level', 1), 1, 100)
    else:
        enemy_hp_ratio = 0.0
        enemy_level = 0.0
    
    features.extend([enemy_hp_ratio, enemy_level])
    
    # Additional derived features
    is_healthy = float(DERIVED_VALUES['is_healthy'](raw_state))
    features.append(is_healthy)
    
    # Pad or truncate to fixed size (20 features)
    features = features[:20]
    while len(features) < 20:
        features.append(0.0)
    
    return np.array(features, dtype=np.float32)


def calculate_reward(current_state: Dict[str, Any], 
                    previous_state: Dict[str, Any]) -> float:
    """
    Calculate reward based on state changes
    
    Args:
        current_state: Current game state
        previous_state: Previous game state
    
    Returns:
        Reward value
    """
    reward = 0.0
    
    # Base survival reward (small positive reward for staying alive)
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
    
    # Money reward (small, to encourage item collection/battles)
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
    
    # Movement exploration bonus (encourage exploring new areas)
    curr_map = current_state.get('player_map', 0)
    prev_map = previous_state.get('player_map', 0)
    if curr_map != prev_map:
        reward += 5.0
    
    # Position change bonus (encourage movement within maps)
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
        # Battle ended - bonus if we won (still have HP)
        if curr_hp > 0:
            reward += 5.0
    
    # Party growth reward
    curr_party = current_state.get('party_count', 0)
    prev_party = previous_state.get('party_count', 0)
    if curr_party > prev_party:
        reward += 20.0 * (curr_party - prev_party)
    
    # Specific location bonuses (encourage reaching important places)
    if curr_map in IMPORTANT_LOCATIONS.values():
        if curr_map != prev_map:
            reward += 10.0
    
    # Time-based penalty (very small, to encourage efficiency)
    reward -= 0.001
    
    return reward


def calculate_shaped_reward(current_state: Dict[str, Any], 
                          previous_state: Dict[str, Any],
                          action_taken: int) -> float:
    """
    More sophisticated reward shaping with action consideration
    
    Args:
        current_state: Current game state
        previous_state: Previous game state
        action_taken: Action that was taken
    
    Returns:
        Shaped reward value
    """
    base_reward = calculate_reward(current_state, previous_state)
    
    # Action-specific shaping
    shaped_reward = base_reward
    
    # Encourage A button usage (for interactions, battles)
    if action_taken == 5:  # A button
        if current_state.get('in_battle', 0):
            shaped_reward += 0.1  # Small bonus for battle actions
        elif current_state.get('text_box_state', 0):
            shaped_reward += 0.05  # Small bonus for text interactions
    
    # Discourage excessive no-ops
    if action_taken == 0:  # No action
        shaped_reward -= 0.01
    
    # Encourage directional movement when not in menus
    if action_taken in [1, 2, 3, 4] and not current_state.get('menu_state', 0):
        shaped_reward += 0.005
    
    return shaped_reward


def create_custom_cnn_policy():
    """
    Create a custom CNN policy for visual observations (if using screen pixels)
    Note: This is for future use if we switch to pixel-based observations
    """
    class CustomCNN(BaseFeaturesExtractor):
        def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
            super().__init__(observation_space, features_dim)
            
            # Assuming 160x144 Game Boy screen, grayscale
            n_input_channels = observation_space.shape[0]
            
            self.cnn = nn.Sequential(
                nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
                nn.ReLU(),
                nn.Flatten(),
            )
            
            # Compute shape by doing one forward pass
            with torch.no_grad():
                n_flatten = self.cnn(
                    torch.as_tensor(observation_space.sample()[None]).float()
                ).shape[1]
            
            self.linear = nn.Sequential(
                nn.Linear(n_flatten, features_dim),
                nn.ReLU()
            )
        
        def forward(self, observations: torch.Tensor) -> torch.Tensor:
            return self.linear(self.cnn(observations))
    
    return CustomCNN


def get_progress_metrics(state: Dict[str, Any]) -> Dict[str, float]:
    """
    Get meaningful progress metrics for monitoring training
    
    Args:
        state: Current game state
    
    Returns:
        Dictionary of progress metrics
    """
    metrics = {
        'level': state.get('player_level', 1),
        'hp_ratio': DERIVED_VALUES['hp_percentage'](state),
        'badges_earned': DERIVED_VALUES['badges_total'](state),
        'party_count': state.get('party_count', 0),
        'money': state.get('money', 0),
        'current_map': state.get('player_map', 0),
        'is_in_battle': float(state.get('in_battle', 0)),
        'experience': state.get('player_exp', 0),
        'position': (state.get('player_x', 0), state.get('player_y', 0))
    }
    
    return metrics


def state_similarity(state1: Dict[str, Any], state2: Dict[str, Any]) -> float:
    """
    Calculate similarity between two game states (for detecting loops/stuck states)
    
    Args:
        state1: First game state
        state2: Second game state
    
    Returns:
        Similarity score (0-1, 1 = identical)
    """
    # Compare key state features
    features_to_compare = [
        'player_x', 'player_y', 'player_map', 'player_hp', 
        'player_level', 'party_count', 'money', 'badges'
    ]
    
    differences = 0
    total_features = len(features_to_compare)
    
    for feature in features_to_compare:
        val1 = state1.get(feature, 0)
        val2 = state2.get(feature, 0)
        
        if val1 != val2:
            differences += 1
    
    similarity = 1.0 - (differences / total_features)
    return similarity


def detect_stuck_state(state_history: List[Dict[str, Any]], 
                      window_size: int = 10,
                      threshold: float = 0.9) -> bool:
    """
    Detect if the agent is stuck in a loop or similar states
    
    Args:
        state_history: List of recent game states
        window_size: Number of states to check
        threshold: Similarity threshold for considering states as "stuck"
    
    Returns:
        True if agent appears to be stuck
    """
    if len(state_history) < window_size:
        return False
    
    recent_states = state_history[-window_size:]
    
    # Check if recent states are too similar
    similarities = []
    for i in range(len(recent_states) - 1):
        sim = state_similarity(recent_states[i], recent_states[i + 1])
        similarities.append(sim)
    
    average_similarity = np.mean(similarities)
    return average_similarity > threshold
