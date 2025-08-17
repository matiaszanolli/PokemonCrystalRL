"""State preprocessing and analysis utilities."""

import numpy as np
from typing import Dict, Any, List
from ..core.memory_map import MEMORY_ADDRESSES, DERIVED_VALUES


def normalize_value(value: float, min_val: float, max_val: float) -> float:
    """Normalize a value to range [0, 1]."""
    if max_val == min_val:
        return 0.0
    return max(0.0, min(1.0, (value - min_val) / (max_val - min_val)))


def preprocess_state(raw_state: Dict[str, Any]) -> np.ndarray:
    """Convert raw game state to normalized observation vector."""
    features = []
    
    # Position features
    player_x = normalize_value(raw_state.get('player_x', 0), 0, 255)
    player_y = normalize_value(raw_state.get('player_y', 0), 0, 255)
    features.extend([player_x, player_y])
    
    # Map ID
    map_id = normalize_value(raw_state.get('player_map', 0), 0, 100)
    features.append(map_id)
    
    # HP ratio
    hp_ratio = DERIVED_VALUES['hp_percentage'](raw_state)
    features.append(hp_ratio)
    
    # Level
    level = normalize_value(raw_state.get('player_level', 1), 1, 100)
    features.append(level)
    
    # Experience
    exp = normalize_value(raw_state.get('player_exp', 0), 0, 1000000)
    features.append(exp)
    
    # Money
    money = normalize_value(raw_state.get('money', 0), 0, 999999)
    features.append(money)
    
    # Badge count
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
    
    # Enemy information
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


def state_similarity(state1: Dict[str, Any], state2: Dict[str, Any]) -> float:
    """Calculate similarity between two game states."""
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
    """Detect if the agent is stuck in a loop or similar states."""
    if len(state_history) < window_size:
        return False
    
    recent_states = state_history[-window_size:]
    
    similarities = []
    for i in range(len(recent_states) - 1):
        sim = state_similarity(recent_states[i], recent_states[i + 1])
        similarities.append(sim)
    
    average_similarity = np.mean(similarities)
    return average_similarity > threshold
