"""Progress tracking and metrics utilities."""

from typing import Dict, Any
from ..core.memory_map import DERIVED_VALUES


def get_progress_metrics(state: Dict[str, Any]) -> Dict[str, float]:
    """Get meaningful progress metrics for monitoring training."""
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
