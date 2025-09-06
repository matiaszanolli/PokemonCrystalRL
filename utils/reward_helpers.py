"""
Pokemon Crystal Reward Calculation Helpers

Helper functions for calculating rewards from game state changes.
"""

from typing import Dict, List, Set, Optional, Tuple, Any
import logging
from config.constants import (
    SCREEN_STATES,
    TRAINING_PARAMS,
    REWARD_VALUES,
)

logger = logging.getLogger(__name__)

def calculate_hp_reward(current: Dict, previous: Dict) -> float:
    """Calculate reward for maintaining/improving health."""
    try:
        # Only calculate health rewards if player has Pokemon
        party_count = current.get('party_count', 0)
        if party_count == 0:
            return 0.0  # No Pokemon = no health rewards/penalties
            
        curr_hp = current.get('player_hp', 0)
        curr_max_hp = current.get('player_max_hp', 1)
        prev_hp = previous.get('player_hp', curr_hp)
        prev_max_hp = previous.get('player_max_hp', curr_max_hp)
        
        # Skip if no valid HP data
        if curr_max_hp == 0 or prev_max_hp == 0:
            return 0.0
        
        curr_hp_pct = curr_hp / curr_max_hp
        prev_hp_pct = prev_hp / prev_max_hp
        
        # Reward health improvement, penalize health loss
        hp_change = curr_hp_pct - prev_hp_pct
        
        if hp_change > 0:
            return hp_change * 5.0  # Reward healing
        elif hp_change < 0:
            return hp_change * 10.0  # Penalty for taking damage
        
        # Small bonus for staying healthy
        if curr_hp_pct > 0.8:
            return 0.1
        elif curr_hp_pct < 0.2:
            return -0.5  # Penalty for being low on health
            
        return 0.0
            
    except Exception as e:
        logger.error(f"Error calculating HP reward: {str(e)}")
        return 0.0

def calculate_level_reward(current: Dict, previous: Dict, curr_screen: str = "", prev_screen: str = "") -> float:
    """Calculate reward for level progression with validation."""
    try:
        curr_level = current.get('player_level', 0)
        prev_level = previous.get('player_level', curr_level)
        
        # Optional screen state validation
        if curr_screen and prev_screen:
            if curr_screen != SCREEN_STATES['OVERWORLD'] or prev_screen != SCREEN_STATES['OVERWORLD']:
                return 0.0
        
        # Optional party count validation (only enforce if provided in inputs)
        if ('party_count' in current or 'party_count' in previous):
            curr_party_count = current.get('party_count', 0)
            prev_party_count = previous.get('party_count', 0)
            if curr_party_count == 0 or prev_party_count == 0:
                return 0.0  # No Pokemon = no level rewards possible
        
        # Guard against impossible level spikes (>100 or huge jumps)
        if curr_level > TRAINING_PARAMS['MAX_LEVEL'] or prev_level > TRAINING_PARAMS['MAX_LEVEL']:
            return 0.0
        
        # Additional validation: levels must be reasonable (1-100)
        if not (1 <= curr_level <= TRAINING_PARAMS['MAX_LEVEL'] and 
                1 <= prev_level <= TRAINING_PARAMS['MAX_LEVEL']):
            return 0.0
        
        if curr_level > prev_level:
            level_gain = curr_level - prev_level
            # Cap level gain to prevent huge memory spike rewards
            level_gain = min(level_gain, 5)  # Max 5 levels per step
            
            # Additional validation: require HP values to be reasonable for this level
            if 'player_hp' in current and 'player_max_hp' in current:
                curr_hp = current.get('player_hp', 0)
                curr_max_hp = current.get('player_max_hp', 0)
                if curr_max_hp <= 0 or curr_hp > curr_max_hp or curr_max_hp < 10:
                    return 0.0  # Suspicious HP values, likely memory glitch
            
            return level_gain * 50.0  # Big reward for leveling up
            
        return 0.0
        
    except Exception as e:
        logger.error(f"Error calculating level reward: {str(e)}")
        return 0.0

def calculate_badge_reward(
    current: Dict,
    previous: Dict,
    curr_screen: str = "",
    prev_screen: str = "",
    badge_milestones: Optional[Set[str]] = None
) -> float:
    """Calculate reward for earning badges with validation."""
    try:
        # Optional screen state validation
        if curr_screen and prev_screen:
            if curr_screen != SCREEN_STATES['OVERWORLD'] or prev_screen != SCREEN_STATES['OVERWORLD']:
                return 0.0
        
        curr_badges = current.get('badges_total', 0)
        prev_badges = previous.get('badges_total', curr_badges)
        
        # Get badge raw values (bitmasks)
        curr_raw = (current.get('badges', 0), current.get('kanto_badges', 0))
        prev_raw = (previous.get('badges', curr_raw[0]), previous.get('kanto_badges', curr_raw[1]))
        
        # Additional validation: avoid early game memory spikes
        if ('party_count' in current and 'player_level' in current):
            early_game = current.get('party_count', 0) == 0 and current.get('player_level', 0) == 0
            if early_game and (0xFF in curr_raw or 0xFF in prev_raw):
                return 0.0
        
        # Additional validation: must have at least one Pokemon
        if 'party_count' in current and current.get('party_count', 0) == 0:
            return 0.0
            
        # Only reward if the total is within plausible range AND actually increased
        if 0 <= curr_badges <= 16 and 0 <= prev_badges <= 16 and curr_badges > prev_badges:
            # Create milestone key to prevent repeat rewards for the same badge
            milestone_key = f"badge_{curr_badges}_{curr_raw[0]}_{curr_raw[1]}"
            
            # Check if we've already rewarded this milestone
            if badge_milestones is not None:
                if milestone_key in badge_milestones:
                    return 0.0  # Already rewarded
                badge_milestones.add(milestone_key)
            
            # Cap to 1 badge per step to prevent jumps awarding huge rewards
            badge_gain = min(curr_badges - prev_badges, 1)
            return badge_gain * REWARD_VALUES['BADGE']
            
        return 0.0
        
    except Exception as e:
        logger.error(f"Error calculating badge reward: {str(e)}")
        return 0.0

def calculate_exploration_reward(
    current: Dict,
    previous: Dict,
    curr_screen: str = "",
    step_counter: int = 0,
    last_reward_step: int = -10000,
    visited_maps: Optional[Set[int]] = None,
    visited_locations: Optional[Set[Tuple[int, int, int]]] = None
) -> float:
    """Calculate reward for exploring new areas."""
    try:
        # Only reward exploration in overworld state
        if curr_screen != SCREEN_STATES['OVERWORLD']:
            return 0.0
        
        curr_map = current.get('player_map', 0)
        curr_x = current.get('player_x', 0)
        curr_y = current.get('player_y', 0)
        
        prev_map = previous.get('player_map', curr_map)
        prev_x = previous.get('player_x', curr_x)
        prev_y = previous.get('player_y', curr_y)
        
        # Validate coordinates
        if not (0 <= curr_x <= 255 and 0 <= curr_y <= 255 and 0 <= curr_map <= 255):
            return 0.0
        if not (0 <= prev_x <= 255 and 0 <= prev_y <= 255 and 0 <= prev_map <= 255):
            return 0.0
        
        current_location = (curr_map, curr_x, curr_y)
        previous_location = (prev_map, prev_x, prev_y)
        
        # Skip if coordinates are exactly the same
        if current_location == previous_location:
            return 0.0
        
        # New map reward - but only if it's a reasonable map change
        if curr_map != prev_map:
            map_diff = abs(curr_map - prev_map)
            # Only reward reasonable map transitions
            if map_diff <= 10:
                # Additional guardrails:
                if visited_maps is not None and curr_map in visited_maps:
                    return 0.0  # Already visited this map
                # Require reasonable coordinate delta
                coord_delta = abs(curr_x - prev_x) + abs(curr_y - prev_y)
                if coord_delta > 8:
                    return 0.0  # Suspicious teleport
                # Rate limit map-entry rewards
                if (step_counter - last_reward_step) < 50:
                    return 0.0
                # All good: record visit and reward
                if visited_maps is not None:
                    visited_maps.add(curr_map)
                if visited_locations is not None:
                    visited_locations.add(current_location)
                return REWARD_VALUES['NEW_MAP']
            else:
                return 0.0  # Suspicious map jump
        
        # Check new location
        if visited_locations is not None and current_location not in visited_locations:
            # Validate movement
            coord_diff = abs(curr_x - prev_x) + abs(curr_y - prev_y)
            if 1 <= coord_diff <= 5:  # Reasonable movement distance
                visited_locations.add(current_location)
                return REWARD_VALUES['NEW_LOCATION']
        
        return 0.0
        
    except Exception as e:
        logger.error(f"Error calculating exploration reward: {str(e)}")
        return 0.0

def calculate_blocked_movement_penalty(
    current: Dict,
    previous: Dict,
    last_action: Optional[str] = None,
    curr_screen: str = "",
    blocked_tracker: Optional[Dict[Tuple[int, int, int, str], int]] = None,
    max_penalty: float = REWARD_VALUES['MAX_BLOCKED_PENALTY']
) -> float:
    """Calculate escalating penalty for repeatedly trying blocked movements."""
    try:
        # Only apply this penalty in overworld state
        if curr_screen != SCREEN_STATES['OVERWORLD']:
            return 0.0
        
        # Get current and previous positions
        curr_map = current.get('player_map', 0)
        curr_x = current.get('player_x', 0)
        curr_y = current.get('player_y', 0)
        
        prev_map = previous.get('player_map', curr_map)
        prev_x = previous.get('player_x', curr_x)
        prev_y = previous.get('player_y', curr_y)
        
        if not last_action:
            return 0.0
            
        # Only track directional movements
        if last_action not in ['up', 'down', 'left', 'right']:
            return 0.0
        
        # Check if position didn't change
        position_unchanged = (
            curr_map == prev_map and 
            curr_x == prev_x and 
            curr_y == prev_y
        )
        
        if position_unchanged and blocked_tracker is not None:
            # Create tracking key: (map, x, y, direction)
            blocked_key = (curr_map, curr_x, curr_y, last_action)
            
            # Increment consecutive blocked attempts
            if blocked_key not in blocked_tracker:
                blocked_tracker[blocked_key] = 0
            blocked_tracker[blocked_key] += 1
            
            consecutive_blocks = blocked_tracker[blocked_key]
            
            # Calculate escalating penalty capped at max
            base_penalty = -0.005
            escalation_factor = consecutive_blocks  # Linear escalation
            penalty = base_penalty * escalation_factor
            penalty = max(penalty, max_penalty)
            
            # Clean up old blocked tracking occasionally
            if len(blocked_tracker) > 100:
                blocked_tracker = {
                    k: v for k, v in blocked_tracker.items() 
                    if v >= 2
                }
            
            return penalty
        else:
            # Position changed - clear any blocked tracking for this location
            if blocked_tracker is not None:
                keys_to_clear = [
                    key for key in blocked_tracker.keys()
                    if key[0] == prev_map and key[1] == prev_x and key[2] == prev_y
                ]
                for key in keys_to_clear:
                    del blocked_tracker[key]
            
            return 0.0
            
    except Exception as e:
        logger.error(f"Error calculating blocked movement penalty: {str(e)}")
        return 0.0

def get_reward_summary(rewards: Dict[str, float]) -> str:
    """Get human-readable reward summary."""
    try:
        summary_parts = []
        for category, value in rewards.items():
            if abs(value) > 0.01:  # Only show significant rewards
                summary_parts.append(f"{category}: {value:+.2f}")
        
        return " | ".join(summary_parts) if summary_parts else "no rewards"
        
    except Exception as e:
        logger.error(f"Error creating reward summary: {str(e)}")
        return "error generating summary"
