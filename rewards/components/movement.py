"""
Movement and Exploration Reward Components

This module provides reward components related to player movement:
- Basic movement rewards
- Exploration of new areas
- Anti-farming protection
- Blocked movement penalties
"""

from typing import Dict, Tuple, Set

from ..component import RewardComponent, StateValidation

class ExplorationRewardComponent(RewardComponent):
    """Rewards for exploring new areas and locations."""

    def __init__(self):
        super().__init__("exploration")
        self.visited_maps: Set[int] = set()
        self.visited_locations: Set[Tuple[int, int, int]] = set()
        self.step_counter = 0
        self.last_map_reward_step = -10_000
        # Anti-farming: Track recent map transitions
        self.recent_map_transitions = []  # List of (from_map, to_map) tuples
        self.map_transition_history_size = 20  # Remember last 20 transitions
        self.map_transition_cooldown = {}  # Track cooldowns for specific transitions
        
    def get_validation_rules(self) -> StateValidation:
        return StateValidation(
            required_fields={'player_map', 'player_x', 'player_y'},
            value_ranges={
                'player_map': (0, 255),
                'player_x': (0, 255),
                'player_y': (0, 255)
            },
            require_screen_state=True,
            allowed_screen_states={'overworld'}
        )
        
    def calculate(self, current_state: Dict, previous_state: Dict) -> Tuple[float, Dict[str, float]]:
        if not (self.validate_state(current_state) and self.validate_state(previous_state)):
            return 0.0, {}
            
        self.step_counter += 1
        
        curr_map = current_state.get('player_map', 0)
        curr_x = current_state.get('player_x', 0)
        curr_y = current_state.get('player_y', 0)
        
        prev_map = previous_state.get('player_map', curr_map)
        prev_x = previous_state.get('player_x', curr_x)
        prev_y = previous_state.get('player_y', curr_y)
        
        current_location = (curr_map, curr_x, curr_y)
        previous_location = (prev_map, prev_x, prev_y)
        
        rewards = {}
        total_reward = 0.0
        
        # Skip if coordinates are exactly the same
        if current_location == previous_location:
            return 0.0, {}
        
        # New map reward - but only if it's a reasonable map change
        if curr_map != prev_map:
            map_diff = abs(curr_map - prev_map)
            if map_diff <= 10:  # Reasonable map transition
                # Track this map transition for anti-farming
                transition = (prev_map, curr_map)
                self.recent_map_transitions.append(transition)
                if len(self.recent_map_transitions) > self.map_transition_history_size:
                    self.recent_map_transitions.pop(0)

                # Check for farming pattern: back-and-forth between same maps
                is_farming = self._detect_map_farming(transition)

                # Only reward if not farming and first time entering this map
                if not is_farming and curr_map not in self.visited_maps:
                    # Rate limit map rewards
                    if (self.step_counter - self.last_map_reward_step) >= 50:
                        self.visited_maps.add(curr_map)
                        self.visited_locations.add(current_location)
                        self.last_map_reward_step = self.step_counter
                        rewards['new_map'] = 10.0
                        total_reward += 10.0
                elif is_farming:
                    # Apply penalty for farming behavior
                    penalty = self._calculate_farming_penalty(transition)
                    rewards['farming_penalty'] = penalty
                    total_reward += penalty
        
        # Check if this location has been visited before
        if current_location not in self.visited_locations:
            # Validate this is actual movement
            coord_diff = abs(curr_x - prev_x) + abs(curr_y - prev_y)
            if 1 <= coord_diff <= 5:  # Reasonable movement distance
                # New unvisited location
                self.visited_locations.add(current_location)
                rewards['new_location'] = 0.1
                total_reward += 0.1
        
        return total_reward, rewards

    def _detect_map_farming(self, transition: Tuple[int, int]) -> bool:
        """Detect if this transition is part of a farming pattern."""
        if len(self.recent_map_transitions) < 4:
            return False

        from_map, to_map = transition

        # Check for back-and-forth pattern in recent history
        recent_transitions = self.recent_map_transitions[-6:]  # Last 6 transitions

        # Count how many times we've seen this exact transition recently
        same_transition_count = recent_transitions.count(transition)

        # Count how many times we've seen the reverse transition recently
        reverse_transition = (to_map, from_map)
        reverse_transition_count = recent_transitions.count(reverse_transition)

        # Farming detected if we've done this transition + reverse multiple times
        total_oscillations = same_transition_count + reverse_transition_count

        # Also check if we're only moving between 2 maps in recent history
        unique_maps_in_transitions = set()
        for t in recent_transitions:
            unique_maps_in_transitions.add(t[0])
            unique_maps_in_transitions.add(t[1])

        # Farming patterns:
        # 1. High oscillation between same two maps
        # 2. Only been on 2 maps recently with multiple transitions
        is_high_oscillation = total_oscillations >= 3
        is_limited_exploration = len(unique_maps_in_transitions) <= 2 and len(recent_transitions) >= 4

        return is_high_oscillation or is_limited_exploration

    def _calculate_farming_penalty(self, transition: Tuple[int, int]) -> float:
        """Calculate escalating penalty for farming behavior."""
        transition_key = f"{transition[0]}_{transition[1]}"

        # Track how many times we've penalized this specific transition
        if transition_key not in self.map_transition_cooldown:
            self.map_transition_cooldown[transition_key] = 0

        self.map_transition_cooldown[transition_key] += 1
        penalty_count = self.map_transition_cooldown[transition_key]

        # Much smaller escalating penalty: -0.1, -0.15, -0.2, etc., capped at -0.5
        base_penalty = -0.1
        escalation = penalty_count * 0.05  # Smaller escalation
        penalty = base_penalty - escalation
        penalty = max(penalty, -0.5)  # Cap at -0.5

        return penalty


class MovementRewardComponent(RewardComponent):
    """Rewards for basic movement with anti-farming protection."""
    
    def __init__(self):
        super().__init__("movement")
        self.recent_locations = []
        self.location_history_size = 10
        self.movement_penalty_tracker = {}
        
    def get_validation_rules(self) -> StateValidation:
        return StateValidation(
            required_fields={'player_map', 'player_x', 'player_y'},
            value_ranges={
                'player_map': (0, 255),
                'player_x': (0, 255),
                'player_y': (0, 255)
            },
            require_screen_state=True,
            allowed_screen_states={'overworld'}
        )
        
    def calculate(self, current_state: Dict, previous_state: Dict) -> Tuple[float, Dict[str, float]]:
        if not (self.validate_state(current_state) and self.validate_state(previous_state)):
            return 0.0, {}
            
        curr_map = current_state.get('player_map', 0)
        curr_x = current_state.get('player_x', 0)
        curr_y = current_state.get('player_y', 0)
        
        prev_map = previous_state.get('player_map', curr_map)
        prev_x = previous_state.get('player_x', curr_x)
        prev_y = previous_state.get('player_y', curr_y)
        
        current_location = (curr_map, curr_x, curr_y)
        previous_location = (prev_map, prev_x, prev_y)
        
        # Update location history for anti-farming
        self.recent_locations.append(current_location)
        if len(self.recent_locations) > self.location_history_size:
            self.recent_locations.pop(0)
        
        # Check if coordinates changed
        position_changed = (curr_x != prev_x) or (curr_y != prev_y)
        map_changed = curr_map != prev_map
        
        # No movement = no reward
        if not position_changed and not map_changed:
            return 0.0, {}
            
        rewards = {}
        total_reward = 0.0
        
        # Check for back-and-forth farming
        if len(self.recent_locations) >= 4:
            recent_unique = list(set(self.recent_locations[-4:]))
            if len(recent_unique) <= 2:  # Only moving between 1-2 locations
                recent_visits = self.recent_locations[-6:].count(current_location)
                if recent_visits >= 3:  # Been here 3+ times recently
                    farming_key = frozenset([current_location, previous_location])
                    if farming_key not in self.movement_penalty_tracker:
                        self.movement_penalty_tracker[farming_key] = 0
                    self.movement_penalty_tracker[farming_key] += 1
                    
                    penalty = -0.01 * self.movement_penalty_tracker[farming_key]
                    penalty = max(penalty, -0.1)  # Cap penalty
                    rewards['farming_penalty'] = penalty
                    return penalty, rewards
        
        # Clean up old penalty tracking occasionally
        if len(self.recent_locations) % 100 == 0:
            current_pairs = set()
            for i in range(len(self.recent_locations) - 1):
                pair = frozenset([self.recent_locations[i], self.recent_locations[i+1]])
                current_pairs.add(pair)
            
            self.movement_penalty_tracker = {
                k: v for k, v in self.movement_penalty_tracker.items() 
                if k in current_pairs
            }
        
        # Reward legitimate movement
        if map_changed:
            map_diff = abs(curr_map - prev_map)
            if map_diff <= 10:
                rewards['map_change'] = 0.02
                total_reward += 0.02
        else:
            coord_diff = abs(curr_x - prev_x) + abs(curr_y - prev_y)
            if 1 <= coord_diff <= 3:
                rewards['movement'] = 0.01
                total_reward += 0.01

        return total_reward, rewards


class BlockedMovementComponent(RewardComponent):
    """Penalties for repeatedly attempting blocked movements."""
    
    def __init__(self):
        super().__init__("blocked_movement")
        self.blocked_movement_tracker = {}
        self.max_blocked_penalty = -0.1
        
    def get_validation_rules(self) -> StateValidation:
        return StateValidation(
            required_fields={'player_map', 'player_x', 'player_y'},
            value_ranges={
                'player_map': (0, 255),
                'player_x': (0, 255),
                'player_y': (0, 255)
            },
            require_screen_state=True,
            allowed_screen_states={'overworld'}
        )
        
    def calculate(self, current_state: Dict, previous_state: Dict) -> Tuple[float, Dict[str, float]]:
        if not (self.validate_state(current_state) and self.validate_state(previous_state)):
            return 0.0, {}
            
        # Only track directional movements
        if not self.last_action or self.last_action.lower() not in ['up', 'down', 'left', 'right']:
            return 0.0, {}
            
        curr_map = current_state.get('player_map', 0)
        curr_x = current_state.get('player_x', 0)
        curr_y = current_state.get('player_y', 0)
        
        prev_map = previous_state.get('player_map', curr_map)
        prev_x = previous_state.get('player_x', curr_x)
        prev_y = previous_state.get('player_y', curr_y)
        
        # Check if position didn't change (blocked movement)
        position_unchanged = (
            curr_map == prev_map and 
            curr_x == prev_x and 
            curr_y == prev_y
        )
        
        if position_unchanged:
            blocked_key = (curr_map, curr_x, curr_y, self.last_action.lower())
            
            if blocked_key not in self.blocked_movement_tracker:
                self.blocked_movement_tracker[blocked_key] = 0
            self.blocked_movement_tracker[blocked_key] += 1
            
            consecutive_blocks = self.blocked_movement_tracker[blocked_key]
            
            base_penalty = -0.005
            escalation_factor = consecutive_blocks
            penalty = base_penalty * escalation_factor
            penalty = max(penalty, self.max_blocked_penalty)
            
            # Clean up tracker if too large
            if len(self.blocked_movement_tracker) > 100:
                self.blocked_movement_tracker = {
                    k: v for k, v in self.blocked_movement_tracker.items() 
                    if v >= 2
                }
            
            return penalty, {'blocked_movement': penalty}
        else:
            # Clear blocked movement tracking for previous location
            keys_to_clear = [
                key for key in self.blocked_movement_tracker.keys()
                if key[0] == prev_map and key[1] == prev_x and key[2] == prev_y
            ]
            for key in keys_to_clear:
                del self.blocked_movement_tracker[key]
            
            return 0.0, {}
