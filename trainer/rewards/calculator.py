"""
Reward calculation system for Pokemon Crystal RL training.
"""

import time
from typing import Dict, List, Tuple
import numpy as np

class PokemonRewardCalculator:
    """Sophisticated reward calculation for Pokemon Crystal"""
    
    def __init__(self):
        self.previous_state = {}
        self.exploration_bonus = {}
        self.last_reward_time = time.time()
        # Track visited locations to prevent reward farming
        self.visited_locations = set()  # Will store (map_id, x, y) tuples
        # Track visited maps to prevent repeated large map-entry rewards
        self.visited_maps = set()
        # Simple step counter and rate limit for map-entry rewards
        self.step_counter = 0
        self.last_map_reward_step = -10_000
        
    def calculate_reward(self, current_state: Dict, previous_state: Dict) -> Tuple[float, Dict[str, float]]:
        """Calculate comprehensive reward based on game progress"""
        # Increment internal step counter for simple rate limiting
        self.step_counter += 1
        
        rewards = {}
        total_reward = 0.0
        
        # 1. Health and survival rewards
        hp_reward = self._calculate_hp_reward(current_state, previous_state)
        rewards['health'] = hp_reward
        total_reward += hp_reward
        
        # 2. Level progression rewards
        level_reward = self._calculate_level_reward(current_state, previous_state)
        rewards['level'] = level_reward
        total_reward += level_reward
        
        # 3. Badge progression rewards (major milestone)
        badge_reward = self._calculate_badge_reward(current_state, previous_state)
        rewards['badges'] = badge_reward
        total_reward += badge_reward
        
        # 4. Money and item rewards
        money_reward = self._calculate_money_reward(current_state, previous_state)
        rewards['money'] = money_reward
        total_reward += money_reward
        
        # 5. Coordinate movement rewards (small rewards for any position change)
        movement_reward = self._calculate_movement_reward(current_state, previous_state)
        rewards['movement'] = movement_reward
        total_reward += movement_reward
        
        # 6. Exploration rewards (larger rewards for completely new areas)
        exploration_reward = self._calculate_exploration_reward(current_state, previous_state)
        rewards['exploration'] = exploration_reward
        total_reward += exploration_reward
        
        # 7. Battle performance rewards
        battle_reward = self._calculate_battle_reward(current_state, previous_state)
        rewards['battle'] = battle_reward
        total_reward += battle_reward
        
        # 8. Progress and efficiency penalties
        efficiency_penalty = self._calculate_efficiency_penalty(current_state)
        rewards['efficiency'] = efficiency_penalty
        total_reward += efficiency_penalty
        
        # 9. Early game progression rewards (getting first Pokemon, etc.)
        progression_reward = self._calculate_progression_reward(current_state, previous_state)
        rewards['progression'] = progression_reward
        total_reward += progression_reward
        
        # 10. Dialogue and interaction rewards (to guide toward first Pokemon)
        dialogue_reward = self._calculate_dialogue_reward(current_state, previous_state)
        rewards['dialogue'] = dialogue_reward
        total_reward += dialogue_reward
        
        # 11. Time-based small negative reward to encourage efficiency
        time_penalty = -0.01  # Small penalty each step
        rewards['time'] = time_penalty
        total_reward += time_penalty
        
        return total_reward, rewards
        
    def _calculate_hp_reward(self, current: Dict, previous: Dict) -> float:
        """Reward for maintaining/improving health"""
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
    
    def _calculate_level_reward(self, current: Dict, previous: Dict) -> float:
        """Reward for leveling up Pokemon with anti-glitch guards"""
        curr_level = current.get('player_level', 0)
        prev_level = previous.get('player_level', curr_level)
        
        # Get screen state and party count for validation
        curr_screen_state = getattr(self, 'last_screen_state', 'unknown')
        prev_screen_state = getattr(self, 'prev_screen_state', curr_screen_state)
        curr_party_count = current.get('party_count', 0)
        prev_party_count = previous.get('party_count', 0)
        
        # CRITICAL FIX: Only award level rewards if we have Pokemon
        # Level changes without Pokemon are memory glitches
        if curr_party_count == 0 or prev_party_count == 0:
            return 0.0  # No Pokemon = no level rewards possible
        
        # CRITICAL FIX: Only award level rewards in overworld state
        # This prevents menu operations from triggering false level changes
        if curr_screen_state != 'overworld' or prev_screen_state != 'overworld':
            return 0.0
        
        # Guard against impossible level spikes (>100 or huge jumps)
        if curr_level > 100 or prev_level > 100:
            return 0.0
        
        # Additional validation: levels must be reasonable (1-100)
        if not (1 <= curr_level <= 100 and 1 <= prev_level <= 100):
            return 0.0
        
        if curr_level > prev_level:
            level_gain = curr_level - prev_level
            # Cap level gain to prevent huge memory spike rewards
            level_gain = min(level_gain, 5)  # Max 5 levels per step
            
            # Additional validation: require HP values to be reasonable for this level
            curr_hp = current.get('player_hp', 0)
            curr_max_hp = current.get('player_max_hp', 0)
            if curr_max_hp < 10 or curr_hp > curr_max_hp:
                return 0.0  # Suspicious HP values, likely memory glitch
            
            return level_gain * 50.0  # Big reward for leveling up
            
        return 0.0
    
    def _calculate_badge_reward(self, current: Dict, previous: Dict) -> float:
        """Huge reward for earning badges (major milestones), with anti-glitch guards."""
        curr_badges = current.get('badges_total', 0)
        prev_badges = previous.get('badges_total', curr_badges)

        # Guard against uninitialized memory spikes (0xFF) early in the game
        curr_raw = (current.get('badges', 0), current.get('kanto_badges', 0))
        prev_raw = (previous.get('badges', curr_raw[0]), previous.get('kanto_badges', curr_raw[1]))
        early_game = current.get('party_count', 0) == 0 and current.get('player_level', 0) == 0
        if early_game and (0xFF in curr_raw or 0xFF in prev_raw):
            return 0.0

        # Only reward if the total is within plausible range
        if 0 <= curr_badges <= 16 and 0 <= prev_badges <= 16 and curr_badges > prev_badges:
            badge_gain = curr_badges - prev_badges
            # Cap to 1 badge per step to prevent jumps awarding huge rewards
            badge_gain = min(badge_gain, 1)
            return badge_gain * 500.0  # Huge reward for badge progress!

        return 0.0
    
    def _calculate_money_reward(self, current: Dict, previous: Dict) -> float:
        """Reward for earning money - with ULTRA strict validation to prevent SELECT button spam"""
        # Money rewards are disabled completely due to reliability issues
        return 0.0
    
    def _calculate_exploration_reward(self, current: Dict, previous: Dict) -> float:
        """Reward for exploring new areas - ONLY in overworld state"""
        # CRITICAL FIX: Only reward exploration when actually in overworld
        # This prevents menu state coordinate fluctuations from giving false rewards
        curr_screen_state = getattr(self, 'last_screen_state', 'unknown')
        prev_screen_state = getattr(self, 'prev_screen_state', curr_screen_state)
        
        # Only consider exploration rewards in overworld state
        if curr_screen_state != 'overworld':
            return 0.0
        
        # Also require previous state to be overworld to prevent menu->overworld transitions
        # from giving false rewards due to coordinate resets
        if prev_screen_state != 'overworld':
            return 0.0
        
        curr_map = current.get('player_map', 0)
        curr_x = current.get('player_x', 0)
        curr_y = current.get('player_y', 0)
        
        prev_map = previous.get('player_map', curr_map)
        prev_x = previous.get('player_x', curr_x)
        prev_y = previous.get('player_y', curr_y)
        
        # Additional validation: coordinates must be reasonable (0-255 range)
        if not (0 <= curr_x <= 255 and 0 <= curr_y <= 255 and 0 <= curr_map <= 255):
            return 0.0
        if not (0 <= prev_x <= 255 and 0 <= prev_y <= 255 and 0 <= prev_map <= 255):
            return 0.0
        
        # Current location tuple
        current_location = (curr_map, curr_x, curr_y)
        previous_location = (prev_map, prev_x, prev_y)
        
        # Skip if coordinates are exactly the same (no movement)
        if current_location == previous_location:
            return 0.0
        
        # New map reward - but only if it's a reasonable map change (adjacent maps)
        if curr_map != prev_map:
            map_diff = abs(curr_map - prev_map)
            # Only reward reasonable map transitions (not huge jumps that indicate glitches)
            if map_diff <= 10:
                # Additional guardrails to prevent exaggerated rewards:
                # 1) Only reward first time we ever enter this map in the session
                if curr_map in self.visited_maps:
                    return 0.0
                # 2) Require that coordinate delta is reasonable (door/edge transition, not teleport)
                coord_delta = abs(curr_x - prev_x) + abs(curr_y - prev_y)
                if coord_delta > 8:
                    return 0.0
                # 3) Rate limit map-entry rewards (e.g., once every 50 steps)
                if (self.step_counter - self.last_map_reward_step) < 50:
                    return 0.0
                # 4) All good: record visit and reward
                self.visited_maps.add(curr_map)
                self.visited_locations.add(current_location)
                self.last_map_reward_step = self.step_counter
                return 10.0  # Reward for entering a new map for the first time
            else:
                # Suspicious map jump - likely a glitch, no reward
                return 0.0
        
        # Check if this location has been visited before
        if current_location not in self.visited_locations:
            # Validate this is actual movement (not coordinate glitch)
            coord_diff = abs(curr_x - prev_x) + abs(curr_y - prev_y)
            if 1 <= coord_diff <= 5:  # Reasonable movement distance
                # New unvisited location! Add to visited set and give reward
                self.visited_locations.add(current_location)
                return 0.25  # Increased reward for discovering new tile
        
        # No reward for revisiting locations or suspicious movements
        return 0.0
    
    def _calculate_movement_reward(self, current: Dict, previous: Dict) -> float:
        """Small reward for any coordinate movement - helps break stuck patterns"""
        # CRITICAL: Only reward movement when actually in overworld
        curr_screen_state = getattr(self, 'last_screen_state', 'unknown')
        prev_screen_state = getattr(self, 'prev_screen_state', curr_screen_state)
        
        # Only consider movement rewards in overworld state
        if curr_screen_state != 'overworld':
            return 0.0
        
        # Also require previous state to be overworld to prevent false rewards
        if prev_screen_state != 'overworld':
            return 0.0
        
        curr_map = current.get('player_map', 0)
        curr_x = current.get('player_x', 0)
        curr_y = current.get('player_y', 0)
        
        prev_map = previous.get('player_map', curr_map)
        prev_x = previous.get('player_x', curr_x)
        prev_y = previous.get('player_y', curr_y)
        
        # Validate coordinates are reasonable
        if not (0 <= curr_x <= 255 and 0 <= curr_y <= 255 and 0 <= curr_map <= 255):
            return 0.0
        if not (0 <= prev_x <= 255 and 0 <= prev_y <= 255 and 0 <= prev_map <= 255):
            return 0.0
        
        # Check if coordinates changed
        position_changed = (curr_x != prev_x) or (curr_y != prev_y)
        map_changed = (curr_map != prev_map)
        
        # Reward any position change
        if position_changed or map_changed:
            # Validate this is reasonable movement (not huge coordinate jumps)
            if map_changed:
                map_diff = abs(curr_map - prev_map)
                if map_diff <= 10:  # Reasonable map transition
                    return 0.15  # Increased reward for changing maps
                else:
                    return 0.0  # Skip suspicious map jumps
            else:
                # Same map, different coordinates
                coord_diff = abs(curr_x - prev_x) + abs(curr_y - prev_y)
                if 1 <= coord_diff <= 3:  # Reasonable single-step movement
                    base_reward = 0.08  # Increased base reward for moving within same map
                    
                    # Add linear movement bonus - check if we're moving in same direction as previous move
                    linear_bonus = self._calculate_linear_movement_bonus(current, previous)
                    return base_reward + linear_bonus
                else:
                    return 0.0  # Skip suspicious coordinate jumps
        
        return 0.0  # No movement, no reward
    
    def _calculate_linear_movement_bonus(self, current: Dict, previous: Dict) -> float:
        """Calculate bonus for linear movement (moving in same direction as previously rewarded moves)"""
        # Only apply linear movement bonus if we're moving
        curr_x = current.get('player_x', 0)
        curr_y = current.get('player_y', 0)
        prev_x = previous.get('player_x', 0)
        prev_y = previous.get('player_y', 0)
        
        # Calculate current movement vector
        dx = curr_x - prev_x
        dy = curr_y - prev_y
        
        # No movement, no bonus
        if dx == 0 and dy == 0:
            return 0.0
        
        # Track movement history for linear bonus calculation
        if not hasattr(self, 'recent_movements'):
            self.recent_movements = []
        
        # Add current movement to history
        self.recent_movements.append((dx, dy))
        
        # Keep only recent movements (last 5 moves)
        if len(self.recent_movements) > 5:
            self.recent_movements.pop(0)
        
        # Need at least 2 movements to calculate linear bonus
        if len(self.recent_movements) < 2:
            return 0.0
        
        # Count consecutive movements in the same direction
        consecutive_same_direction = 1  # Current move counts as 1
        current_direction = (dx, dy)
        
        # Look backwards through recent movements
        for i in range(len(self.recent_movements) - 2, -1, -1):
            prev_direction = self.recent_movements[i]
            if prev_direction == current_direction:
                consecutive_same_direction += 1
            else:
                break  # Stop at first different direction
        
        # Apply escalating bonus for consecutive linear movement
        if consecutive_same_direction >= 2:
            # Bonus increases with consecutive moves but caps at reasonable level
            linear_bonus = min(consecutive_same_direction * 0.02, 0.1)
            return linear_bonus
        
        return 0.0
    
    def _calculate_battle_reward(self, current: Dict, previous: Dict) -> float:
        """Reward for battle performance"""
        curr_in_battle = current.get('in_battle', 0)
        prev_in_battle = previous.get('in_battle', 0)
        
        # Entered battle
        if curr_in_battle == 1 and prev_in_battle == 0:
            return 2.0  # Small reward for engaging in battle
            
        # Exited battle (assuming victory if health didn't drop significantly)
        if curr_in_battle == 0 and prev_in_battle == 1:
            curr_hp_pct = current.get('player_hp', 0) / max(current.get('player_max_hp', 1), 1)
            if curr_hp_pct > 0.5:  # Likely won the battle
                return 20.0  # Good reward for winning battle
            else:
                return -5.0  # Penalty for losing/fleeing
                
        return 0.0
    
    def _calculate_efficiency_penalty(self, current: Dict) -> float:
        """Penalty for inefficient actions"""
        # Penalty for being stuck (no progress)
        # This could be enhanced with more sophisticated stuck detection
        return 0.0  # Placeholder for now
    
    def _calculate_progression_reward(self, current: Dict, previous: Dict) -> float:
        """Reward for early game progression milestones - with strict validation"""
        curr_party_count = current.get('party_count', 0)
        prev_party_count = previous.get('party_count', 0)
        
        # Get screen state to prevent menu-state false rewards
        curr_screen_state = getattr(self, 'last_screen_state', 'unknown')
        prev_screen_state = getattr(self, 'prev_screen_state', curr_screen_state)
        
        # CRITICAL FIX: Only award progression rewards in consistent overworld states
        # This prevents menu operations from triggering false party_count changes
        if curr_screen_state != 'overworld' or prev_screen_state != 'overworld':
            return 0.0
        
        # Additional validation: party count must be reasonable and stable
        if not (0 <= curr_party_count <= 6 and 0 <= prev_party_count <= 6):
            return 0.0
        
        # Require multiple corroborating signals for genuine Pokemon acquisition
        if curr_party_count > prev_party_count:
            # Check if we have corroborating evidence of genuine progression
            curr_level = current.get('player_level', 0)
            prev_level = previous.get('player_level', 0)
            curr_hp = current.get('player_hp', 0)
            curr_max_hp = current.get('player_max_hp', 0)
            
            # First Pokemon: require level > 0 and reasonable HP values
            if prev_party_count == 0 and curr_party_count == 1:
                if curr_level > 0 and curr_max_hp > 0 and curr_hp <= curr_max_hp:
                    # Track this as a major milestone to prevent repeated rewards
                    milestone_key = f"first_pokemon_{curr_level}_{curr_max_hp}"
                    if not hasattr(self, 'progression_milestones'):
                        self.progression_milestones = set()
                    
                    if milestone_key not in self.progression_milestones:
                        self.progression_milestones.add(milestone_key)
                        return 100.0  # First Pokemon is a huge milestone!
                    else:
                        return 0.0  # Already rewarded this milestone
                else:
                    return 0.0  # Invalid Pokemon data, likely memory glitch
            
            # Additional Pokemon: require reasonable level progression
            elif prev_party_count > 0 and curr_party_count <= 6:
                # Require stable level values (not memory glitches)
                if 1 <= curr_level <= 100 and abs(curr_level - prev_level) <= 5:
                    party_milestone_key = f"party_{curr_party_count}_{curr_level}"
                    if not hasattr(self, 'progression_milestones'):
                        self.progression_milestones = set()
                    
                    if party_milestone_key not in self.progression_milestones:
                        self.progression_milestones.add(party_milestone_key)
                        return 25.0  # Additional Pokemon rewarded
                    else:
                        return 0.0  # Already rewarded this milestone
        
        return 0.0
    
    def _calculate_dialogue_reward(self, current: Dict, previous: Dict) -> float:
        """Reward for dialogue progression to guide toward first Pokemon"""
        # Get screen state to determine if we're in dialogue
        curr_screen_state = getattr(self, 'last_screen_state', 'unknown')
        prev_screen_state = getattr(self, 'prev_screen_state', curr_screen_state)
        
        # Only reward dialogue progression before getting first Pokemon
        party_count = current.get('party_count', 0)
        if party_count > 0:
            return 0.0  # No dialogue rewards after getting first Pokemon
        
        # Small reward for being in dialogue state (encourages talking to NPCs)
        if curr_screen_state == 'dialogue':
            return 0.05  # Small positive reward for dialogue engagement
        
        # Small reward for transitioning into dialogue (finding NPCs to talk to)
        if prev_screen_state == 'overworld' and curr_screen_state == 'dialogue':
            return 0.1  # Reward for initiating dialogue
        
        # Small reward for progressing through dialogue sequences
        if prev_screen_state == 'dialogue' and curr_screen_state == 'dialogue':
            return 0.02  # Small reward for progressing dialogue
        
        return 0.0
    
    def get_reward_summary(self, rewards: Dict[str, float]) -> str:
        """Get human-readable reward summary"""
        summary_parts = []
        for category, value in rewards.items():
            if abs(value) > 0.01:  # Only show significant rewards
                summary_parts.append(f"{category}: {value:+.2f}")
        
        return " | ".join(summary_parts) if summary_parts else "no rewards"
