"""
Reward Calculator - Multi-factor reward system for Pokemon Crystal RL

Extracted from LLMTrainer to handle reward calculation, state tracking,
and progress evaluation with various reward factors.
"""

import time
import logging
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from collections import deque, defaultdict

try:
    from config.memory_addresses import MEMORY_ADDRESSES
except ImportError:
    print("⚠️  Memory addresses not available")
    MEMORY_ADDRESSES = {}

try:
    from utils.memory_reader import build_observation
except ImportError:
    print("⚠️  Memory reader not available")
    build_observation = None


@dataclass
class RewardConfig:
    """Configuration for reward calculation."""
    exploration_weight: float = 1.0
    progress_weight: float = 2.0
    battle_weight: float = 1.5
    item_weight: float = 1.0
    dialogue_weight: float = 0.1
    stuck_penalty: float = -0.1
    invalid_action_penalty: float = -0.05
    time_penalty: float = -0.001
    
    # Progress thresholds
    badge_reward: float = 100.0
    level_up_reward: float = 10.0
    new_area_reward: float = 5.0
    item_reward: float = 2.0
    dialogue_reward: float = 0.1


class RewardCalculator:
    """Calculates multi-factor rewards for Pokemon Crystal RL training."""
    
    def __init__(self, config: RewardConfig = None):
        self.config = config or RewardConfig()
        self.logger = logging.getLogger("RewardCalculator")
        
        # State tracking
        self.previous_state = {}
        self.visited_locations = set()
        self.previous_badges = 0
        self.previous_level = 0
        self.previous_money = 0
        self.previous_party_count = 0
        
        # Reward history
        self.reward_history = deque(maxlen=1000)
        self.total_reward = 0.0
        self.episode_reward = 0.0
        
        # Progress tracking
        self.location_visit_count = defaultdict(int)
        self.dialogue_interactions = 0
        self.battles_won = 0
        self.items_obtained = 0
        
        # Time tracking
        self.start_time = time.time()
        self.last_reward_time = self.start_time

        # Anti-farming tracking
        self.recent_map_transitions = []  # List of (from_map, to_map) tuples
        self.map_transition_history_size = 20  # Remember last 20 transitions
        self.map_transition_cooldown = {}  # Track cooldowns for specific transitions
        self.step_counter = 0
        self.last_map_reward_step = -10_000
    
    def calculate_reward(self, 
                        current_state: Dict[str, Any],
                        action: int,
                        screen_analysis: Optional[Dict[str, Any]] = None) -> float:
        """Calculate comprehensive reward for current state and action.
        
        Args:
            current_state: Current game state
            action: Action taken (1-8)
            screen_analysis: Optional screen analysis data
            
        Returns:
            float: Calculated reward
        """
        try:
            reward = 0.0
            reward_breakdown = {}
            
            # Progress rewards (badges, levels, money)
            progress_reward = self._calculate_progress_reward(current_state)
            reward += progress_reward * self.config.progress_weight
            reward_breakdown['progress'] = progress_reward
            
            # Exploration rewards
            exploration_reward = self._calculate_exploration_reward(current_state)
            reward += exploration_reward * self.config.exploration_weight
            reward_breakdown['exploration'] = exploration_reward
            
            # Battle rewards
            battle_reward = self._calculate_battle_reward(current_state, screen_analysis)
            reward += battle_reward * self.config.battle_weight
            reward_breakdown['battle'] = battle_reward
            
            # Item and interaction rewards
            interaction_reward = self._calculate_interaction_reward(current_state, screen_analysis)
            reward += interaction_reward * self.config.item_weight
            reward_breakdown['interaction'] = interaction_reward
            
            # Dialogue rewards
            dialogue_reward = self._calculate_dialogue_reward(screen_analysis)
            reward += dialogue_reward * self.config.dialogue_weight
            reward_breakdown['dialogue'] = dialogue_reward
            
            # Time and stuck penalties
            penalty_reward = self._calculate_penalties(current_state, action)
            reward += penalty_reward
            reward_breakdown['penalties'] = penalty_reward
            
            # Update tracking
            self._update_tracking(current_state, reward, reward_breakdown)
            
            return reward
            
        except Exception as e:
            self.logger.error(f"Error calculating reward: {e}")
            return 0.0
    
    def _calculate_progress_reward(self, current_state: Dict[str, Any]) -> float:
        """Calculate rewards for game progress (badges, levels, money)."""
        reward = 0.0
        
        # Badge progress
        current_badges = current_state.get('badges', 0)
        if current_badges > self.previous_badges:
            badge_gain = current_badges - self.previous_badges
            reward += badge_gain * self.config.badge_reward
            self.logger.info(f"Badge reward: +{badge_gain * self.config.badge_reward}")
            self.previous_badges = current_badges
        
        # Level up reward
        current_level = current_state.get('player_level', 0)
        if current_level > self.previous_level:
            level_gain = current_level - self.previous_level
            reward += level_gain * self.config.level_up_reward
            self.logger.info(f"Level up reward: +{level_gain * self.config.level_up_reward}")
            self.previous_level = current_level
        
        # Money progress (small reward for earning money)
        current_money = current_state.get('money', 0)
        if current_money > self.previous_money:
            money_gain = (current_money - self.previous_money) * 0.001  # Small multiplier
            reward += min(money_gain, 5.0)  # Cap money reward
            self.previous_money = current_money
        
        # Party growth
        current_party = current_state.get('party_count', 0)
        if current_party > self.previous_party_count:
            party_gain = current_party - self.previous_party_count
            reward += party_gain * 5.0  # Reward for catching/obtaining Pokemon
            self.logger.info(f"Party growth reward: +{party_gain * 5.0}")
            self.previous_party_count = current_party
        
        return reward
    
    def _calculate_exploration_reward(self, current_state: Dict[str, Any]) -> float:
        """Calculate rewards for exploring new areas."""
        reward = 0.0
        
        # Get current location
        player_map = current_state.get('player_map', 0)
        player_x = current_state.get('player_x', 0)
        player_y = current_state.get('player_y', 0)
        current_location = (player_map, player_x, player_y)
        prev_map = self.previous_state.get('player_map', player_map)

        # Track step counter for rate limiting
        self.step_counter += 1

        # Check for map transitions and apply anti-farming logic
        if player_map != prev_map:
            # Track this map transition for anti-farming
            transition = (prev_map, player_map)
            self.recent_map_transitions.append(transition)
            if len(self.recent_map_transitions) > self.map_transition_history_size:
                self.recent_map_transitions.pop(0)

            # Check for farming pattern: back-and-forth between same maps
            is_farming = self._detect_map_farming(transition)

            # Only reward if not farming and first time entering this map
            if not is_farming:
                # Check if this is a new map entirely
                visited_maps = {loc[0] for loc in self.visited_locations}
                if player_map not in visited_maps:
                    # Rate limit map rewards
                    if (self.step_counter - self.last_map_reward_step) >= 50:
                        reward += self.config.new_area_reward * 2
                        self.last_map_reward_step = self.step_counter
                        self.logger.info(f"New map exploration (legitimate): +{self.config.new_area_reward * 2}")
            else:
                # Apply penalty for farming behavior
                penalty = self._calculate_farming_penalty(transition)
                reward += penalty
                self.logger.info(f"Map farming detected: {penalty:.1f}")

        # Reward for visiting new locations (but smaller reward to prevent exploitation)
        if current_location not in self.visited_locations:
            self.visited_locations.add(current_location)
            reward += self.config.new_area_reward
        
        # Track location visits for stuck detection
        self.location_visit_count[current_location] += 1
        
        # Small penalty for revisiting same location too often (much more conservative)
        if self.location_visit_count[current_location] > 30:
            # Very small penalty, capped at -0.1 maximum
            visit_penalty = min(0.002 * (self.location_visit_count[current_location] - 30), 0.1)
            reward -= visit_penalty
        
        return reward
    
    def _calculate_battle_reward(self, 
                               current_state: Dict[str, Any],
                               screen_analysis: Optional[Dict[str, Any]]) -> float:
        """Calculate rewards for battle-related activities."""
        reward = 0.0
        
        # Check if in battle
        in_battle = current_state.get('in_battle', 0)
        if in_battle:
            # Small reward for being in battle (engagement)
            reward += 0.1
            
            # Check for battle victory (health changes, exp gain)
            # This would require more detailed battle state tracking
            # For now, give small reward for battle participation
        
        # Battle state analysis from screen
        if screen_analysis:
            battle_state = screen_analysis.get('battle_state', '')
            if 'victory' in battle_state.lower():
                reward += 10.0
                self.battles_won += 1
                self.logger.info("Battle victory reward: +10.0")
            elif 'defeat' in battle_state.lower():
                reward -= 5.0  # Penalty for losing
        
        return reward
    
    def _calculate_interaction_reward(self, 
                                    current_state: Dict[str, Any],
                                    screen_analysis: Optional[Dict[str, Any]]) -> float:
        """Calculate rewards for item collection and interactions."""
        reward = 0.0
        
        # This would need item tracking from memory
        # For now, provide small rewards for interaction attempts
        if screen_analysis:
            state_type = screen_analysis.get('state', 'unknown')
            
            # Reward for menu interactions (likely item collection)
            if state_type == 'menu' and not hasattr(self, '_last_menu_time'):
                reward += self.config.item_reward * 0.1
                self._last_menu_time = time.time()
            elif hasattr(self, '_last_menu_time') and time.time() - self._last_menu_time > 5:
                delattr(self, '_last_menu_time')
        
        return reward
    
    def _calculate_dialogue_reward(self, screen_analysis: Optional[Dict[str, Any]]) -> float:
        """Calculate rewards for dialogue interactions."""
        reward = 0.0
        
        if screen_analysis:
            state_type = screen_analysis.get('state', 'unknown')
            
            if state_type == 'dialogue':
                reward += self.config.dialogue_reward
                self.dialogue_interactions += 1
        
        return reward
    
    def _calculate_penalties(self, current_state: Dict[str, Any], action: int) -> float:
        """Calculate penalties for undesirable behaviors."""
        penalty = 0.0

        # Time penalty (encourage efficiency) - REMOVED to prevent massive accumulation
        # The original logic was flawed as it would continuously apply penalties
        # while last_reward_time only updated on positive rewards

        # Stuck penalty (same location for too long)
        player_map = current_state.get('player_map', 0)
        player_x = current_state.get('player_x', 0)
        player_y = current_state.get('player_y', 0)
        current_location = (player_map, player_x, player_y)

        if self.location_visit_count[current_location] > 20:
            penalty += self.config.stuck_penalty

        # Invalid action penalty (if action doesn't make sense in context)
        if not self._is_action_valid(current_state, action):
            penalty += self.config.invalid_action_penalty

        return penalty
    
    def _is_action_valid(self, current_state: Dict[str, Any], action: int) -> bool:
        """Check if action is valid in current context."""
        # Basic validation - all actions 1-8 are generally valid
        if not isinstance(action, int) or not (1 <= action <= 8):
            return False
        
        # Context-specific validation could be added here
        # For example, checking if START/SELECT make sense based on game state
        
        return True
    
    def _update_tracking(self,
                        current_state: Dict[str, Any],
                        reward: float,
                        reward_breakdown: Dict[str, float]) -> None:
        """Update reward tracking and statistics."""
        # Update totals
        self.total_reward += reward
        self.episode_reward += reward

        # Update history
        self.reward_history.append({
            'reward': reward,
            'breakdown': reward_breakdown,
            'timestamp': time.time(),
            'state': current_state.copy()
        })

        # Update previous state
        self.previous_state = current_state.copy()

        # Update timing - always update, not just on positive rewards
        self.last_reward_time = time.time()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get reward calculation statistics.
        
        Returns:
            Dict: Statistics about rewards and progress
        """
        recent_rewards = [r['reward'] for r in list(self.reward_history)[-100:]]
        
        return {
            'total_reward': self.total_reward,
            'episode_reward': self.episode_reward,
            'average_reward': np.mean(recent_rewards) if recent_rewards else 0.0,
            'reward_std': np.std(recent_rewards) if recent_rewards else 0.0,
            'positive_rewards': sum(1 for r in recent_rewards if r > 0),
            'negative_rewards': sum(1 for r in recent_rewards if r < 0),
            'total_locations_visited': len(self.visited_locations),
            'dialogue_interactions': self.dialogue_interactions,
            'battles_won': self.battles_won,
            'items_obtained': self.items_obtained,
            'session_duration': time.time() - self.start_time
        }
    
    def reset_episode(self) -> float:
        """Reset episode-specific tracking and return episode reward.
        
        Returns:
            float: Final episode reward
        """
        final_reward = self.episode_reward
        self.episode_reward = 0.0
        
        # Keep some state between episodes
        # Reset location visit counts but keep visited locations
        self.location_visit_count.clear()
        
        return final_reward
    
    def save_progress(self, filepath: str) -> bool:
        """Save reward calculation progress to file.
        
        Args:
            filepath: Path to save progress data
            
        Returns:
            bool: True if save successful
        """
        try:
            import json
            
            progress_data = {
                'total_reward': self.total_reward,
                'visited_locations': list(self.visited_locations),
                'previous_badges': self.previous_badges,
                'previous_level': self.previous_level,
                'previous_money': self.previous_money,
                'previous_party_count': self.previous_party_count,
                'dialogue_interactions': self.dialogue_interactions,
                'battles_won': self.battles_won,
                'items_obtained': self.items_obtained,
                'statistics': self.get_statistics()
            }
            
            with open(filepath, 'w') as f:
                json.dump(progress_data, f, indent=2)
            
            self.logger.info(f"Progress saved to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save progress: {e}")
            return False
    
    def load_progress(self, filepath: str) -> bool:
        """Load reward calculation progress from file.
        
        Args:
            filepath: Path to load progress data
            
        Returns:
            bool: True if load successful
        """
        try:
            import json
            import os
            
            if not os.path.exists(filepath):
                return False
            
            with open(filepath, 'r') as f:
                progress_data = json.load(f)
            
            # Restore progress
            self.total_reward = progress_data.get('total_reward', 0.0)
            self.visited_locations = set(tuple(loc) if isinstance(loc, list) else loc 
                                       for loc in progress_data.get('visited_locations', []))
            self.previous_badges = progress_data.get('previous_badges', 0)
            self.previous_level = progress_data.get('previous_level', 0)
            self.previous_money = progress_data.get('previous_money', 0)
            self.previous_party_count = progress_data.get('previous_party_count', 0)
            self.dialogue_interactions = progress_data.get('dialogue_interactions', 0)
            self.battles_won = progress_data.get('battles_won', 0)
            self.items_obtained = progress_data.get('items_obtained', 0)
            
            self.logger.info(f"Progress loaded from {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load progress: {e}")
            return False

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