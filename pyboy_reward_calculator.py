#!/usr/bin/env python3
"""
Enhanced Pokemon Reward Calculator using PyBoy-based state detection

This replaces the unreliable memory address approach with PyBoy's 
built-in debugging features for more accurate reward calculation.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
from pyboy_state_detector import PyBoyStateDetector

class PyBoyRewardCalculator:
    """Enhanced reward calculator using PyBoy's reliable state detection"""
    
    def __init__(self, pyboy):
        self.pyboy = pyboy
        self.detector = PyBoyStateDetector(pyboy)
        
        # Reward tracking
        self.visited_game_states = set()
        self.last_reward_time = 0
        self.consecutive_stuck_actions = 0
        
        # Store previous state for comparison
        self.previous_state = None
    
    def calculate_reward(self, action: str) -> Tuple[float, Dict[str, float]]:
        """
        Calculate reward using PyBoy-based state detection
        
        Returns:
            Tuple of (total_reward, reward_breakdown)
        """
        # Get current state using PyBoy detector
        current_state = self.detector.get_comprehensive_state()
        
        # Initialize reward breakdown
        rewards = {
            'exploration': 0.0,
            'movement': 0.0,
            'area_change': 0.0,
            'map_transition': 0.0,
            'stuck_penalty': 0.0,
            'time_penalty': -0.01  # Small time penalty to encourage progress
        }
        
        if self.previous_state is not None:
            # 1. Exploration rewards using PyBoy detection
            rewards['exploration'] = self._calculate_exploration_reward(
                self.previous_state, current_state
            )
            
            # 2. Movement rewards based on coordinate changes
            rewards['movement'] = self._calculate_movement_reward(
                self.previous_state, current_state
            )
            
            # 3. Area change rewards (more sensitive than map transitions)
            rewards['area_change'] = self._calculate_area_change_reward(
                self.previous_state, current_state
            )
            
            # 4. Map transition rewards (big rewards for new areas)
            rewards['map_transition'] = self._calculate_map_transition_reward(
                self.previous_state, current_state
            )
            
            # 5. Anti-stuck penalty
            rewards['stuck_penalty'] = self._calculate_stuck_penalty(current_state)
        
        # Store current state for next iteration
        self.previous_state = current_state
        
        # Calculate total reward
        total_reward = sum(rewards.values())
        
        return total_reward, rewards
    
    def _calculate_exploration_reward(self, prev_state: Dict, curr_state: Dict) -> float:
        """Calculate exploration reward using game area analysis"""
        return self.detector.get_exploration_reward(prev_state, curr_state)
    
    def _calculate_movement_reward(self, prev_state: Dict, curr_state: Dict) -> float:
        """Reward coordinate-based movement"""
        prev_x = prev_state.get('memory_x', 0)
        prev_y = prev_state.get('memory_y', 0)
        curr_x = curr_state.get('memory_x', 0)
        curr_y = curr_state.get('memory_y', 0)
        
        # Check for coordinate movement
        if (prev_x != curr_x or prev_y != curr_y) and (curr_x > 0 or curr_y > 0):
            return 0.02  # Small reward for any movement
        
        return 0.0
    
    def _calculate_area_change_reward(self, prev_state: Dict, curr_state: Dict) -> float:
        """Reward changes in game area (more sensitive than map transitions)"""
        prev_sum = prev_state.get('game_area_sum', 0)
        curr_sum = curr_state.get('game_area_sum', 0)
        
        if abs(prev_sum - curr_sum) > 50:  # Even small area changes
            change_magnitude = min(abs(prev_sum - curr_sum) / 1000, 0.5)
            return change_magnitude
        
        return 0.0
    
    def _calculate_map_transition_reward(self, prev_state: Dict, curr_state: Dict) -> float:
        """Large reward for actual map transitions"""
        if self.detector.detect_map_transition(prev_state, curr_state):
            return 20.0  # Large reward for changing maps
        
        return 0.0
    
    def _calculate_stuck_penalty(self, curr_state: Dict) -> float:
        """Penalty for being stuck in the same position too long"""
        stuck_count = curr_state.get('stuck_count', 0)
        
        if stuck_count > 10:
            # Increasing penalty for being stuck
            penalty = -min((stuck_count - 10) * 0.01, 0.1)
            return penalty
        
        return 0.0
    
    def get_reward_summary(self, rewards: Dict[str, float]) -> str:
        """Get a human-readable summary of rewards"""
        active_rewards = []
        
        for category, value in rewards.items():
            if abs(value) > 0.001:  # Only show non-zero rewards
                if value > 0:
                    active_rewards.append(f"{category}: +{value:.3f}")
                else:
                    active_rewards.append(f"{category}: {value:.3f}")
        
        return ", ".join(active_rewards) if active_rewards else "no rewards"
    
    def should_explore_more_aggressively(self) -> bool:
        """Determine if we should try more aggressive exploration"""
        if not self.previous_state:
            return False
            
        stuck_count = self.previous_state.get('stuck_count', 0)
        return stuck_count > 20
    
    def get_exploration_suggestions(self) -> List[str]:
        """Get suggestions for better exploration based on current state"""
        if not self.previous_state:
            return []
        
        suggestions = []
        
        stuck_count = self.previous_state.get('stuck_count', 0)
        if stuck_count > 15:
            suggestions.append("Try pressing 'A' to interact with objects")
            suggestions.append("Try different button sequences like 'B' then movement")
            suggestions.append("Try 'START' to open menu, then 'B' to close")
        
        return suggestions


def test_pyboy_reward_calculator():
    """Test the PyBoy-based reward calculator"""
    from pyboy import PyBoy
    import os
    
    rom_path = 'roms/pokemon_crystal.gbc'
    save_state_path = rom_path + '.state'
    
    if not os.path.exists(save_state_path):
        print("No save state found!")
        return
    
    pyboy = PyBoy(rom_path, window='null', debug=True)
    
    # Load save state
    with open(save_state_path, 'rb') as f:
        pyboy.load_state(f)
    
    # Initialize calculator
    calculator = PyBoyRewardCalculator(pyboy)
    
    print("🧪 Testing PyBoy Reward Calculator")
    print("=" * 50)
    
    # Test different actions
    actions_to_test = ['down', 'up', 'left', 'right', 'a', 'b', 'start']
    
    total_rewards = 0
    
    for i, action in enumerate(actions_to_test):
        print(f"\\n🎮 Action {i+1}: {action.upper()}")
        
        # Perform the action
        pyboy.button_press(action)
        for _ in range(8):
            pyboy.tick()
        pyboy.button_release(action)
        
        # Wait for state to stabilize
        for _ in range(4):
            pyboy.tick()
        
        # Calculate reward
        reward, breakdown = calculator.calculate_reward(action)
        total_rewards += reward
        
        print(f"  Reward: {reward:.3f}")
        print(f"  Breakdown: {calculator.get_reward_summary(breakdown)}")
        
        # Check for exploration suggestions
        if calculator.should_explore_more_aggressively():
            suggestions = calculator.get_exploration_suggestions()
            if suggestions:
                print(f"  💡 Suggestions: {', '.join(suggestions)}")
        
        # If we got a large reward, we probably found something good
        if reward > 1.0:
            print(f"  🎉 Large reward detected! This action might have triggered something important.")
    
    print(f"\\n📊 Total rewards over {len(actions_to_test)} actions: {total_rewards:.3f}")
    print(f"Average reward per action: {total_rewards/len(actions_to_test):.3f}")
    
    pyboy.stop()
    print("\\n✅ PyBoy reward calculator test complete!")


if __name__ == "__main__":
    test_pyboy_reward_calculator()
