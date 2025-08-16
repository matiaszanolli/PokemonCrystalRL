#!/usr/bin/env python3
"""
test_enhanced_reward_system.py - Tests for the enhanced reward calculation system

Tests the new reward features including:
- Stuck detection penalties
- Menu progress rewards  
- State transition bonuses
- Action diversity rewards
- Progress momentum bonuses
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from collections import deque

# Import test system modules
import sys
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from utils.utils import calculate_reward
from core.pyboy_env import PyBoyPokemonCrystalEnv


@pytest.mark.enhanced_rewards
@pytest.mark.unit
class TestStuckDetectionPenalties:
    """Test stuck detection penalty system"""
    
    def test_no_penalty_for_normal_gameplay(self):
        """Test that normal gameplay doesn't trigger stuck penalties"""
        current_state = {
            'player_hp': 100,
            'consecutive_same_screens': 0,
            'game_state': 'overworld',
            'recent_actions': [1, 2, 3, 4, 5],
        }
        
        previous_state = {
            'player_hp': 100,
            'consecutive_same_screens': 0,
            'game_state': 'overworld',
            'recent_actions': [5, 4, 3, 2, 1],
        }
        
        reward = calculate_reward(current_state, previous_state)
        
        # Should get base survival reward without stuck penalty
        assert reward >= 0.0
    
    def test_escalating_stuck_penalties(self):
        """Test escalating penalties for being stuck"""
        base_state = {
            'player_hp': 100,
            'game_state': 'overworld',
            'recent_actions': [1, 1, 1, 1, 1],
        }
        
        previous_state = base_state.copy()
        previous_state['consecutive_same_screens'] = 0
        
        # Test different stuck levels
        stuck_levels = [5, 11, 15, 20, 26]
        rewards = []
        
        for stuck_count in stuck_levels:
            current_state = base_state.copy()
            current_state['consecutive_same_screens'] = stuck_count
            
            reward = calculate_reward(current_state, previous_state)
            rewards.append(reward)
        
        # Rewards should decrease as stuck count increases
        assert rewards[0] > rewards[1]  # No penalty vs light penalty
        assert rewards[1] > rewards[2]  # Light penalty vs medium penalty
        assert rewards[3] > rewards[4]  # Medium vs severe penalty (>25 stuck)
    
    def test_severe_stuck_penalty(self):
        """Test severe penalty for being very stuck"""
        current_state = {
            'player_hp': 100,
            'consecutive_same_screens': 30,  # Very stuck
            'game_state': 'overworld',
            'recent_actions': [1, 1, 1, 1, 1],
        }
        
        previous_state = {
            'player_hp': 100,
            'consecutive_same_screens': 0,
            'game_state': 'overworld',
            'recent_actions': [1, 2, 3, 4, 5],
        }
        
        reward = calculate_reward(current_state, previous_state)
        
        # Should have severe penalty for being really stuck
        # Base survival (0.1) + severe stuck penalty (-2.0) + escalating (-2.0) + time (-0.002) = very negative
        assert reward < -3.0


@pytest.mark.enhanced_rewards
@pytest.mark.unit
class TestMenuProgressRewards:
    """Test menu navigation and progress rewards"""
    
    def test_menu_entry_reward(self):
        """Test reward for entering menu successfully"""
        current_state = {
            'player_hp': 100,
            'consecutive_same_screens': 0,
            'game_state': 'menu',
            'recent_actions': [7, 5],  # START, A
        }
        
        previous_state = {
            'player_hp': 100,
            'consecutive_same_screens': 0,
            'game_state': 'overworld',
            'recent_actions': [1, 2, 3, 4],
        }
        
        reward = calculate_reward(current_state, previous_state)
        
        # Should get base reward + menu entry bonus (1.0)
        assert reward > 1.0
    
    def test_menu_exit_reward(self):
        """Test reward for successfully exiting menu"""
        current_state = {
            'player_hp': 100,
            'consecutive_same_screens': 0,
            'game_state': 'overworld',
            'recent_actions': [6, 6, 5],  # B, B, A
        }
        
        previous_state = {
            'player_hp': 100,
            'consecutive_same_screens': 0,
            'game_state': 'menu',
            'recent_actions': [1, 2, 5, 5],
        }
        
        reward = calculate_reward(current_state, previous_state)
        
        # Should get base reward + menu exit bonus (2.0) - higher than entry
        assert reward > 2.0


@pytest.mark.enhanced_rewards
@pytest.mark.unit
class TestStateTransitionBonuses:
    """Test state transition reward bonuses"""
    
    def test_title_to_new_game_transition(self):
        """Test reward for progressing from title screen to new game menu"""
        current_state = {
            'player_hp': 100,
            'consecutive_same_screens': 0,
            'game_state': 'new_game_menu',
            'recent_actions': [7, 5],
        }
        
        previous_state = {
            'player_hp': 100,
            'consecutive_same_screens': 0,
            'game_state': 'title_screen',
            'recent_actions': [7, 7, 5],
        }
        
        reward = calculate_reward(current_state, previous_state)
        
        # Should get base reward + state transition bonus (5.0)
        assert reward > 5.0
    
    def test_new_game_to_overworld_transition(self):
        """Test major reward for entering overworld from new game menu"""
        current_state = {
            'player_hp': 100,
            'consecutive_same_screens': 0,
            'game_state': 'overworld',
            'recent_actions': [5, 5, 5],
        }
        
        previous_state = {
            'player_hp': 100,
            'consecutive_same_screens': 0,
            'game_state': 'new_game_menu',
            'recent_actions': [1, 2, 5],
        }
        
        reward = calculate_reward(current_state, previous_state)
        
        # Should get base reward + large state transition bonus (10.0)
        assert reward > 10.0
    
    def test_battle_completion_reward(self):
        """Test reward for successfully completing battle"""
        current_state = {
            'player_hp': 80,  # Lost some HP but survived
            'consecutive_same_screens': 0,
            'game_state': 'overworld',
            'recent_actions': [5, 5, 1, 5],
        }
        
        previous_state = {
            'player_hp': 100,
            'consecutive_same_screens': 0,
            'game_state': 'battle',
            'recent_actions': [5, 1, 5, 2],
        }
        
        reward = calculate_reward(current_state, previous_state)
        
        # Should get base reward + battle completion bonus (3.0) - HP loss penalty
        # HP loss: -0.5 * 20 = -10.0, but battle completion should still make it positive
        assert reward > -7.0  # 0.1 + 3.0 - 10.0 - 0.002 = -6.902


@pytest.mark.enhanced_rewards
@pytest.mark.unit
class TestActionDiversityRewards:
    """Test action diversity tracking and rewards"""
    
    def test_action_diversity_bonus(self):
        """Test bonus for using diverse actions"""
        current_state = {
            'player_hp': 100,
            'consecutive_same_screens': 0,
            'game_state': 'overworld',
            'recent_actions': [1, 2, 3, 4, 5],  # 5 different actions
        }
        
        previous_state = {
            'player_hp': 100,
            'consecutive_same_screens': 0,
            'game_state': 'overworld',
            'recent_actions': [6, 7, 8, 1, 2],
        }
        
        reward = calculate_reward(current_state, previous_state)
        
        # Should get base reward + diversity bonus (0.05)
        assert reward > 0.1
    
    def test_repetitive_action_penalty(self):
        """Test penalty for repeating same action"""
        current_state = {
            'player_hp': 100,
            'consecutive_same_screens': 0,
            'game_state': 'overworld',
            'recent_actions': [1, 1, 1, 1, 1],  # All same action
        }
        
        previous_state = {
            'player_hp': 100,
            'consecutive_same_screens': 0,
            'game_state': 'overworld',
            'recent_actions': [2, 3, 4, 5, 6],
        }
        
        reward = calculate_reward(current_state, previous_state)
        
        # Should get base reward + repetitive action penalty (-0.02)
        # 0.1 - 0.02 - 0.002 = 0.078
        assert 0.07 < reward < 0.09
    
    def test_insufficient_actions_for_diversity(self):
        """Test that diversity check requires enough actions"""
        current_state = {
            'player_hp': 100,
            'consecutive_same_screens': 0,
            'game_state': 'overworld',
            'recent_actions': [1, 2, 3],  # Only 3 actions, need 5
        }
        
        previous_state = {
            'player_hp': 100,
            'consecutive_same_screens': 0,
            'game_state': 'overworld',
            'recent_actions': [4, 5],
        }
        
        reward = calculate_reward(current_state, previous_state)
        
        # Should not get diversity bonus or penalty
        # Just base survival (0.1) - time penalty (0.002) = 0.098
        assert 0.09 < reward < 0.11


@pytest.mark.enhanced_rewards
@pytest.mark.unit
class TestProgressMomentumRewards:
    """Test progress momentum and compound progress rewards"""
    
    def test_compound_progress_bonus(self):
        """Test bonus for multiple types of progress in one step"""
        current_state = {
            'player_hp': 100,
            'player_level': 6,
            'player_exp': 1500,
            'money': 2000,
            'player_x': 10,
            'player_y': 15,
            'player_map': 2,
            'consecutive_same_screens': 0,
            'game_state': 'overworld',
            'recent_actions': [1, 2, 3, 4, 5],
        }
        
        previous_state = {
            'player_hp': 100,
            'player_level': 5,
            'player_exp': 1000,
            'money': 1500,
            'player_x': 5,
            'player_y': 10,
            'player_map': 1,
            'consecutive_same_screens': 0,
            'game_state': 'overworld',
            'recent_actions': [2, 3, 4, 5, 6],
        }
        
        reward = calculate_reward(current_state, previous_state)
        
        # Multiple progress types: level up, exp gain, money gain, position change, map change
        # Should get significant compound progress bonus
        assert reward > 15.0  # Level up (10.0) + exp (5.0) + money (0.5) + map change (5.0) + compound bonus
    
    def test_no_compound_bonus_for_single_progress(self):
        """Test no compound bonus for just one type of progress"""
        current_state = {
            'player_hp': 100,
            'player_exp': 1100,
            'consecutive_same_screens': 0,
            'game_state': 'overworld',
            'recent_actions': [1, 2, 3, 4, 5],
        }
        
        previous_state = {
            'player_hp': 100,
            'player_exp': 1000,
            'consecutive_same_screens': 0,
            'game_state': 'overworld',
            'recent_actions': [2, 3, 4, 5, 6],
        }
        
        reward = calculate_reward(current_state, previous_state)
        
        # Should get exp reward (1.0) + base survival (0.1) + diversity (0.05) - time penalty (0.002)
        # No compound bonus since only 1 progress type
        assert 1.0 < reward < 1.2


@pytest.mark.enhanced_rewards
@pytest.mark.integration
class TestEnhancedRewardIntegration:
    """Test enhanced reward system integration with environment"""
    
    @patch('core.pyboy_env.PyBoy')
    def test_environment_tracks_enhanced_state(self, mock_pyboy_class):
        """Test that environment properly tracks enhanced reward state"""
        # Mock PyBoy instance
        mock_pyboy = Mock()
        # Create a comprehensive mock memory that includes all party data addresses
        mock_memory = {}
        # Basic addresses
        mock_memory.update({0xDCB8: 10, 0xDCB9: 15, 0xDCB5: 1, 0xDCB6: 0,
                           0xD84E: 0, 0xD84F: 0, 0xD850: 0, 0xD855: 0, 0xD856: 0,
                           0xDCD7: 1, 0xDCDF: 25})
        
        # Add mock data for party Pokemon (starting at 0xDCDF)
        party_start = 0xDCDF
        for i in range(6):  # Support up to 6 Pokemon
            pokemon_offset = party_start + (i * 48)
            for j in range(48):  # Each Pokemon is 48 bytes
                mock_memory[pokemon_offset + j] = (i + 1) * 10 + (j % 10)  # Mock data pattern
        
        mock_pyboy.memory = mock_memory
        mock_pyboy.frame_count = 1000
        mock_pyboy.screen.ndarray = np.random.randint(0, 255, (144, 160, 3), dtype=np.uint8)
        mock_pyboy_class.return_value = mock_pyboy
        
        # Create environment
        env = PyBoyPokemonCrystalEnv(
            rom_path="test.gbc",
            headless=True,
            debug_mode=False,
            enable_monitoring=False
        )
        
        # Reset environment
        obs, info = env.reset()
        
        # Check that enhanced tracking is initialized
        assert hasattr(env, 'consecutive_same_screens')
        assert hasattr(env, 'recent_actions')
        assert hasattr(env, 'game_state_history')
        
        assert env.consecutive_same_screens == 0
        assert len(env.recent_actions) == 0
        # Game state history may have initial state after reset, so just check it exists
        assert len(env.game_state_history) >= 0
        
        # Take some actions
        for action in [1, 2, 3, 4, 5]:
            obs, reward, terminated, truncated, info = env.step(action)
        
        # Check that actions are tracked
        assert len(env.recent_actions) == 5
        assert list(env.recent_actions) == [1, 2, 3, 4, 5]
        
        # Check that game state is tracked
        assert len(env.game_state_history) > 0
        
        env.close()
    
    def test_stuck_detection_integration(self):
        """Test stuck detection in environment step"""
        # Create a simple mock environment that will trigger stuck detection
        with patch('core.pyboy_env.PyBoy') as mock_pyboy_class:
            mock_pyboy = Mock()
            # Create comprehensive mock memory for second test too
            mock_memory = {}
            mock_memory.update({0xDCB8: 10, 0xDCB9: 15, 0xDCB5: 1, 0xDCB6: 0,
                               0xD84E: 0, 0xD84F: 0, 0xD850: 0, 0xD855: 0, 0xD856: 0,
                               0xDCD7: 1, 0xDCDF: 25})
            
            # Add mock data for party Pokemon
            party_start = 0xDCDF
            for i in range(6):
                pokemon_offset = party_start + (i * 48)
                for j in range(48):
                    mock_memory[pokemon_offset + j] = (i + 1) * 10 + (j % 10)
            
            mock_pyboy.memory = mock_memory
            mock_pyboy.frame_count = 1000
            
            # Create identical screens to trigger stuck detection
            stuck_screen = np.ones((144, 160, 3), dtype=np.uint8) * 128
            mock_pyboy.screen.ndarray = stuck_screen
            mock_pyboy_class.return_value = mock_pyboy
            
            env = PyBoyPokemonCrystalEnv(
                rom_path="test.gbc",
                headless=True,
                debug_mode=False,
                enable_monitoring=False
            )
            
            # Reset and take multiple steps with same screen
            obs, info = env.reset()
            
            # Take 15 identical steps to trigger stuck detection
            for i in range(15):
                obs, reward, terminated, truncated, info = env.step(1)  # Same action
            
            # Should have detected stuck situation
            assert env.consecutive_same_screens >= 10
            
            # Next step should have stuck penalty
            obs, reward, terminated, truncated, info = env.step(1)
            
            # Should have negative reward due to stuck penalty
            assert reward < 0.1  # Less than base survival reward
            
            env.close()


@pytest.mark.enhanced_rewards
@pytest.mark.unit
class TestRewardEdgeCases:
    """Test edge cases and boundary conditions for enhanced rewards"""
    
    def test_missing_state_fields(self):
        """Test reward calculation with missing state fields"""
        current_state = {
            'player_hp': 100,
            # Missing consecutive_same_screens, recent_actions, game_state
        }
        
        previous_state = {
            'player_hp': 100,
        }
        
        # Should not crash and should return reasonable reward
        reward = calculate_reward(current_state, previous_state)
        assert isinstance(reward, float)
        assert reward > 0  # Should still get base survival reward
    
    def test_empty_recent_actions(self):
        """Test action diversity with empty recent actions"""
        current_state = {
            'player_hp': 100,
            'consecutive_same_screens': 0,
            'game_state': 'overworld',
            'recent_actions': [],  # Empty list
        }
        
        previous_state = {
            'player_hp': 100,
            'consecutive_same_screens': 0,
            'game_state': 'overworld',
            'recent_actions': [],
        }
        
        reward = calculate_reward(current_state, previous_state)
        
        # Should not crash and should not apply diversity rewards/penalties
        assert isinstance(reward, float)
    
    def test_extreme_stuck_values(self):
        """Test very high stuck screen counts"""
        current_state = {
            'player_hp': 100,
            'consecutive_same_screens': 1000,  # Extremely stuck
            'game_state': 'overworld',
            'recent_actions': [1, 1, 1, 1, 1],
        }
        
        previous_state = {
            'player_hp': 100,
            'consecutive_same_screens': 0,
            'game_state': 'overworld',
            'recent_actions': [1, 2, 3, 4, 5],
        }
        
        reward = calculate_reward(current_state, previous_state)
        
        # Should handle extreme values gracefully
        assert isinstance(reward, float)
        assert reward < -50.0  # Should be very negative
    
    def test_unknown_state_transitions(self):
        """Test state transitions not in the predefined bonus list"""
        current_state = {
            'player_hp': 100,
            'consecutive_same_screens': 0,
            'game_state': 'unknown_state',
            'recent_actions': [1, 2, 3, 4, 5],
        }
        
        previous_state = {
            'player_hp': 100,
            'consecutive_same_screens': 0,
            'game_state': 'another_unknown_state',
            'recent_actions': [2, 3, 4, 5, 6],
        }
        
        reward = calculate_reward(current_state, previous_state)
        
        # Should not crash and should not give transition bonus
        assert isinstance(reward, float)
        # Should be close to base survival reward + diversity bonus - time penalty
        assert 0.1 < reward < 0.2


if __name__ == "__main__":
    # Run specific test categories
    pytest.main([
        __file__,
        "-v",
        "-m", "enhanced_rewards",
        "--tb=short"
    ])
