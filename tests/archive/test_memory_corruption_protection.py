#!/usr/bin/env python3
"""
test_memory_corruption_protection.py - Tests for memory corruption protection

Tests the memory corruption protection systems that prevent false reward spikes
from uninitialized memory reads in the early game, including:
- Badge corruption detection and sanitization
- Level corruption detection and sanitization  
- Reward system anti-glitch measures
- Memory state validation
- Integration with the trainer's reward calculator
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from environments.state.memory_map import DERIVED_VALUES, get_badges_earned, BADGE_MASKS
from llm_trainer import PokemonRewardCalculator

pytestmark = pytest.mark.skip("Memory corruption protection tests disabled - testing archived functionality")


@pytest.mark.memory_corruption
@pytest.mark.unit
class TestMemoryCorruptionProtection:
    """Test memory corruption protection in derived values"""
    
    def test_badges_total_corruption_protection(self):
        """Test that badges_total handles memory corruption gracefully"""
        badges_total = DERIVED_VALUES['badges_total']
        
        # Early game conditions: no party, no level
        early_game_base = {'party_count': 0, 'player_level': 0}
        
        # Test 0xFF corruption (common uninitialized value)
        state = {**early_game_base, 'badges': 0xFF, 'kanto_badges': 0}
        assert badges_total(state) == 0, "0xFF Johto badges in early game should return 0"
        
        state = {**early_game_base, 'badges': 0, 'kanto_badges': 0xFF}
        assert badges_total(state) == 0, "0xFF Kanto badges in early game should return 0"
        
        state = {**early_game_base, 'badges': 0xFF, 'kanto_badges': 0xFF}
        assert badges_total(state) == 0, "0xFF both badges in early game should return 0"
        
        # Test values > 0x80 corruption
        state = {**early_game_base, 'badges': 0x90, 'kanto_badges': 0}
        assert badges_total(state) == 0, "High Johto badge value in early game should return 0"
        
        state = {**early_game_base, 'badges': 0, 'kanto_badges': 0xA0}
        assert badges_total(state) == 0, "High Kanto badge value in early game should return 0"
        
        # Test impossible level corruption
        state = {'party_count': 1, 'player_level': 122, 'badges': 0xFF, 'kanto_badges': 0xFF}
        assert badges_total(state) == 0, "Impossible level should trigger corruption protection"
        
        state = {'party_count': 1, 'player_level': 255, 'badges': 0x0F, 'kanto_badges': 0x07}
        assert badges_total(state) == 0, "Level 255 should trigger corruption protection"
    
    def test_badges_total_valid_cases(self):
        """Test that badges_total works correctly for valid cases"""
        badges_total = DERIVED_VALUES['badges_total']
        
        # Valid mid-game state
        valid_state_base = {'party_count': 3, 'player_level': 25}
        
        # Test normal badge progression
        state = {**valid_state_base, 'badges': 0x0F, 'kanto_badges': 0}  # 4 badges
        assert badges_total(state) == 4
        
        state = {**valid_state_base, 'badges': 0, 'kanto_badges': 0x07}  # 3 badges
        assert badges_total(state) == 3
        
        state = {**valid_state_base, 'badges': 0x0F, 'kanto_badges': 0x07}  # 7 badges
        assert badges_total(state) == 7
        
        # Test valid endgame state
        endgame_state = {'party_count': 6, 'player_level': 80}
        state = {**endgame_state, 'badges': 0xFF, 'kanto_badges': 0xFF}  # All 16 badges
        assert badges_total(state) == 16
    
    def test_badges_total_edge_cases(self):
        """Test edge cases for badges_total corruption protection"""
        badges_total = DERIVED_VALUES['badges_total']
        
        # Test party_count = 0 but level > 0 (still early game)
        state = {'party_count': 0, 'player_level': 5, 'badges': 0xFF, 'kanto_badges': 0}
        assert badges_total(state) == 0, "No party but some level should still trigger protection"
        
        # Test level = 0 but party_count > 0 (still early game)
        state = {'party_count': 1, 'player_level': 0, 'badges': 0xFF, 'kanto_badges': 0}
        assert badges_total(state) == 0, "Some party but no level should still trigger protection"
        
        # Test exactly level 100 (valid endgame)
        state = {'party_count': 6, 'player_level': 100, 'badges': 0xFF, 'kanto_badges': 0xFF}
        assert badges_total(state) == 16, "Level 100 should be valid"
        
        # Test level 101 (impossible, should trigger protection)
        state = {'party_count': 6, 'player_level': 101, 'badges': 0xFF, 'kanto_badges': 0xFF}
        assert badges_total(state) == 0, "Level 101 should trigger protection"
    
    def test_badges_total_capping(self):
        """Test that badges_total caps at maximum possible badges"""
        badges_total = DERIVED_VALUES['badges_total']
        
        # Test maximum valid badges
        state = {'party_count': 6, 'player_level': 50, 'badges': 0xFF, 'kanto_badges': 0xFF}
        result = badges_total(state)
        assert result == 16, f"Maximum badges should be 16, got {result}"
        
        # The function should inherently cap due to bit counting, but verify
        assert result <= 16, "Badge count should never exceed 16"


@pytest.mark.memory_corruption
@pytest.mark.unit
class TestRewardSystemCorruptionProtection:
    """Test reward system protection against memory corruption"""
    
    @pytest.fixture
    def reward_calculator(self):
        """Create a reward calculator instance for testing"""
        return PokemonRewardCalculator()
    
    def test_badge_reward_corruption_protection(self, reward_calculator):
        """Test badge reward corruption protection"""
        
        # Early game conditions
        early_game_current = {
            'badges_total': 8,  # False positive from 0xFF
            'badges': 0xFF,
            'kanto_badges': 0,
            'party_count': 0,
            'player_level': 0
        }
        
        early_game_previous = {
            'badges_total': 0,
            'badges': 0,
            'kanto_badges': 0,
            'party_count': 0,
            'player_level': 0
        }
        
        # This should return 0 reward due to corruption protection
        reward = reward_calculator._calculate_badge_reward(early_game_current, early_game_previous)
        assert reward == 0.0, "Early game 0xFF corruption should not grant badge rewards"
        
        # Test with both badges corrupted
        early_game_current['kanto_badges'] = 0xFF
        reward = reward_calculator._calculate_badge_reward(early_game_current, early_game_previous)
        assert reward == 0.0, "Early game 0xFF corruption in both badges should not grant rewards"
        
        # Test corruption in previous state
        early_game_previous['badges'] = 0xFF
        reward = reward_calculator._calculate_badge_reward(early_game_current, early_game_previous)
        assert reward == 0.0, "0xFF corruption in previous state should not grant rewards"
    
    def test_badge_reward_valid_cases(self, reward_calculator):
        """Test badge reward system with valid cases"""
        
        # Valid mid-game badge earning
        valid_current = {
            'badges_total': 3,
            'badges': 0x07,  # 3 badges
            'kanto_badges': 0,
            'party_count': 2,
            'player_level': 15
        }
        
        valid_previous = {
            'badges_total': 2,
            'badges': 0x03,  # 2 badges
            'kanto_badges': 0,
            'party_count': 2,
            'player_level': 15
        }
        
        reward = reward_calculator._calculate_badge_reward(valid_current, valid_previous)
        assert reward == 500.0, "Valid badge gain should grant 500 points per badge"
    
    def test_badge_reward_gain_capping(self, reward_calculator):
        """Test that badge gains are capped to prevent huge spikes"""
        
        # Simulate impossible multi-badge jump (could happen due to memory issues)
        current_state = {
            'badges_total': 8,  # Jumped from 0 to 8
            'badges': 0xFF,
            'kanto_badges': 0,
            'party_count': 3,
            'player_level': 20
        }
        
        previous_state = {
            'badges_total': 0,
            'badges': 0,
            'kanto_badges': 0,
            'party_count': 3,
            'player_level': 20
        }
        
        reward = reward_calculator._calculate_badge_reward(current_state, previous_state)
        assert reward == 500.0, "Badge gains should be capped to 1 badge per step (500 points max)"
    
    def test_badge_reward_out_of_range_protection(self, reward_calculator):
        """Test protection against impossible badge counts"""
        
        # Test negative badges (shouldn't happen but protect anyway)
        current_state = {
            'badges_total': -1,
            'badges': 0,
            'kanto_badges': 0,
            'party_count': 1,
            'player_level': 5
        }
        
        previous_state = {
            'badges_total': 0,
            'badges': 0,
            'kanto_badges': 0,
            'party_count': 1,
            'player_level': 5
        }
        
        reward = reward_calculator._calculate_badge_reward(current_state, previous_state)
        assert reward == 0.0, "Negative badge count should not grant rewards"
        
        # Test impossibly high badge count (>16)
        current_state = {
            'badges_total': 25,
            'badges': 0xFF,
            'kanto_badges': 0xFF,
            'party_count': 1,
            'player_level': 5
        }
        
        reward = reward_calculator._calculate_badge_reward(current_state, previous_state)
        assert reward == 0.0, "Impossible badge count (>16) should not grant rewards"
    
    def test_level_reward_corruption_protection(self, reward_calculator):
        """Test level reward corruption protection"""
        
        # Test impossible level spike
        current_state = {'player_level': 122}  # Impossible level
        previous_state = {'player_level': 5}
        
        reward = reward_calculator._calculate_level_reward(current_state, previous_state)
        assert reward == 0.0, "Impossible level should not grant rewards"
        
        # Test corruption in previous state
        current_state = {'player_level': 10}
        previous_state = {'player_level': 150}  # Impossible previous level
        
        reward = reward_calculator._calculate_level_reward(current_state, previous_state)
        assert reward == 0.0, "Impossible previous level should not grant rewards"
    
    def test_level_reward_gain_capping(self, reward_calculator):
        """Test that level gains are capped to prevent huge spikes"""
        
        # Test massive level jump (shouldn't happen normally)
        current_state = {'player_level': 50}  # Jumped 45 levels at once
        previous_state = {'player_level': 5}
        
        reward = reward_calculator._calculate_level_reward(current_state, previous_state)
        expected_reward = 5 * 50.0  # Capped at 5 levels max = 250 points
        assert reward == expected_reward, f"Level gains should be capped at 5 levels per step, got {reward}"
    
    def test_level_reward_valid_cases(self, reward_calculator):
        """Test level reward system with valid cases"""
        
        # Normal 1-level gain
        current_state = {'player_level': 6}
        previous_state = {'player_level': 5}
        
        reward = reward_calculator._calculate_level_reward(current_state, previous_state)
        assert reward == 50.0, "Normal 1-level gain should grant 50 points"
        
        # Normal 2-level gain
        current_state = {'player_level': 7}
        previous_state = {'player_level': 5}
        
        reward = reward_calculator._calculate_level_reward(current_state, previous_state)
        assert reward == 100.0, "2-level gain should grant 100 points"
        
        # Valid 5-level gain (at the cap)
        current_state = {'player_level': 10}
        previous_state = {'player_level': 5}
        
        reward = reward_calculator._calculate_level_reward(current_state, previous_state)
        assert reward == 250.0, "5-level gain should grant 250 points"


@pytest.mark.memory_corruption
@pytest.mark.integration
class TestMemoryCorruptionIntegration:
    """Test memory corruption protection in integration scenarios"""
    
    @pytest.fixture
    def reward_calculator(self):
        """Create a reward calculator for integration testing"""
        return PokemonRewardCalculator()
    
    def test_early_game_corruption_scenario(self, reward_calculator):
        """Test complete early game corruption scenario"""
        
        # Simulate early game state with memory corruption
        corrupted_current = {
            'badges': 0xFF,     # Corrupted memory
            'kanto_badges': 0,
            'party_count': 0,   # No Pokemon yet
            'player_level': 122, # Impossible level
            'player_hp': 0,
            'player_max_hp': 0,
            'money': 0,
            'player_x': 0,
            'player_y': 0,
            'player_map': 0
        }
        
        # Calculate badges_total using the safe function (simulates trainer behavior)
        corrupted_current['badges_total'] = DERIVED_VALUES['badges_total'](corrupted_current)
        
        clean_previous = {
            'badges_total': 0,
            'badges': 0,
            'kanto_badges': 0,
            'party_count': 0,
            'player_level': 0,
            'player_hp': 0,
            'player_max_hp': 0,
            'money': 0,
            'player_x': 0,
            'player_y': 0,
            'player_map': 0
        }
        
        # Calculate full reward breakdown
        total_reward, reward_breakdown = reward_calculator.calculate_reward(
            corrupted_current, clean_previous
        )
        
        # Badge rewards should be 0 due to corruption protection
        assert reward_breakdown['badges'] == 0.0, "Badge rewards should be blocked"
        
        # Level rewards should be 0 due to impossible level
        assert reward_breakdown['level'] == 0.0, "Level rewards should be blocked"
        
        # Total reward should be minimal (just time penalty)
        assert total_reward < 1.0, f"Total reward should be minimal, got {total_reward}"
        
        # Should not have the massive 4000+ point spike
        assert total_reward < 100.0, "Should not have massive reward spike from corruption"
    
    def test_normal_progression_not_affected(self, reward_calculator):
        """Test that normal progression is not affected by corruption protection"""
        
        # Normal badge earning scenario
        current_state = {
            'badges_total': 1,
            'badges': 0x01,  # First badge (ZEPHYR)
            'kanto_badges': 0,
            'party_count': 1,
            'player_level': 12,
            'player_hp': 45,
            'player_max_hp': 50,
            'money': 1500
        }
        
        previous_state = {
            'badges_total': 0,
            'badges': 0,
            'kanto_badges': 0,
            'party_count': 1,
            'player_level': 11,
            'player_hp': 45,
            'player_max_hp': 50,
            'money': 1200
        }
        
        total_reward, reward_breakdown = reward_calculator.calculate_reward(
            current_state, previous_state
        )
        
        # Should get proper badge reward
        assert reward_breakdown['badges'] == 500.0, "Normal badge earning should work"
        
        # Should get proper level reward  
        assert reward_breakdown['level'] == 50.0, "Normal level gain should work"
        
        # Should get money reward
        assert reward_breakdown['money'] > 0, "Money increase should give reward"
        
        # Total should be substantial but reasonable
        assert total_reward > 500.0, "Normal progression should give good rewards"
        assert total_reward < 1000.0, "But not excessive rewards"
    
    def test_mixed_corruption_scenarios(self, reward_calculator):
        """Test various mixed corruption scenarios"""
        
        scenarios = [
            {
                'name': 'Johto badges corrupted only',
                'current': {'badges': 0xFF, 'kanto_badges': 0, 'party_count': 0, 'player_level': 0},
                'previous': {'badges': 0, 'kanto_badges': 0, 'party_count': 0, 'player_level': 0},
                'expected_badge_reward': 0.0
            },
            {
                'name': 'Kanto badges corrupted only',
                'current': {'badges': 0, 'kanto_badges': 0xFF, 'party_count': 0, 'player_level': 0},
                'previous': {'badges': 0, 'kanto_badges': 0, 'party_count': 0, 'player_level': 0},
                'expected_badge_reward': 0.0
            },
            {
                'name': 'High badge values',
                'current': {'badges': 0x90, 'kanto_badges': 0xA0, 'party_count': 0, 'player_level': 0},
                'previous': {'badges': 0, 'kanto_badges': 0, 'party_count': 0, 'player_level': 0},
                'expected_badge_reward': 0.0
            },
            {
                'name': 'Valid progression',
                'current': {'badges': 0x03, 'kanto_badges': 0, 'party_count': 2, 'player_level': 15},
                'previous': {'badges': 0x01, 'kanto_badges': 0, 'party_count': 2, 'player_level': 15},
                'expected_badge_reward': 500.0  # 1 badge gained
            }
        ]
        
        for scenario in scenarios:
            # Add badges_total based on the manual calculation
            current = scenario['current'].copy()
            previous = scenario['previous'].copy()
            
            # Manually calculate badges_total using our protected function
            current['badges_total'] = DERIVED_VALUES['badges_total'](current)
            previous['badges_total'] = DERIVED_VALUES['badges_total'](previous)
            
            reward = reward_calculator._calculate_badge_reward(current, previous)
            
            assert reward == scenario['expected_badge_reward'], \
                f"Scenario '{scenario['name']}' failed: expected {scenario['expected_badge_reward']}, got {reward}"
    
    def test_corruption_protection_doesnt_break_endgame(self, reward_calculator):
        """Test that corruption protection doesn't interfere with valid endgame scenarios"""
        
        # Simulate completing Elite Four (all 16 badges earned legitimately)
        endgame_current = {
            'badges_total': 16,
            'badges': 0xFF,
            'kanto_badges': 0xFF,
            'party_count': 6,
            'player_level': 55,  # Valid endgame level
            'player_hp': 200,
            'player_max_hp': 200
        }
        
        near_endgame_previous = {
            'badges_total': 15,
            'badges': 0x7F,  # Missing one badge
            'kanto_badges': 0xFF,
            'party_count': 6,
            'player_level': 55,
            'player_hp': 200,
            'player_max_hp': 200
        }
        
        reward = reward_calculator._calculate_badge_reward(endgame_current, near_endgame_previous)
        assert reward == 500.0, "Valid endgame badge earning should work normally"
        
        # Test level 100 (maximum possible)
        level_100_current = {'player_level': 100}
        level_99_previous = {'player_level': 99}
        
        reward = reward_calculator._calculate_level_reward(level_100_current, level_99_previous)
        assert reward == 50.0, "Reaching level 100 should grant normal rewards"


@pytest.mark.memory_corruption
@pytest.mark.performance
class TestCorruptionProtectionPerformance:
    """Test that corruption protection doesn't significantly impact performance"""
    
    def test_badges_total_performance(self):
        """Test performance of badges_total with corruption checks"""
        import time
        badges_total = DERIVED_VALUES['badges_total']
        
        # Test data with various corruption patterns
        test_states = [
            {'badges': 0, 'kanto_badges': 0, 'party_count': 1, 'player_level': 5},
            {'badges': 0xFF, 'kanto_badges': 0, 'party_count': 0, 'player_level': 0},
            {'badges': 0x0F, 'kanto_badges': 0x07, 'party_count': 3, 'player_level': 25},
            {'badges': 0x90, 'kanto_badges': 0xA0, 'party_count': 0, 'player_level': 0},
            {'badges': 0xFF, 'kanto_badges': 0xFF, 'party_count': 6, 'player_level': 50},
        ]
        
        # Time the function calls
        iterations = 1000
        start_time = time.time()
        
        for _ in range(iterations):
            for state in test_states:
                badges_total(state)
        
        end_time = time.time()
        duration = end_time - start_time
        calls_per_second = (iterations * len(test_states)) / duration
        
        # Should be able to handle thousands of calls per second
        assert calls_per_second > 1000, f"Performance too slow: {calls_per_second:.1f} calls/sec"
        
        # Should complete in reasonable time
        assert duration < 1.0, f"Function calls took too long: {duration:.3f}s"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-m", "memory_corruption"])
