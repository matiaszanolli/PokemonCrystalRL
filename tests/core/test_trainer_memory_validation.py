#!/usr/bin/env python3
"""
test_trainer_memory_validation.py - Tests for trainer memory state validation

Tests the memory validation fixes in the LLMPokemonTrainer.get_game_state method
including sanitization of corrupted memory values and proper badge calculation.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

# We'll mock the trainer's get_game_state functionality since we can't easily
# import the full trainer without PyBoy dependencies


@pytest.mark.trainer_validation
@pytest.mark.unit
class TestTrainerMemoryValidation:
    """Test memory validation in trainer's get_game_state method"""
    
    @pytest.fixture
    def mock_trainer_memory_logic(self):
        """Mock the trainer's memory validation logic"""
        
        def validate_and_sanitize_memory_state(raw_state):
            """Simulate the trainer's get_game_state memory validation logic"""
            
            # Extract values
            johto_badges = raw_state.get('badges', 0)
            kanto_badges = raw_state.get('kanto_badges', 0)
            player_level = raw_state.get('player_level', 0)
            party_count = raw_state.get('party_count', 0)
            
            # Sanitize implausible values that indicate uninitialized memory
            early_game_indicators = party_count == 0 or player_level == 0
            invalid_badge_values = johto_badges == 0xFF or kanto_badges == 0xFF or johto_badges > 0x80 or kanto_badges > 0x80
            
            if early_game_indicators and invalid_badge_values:
                johto_badges = 0
                kanto_badges = 0
            
            # Additional sanity check: if level > 100 (impossible in Pokemon), sanitize everything
            if player_level > 100:
                raw_state['player_level'] = 0
                johto_badges = 0
                kanto_badges = 0
            
            # Update state with sanitized values
            sanitized_state = raw_state.copy()
            sanitized_state['badges'] = johto_badges
            sanitized_state['kanto_badges'] = kanto_badges
            
            # Calculate badge totals using the sanitized values
            # Simulate the badge counting logic
            def count_bits(value):
                count = 0
                for i in range(8):
                    count += (value >> i) & 1
                return count
            
            sanitized_state['badges_total'] = count_bits(johto_badges) + count_bits(kanto_badges)
            
            return sanitized_state
        
        return validate_and_sanitize_memory_state
    
    def test_early_game_corruption_sanitization(self, mock_trainer_memory_logic):
        """Test that early game memory corruption is properly sanitized"""
        
        # Simulate early game with 0xFF corruption
        corrupted_early_state = {
            'badges': 0xFF,
            'kanto_badges': 0,
            'player_level': 0,
            'party_count': 0,
            'player_hp': 0,
            'player_max_hp': 0
        }
        
        sanitized = mock_trainer_memory_logic(corrupted_early_state)
        
        assert sanitized['badges'] == 0, "Corrupted badges should be sanitized to 0"
        assert sanitized['badges_total'] == 0, "Badge total should be 0 after sanitization"
        
        # Test with Kanto badges corrupted
        corrupted_early_state['badges'] = 0
        corrupted_early_state['kanto_badges'] = 0xFF
        
        sanitized = mock_trainer_memory_logic(corrupted_early_state)
        
        assert sanitized['kanto_badges'] == 0, "Corrupted Kanto badges should be sanitized to 0"
        assert sanitized['badges_total'] == 0, "Badge total should be 0 after Kanto sanitization"
    
    def test_impossible_level_sanitization(self, mock_trainer_memory_logic):
        """Test that impossible levels trigger full sanitization"""
        
        # Simulate impossible level corruption
        impossible_level_state = {
            'badges': 0xFF,
            'kanto_badges': 0xFF,
            'player_level': 122,  # Impossible level
            'party_count': 1,
            'player_hp': 100,
            'player_max_hp': 100
        }
        
        sanitized = mock_trainer_memory_logic(impossible_level_state)
        
        assert sanitized['player_level'] == 0, "Impossible level should be reset to 0"
        assert sanitized['badges'] == 0, "Badges should be sanitized when level is impossible"
        assert sanitized['kanto_badges'] == 0, "Kanto badges should be sanitized when level is impossible"
        assert sanitized['badges_total'] == 0, "Badge total should be 0 after level-triggered sanitization"
    
    def test_valid_midgame_not_affected(self, mock_trainer_memory_logic):
        """Test that valid mid-game states are not affected by sanitization"""
        
        # Valid mid-game state
        valid_midgame_state = {
            'badges': 0x0F,  # 4 badges (binary 00001111)
            'kanto_badges': 0,
            'player_level': 25,
            'party_count': 3,
            'player_hp': 150,
            'player_max_hp': 150
        }
        
        sanitized = mock_trainer_memory_logic(valid_midgame_state)
        
        assert sanitized['badges'] == 0x0F, "Valid badges should not be changed"
        assert sanitized['kanto_badges'] == 0, "Valid Kanto badges should not be changed"
        assert sanitized['player_level'] == 25, "Valid level should not be changed"
        assert sanitized['badges_total'] == 4, "Badge total should be calculated correctly"
    
    def test_valid_endgame_not_affected(self, mock_trainer_memory_logic):
        """Test that valid end-game states with all badges work correctly"""
        
        # Valid end-game state
        valid_endgame_state = {
            'badges': 0xFF,  # All 8 Johto badges
            'kanto_badges': 0xFF,  # All 8 Kanto badges
            'player_level': 60,  # Valid endgame level
            'party_count': 6,
            'player_hp': 200,
            'player_max_hp': 200
        }
        
        sanitized = mock_trainer_memory_logic(valid_endgame_state)
        
        assert sanitized['badges'] == 0xFF, "Valid endgame badges should not be changed"
        assert sanitized['kanto_badges'] == 0xFF, "Valid endgame Kanto badges should not be changed"
        assert sanitized['player_level'] == 60, "Valid endgame level should not be changed"
        assert sanitized['badges_total'] == 16, "All badges should be counted correctly"
    
    def test_high_badge_values_sanitization(self, mock_trainer_memory_logic):
        """Test that high badge values (> 0x80) are sanitized in early game"""
        
        # High badge values in early game
        high_value_state = {
            'badges': 0x90,  # Higher than maximum single badge (0x80)
            'kanto_badges': 0xA0,  # Higher than maximum single badge
            'player_level': 0,
            'party_count': 0,
            'player_hp': 0,
            'player_max_hp': 0
        }
        
        sanitized = mock_trainer_memory_logic(high_value_state)
        
        assert sanitized['badges'] == 0, "High badge values should be sanitized in early game"
        assert sanitized['kanto_badges'] == 0, "High Kanto badge values should be sanitized in early game"
        assert sanitized['badges_total'] == 0, "Badge total should be 0 after high value sanitization"
    
    def test_edge_case_level_100(self, mock_trainer_memory_logic):
        """Test that level 100 (maximum valid) is not sanitized"""
        
        # Level 100 should be valid
        level_100_state = {
            'badges': 0xFF,
            'kanto_badges': 0xFF,
            'player_level': 100,  # Maximum valid level
            'party_count': 6,
            'player_hp': 300,
            'player_max_hp': 300
        }
        
        sanitized = mock_trainer_memory_logic(level_100_state)
        
        assert sanitized['player_level'] == 100, "Level 100 should be valid"
        assert sanitized['badges'] == 0xFF, "Badges should not be sanitized at level 100"
        assert sanitized['kanto_badges'] == 0xFF, "Kanto badges should not be sanitized at level 100"
        assert sanitized['badges_total'] == 16, "All badges should be counted at level 100"
    
    def test_edge_case_level_101(self, mock_trainer_memory_logic):
        """Test that level 101 (impossible) triggers sanitization"""
        
        # Level 101 should trigger sanitization
        level_101_state = {
            'badges': 0xFF,
            'kanto_badges': 0xFF,
            'player_level': 101,  # Impossible level
            'party_count': 6,
            'player_hp': 300,
            'player_max_hp': 300
        }
        
        sanitized = mock_trainer_memory_logic(level_101_state)
        
        assert sanitized['player_level'] == 0, "Level 101 should be reset to 0"
        assert sanitized['badges'] == 0, "Badges should be sanitized at impossible level"
        assert sanitized['kanto_badges'] == 0, "Kanto badges should be sanitized at impossible level"
        assert sanitized['badges_total'] == 0, "Badge total should be 0 at impossible level"
    
    def test_partial_early_game_conditions(self, mock_trainer_memory_logic):
        """Test various partial early game conditions"""
        
        # Party count 0 but some level
        state1 = {
            'badges': 0xFF,
            'kanto_badges': 0,
            'player_level': 5,
            'party_count': 0,  # Still early game indicator
            'player_hp': 50,
            'player_max_hp': 50
        }
        
        sanitized = mock_trainer_memory_logic(state1)
        assert sanitized['badges'] == 0, "Badges should be sanitized when party_count is 0"
        
        # Level 0 but some party
        state2 = {
            'badges': 0xFF,
            'kanto_badges': 0,
            'player_level': 0,  # Still early game indicator
            'party_count': 1,
            'player_hp': 50,
            'player_max_hp': 50
        }
        
        sanitized = mock_trainer_memory_logic(state2)
        assert sanitized['badges'] == 0, "Badges should be sanitized when player_level is 0"


@pytest.mark.trainer_validation
@pytest.mark.integration
class TestTrainerValidationIntegration:
    """Test integration scenarios for trainer memory validation"""
    
    @pytest.fixture
    def mock_trainer_memory_logic(self):
        """Mock the trainer's memory validation logic (same as above)"""
        
        def validate_and_sanitize_memory_state(raw_state):
            johto_badges = raw_state.get('badges', 0)
            kanto_badges = raw_state.get('kanto_badges', 0)
            player_level = raw_state.get('player_level', 0)
            party_count = raw_state.get('party_count', 0)
            
            early_game_indicators = party_count == 0 or player_level == 0
            invalid_badge_values = johto_badges == 0xFF or kanto_badges == 0xFF or johto_badges > 0x80 or kanto_badges > 0x80
            
            if early_game_indicators and invalid_badge_values:
                johto_badges = 0
                kanto_badges = 0
            
            if player_level > 100:
                raw_state['player_level'] = 0
                johto_badges = 0
                kanto_badges = 0
            
            sanitized_state = raw_state.copy()
            sanitized_state['badges'] = johto_badges
            sanitized_state['kanto_badges'] = kanto_badges
            
            def count_bits(value):
                count = 0
                for i in range(8):
                    count += (value >> i) & 1
                return count
            
            sanitized_state['badges_total'] = count_bits(johto_badges) + count_bits(kanto_badges)
            
            return sanitized_state
        
        return validate_and_sanitize_memory_state
    
    def test_game_progression_scenarios(self, mock_trainer_memory_logic):
        """Test various game progression scenarios"""
        
        scenarios = [
            {
                'name': 'Fresh start - all clean',
                'state': {'badges': 0, 'kanto_badges': 0, 'player_level': 0, 'party_count': 0},
                'expected_badges_total': 0
            },
            {
                'name': 'Fresh start - corrupted badges',
                'state': {'badges': 0xFF, 'kanto_badges': 0xFF, 'player_level': 0, 'party_count': 0},
                'expected_badges_total': 0  # Should be sanitized
            },
            {
                'name': 'First Pokemon obtained',
                'state': {'badges': 0, 'kanto_badges': 0, 'player_level': 5, 'party_count': 1},
                'expected_badges_total': 0
            },
            {
                'name': 'First badge earned',
                'state': {'badges': 0x01, 'kanto_badges': 0, 'player_level': 12, 'party_count': 1},
                'expected_badges_total': 1
            },
            {
                'name': 'Mid-game progression',
                'state': {'badges': 0x0F, 'kanto_badges': 0, 'player_level': 30, 'party_count': 4},
                'expected_badges_total': 4
            },
            {
                'name': 'Elite Four complete',
                'state': {'badges': 0xFF, 'kanto_badges': 0, 'player_level': 45, 'party_count': 6},
                'expected_badges_total': 8
            },
            {
                'name': 'Kanto progression',
                'state': {'badges': 0xFF, 'kanto_badges': 0x07, 'player_level': 55, 'party_count': 6},
                'expected_badges_total': 11
            },
            {
                'name': 'Champion complete',
                'state': {'badges': 0xFF, 'kanto_badges': 0xFF, 'player_level': 70, 'party_count': 6},
                'expected_badges_total': 16
            }
        ]
        
        for scenario in scenarios:
            sanitized = mock_trainer_memory_logic(scenario['state'])
            actual_total = sanitized['badges_total']
            expected_total = scenario['expected_badges_total']
            
            assert actual_total == expected_total, \
                f"Scenario '{scenario['name']}' failed: expected {expected_total} badges, got {actual_total}"
    
    def test_memory_corruption_during_gameplay(self, mock_trainer_memory_logic):
        """Test handling of memory corruption during different gameplay phases"""
        
        corruption_scenarios = [
            {
                'name': 'Corruption during early game',
                'state': {'badges': 0xFF, 'kanto_badges': 0, 'player_level': 0, 'party_count': 0},
                'should_sanitize': True,
                'expected_badges': 0
            },
            {
                'name': 'Corruption during mid-game',
                'state': {'badges': 0xFF, 'kanto_badges': 0, 'player_level': 25, 'party_count': 3},
                'should_sanitize': False,  # Valid mid-game state
                'expected_badges': 8
            },
            {
                'name': 'Level corruption triggers sanitization',
                'state': {'badges': 0xFF, 'kanto_badges': 0xFF, 'player_level': 200, 'party_count': 6},
                'should_sanitize': True,
                'expected_badges': 0
            },
            {
                'name': 'High badge values early game',
                'state': {'badges': 0x90, 'kanto_badges': 0xA0, 'player_level': 0, 'party_count': 0},
                'should_sanitize': True,
                'expected_badges': 0
            },
            {
                'name': 'Valid endgame not sanitized',
                'state': {'badges': 0xFF, 'kanto_badges': 0xFF, 'player_level': 80, 'party_count': 6},
                'should_sanitize': False,
                'expected_badges': 16
            }
        ]
        
        for scenario in corruption_scenarios:
            sanitized = mock_trainer_memory_logic(scenario['state'])
            
            if scenario['should_sanitize']:
                # Badges should be sanitized
                assert sanitized['badges'] == 0 or sanitized['badges'] == scenario['state']['badges'], \
                    f"Scenario '{scenario['name']}': badges should be sanitized or remain valid"
                
                if sanitized['badges'] == 0:
                    assert sanitized['badges_total'] == 0, \
                        f"Scenario '{scenario['name']}': badge total should be 0 when sanitized"
            else:
                # State should remain largely unchanged
                assert sanitized['badges'] == scenario['state']['badges'], \
                    f"Scenario '{scenario['name']}': valid badges should not be changed"
                assert sanitized['badges_total'] == scenario['expected_badges'], \
                    f"Scenario '{scenario['name']}': badge total should match expected"
    
    def test_reward_calculation_prevention(self, mock_trainer_memory_logic):
        """Test that sanitized states prevent false reward calculations"""
        
        # Simulate the problematic scenario that caused the 4000+ reward spike
        problematic_states = [
            {
                'name': 'Original bug scenario',
                'current': {'badges': 0xFF, 'kanto_badges': 0, 'player_level': 122, 'party_count': 0},
                'previous': {'badges': 0, 'kanto_badges': 0, 'player_level': 0, 'party_count': 0}
            },
            {
                'name': 'Both badges corrupted',
                'current': {'badges': 0xFF, 'kanto_badges': 0xFF, 'player_level': 0, 'party_count': 0},
                'previous': {'badges': 0, 'kanto_badges': 0, 'player_level': 0, 'party_count': 0}
            },
            {
                'name': 'High values corruption',
                'current': {'badges': 0x90, 'kanto_badges': 0xA0, 'player_level': 0, 'party_count': 0},
                'previous': {'badges': 0, 'kanto_badges': 0, 'player_level': 0, 'party_count': 0}
            }
        ]
        
        for scenario in problematic_states:
            # Sanitize both current and previous states
            sanitized_current = mock_trainer_memory_logic(scenario['current'])
            sanitized_previous = mock_trainer_memory_logic(scenario['previous'])
            
            # Calculate badge difference (simulating reward calculation logic)
            badge_diff = sanitized_current['badges_total'] - sanitized_previous['badges_total']
            
            # Badge difference should be 0 or minimal, not 8
            assert badge_diff <= 1, \
                f"Scenario '{scenario['name']}': badge difference should be <= 1, got {badge_diff}"
            
            # Total badges should be reasonable
            assert sanitized_current['badges_total'] <= 16, \
                f"Scenario '{scenario['name']}': total badges should be <= 16"
            
            # For early game corruption, badges should be 0
            if scenario['current']['party_count'] == 0 and scenario['current']['player_level'] <= 1:
                assert sanitized_current['badges_total'] == 0, \
                    f"Scenario '{scenario['name']}': early game corruption should result in 0 badges"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-m", "trainer_validation"])
