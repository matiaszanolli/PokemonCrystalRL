"""
Enhanced Badge Protection Tests

Comprehensive tests for the enhanced badge protection system that prevents
false badge detection due to memory corruption in early game states.
"""

import pytest
from rewards.components.progress import BadgeRewardComponent
from rewards.calculator import PokemonRewardCalculator


class TestEnhancedBadgeProtection:
    """Test enhanced badge protection in both component and main calculator."""

    def setup_method(self):
        """Set up test fixtures."""
        self.component = BadgeRewardComponent()
        self.calculator = PokemonRewardCalculator()

    def test_component_blocks_no_pokemon(self):
        """Badge component should block rewards when party_count = 0."""
        current = {
            'badges_total': 1,
            'badges': 1,
            'party_count': 0,
            'player_level': 5,
            'player_hp': 20,
            'player_max_hp': 24
        }
        previous = {
            'badges_total': 0,
            'badges': 0,
            'party_count': 0,
            'player_level': 5
        }

        reward, details = self.component.calculate(current, previous)
        assert reward == 0.0
        assert details == {}

    def test_component_blocks_no_level(self):
        """Badge component should block rewards when player_level = 0."""
        current = {
            'badges_total': 1,
            'badges': 1,
            'party_count': 1,
            'player_level': 0,
            'player_hp': 0,
            'player_max_hp': 0
        }
        previous = {
            'badges_total': 0,
            'badges': 0,
            'party_count': 1,
            'player_level': 0
        }

        reward, details = self.component.calculate(current, previous)
        assert reward == 0.0
        assert details == {}

    def test_component_blocks_no_hp(self):
        """Badge component should block rewards when HP values are 0."""
        current = {
            'badges_total': 1,
            'badges': 1,
            'party_count': 1,
            'player_level': 1,
            'player_hp': 0,
            'player_max_hp': 0
        }
        previous = {
            'badges_total': 0,
            'badges': 0,
            'party_count': 1,
            'player_level': 1
        }

        reward, details = self.component.calculate(current, previous)
        assert reward == 0.0
        assert details == {}

    def test_component_blocks_early_game_combo(self):
        """Badge component should block rewards in early game combination scenarios."""
        current = {
            'badges_total': 1,
            'badges': 1,
            'party_count': 0,
            'player_level': 3,  # <= 5 and party_count = 0
            'player_hp': 10,
            'player_max_hp': 15
        }
        previous = {
            'badges_total': 0,
            'badges': 0,
            'party_count': 0,
            'player_level': 3
        }

        reward, details = self.component.calculate(current, previous)
        assert reward == 0.0
        assert details == {}

    def test_component_allows_valid_badges(self):
        """Badge component should allow rewards for valid badge progression."""
        current = {
            'badges_total': 1,
            'badges': 1,
            'party_count': 2,
            'player_level': 15,
            'player_hp': 45,
            'player_max_hp': 50
        }
        previous = {
            'badges_total': 0,
            'badges': 0,
            'party_count': 2,
            'player_level': 15
        }

        reward, details = self.component.calculate(current, previous)
        assert reward == 500.0  # 1 badge * 500.0
        assert 'badge_earned' in details

    def test_main_calculator_blocks_no_pokemon(self):
        """Main calculator should block badge rewards when party_count = 0."""
        current = {
            'badges_total': 1,
            'badges': 1,
            'party_count': 0,
            'player_level': 5,
            'player_hp': 20,
            'player_max_hp': 24
        }
        previous = {
            'badges_total': 0,
            'badges': 0,
            'party_count': 0,
            'player_level': 5
        }

        reward = self.calculator._calculate_badge_reward(current, previous)
        assert reward == 0.0

    def test_main_calculator_blocks_no_level(self):
        """Main calculator should block badge rewards when player_level = 0."""
        current = {
            'badges_total': 1,
            'badges': 1,
            'party_count': 1,
            'player_level': 0,
            'player_hp': 0,
            'player_max_hp': 0
        }
        previous = {
            'badges_total': 0,
            'badges': 0,
            'party_count': 1,
            'player_level': 0
        }

        reward = self.calculator._calculate_badge_reward(current, previous)
        assert reward == 0.0

    def test_main_calculator_blocks_memory_corruption(self):
        """Main calculator should block badge rewards with memory corruption indicators."""
        current = {
            'badges_total': 2,
            'badges': 0xFF,  # Memory corruption
            'party_count': 0,
            'player_level': 0,
            'player_hp': 0,
            'player_max_hp': 0
        }
        previous = {
            'badges_total': 0,
            'badges': 0,
            'party_count': 0,
            'player_level': 0
        }

        reward = self.calculator._calculate_badge_reward(current, previous)
        assert reward == 0.0

    def test_main_calculator_allows_valid_badges(self):
        """Main calculator should allow rewards for valid badge progression."""
        current = {
            'badges_total': 1,
            'badges': 1,
            'party_count': 2,
            'player_level': 15,
            'player_hp': 45,
            'player_max_hp': 50
        }
        previous = {
            'badges_total': 0,
            'badges': 0,
            'party_count': 2,
            'player_level': 15
        }

        reward = self.calculator._calculate_badge_reward(current, previous)
        assert reward == 500.0  # 1 badge * 500.0

    def test_milestone_prevention(self):
        """Both systems should prevent duplicate badge rewards for same milestone."""
        current = {
            'badges_total': 1,
            'badges': 1,
            'party_count': 2,
            'player_level': 15,
            'player_hp': 45,
            'player_max_hp': 50
        }
        previous = {
            'badges_total': 0,
            'badges': 0,
            'party_count': 2,
            'player_level': 15
        }

        # First call should give reward
        reward1 = self.calculator._calculate_badge_reward(current, previous)
        assert reward1 == 500.0

        # Second call with same milestone should give no reward
        reward2 = self.calculator._calculate_badge_reward(current, previous)
        assert reward2 == 0.0

    def test_realistic_corruption_scenario(self):
        """Test realistic memory corruption scenario from logs."""
        # Based on actual log data that was causing 18,000+ point rewards
        current = {
            'badges_total': 2,  # False badge detection
            'badges': 2,
            'party_count': 0,   # No Pokemon (should block)
            'player_level': 0,  # No level (should block)
            'player_hp': 0,     # No HP (should block)
            'player_max_hp': 0, # No max HP (should block)
            'current_map': 3,
            'money': 0
        }
        previous = {
            'badges_total': 0,
            'badges': 0,
            'party_count': 0,
            'player_level': 0,
            'player_hp': 0,
            'player_max_hp': 0
        }

        # Both systems should block this
        component_reward, _ = self.component.calculate(current, previous)
        calculator_reward = self.calculator._calculate_badge_reward(current, previous)

        assert component_reward == 0.0
        assert calculator_reward == 0.0