#!/usr/bin/env python3
"""
Test Suite for Constants Module

Tests all game constants including locations, Pokemon species, status conditions,
badge masks, derived values, screen states, actions, and training parameters.
"""

import pytest

from config.constants import (
    LOCATIONS, POKEMON_SPECIES, STATUS_CONDITIONS, BADGE_MASKS,
    DERIVED_VALUES, SCREEN_STATES, AVAILABLE_ACTIONS, MOVEMENT_DIRECTIONS,
    SCREEN_DIMENSIONS, TRAINING_PARAMS, REWARD_VALUES
)


class TestLocations:
    """Test LOCATIONS constant definitions."""

    def test_locations_exists(self):
        """Test that LOCATIONS dictionary exists and is not empty."""
        assert LOCATIONS is not None
        assert isinstance(LOCATIONS, dict)
        assert len(LOCATIONS) > 0

    def test_locations_structure(self):
        """Test LOCATIONS dictionary structure."""
        for location_id, location_name in LOCATIONS.items():
            assert isinstance(location_id, int)
            assert isinstance(location_name, str)
            assert len(location_name) > 0

    def test_key_locations(self):
        """Test that key locations are defined."""
        expected_locations = {
            24: "Player's Bedroom",
            25: "Player's House",
            26: "New Bark Town",
            27: "Prof. Elm's Lab"
        }

        for location_id, expected_name in expected_locations.items():
            assert location_id in LOCATIONS
            assert LOCATIONS[location_id] == expected_name

    def test_location_ids_range(self):
        """Test that location IDs are in reasonable range."""
        for location_id in LOCATIONS.keys():
            assert 0 <= location_id <= 255  # Game Boy address space


class TestPokemonSpecies:
    """Test POKEMON_SPECIES constant definitions."""

    def test_pokemon_species_exists(self):
        """Test that POKEMON_SPECIES dictionary exists."""
        assert POKEMON_SPECIES is not None
        assert isinstance(POKEMON_SPECIES, dict)
        assert len(POKEMON_SPECIES) > 0

    def test_pokemon_species_structure(self):
        """Test POKEMON_SPECIES dictionary structure."""
        for species_id, species_name in POKEMON_SPECIES.items():
            assert isinstance(species_id, int)
            assert isinstance(species_name, str)
            assert len(species_name) > 0

    def test_starter_pokemon(self):
        """Test that starter Pokemon are defined."""
        expected_starters = {
            152: "Chikorita",
            155: "Cyndaquil",
            158: "Totodile"
        }

        for species_id, expected_name in expected_starters.items():
            assert species_id in POKEMON_SPECIES
            assert POKEMON_SPECIES[species_id] == expected_name

    def test_common_pokemon(self):
        """Test that common Pokemon are defined."""
        common_pokemon = [
            (0, "None"),
            (16, "Pidgey"),
            (19, "Rattata"),
            (129, "Magikarp")
        ]

        for species_id, expected_name in common_pokemon:
            assert species_id in POKEMON_SPECIES
            assert POKEMON_SPECIES[species_id] == expected_name

    def test_pokemon_ids_range(self):
        """Test that Pokemon IDs are in valid range."""
        for species_id in POKEMON_SPECIES.keys():
            assert 0 <= species_id <= 255  # Game Boy Pokemon ID range


class TestStatusConditions:
    """Test STATUS_CONDITIONS constant definitions."""

    def test_status_conditions_exists(self):
        """Test that STATUS_CONDITIONS dictionary exists."""
        assert STATUS_CONDITIONS is not None
        assert isinstance(STATUS_CONDITIONS, dict)
        assert len(STATUS_CONDITIONS) > 0

    def test_status_conditions_structure(self):
        """Test STATUS_CONDITIONS dictionary structure."""
        for status_id, status_name in STATUS_CONDITIONS.items():
            assert isinstance(status_id, int)
            assert isinstance(status_name, str)
            assert len(status_name) > 0

    def test_standard_status_conditions(self):
        """Test that standard status conditions are defined."""
        expected_conditions = {
            0: "Healthy",
            1: "Sleep",
            2: "Poison",
            3: "Burn",
            4: "Freeze",
            5: "Paralysis"
        }

        for status_id, expected_name in expected_conditions.items():
            assert status_id in STATUS_CONDITIONS
            assert STATUS_CONDITIONS[status_id] == expected_name

    def test_status_ids_range(self):
        """Test that status IDs are in valid range."""
        for status_id in STATUS_CONDITIONS.keys():
            assert 0 <= status_id <= 15  # Reasonable range for status effects


class TestBadgeMasks:
    """Test BADGE_MASKS constant definitions."""

    def test_badge_masks_exists(self):
        """Test that BADGE_MASKS dictionary exists."""
        assert BADGE_MASKS is not None
        assert isinstance(BADGE_MASKS, dict)
        assert len(BADGE_MASKS) > 0

    def test_johto_badges(self):
        """Test Johto badge masks."""
        assert 'johto' in BADGE_MASKS
        johto_badges = BADGE_MASKS['johto']

        expected_badges = {
            'zephyr': 0x01,
            'hive': 0x02,
            'plain': 0x04,
            'fog': 0x08,
            'storm': 0x10,
            'mineral': 0x20,
            'glacier': 0x40,
            'rising': 0x80
        }

        for badge_name, expected_mask in expected_badges.items():
            assert badge_name in johto_badges
            assert johto_badges[badge_name] == expected_mask

    def test_badge_mask_powers_of_two(self):
        """Test that badge masks are powers of two (bitfield)."""
        johto_badges = BADGE_MASKS['johto']

        for badge_name, mask in johto_badges.items():
            # Check that mask is a power of two
            assert mask > 0
            assert (mask & (mask - 1)) == 0  # Power of two check

    def test_badge_mask_uniqueness(self):
        """Test that all badge masks are unique."""
        johto_badges = BADGE_MASKS['johto']
        masks = list(johto_badges.values())
        assert len(masks) == len(set(masks))  # All unique


class TestDerivedValues:
    """Test DERIVED_VALUES constant definitions."""

    def test_derived_values_exists(self):
        """Test that DERIVED_VALUES dictionary exists."""
        assert DERIVED_VALUES is not None
        assert isinstance(DERIVED_VALUES, dict)
        assert len(DERIVED_VALUES) > 0

    def test_derived_value_functions(self):
        """Test that derived value functions work correctly."""
        # Test badges_total function
        assert 'badges_total' in DERIVED_VALUES
        badges_func = DERIVED_VALUES['badges_total']
        assert callable(badges_func)
        assert badges_func({'badges_count': 5}) == 5
        assert badges_func({}) == 0  # Default value

        # Test health_percentage function
        assert 'health_percentage' in DERIVED_VALUES
        health_func = DERIVED_VALUES['health_percentage']
        assert callable(health_func)
        assert health_func({'player_hp': 50, 'player_max_hp': 100}) == 50.0
        assert health_func({'player_hp': 100, 'player_max_hp': 100}) == 100.0
        assert health_func({}) == 0.0  # Default values

        # Test has_pokemon function
        assert 'has_pokemon' in DERIVED_VALUES
        pokemon_func = DERIVED_VALUES['has_pokemon']
        assert callable(pokemon_func)
        assert pokemon_func({'party_count': 1}) is True
        assert pokemon_func({'party_count': 0}) is False
        assert pokemon_func({}) is False  # Default value

        # Test location_key function
        assert 'location_key' in DERIVED_VALUES
        location_func = DERIVED_VALUES['location_key']
        assert callable(location_func)
        assert location_func({'player_map': 1, 'player_x': 10, 'player_y': 20}) == "1_10_20"
        assert location_func({}) == "0_0_0"  # Default values

    def test_health_percentage_edge_cases(self):
        """Test health percentage calculation edge cases."""
        health_func = DERIVED_VALUES['health_percentage']

        # Division by zero protection
        assert health_func({'player_hp': 10, 'player_max_hp': 0}) == 1000.0  # Protected by max(1)

        # Normal cases
        assert health_func({'player_hp': 25, 'player_max_hp': 50}) == 50.0
        assert health_func({'player_hp': 0, 'player_max_hp': 100}) == 0.0


class TestScreenStates:
    """Test SCREEN_STATES constant definitions."""

    def test_screen_states_exists(self):
        """Test that SCREEN_STATES dictionary exists."""
        assert SCREEN_STATES is not None
        assert isinstance(SCREEN_STATES, dict)
        assert len(SCREEN_STATES) > 0

    def test_standard_screen_states(self):
        """Test that standard screen states are defined."""
        expected_states = {
            'LOADING': 'loading',
            'DIALOGUE': 'dialogue',
            'MENU': 'menu',
            'BATTLE': 'battle',
            'OVERWORLD': 'overworld',
            'SETTINGS_MENU': 'settings_menu',
            'UNKNOWN': 'unknown'
        }

        for state_key, expected_value in expected_states.items():
            assert state_key in SCREEN_STATES
            assert SCREEN_STATES[state_key] == expected_value

    def test_screen_state_values_format(self):
        """Test that screen state values are properly formatted."""
        for state_key, state_value in SCREEN_STATES.items():
            assert isinstance(state_key, str)
            assert isinstance(state_value, str)
            assert state_key.isupper()  # Constants should be uppercase
            assert state_value.islower() or '_' in state_value  # Values should be lowercase/snake_case


class TestAvailableActions:
    """Test AVAILABLE_ACTIONS constant definitions."""

    def test_available_actions_exists(self):
        """Test that AVAILABLE_ACTIONS dictionary exists."""
        assert AVAILABLE_ACTIONS is not None
        assert isinstance(AVAILABLE_ACTIONS, dict)
        assert len(AVAILABLE_ACTIONS) > 0

    def test_movement_actions(self):
        """Test movement actions are properly defined."""
        assert 'MOVEMENT' in AVAILABLE_ACTIONS
        movement_actions = AVAILABLE_ACTIONS['MOVEMENT']
        expected_movements = ['up', 'down', 'left', 'right']

        assert isinstance(movement_actions, list)
        assert set(movement_actions) == set(expected_movements)

    def test_button_actions(self):
        """Test button actions are properly defined."""
        assert 'BUTTONS' in AVAILABLE_ACTIONS
        button_actions = AVAILABLE_ACTIONS['BUTTONS']
        expected_buttons = ['a', 'b', 'start', 'select']

        assert isinstance(button_actions, list)
        assert set(button_actions) == set(expected_buttons)

    def test_forbidden_initial_actions(self):
        """Test forbidden initial actions are properly defined."""
        assert 'FORBIDDEN_INITIAL' in AVAILABLE_ACTIONS
        forbidden_actions = AVAILABLE_ACTIONS['FORBIDDEN_INITIAL']
        expected_forbidden = ['start', 'select']

        assert isinstance(forbidden_actions, list)
        assert set(forbidden_actions) == set(expected_forbidden)

    def test_action_consistency(self):
        """Test that forbidden actions are subset of button actions."""
        button_actions = AVAILABLE_ACTIONS['BUTTONS']
        forbidden_actions = AVAILABLE_ACTIONS['FORBIDDEN_INITIAL']

        for action in forbidden_actions:
            assert action in button_actions


class TestMovementDirections:
    """Test MOVEMENT_DIRECTIONS constant definitions."""

    def test_movement_directions_exists(self):
        """Test that MOVEMENT_DIRECTIONS dictionary exists."""
        assert MOVEMENT_DIRECTIONS is not None
        assert isinstance(MOVEMENT_DIRECTIONS, dict)
        assert len(MOVEMENT_DIRECTIONS) > 0

    def test_standard_directions(self):
        """Test standard direction encodings."""
        expected_directions = {
            'DOWN': 0,
            'UP': 2,
            'LEFT': 4,
            'RIGHT': 6,
            'STANDING': 8
        }

        for direction, expected_value in expected_directions.items():
            assert direction in MOVEMENT_DIRECTIONS
            assert MOVEMENT_DIRECTIONS[direction] == expected_value

    def test_direction_values_even(self):
        """Test that direction values are even (Game Boy standard)."""
        for direction, value in MOVEMENT_DIRECTIONS.items():
            assert isinstance(value, int)
            assert value % 2 == 0  # Game Boy directions are even numbers


class TestScreenDimensions:
    """Test SCREEN_DIMENSIONS constant definitions."""

    def test_screen_dimensions_exists(self):
        """Test that SCREEN_DIMENSIONS dictionary exists."""
        assert SCREEN_DIMENSIONS is not None
        assert isinstance(SCREEN_DIMENSIONS, dict)
        assert len(SCREEN_DIMENSIONS) > 0

    def test_gameboy_dimensions(self):
        """Test Game Boy screen dimensions."""
        expected_dimensions = {
            'WIDTH': 160,
            'HEIGHT': 144,
            'PIXELS': 160 * 144
        }

        for dim_key, expected_value in expected_dimensions.items():
            assert dim_key in SCREEN_DIMENSIONS
            assert SCREEN_DIMENSIONS[dim_key] == expected_value

    def test_pixel_calculation(self):
        """Test that pixel count is calculated correctly."""
        width = SCREEN_DIMENSIONS['WIDTH']
        height = SCREEN_DIMENSIONS['HEIGHT']
        pixels = SCREEN_DIMENSIONS['PIXELS']

        assert pixels == width * height


class TestTrainingParams:
    """Test TRAINING_PARAMS constant definitions."""

    def test_training_params_exists(self):
        """Test that TRAINING_PARAMS dictionary exists."""
        assert TRAINING_PARAMS is not None
        assert isinstance(TRAINING_PARAMS, dict)
        assert len(TRAINING_PARAMS) > 0

    def test_standard_training_params(self):
        """Test standard training parameters."""
        expected_params = {
            'LLM_INTERVAL': 20,
            'STUCK_THRESHOLD': 100,
            'MAX_PARTY_SIZE': 6,
            'MAX_MONEY': 999999,
            'PARTY_SLOT_SIZE': 44,
            'MAX_LEVEL': 100
        }

        for param_key, expected_value in expected_params.items():
            assert param_key in TRAINING_PARAMS
            assert TRAINING_PARAMS[param_key] == expected_value

    def test_training_param_ranges(self):
        """Test that training parameters are in reasonable ranges."""
        assert TRAINING_PARAMS['LLM_INTERVAL'] > 0
        assert TRAINING_PARAMS['STUCK_THRESHOLD'] > 0
        assert TRAINING_PARAMS['MAX_PARTY_SIZE'] == 6  # Pokemon game standard
        assert TRAINING_PARAMS['MAX_MONEY'] > 0
        assert TRAINING_PARAMS['PARTY_SLOT_SIZE'] > 0
        assert TRAINING_PARAMS['MAX_LEVEL'] == 100  # Pokemon game standard


class TestRewardValues:
    """Test REWARD_VALUES constant definitions."""

    def test_reward_values_exists(self):
        """Test that REWARD_VALUES exists."""
        # REWARD_VALUES might not be defined yet, so test conditionally
        try:
            from config.constants import REWARD_VALUES
            assert REWARD_VALUES is not None
            assert isinstance(REWARD_VALUES, dict)
        except ImportError:
            # REWARD_VALUES not defined, skip test
            pytest.skip("REWARD_VALUES not yet implemented")


class TestConstantsIntegration:
    """Test integration between different constant definitions."""

    def test_movement_actions_match_directions(self):
        """Test that movement actions match direction definitions."""
        movement_actions = AVAILABLE_ACTIONS['MOVEMENT']
        direction_names = [name.lower() for name in MOVEMENT_DIRECTIONS.keys()
                          if name != 'STANDING']

        # Should have corresponding directions for most movements
        for action in movement_actions:
            assert any(action == direction for direction in direction_names)

    def test_pokemon_species_none_exists(self):
        """Test that 'None' species (ID 0) exists."""
        assert 0 in POKEMON_SPECIES
        assert POKEMON_SPECIES[0] == "None"

    def test_status_healthy_is_zero(self):
        """Test that healthy status is ID 0."""
        assert 0 in STATUS_CONDITIONS
        assert STATUS_CONDITIONS[0] == "Healthy"

    def test_badge_count_matches_masks(self):
        """Test that badge count matches expected number."""
        johto_badges = BADGE_MASKS['johto']
        assert len(johto_badges) == 8  # 8 Johto gym badges


if __name__ == "__main__":
    pytest.main([__file__, "-v"])