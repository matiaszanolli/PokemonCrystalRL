"""
Memory Corruption Protection Tests

Tests for the enhanced memory reading system that handles corruption
gracefully and maintains stable position readings.
"""

import pytest
from unittest.mock import MagicMock
from utils.memory_reader import (
    get_safe_memory,
    read_location,
    read_party_pokemon,
    build_observation,
    _position_history,
    _last_good_position
)


class TestMemoryCorruptionProtection:
    """Test memory corruption detection and protection mechanisms."""

    def setup_method(self):
        """Reset global state before each test."""
        global _position_history, _last_good_position
        _position_history.clear()
        _last_good_position = (1, 5, 4, 0)

    def test_get_safe_memory_bounds_checking(self):
        """get_safe_memory should handle out-of-bounds values."""
        memory = [100, 200, 50]

        # Valid read
        assert get_safe_memory(memory, 1) == 200

        # Out of bounds
        assert get_safe_memory(memory, 10, default=42) == 42

        # Invalid value type
        memory_bad = [100, "invalid", 50]
        assert get_safe_memory(memory_bad, 1, default=0) == 0

    def test_read_location_corruption_detection(self):
        """read_location should detect and handle memory corruption patterns."""
        # Mock memory with corruption patterns
        memory = MagicMock()

        # Test 1: All zeros corruption
        memory.__getitem__.side_effect = lambda addr: {
            0x1000: 0,  # map_id
            0x1001: 0,  # x
            0x1002: 0,  # y
            0x1003: 0   # facing
        }.get(addr, 0)

        with pytest.MonkeyPatch.context() as m:
            m.setattr('utils.memory_reader.MEMORY_ADDRESSES', {
                'player_map': 0x1000,
                'player_x': 0x1001,
                'player_y': 0x1002,
                'player_direction': 0x1003
            })

            result = read_location(memory)
            # Should return last good position, not corrupted (0,0,0,0)
            assert result == (1, 5, 4, 0)

    def test_read_location_invalid_coordinates(self):
        """read_location should handle invalid coordinate values."""
        memory = MagicMock()

        # Test invalid coordinates (0xFF)
        memory.__getitem__.side_effect = lambda addr: {
            0x1000: 3,     # valid map_id
            0x1001: 0xFF,  # invalid x
            0x1002: 0xFF,  # invalid y
            0x1003: 0      # valid facing
        }.get(addr, 0)

        with pytest.MonkeyPatch.context() as m:
            m.setattr('utils.memory_reader.MEMORY_ADDRESSES', {
                'player_map': 0x1000,
                'player_x': 0x1001,
                'player_y': 0x1002,
                'player_direction': 0x1003
            })

            result = read_location(memory)
            # Should return last good position
            assert result == (1, 5, 4, 0)

    def test_read_location_valid_position_tracking(self):
        """read_location should track valid positions and update last good position."""
        memory = MagicMock()

        # Valid position data
        memory.__getitem__.side_effect = lambda addr: {
            0x1000: 4,  # map_id
            0x1001: 10, # x
            0x1002: 15, # y
            0x1003: 2   # facing
        }.get(addr, 0)

        with pytest.MonkeyPatch.context() as m:
            m.setattr('utils.memory_reader.MEMORY_ADDRESSES', {
                'player_map': 0x1000,
                'player_x': 0x1001,
                'player_y': 0x1002,
                'player_direction': 0x1003
            })

            result = read_location(memory)
            assert result == (4, 10, 15, 0)  # facing gets normalized to 0

            # Global state should be updated
            global _last_good_position
            assert _last_good_position == (4, 10, 15, 0)

    def test_read_party_pokemon_empty_slot_detection(self):
        """read_party_pokemon should detect empty slots and corruption."""
        memory = MagicMock()

        # Mock empty slot with species = 0
        memory.__getitem__.side_effect = lambda addr: 0

        with pytest.MonkeyPatch.context() as m:
            m.setattr('utils.memory_reader.MEMORY_ADDRESSES', {'party_data_start': 0x2000})
            m.setattr('utils.memory_reader.TRAINING_PARAMS', {
                'MAX_PARTY_SIZE': 6,
                'PARTY_SLOT_SIZE': 48,
                'MAX_LEVEL': 100
            })
            m.setattr('utils.memory_reader.POKEMON_SPECIES', {1, 2, 3, 4, 5})
            m.setattr('utils.memory_reader.STATUS_CONDITIONS', {0, 1, 2, 3})

            result = read_party_pokemon(memory, 0)

            # Should return empty slot structure
            expected = {
                "species": 0, "held_item": 0, "hp": 0, "max_hp": 0,
                "level": 0, "status": 0, "moves": [0, 0, 0, 0], "pp": [0, 0, 0, 0]
            }
            assert result == expected

    def test_read_party_pokemon_corruption_detection(self):
        """read_party_pokemon should detect and handle corrupted Pokemon data."""
        memory = MagicMock()

        # Mock corrupted data with invalid species
        memory.__getitem__.side_effect = lambda addr: {
            0x2000: 255,  # Invalid species (corruption indicator)
            0x2008: 240,  # Invalid level
        }.get(addr, 0)

        with pytest.MonkeyPatch.context() as m:
            m.setattr('utils.memory_reader.MEMORY_ADDRESSES', {'party_data_start': 0x2000})
            m.setattr('utils.memory_reader.TRAINING_PARAMS', {
                'MAX_PARTY_SIZE': 6,
                'PARTY_SLOT_SIZE': 48,
                'MAX_LEVEL': 100
            })
            m.setattr('utils.memory_reader.POKEMON_SPECIES', {1, 2, 3, 4, 5})
            m.setattr('utils.memory_reader.STATUS_CONDITIONS', {0, 1, 2, 3})

            result = read_party_pokemon(memory, 0)

            # Should return empty slot due to corruption
            expected = {
                "species": 0, "held_item": 0, "hp": 0, "max_hp": 0,
                "level": 0, "status": 0, "moves": [0, 0, 0, 0], "pp": [0, 0, 0, 0]
            }
            assert result == expected

    def test_build_observation_corruption_resilience(self):
        """build_observation should handle memory corruption gracefully."""
        memory = MagicMock()

        # Mock corrupted memory that returns invalid values
        memory.__getitem__.side_effect = lambda addr: 0xFF

        with pytest.MonkeyPatch.context() as m:
            m.setattr('utils.memory_reader.MEMORY_ADDRESSES', {
                'party_count': 0x1000,
                'party_data_start': 0x2000,
                'money_low': 0x3000,
                'money_mid': 0x3001,
                'money_high': 0x3002,
                'badges': 0x4000,
                'player_map': 0x5000,
                'player_x': 0x5001,
                'player_y': 0x5002,
                'player_direction': 0x5003,
                'in_battle': 0x6000,
                'battle_turn': 0x6001,
                'enemy_species': 0x6002,
                'enemy_hp_low': 0x6003,
                'enemy_hp_high': 0x6004,
                'enemy_level': 0x6005,
                'step_counter': 0x7000
            })
            m.setattr('utils.memory_reader.TRAINING_PARAMS', {
                'MAX_PARTY_SIZE': 6,
                'PARTY_SLOT_SIZE': 48,
                'MAX_LEVEL': 100,
                'MAX_MONEY': 999999
            })
            m.setattr('utils.memory_reader.POKEMON_SPECIES', {1, 2, 3, 4, 5})
            m.setattr('utils.memory_reader.STATUS_CONDITIONS', {0, 1, 2, 3})

            result = build_observation(memory)

            # Should return valid state structure despite corruption
            assert isinstance(result, dict)
            assert result['party_count'] == 0  # Invalid party count reset
            assert result['money'] == 0  # Invalid money reset
            assert result['has_pokemon'] == False
            assert result['health_percentage'] == 0

    def test_position_history_management(self):
        """Position history should be properly managed and bounded."""
        memory = MagicMock()

        with pytest.MonkeyPatch.context() as m:
            m.setattr('utils.memory_reader.MEMORY_ADDRESSES', {
                'player_map': 0x1000,
                'player_x': 0x1001,
                'player_y': 0x1002,
                'player_direction': 0x1003
            })

            # Add more than 10 positions to test history bounds
            for i in range(15):
                memory.__getitem__.side_effect = lambda addr, i=i: {
                    0x1000: 1,    # map_id
                    0x1001: i,    # x varies
                    0x1002: 20,   # y constant
                    0x1003: 0     # facing
                }.get(addr, 0)

                read_location(memory)

            # History should be bounded to 10 entries
            global _position_history
            assert len(_position_history) <= 10