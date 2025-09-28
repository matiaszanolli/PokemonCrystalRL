#!/usr/bin/env python3
"""
Test Suite for Memory Addresses Module

Tests the validated memory address mappings for Pokemon Crystal ROM data structures.
"""

import pytest

from config.memory_addresses import MEMORY_ADDRESSES


class TestMemoryAddresses:
    """Test MEMORY_ADDRESSES dictionary functionality."""

    def test_memory_addresses_exists(self):
        """Test that MEMORY_ADDRESSES dictionary exists and is not empty."""
        assert MEMORY_ADDRESSES is not None
        assert isinstance(MEMORY_ADDRESSES, dict)
        assert len(MEMORY_ADDRESSES) > 0

    def test_memory_addresses_structure(self):
        """Test MEMORY_ADDRESSES dictionary structure."""
        for addr_name, addr_value in MEMORY_ADDRESSES.items():
            assert isinstance(addr_name, str)
            assert isinstance(addr_value, int)
            assert len(addr_name) > 0
            assert addr_value > 0

    def test_party_pokemon_addresses(self):
        """Test party and Pokemon data addresses."""
        party_addresses = [
            'party_count', 'player_species', 'player_held_item',
            'player_hp', 'player_hp_high', 'player_max_hp', 'player_max_hp_high',
            'player_level', 'player_status'
        ]

        for addr in party_addresses:
            assert addr in MEMORY_ADDRESSES
            assert isinstance(MEMORY_ADDRESSES[addr], int)
            assert MEMORY_ADDRESSES[addr] > 0

    def test_location_movement_addresses(self):
        """Test location and movement addresses."""
        location_addresses = [
            'player_map', 'player_x', 'player_y', 'player_direction'
        ]

        for addr in location_addresses:
            assert addr in MEMORY_ADDRESSES
            assert isinstance(MEMORY_ADDRESSES[addr], int)
            assert MEMORY_ADDRESSES[addr] > 0

    def test_resource_progress_addresses(self):
        """Test resource and progress addresses."""
        resource_addresses = [
            'money_low', 'money_mid', 'money_high', 'badges'
        ]

        for addr in resource_addresses:
            assert addr in MEMORY_ADDRESSES
            assert isinstance(MEMORY_ADDRESSES[addr], int)
            assert MEMORY_ADDRESSES[addr] > 0

    def test_battle_state_addresses(self):
        """Test battle state addresses."""
        battle_addresses = [
            'in_battle', 'battle_turn', 'enemy_species',
            'enemy_hp_low', 'enemy_hp_high', 'enemy_level',
            'player_active_slot', 'move_selected'
        ]

        for addr in battle_addresses:
            assert addr in MEMORY_ADDRESSES
            assert isinstance(MEMORY_ADDRESSES[addr], int)
            assert MEMORY_ADDRESSES[addr] > 0

    def test_misc_addresses(self):
        """Test miscellaneous addresses."""
        misc_addresses = ['step_counter', 'game_time_hours']

        for addr in misc_addresses:
            assert addr in MEMORY_ADDRESSES
            assert isinstance(MEMORY_ADDRESSES[addr], int)
            assert MEMORY_ADDRESSES[addr] > 0

    def test_address_ranges(self):
        """Test that addresses are in valid Game Boy memory ranges."""
        # Game Boy Color memory address ranges
        valid_ranges = [
            (0x8000, 0xFFFF),  # Various RAM and register areas
            (0xC000, 0xFDFF),  # Work RAM areas
            (0xD000, 0xDFFF),  # Work RAM bank 1 (most Pokemon Crystal data)
        ]

        for addr_name, addr_value in MEMORY_ADDRESSES.items():
            # Check if address is in any valid range
            in_valid_range = any(start <= addr_value <= end for start, end in valid_ranges)
            assert in_valid_range, f"Address {addr_name}=0x{addr_value:04X} not in valid range"

    def test_address_format(self):
        """Test that addresses are properly formatted."""
        for addr_name, addr_value in MEMORY_ADDRESSES.items():
            # Should be positive integers
            assert isinstance(addr_value, int)
            assert addr_value > 0

            # Should be valid 16-bit addresses
            assert addr_value <= 0xFFFF

    def test_hp_address_consistency(self):
        """Test HP address consistency (low/high byte pairs)."""
        # Player HP addresses should be consecutive
        player_hp = MEMORY_ADDRESSES['player_hp']
        player_hp_high = MEMORY_ADDRESSES['player_hp_high']
        assert player_hp_high == player_hp + 1

        # Player max HP addresses should be consecutive
        player_max_hp = MEMORY_ADDRESSES['player_max_hp']
        player_max_hp_high = MEMORY_ADDRESSES['player_max_hp_high']
        assert player_max_hp_high == player_max_hp + 1

        # Enemy HP addresses should be consecutive
        enemy_hp_low = MEMORY_ADDRESSES['enemy_hp_low']
        enemy_hp_high = MEMORY_ADDRESSES['enemy_hp_high']
        assert enemy_hp_high == enemy_hp_low + 1

    def test_money_address_consistency(self):
        """Test money address consistency (3-byte little-endian)."""
        money_low = MEMORY_ADDRESSES['money_low']
        money_mid = MEMORY_ADDRESSES['money_mid']
        money_high = MEMORY_ADDRESSES['money_high']

        # Money addresses should be consecutive
        assert money_mid == money_low + 1
        assert money_high == money_mid + 1

    def test_position_address_consistency(self):
        """Test position address consistency."""
        player_x = MEMORY_ADDRESSES['player_x']
        player_y = MEMORY_ADDRESSES['player_y']
        player_map = MEMORY_ADDRESSES['player_map']

        # Position addresses should be close together
        assert abs(player_x - player_y) <= 10
        assert abs(player_x - player_map) <= 10

    def test_battle_address_grouping(self):
        """Test that battle addresses are logically grouped."""
        battle_addresses = [
            'in_battle', 'battle_turn', 'enemy_species',
            'enemy_hp_low', 'enemy_hp_high', 'enemy_level',
            'player_active_slot', 'move_selected'
        ]

        battle_values = [MEMORY_ADDRESSES[addr] for addr in battle_addresses]
        min_addr = min(battle_values)
        max_addr = max(battle_values)
        address_span = max_addr - min_addr

        # Battle addresses should be reasonably close together
        assert address_span < 1000, "Battle addresses too spread out"

    def test_party_address_grouping(self):
        """Test that party addresses are logically grouped."""
        party_addresses = [
            'party_count', 'player_species', 'player_held_item',
            'player_hp', 'player_hp_high', 'player_max_hp', 'player_max_hp_high',
            'player_level', 'player_status'
        ]

        party_values = [MEMORY_ADDRESSES[addr] for addr in party_addresses]
        min_addr = min(party_values)
        max_addr = max(party_values)
        address_span = max_addr - min_addr

        # Party addresses should be reasonably close together
        assert address_span < 100, "Party addresses too spread out"

    def test_address_uniqueness(self):
        """Test that addresses are unique where appropriate."""
        addresses = list(MEMORY_ADDRESSES.values())

        # Count duplicates (some may be legitimate for multi-byte values)
        duplicates = len(addresses) - len(set(addresses))

        # Allow some duplication but not too much
        duplication_ratio = duplicates / len(addresses)
        assert duplication_ratio < 0.2, "Too many duplicate addresses"

    def test_specific_known_addresses(self):
        """Test specific known validated addresses."""
        # These addresses are marked as VERIFIED in the source
        verified_addresses = {
            'player_map': 0xDCBA,
            'player_x': 0xDCB8,
            'player_y': 0xDCB9,
            'player_direction': 0xDCBB
        }

        for addr_name, expected_value in verified_addresses.items():
            assert addr_name in MEMORY_ADDRESSES
            assert MEMORY_ADDRESSES[addr_name] == expected_value

    def test_address_naming_conventions(self):
        """Test address naming conventions."""
        for addr_name in MEMORY_ADDRESSES.keys():
            # Should be lowercase with underscores
            assert addr_name.islower() or '_' in addr_name
            assert ' ' not in addr_name  # No spaces
            assert addr_name.replace('_', '').isalnum()  # Only letters, numbers, underscores

    def test_address_categories_coverage(self):
        """Test that we have good coverage of address categories."""
        categories = {
            'party': ['party_count', 'player_species', 'player_hp'],
            'location': ['player_map', 'player_x', 'player_y'],
            'resources': ['money_low', 'badges'],
            'battle': ['in_battle', 'enemy_species'],
            'misc': ['step_counter', 'game_time_hours']
        }

        for category, required_addresses in categories.items():
            for addr in required_addresses:
                assert addr in MEMORY_ADDRESSES, f"Missing {category} address: {addr}"

    def test_party_slot_structure(self):
        """Test party slot structure consistency."""
        # Player Pokemon data should follow logical structure
        player_species = MEMORY_ADDRESSES['player_species']
        player_held_item = MEMORY_ADDRESSES['player_held_item']

        # Item should come after species
        assert player_held_item > player_species

    def test_multi_byte_consistency(self):
        """Test multi-byte value consistency."""
        # Test that multi-byte values have proper byte ordering
        multi_byte_pairs = [
            ('player_hp', 'player_hp_high'),
            ('player_max_hp', 'player_max_hp_high'),
            ('enemy_hp_low', 'enemy_hp_high')
        ]

        for low_byte, high_byte in multi_byte_pairs:
            assert low_byte in MEMORY_ADDRESSES
            assert high_byte in MEMORY_ADDRESSES
            assert MEMORY_ADDRESSES[high_byte] == MEMORY_ADDRESSES[low_byte] + 1

    def test_resource_addresses_logic(self):
        """Test resource addresses logical consistency."""
        # Money should be 3-byte value
        money_addresses = ['money_low', 'money_mid', 'money_high']
        money_values = [MEMORY_ADDRESSES[addr] for addr in money_addresses]

        # Should be consecutive
        assert money_values[1] == money_values[0] + 1
        assert money_values[2] == money_values[1] + 1

    def test_battle_state_logic(self):
        """Test battle state addresses logical consistency."""
        # Battle flag and battle-related data should be separate
        in_battle = MEMORY_ADDRESSES['in_battle']
        enemy_species = MEMORY_ADDRESSES['enemy_species']

        # These shouldn't be the same address
        assert in_battle != enemy_species


class TestMemoryAddressValidation:
    """Test memory address validation and verification."""

    def test_essential_addresses_present(self):
        """Test that essential addresses for core functionality are present."""
        essential_addresses = [
            'player_x', 'player_y', 'player_map',  # Position
            'player_hp', 'player_max_hp', 'player_level',  # Player stats
            'party_count',  # Party info
            'badges', 'money_low',  # Progress
            'in_battle'  # Game state
        ]

        for addr in essential_addresses:
            assert addr in MEMORY_ADDRESSES, f"Essential address {addr} missing"

    def test_verified_addresses_documented(self):
        """Test that verified addresses are properly documented."""
        # Addresses marked as VERIFIED in source should be present
        verified_addresses = [
            'player_map', 'player_x', 'player_y', 'player_direction'
        ]

        for addr in verified_addresses:
            assert addr in MEMORY_ADDRESSES
            # Value should be in expected Game Boy memory range
            assert 0x8000 <= MEMORY_ADDRESSES[addr] <= 0xFFFF

    def test_party_data_structure(self):
        """Test party data structure addresses."""
        party_addresses = [
            'party_count', 'player_species', 'player_held_item',
            'player_hp', 'player_max_hp', 'player_level', 'player_status'
        ]

        # All party addresses should exist
        for addr in party_addresses:
            assert addr in MEMORY_ADDRESSES

        # Party data should be in a logical memory region
        party_values = [MEMORY_ADDRESSES[addr] for addr in party_addresses]
        min_party = min(party_values)
        max_party = max(party_values)

        # Should all be in the same general area
        assert max_party - min_party < 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])