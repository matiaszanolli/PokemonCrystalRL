#!/usr/bin/env python3
"""
Test Suite for Memory Map Module

Tests the memory address definitions and mappings for Pokemon Crystal game state.
"""

import pytest

from environments.state.memory_map import MEMORY_ADDRESSES


class TestMemoryAddresses:
    """Test memory address definitions."""

    def test_memory_addresses_exists(self):
        """Test that MEMORY_ADDRESSES dictionary exists."""
        assert MEMORY_ADDRESSES is not None
        assert isinstance(MEMORY_ADDRESSES, dict)
        assert len(MEMORY_ADDRESSES) > 0

    def test_player_position_addresses(self):
        """Test player position memory addresses."""
        required_addresses = ['player_x', 'player_y', 'player_map', 'player_direction']

        for addr in required_addresses:
            assert addr in MEMORY_ADDRESSES
            assert isinstance(MEMORY_ADDRESSES[addr], int)
            assert MEMORY_ADDRESSES[addr] > 0

    def test_player_stats_addresses(self):
        """Test player stats memory addresses."""
        required_addresses = [
            'player_hp', 'player_max_hp', 'player_level',
            'player_exp', 'player_exp_to_next'
        ]

        for addr in required_addresses:
            assert addr in MEMORY_ADDRESSES
            assert isinstance(MEMORY_ADDRESSES[addr], int)
            assert MEMORY_ADDRESSES[addr] > 0

    def test_party_information_addresses(self):
        """Test party information memory addresses."""
        required_addresses = [
            'party_count', 'party_species', 'party_hp',
            'party_max_hp', 'party_level', 'party_status'
        ]

        for addr in required_addresses:
            assert addr in MEMORY_ADDRESSES
            assert isinstance(MEMORY_ADDRESSES[addr], int)
            assert MEMORY_ADDRESSES[addr] > 0

    def test_items_inventory_addresses(self):
        """Test items and inventory memory addresses."""
        required_addresses = ['money', 'bag_count', 'bag_items']

        for addr in required_addresses:
            assert addr in MEMORY_ADDRESSES
            assert isinstance(MEMORY_ADDRESSES[addr], int)
            assert MEMORY_ADDRESSES[addr] > 0

    def test_game_progress_addresses(self):
        """Test game progress memory addresses."""
        required_addresses = [
            'badges', 'kanto_badges', 'elite_four_beaten', 'champion_beaten'
        ]

        for addr in required_addresses:
            assert addr in MEMORY_ADDRESSES
            assert isinstance(MEMORY_ADDRESSES[addr], int)
            assert MEMORY_ADDRESSES[addr] > 0

    def test_battle_state_addresses(self):
        """Test battle state memory addresses."""
        required_addresses = [
            'in_battle', 'battle_type', 'enemy_hp',
            'enemy_max_hp', 'enemy_level', 'enemy_species'
        ]

        for addr in required_addresses:
            assert addr in MEMORY_ADDRESSES
            assert isinstance(MEMORY_ADDRESSES[addr], int)
            assert MEMORY_ADDRESSES[addr] > 0

    def test_menu_ui_state_addresses(self):
        """Test menu and UI state memory addresses."""
        # menu_state should be present
        assert 'menu_state' in MEMORY_ADDRESSES
        assert isinstance(MEMORY_ADDRESSES['menu_state'], int)
        assert MEMORY_ADDRESSES['menu_state'] > 0

    def test_address_ranges(self):
        """Test that addresses are in valid Game Boy memory ranges."""
        # Game Boy Color RAM addresses are typically in specific ranges
        valid_ranges = [
            (0x8000, 0xFFFF),  # Various RAM and register areas
            (0xC000, 0xFDFF),  # Work RAM areas
            (0xD000, 0xDFFF),  # Work RAM bank 1
        ]

        for name, addr in MEMORY_ADDRESSES.items():
            # Check if address is in any valid range
            in_valid_range = any(start <= addr <= end for start, end in valid_ranges)
            assert in_valid_range, f"Address {name}=0x{addr:04X} is not in valid range"

    def test_address_uniqueness(self):
        """Test that addresses are unique where appropriate."""
        # Some addresses may legitimately be the same (multi-byte values)
        # but most should be unique
        addresses = list(MEMORY_ADDRESSES.values())
        unique_addresses = set(addresses)

        # Allow some duplication but not too much
        duplication_ratio = len(unique_addresses) / len(addresses)
        assert duplication_ratio > 0.8, "Too many duplicate addresses found"

    def test_address_format(self):
        """Test that addresses are properly formatted."""
        for name, addr in MEMORY_ADDRESSES.items():
            # Should be positive integers
            assert isinstance(addr, int)
            assert addr > 0

            # Should be valid 16-bit addresses
            assert addr <= 0xFFFF

    def test_specific_known_addresses(self):
        """Test specific known address values if documented."""
        # These are example tests - adjust based on actual ROM analysis

        # Player position addresses should be close together
        player_x = MEMORY_ADDRESSES['player_x']
        player_y = MEMORY_ADDRESSES['player_y']
        assert abs(player_x - player_y) <= 10, "Player X/Y addresses should be close"

        # HP addresses should be close together
        hp = MEMORY_ADDRESSES['player_hp']
        max_hp = MEMORY_ADDRESSES['player_max_hp']
        assert abs(hp - max_hp) <= 10, "HP addresses should be close"

    def test_address_comments_consistency(self):
        """Test that address comments are consistent with names."""
        # This would require reading the source file to check comments
        # For now, just verify the structure is consistent
        position_addresses = [
            'player_x', 'player_y', 'player_map', 'player_direction'
        ]

        for addr in position_addresses:
            assert addr in MEMORY_ADDRESSES
            # All position addresses should be in similar memory region
            base_addr = MEMORY_ADDRESSES['player_x']
            current_addr = MEMORY_ADDRESSES[addr]
            assert abs(current_addr - base_addr) < 50, f"{addr} too far from base position"

    def test_multi_byte_addresses(self):
        """Test addresses that represent multi-byte values."""
        # Some values like HP, money, etc. are multi-byte
        multi_byte_fields = [
            'player_hp', 'player_max_hp', 'player_exp', 'money'
        ]

        for field in multi_byte_fields:
            assert field in MEMORY_ADDRESSES
            addr = MEMORY_ADDRESSES[field]
            assert isinstance(addr, int)
            assert addr > 0

    def test_array_addresses(self):
        """Test addresses that represent arrays or lists."""
        array_fields = [
            'party_species', 'party_hp', 'party_max_hp',
            'party_level', 'party_status', 'bag_items'
        ]

        for field in array_fields:
            assert field in MEMORY_ADDRESSES
            addr = MEMORY_ADDRESSES[field]
            assert isinstance(addr, int)
            assert addr > 0

    def test_battle_addresses_logical_grouping(self):
        """Test that battle-related addresses are logically grouped."""
        battle_fields = [
            'in_battle', 'battle_type', 'enemy_hp',
            'enemy_max_hp', 'enemy_level', 'enemy_species'
        ]

        battle_addresses = [MEMORY_ADDRESSES[field] for field in battle_fields]

        # Battle addresses should be in the same general memory region
        min_addr = min(battle_addresses)
        max_addr = max(battle_addresses)
        address_span = max_addr - min_addr

        # Allow reasonable span but not too spread out
        assert address_span < 1000, "Battle addresses too spread out"

    def test_progress_addresses_logical_grouping(self):
        """Test that progress-related addresses are logically grouped."""
        progress_fields = [
            'badges', 'kanto_badges', 'elite_four_beaten', 'champion_beaten'
        ]

        progress_addresses = [MEMORY_ADDRESSES[field] for field in progress_fields]

        # Progress addresses should be in the same general memory region
        min_addr = min(progress_addresses)
        max_addr = max(progress_addresses)
        address_span = max_addr - min_addr

        # Allow reasonable span
        assert address_span < 100, "Progress addresses too spread out"


class TestMemoryMapCompleteness:
    """Test completeness of memory map."""

    def test_minimum_required_addresses(self):
        """Test that minimum required addresses are present."""
        essential_addresses = [
            # Position
            'player_x', 'player_y', 'player_map',
            # Stats
            'player_hp', 'player_max_hp', 'player_level',
            # Party
            'party_count',
            # Progress
            'badges', 'money',
            # Battle
            'in_battle'
        ]

        for addr in essential_addresses:
            assert addr in MEMORY_ADDRESSES, f"Essential address {addr} missing"

    def test_address_coverage(self):
        """Test that we have good coverage of game state."""
        categories = {
            'position': ['player_x', 'player_y', 'player_map'],
            'stats': ['player_hp', 'player_max_hp', 'player_level'],
            'party': ['party_count', 'party_species'],
            'inventory': ['money', 'bag_count'],
            'progress': ['badges'],
            'battle': ['in_battle', 'enemy_hp']
        }

        for category, addresses in categories.items():
            for addr in addresses:
                assert addr in MEMORY_ADDRESSES, f"{category} address {addr} missing"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])