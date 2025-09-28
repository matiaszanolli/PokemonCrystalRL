#!/usr/bin/env python3
"""
Test Suite for Addresses Module

Tests the structured address mappings organized by categories
(party, location, resource, battle, misc).
"""

import pytest

# Try importing from addresses module if it exists
try:
    from config.addresses import (
        PARTY_ADDRESSES, LOCATION_ADDRESSES, RESOURCE_ADDRESSES,
        BATTLE_ADDRESSES, MISC_ADDRESSES
    )
    ADDRESSES_MODULE_EXISTS = True
except ImportError:
    ADDRESSES_MODULE_EXISTS = False


@pytest.mark.skipif(not ADDRESSES_MODULE_EXISTS, reason="addresses module not found")
class TestPartyAddresses:
    """Test PARTY_ADDRESSES constant definitions."""

    def test_party_addresses_exists(self):
        """Test that PARTY_ADDRESSES dictionary exists."""
        assert PARTY_ADDRESSES is not None
        assert isinstance(PARTY_ADDRESSES, dict)
        assert len(PARTY_ADDRESSES) > 0

    def test_party_addresses_structure(self):
        """Test PARTY_ADDRESSES dictionary structure."""
        for addr_name, addr_value in PARTY_ADDRESSES.items():
            assert isinstance(addr_name, str)
            assert isinstance(addr_value, int)
            assert len(addr_name) > 0
            assert addr_value > 0

    def test_essential_party_addresses(self):
        """Test that essential party addresses are defined."""
        essential_addresses = [
            'party_count', 'player_species', 'player_hp',
            'player_max_hp', 'player_level'
        ]

        for addr in essential_addresses:
            assert addr in PARTY_ADDRESSES

    def test_party_hp_consistency(self):
        """Test HP address consistency (low/high byte pairs)."""
        if 'player_hp' in PARTY_ADDRESSES and 'player_hp_high' in PARTY_ADDRESSES:
            player_hp = PARTY_ADDRESSES['player_hp']
            player_hp_high = PARTY_ADDRESSES['player_hp_high']
            assert player_hp_high == player_hp + 1

        if 'player_max_hp' in PARTY_ADDRESSES and 'player_max_hp_high' in PARTY_ADDRESSES:
            player_max_hp = PARTY_ADDRESSES['player_max_hp']
            player_max_hp_high = PARTY_ADDRESSES['player_max_hp_high']
            assert player_max_hp_high == player_max_hp + 1


@pytest.mark.skipif(not ADDRESSES_MODULE_EXISTS, reason="addresses module not found")
class TestLocationAddresses:
    """Test LOCATION_ADDRESSES constant definitions."""

    def test_location_addresses_exists(self):
        """Test that LOCATION_ADDRESSES dictionary exists."""
        assert LOCATION_ADDRESSES is not None
        assert isinstance(LOCATION_ADDRESSES, dict)
        assert len(LOCATION_ADDRESSES) > 0

    def test_location_addresses_structure(self):
        """Test LOCATION_ADDRESSES dictionary structure."""
        for addr_name, addr_value in LOCATION_ADDRESSES.items():
            assert isinstance(addr_name, str)
            assert isinstance(addr_value, int)
            assert len(addr_name) > 0
            assert addr_value > 0

    def test_essential_location_addresses(self):
        """Test that essential location addresses are defined."""
        essential_addresses = ['player_map', 'player_x', 'player_y']

        for addr in essential_addresses:
            assert addr in LOCATION_ADDRESSES

    def test_verified_location_addresses(self):
        """Test verified location addresses have correct values."""
        # These addresses are marked as VERIFIED in the source
        verified_addresses = {
            'player_map': 0xDCBA,
            'player_x': 0xDCB8,
            'player_y': 0xDCB9
        }

        for addr_name, expected_value in verified_addresses.items():
            if addr_name in LOCATION_ADDRESSES:
                assert LOCATION_ADDRESSES[addr_name] == expected_value

    def test_location_address_grouping(self):
        """Test that location addresses are close together."""
        if 'player_x' in LOCATION_ADDRESSES and 'player_y' in LOCATION_ADDRESSES:
            player_x = LOCATION_ADDRESSES['player_x']
            player_y = LOCATION_ADDRESSES['player_y']
            assert abs(player_x - player_y) <= 10


@pytest.mark.skipif(not ADDRESSES_MODULE_EXISTS, reason="addresses module not found")
class TestResourceAddresses:
    """Test RESOURCE_ADDRESSES constant definitions."""

    def test_resource_addresses_exists(self):
        """Test that RESOURCE_ADDRESSES dictionary exists."""
        assert RESOURCE_ADDRESSES is not None
        assert isinstance(RESOURCE_ADDRESSES, dict)
        assert len(RESOURCE_ADDRESSES) > 0

    def test_resource_addresses_structure(self):
        """Test RESOURCE_ADDRESSES dictionary structure."""
        for addr_name, addr_value in RESOURCE_ADDRESSES.items():
            assert isinstance(addr_name, str)
            assert isinstance(addr_value, int)
            assert len(addr_name) > 0
            assert addr_value > 0

    def test_essential_resource_addresses(self):
        """Test that essential resource addresses are defined."""
        essential_addresses = ['badges']

        for addr in essential_addresses:
            assert addr in RESOURCE_ADDRESSES

    def test_money_address_consistency(self):
        """Test money address consistency (3-byte little-endian)."""
        money_addresses = ['money_low', 'money_mid', 'money_high']
        money_present = [addr for addr in money_addresses if addr in RESOURCE_ADDRESSES]

        if len(money_present) >= 3:
            money_low = RESOURCE_ADDRESSES['money_low']
            money_mid = RESOURCE_ADDRESSES['money_mid']
            money_high = RESOURCE_ADDRESSES['money_high']

            # Money addresses should be consecutive
            assert money_mid == money_low + 1
            assert money_high == money_mid + 1


@pytest.mark.skipif(not ADDRESSES_MODULE_EXISTS, reason="addresses module not found")
class TestBattleAddresses:
    """Test BATTLE_ADDRESSES constant definitions."""

    def test_battle_addresses_exists(self):
        """Test that BATTLE_ADDRESSES dictionary exists."""
        assert BATTLE_ADDRESSES is not None
        assert isinstance(BATTLE_ADDRESSES, dict)
        assert len(BATTLE_ADDRESSES) > 0

    def test_battle_addresses_structure(self):
        """Test BATTLE_ADDRESSES dictionary structure."""
        for addr_name, addr_value in BATTLE_ADDRESSES.items():
            assert isinstance(addr_name, str)
            assert isinstance(addr_value, int)
            assert len(addr_name) > 0
            assert addr_value > 0

    def test_essential_battle_addresses(self):
        """Test that essential battle addresses are defined."""
        essential_addresses = ['in_battle']

        for addr in essential_addresses:
            assert addr in BATTLE_ADDRESSES

    def test_enemy_hp_consistency(self):
        """Test enemy HP address consistency."""
        if 'enemy_hp_low' in BATTLE_ADDRESSES and 'enemy_hp_high' in BATTLE_ADDRESSES:
            enemy_hp_low = BATTLE_ADDRESSES['enemy_hp_low']
            enemy_hp_high = BATTLE_ADDRESSES['enemy_hp_high']
            assert enemy_hp_high == enemy_hp_low + 1

    def test_battle_address_grouping(self):
        """Test that battle addresses are logically grouped."""
        battle_values = list(BATTLE_ADDRESSES.values())
        if len(battle_values) >= 2:
            min_addr = min(battle_values)
            max_addr = max(battle_values)
            address_span = max_addr - min_addr

            # Battle addresses should be reasonably close together
            assert address_span < 1000, "Battle addresses too spread out"


@pytest.mark.skipif(not ADDRESSES_MODULE_EXISTS, reason="addresses module not found")
class TestMiscAddresses:
    """Test MISC_ADDRESSES constant definitions."""

    def test_misc_addresses_exists(self):
        """Test that MISC_ADDRESSES dictionary exists."""
        assert MISC_ADDRESSES is not None
        assert isinstance(MISC_ADDRESSES, dict)
        assert len(MISC_ADDRESSES) >= 0  # May be empty

    def test_misc_addresses_structure(self):
        """Test MISC_ADDRESSES dictionary structure."""
        for addr_name, addr_value in MISC_ADDRESSES.items():
            assert isinstance(addr_name, str)
            assert isinstance(addr_value, int)
            assert len(addr_name) > 0
            assert addr_value > 0


@pytest.mark.skipif(not ADDRESSES_MODULE_EXISTS, reason="addresses module not found")
class TestAddressesIntegration:
    """Test integration between different address categories."""

    def test_address_uniqueness_across_categories(self):
        """Test that addresses are unique across all categories."""
        all_addresses = {}

        # Collect all addresses from all categories
        for category_name, category_dict in [
            ('PARTY', PARTY_ADDRESSES),
            ('LOCATION', LOCATION_ADDRESSES),
            ('RESOURCE', RESOURCE_ADDRESSES),
            ('BATTLE', BATTLE_ADDRESSES),
            ('MISC', MISC_ADDRESSES)
        ]:
            for addr_name, addr_value in category_dict.items():
                if addr_value in all_addresses:
                    # Some duplication might be legitimate (multi-byte values)
                    existing_name = all_addresses[addr_value]
                    print(f"Warning: Address 0x{addr_value:04X} used by both {existing_name} and {addr_name}")
                all_addresses[addr_value] = addr_name

        # Allow some duplication but not too much
        total_addresses = sum(len(cat) for cat in [
            PARTY_ADDRESSES, LOCATION_ADDRESSES, RESOURCE_ADDRESSES,
            BATTLE_ADDRESSES, MISC_ADDRESSES
        ])
        unique_addresses = len(all_addresses)
        duplication_ratio = (total_addresses - unique_addresses) / total_addresses

        assert duplication_ratio < 0.3, "Too much address duplication across categories"

    def test_memory_range_coverage(self):
        """Test that addresses cover expected memory ranges."""
        all_addresses = []

        # Collect all address values
        for category_dict in [
            PARTY_ADDRESSES, LOCATION_ADDRESSES, RESOURCE_ADDRESSES,
            BATTLE_ADDRESSES, MISC_ADDRESSES
        ]:
            all_addresses.extend(category_dict.values())

        if all_addresses:
            min_addr = min(all_addresses)
            max_addr = max(all_addresses)

            # Should be in Game Boy memory range
            assert 0x8000 <= min_addr <= 0xFFFF
            assert 0x8000 <= max_addr <= 0xFFFF

    def test_category_logical_separation(self):
        """Test that address categories are logically separated."""
        # Party and location addresses should generally be in different regions
        if PARTY_ADDRESSES and LOCATION_ADDRESSES:
            party_values = list(PARTY_ADDRESSES.values())
            location_values = list(LOCATION_ADDRESSES.values())

            party_avg = sum(party_values) / len(party_values)
            location_avg = sum(location_values) / len(location_values)

            # They should be in different memory regions (allowing some overlap)
            assert abs(party_avg - location_avg) > 100


class TestAddressesModuleFallback:
    """Test fallback behavior when addresses module doesn't exist."""

    @pytest.mark.skipif(ADDRESSES_MODULE_EXISTS, reason="addresses module exists")
    def test_addresses_module_missing_handled(self):
        """Test that missing addresses module is handled gracefully."""
        # This test runs when the addresses module is not found
        # It verifies that the test suite handles the missing module correctly
        assert not ADDRESSES_MODULE_EXISTS


class TestAddressValidationHelpers:
    """Test helper functions for address validation."""

    @pytest.mark.skipif(not ADDRESSES_MODULE_EXISTS, reason="addresses module not found")
    def test_all_addresses_in_valid_range(self):
        """Test that all addresses are in valid Game Boy memory ranges."""
        valid_ranges = [
            (0x8000, 0xFFFF),  # Various RAM and register areas
            (0xC000, 0xFDFF),  # Work RAM areas
            (0xD000, 0xDFFF),  # Work RAM bank 1
        ]

        all_categories = [
            ('PARTY', PARTY_ADDRESSES),
            ('LOCATION', LOCATION_ADDRESSES),
            ('RESOURCE', RESOURCE_ADDRESSES),
            ('BATTLE', BATTLE_ADDRESSES),
            ('MISC', MISC_ADDRESSES)
        ]

        for category_name, category_dict in all_categories:
            for addr_name, addr_value in category_dict.items():
                in_valid_range = any(start <= addr_value <= end for start, end in valid_ranges)
                assert in_valid_range, f"{category_name}.{addr_name}=0x{addr_value:04X} not in valid range"

    @pytest.mark.skipif(not ADDRESSES_MODULE_EXISTS, reason="addresses module not found")
    def test_address_naming_conventions(self):
        """Test that address names follow consistent conventions."""
        all_categories = [
            PARTY_ADDRESSES, LOCATION_ADDRESSES, RESOURCE_ADDRESSES,
            BATTLE_ADDRESSES, MISC_ADDRESSES
        ]

        for category_dict in all_categories:
            for addr_name in category_dict.keys():
                # Should be lowercase with underscores
                assert addr_name.islower() or '_' in addr_name
                assert ' ' not in addr_name  # No spaces
                assert addr_name.replace('_', '').isalnum()  # Only letters, numbers, underscores


if __name__ == "__main__":
    pytest.main([__file__, "-v"])