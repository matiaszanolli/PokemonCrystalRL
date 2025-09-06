"""
Pokemon Crystal Memory Management Package
========================================

This package contains consolidated memory mapping information for Pokemon Crystal,
including address definitions, conflict resolution, and testing utilities.
"""

from .addresses import (
    MEMORY_ADDRESSES,
    ADDRESS_GROUPS,
    DERIVED_VALUES,
    IMPORTANT_LOCATIONS,
    POKEMON_SPECIES,
    STATUS_CONDITIONS,
    BADGE_MASKS,
    get_badges_earned,
    validate_memory_addresses,
    get_address_conflicts,
    test_address_group,
    get_best_addresses_for_rom,
)

__all__ = [
    'MEMORY_ADDRESSES',
    'ADDRESS_GROUPS', 
    'DERIVED_VALUES',
    'IMPORTANT_LOCATIONS',
    'POKEMON_SPECIES',
    'STATUS_CONDITIONS',
    'BADGE_MASKS',
    'get_badges_earned',
    'validate_memory_addresses',
    'get_address_conflicts',
    'test_address_group',
    'get_best_addresses_for_rom',
]