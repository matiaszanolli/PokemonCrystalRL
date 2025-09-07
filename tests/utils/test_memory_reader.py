"""
Tests for Pokemon Crystal memory reading utilities.
"""

import pytest
import numpy as np
from typing import Dict

from utils.memory_reader import (
    get_safe_memory,
    read_party_pokemon,
    read_money,
    read_location,
    build_observation
)

from config.memory_addresses import MEMORY_ADDRESSES
from config.constants import (
    POKEMON_SPECIES,
    STATUS_CONDITIONS,
    TRAINING_PARAMS
)

from tests.fixtures.pyboy_helpers import (
    mock_pyboy,
    mock_pyboy_with_pokemon,
    mock_pyboy_in_battle,
    mock_pyboy_exploring
)

def test_get_safe_memory(mock_pyboy):
    """Test safe memory reading with bounds checking."""
    # Test valid memory read
    mock_pyboy.set_memory(0xD163, 42)
    assert get_safe_memory(mock_pyboy.memory, 0xD163) == 42
    
    # Test invalid value
    mock_pyboy.memory.memory[0xD164] = 300  # Invalid byte value
    assert get_safe_memory(mock_pyboy.memory, 0xD164) == 0
    
    # Test missing address
    assert get_safe_memory(mock_pyboy.memory, 0xFFFF) == 0
    
    # Test with custom default
    assert get_safe_memory(mock_pyboy.memory, 0xFFFF, default=42) == 42

def test_read_party_pokemon_empty(mock_pyboy):
    """Test reading empty party slot."""
    pokemon = read_party_pokemon(mock_pyboy.memory, 0)
    assert pokemon['species'] == 0
    assert pokemon['level'] == 0
    assert pokemon['hp'] == 0
    assert pokemon['max_hp'] == 0
    assert pokemon['status'] == 0
    assert len(pokemon['moves']) == 4
    assert len(pokemon['pp']) == 4

def test_read_party_pokemon_valid(mock_pyboy_with_pokemon):
    """Test reading valid Pokemon data."""
    pokemon = read_party_pokemon(mock_pyboy_with_pokemon.memory, 0)
    assert pokemon['species'] == 155  # Cyndaquil
    assert pokemon['level'] == 5
    assert pokemon['hp'] == 20
    assert pokemon['max_hp'] == 24
    assert pokemon['status'] == 0  # Healthy

def test_read_party_pokemon_invalid_slot(mock_pyboy_with_pokemon):
    """Test reading invalid party slot."""
    # Test negative slot
    pokemon = read_party_pokemon(mock_pyboy_with_pokemon.memory, -1)
    assert pokemon['species'] == 0
    
    # Test out of range slot
    pokemon = read_party_pokemon(mock_pyboy_with_pokemon.memory, 6)
    assert pokemon['species'] == 0

def test_read_money(mock_pyboy_exploring):
    """Test reading money value."""
    # Test basic money reading
    assert read_money(mock_pyboy_exploring.memory) == 100
    
    # Test larger values
    mock_pyboy_exploring.set_memory(MEMORY_ADDRESSES['money_low'], 0xFF)
    mock_pyboy_exploring.set_memory(MEMORY_ADDRESSES['money_mid'], 0x01)
    assert read_money(mock_pyboy_exploring.memory) == 511
    
    # Test max value limit
    mock_pyboy_exploring.set_memory(MEMORY_ADDRESSES['money_low'], 0xFF)
    mock_pyboy_exploring.set_memory(MEMORY_ADDRESSES['money_mid'], 0xFF)
    mock_pyboy_exploring.set_memory(MEMORY_ADDRESSES['money_high'], 0xFF)
    assert read_money(mock_pyboy_exploring.memory) == 0  # Should reset on overflow

def test_read_location(mock_pyboy_exploring):
    """Test reading location data."""
    map_id, x, y, facing = read_location(mock_pyboy_exploring.memory)
    assert map_id == 26  # New Bark Town
    assert x == 10
    assert y == 12
    assert facing == 0  # Down
    
    # Test invalid coordinates
    mock_pyboy_exploring.set_memory(MEMORY_ADDRESSES['player_x'], 0xFF)
    mock_pyboy_exploring.set_memory(MEMORY_ADDRESSES['player_y'], 0xFF)
    map_id, x, y, facing = read_location(mock_pyboy_exploring.memory)
    assert x == 0  # Should reset invalid coordinates
    assert y == 0
    
    # Test invalid direction
    mock_pyboy_exploring.set_memory(MEMORY_ADDRESSES['player_direction'], 7)
    map_id, x, y, facing = read_location(mock_pyboy_exploring.memory)
    assert facing == 0  # Should reset invalid direction

def test_build_observation_empty(mock_pyboy):
    """Test building observation from empty game state."""
    state = build_observation(mock_pyboy.memory)
    assert state['party_count'] == 0
    assert state['money'] == 0
    assert state['badges'] == 0
    assert state['badges_count'] == 0
    assert state['has_pokemon'] is False
    assert state['health_percentage'] == 0

def test_build_observation_with_pokemon(mock_pyboy_with_pokemon):
    """Test building observation with Pokemon data."""
    state = build_observation(mock_pyboy_with_pokemon.memory)
    assert state['party_count'] == 1
    assert state['player_species'] == 155  # Cyndaquil
    assert state['player_level'] == 5
    assert state['player_hp'] == 20
    assert state['player_max_hp'] == 24
    assert state['has_pokemon'] is True
    assert state['health_percentage'] == (20 / 24) * 100

def test_build_observation_in_battle(mock_pyboy_in_battle):
    """Test building observation during battle."""
    state = build_observation(mock_pyboy_in_battle.memory)
    assert state['in_battle'] is True
    assert state['battle_turn'] == 2
    assert state['enemy_species'] == 16  # Pidgey
    assert state['enemy_level'] == 4
    assert state['enemy_hp'] == 15

def test_build_observation_exploring(mock_pyboy_exploring):
    """Test building observation during exploration."""
    state = build_observation(mock_pyboy_exploring.memory)
    assert state['player_map'] == 26  # New Bark Town
    assert state['coords'] == (10, 12)
    assert state['money'] == 100
    assert state['facing'] == 0  # Down
    assert not state['in_battle']

def test_build_observation_handles_corruption(mock_pyboy):
    """Test observation building with corrupted memory values."""
    # Set some invalid memory values
    mock_pyboy.set_memory(MEMORY_ADDRESSES['party_count'], 0xFF)  # Invalid party count
    mock_pyboy.set_memory(MEMORY_ADDRESSES['player_level'], 0xFF)  # Invalid level
    mock_pyboy.set_memory(MEMORY_ADDRESSES['player_hp'], 0xFF)    # HP higher than max
    mock_pyboy.set_memory(MEMORY_ADDRESSES['player_max_hp'], 0x01)
    
    state = build_observation(mock_pyboy.memory)
    assert state['party_count'] == 0  # Should reset invalid party count
    assert state['has_pokemon'] is False
    assert state['health_percentage'] == 0
