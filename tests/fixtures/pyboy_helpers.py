"""
PyBoy Test Fixtures and Helpers

Common fixtures and helper functions for testing with PyBoy emulator.
"""

import pytest
import numpy as np
from typing import Dict, Optional
from unittest.mock import MagicMock
from config.constants import MEMORY_ADDRESSES, TRAINING_PARAMS

class MockMemory:
    """Mock PyBoy memory for testing."""
    def __init__(self, initial_state: Optional[Dict[int, int]] = None):
        self.memory = initial_state or {}
        
    def __getitem__(self, key):
        return self.memory.get(key, 0)
        
    def __setitem__(self, key, value):
        if not isinstance(value, int) or not 0 <= value <= 255:
            raise ValueError(f"Invalid memory value: {value}")
        self.memory[key] = value

class MockPyBoy:
    """Mock PyBoy emulator for testing."""
    def __init__(self, initial_memory: Optional[Dict[int, int]] = None):
        self.memory = MockMemory(initial_memory)
        self.screen = MagicMock()
        self.screen.ndarray = np.zeros((144, 160, 3), dtype=np.uint8)
        
    def set_memory(self, addr: int, value: int) -> None:
        """Set memory value at address."""
        self.memory[addr] = value
        
    def get_memory(self, addr: int) -> int:
        """Get memory value at address."""
        return self.memory[addr]
        
    def set_screen(self, array: np.ndarray) -> None:
        """Set screen content."""
        if array.shape != (144, 160, 3):
            raise ValueError("Invalid screen dimensions")
        self.screen.ndarray = array.copy()

@pytest.fixture
def mock_pyboy():
    """Basic mock PyBoy instance."""
    return MockPyBoy()

@pytest.fixture
def mock_pyboy_with_pokemon():
    """Mock PyBoy with basic Pokemon data."""
    initial_memory = {
        MEMORY_ADDRESSES['party_count']: 1,
        MEMORY_ADDRESSES['player_level']: 5,
        MEMORY_ADDRESSES['player_hp']: 20,
        MEMORY_ADDRESSES['player_hp_high']: 0,
        MEMORY_ADDRESSES['player_max_hp']: 24,
        MEMORY_ADDRESSES['player_max_hp_high']: 0,
        MEMORY_ADDRESSES['player_status']: 0,
        MEMORY_ADDRESSES['player_species']: 155,  # Cyndaquil
    }
    return MockPyBoy(initial_memory)

@pytest.fixture
def mock_pyboy_in_battle():
    """Mock PyBoy in a battle state."""
    initial_memory = {
        MEMORY_ADDRESSES['in_battle']: 1,
        MEMORY_ADDRESSES['battle_turn']: 2,
        MEMORY_ADDRESSES['enemy_species']: 16,  # Pidgey
        MEMORY_ADDRESSES['enemy_level']: 4,
        MEMORY_ADDRESSES['enemy_hp_low']: 15,
        MEMORY_ADDRESSES['enemy_hp_high']: 0,
        MEMORY_ADDRESSES['player_active_slot']: 0,
        MEMORY_ADDRESSES['move_selected']: 0,
    }
    return MockPyBoy(initial_memory)

@pytest.fixture
def mock_pyboy_exploring():
    """Mock PyBoy in exploration state."""
    initial_memory = {
        MEMORY_ADDRESSES['player_map']: 26,  # New Bark Town
        MEMORY_ADDRESSES['player_x']: 10,
        MEMORY_ADDRESSES['player_y']: 12,
        MEMORY_ADDRESSES['player_direction']: 0,  # Down
        MEMORY_ADDRESSES['money_low']: 100,
        MEMORY_ADDRESSES['money_mid']: 0,
        MEMORY_ADDRESSES['money_high']: 0,
        MEMORY_ADDRESSES['badges']: 0,
    }
    return MockPyBoy(initial_memory)

@pytest.fixture
def mock_screen_overworld():
    """Mock screen data for overworld state."""
    screen = np.ones((144, 160, 3), dtype=np.uint8) * 128  # Gray background
    # Add some "sprite" variation
    screen[50:90, 60:100] = 200  # Lighter region
    return screen

@pytest.fixture
def mock_screen_battle():
    """Mock screen data for battle state."""
    screen = np.zeros((144, 160, 3), dtype=np.uint8)  # Dark background
    # Add high-contrast battle elements
    screen[20:60, 100:140] = 255  # Enemy area
    screen[80:120, 20:60] = 255  # Player area
    return screen

@pytest.fixture
def mock_screen_menu():
    """Mock screen data for menu state."""
    screen = np.zeros((144, 160, 3), dtype=np.uint8)  # Dark background
    # Add menu box
    screen[20:124, 10:150] = 255  # White menu area
    return screen

@pytest.fixture
def mock_screen_dialogue():
    """Mock screen data for dialogue state."""
    screen = np.ones((144, 160, 3), dtype=np.uint8) * 128  # Gray background
    # Add dialogue box
    screen[100:140, 5:155] = 255  # White dialogue area
    return screen

def create_game_state(**kwargs) -> Dict:
    """Create a test game state with defaults."""
    state = {
        'party_count': 0,
        'player_map': 24,  # Player's room
        'player_x': 1,
        'player_y': 1,
        'facing': 0,
        'money': 0,
        'badges': 0,
        'badges_count': 0,
        'in_battle': 0,
        'step_counter': 0,
    }
    state.update(kwargs)
    return state
