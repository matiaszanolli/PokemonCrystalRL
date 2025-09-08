"""
PyBoy Test Fixtures and Helpers

Common fixtures and helper functions for testing with PyBoy emulator.
"""

import pytest
import numpy as np
from typing import Dict, Optional
from unittest.mock import MagicMock
from config.memory_addresses import MEMORY_ADDRESSES
from config.constants import TRAINING_PARAMS

class MockMemory:
    """Mock PyBoy memory for testing."""
    def __init__(self, initial_state: Optional[Dict[int, int]] = None):
        self.memory = initial_state or {}
        
    def __getitem__(self, key):
        if key not in self.memory:
            raise KeyError(f"Memory address {key:X} not found")
        return self.memory[key]
        
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
    
    # Create high variance overworld with dramatic color differences
    # Need >10 unique colors and variance >3000
    
    # Create extreme contrast areas to boost variance
    screen[0:40, 0:80] = [255, 255, 255]    # Bright white area
    screen[40:80, 0:80] = [0, 0, 0]         # Black area (max contrast)
    screen[80:120, 0:80] = [255, 0, 0]      # Bright red
    screen[120:144, 0:80] = [0, 255, 0]     # Bright green
    
    screen[0:40, 80:160] = [0, 0, 255]      # Bright blue
    screen[40:80, 80:160] = [255, 255, 0]   # Bright yellow
    screen[80:120, 80:160] = [255, 0, 255]  # Bright magenta
    screen[120:144, 80:160] = [0, 255, 255] # Bright cyan
    
    # Add more distinct colors to get >10
    screen[20:40, 40:60] = [128, 64, 32]    # Brown
    screen[60:80, 60:100] = [192, 192, 192] # Light gray
    screen[100:120, 100:140] = [64, 128, 255] # Light blue
    screen[20:60, 120:140] = [255, 128, 64] # Orange
    
    return screen

@pytest.fixture
def mock_screen_battle():
    """Mock screen data for battle state."""
    screen = np.zeros((144, 160, 3), dtype=np.uint8)  # Dark background
    
    # Battle needs variance > 20000 and colors > 8
    # Create extreme high-contrast battle elements with many colors
    screen[10:50, 100:140] = [255, 0, 0]      # Bright red enemy area
    screen[50:90, 100:140] = [255, 255, 255]  # White highlights  
    screen[80:120, 20:60] = [0, 255, 0]       # Bright green player area
    screen[90:130, 20:60] = [255, 255, 0]     # Yellow player highlights
    
    # Add battle effects and UI elements with more colors
    screen[0:20, 0:160] = [0, 0, 255]         # Blue battle UI bar
    screen[124:144, 0:160] = [255, 0, 255]    # Magenta battle UI
    screen[60:80, 40:120] = [0, 255, 255]     # Cyan battle effects
    screen[40:60, 80:100] = [128, 64, 192]    # Purple battle element
    screen[100:120, 80:100] = [255, 128, 0]   # Orange battle effect
    
    # Add more distinct colors to get >8 
    screen[30:50, 130:150] = [64, 192, 128]   # Color 9: Teal
    screen[70:90, 130:150] = [192, 64, 128]   # Color 10: Pink
    
    # Add extreme noise patterns to boost variance >20000
    np.random.seed(123)  # Reproducible for tests
    for i in range(0, 144, 1):  # Every pixel, not every 2nd
        for j in range(0, 160, 1):
            if np.random.random() > 0.5:  # 50% of pixels get extreme variation
                screen[i, j] = np.random.choice([0, 255], 3)  # Pure white or black noise
    
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
