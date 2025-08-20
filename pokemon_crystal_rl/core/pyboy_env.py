"""
PyBoy-based environment wrapper for Pokémon Crystal RL training
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os
import collections
from typing import Dict, Any, Tuple, Optional

try:
    from pyboy import PyBoy
except ImportError:
    print("Warning: Could not import PyBoy. PyBoy functionality will be disabled.")
    PyBoy = None

from pokemon_crystal_rl.core.memory_map import MEMORY_ADDRESSES
from pokemon_crystal_rl.utils import calculate_reward, preprocess_state
from pokemon_crystal_rl.monitoring.monitoring_client import MonitoringClient


class PyBoyPokemonCrystalEnv(gym.Env):
    """
    OpenAI Gym environment for Pokémon Crystal using PyBoy emulator
    """
    
    def __init__(self, 
                 rom_path: str = "../pokecrystal.gbc",
                 save_state_path: str = "../pokecrystal.ss1",
                 max_steps: int = 10000,
                 render_mode: Optional[str] = None,
                 headless: bool = True,
                 debug_mode: bool = False,
                 enable_monitoring: bool = True,
                 monitor_server_url: str = "http://localhost:5000"):
        """
        Initialize the PyBoy Pokémon Crystal environment
        
        Args:
            rom_path: Path to the Pokémon Crystal ROM file
            save_state_path: Path to the save state file
            max_steps: Maximum steps per episode
            render_mode: Rendering mode ('human', 'rgb_array', or None)
            headless: Whether to run in headless mode (no GUI)
            debug_mode: Enable debug mode for detailed logging
        """
        super().__init__()
        
        self.rom_path = os.path.abspath(rom_path)
        self.save_state_path = os.path.abspath(save_state_path) if save_state_path else None
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.headless = headless
        self.debug_mode = debug_mode
        self.enable_monitoring = enable_monitoring
        
        # Initialize monitoring client
        self.monitor = None
        if enable_monitoring:
            try:
                self.monitor = MonitoringClient(monitor_server_url, auto_start=True)
                if self.debug_mode:
                    print(f"✓ Monitoring enabled: {self.monitor.is_server_available()}")
            except Exception as e:
                if self.debug_mode:
                    print(f"⚠ Monitoring initialization failed: {e}")
                self.monitor = None
        
        # Action space: 9 possible actions (including no-op)
        # 0: No action, 1: Up, 2: Down, 3: Left, 4: Right, 
        # 5: A, 6: B, 7: Start, 8: Select
        self.action_space = spaces.Discrete(9)
        
        # Observation space: normalized game state values
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(20,), dtype=np.float32
        )
        
        # Initialize PyBoy
        self.pyboy = None
        self.window_wrapper = None
        
        # Environment state
        self.step_count = 0
        self.episode_reward = 0
        self.previous_state = None
        self.current_state = None
        self.episode_number = 0
        
        # Enhanced state tracking for rewards
        self.consecutive_same_screens = 0
        self.last_screen_hash = None
        self.recent_actions = collections.deque(maxlen=10)  # Track last 10 actions
        self.game_state_history = collections.deque(maxlen=5)  # Track recent game states
        
        # Action mapping for monitoring
        self.action_map = {
            0: "NONE", 1: "UP", 2: "DOWN", 3: "LEFT", 4: "RIGHT",
            5: "A", 6: "B", 7: "START", 8: "SELECT"
        }
        
        # Memory addresses for Pokemon Crystal (verified addresses)
        self.memory_addresses = {
            'player_x': 0xDCB8,
            'player_y': 0xDCB9, 
            'player_map': 0xDCB5,
            'player_direction': 0xDCB6,
            'money': 0xD84E,  # 3 bytes BCD format: $xxxx stored as BCD
            'badges': 0xD855,  # Johto badges byte
            'kanto_badges': 0xD856,  # Kanto badges byte  
            'party_count': 0xDCD7,
            'party_pokemon': 0xDCDF,  # Start of party data
            'player_name': 0xD47D,
            'options': 0xD355,
        }
        
    def _read_bcd_money(self) -> int:
        """Read money value in BCD format (3 bytes). In Pokémon Crystal (USA),
        money is stored in Ᵽ (Pokédollar) format as BCD digits.
        Memory layout: Three bytes, each byte holding two BCD digits.
        Example for Ᵽ123456:
        Byte 0: 0x12 (first two digits, 12)
        Byte 1: 0x34 (middle two digits, 34)
        Byte 2: 0x56 (last two digits, 56)
        
        Each byte encodes two decimal digits in BCD format:
        - High nibble (4 bits) = 10s digit
        - Low nibble (4 bits) = 1s digit
        """        
        money_addr = self.memory_addresses['money']
        
        # Read all three bytes
        byte0 = self.pyboy.memory[money_addr]
        byte1 = self.pyboy.memory[money_addr + 1]
        byte2 = self.pyboy.memory[money_addr + 2]
        
        # Check for invalid BCD digits
        for byte in [byte0, byte1, byte2]:
            high = (byte >> 4) & 0xF
            low = byte & 0xF
            if high > 9 or low > 9:
                if self.debug_mode:
                    print(f"Warning: Invalid BCD digit in money value")
                return 0
        
        # Extract BCD digits from each byte
        d0 = (byte0 >> 4) & 0xF  # 100,000s
        d1 = byte0 & 0xF         # 10,000s
        d2 = (byte1 >> 4) & 0xF  # 1,000s
        d3 = byte1 & 0xF         # 100s
        d4 = (byte2 >> 4) & 0xF  # 10s
        d5 = byte2 & 0xF         # 1s
        
        # Combine digits with place values to get decimal result
        result = (d0 * 100000 + 
                 d1 * 10000 + 
                 d2 * 1000 + 
                 d3 * 100 + 
                 d4 * 10 + 
                 d5)
        
        return result
