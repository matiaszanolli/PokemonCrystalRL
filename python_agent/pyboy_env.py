"""
pyboy_env.py - PyBoy-based environment wrapper for Pokémon Crystal RL training
"""

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces

import numpy as np
import os
from typing import Dict, Any, Tuple, Optional
from pyboy import PyBoy

from memory_map import MEMORY_ADDRESSES
from utils import calculate_reward, preprocess_state


class PyBoyPokemonCrystalEnv(gym.Env):
    """
    OpenAI Gym environment for Pokémon Crystal using PyBoy emulator
    """
    
    def __init__(self, 
                 rom_path: str = "../pokecrystal.gbc",
                 save_state_path: str = "../pokecrystal.ss1",
                 max_steps: int = 10000,
                 render_mode: Optional[str] = None,
                 headless: bool = True):
        """
        Initialize the PyBoy Pokémon Crystal environment
        
        Args:
            rom_path: Path to the Pokémon Crystal ROM file
            save_state_path: Path to the save state file
            max_steps: Maximum steps per episode
            render_mode: Rendering mode ('human', 'rgb_array', or None)
            headless: Whether to run in headless mode (no GUI)
        """
        super().__init__()
        
        self.rom_path = os.path.abspath(rom_path)
        self.save_state_path = os.path.abspath(save_state_path) if save_state_path else None
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.headless = headless
        
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
        
        # Memory addresses (PyBoy uses different addressing)
        self.memory_addresses = {
            'player_x': 0xDCB8,
            'player_y': 0xDCB9,
            'player_map': 0xDCB5,
            'player_direction': 0xDCB6,
            'money': [0xD84E, 0xD84F, 0xD850],  # 3 bytes BCD
            'badges': [0xD857, 0xD858],  # 2 bytes
            'party_count': 0xDCD7,
            'party_pokemon': 0xDCDF,  # Start of party data
            'player_name': 0xD47D,
            'options': 0xD355,
        }
        
    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """Reset the environment to initial state"""
        super().reset(seed=seed)
        
        if self.pyboy is not None:
            self.pyboy.stop()
        
        # Initialize PyBoy with the ROM
        self.pyboy = PyBoy(
            self.rom_path,
            window="null" if self.headless else "SDL2",
            debug=False,
            sound=False,
            sound_emulated=False
        )
        
        # No need for window wrapper with PyBoy button methods
        
        # Load save state if available
        if self.save_state_path and os.path.exists(self.save_state_path):
            print(f"Loading save state from: {self.save_state_path}")
            with open(self.save_state_path, 'rb') as f:
                self.pyboy.load_state(f)
        else:
            print("No save state found, starting from beginning")
        
        # Let the game run for a few frames to initialize
        for _ in range(30):
            self.pyboy.tick()
        
        # Reset episode tracking
        self.step_count = 0
        self.episode_reward = 0
        self.previous_state = None
        
        # Get initial observation
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one step in the environment"""
        self.step_count += 1
        
        # Execute action
        self._execute_action(action)
        
        # Run emulator for a few frames to process the action
        for _ in range(8):  # Process action over multiple frames
            self.pyboy.tick()
        
        # Get observation and reward
        obs = self._get_observation()
        reward = self._calculate_reward()
        
        # Check if episode is done
        terminated = self._is_terminated()
        truncated = self.step_count >= self.max_steps
        
        # Update episode reward
        self.episode_reward += reward
        
        info = self._get_info()
        
        return obs, reward, terminated, truncated, info
    
    def render(self):
        """Render the environment"""
        if self.render_mode == "human" and not self.headless:
            # PyBoy handles rendering automatically in non-headless mode
            pass
        elif self.render_mode == "rgb_array":
            return np.array(self.pyboy.screen_image())
    
    def close(self):
        """Clean up environment"""
        if self.pyboy:
            self.pyboy.stop()
    
    def _execute_action(self, action: int):
        """Execute the given action"""
        # Release all buttons first
        self.pyboy.button_release("up")
        self.pyboy.button_release("down")
        self.pyboy.button_release("left")
        self.pyboy.button_release("right")
        self.pyboy.button_release("a")
        self.pyboy.button_release("b")
        self.pyboy.button_release("start")
        self.pyboy.button_release("select")
        
        # Execute the specific action
        if action == 1:  # Up
            self.pyboy.button_press("up")
        elif action == 2:  # Down
            self.pyboy.button_press("down")
        elif action == 3:  # Left
            self.pyboy.button_press("left")
        elif action == 4:  # Right
            self.pyboy.button_press("right")
        elif action == 5:  # A
            self.pyboy.button_press("a")
        elif action == 6:  # B
            self.pyboy.button_press("b")
        elif action == 7:  # Start
            self.pyboy.button_press("start")
        elif action == 8:  # Select
            self.pyboy.button_press("select")
        # action == 0 is no-op, no button press needed
    
    def _get_observation(self) -> np.ndarray:
        """Convert game state to observation vector"""
        self._update_state()
        if self.current_state is None:
            return np.zeros(self.observation_space.shape, dtype=np.float32)
        
        return preprocess_state(self.current_state)
    
    def _update_state(self):
        """Update current game state by reading from memory"""
        if not self.pyboy:
            return
        
        self.previous_state = self.current_state
        
        # Read game state from memory
        state = {
            'player': {
                'x': self.pyboy.memory[self.memory_addresses['player_x']],
                'y': self.pyboy.memory[self.memory_addresses['player_y']],
                'map': self.pyboy.memory[self.memory_addresses['player_map']],
                'direction': self.pyboy.memory[self.memory_addresses['player_direction']],
                'money': self._read_bcd_money(),
                'badges': self._read_badges(),
            },
            'party': self._read_party_data(),
            'frame_count': self.step_count,
            'game_time': self.pyboy.frame_count,
        }
        
        self.current_state = state
    
    def _read_bcd_money(self) -> int:
        """Read money value in BCD format"""
        money = 0
        for i, addr in enumerate(self.memory_addresses['money']):
            byte_val = self.pyboy.memory[addr]
            tens = (byte_val >> 4) & 0xF
            ones = byte_val & 0xF
            money = money * 100 + tens * 10 + ones
        return money
    
    def _read_badges(self) -> int:
        """Read badge count"""
        badge_byte1 = self.pyboy.memory[self.memory_addresses['badges'][0]]
        badge_byte2 = self.pyboy.memory[self.memory_addresses['badges'][1]]
        return (badge_byte1 << 8) | badge_byte2
    
    def _read_party_data(self) -> list:
        """Read party Pokemon data"""
        party = []
        party_count = self.pyboy.memory[self.memory_addresses['party_count']]
        
        if party_count > 6:  # Sanity check
            party_count = 0
        
        party_start = self.memory_addresses['party_pokemon']
        
        for i in range(min(party_count, 6)):
            pokemon_offset = party_start + (i * 48)  # Each party Pokemon is 48 bytes
            
            pokemon = {
                'species': self.pyboy.memory[pokemon_offset],
                'level': self.pyboy.memory[pokemon_offset + 31],
                'hp': self._read_16bit(pokemon_offset + 34),
                'max_hp': self._read_16bit(pokemon_offset + 36),
                'status': self.pyboy.memory[pokemon_offset + 32],
                'experience': self._read_24bit(pokemon_offset + 8),
            }
            party.append(pokemon)
        
        return party
    
    def _read_16bit(self, address: int) -> int:
        """Read a 16-bit big-endian value from memory"""
        high = self.pyboy.memory[address]
        low = self.pyboy.memory[address + 1]
        return (high << 8) | low
    
    def _read_24bit(self, address: int) -> int:
        """Read a 24-bit big-endian value from memory"""
        high = self.pyboy.memory[address]
        mid = self.pyboy.memory[address + 1]
        low = self.pyboy.memory[address + 2]
        return (high << 16) | (mid << 8) | low
    
    def _calculate_reward(self) -> float:
        """Calculate reward based on current and previous states"""
        if self.current_state is None or self.previous_state is None:
            return 0.0
        
        return calculate_reward(self.current_state, self.previous_state)
    
    def _is_terminated(self) -> bool:
        """Check if episode should be terminated"""
        if self.current_state is None:
            return False
        
        # Check if all party Pokemon are fainted
        party = self.current_state.get('party', [])
        if party:
            all_fainted = all(pokemon['hp'] <= 0 for pokemon in party)
            if all_fainted:
                return True
        
        return False
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional info about the current state"""
        info = {
            'step_count': self.step_count,
            'episode_reward': self.episode_reward,
            'raw_state': self.current_state
        }
        
        if self.current_state and 'player' in self.current_state:
            player = self.current_state['player']
            info.update({
                'player_position': (player.get('x', 0), player.get('y', 0)),
                'player_map': player.get('map', 0),
                'player_money': player.get('money', 0),
                'badges': player.get('badges', 0),
                'party_size': len(self.current_state.get('party', [])),
            })
            
            # Add party Pokemon info
            party = self.current_state.get('party', [])
            if party:
                info['party_levels'] = [p['level'] for p in party]
                info['party_hp'] = [p['hp'] for p in party]
                info['party_species'] = [p['species'] for p in party]
        
        return info
    
    def save_state(self, filename: str):
        """Save current game state"""
        if self.pyboy:
            with open(filename, 'wb') as f:
                self.pyboy.save_state(f)
    
    def load_state(self, filename: str):
        """Load game state from file"""
        if self.pyboy and os.path.exists(filename):
            with open(filename, 'rb') as f:
                self.pyboy.load_state(f)


# Register the environment
gym.register(
    id='PyBoyPokemonCrystal-v0',
    entry_point='pyboy_env:PyBoyPokemonCrystalEnv',
    max_episode_steps=10000,
)
