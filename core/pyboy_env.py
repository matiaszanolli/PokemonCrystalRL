"""
PyBoy-based environment wrapper for Pokémon Crystal RL training
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os
import collections
from typing import Dict, Any, Tuple, Optional
from pyboy import PyBoy
from core.memory_map import MEMORY_ADDRESSES
from utils.utils import calculate_reward
from utils.preprocess_state import preprocess_state
from monitoring.monitoring_client import MonitoringClient

class PyBoyPokemonCrystalEnv(gym.Env):
    """
    OpenAI Gym Environment for Pokémon Crystal Using PyBoy Emulator
    
    This environment provides an interface for reinforcement learning with Pokémon Crystal,
    utilizing the PyBoy emulator for game interaction and state management.
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
            enable_monitoring: Enable monitoring client integration
            monitor_server_url: URL for the monitoring server
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
    
    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, dict]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        # Initialize/reset PyBoy if not already done
        if self.pyboy is None:
            try:
                self.pyboy = PyBoy(self.rom_path, window_type="headless" if self.headless else "SDL2")
                if self.save_state_path and os.path.exists(self.save_state_path):
                    self.pyboy.load_state(self.save_state_path)
            except Exception as e:
                if self.debug_mode:
                    print(f"Warning: Could not initialize PyBoy: {e}")
                # For testing, we'll continue without PyBoy
        
        # Reset environment state
        self.step_count = 0
        self.episode_reward = 0
        self.consecutive_same_screens = 0
        self.last_screen_hash = None
        self.recent_actions.clear()
        self.game_state_history.clear()
        self.previous_state = None
        
        # Get initial observation
        observation = self._get_observation()
        info = self._get_info()
        
        # Update monitoring
        if self.monitor:
            self.episode_number += 1
            try:
                self.monitor.update_episode(self.episode_number, 0.0, 0, False)
            except AttributeError:
                pass  # Monitor doesn't have this method
        
        return observation, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Execute action and return new state."""
        if self.pyboy is None:
            # For testing without PyBoy, just return dummy values
            observation = np.zeros(20, dtype=np.float32)
            reward = 0.0
            terminated = False
            truncated = self.step_count >= self.max_steps
            info = {}
            self.step_count += 1
            return observation, reward, terminated, truncated, info
        
        # Store previous state for reward calculation
        self.previous_state = self._get_info()
        
        # Execute action
        self._execute_action(action)
        self.recent_actions.append(action)
        
        # Update state
        self.step_count += 1
        observation = self._get_observation()
        reward = self._calculate_reward()
        terminated = self._is_terminated()
        truncated = self.step_count >= self.max_steps
        info = self._get_info()
        
        # Update episode reward
        self.episode_reward += reward
        
        return observation, reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        """Get current observation as normalized array."""
        if self.pyboy is None:
            return np.zeros(20, dtype=np.float32)
        
        # Get current game state info
        info = self._get_info()
        
        # Get screen data for stuck detection
        try:
            screen = self.pyboy.screen_image()
            new_hash = self._get_screen_hash(screen)
            if new_hash == self.last_screen_hash:
                self.consecutive_same_screens += 1
            else:
                self.consecutive_same_screens = 0
            self.last_screen_hash = new_hash
        except:
            pass  # Skip screen processing if it fails
        
        # Add stuck detection info to state
        info['consecutive_same_screens'] = self.consecutive_same_screens
        info['recent_actions'] = list(self.recent_actions)
        info['step_count'] = self.step_count
        info['episode_reward'] = self.episode_reward
        
        # Process state using utility function
        try:
            processed_state = preprocess_state(info)
        except:
            # Fallback if preprocess_state fails
            processed_state = np.zeros(20, dtype=np.float32)
        
        return processed_state

    def _calculate_reward(self) -> float:
        """Calculate reward for current state."""
        current_state = self._get_info()
        current_state['consecutive_same_screens'] = self.consecutive_same_screens
        current_state['recent_actions'] = list(self.recent_actions)
        
        try:
            return calculate_reward(
                current_state=current_state,
                previous_state=self.previous_state,
                consecutive_same_screens=self.consecutive_same_screens,
                recent_actions=list(self.recent_actions)
            )
        except:
            return 0.0  # Fallback reward

    def _get_info(self) -> dict:
        """Get current environment info."""
        if self.pyboy is None:
            return {
                'player_x': 0,
                'player_y': 0,
                'player_map': 0,
                'money': 0,
                'badges': 0,
                'party': {'count': 0, 'pokemon': []},
                'step_count': self.step_count,
                'episode_reward': self.episode_reward,
                'consecutive_same_screens': self.consecutive_same_screens
            }
        
        try:
            return {
                'player_x': self.pyboy.memory[self.memory_addresses['player_x']],
                'player_y': self.pyboy.memory[self.memory_addresses['player_y']],
                'player_map': self.pyboy.memory[self.memory_addresses['player_map']],
                'money': self._read_bcd_money(),
                'badges': self._read_badges(),
                'party': self._read_party_data(),
                'step_count': self.step_count,
                'episode_reward': self.episode_reward,
                'consecutive_same_screens': self.consecutive_same_screens
            }
        except Exception as e:
            if self.debug_mode:
                print(f"Warning: Error reading game info: {e}")
            return {
                'player_x': 0,
                'player_y': 0,
                'player_map': 0,
                'money': 0,
                'badges': 0,
                'party': {'count': 0, 'pokemon': []},
                'step_count': self.step_count,
                'episode_reward': self.episode_reward,
                'consecutive_same_screens': self.consecutive_same_screens
            }

    def _read_badges(self) -> int:
        """Read badge data from memory."""
        if self.pyboy is None:
            return 0
            
        try:
            johto_badges = self.pyboy.memory[self.memory_addresses['badges']]
            kanto_badges = self.pyboy.memory[self.memory_addresses['kanto_badges']]
            return (johto_badges << 8) | kanto_badges
        except:
            return 0

    def _read_party_data(self) -> dict:
        """Read Pokemon party data from memory."""
        if self.pyboy is None:
            return {'count': 0, 'pokemon': []}
            
        try:
            count = self.pyboy.memory[self.memory_addresses['party_count']]
            party = []
            
            for i in range(min(count, 6)):
                pokemon_addr = self.memory_addresses['party_pokemon'] + (i * 44)  # Each Pokemon entry is 44 bytes
                try:
                    species = self.pyboy.memory[pokemon_addr]
                    level = self.pyboy.memory[pokemon_addr + 31]
                    hp = (self.pyboy.memory[pokemon_addr + 1] << 8) | self.pyboy.memory[pokemon_addr + 2]
                    
                    party.append({
                        'species': species,
                        'level': level,
                        'hp': hp
                    })
                except KeyError:
                    # Skip this Pokemon if memory address doesn't exist
                    break
            
            return {
                'count': len(party),  # Use actual count of successfully read Pokemon
                'pokemon': party
            }
        except:
            return {'count': 0, 'pokemon': []}

    def _execute_action(self, action: int) -> None:
        """Execute the given action in the environment."""
        if self.pyboy is None:
            return
            
        try:
            # Clear previous input
            self.pyboy.send_input(self.pyboy.WindowEvent.RELEASE_ARROW_UP)
            self.pyboy.send_input(self.pyboy.WindowEvent.RELEASE_ARROW_DOWN)
            self.pyboy.send_input(self.pyboy.WindowEvent.RELEASE_ARROW_LEFT)
            self.pyboy.send_input(self.pyboy.WindowEvent.RELEASE_ARROW_RIGHT)
            self.pyboy.send_input(self.pyboy.WindowEvent.RELEASE_BUTTON_A)
            self.pyboy.send_input(self.pyboy.WindowEvent.RELEASE_BUTTON_B)
            self.pyboy.send_input(self.pyboy.WindowEvent.RELEASE_BUTTON_START)
            self.pyboy.send_input(self.pyboy.WindowEvent.RELEASE_BUTTON_SELECT)
            
            # Map action to PyBoy input
            action_map = {
                1: self.pyboy.WindowEvent.PRESS_ARROW_UP,
                2: self.pyboy.WindowEvent.PRESS_ARROW_DOWN,
                3: self.pyboy.WindowEvent.PRESS_ARROW_LEFT,
                4: self.pyboy.WindowEvent.PRESS_ARROW_RIGHT,
                5: self.pyboy.WindowEvent.PRESS_BUTTON_A,
                6: self.pyboy.WindowEvent.PRESS_BUTTON_B,
                7: self.pyboy.WindowEvent.PRESS_BUTTON_START,
                8: self.pyboy.WindowEvent.PRESS_BUTTON_SELECT
            }
            
            if action in action_map:
                self.pyboy.send_input(action_map[action])
            
            # Tick the emulator
            self.pyboy.tick()
        except:
            pass  # Skip action execution if it fails

    def _is_terminated(self) -> bool:
        """Check if episode should terminate."""
        if self.pyboy is None:
            return False  # Don't terminate if PyBoy isn't initialized
            
        # Check for game over conditions
        if self.consecutive_same_screens > 100:  # Stuck detection
            return True
            
        try:
            party = self._read_party_data()
            if party['count'] > 0:
                all_fainted = all(p['hp'] == 0 for p in party['pokemon'])
                if all_fainted:
                    return True
        except:
            pass  # Skip termination check if it fails
                
        return False

    def _get_screen_hash(self, screen) -> int:
        """Generate hash of screen for stuck detection."""
        if screen is None:
            return 0
        try:
            return hash(screen.tobytes())
        except:
            return 0

    def render(self, mode: str = 'human'):
        """Render the environment."""
        if self.pyboy is None:
            return None
            
        try:
            if mode == 'rgb_array':
                return np.array(self.pyboy.screen_image())
            elif mode == 'human':
                if not self.headless:
                    # PyBoy handles the display automatically when not in headless mode
                    return None
        except:
            pass  # Skip rendering if it fails
        return None

    def close(self):
        """Clean up resources."""
        if self.pyboy is not None:
            try:
                self.pyboy.stop()
            except:
                pass  # Ignore errors during cleanup
            self.pyboy = None
        
        if self.monitor:
            # Send final episode update if monitor has the method
            try:
                self.monitor.update_episode(
                    self.episode_number, 
                    self.episode_reward, 
                    self.step_count, 
                    False
                )
            except AttributeError:
                pass
        
    def _read_bcd_money(self) -> int:
        """Read money value in BCD format (3 bytes). In Pokémon Crystal (USA),
        money is stored in $ (Pokédollar) format as BCD digits.
        Memory layout: Three bytes, each byte holding two BCD digits.
        Example for 123456:
        Byte 0: 0x12 (first two digits, 12)
        Byte 1: 0x34 (middle two digits, 34)
        Byte 2: 0x56 (last two digits, 56)
        
        Each byte encodes two decimal digits in BCD format:
        - High nibble (4 bits) = 10s digit
        - Low nibble (4 bits) = 1s digit
        """        
        if self.pyboy is None:
            return 0
            
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
