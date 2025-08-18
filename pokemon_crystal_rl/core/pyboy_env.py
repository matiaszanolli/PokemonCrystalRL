"""
pyboy_env.py - PyBoy-based environment wrapper for Pokémon Crystal RL training
"""

import gymnasium as gym
from gymnasium import spaces

import numpy as np
import os
import collections
from typing import Dict, Any, Tuple, Optional
from pyboy import PyBoy

try:
    from .memory_map import MEMORY_ADDRESSES
except ImportError:
    from memory_map import MEMORY_ADDRESSES

try:
    from pokemon_crystal_rl.utils import calculate_reward, preprocess_state
except ImportError:
    from pokemon_crystal_rl.utils import calculate_reward, preprocess_state

try:
    from pokemon_crystal_rl.monitoring.monitoring_client import MonitoringClient
except ImportError:
    from monitoring.monitoring_client import MonitoringClient


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
        self.episode_number += 1
        
        # Reset enhanced state tracking
        self.consecutive_same_screens = 0
        self.last_screen_hash = None
        self.recent_actions.clear()
        self.game_state_history.clear()
        
        # Update monitoring if enabled
        if self.monitor:
            self.monitor.current_episode = self.episode_number
            self.monitor.current_step = 0
        
        # Get initial observation
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one step in the environment"""
        self.step_count += 1
        
        # Track action for enhanced rewards
        self.recent_actions.append(action)
        
        # Execute action
        self._execute_action(action)
        
        # Run emulator for a few frames to process the action
        for _ in range(8):  # Process action over multiple frames
            self.pyboy.tick()
        
        # Update screen hash tracking for stuck detection
        screenshot = self._get_screen_array()
        current_screen_hash = self._get_screen_hash(screenshot)
        
        if current_screen_hash == self.last_screen_hash:
            self.consecutive_same_screens += 1
        else:
            self.consecutive_same_screens = 0
            self.last_screen_hash = current_screen_hash
        
        # Get observation and reward
        obs = self._get_observation()
        reward = self._calculate_reward()
        
        # Check if episode is done
        terminated = self._is_terminated()
        truncated = self.step_count >= self.max_steps
        
        # Update episode reward
        self.episode_reward += reward
        
        info = self._get_info()
        
        # Send monitoring updates if enabled
        if self.monitor:
            self._send_monitoring_updates(action, reward, info)
        
        return obs, reward, terminated, truncated, info
    
    def render(self):
        """Render the environment"""
        if self.render_mode == "human" and not self.headless:
            # PyBoy handles rendering automatically in non-headless mode
            pass
        elif self.render_mode == "rgb_array":
            return self._get_screen_array()
    
    def close(self):
        """Clean up environment"""
        if self.pyboy:
            self.pyboy.stop()
        
        # Send final monitoring update
        if self.monitor:
            self.monitor.send_performance_update()
    
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
        
        # Basic game state detection for enhanced rewards
        screenshot = self._get_screen_array()
        game_state = self._detect_basic_game_state(screenshot)
        
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
            # Enhanced reward tracking
            'consecutive_same_screens': self.consecutive_same_screens,
            'recent_actions': list(self.recent_actions),
            'game_state': game_state,
        }
        
        # Add to game state history
        self.game_state_history.append(game_state)
        
        self.current_state = state
    
    def _read_bcd_money(self) -> int:
        """Read money value in BCD format (3 bytes starting at money address)"""
        money = 0
        money_addr = self.memory_addresses['money']
        
        # Read 3 bytes of BCD money data
        for i in range(3):
            byte_val = self.pyboy.memory[money_addr + i]
            high = (byte_val >> 4) & 0xF
            low = byte_val & 0xF
            money = money * 100 + high * 10 + low
        
        return money
    
    def _read_badges(self) -> int:
        """Read badge count from Johto and Kanto badge bytes"""
        johto_badges = self.pyboy.memory[self.memory_addresses['badges']]
        kanto_badges = self.pyboy.memory[self.memory_addresses['kanto_badges']]
        
        # Count number of bits set in both badge bytes
        total_badges = bin(johto_badges).count('1') + bin(kanto_badges).count('1')
        return total_badges
    
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
    
    def get_game_state(self) -> Dict[str, Any]:
        """Get the current game state for external use"""
        self._update_state()
        return self.current_state if self.current_state else {}
    
    def get_screenshot(self) -> np.ndarray:
        """Get current screenshot as RGB numpy array"""
        return self._get_screen_array()
    
    def _get_screen_array(self) -> np.ndarray:
        """Get screen data as numpy array using the correct PyBoy API"""
        if self.pyboy:
            try:
                # Use the PyBoy screen.ndarray method (confirmed working)
                if hasattr(self.pyboy, 'screen') and hasattr(self.pyboy.screen, 'ndarray'):
                    screen_data = self.pyboy.screen.ndarray.copy()
                    
                    if self.debug_mode:
                        print(f"🔍 Screen data shape: {screen_data.shape}, dtype: {screen_data.dtype}")
                    
                    # Handle different channel formats
                    if screen_data.shape[-1] == 4:  # RGBA format
                        # Convert RGBA to RGB by dropping alpha channel
                        screen_rgb = screen_data[:, :, :3]
                        return screen_rgb.astype(np.uint8)
                    elif screen_data.shape[-1] == 3:  # Already RGB
                        return screen_data.astype(np.uint8)
                    elif len(screen_data.shape) == 2:  # Grayscale
                        # Convert grayscale to RGB
                        screen_rgb = np.stack([screen_data, screen_data, screen_data], axis=-1)
                        return screen_rgb.astype(np.uint8)
                    else:
                        # Unknown format, try to reshape to RGB
                        if screen_data.size >= 144 * 160 * 3:
                            reshaped = screen_data.flatten()[:144*160*3].reshape((144, 160, 3))
                            return reshaped.astype(np.uint8)
                
                # Fallback: try PIL image method
                elif hasattr(self.pyboy, 'screen') and hasattr(self.pyboy.screen, 'image'):
                    pil_img = self.pyboy.screen.image()
                    screen_array = np.array(pil_img)
                    
                    # Convert PIL image to RGB if needed
                    if screen_array.shape[-1] == 4:  # RGBA
                        return screen_array[:, :, :3].astype(np.uint8)
                    elif screen_array.shape[-1] == 3:  # RGB
                        return screen_array.astype(np.uint8)
                
                # Last resort: try legacy screen_image method
                elif hasattr(self.pyboy, 'screen_image'):
                    screen_image = self.pyboy.screen_image()
                    screen_array = np.array(screen_image)
                    if len(screen_array.shape) >= 3 and screen_array.shape[-1] >= 3:
                        return screen_array[:, :, :3].astype(np.uint8)
                
            except Exception as e:
                if self.debug_mode:
                    print(f"🔍 Screen capture error: {e}")
                    if hasattr(self.pyboy, 'screen'):
                        print(f"🔍 Available screen methods: {[m for m in dir(self.pyboy.screen) if not m.startswith('_')]}")
        
        # Return empty array if no emulator or capture failed
        return np.zeros((144, 160, 3), dtype=np.uint8)
    
    def _send_monitoring_updates(self, action: int, reward: float, info: Dict[str, Any]):
        """Send updates to monitoring system"""
        if not self.monitor:
            return
        
        try:
            # Get current game state
            self._update_state()
            
            # Get player position and game info
            player_x = 0
            player_y = 0
            map_id = 0
            screen_type = "unknown"
            
            if self.current_state and 'player' in self.current_state:
                player = self.current_state['player']
                player_x = player.get('x', 0)
                player_y = player.get('y', 0)
                map_id = player.get('map', 0)
                
                # Basic screen type detection based on map or other criteria
                if map_id == 0:
                    screen_type = "menu"
                elif map_id > 0:
                    screen_type = "overworld"
            
            # Update step information
            self.monitor.update_step(
                step=self.step_count,
                reward=reward,
                action=self.action_map.get(action, "UNKNOWN"),
                screen_type=screen_type,
                map_id=map_id,
                player_x=player_x,
                player_y=player_y
            )
            
            # Update screenshot every few steps
            if self.step_count % 5 == 0:  # Update screenshot every 5 steps
                screenshot = self.get_screenshot()
                if screenshot is not None and screenshot.size > 0:
                    self.monitor.update_screenshot(screenshot)
            
        except Exception as e:
            if self.debug_mode:
                print(f"⚠ Monitoring update failed: {e}")
    
    def send_episode_end(self, success: bool = False):
        """Send episode end notification to monitoring"""
        if self.monitor:
            self.monitor.update_episode(
                episode=self.episode_number,
                total_reward=self.episode_reward,
                steps=self.step_count,
                success=success
            )
    
    def send_llm_decision(self, action: str, reasoning: str = "", context: Dict = None):
        """Send LLM decision to monitoring"""
        if self.monitor:
            self.monitor.update_llm_decision(action, reasoning, context)
    
    def send_text_update(self, text: str, text_type: str = "dialogue"):
        """Send text detection to monitoring"""
        if self.monitor:
            self.monitor.update_text(text, text_type)
    
    def is_monitoring_available(self) -> bool:
        """Check if monitoring is available"""
        return self.monitor is not None and self.monitor.is_server_available()
    
    def _get_screen_hash(self, screenshot: np.ndarray) -> int:
        """Get a hash of the screen for stuck detection"""
        if screenshot is None or screenshot.size == 0:
            return 0
        
        try:
            # Use aggressive sampling for performance
            sampled = screenshot[::8, ::8]  # Sample every 8th pixel
            if sampled.size == 0:
                return 0
            
            # Simple hash using mean and std
            mean_val = int(np.mean(sampled))
            std_val = int(np.std(sampled)) if sampled.size > 1 else 0
            
            # Add spatial features for better discrimination
            h, w = sampled.shape[:2]
            if h > 1 and w > 1:
                tl = int(np.mean(sampled[:h//2, :w//2]))  # Top-left
                br = int(np.mean(sampled[h//2:, w//2:]))  # Bottom-right
            else:
                tl, br = mean_val, mean_val
            
            return hash((mean_val, std_val, tl, br))
        except Exception:
            return 0
    
    def _detect_basic_game_state(self, screenshot: np.ndarray) -> str:
        """Basic game state detection for enhanced rewards"""
        if screenshot is None or screenshot.size == 0:
            return "unknown"
        
        try:
            # Simple brightness-based detection
            mean_brightness = np.mean(screenshot)
            color_variance = np.var(screenshot)
            
            # Basic state detection logic
            if mean_brightness < 10:
                return "loading"
            elif mean_brightness > 240:
                return "intro_sequence"
            elif mean_brightness >= 200 and color_variance < 100:
                return "title_screen"
            elif color_variance > 2000:
                return "title_screen"  # High contrast (logos, flashes)
            elif 95 <= mean_brightness <= 105:
                return "battle"
            elif 120 <= mean_brightness <= 180:
                return "menu"
            elif color_variance > 800:
                return "overworld"
            else:
                return "dialogue"  # Default assumption for medium brightness
        except Exception:
            return "unknown"


# Register the environment
gym.register(
    id='PyBoyPokemonCrystal-v0',
    entry_point='pokemon_crystal_rl.core.pyboy_env:PyBoyPokemonCrystalEnv',
    max_episode_steps=10000,
    reward_threshold=1000.0,  # Target score for considering the environment solved
)
