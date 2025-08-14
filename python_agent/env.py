"""
env.py - Gym-style environment wrapper for Pokémon Crystal RL training
"""

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces
import numpy as np
import json
import os
import time
import subprocess
from typing import Dict, Any, Tuple, Optional

from memory_map import MEMORY_ADDRESSES
from utils import calculate_reward, preprocess_state


class PokemonCrystalEnv(gym.Env):
    """
    OpenAI Gym environment for Pokémon Crystal reinforcement learning
    """
    
    def __init__(self, 
                 emulator_path: str = "bizhawk",
                 rom_path: str = "pokemon_crystal.gbc",
                 lua_script_path: str = "../lua_bridge/crystal_bridge.lua",
                 max_steps: int = 10000,
                 render_mode: Optional[str] = None):
        """
        Initialize the Pokémon Crystal environment
        
        Args:
            emulator_path: Path to the emulator executable
            rom_path: Path to the Pokémon Crystal ROM file
            lua_script_path: Path to the Lua bridge script
            max_steps: Maximum steps per episode
            render_mode: Rendering mode ('human', 'rgb_array', or None)
        """
        super().__init__()
        
        self.emulator_path = emulator_path
        self.rom_path = rom_path
        self.lua_script_path = lua_script_path
        self.max_steps = max_steps
        self.render_mode = render_mode
        
        # Action space: 9 possible actions (including no-op)
        # 0: No action, 1: Up, 2: Down, 3: Left, 4: Right, 
        # 5: A, 6: B, 7: Start, 8: Select
        self.action_space = spaces.Discrete(9)
        
        # Observation space: normalized game state values
        # This will be customized based on what state information we want to include
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(20,), dtype=np.float32
        )
        
        # Environment state
        self.emulator_process = None
        self.current_state = None
        self.step_count = 0
        self.episode_reward = 0
        self.previous_state = None
        
        # File paths for communication with Lua script
        self.state_file = "state.json"
        self.action_file = "action.txt"
        
    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """Reset the environment to initial state"""
        super().reset(seed=seed)
        
        # Clean up previous episode
        self._cleanup_files()
        
        # Start emulator if not running
        if self.emulator_process is None or self.emulator_process.poll() is not None:
            self._start_emulator()
        
        # Wait for initial state
        self._wait_for_state()
        
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
        
        # Send action to emulator
        self._send_action(action)
        
        # Wait for next state
        self._wait_for_state()
        
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
        """Render the environment (emulator handles this)"""
        pass
    
    def close(self):
        """Clean up environment"""
        self._cleanup_files()
        if self.emulator_process:
            self.emulator_process.terminate()
            self.emulator_process.wait()
    
    def _start_emulator(self):
        """Start the emulator with the Lua script"""
        cmd = [
            self.emulator_path,
            self.rom_path,
            "--lua=" + self.lua_script_path
        ]
        
        try:
            self.emulator_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            print("Emulator started successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to start emulator: {e}")
    
    def _send_action(self, action: int):
        """Send action to Lua script via file"""
        try:
            with open(self.action_file, 'w') as f:
                f.write(str(action))
        except Exception as e:
            raise RuntimeError(f"Failed to send action: {e}")
    
    def _wait_for_state(self, timeout: float = 5.0):
        """Wait for state file to be updated by Lua script"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if os.path.exists(self.state_file):
                try:
                    with open(self.state_file, 'r') as f:
                        state_data = json.load(f)
                    
                    # Update current state
                    self.previous_state = self.current_state
                    self.current_state = state_data
                    return
                except (json.JSONDecodeError, IOError):
                    # File might be being written, wait a bit
                    time.sleep(0.01)
                    continue
            
            time.sleep(0.01)
        
        raise TimeoutError("Timeout waiting for state update from emulator")
    
    def _get_observation(self) -> np.ndarray:
        """Convert game state to observation vector"""
        if self.current_state is None:
            return np.zeros(self.observation_space.shape, dtype=np.float32)
        
        return preprocess_state(self.current_state)
    
    def _calculate_reward(self) -> float:
        """Calculate reward based on current and previous states"""
        if self.current_state is None or self.previous_state is None:
            return 0.0
        
        return calculate_reward(self.current_state, self.previous_state)
    
    def _is_terminated(self) -> bool:
        """Check if episode should be terminated"""
        if self.current_state is None:
            return False
        
        # Example termination conditions:
        # - Player's HP reaches 0
        # - Game over screen
        # - Specific achievement reached (e.g., beating Elite Four)
        
        if self.current_state.get('player_hp', 1) <= 0:
            return True
        
        # Add more termination conditions as needed
        return False
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional info about the current state"""
        info = {
            'step_count': self.step_count,
            'episode_reward': self.episode_reward,
            'raw_state': self.current_state
        }
        
        if self.current_state:
            info.update({
                'player_position': (
                    self.current_state.get('player_x', 0),
                    self.current_state.get('player_y', 0)
                ),
                'player_level': self.current_state.get('player_level', 1),
                'player_hp': self.current_state.get('player_hp', 0),
                'player_money': self.current_state.get('money', 0),
                'badges': self.current_state.get('badges', 0)
            })
        
        return info
    
    def _cleanup_files(self):
        """Clean up temporary communication files"""
        for file_path in [self.state_file, self.action_file]:
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except OSError:
                    pass


# Register the environment
gym.register(
    id='PokemonCrystal-v0',
    entry_point='env:PokemonCrystalEnv',
    max_episode_steps=10000,
)
