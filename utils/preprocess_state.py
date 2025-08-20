"""State preprocessing utilities."""

import numpy as np
from typing import Dict, Any, Union


def preprocess_state(screen: Union[Dict[str, Any], np.ndarray]) -> np.ndarray:
    """Preprocess the game state or screen data for the neural network.
    
    Args:
        screen: Raw screen data from PyBoy or game state dictionary
        
    Returns:
        np.ndarray: Processed observation vector or screen data
    """
    # If input is a game state dictionary, extract relevant features
    if isinstance(screen, dict):
        # Initialize observation vector with zeros
        obs = np.zeros(20, dtype=np.float32)
        
        # Extract player position and normalize
        if 'player' in screen:
            player = screen['player']
            obs[0] = player.get('x', 0) / 255.0  # Normalize position
            obs[1] = player.get('y', 0) / 255.0
            obs[2] = player.get('map', 0) / 255.0
            obs[3] = player.get('money', 0) / 999999.0  # Normalize money
            obs[4] = player.get('badges', 0) / 16.0  # Normalize badge count
        
        # Extract party information
        party = screen.get('party', [])
        if party:
            # Average party level
            levels = [p.get('level', 0) for p in party]
            obs[5] = np.mean(levels) / 100.0 if levels else 0
            
            # Party size
            obs[6] = len(party) / 6.0
            
            # Average HP percentage
            hp_percentages = [p.get('hp', 0) / p.get('max_hp', 1) for p in party]
            obs[7] = np.mean(hp_percentages) if hp_percentages else 0
        
        # Game state flags
        obs[8] = 1.0 if screen.get('in_battle', False) else 0.0
        obs[9] = 1.0 if screen.get('text_box_active', False) else 0.0
        obs[10] = 1.0 if screen.get('in_menu', False) else 0.0
        
        # Recent actions (last 5 normalized)
        recent_actions = screen.get('recent_actions', [])
        for i in range(5):
            if i < len(recent_actions):
                obs[11 + i] = recent_actions[i] / 8.0  # 8 possible actions
        
        # Additional state information
        obs[16] = screen.get('consecutive_same_screens', 0) / 100.0
        obs[17] = screen.get('frame_count', 0) / 10000.0
        obs[18] = screen.get('episode_reward', 0) / 1000.0
        
        # Game progress indicator (could be based on badges, money, etc.)
        progress = max(screen.get('badges', 0) / 16.0,
                      screen.get('money', 0) / 999999.0)
        obs[19] = progress
        
        return obs
    
    # Handle screen data input
    elif isinstance(screen, np.ndarray):
        # Ensure the input is a numpy array
        if not isinstance(screen, np.ndarray):
            raise ValueError("Screen must be a numpy array")
        
        # Convert grayscale to RGB if needed
        if len(screen.shape) == 2:
            screen = np.stack((screen,) * 3, axis=-1)
        
        # Convert RGBA to RGB if needed
        elif len(screen.shape) == 3 and screen.shape[2] == 4:
            screen = screen[:, :, :3]
        
        # Ensure correct shape
        if len(screen.shape) != 3 or screen.shape[2] != 3:
            raise ValueError(f"Invalid screen shape: {screen.shape}")
        
        # Ensure uint8 type
        if screen.dtype != np.uint8:
            screen = (screen * 255).astype(np.uint8)
        
        return screen
    else:
        raise ValueError("Input must be either a game state dictionary or numpy array")
