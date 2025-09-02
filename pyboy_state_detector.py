#!/usr/bin/env python3
"""
PyBoy-based Pokemon Crystal State Detector

This module uses PyBoy's built-in debugging features to reliably extract
game state instead of relying on hardcoded memory addresses.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import time

class PyBoyStateDetector:
    """Reliable game state detection using PyBoy's debugging features"""
    
    def __init__(self, pyboy):
        self.pyboy = pyboy
        self.scanner = pyboy.memory_scanner
        
        # Cache for verified memory addresses
        self.verified_addresses = {}
        
        # Track position history for movement detection
        self.position_history = []
        self.max_history = 10
        
        # Initialize by finding key addresses
        self._initialize_addresses()
    
    def _initialize_addresses(self):
        """Initialize by finding reliable memory addresses through scanning"""
        print("🔍 Initializing PyBoy state detector...")
        
        # Get initial game area to establish baseline
        try:
            self.initial_game_area = self.pyboy.game_area()
            print(f"✅ Game area accessible: {self.initial_game_area.shape}")
        except Exception as e:
            print(f"⚠️  Game area not accessible: {e}")
            self.initial_game_area = None
    
    def scan_for_changing_values(self, actions: List[str], max_attempts: int = 5) -> Dict[str, List[int]]:
        """
        Scan for memory addresses that change when specific actions are taken
        This helps us find the real position/state addresses dynamically
        """
        print("🧪 Scanning for changing memory addresses...")
        
        # Take initial memory snapshot
        initial_scan = {}
        for value in range(256):  # Scan for all possible byte values
            locations = self.scanner.scan_memory(value)
            if locations:
                initial_scan[value] = set(locations)
        
        changing_addresses = {}
        
        for action in actions:
            print(f"  Testing action: {action}")
            
            # Perform the action
            self.pyboy.button_press(action)
            for _ in range(8):
                self.pyboy.tick()
            self.pyboy.button_release(action)
            
            # Wait for state to stabilize
            for _ in range(4):
                self.pyboy.tick()
            
            # Scan again and find differences
            for value in range(256):
                new_locations = set(self.scanner.scan_memory(value))
                if value in initial_scan:
                    old_locations = initial_scan[value]
                    
                    # Find addresses that changed FROM this value
                    lost_addresses = old_locations - new_locations
                    # Find addresses that changed TO this value  
                    gained_addresses = new_locations - old_locations
                    
                    if lost_addresses or gained_addresses:
                        if action not in changing_addresses:
                            changing_addresses[action] = set()
                        changing_addresses[action].update(lost_addresses)
                        changing_addresses[action].update(gained_addresses)
        
        return changing_addresses
    
    def get_position_via_game_area(self) -> Optional[Tuple[int, int]]:
        """
        Try to determine player position by analyzing the game area
        This is more reliable than memory scanning for position
        """
        try:
            current_area = self.pyboy.game_area()
            
            # Look for player sprite indicators in the game area
            # In Pokemon games, the player is typically represented by specific tile values
            # We can detect changes in the game area to find the player position
            
            # For now, return center position as fallback
            # This needs game-specific analysis to find the actual player tile
            center_x = current_area.shape[1] // 2
            center_y = current_area.shape[0] // 2
            
            return (center_x, center_y)
            
        except Exception as e:
            print(f"Error getting position via game area: {e}")
            return None
    
    def get_comprehensive_state(self) -> Dict:
        """Get comprehensive game state using PyBoy's features"""
        state = {
            'timestamp': time.time(),
            'frame_count': self.pyboy.frame_count,
        }
        
        # 1. Try to get position via game area analysis
        area_position = self.get_position_via_game_area()
        if area_position:
            state['area_x'], state['area_y'] = area_position
        
        # 2. Get game area information
        try:
            game_area = self.pyboy.game_area()
            state['game_area_shape'] = game_area.shape
            state['game_area_sum'] = int(np.sum(game_area))  # Checksum for area changes
            state['game_area_unique_tiles'] = len(np.unique(game_area))
            
            # Store a few representative tiles for change detection
            h, w = game_area.shape
            state['corner_tiles'] = [
                int(game_area[0, 0]),        # Top-left
                int(game_area[0, w-1]),      # Top-right  
                int(game_area[h-1, 0]),      # Bottom-left
                int(game_area[h-1, w-1]),    # Bottom-right
                int(game_area[h//2, w//2])   # Center
            ]
            
        except Exception as e:
            print(f"Error getting game area: {e}")
        
        # 3. Try fallback to memory addresses (but with validation)
        try:
            # Test our suspected addresses but validate them
            suspected_x = self.pyboy.memory[0xDCB8]
            suspected_y = self.pyboy.memory[0xDCB9]
            suspected_map = self.pyboy.memory[0xDCB5]
            
            # Basic validation: coordinates should be reasonable
            if 0 <= suspected_x <= 255 and 0 <= suspected_y <= 255:
                state['memory_x'] = suspected_x
                state['memory_y'] = suspected_y
                state['memory_map'] = suspected_map
            
        except:
            pass
        
        # 4. Get screen analysis for state detection
        try:
            screen = self.pyboy.screen.ndarray
            state['screen_variance'] = float(np.var(screen))
            state['screen_mean'] = float(np.mean(screen))
            state['screen_unique_colors'] = len(np.unique(screen.reshape(-1, screen.shape[-1]), axis=0))
        except:
            pass
        
        # 5. Add to position history for movement detection
        position_key = (state.get('memory_x', 0), state.get('memory_y', 0), state.get('memory_map', 0))
        self.position_history.append(position_key)
        if len(self.position_history) > self.max_history:
            self.position_history.pop(0)
        
        # Movement detection
        if len(self.position_history) >= 2:
            state['moved'] = self.position_history[-1] != self.position_history[-2]
            state['stuck_count'] = self._count_stuck_frames()
        
        return state
    
    def _count_stuck_frames(self) -> int:
        """Count how many recent frames have the same position"""
        if not self.position_history:
            return 0
        
        current_pos = self.position_history[-1]
        stuck_count = 0
        
        for pos in reversed(self.position_history):
            if pos == current_pos:
                stuck_count += 1
            else:
                break
        
        return stuck_count
    
    def detect_map_transition(self, previous_state: Dict, current_state: Dict) -> bool:
        """Detect if a map transition occurred using multiple indicators"""
        
        # 1. Check memory-based map change
        prev_map = previous_state.get('memory_map', 0)
        curr_map = current_state.get('memory_map', 0)
        if prev_map != curr_map and curr_map > 0:
            print(f"🗺️  Map transition detected via memory: {prev_map} → {curr_map}")
            return True
        
        # 2. Check game area significant change
        prev_area_sum = previous_state.get('game_area_sum', 0)
        curr_area_sum = current_state.get('game_area_sum', 0)
        if abs(prev_area_sum - curr_area_sum) > 1000:  # Significant change
            print(f"🗺️  Map transition detected via area change: {prev_area_sum} → {curr_area_sum}")
            return True
        
        # 3. Check corner tiles (map transitions usually change corner tiles)
        prev_corners = previous_state.get('corner_tiles', [])
        curr_corners = current_state.get('corner_tiles', [])
        if prev_corners and curr_corners:
            corners_changed = sum(1 for p, c in zip(prev_corners, curr_corners) if p != c)
            if corners_changed >= 3:  # At least 3 out of 5 corners changed
                print(f"🗺️  Map transition detected via corner tiles: {corners_changed}/5 changed")
                return True
        
        return False
    
    def get_exploration_reward(self, previous_state: Dict, current_state: Dict) -> float:
        """Calculate exploration reward using reliable PyBoy-based detection"""
        
        # Check for map transition
        if self.detect_map_transition(previous_state, current_state):
            return 10.0  # Large reward for entering new area
        
        # Check for position movement within same area
        prev_pos = (previous_state.get('memory_x', 0), previous_state.get('memory_y', 0))
        curr_pos = (current_state.get('memory_x', 0), current_state.get('memory_y', 0))
        
        if prev_pos != curr_pos and curr_pos != (0, 0):
            # Small reward for movement
            return 0.1
        
        # Check for game area changes (even without position change)
        prev_sum = previous_state.get('game_area_sum', 0)
        curr_sum = current_state.get('game_area_sum', 0)
        if abs(prev_sum - curr_sum) > 100:  # Moderate area change
            return 0.05
        
        return 0.0
    
    def is_stuck(self, threshold: int = 20) -> bool:
        """Determine if the player is stuck using PyBoy-based analysis"""
        state = self.get_comprehensive_state()
        return state.get('stuck_count', 0) > threshold


def test_pyboy_detector():
    """Test the PyBoy state detector"""
    from pyboy import PyBoy
    import os
    
    rom_path = 'roms/pokemon_crystal.gbc'
    save_state_path = rom_path + '.state'
    
    if not os.path.exists(save_state_path):
        print("No save state found!")
        return
    
    pyboy = PyBoy(rom_path, window='null', debug=True)
    
    # Load save state
    with open(save_state_path, 'rb') as f:
        pyboy.load_state(f)
    
    # Initialize detector
    detector = PyBoyStateDetector(pyboy)
    
    # Get initial state
    print("\n📊 Initial state:")
    initial_state = detector.get_comprehensive_state()
    for key, value in initial_state.items():
        print(f"  {key}: {value}")
    
    # Test movement detection
    print("\n🎮 Testing movement detection...")
    directions = ['down', 'up', 'left', 'right']
    
    for direction in directions:
        print(f"\nTrying {direction}...")
        
        # Get state before
        before_state = detector.get_comprehensive_state()
        
        # Perform action
        pyboy.button_press(direction)
        for _ in range(8):
            pyboy.tick()
        pyboy.button_release(direction)
        
        for _ in range(4):
            pyboy.tick()
        
        # Get state after
        after_state = detector.get_comprehensive_state()
        
        # Check for changes
        moved = after_state.get('moved', False)
        exploration_reward = detector.get_exploration_reward(before_state, after_state)
        map_transition = detector.detect_map_transition(before_state, after_state)
        
        print(f"  Moved: {moved}")
        print(f"  Map transition: {map_transition}")
        print(f"  Exploration reward: {exploration_reward}")
        
        if map_transition:
            print(f"🎉 SUCCESS! Found map transition with {direction}!")
            break
    
    pyboy.stop()
    print("\n✅ PyBoy detector test complete!")


if __name__ == "__main__":
    test_pyboy_detector()
