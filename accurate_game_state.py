#!/usr/bin/env python3
"""
Accurate Pokemon Crystal Game State Extractor

Using validated memory addresses for reliable state detection and reward calculation.
"""

from typing import Dict, List, Tuple, Optional
import time

class AccurateGameState:
    """Accurate game state extraction using validated memory addresses"""
    
    def __init__(self, pyboy):
        self.pyboy = pyboy
        self.memory = pyboy.memory
        
        # Track state history for change detection
        self.state_history = []
        self.max_history = 10
        
        # Visited locations for exploration tracking
        self.visited_locations = set()
        self.visited_maps = set()
        
    def get_complete_state(self) -> Dict:
        """Get comprehensive game state using accurate memory addresses"""
        
        # Party & Pokemon - First get the actual party count from memory
        # The party count is stored at the start of the party data structure
        party_count_addr = 0xD163
        actual_party_count = self.memory[party_count_addr]
        
        party = []
        for i in range(6):
            base = 0xD163 + 1 + i * 44  # +1 to skip the party count byte
            pokemon_data = {
                "species": self.memory[base],
                "held_item": self.memory[base + 1],
                "hp": self.memory[base + 4] + (self.memory[base + 5] << 8),
                "max_hp": self.memory[base + 6] + (self.memory[base + 7] << 8),
                "level": self.memory[base + 8],
                "status": self.memory[base + 9],
                "moves": [self.memory[base + 10 + j] for j in range(4)],
                "pp": [self.memory[base + 14 + j] for j in range(4)]
            }
            party.append(pokemon_data)
        
        # Money (3 bytes, little-endian)
        money = (self.memory[0xD347] + 
                (self.memory[0xD348] << 8) + 
                (self.memory[0xD349] << 16))
        
        # Location and movement
        map_id = self.memory[0xD35D]
        player_x = self.memory[0xD361]
        player_y = self.memory[0xD362]
        facing = self.memory[0xD363]
        
        # Step counter for movement tracking
        step_counter = self.memory[0xD164]
        
        # Battle state
        battle_active = bool(self.memory[0xD057])
        turn_counter = self.memory[0xD068] if battle_active else 0
        
        # Battle details if in battle
        battle_info = {}
        if battle_active:
            battle_info = {
                "opponent_species": self.memory[0xD0A5],
                "opponent_hp": self.memory[0xD0A8] + (self.memory[0xD0A9] << 8),
                "opponent_level": self.memory[0xD0AA],
                "player_active_slot": self.memory[0xD05E],
                "move_selected": self.memory[0xD05F]
            }
        
        # Progress tracking
        badges = self.memory[0xD359]  # Johto badges
        
        # Time played (optional for analytics)
        time_played_hours = self.memory[0xD3E1]
        
        # Pokedex progress (seen/caught flags)
        pokedex_seen = []
        pokedex_caught = []
        for i in range(7):  # 0xD350-0xD356
            seen_byte = self.memory[0xD350 + i]
            caught_byte = self.memory[0xD357 + i] if i < 6 else 0  # Caught flags might be different
            pokedex_seen.append(seen_byte)
            pokedex_caught.append(caught_byte)
        
        # Compile complete state
        state = {
            "timestamp": time.time(),
            "frame_count": self.pyboy.frame_count,
            
            # Location & Movement
            "map_id": map_id,
            "player_x": player_x,
            "player_y": player_y,
            "facing": facing,
            "coords": (player_x, player_y),
            "step_counter": step_counter,
            
            # Party & Pokemon
            "party": party,
            "party_count": actual_party_count if actual_party_count <= 6 else sum(1 for p in party if p["species"] > 0),
            
            # Player resources
            "money": money,
            "badges": badges,
            "badges_count": bin(badges).count('1'),
            
            # Battle state
            "in_battle": battle_active,
            "battle_turn": turn_counter,
            "battle_info": battle_info,
            
            # Progress tracking
            "time_played_hours": time_played_hours,
            "pokedex_seen": pokedex_seen,
            "pokedex_caught": pokedex_caught,
            
            # Derived values
            "total_hp": sum(p["hp"] for p in party if p["species"] > 0),
            "total_max_hp": sum(p["max_hp"] for p in party if p["species"] > 0),
            "average_level": sum(p["level"] for p in party if p["species"] > 0) / max(1, sum(1 for p in party if p["species"] > 0)),
            "has_pokemon": any(p["species"] > 0 for p in party)
        }
        
        # Add to history
        self.state_history.append(state)
        if len(self.state_history) > self.max_history:
            self.state_history.pop(0)
        
        # Track visited locations
        location_key = (map_id, player_x, player_y)
        self.visited_locations.add(location_key)
        self.visited_maps.add(map_id)
        
        return state
    
    def detect_changes(self, previous_state: Dict, current_state: Dict) -> Dict[str, bool]:
        """Detect various types of changes between states"""
        changes = {
            "moved": False,
            "map_changed": False,
            "battle_started": False,
            "battle_ended": False,
            "pokemon_gained": False,
            "pokemon_lost": False,
            "level_gained": False,
            "money_changed": False,
            "badge_gained": False,
            "hp_changed": False,
            "step_progressed": False
        }
        
        if not previous_state:
            return changes
        
        # Movement detection
        prev_coords = previous_state.get("coords", (0, 0))
        curr_coords = current_state.get("coords", (0, 0))
        changes["moved"] = prev_coords != curr_coords
        
        # Map change detection
        prev_map = previous_state.get("map_id", 0)
        curr_map = current_state.get("map_id", 0)
        changes["map_changed"] = prev_map != curr_map and curr_map > 0
        
        # Battle state changes
        prev_battle = previous_state.get("in_battle", False)
        curr_battle = current_state.get("in_battle", False)
        changes["battle_started"] = not prev_battle and curr_battle
        changes["battle_ended"] = prev_battle and not curr_battle
        
        # Pokemon changes
        prev_party_count = previous_state.get("party_count", 0)
        curr_party_count = current_state.get("party_count", 0)
        changes["pokemon_gained"] = curr_party_count > prev_party_count
        changes["pokemon_lost"] = curr_party_count < prev_party_count
        
        # Level changes
        prev_avg_level = previous_state.get("average_level", 0)
        curr_avg_level = current_state.get("average_level", 0)
        changes["level_gained"] = curr_avg_level > prev_avg_level
        
        # Money changes
        prev_money = previous_state.get("money", 0)
        curr_money = current_state.get("money", 0)
        changes["money_changed"] = prev_money != curr_money
        
        # Badge changes
        prev_badges = previous_state.get("badges_count", 0)
        curr_badges = current_state.get("badges_count", 0)
        changes["badge_gained"] = curr_badges > prev_badges
        
        # HP changes
        prev_hp = previous_state.get("total_hp", 0)
        curr_hp = current_state.get("total_hp", 0)
        changes["hp_changed"] = prev_hp != curr_hp
        
        # Step counter progression
        prev_steps = previous_state.get("step_counter", 0)
        curr_steps = current_state.get("step_counter", 0)
        changes["step_progressed"] = curr_steps > prev_steps
        
        return changes
    
    def calculate_exploration_reward(self, previous_state: Dict, current_state: Dict) -> float:
        """Calculate exploration reward based on accurate state detection"""
        if not previous_state:
            return 0.0
        
        reward = 0.0
        changes = self.detect_changes(previous_state, current_state)
        
        # Map transition reward (big!)
        if changes["map_changed"]:
            new_map = current_state.get("map_id", 0)
            if new_map not in self.visited_maps:
                reward += 20.0  # First time visiting this map
                print(f"🗺️  NEW MAP DISCOVERED: {new_map} (+20.0)")
            else:
                reward += 5.0   # Revisiting known map
                print(f"🗺️  Map transition: {previous_state.get('map_id')} → {new_map} (+5.0)")
        
        # Movement within map
        elif changes["moved"]:
            curr_location = (current_state.get("map_id", 0), 
                           current_state.get("player_x", 0), 
                           current_state.get("player_y", 0))
            
            if curr_location not in self.visited_locations:
                reward += 0.5  # New location within same map
                print(f"📍 New location: {curr_location} (+0.5)")
            else:
                reward += 0.1  # Revisiting location
        
        # Step counter progression (very small reward for any movement)
        elif changes["step_progressed"]:
            reward += 0.02
        
        return reward
    
    def calculate_progression_reward(self, previous_state: Dict, current_state: Dict) -> float:
        """Calculate progression-based rewards"""
        if not previous_state:
            return 0.0
        
        reward = 0.0
        changes = self.detect_changes(previous_state, current_state)
        
        # Pokemon acquisition (huge reward for first Pokemon!)
        if changes["pokemon_gained"]:
            curr_count = current_state.get("party_count", 0)
            if curr_count == 1:
                reward += 100.0  # First Pokemon is massive milestone
                print(f"🎉 FIRST POKEMON ACQUIRED! (+100.0)")
            else:
                reward += 25.0   # Additional Pokemon
                print(f"🎉 New Pokemon acquired! Party count: {curr_count} (+25.0)")
        
        # Level progression
        if changes["level_gained"]:
            reward += 10.0
            print(f"📈 Level gained! Avg level: {current_state.get('average_level', 0):.1f} (+10.0)")
        
        # Badge acquisition
        if changes["badge_gained"]:
            badge_count = current_state.get("badges_count", 0)
            reward += 50.0
            print(f"🏆 Badge earned! Total badges: {badge_count} (+50.0)")
        
        # Battle rewards
        if changes["battle_started"]:
            reward += 2.0
            print(f"⚔️  Battle started! (+2.0)")
        
        if changes["battle_ended"]:
            # Check if we likely won (still have HP)
            total_hp = current_state.get("total_hp", 0)
            if total_hp > 0:
                reward += 15.0
                print(f"🏆 Battle won! (+15.0)")
            else:
                reward -= 5.0
                print(f"💀 Battle lost (-5.0)")
        
        return reward
    
    def get_reward_breakdown(self, previous_state: Dict, current_state: Dict) -> Tuple[float, Dict[str, float]]:
        """Get complete reward breakdown"""
        rewards = {
            "exploration": self.calculate_exploration_reward(previous_state, current_state),
            "progression": self.calculate_progression_reward(previous_state, current_state),
            "time_penalty": -0.01  # Small penalty to encourage progress
        }
        
        total_reward = sum(rewards.values())
        return total_reward, rewards
    
    def is_stuck(self, threshold: int = 20) -> bool:
        """Determine if player is stuck based on recent history"""
        if len(self.state_history) < threshold:
            return False
        
        recent_states = self.state_history[-threshold:]
        recent_coords = [state.get("coords", (0, 0)) for state in recent_states]
        recent_maps = [state.get("map_id", 0) for state in recent_states]
        
        # Check if coordinates haven't changed
        coords_stuck = len(set(recent_coords)) == 1
        maps_stuck = len(set(recent_maps)) == 1
        
        return coords_stuck and maps_stuck


def test_accurate_state():
    """Test the accurate game state extractor"""
    from pyboy import PyBoy
    import os
    
    rom_path = 'roms/pokemon_crystal.gbc'
    save_state_path = rom_path + '.state'
    
    if not os.path.exists(save_state_path):
        print("No save state found!")
        return
    
    pyboy = PyBoy(rom_path, window='null', debug=False)
    
    # Load save state
    with open(save_state_path, 'rb') as f:
        pyboy.load_state(f)
    
    # Initialize state tracker
    state_tracker = AccurateGameState(pyboy)
    
    print("🔍 Testing Accurate Game State Extractor")
    print("=" * 50)
    
    # Get initial state
    initial_state = state_tracker.get_complete_state()
    
    print(f"📍 INITIAL STATE:")
    print(f"  Map: {initial_state['map_id']}")
    print(f"  Coordinates: {initial_state['coords']}")
    print(f"  Party count: {initial_state['party_count']}")
    print(f"  Money: ¥{initial_state['money']}")
    print(f"  Badges: {initial_state['badges_count']}")
    print(f"  Step counter: {initial_state['step_counter']}")
    print(f"  Has Pokemon: {initial_state['has_pokemon']}")
    
    # Test various actions
    actions = ['down', 'up', 'left', 'right', 'a', 'b', 'start', 'select']
    total_reward = 0
    
    previous_state = initial_state
    
    for i, action in enumerate(actions):
        print(f"\n🎮 Action {i+1}: {action.upper()}")
        
        # Perform action
        pyboy.button_press(action)
        for _ in range(10):
            pyboy.tick()
        pyboy.button_release(action)
        
        # Wait for state to settle
        for _ in range(6):
            pyboy.tick()
        
        # Get new state
        current_state = state_tracker.get_complete_state()
        
        # Calculate rewards
        reward, breakdown = state_tracker.get_reward_breakdown(previous_state, current_state)
        total_reward += reward
        
        # Show results
        print(f"  Reward: {reward:.3f}")
        if reward != -0.01:  # Show breakdown if more than just time penalty
            for category, value in breakdown.items():
                if abs(value) > 0.001:
                    print(f"    {category}: {value:+.3f}")
        
        # Show state changes
        changes = state_tracker.detect_changes(previous_state, current_state)
        active_changes = [name for name, active in changes.items() if active]
        if active_changes:
            print(f"  Changes: {', '.join(active_changes)}")
        
        # Update for next iteration
        previous_state = current_state
        
        # If we got a big reward, we found something important
        if reward > 1.0:
            print(f"  🎉 BREAKTHROUGH! Large reward detected!")
    
    print(f"\n📊 SUMMARY:")
    print(f"  Total reward: {total_reward:.3f}")
    print(f"  Average reward: {total_reward/len(actions):.3f}")
    print(f"  Visited locations: {len(state_tracker.visited_locations)}")
    print(f"  Visited maps: {list(state_tracker.visited_maps)}")
    print(f"  Is stuck: {state_tracker.is_stuck()}")
    
    pyboy.stop()
    print("\n✅ Accurate state test complete!")


if __name__ == "__main__":
    test_accurate_state()
