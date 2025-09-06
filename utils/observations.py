"""
Pokemon Crystal Game State Observation Builder

Contains functions for building comprehensive game state observations
from memory addresses and Pokemon Crystal ROM data.
"""

from typing import Dict, List
from config.addresses import (
    MEMORY_ADDRESSES, 
    PARTY_SIZE, 
    PARTY_BYTES_PER_POKEMON, 
    PARTY_BASE_ADDRESS,
    MOVES_PER_POKEMON,
    PP_PER_POKEMON
)


def build_observation(memory) -> Dict:
    """Build complete game state observation using validated memory addresses
    
    Args:
        memory: PyBoy memory interface for reading game state
        
    Returns:
        Dict containing comprehensive game state information including:
        - Party Pokemon data (species, HP, level, moves, etc.)
        - Location and coordinates
        - Money and badges
        - Battle state
        - Derived gameplay values
    """
    
    # Party data - using validated party structure (44 bytes per Pokemon)
    party = []
    party_count = memory[MEMORY_ADDRESSES['party_count']] if memory[MEMORY_ADDRESSES['party_count']] <= 6 else 0  # Validate party count
    
    for i in range(PARTY_SIZE):  # Always check all 6 slots
        base = PARTY_BASE_ADDRESS + i * PARTY_BYTES_PER_POKEMON
        try:
            # PyBoy memory access doesn't support len(), so we'll use try/except for bounds checking

            species = memory[base] if i < party_count else 0
            held_item = memory[base + 1] if i < party_count else 0
            hp = memory[base + 4] + (memory[base + 5] << 8) if i < party_count else 0
            max_hp = memory[base + 6] + (memory[base + 7] << 8) if i < party_count else 0
            level = memory[base + 8] if i < party_count else 0
            status = memory[base + 9] if i < party_count else 0
            moves = [memory[base + 10 + j] for j in range(MOVES_PER_POKEMON)] if i < party_count else [0, 0, 0, 0]
            pp = [memory[base + 14 + j] for j in range(PP_PER_POKEMON)] if i < party_count else [0, 0, 0, 0]

            # Validate critical values
            if i < party_count:
                # Pokemon level should be between 1-100
                if not 0 <= level <= 100:
                    print(f"Warning: Invalid level {level} for Pokemon {i}, resetting to 0")
                    level = 0
                # HP should never be more than max HP
                if hp > max_hp:
                    print(f"Warning: HP {hp} exceeds max HP {max_hp} for Pokemon {i}, capping")
                    hp = max_hp
            
            party.append({
                "species": species,
                "held_item": held_item,
                "hp": hp,
                "max_hp": max_hp,
                "level": level,
                "status": status,
                "moves": moves,
                "pp": pp
            })
        except (IndexError, KeyError) as e:
            print(f"Error reading Pokemon {i} data: {str(e)}")
            party.append({
                "species": 0, "held_item": 0, "hp": 0, "max_hp": 0,
                "level": 0, "status": 0, "moves": [0, 0, 0, 0], "pp": [0, 0, 0, 0]
            })
        except Exception as e:
            print(f"Unexpected error reading Pokemon {i} data: {str(e)}")
            party.append({
                "species": 0, "held_item": 0, "hp": 0, "max_hp": 0,
                "level": 0, "status": 0, "moves": [0, 0, 0, 0], "pp": [0, 0, 0, 0]
            })
    
    # Money - using validated 3-byte little-endian structure
    money = _read_money(memory)
    
    # Location and coordinates - using VERIFIED addresses
    location_data = _read_location_data(memory)
    
    # Battle state - using validated battle structure
    battle_data = _read_battle_data(memory)
    
    # Badge and progression - using validated badge structure
    badges_data = _read_badges_data(memory)
    
    # Step counter for exploration
    step_counter = _read_step_counter(memory)
    
    # Compile complete state
    observation = {
        "party": party,
        "party_count": party_count,
        "money": money,
        "badges": badges_data["badges"],
        "badges_count": badges_data["badges_count"],
        "badges_total": badges_data["badges_count"],  # For compatibility
        "map_id": location_data["map_id"],
        "player_map": location_data["map_id"],  # For compatibility
        "coords": (location_data["player_x"], location_data["player_y"]),
        "player_x": location_data["player_x"],
        "player_y": location_data["player_y"],
        "facing": location_data["facing"],
        "player_direction": location_data["facing"],
        "in_battle": battle_data["in_battle"],
        "battle_turn": battle_data["battle_turn"],
        "enemy_species": battle_data["enemy_species"],
        "enemy_hp": battle_data["enemy_hp"],
        "enemy_level": battle_data["enemy_level"],
        "step_counter": step_counter,
        
        # Derived values for first Pokemon (main player stats)
        "player_species": party[0]["species"] if party_count > 0 else 0,
        "player_hp": party[0]["hp"] if party_count > 0 else 0,
        "player_max_hp": party[0]["max_hp"] if party_count > 0 else 0,
        "player_level": party[0]["level"] if party_count > 0 else 0,
        "player_status": party[0]["status"] if party_count > 0 else 0,
        "has_pokemon": party_count > 0,
        "health_percentage": (party[0]["hp"] / max(party[0]["max_hp"], 1)) * 100 if party_count > 0 else 0,
    }
    
    return observation


def _read_money(memory) -> int:
    """Read money value from memory using 3-byte little-endian format"""
    try:
        # Validate individual bytes are within valid range (0-255)
        for addr in [MEMORY_ADDRESSES['money_low'], MEMORY_ADDRESSES['money_mid'], MEMORY_ADDRESSES['money_high']]:
            if not 0 <= memory[addr] <= 255:
                raise ValueError(f"Invalid money byte value at {addr:X}: {memory[addr]}")
        
        # Calculate money using little-endian bytes
        money = (memory[MEMORY_ADDRESSES['money_low']] + 
                (memory[MEMORY_ADDRESSES['money_mid']] << 8) + 
                (memory[MEMORY_ADDRESSES['money_high']] << 16))
        
        # Validate final money value (reasonable max of 999,999)
        if money > 999999:
            print(f"Warning: Unusually high money value {money}, resetting to 0")
            money = 0
            
        return money
            
    except (IndexError, KeyError) as e:
        print(f"Error reading money data: {str(e)}")
        return 0
    except ValueError as e:
        print(f"Invalid money value: {str(e)}")
        return 0
    except Exception as e:
        print(f"Unexpected error reading money: {str(e)}")
        return 0


def _read_location_data(memory) -> Dict:
    """Read player location and coordinates from memory"""
    try:
        # Read location data
        map_id = memory[MEMORY_ADDRESSES['player_map']]
        player_x = memory[MEMORY_ADDRESSES['player_x']]
        player_y = memory[MEMORY_ADDRESSES['player_y']] 
        facing = memory[MEMORY_ADDRESSES['player_direction']]
        
        # Validate location data is within reasonable ranges
        if not 0 <= map_id <= 255:
            print(f"Warning: Invalid map ID {map_id}, resetting to 0")
            map_id = 0
        if not 0 <= player_x <= 255:
            print(f"Warning: Invalid X coordinate {player_x}, resetting to 0")
            player_x = 0
        if not 0 <= player_y <= 255:
            print(f"Warning: Invalid Y coordinate {player_y}, resetting to 0")
            player_y = 0
        # Game Boy direction encoding: 0=Down, 2=Up, 4=Left, 6=Right, 8=Standing still
        # Also allow some other values that might occur during transitions
        valid_directions = [0, 2, 4, 6, 8]  # Common Game Boy direction values
        if facing not in valid_directions:
            print(f"Warning: Invalid direction {facing}, resetting to 0")
            facing = 0
        
        return {
            "map_id": map_id,
            "player_x": player_x,
            "player_y": player_y,
            "facing": facing
        }
            
    except (IndexError, KeyError) as e:
        print(f"Error reading location data: {str(e)}")
        return {"map_id": 0, "player_x": 0, "player_y": 0, "facing": 0}
    except Exception as e:
        print(f"Unexpected error reading location data: {str(e)}")
        return {"map_id": 0, "player_x": 0, "player_y": 0, "facing": 0}


def _read_battle_data(memory) -> Dict:
    """Read battle state information from memory"""
    try:
        battle_flag = memory[MEMORY_ADDRESSES['in_battle']]
        turn_count = memory[MEMORY_ADDRESSES['battle_turn']] if battle_flag else 0
        enemy_species = memory[MEMORY_ADDRESSES['enemy_species']] if battle_flag else 0
        enemy_hp = (memory[MEMORY_ADDRESSES['enemy_hp_low']] + 
                   (memory[MEMORY_ADDRESSES['enemy_hp_high']] << 8)) if battle_flag else 0
        enemy_level = memory[MEMORY_ADDRESSES['enemy_level']] if battle_flag else 0
        
        return {
            "in_battle": bool(battle_flag),
            "battle_turn": turn_count,
            "enemy_species": enemy_species,
            "enemy_hp": enemy_hp,
            "enemy_level": enemy_level
        }
    except:
        return {
            "in_battle": False,
            "battle_turn": 0,
            "enemy_species": 0,
            "enemy_hp": 0,
            "enemy_level": 0
        }


def _read_badges_data(memory) -> Dict:
    """Read badge progression data from memory"""
    try:
        badges = memory[MEMORY_ADDRESSES['badges']]
        badges_count = bin(badges).count('1')
        
        return {
            "badges": badges,
            "badges_count": badges_count
        }
    except:
        return {
            "badges": 0,
            "badges_count": 0
        }


def _read_step_counter(memory) -> int:
    """Read step counter for exploration tracking"""
    try:
        return memory[MEMORY_ADDRESSES['step_counter']]
    except:
        return 0