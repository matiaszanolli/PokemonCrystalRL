"""
Pokemon Crystal Memory Reader Utilities

Safe utilities for reading and validating Game Boy memory values.
"""

from typing import Dict, Optional, List, Tuple, Any
import logging
from config.memory_addresses import MEMORY_ADDRESSES
from config.constants import (
    TRAINING_PARAMS,
    POKEMON_SPECIES,
    STATUS_CONDITIONS,
)

logger = logging.getLogger(__name__)

def get_safe_memory(memory, address: int, default: int = 0) -> int:
    """Safely read memory address with bounds checking and validation."""
    try:
        value = memory[address]
        if not isinstance(value, int) or not 0 <= value <= 255:
            logger.warning(f"Invalid memory value at {address:X}: {value}")
            return default
        return value
    except (IndexError, KeyError) as e:
        logger.warning(f"Error reading memory at {address:X}: {str(e)}")
        return default
    except Exception as e:
        logger.error(f"Unexpected error reading memory at {address:X}: {str(e)}")
        return default

def read_party_pokemon(memory, slot: int) -> Dict[str, Any]:
    """Read single Pokemon data from party slot with validation."""
    if not 0 <= slot < TRAINING_PARAMS['MAX_PARTY_SIZE']:
        return {
            "species": 0, "held_item": 0, "hp": 0, "max_hp": 0,
            "level": 0, "status": 0, "moves": [0, 0, 0, 0], "pp": [0, 0, 0, 0]
        }

    try:
        base = MEMORY_ADDRESSES['party_count'] + slot * TRAINING_PARAMS['PARTY_SLOT_SIZE']
        
        species = get_safe_memory(memory, base)
        if species not in POKEMON_SPECIES:
            logger.warning(f"Invalid Pokemon species {species} in slot {slot}")
            species = 0
            
        level = get_safe_memory(memory, base + 8)
        if not 0 <= level <= TRAINING_PARAMS['MAX_LEVEL']:
            logger.warning(f"Invalid level {level} for Pokemon {slot}")
            level = 0
            
        hp = get_safe_memory(memory, base + 4) + (get_safe_memory(memory, base + 5) << 8)
        max_hp = get_safe_memory(memory, base + 6) + (get_safe_memory(memory, base + 7) << 8)
        
        if hp > max_hp:
            logger.warning(f"HP {hp} exceeds max HP {max_hp} for Pokemon {slot}")
            hp = max_hp
            
        status = get_safe_memory(memory, base + 9)
        if status not in STATUS_CONDITIONS:
            logger.warning(f"Invalid status {status} for Pokemon {slot}")
            status = 0
            
        moves = [get_safe_memory(memory, base + 10 + i) for i in range(4)]
        pp = [get_safe_memory(memory, base + 14 + i) for i in range(4)]
        
        return {
            "species": species,
            "held_item": get_safe_memory(memory, base + 1),
            "hp": hp,
            "max_hp": max_hp,
            "level": level,
            "status": status,
            "moves": moves,
            "pp": pp
        }
        
    except Exception as e:
        logger.error(f"Error reading Pokemon {slot} data: {str(e)}")
        return {
            "species": 0, "held_item": 0, "hp": 0, "max_hp": 0,
            "level": 0, "status": 0, "moves": [0, 0, 0, 0], "pp": [0, 0, 0, 0]
        }

def read_money(memory) -> int:
    """Read money value from memory (3-byte little-endian) with validation."""
    try:
        # Read individual bytes
        low = get_safe_memory(memory, MEMORY_ADDRESSES['money_low'])
        mid = get_safe_memory(memory, MEMORY_ADDRESSES['money_mid'])
        high = get_safe_memory(memory, MEMORY_ADDRESSES['money_high'])
        
        # Calculate total
        money = low + (mid << 8) + (high << 16)
        
        # Validate reasonable range
        if money > TRAINING_PARAMS['MAX_MONEY']:
            logger.warning(f"Unusually high money value {money}, resetting to 0")
            return 0
            
        return money
        
    except Exception as e:
        logger.error(f"Error reading money value: {str(e)}")
        return 0

def read_location(memory) -> Tuple[int, int, int, int]:
    """Read and validate player location data."""
    try:
        map_id = get_safe_memory(memory, MEMORY_ADDRESSES['player_map'])
        x = get_safe_memory(memory, MEMORY_ADDRESSES['player_x'])
        y = get_safe_memory(memory, MEMORY_ADDRESSES['player_y'])
        facing = get_safe_memory(memory, MEMORY_ADDRESSES['player_direction'])
        
        # Validate coordinates are in valid range
        if not 0 <= x <= 255:
            logger.warning(f"Invalid X coordinate {x}, resetting to 0")
            x = 0
        if not 0 <= y <= 255:
            logger.warning(f"Invalid Y coordinate {y}, resetting to 0")
            y = 0
            
        # Validate map ID and facing direction
        if not 0 <= map_id <= 255:
            logger.warning(f"Invalid map ID {map_id}, resetting to 0")
            map_id = 0
        if facing not in [0, 2, 4, 6, 8]:  # Valid GB directions
            logger.warning(f"Invalid direction {facing}, resetting to 0")
            facing = 0
            
        return map_id, x, y, facing
        
    except Exception as e:
        logger.error(f"Error reading location data: {str(e)}")
        return 0, 0, 0, 0

def build_observation(memory) -> Dict[str, Any]:
    """Build complete game state observation with validated memory values."""
    try:
        # Get party Pokemon data
        party = []
        party_count = get_safe_memory(memory, MEMORY_ADDRESSES['party_count'])
        if party_count > TRAINING_PARAMS['MAX_PARTY_SIZE']:
            logger.warning(f"Invalid party count {party_count}, resetting to 0")
            party_count = 0
            
        for i in range(TRAINING_PARAMS['MAX_PARTY_SIZE']):
            pokemon = read_party_pokemon(memory, i)
            party.append(pokemon)
            
        # Get money
        money = read_money(memory)
            
        # Get badges
        badges = get_safe_memory(memory, MEMORY_ADDRESSES['badges'])
        badges_count = bin(badges).count('1')
        
        # Get location
        map_id, player_x, player_y, facing = read_location(memory)
        
        # Get battle state
        battle_flag = get_safe_memory(memory, MEMORY_ADDRESSES['in_battle'])
        turn_count = get_safe_memory(memory, MEMORY_ADDRESSES['battle_turn']) if battle_flag else 0
        enemy_species = get_safe_memory(memory, MEMORY_ADDRESSES['enemy_species']) if battle_flag else 0
        enemy_hp = (get_safe_memory(memory, MEMORY_ADDRESSES['enemy_hp_low']) + 
                   (get_safe_memory(memory, MEMORY_ADDRESSES['enemy_hp_high']) << 8)) if battle_flag else 0
        enemy_level = get_safe_memory(memory, MEMORY_ADDRESSES['enemy_level']) if battle_flag else 0
        
        # Get misc
        step_counter = get_safe_memory(memory, MEMORY_ADDRESSES['step_counter'])
        
        # Compile complete state
        state = {
            "party": party,
            "party_count": party_count,
            "money": money,
            "badges": badges,
            "badges_count": badges_count,
            "badges_total": badges_count,  # For compatibility
            "map_id": map_id,
            "player_map": map_id,  # For compatibility
            "coords": (player_x, player_y),
            "player_x": player_x,
            "player_y": player_y,
            "facing": facing,
            "player_direction": facing,
            "in_battle": bool(battle_flag),
            "battle_turn": turn_count,
            "enemy_species": enemy_species,
            "enemy_hp": enemy_hp,
            "enemy_level": enemy_level,
            "step_counter": step_counter,
        }
        
        # Add derived player stats if we have Pokemon
        if party_count > 0:
            state.update({
                "player_species": party[0]["species"],
                "player_hp": party[0]["hp"],
                "player_max_hp": party[0]["max_hp"],
                "player_level": party[0]["level"],
                "player_status": party[0]["status"],
                "has_pokemon": True,
                "health_percentage": (party[0]["hp"] / max(party[0]["max_hp"], 1)) * 100,
            })
        else:
            state.update({
                "player_species": 0,
                "player_hp": 0,
                "player_max_hp": 0,
                "player_level": 0,
                "player_status": 0,
                "has_pokemon": False,
                "health_percentage": 0,
            })
            
        return state
        
    except Exception as e:
        logger.error(f"Error building observation: {str(e)}")
        # Return minimal valid state
        return {
            "party": [], "party_count": 0, "money": 0,
            "badges": 0, "badges_count": 0, "badges_total": 0,
            "map_id": 0, "player_map": 0, "coords": (0, 0),
            "player_x": 0, "player_y": 0, "facing": 0,
            "player_direction": 0, "in_battle": False,
            "battle_turn": 0, "enemy_species": 0, "enemy_hp": 0,
            "enemy_level": 0, "step_counter": 0,
            "player_species": 0, "player_hp": 0, "player_max_hp": 0,
            "player_level": 0, "player_status": 0,
            "has_pokemon": False, "health_percentage": 0,
        }
