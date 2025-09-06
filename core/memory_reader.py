#!/usr/bin/env python3
"""Core game state management for Pokemon Crystal.

Handles memory mapping, state validation, and observation building."""

from typing import Dict, List, Optional
import logging

# Memory address mappings for Pokemon Crystal
MEMORY_ADDRESSES = {
    # Party and Pokemon data
    'party_count': 0xD163,      # Number of Pokemon in party
    'player_species': 0xD163,   # Species of first Pokemon (party slot 0 + 0)
    'player_held_item': 0xD164, # Held item of first Pokemon (party slot 0 + 1)
    'player_hp': 0xD167,        # Current HP of first Pokemon (low byte)
    'player_hp_high': 0xD168,   # Current HP of first Pokemon (high byte)
    'player_max_hp': 0xD169,    # Max HP of first Pokemon (low byte)
    'player_max_hp_high': 0xD16A, # Max HP of first Pokemon (high byte)
    'player_level': 0xD16B,     # Level of first Pokemon
    'player_status': 0xD16C,    # Status condition of first Pokemon
    
    # Location and movement
    'player_map': 0xDCBA,       # Current map ID
    'player_x': 0xDCB8,         # Player X coordinate
    'player_y': 0xDCB9,         # Player Y coordinate
    'player_direction': 0xDCBB, # Direction player is facing
    
    # Resources and progress
    'money_low': 0xD347,        # Money (3 bytes little-endian)
    'money_mid': 0xD348,
    'money_high': 0xD349,
    'badges': 0xD359,           # Badge flags
    
    # Battle state
    'in_battle': 0xD057,        # Battle active flag
    'battle_turn': 0xD068,      # Turn counter
    'enemy_species': 0xD0A5,    # Opponent Pokemon species
    'enemy_hp_low': 0xD0A8,     # Opponent HP (2 bytes)
    'enemy_hp_high': 0xD0A9,
    'enemy_level': 0xD0AA,      # Opponent level
    'player_active_slot': 0xD05E, # Active Pokemon slot
    'move_selected': 0xD05F,    # Selected move
    
    # Misc
    'step_counter': 0xD164,     # Step counter
    'game_time_hours': 0xD3E1,  # Time played
}

class GameState:
    """Manages Pokemon Crystal game state extraction and validation."""
    
    def __init__(self):
        self.logger = logging.getLogger("game_state")
        self.previous_state = {}
    
    def get_state(self, memory) -> Dict:
        """Extract complete game state from memory with validation."""
        state = {}
        
        # Get party information
        state.update(self._get_party_state(memory))
        
        # Get location and movement
        state.update(self._get_location_state(memory))
        
        # Get resources (money, badges)
        state.update(self._get_resource_state(memory))
        
        # Get battle state if in battle
        state.update(self._get_battle_state(memory))
        
        # Store for delta calculations
        self.previous_state = state
        
        return state
    
    def _get_party_state(self, memory) -> Dict:
        """Extract and validate party state."""
        try:
            party = []
            party_count = memory[MEMORY_ADDRESSES['party_count']]
            
            # Validate party count
            if not 0 <= party_count <= 6:
                self.logger.warning(f"Invalid party count {party_count}, resetting to 0")
                party_count = 0
            
            # Process each party slot
            for i in range(6):
                base = MEMORY_ADDRESSES['party_count'] + i * 44
                if i < party_count:
                    pokemon = self._read_pokemon_data(memory, base)
                else:
                    pokemon = self._empty_pokemon()
                party.append(pokemon)
            
            return {
                'party': party,
                'party_count': party_count,
                'has_pokemon': party_count > 0
            }
            
        except Exception as e:
            self.logger.error(f"Error reading party state: {e}")
            return {
                'party': [self._empty_pokemon() for _ in range(6)],
                'party_count': 0,
                'has_pokemon': False
            }
    
    def _get_location_state(self, memory) -> Dict:
        """Extract and validate location state."""
        try:
            map_id = memory[MEMORY_ADDRESSES['player_map']]
            x = memory[MEMORY_ADDRESSES['player_x']]
            y = memory[MEMORY_ADDRESSES['player_y']]
            direction = memory[MEMORY_ADDRESSES['player_direction']]
            
            # Validate coordinates
            if not all(0 <= v <= 255 for v in (map_id, x, y)):
                self.logger.warning("Invalid coordinates detected")
                map_id = x = y = 0
            
            # Validate direction
            valid_dirs = [0, 2, 4, 6, 8]  # Down, Up, Left, Right, Still
            if direction not in valid_dirs:
                direction = 0
            
            return {
                'player_map': map_id,
                'player_x': x,
                'player_y': y,
                'player_direction': direction,
                'position': {'x': x, 'y': y}
            }
            
        except Exception as e:
            self.logger.error(f"Error reading location state: {e}")
            return {
                'player_map': 0,
                'player_x': 0,
                'player_y': 0,
                'player_direction': 0,
                'position': {'x': 0, 'y': 0}
            }
    
    def _get_resource_state(self, memory) -> Dict:
        """Extract and validate resource state (money, badges)."""
        try:
            # Read money (3-byte little endian)
            money_bytes = [
                memory[MEMORY_ADDRESSES['money_low']],
                memory[MEMORY_ADDRESSES['money_mid']],
                memory[MEMORY_ADDRESSES['money_high']]
            ]
            
            # Validate and calculate money
            if all(0 <= b <= 255 for b in money_bytes):
                money = money_bytes[0] + (money_bytes[1] << 8) + (money_bytes[2] << 16)
                if money > 999999:  # Cap at reasonable maximum
                    money = 999999
            else:
                money = 0
            
            # Read badges
            badges = memory[MEMORY_ADDRESSES['badges']]
            badges_count = bin(badges).count('1')
            
            return {
                'money': money,
                'badges': badges,
                'badges_count': badges_count,
                'badges_total': badges_count
            }
            
        except Exception as e:
            self.logger.error(f"Error reading resource state: {e}")
            return {
                'money': 0,
                'badges': 0,
                'badges_count': 0,
                'badges_total': 0
            }
    
    def _get_battle_state(self, memory) -> Dict:
        """Extract and validate battle state."""
        try:
            in_battle = memory[MEMORY_ADDRESSES['in_battle']]
            
            if in_battle:
                state = {
                    'in_battle': True,
                    'battle_turn': memory[MEMORY_ADDRESSES['battle_turn']],
                    'enemy_species': memory[MEMORY_ADDRESSES['enemy_species']],
                    'enemy_level': memory[MEMORY_ADDRESSES['enemy_level']],
                    'enemy_hp': memory[MEMORY_ADDRESSES['enemy_hp_low']] + 
                               (memory[MEMORY_ADDRESSES['enemy_hp_high']] << 8),
                    'player_active_slot': memory[MEMORY_ADDRESSES['player_active_slot']],
                    'move_selected': memory[MEMORY_ADDRESSES['move_selected']]
                }
            else:
                state = {
                    'in_battle': False,
                    'battle_turn': 0,
                    'enemy_species': 0,
                    'enemy_level': 0,
                    'enemy_hp': 0,
                    'player_active_slot': 0,
                    'move_selected': 0
                }
            
            return state
            
        except Exception as e:
            self.logger.error(f"Error reading battle state: {e}")
            return {
                'in_battle': False,
                'battle_turn': 0,
                'enemy_species': 0,
                'enemy_level': 0,
                'enemy_hp': 0,
                'player_active_slot': 0,
                'move_selected': 0
            }
    
    def _read_pokemon_data(self, memory, base: int) -> Dict:
        """Read single Pokemon's data from memory with validation."""
        try:
            species = memory[base]
            held_item = memory[base + 1]
            
            # Read HP values
            hp = memory[base + 4] + (memory[base + 5] << 8)
            max_hp = memory[base + 6] + (memory[base + 7] << 8)
            
            # Validate HP
            if hp > max_hp:
                hp = max_hp
            
            # Read level and validate
            level = memory[base + 8]
            if not 0 <= level <= 100:
                level = 0
            
            # Read other stats
            status = memory[base + 9]
            moves = [memory[base + 10 + i] for i in range(4)]
            pp = [memory[base + 14 + i] for i in range(4)]
            
            return {
                'species': species,
                'held_item': held_item,
                'hp': hp,
                'max_hp': max_hp,
                'level': level,
                'status': status,
                'moves': moves,
                'pp': pp
            }
            
        except Exception as e:
            self.logger.error(f"Error reading Pokemon data at {base:X}: {e}")
            return self._empty_pokemon()
    
    def _empty_pokemon(self) -> Dict:
        """Return empty Pokemon data structure."""
        return {
            'species': 0,
            'held_item': 0,
            'hp': 0,
            'max_hp': 0,
            'level': 0,
            'status': 0,
            'moves': [0, 0, 0, 0],
            'pp': [0, 0, 0, 0]
        }
