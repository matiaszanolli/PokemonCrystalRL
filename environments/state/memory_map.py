"""
memory_map.py - Memory addresses for Pokémon Crystal game state
"""

# Memory addresses for Pokémon Crystal (Game Boy Color)
# These addresses may need to be verified/updated for your specific ROM version

MEMORY_ADDRESSES: dict[str, int] = {
    # Player position and map data
    'player_x': 0xDCB8,           # Player X coordinate
    'player_y': 0xDCB9,           # Player Y coordinate
    'player_map': 0xDCB5,         # Current map/location ID
    'player_direction': 0xDCBA,   # Direction player is facing
    
    # Player stats
    'player_hp': 0xDCDA,          # Current HP (2 bytes, little endian)
    'player_max_hp': 0xDCDB,      # Maximum HP (2 bytes, little endian)
    'player_level': 0xDCD3,       # Player's current level
    'player_exp': 0xDCD5,         # Current experience points (3 bytes)
    'player_exp_to_next': 0xDCD8, # Experience needed for next level
    
    # Party information
    'party_count': 0xDCD7,        # Number of Pokémon in party
    'party_species': 0xDCE9,      # Species of each party Pokémon (6 bytes)
    'party_hp': 0xDD0F,          # HP of each party Pokémon (12 bytes, 2 per Pokémon)
    'party_max_hp': 0xDD1B,      # Max HP of each party Pokémon (12 bytes)
    'party_level': 0xDD27,       # Level of each party Pokémon (6 bytes)
    'party_status': 0xDD33,      # Status conditions (6 bytes)
    
    # Items and inventory
    'money': 0xD84E,             # Player's money (3 bytes, BCD format)
    'bag_count': 0xD892,         # Number of items in bag
    'bag_items': 0xD893,         # Bag items (2 bytes per item: ID + quantity)
    
    # Game progress
    'badges': 0xD855,            # Badge bits (Johto badges in lower 8 bits)
    'kanto_badges': 0xD856,      # Kanto badge bits
    'elite_four_beaten': 0xD857, # Elite Four completion flag
    'champion_beaten': 0xD858,   # Champion beaten flag
    
    # Battle state
    'in_battle': 0xD057,         # 1 if in battle, 0 otherwise
    'battle_type': 0xD058,       # Type of battle (wild, trainer, etc.)
    'enemy_hp': 0xCFE6,         # Enemy Pokémon current HP
    'enemy_max_hp': 0xCFE8,     # Enemy Pokémon max HP
    'enemy_level': 0xCFE3,      # Enemy Pokémon level
    'enemy_species': 0xCFE0,    # Enemy Pokémon species
    
    # Menu and UI state
    'menu_state': 0xD0A0,       # Current menu state
    'text_box_state': 0xD0A1,   # Text box display state
    'overworld_state': 0xD0A2,  # Overworld interaction state
    
    # Time and day/night
    'time_of_day': 0xD269,      # 1=morning, 2=day, 3=evening, 4=night
    'day_of_week': 0xD26A,      # Day of the week
    
    # Important flags and events
    'rival_name': 0xD2B7,       # Rival's name (10 bytes)
    'player_name': 0xD47D,      # Player's name (10 bytes)
    
    # PC and storage
    'pc_box_count': 0xDA80,     # Number of Pokémon in current PC box
    
    # Audio and graphics
    'music_id': 0xD0C0,         # Current music track ID
    'sound_id': 0xD0C1,         # Current sound effect ID
    
    # RNG and luck factors
    'rng_seed': 0xD26B,         # Random number generator seed
    
    # Movement and interaction
    'can_move': 0xD0B0,         # Whether player can move
    'surf_state': 0xD0B1,       # Whether player is surfing
    'bike_state': 0xD0B2,       # Whether player is on bike
    
    # Special events and cutscenes
    'cutscene_flag': 0xD0C5,    # Cutscene or special event flag
    
    # Pokémon Center and healing
    'last_pokecenter': 0xD2A0,  # Last Pokémon Center visited
}

# Additional derived values that can be calculated
DERIVED_VALUES: dict[str, callable] = {
    'hp_percentage': lambda state: (
        state.get('player_hp', 0) / max(state.get('player_max_hp', 1), 1)
    ),
    'party_alive_count': lambda state: sum(
        1 for i in range(state.get('party_count', 0))
        if state.get(f'party_hp_{i}', 0) > 0
    ),
    'badges_total': lambda state: (
        # Early game protection - if no party/no level, consider badges invalid
        0 if (state.get('party_count', 0) == 0 or state.get('player_level', 0) == 0) else
        # Impossible level protection (>100)
        0 if state.get('player_level', 0) > 100 else
        # Count valid badges while handling potential corruption
        bin((
            # Only count if values are reasonable (no uninitialized/corrupted values)
            (state.get('badges', 0) if 0 <= state.get('badges', 0) <= 0xFF else 0) |
            ((state.get('kanto_badges', 0) if 0 <= state.get('kanto_badges', 0) <= 0xFF else 0) << 8)
        )).count('1')
    ),
    'is_healthy': lambda state: state.get('player_hp', 0) > (
        state.get('player_max_hp', 1) * 0.5
    ),
}

# Map IDs for important locations (these may need verification)
IMPORTANT_LOCATIONS: dict[str, int] = {
    'NEW_BARK_TOWN': 1,
    'ROUTE_29': 2,
    'CHERRYGROVE_CITY': 3,
    'ROUTE_30': 4,
    'ROUTE_31': 5,
    'VIOLET_CITY': 6,
    'SPROUT_TOWER': 7,
    'ROUTE_32': 8,
    'RUINS_OF_ALPH': 9,
    'UNION_CAVE': 10,
    'ROUTE_33': 11,
    'AZALEA_TOWN': 12,
    'SLOWPOKE_WELL': 13,
    'ILEX_FOREST': 14,
    'ROUTE_34': 15,
    'GOLDENROD_CITY': 16,
    'NATIONAL_PARK': 17,
    'ROUTE_35': 18,
    'ROUTE_36': 19,
    'ROUTE_37': 20,
    'ECRUTEAK_CITY': 21,
    # Add more locations as needed
}

# Pokémon species IDs (first generation + some Johto)
POKEMON_SPECIES: dict[str, int] = {
    'BULBASAUR': 1,
    'IVYSAUR': 2,
    'VENUSAUR': 3,
    'CHARMANDER': 4,
    'CHARMELEON': 5,
    'CHARIZARD': 6,
    'SQUIRTLE': 7,
    'WARTORTLE': 8,
    'BLASTOISE': 9,
    'CATERPIE': 10,
    # ... add more as needed
    'CHIKORITA': 152,
    'BAYLEEF': 153,
    'MEGANIUM': 154,
    'CYNDAQUIL': 155,
    'QUILAVA': 156,
    'TYPHLOSION': 157,
    'TOTODILE': 158,
    'CROCONAW': 159,
    'FERALIGATR': 160,
    # Add more Johto Pokémon as needed
}

# Status condition flags
STATUS_CONDITIONS: dict[str, int] = {
    'NONE': 0x00,
    'SLEEP': 0x01,
    'POISON': 0x02,
    'BURN': 0x04,
    'FREEZE': 0x08,
    'PARALYSIS': 0x10,
    'TOXIC': 0x20,
}

# Badge bit masks
BADGE_MASKS: dict[str, int] = {
    # Johto badges
    'ZEPHYR': 0x01,
    'HIVE': 0x02,
    'PLAIN': 0x04,
    'FOG': 0x08,
    'STORM': 0x10,
    'MINERAL': 0x20,
    'GLACIER': 0x40,
    'RISING': 0x80,
    
    # Kanto badges (in kanto_badges byte)
    'BOULDER': 0x01,
    'CASCADE': 0x02,
    'THUNDER': 0x04,
    'RAINBOW': 0x08,
    'SOUL': 0x10,
    'MARSH': 0x20,
    'VOLCANO': 0x40,
    'EARTH': 0x80,
}

def get_badges_earned(badges_byte: int, kanto_badges_byte: int = 0) -> list:
    """
    Get list of badges earned based on badge bytes
    
    Args:
        badges_byte: Johto badges byte
        kanto_badges_byte: Kanto badges byte
    
    Returns:
        List of badge names earned
    """
    earned_badges = []
    
    # Check Johto badges
    for badge_name, mask in BADGE_MASKS.items():
        if badge_name in ['BOULDER', 'CASCADE', 'THUNDER', 'RAINBOW', 
                         'SOUL', 'MARSH', 'VOLCANO', 'EARTH']:
            continue  # Skip Kanto badges in this loop
        
        if badges_byte & mask:
            earned_badges.append(badge_name)
    
    # Check Kanto badges
    for badge_name, mask in BADGE_MASKS.items():
        if badge_name not in ['BOULDER', 'CASCADE', 'THUNDER', 'RAINBOW', 
                             'SOUL', 'MARSH', 'VOLCANO', 'EARTH']:
            continue  # Skip Johto badges in this loop
        
        if kanto_badges_byte & mask:
            earned_badges.append(badge_name)
    
    return earned_badges


def validate_memory_addresses():
    """
    Validate that memory addresses are reasonable for Game Boy Color
    """
    for name, addr in MEMORY_ADDRESSES.items():
        if not (0x8000 <= addr <= 0xFFFF):
            print(f"Warning: {name} address {hex(addr)} may be invalid for GBC")


if __name__ == "__main__":
    validate_memory_addresses()
    print(f"Loaded {len(MEMORY_ADDRESSES)} memory addresses")
    print(f"Loaded {len(IMPORTANT_LOCATIONS)} important locations")
    print(f"Loaded {len(BADGE_MASKS)} badge definitions")
