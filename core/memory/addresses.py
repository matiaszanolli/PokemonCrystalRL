"""
Consolidated Memory Map for Pokemon Crystal
==========================================

This file contains all memory addresses from both memory_map.py and memory_map_new.py
with clear documentation about conflicts and different ROM versions.

IMPORTANT: Different Pokemon Crystal ROM versions may have different memory layouts.
When there are conflicts, both addresses are preserved with clear comments about their source.
"""

# Memory addresses for Pokémon Crystal (Game Boy Color)
# Consolidated from memory_map.py and memory_map_new.py

MEMORY_ADDRESSES = {
    # ===== PLAYER POSITION AND MAP DATA =====
    # NOTE: Conflicts exist - test both addresses for your ROM version
    
    'player_x': 0xDCB8,                    # Player X coordinate (from original memory_map.py - VERIFIED)
    'player_y': 0xDCB9,                    # Player Y coordinate (from original memory_map.py - VERIFIED)
    
    # CONFLICT: Different map address in each file
    'player_map': 0xDCBA,                  # Current map/location ID (from original - updated from test)
    'player_map_alt': 0xDCB5,              # Alternative map location (from memory_map_new.py)
    
    # CONFLICT: Different direction address
    'player_direction': 0xDCBB,            # Direction player is facing (from original - updated)
    'player_direction_alt': 0xDCBA,        # Alternative direction location (from memory_map_new.py)
    
    # Additional coordinate locations found in memory_map_new.py analysis
    'alt_x': 0xD4BC,                       # Potential X coordinate (found value 29)
    'alt_y': 0xD4BD,                       # Potential Y coordinate (found value 10)
    
    # ===== PLAYER STATS =====
    # CONFLICT: Completely different addresses for player stats
    
    # Original addresses (memory_map.py)
    'player_hp': 0xDCDA,                   # Current HP (2 bytes, little endian) - ORIGINAL
    'player_max_hp': 0xDCDB,               # Maximum HP (2 bytes, little endian) - ORIGINAL  
    'player_level': 0xDCD3,                # Player's current level - ORIGINAL
    'player_exp': 0xDCD5,                  # Current experience points (3 bytes) - ORIGINAL
    'player_exp_to_next': 0xDCD8,          # Experience needed for next level - ORIGINAL
    
    # Alternative addresses (memory_map_new.py - based on ROM analysis)
    'player_hp_alt': 0xD16C,               # Current HP (2 bytes, little endian) - NEW
    'player_max_hp_alt': 0xD16E,           # Maximum HP (2 bytes, little endian) - NEW
    'player_level_alt': 0xD172,            # Player's current level - NEW  
    'player_exp_alt': 0xD179,              # Current experience points (3 bytes) - NEW
    'player_exp_to_next_alt': 0xD17C,      # Experience needed for next level - NEW
    
    # ===== PARTY INFORMATION =====
    # CONFLICT: Different base addresses for party data
    
    # Original addresses (memory_map.py)
    'party_count': 0xDCD7,                 # Number of Pokémon in party - ORIGINAL
    'party_species': 0xDCE9,               # Species of each party Pokémon (6 bytes) - ORIGINAL
    'party_hp': 0xDD0F,                    # HP of each party Pokémon (12 bytes, 2 per Pokémon) - ORIGINAL
    'party_max_hp': 0xDD1B,                # Max HP of each party Pokémon (12 bytes) - ORIGINAL
    'party_level': 0xDD27,                 # Level of each party Pokémon (6 bytes) - ORIGINAL
    'party_status': 0xDD33,                # Status conditions (6 bytes) - ORIGINAL
    
    # Alternative addresses (memory_map_new.py)
    'party_count_alt': 0xD163,             # Number of Pokémon in party - NEW
    'party_species_alt': 0xD164,           # Species of each party Pokémon (6 bytes) - NEW
    'party_hp_alt': 0xD16C,                # HP of each party Pokémon (12 bytes, 2 per Pokémon) - NEW
    'party_max_hp_alt': 0xD178,            # Max HP of each party Pokémon (12 bytes) - NEW
    'party_level_alt': 0xD184,             # Level of each party Pokémon (6 bytes) - NEW
    'party_status_alt': 0xD18A,            # Status conditions (6 bytes) - NEW
    
    # ===== ITEMS AND INVENTORY =====
    # CONFLICT: Different money address
    
    'money': 0xD84E,                       # Player's money (3 bytes, BCD format) - ORIGINAL
    'money_alt': 0xD84A,                   # Player's money - FOUND IN ANALYSIS (memory_map_new.py)
    'bag_count': 0xD892,                   # Number of items in bag
    'bag_items': 0xD893,                   # Bag items (2 bytes per item: ID + quantity)
    
    # ===== GAME PROGRESS =====
    # CONFLICT: Different badge addresses
    
    # Original addresses
    'badges': 0xD855,                      # Badge bits (Johto badges in lower 8 bits) - ORIGINAL
    'kanto_badges': 0xD856,                # Kanto badge bits - ORIGINAL
    'elite_four_beaten': 0xD857,           # Elite Four completion flag - ORIGINAL
    'champion_beaten': 0xD858,             # Champion beaten flag - ORIGINAL
    
    # Alternative addresses  
    'badges_alt': 0xD857,                  # Badge bits (Johto badges in lower 8 bits) - NEW
    'kanto_badges_alt': 0xD858,            # Kanto badge bits - NEW
    'elite_four_beaten_alt': 0xD864,       # Elite Four completion flag - NEW
    'champion_beaten_alt': 0xD865,         # Champion beaten flag - NEW
    
    # ===== BATTLE STATE =====
    # These appear consistent between both files
    
    'in_battle': 0xD057,                   # 1 if in battle, 0 otherwise
    'battle_type': 0xD058,                 # Type of battle (wild, trainer, etc.)
    'enemy_hp': 0xCFE6,                    # Enemy Pokémon current HP
    'enemy_max_hp': 0xCFE8,                # Enemy Pokémon max HP
    'enemy_level': 0xCFE3,                 # Enemy Pokémon level
    'enemy_species': 0xCFE0,               # Enemy Pokémon species
    
    # ===== MENU AND UI STATE =====
    # These appear consistent between both files
    
    'menu_state': 0xD0A0,                  # Current menu state
    'text_box_state': 0xD0A1,              # Text box display state
    'overworld_state': 0xD0A2,             # Overworld interaction state
    
    # ===== TIME AND DAY/NIGHT =====
    # These appear consistent between both files
    
    'time_of_day': 0xD269,                 # 1=morning, 2=day, 3=evening, 4=night - Found value!
    'day_of_week': 0xD26A,                 # Day of the week
    
    # ===== IMPORTANT FLAGS AND EVENTS =====
    # These appear consistent between both files
    
    'rival_name': 0xD2B7,                  # Rival's name (10 bytes)
    'player_name': 0xD47D,                 # Player's name (10 bytes)
    
    # ===== PC AND STORAGE =====
    # These appear consistent between both files
    
    'pc_box_count': 0xDA80,                # Number of Pokémon in current PC box
    
    # ===== AUDIO AND GRAPHICS =====
    # These appear consistent between both files
    
    'music_id': 0xD0C0,                    # Current music track ID
    'sound_id': 0xD0C1,                    # Current sound effect ID
    
    # ===== RNG AND LUCK FACTORS =====
    # These appear consistent between both files
    
    'rng_seed': 0xD26B,                    # Random number generator seed
    
    # ===== MOVEMENT AND INTERACTION =====
    # These appear consistent between both files
    
    'can_move': 0xD0B0,                    # Whether player can move
    'surf_state': 0xD0B1,                  # Whether player is surfing
    'bike_state': 0xD0B2,                  # Whether player is on bike
    
    # ===== SPECIAL EVENTS AND CUTSCENES =====
    # These appear consistent between both files
    
    'cutscene_flag': 0xD0C5,               # Cutscene or special event flag
    
    # ===== POKÉMON CENTER AND HEALING =====
    # These appear consistent between both files
    
    'last_pokecenter': 0xD2A0,             # Last Pokémon Center visited
}

# Address groups for easier testing and validation
ADDRESS_GROUPS = {
    'player_position': {
        'primary': ['player_x', 'player_y', 'player_map', 'player_direction'],
        'alternative': ['alt_x', 'alt_y', 'player_map_alt', 'player_direction_alt'],
    },
    'player_stats': {
        'primary': ['player_hp', 'player_max_hp', 'player_level', 'player_exp', 'player_exp_to_next'],
        'alternative': ['player_hp_alt', 'player_max_hp_alt', 'player_level_alt', 'player_exp_alt', 'player_exp_to_next_alt'],
    },
    'party_info': {
        'primary': ['party_count', 'party_species', 'party_hp', 'party_max_hp', 'party_level', 'party_status'],
        'alternative': ['party_count_alt', 'party_species_alt', 'party_hp_alt', 'party_max_hp_alt', 'party_level_alt', 'party_status_alt'],
    },
    'money': {
        'primary': ['money'],
        'alternative': ['money_alt'],
    },
    'badges': {
        'primary': ['badges', 'kanto_badges', 'elite_four_beaten', 'champion_beaten'],
        'alternative': ['badges_alt', 'kanto_badges_alt', 'elite_four_beaten_alt', 'champion_beaten_alt'],
    }
}

def _safe_badges_total(state):
    """
    Safely compute total badges, filtering uninitialized memory spikes.
    Treat invalid bytes as uninitialized if conditions suggest memory corruption.
    
    This is the enhanced version from memory_map.py with better corruption detection.
    """
    # Try primary badge addresses first
    johto = state.get('badges', 0)
    kanto = state.get('kanto_badges', 0)
    
    # If primary addresses seem invalid, try alternatives
    if johto == 0xFF or kanto == 0xFF:
        johto = state.get('badges_alt', johto)
        kanto = state.get('kanto_badges_alt', kanto)
    
    party_count = state.get('party_count', state.get('party_count_alt', 0))
    player_level = state.get('player_level', state.get('player_level_alt', 0))

    # Impossible level always indicates corruption, regardless of game state
    if player_level > 100:
        return 0

    # Early game indicators - EITHER condition suggests early game
    early_game = (party_count == 0 or player_level == 0)
    
    # Invalid badge values (0xFF, or values > 0x80 which is only the last badge)
    invalid_badges = (
        johto == 0xFF or kanto == 0xFF or 
        johto > 0x80 or kanto > 0x80
    )
    
    # If early game indicators with invalid badges, treat as corruption
    if early_game and invalid_badges:
        return 0
    
    # Sanity bounds: max 8 badges each region = 16 total
    total = 0
    jb = johto & 0xFF
    kb = kanto & 0xFF
    
    # Count bits explicitly to avoid surprises
    for i in range(8):
        total += (jb >> i) & 1
    for i in range(8):
        total += (kb >> i) & 1
    
    # Cap at maximum possible badges
    return min(total, 16)

def _simple_badges_total(state):
    """
    Simple badge calculation from memory_map_new.py - less robust but faster.
    """
    badges = state.get('badges', state.get('badges_alt', 0))
    kanto_badges = state.get('kanto_badges', state.get('kanto_badges_alt', 0))
    return bin(badges | (kanto_badges << 8)).count('1')

# Additional derived values that can be calculated
DERIVED_VALUES = {
    'hp_percentage': lambda state: (
        state.get('player_hp', state.get('player_hp_alt', 0)) / 
        max(state.get('player_max_hp', state.get('player_max_hp_alt', 1)), 1)
    ),
    'party_alive_count': lambda state: sum(
        1 for i in range(state.get('party_count', state.get('party_count_alt', 0)))
        if state.get(f'party_hp_{i}', 0) > 0
    ),
    'badges_total': _safe_badges_total,          # Use the robust version by default
    'badges_total_simple': _simple_badges_total,  # Alternative simple version
    'is_healthy': lambda state: (
        state.get('player_hp', state.get('player_hp_alt', 0)) > 
        (state.get('player_max_hp', state.get('player_max_hp_alt', 1)) * 0.5)
    ),
}

# Map IDs for important locations (consistent in both files)
IMPORTANT_LOCATIONS = {
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

# Pokémon species IDs (first generation + some Johto) - consistent in both files
POKEMON_SPECIES = {
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

# Status condition flags - consistent in both files
STATUS_CONDITIONS = {
    'NONE': 0x00,
    'SLEEP': 0x01,
    'POISON': 0x02,
    'BURN': 0x04,
    'FREEZE': 0x08,
    'PARALYSIS': 0x10,
    'TOXIC': 0x20,
}

# Badge bit masks - consistent in both files
BADGE_MASKS = {
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
    issues = []
    for name, addr in MEMORY_ADDRESSES.items():
        if not (0x8000 <= addr <= 0xFFFF):
            issues.append(f"Warning: {name} address {hex(addr)} may be invalid for GBC")
    
    return issues

def get_address_conflicts():
    """
    Return a summary of address conflicts between the original files
    
    Returns:
        dict: Mapping of conflict categories to conflicting addresses
    """
    conflicts = {
        'player_position': {
            'player_map': {'original': 0xDCBA, 'alternative': 0xDCB5},
            'player_direction': {'original': 0xDCBB, 'alternative': 0xDCBA},
        },
        'player_stats': {
            'player_hp': {'original': 0xDCDA, 'alternative': 0xD16C},
            'player_max_hp': {'original': 0xDCDB, 'alternative': 0xD16E},
            'player_level': {'original': 0xDCD3, 'alternative': 0xD172},
            'player_exp': {'original': 0xDCD5, 'alternative': 0xD179},
            'player_exp_to_next': {'original': 0xDCD8, 'alternative': 0xD17C},
        },
        'party_info': {
            'party_count': {'original': 0xDCD7, 'alternative': 0xD163},
            'party_species': {'original': 0xDCE9, 'alternative': 0xD164},
            'party_hp': {'original': 0xDD0F, 'alternative': 0xD16C},
            'party_max_hp': {'original': 0xDD1B, 'alternative': 0xD178},
            'party_level': {'original': 0xDD27, 'alternative': 0xD184},
            'party_status': {'original': 0xDD33, 'alternative': 0xD18A},
        },
        'money': {
            'money': {'original': 0xD84E, 'alternative': 0xD84A},
        },
        'badges': {
            'badges': {'original': 0xD855, 'alternative': 0xD857},
            'kanto_badges': {'original': 0xD856, 'alternative': 0xD858},
            'elite_four_beaten': {'original': 0xD857, 'alternative': 0xD864},
            'champion_beaten': {'original': 0xD858, 'alternative': 0xD865},
        }
    }
    return conflicts

def test_address_group(memory_reader, group_name: str, use_alternative: bool = False):
    """
    Test a specific address group to determine which version works for your ROM
    
    Args:
        memory_reader: Memory reader instance with read_memory method
        group_name: Name of the address group to test
        use_alternative: Whether to test alternative addresses
    
    Returns:
        dict: Results of testing the address group
    """
    if group_name not in ADDRESS_GROUPS:
        raise ValueError(f"Unknown address group: {group_name}")
    
    group = ADDRESS_GROUPS[group_name]
    addresses_to_test = group['alternative'] if use_alternative else group['primary']
    
    results = {}
    for addr_name in addresses_to_test:
        if addr_name in MEMORY_ADDRESSES:
            addr = MEMORY_ADDRESSES[addr_name]
            try:
                value = memory_reader.read_memory(addr)
                results[addr_name] = {
                    'address': hex(addr),
                    'value': value,
                    'valid': True
                }
            except Exception as e:
                results[addr_name] = {
                    'address': hex(addr),
                    'error': str(e),
                    'valid': False
                }
    
    return results

def get_best_addresses_for_rom(test_results: dict):
    """
    Analyze test results to recommend the best address set for a specific ROM
    
    Args:
        test_results: Results from testing both primary and alternative address groups
    
    Returns:
        dict: Recommended addresses for this ROM version
    """
    recommendations = {}
    
    for group_name in ADDRESS_GROUPS:
        primary_results = test_results.get(f"{group_name}_primary", {})
        alt_results = test_results.get(f"{group_name}_alternative", {})
        
        # Count valid addresses in each group
        primary_valid = sum(1 for result in primary_results.values() if result.get('valid', False))
        alt_valid = sum(1 for result in alt_results.values() if result.get('valid', False))
        
        # Recommend the group with more valid addresses
        if alt_valid > primary_valid:
            recommendations[group_name] = 'alternative'
        else:
            recommendations[group_name] = 'primary'
    
    return recommendations

if __name__ == "__main__":
    issues = validate_memory_addresses()
    conflicts = get_address_conflicts()
    
    print("=== CONSOLIDATED POKEMON CRYSTAL MEMORY MAP ===")
    print(f"Loaded {len(MEMORY_ADDRESSES)} memory addresses")
    print(f"Loaded {len(IMPORTANT_LOCATIONS)} important locations")
    print(f"Loaded {len(BADGE_MASKS)} badge definitions")
    print()
    
    if issues:
        print("VALIDATION ISSUES:")
        for issue in issues:
            print(f"  {issue}")
        print()
    
    print("ADDRESS CONFLICTS DETECTED:")
    for category, category_conflicts in conflicts.items():
        print(f"\n{category.upper()}:")
        for addr_name, addresses in category_conflicts.items():
            print(f"  {addr_name}:")
            print(f"    Original:    {hex(addresses['original'])}")
            print(f"    Alternative: {hex(addresses['alternative'])}")
    
    print()
    print("RECOMMENDATIONS:")
    print("1. Test both address sets with your specific ROM version")
    print("2. Use test_address_group() function to validate addresses")
    print("3. Use get_best_addresses_for_rom() to get recommendations")
    print("4. Consider that different ROM versions may need different addresses")