"""Memory address mappings for Pokemon Crystal."""

# Party and Pokemon data - Based on party structure analysis
PARTY_ADDRESSES = {
    'party_count': 0xD163,      # Number of Pokemon in party
    'player_species': 0xD163,   # Species of first Pokemon (party slot 0 + 0)
    'player_held_item': 0xD164, # Held item of first Pokemon (party slot 0 + 1)
    'player_hp': 0xD167,        # Current HP of first Pokemon (party slot 0 + 4, low byte)
    'player_hp_high': 0xD168,   # Current HP of first Pokemon (party slot 0 + 4, high byte)
    'player_max_hp': 0xD169,    # Max HP of first Pokemon (party slot 0 + 6, low byte)
    'player_max_hp_high': 0xD16A, # Max HP of first Pokemon (party slot 0 + 6, high byte)
    'player_level': 0xD16B,     # Level of first Pokemon (party slot 0 + 8)
    'player_status': 0xD16C,    # Status condition of first Pokemon (party slot 0 + 9)
}

# Location and movement - VERIFIED ADDRESSES from coordinate testing
LOCATION_ADDRESSES = {
    'player_map': 0xDCBA,       # Current map ID (VERIFIED)
    'player_x': 0xDCB8,         # Player X coordinate (VERIFIED)
    'player_y': 0xDCB9,         # Player Y coordinate (VERIFIED)
    'player_direction': 0xDCBB, # Direction player is facing (VERIFIED)
}

# Resources and progress - From money/badge analysis 
RESOURCE_ADDRESSES = {
    'money_low': 0xD347,        # Money (low byte, 3 bytes little-endian)
    'money_mid': 0xD348,        # Money (mid byte)
    'money_high': 0xD349,       # Money (high byte)
    'badges': 0xD359,           # Badge flags (bit flags for 8 Johto badges)
}

# Battle state - From battle analysis
BATTLE_ADDRESSES = {
    'in_battle': 0xD057,        # Battle active flag (0=overworld, 1=battle)
    'battle_turn': 0xD068,      # Turn counter in battle
    'enemy_species': 0xD0A5,    # Opponent Pokemon species
    'enemy_hp_low': 0xD0A8,     # Opponent HP (low byte, 2 bytes)
    'enemy_hp_high': 0xD0A9,    # Opponent HP (high byte)
    'enemy_level': 0xD0AA,      # Opponent Pokemon level
    'player_active_slot': 0xD05E, # Player active Pokemon slot (0-5)
    'move_selected': 0xD05F,    # Move selected (0-3)
}

# Misc useful - From misc analysis
MISC_ADDRESSES = {
    'step_counter': 0xD164,     # Step counter for movement tracking
    'game_time_hours': 0xD3E1,  # Time played (hours)
}

# Combined address map
MEMORY_ADDRESSES = {
    **PARTY_ADDRESSES,
    **LOCATION_ADDRESSES,
    **RESOURCE_ADDRESSES,
    **BATTLE_ADDRESSES,
    **MISC_ADDRESSES,
}
