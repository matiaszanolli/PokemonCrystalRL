"""
Game-wide constants for Pokemon Crystal RL

Contains memory addresses, Pokemon data, status effects, and other game-related constants.
"""

# Note: MEMORY_ADDRESSES moved to memory_addresses.py to avoid duplication

# Important game locations
LOCATIONS = {
    24: "Player's Bedroom",
    25: "Player's House", 
    26: "New Bark Town",
    27: "Prof. Elm's Lab",
    28: "Route 29",
    29: "Route 30",
    30: "Cherrygrove City",
}

# Pokemon species IDs (partial list for important ones)
POKEMON_SPECIES = {
    0: "None",
    152: "Chikorita",
    155: "Cyndaquil", 
    158: "Totodile",
    16: "Pidgey",
    19: "Rattata",
    129: "Magikarp",
}

# Status conditions
STATUS_CONDITIONS = {
    0: "Healthy",
    1: "Sleep",
    2: "Poison",
    3: "Burn",
    4: "Freeze",
    5: "Paralysis",
}

# Badge masks for checking individual badges
BADGE_MASKS = {
    'johto': {
        'zephyr': 0x01, 'hive': 0x02, 'plain': 0x04, 'fog': 0x08,
        'storm': 0x10, 'mineral': 0x20, 'glacier': 0x40, 'rising': 0x80
    }
}

# Derived value functions
DERIVED_VALUES = {
    'badges_total': lambda state: state.get('badges_count', 0),
    'health_percentage': lambda state: (state.get('player_hp', 0) / max(state.get('player_max_hp', 1), 1)) * 100,
    'has_pokemon': lambda state: state.get('party_count', 0) > 0,
    'location_key': lambda state: f"{state.get('player_map', 0)}_{state.get('player_x', 0)}_{state.get('player_y', 0)}",
}

# Screen state definitions
SCREEN_STATES = {
    'LOADING': 'loading',
    'DIALOGUE': 'dialogue',
    'MENU': 'menu',
    'BATTLE': 'battle',
    'OVERWORLD': 'overworld',
    'SETTINGS_MENU': 'settings_menu',
    'UNKNOWN': 'unknown',
}

# Action definitions
AVAILABLE_ACTIONS = {
    'MOVEMENT': ['up', 'down', 'left', 'right'],
    'BUTTONS': ['a', 'b', 'start', 'select'],
    'FORBIDDEN_INITIAL': ['start', 'select'],  # Actions forbidden until first Pokemon
}

# Movement direction encoding (Game Boy standard)
MOVEMENT_DIRECTIONS = {
    'DOWN': 0,
    'UP': 2,
    'LEFT': 4,
    'RIGHT': 6,
    'STANDING': 8,
}

# Default Game Boy screen dimensions
SCREEN_DIMENSIONS = {
    'WIDTH': 160,
    'HEIGHT': 144,
    'PIXELS': 160 * 144,
}

# Training parameters
TRAINING_PARAMS = {
    'LLM_INTERVAL': 20,  # Actions between LLM decisions
    'STUCK_THRESHOLD': 100,  # Actions without reward before stuck detection
    'MAX_PARTY_SIZE': 6,
    'MAX_MONEY': 999999,
    'PARTY_SLOT_SIZE': 44,  # Bytes per Pokemon in party data
    'MAX_LEVEL': 100,
}

# Reward constants
REWARD_VALUES = {
    'FIRST_POKEMON': 100.0,
    'ADDITIONAL_POKEMON': 25.0,
    'BADGE': 500.0,
    'BATTLE_START': 2.0,
    'BATTLE_WIN': 20.0,
    'BATTLE_LOSE': -5.0,
    'DIALOGUE_ENGAGE': 0.1,
    'DIALOGUE_PROGRESS': 0.02,
    'NEW_MAP': 10.0,
    'NEW_LOCATION': 0.1,
    'MAP_MOVE': 0.02,
    'LOCAL_MOVE': 0.01,
    'MAX_BLOCKED_PENALTY': -0.1,
    'TIME_PENALTY': -0.01,
}
