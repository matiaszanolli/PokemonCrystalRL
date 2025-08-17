#!/usr/bin/env python3
"""
Game state definitions for Pokemon Crystal RL agent
"""

from enum import Enum, auto

class PyBoyGameState(Enum):
    """Game states for Pokemon Crystal"""
    UNKNOWN = auto()
    TITLE_SCREEN = auto()
    INTRO = auto()
    OVERWORLD = auto()
    MENU = auto()
    DIALOGUE = auto()
    BATTLE = auto()
    EVOLUTION = auto()
    TRANSITION = auto()
    POKEMON_CENTER = auto()
    SHOP = auto()
    GAME_OVER = auto()

# Common UI elements for each state
STATE_UI_ELEMENTS = {
    PyBoyGameState.TITLE_SCREEN: ["title_text", "start_button"],
    PyBoyGameState.INTRO: ["intro_text", "professor_sprite"],
    PyBoyGameState.OVERWORLD: ["player_sprite", "map_elements"],
    PyBoyGameState.MENU: ["menu_box", "menu_cursor"],
    PyBoyGameState.DIALOGUE: ["dialogue_box", "text_area"],
    PyBoyGameState.BATTLE: ["healthbar", "battle_menu", "pokemon_sprites"],
    PyBoyGameState.EVOLUTION: ["evolution_animation", "pokemon_sprite"],
    PyBoyGameState.POKEMON_CENTER: ["nurse_sprite", "healing_machine"],
    PyBoyGameState.SHOP: ["shop_menu", "item_list"],
}

# State transition rules
VALID_STATE_TRANSITIONS = {
    PyBoyGameState.TITLE_SCREEN: [PyBoyGameState.INTRO],
    PyBoyGameState.INTRO: [PyBoyGameState.OVERWORLD],
    PyBoyGameState.OVERWORLD: [
        PyBoyGameState.MENU,
        PyBoyGameState.BATTLE,
        PyBoyGameState.DIALOGUE,
        PyBoyGameState.POKEMON_CENTER,
        PyBoyGameState.SHOP
    ],
    PyBoyGameState.MENU: [PyBoyGameState.OVERWORLD],
    PyBoyGameState.DIALOGUE: [
        PyBoyGameState.OVERWORLD,
        PyBoyGameState.MENU,
        PyBoyGameState.BATTLE
    ],
    PyBoyGameState.BATTLE: [
        PyBoyGameState.OVERWORLD,
        PyBoyGameState.EVOLUTION,
        PyBoyGameState.DIALOGUE
    ],
    PyBoyGameState.EVOLUTION: [PyBoyGameState.BATTLE, PyBoyGameState.OVERWORLD],
    PyBoyGameState.POKEMON_CENTER: [PyBoyGameState.OVERWORLD],
    PyBoyGameState.SHOP: [PyBoyGameState.OVERWORLD],
    PyBoyGameState.GAME_OVER: [PyBoyGameState.TITLE_SCREEN],
    PyBoyGameState.UNKNOWN: list(PyBoyGameState),  # Can transition to any state
    PyBoyGameState.TRANSITION: list(PyBoyGameState),  # Can transition to any state
}

# State timeouts (in frames)
STATE_TIMEOUTS = {
    PyBoyGameState.TITLE_SCREEN: 3600,  # 1 minute
    PyBoyGameState.INTRO: 1800,  # 30 seconds
    PyBoyGameState.DIALOGUE: 600,  # 10 seconds
    PyBoyGameState.BATTLE: 3600,  # 1 minute
    PyBoyGameState.EVOLUTION: 900,  # 15 seconds
    PyBoyGameState.TRANSITION: 180,  # 3 seconds
}

# Default timeout for unlisted states
DEFAULT_STATE_TIMEOUT = 1800  # 30 seconds
