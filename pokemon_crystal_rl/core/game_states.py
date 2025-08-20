"""
Game state definitions and management for Pokemon Crystal RL agent.
"""

from enum import Enum, auto

class PyBoyGameState(Enum):
    """Game states for Pokemon Crystal"""
    UNKNOWN = "unknown"
    TITLE_SCREEN = "title_screen"
    INTRO = "intro"
    OVERWORLD = "overworld"
    MENU = "menu"
    DIALOGUE = "dialogue"
    BATTLE = "battle"
    EVOLUTION = "evolution"
    TRANSITION = "transition"
    POKEMON_CENTER = "pokemon_center"
    SHOP = "shop"
    GAME_OVER = "game_over"
    LOADING = "loading"
    INTRO_SEQUENCE = "intro_sequence"
    NEW_GAME_MENU = "new_game_menu"
    NAME_SELECTION = "name_selection"
    POKEMART = "pokemart"
    PC_MENU = "pc_menu"
    BAG = "bag"
    TRAINER_CARD = "trainer_card"
    SAVE_MENU = "save_menu"
    POKEMON_LIST = "pokemon_list"
    POKEMON_STATS = "pokemon_stats"
    POKEMON_MOVES = "pokemon_moves"
    NEW_GAME = "new_game"
    PC_BOX = "pc_box"
    CONTINUE = "continue"
    PAUSE = "pause"

    @classmethod
    def from_str(cls, state_str: str) -> "PyBoyGameState":
        """Convert string to enum value."""
        try:
            return cls(state_str)
        except ValueError:
            return cls.UNKNOWN

    @property
    def allows_input(self) -> bool:
        """Whether the state allows user input."""
        non_input_states = {
            PyBoyGameState.LOADING,
            PyBoyGameState.INTRO,
            PyBoyGameState.EVOLUTION
        }
        return self not in non_input_states

    @property
    def requires_action(self) -> bool:
        """Whether the state requires an action to progress."""
        action_required_states = {
            PyBoyGameState.TITLE_SCREEN,
            PyBoyGameState.NEW_GAME,
            PyBoyGameState.CONTINUE,
            PyBoyGameState.DIALOGUE,
            PyBoyGameState.MENU
        }
        return self in action_required_states

    @property
    def is_interactive(self) -> bool:
        """Whether the state is part of normal gameplay interaction."""
        interactive_states = {
            PyBoyGameState.OVERWORLD,
            PyBoyGameState.BATTLE,
            PyBoyGameState.MENU,
            PyBoyGameState.PC_BOX,
            PyBoyGameState.PAUSE
        }
        return self in interactive_states

    @property
    def is_transitional(self) -> bool:
        """Whether the state is transitional."""
        transitional_states = {
            PyBoyGameState.LOADING,
            PyBoyGameState.INTRO_SEQUENCE,
            PyBoyGameState.EVOLUTION
        }
        return self in transitional_states

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

# State transition rewards
STATE_TRANSITION_REWARDS = {
    # Progress rewards
    (PyBoyGameState.TITLE_SCREEN, PyBoyGameState.NEW_GAME_MENU): 5.0,  # Starting new game
    (PyBoyGameState.NEW_GAME_MENU, PyBoyGameState.OVERWORLD): 10.0,  # First entry to overworld
    (PyBoyGameState.BATTLE, PyBoyGameState.OVERWORLD): 3.0,  # Completing battle
    (PyBoyGameState.DIALOGUE, PyBoyGameState.OVERWORLD): 2.0,  # Completing dialogue
    
    # Menu usage rewards
    (PyBoyGameState.OVERWORLD, PyBoyGameState.MENU): 1.0,  # Opening menu
    (PyBoyGameState.MENU, PyBoyGameState.OVERWORLD): 2.0,  # Exiting menu properly
    (PyBoyGameState.POKEMON_LIST, PyBoyGameState.POKEMON_STATS): 0.5,  # Checking Pokemon stats
    (PyBoyGameState.POKEMON_STATS, PyBoyGameState.POKEMON_MOVES): 0.5,  # Checking Pokemon moves
    
    # Survival/utility state rewards
    (PyBoyGameState.OVERWORLD, PyBoyGameState.POKEMON_CENTER): 1.0,  # Seeking healing
    (PyBoyGameState.POKEMON_CENTER, PyBoyGameState.OVERWORLD): 2.0,  # Finished healing
    (PyBoyGameState.OVERWORLD, PyBoyGameState.POKEMART): 1.0,  # Buying items
    (PyBoyGameState.POKEMART, PyBoyGameState.OVERWORLD): 2.0,  # Finished shopping
    
    # Save/load rewards
    (PyBoyGameState.MENU, PyBoyGameState.SAVE_MENU): 0.5,  # Attempting to save
    (PyBoyGameState.SAVE_MENU, PyBoyGameState.OVERWORLD): 2.0,  # Successfully saved
}
