"""Constants for Pokemon Crystal RL agent."""

from enum import Enum

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
    CUTSCENE = "cutscene"

# State transition rewards
STATE_TRANSITION_REWARDS = {
    # Progress rewards
    (PyBoyGameState.TITLE_SCREEN, PyBoyGameState.NEW_GAME_MENU): 5.0,
    (PyBoyGameState.NEW_GAME_MENU, PyBoyGameState.OVERWORLD): 10.0,
    (PyBoyGameState.BATTLE, PyBoyGameState.OVERWORLD): 3.0,
    (PyBoyGameState.DIALOGUE, PyBoyGameState.OVERWORLD): 2.0,
    
    # Menu usage rewards
    (PyBoyGameState.OVERWORLD, PyBoyGameState.MENU): 1.0,
    (PyBoyGameState.MENU, PyBoyGameState.OVERWORLD): 2.0,
    (PyBoyGameState.POKEMON_LIST, PyBoyGameState.POKEMON_STATS): 0.5,
    (PyBoyGameState.POKEMON_STATS, PyBoyGameState.POKEMON_MOVES): 0.5,
    
    # Survival/utility state rewards
    (PyBoyGameState.OVERWORLD, PyBoyGameState.POKEMON_CENTER): 1.0,
    (PyBoyGameState.POKEMON_CENTER, PyBoyGameState.OVERWORLD): 2.0,
    (PyBoyGameState.OVERWORLD, PyBoyGameState.POKEMART): 1.0,
    (PyBoyGameState.POKEMART, PyBoyGameState.OVERWORLD): 2.0,
    
    # Save/load rewards
    (PyBoyGameState.MENU, PyBoyGameState.SAVE_MENU): 0.5,
    (PyBoyGameState.SAVE_MENU, PyBoyGameState.OVERWORLD): 2.0,
}
