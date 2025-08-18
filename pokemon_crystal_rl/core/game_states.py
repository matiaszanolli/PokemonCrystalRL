"""Game state tracking and management for PyBoy Pokemon Crystal environment."""

from enum import Enum


class PyBoyGameState(Enum):
    """Game states for Pokemon Crystal."""
    UNKNOWN = "unknown"
    LOADING = "loading"
    INTRO_SEQUENCE = "intro_sequence"
    TITLE_SCREEN = "title_screen"
    NEW_GAME_MENU = "new_game_menu"
    NAME_SELECTION = "name_selection"
    OVERWORLD = "overworld"
    BATTLE = "battle"
    MENU = "menu"
    DIALOGUE = "dialogue"
    CUTSCENE = "cutscene"
    BLACK_SCREEN = "black_screen"
    OPTIONS = "options"
    POKEMON_CENTER = "pokemon_center"
    POKEMART = "pokemart"
    PC_MENU = "pc_menu"
    BAG = "bag"
    TRAINER_CARD = "trainer_card"
    SAVE_MENU = "save_menu"
    POKEMON_LIST = "pokemon_list"
    POKEMON_STATS = "pokemon_stats"
    POKEMON_MOVES = "pokemon_moves"


class GameStateTransition:
    """Represents a transition between game states."""
    def __init__(self, from_state: PyBoyGameState, to_state: PyBoyGameState, reward: float = 0.0):
        self.from_state = from_state
        self.to_state = to_state
        self.reward = reward
        self.valid = True


# State transition rewards
STATE_TRANSITION_REWARDS = {
    # Progress rewards
    (PyBoyGameState.TITLE_SCREEN, PyBoyGameState.NEW_GAME_MENU): 5.0,  # Starting new game
    (PyBoyGameState.NEW_GAME_MENU, PyBoyGameState.OVERWORLD): 10.0,  # First entry to overworld
    (PyBoyGameState.BATTLE, PyBoyGameState.OVERWORLD): 3.0,  # Completing battle
    (PyBoyGameState.DIALOGUE, PyBoyGameState.OVERWORLD): 2.0,  # Completing dialogue
    (PyBoyGameState.CUTSCENE, PyBoyGameState.OVERWORLD): 2.0,  # Completing cutscene
    
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
