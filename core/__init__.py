"""Pokemon Crystal RL Core Systems

This package contains the core systems and utilities used by the Pokemon Crystal
RL agent, including state detection, vision processing, choice recognition, and
other shared functionality.
"""

from pokemon_crystal_rl.core.game_states import PyBoyGameState, STATE_UI_ELEMENTS, STATE_TRANSITION_REWARDS
from pokemon_crystal_rl.core.vision_processor import DetectedText, GameUIElement, VisualContext, PokemonVisionProcessor
from pokemon_crystal_rl.core.choice_recognition_system import (
    ChoiceType,
    ChoicePosition,
    ChoiceContext,
    RecognizedChoice,
    ChoiceRecognitionSystem
)
