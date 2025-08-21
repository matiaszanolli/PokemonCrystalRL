"""Pokemon Crystal RL Core Systems

This package contains the core systems and utilities used by the Pokemon Crystal
RL agent, including state detection, vision processing, choice recognition, and
other shared functionality.
"""

from shared_types import PyBoyGameState
from core.game_states import STATE_UI_ELEMENTS, STATE_TRANSITION_REWARDS
from trainer.choice_recognition_system import (
    ChoiceType,
    ChoicePosition,
    ChoiceContext,
    RecognizedChoice,
    ChoiceRecognitionSystem
)
