"""Pokemon Crystal RL Core Systems

This package contains the core systems and utilities used by the Pokemon Crystal
RL agent, including state detection, vision processing, choice recognition, and
other shared functionality.
"""

from shared_types import PyBoyGameState
from .state.machine import STATE_UI_ELEMENTS, STATE_TRANSITION_REWARDS
from .choice_recognition import (
    ChoiceType,
    ChoicePosition,
    ChoiceContext,
    RecognizedChoice,
    ChoiceRecognitionSystem
)
from .vision_processor import DetectedText, VisualContext, UnifiedVisionProcessor
