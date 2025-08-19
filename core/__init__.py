"""Pokemon Crystal RL Core Systems

This package contains the core systems and utilities used by the Pokemon Crystal
RL agent, including state detection, vision processing, choice recognition, and
other shared functionality.
"""

from .vision_processor import DetectedText, GameUIElement, VisualContext
from .choice_recognition_system import (
    ChoiceType,
    ChoicePosition,
    ChoiceContext,
    RecognizedChoice,
    ChoiceRecognitionSystem
)
