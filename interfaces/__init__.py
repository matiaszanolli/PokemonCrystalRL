"""Pokemon Crystal RL Interfaces.

This package provides the core interfaces used throughout the Pokemon Crystal RL project.
These interfaces help enforce clean separation of concerns and avoid circular dependencies
between components.

Usage:
    from interfaces.trainers import TrainerInterface
    from interfaces.monitoring import MonitoringComponent
    from interfaces.vision import VisionProcessorInterface
"""

from .monitoring import (
    MonitoringComponent,
    MonitoringStats,
    ScreenCaptureComponent,
    WebMonitorInterface,
    StatsCollectorInterface,
    TrainingMonitorInterface,
    ErrorHandlerInterface,
)

from .trainers import (
    TrainerConfig,
    TrainingState,
    GameState,
    PyBoyInterface,
    AgentInterface,
    RewardCalculatorInterface,
    TrainerInterface,
)

from .vision import (
    BoundingBox,
    DetectedText,
    GameUIElement,
    VisualContext,
    FontDecoderInterface,
    ScreenProcessorInterface,
    VisionProcessorInterface,
)

# Export package info
__version__ = '0.1.0'
__author__ = 'Pokemon Crystal RL Team'
