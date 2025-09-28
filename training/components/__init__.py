"""
Training Components Package

Extracted components from monolithic trainer modules for better separation of concerns.
"""

from .emulation_manager import EmulationManager, EmulationConfig
from .llm_decision_engine import LLMDecisionEngine, LLMConfig
from .llm_manager import LLMManager
from .reward_calculator import RewardCalculator, RewardConfig
from .statistics_tracker import StatisticsTracker, TrainingSession, PerformanceMetrics
from .screen_capture_manager import ScreenCaptureManager, ScreenCaptureConfig
from .error_recovery_system import ErrorRecoverySystem, RecoveryConfig, ErrorSeverity, RecoveryStrategy

__all__ = [
    'EmulationManager', 'EmulationConfig',
    'LLMDecisionEngine', 'LLMConfig',
    'LLMManager',
    'RewardCalculator', 'RewardConfig',
    'StatisticsTracker', 'TrainingSession', 'PerformanceMetrics',
    'ScreenCaptureManager', 'ScreenCaptureConfig',
    'ErrorRecoverySystem', 'RecoveryConfig', 'ErrorSeverity', 'RecoveryStrategy'
]