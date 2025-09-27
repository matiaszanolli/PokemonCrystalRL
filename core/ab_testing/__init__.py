"""
A/B Testing Framework - Core Components

This module provides comprehensive A/B testing capabilities for the Pokemon Crystal RL platform,
enabling data-driven optimization of agent strategies, plugin configurations, and training parameters.

Key Components:
- ExperimentManager: Core experiment orchestration
- ConfigurationComparator: Plugin/agent configuration testing
- StatisticalAnalyzer: Performance analysis and significance testing
- ExperimentMetrics: Comprehensive metrics collection and analysis
- ExperimentScheduler: Automated experiment scheduling and execution
- AutomationTemplates: Pre-built automation workflows and templates
"""

from .experiment_manager import ExperimentManager
from .experiment_models import ExperimentStatus, ExperimentType
from .configuration_comparator import ConfigurationComparator, ConfigurationVariant, ConfigurationType
from .statistical_analyzer import StatisticalAnalyzer, StatisticalTest, AnalysisResult
from .experiment_scheduler import ExperimentScheduler, ScheduleConfig, ScheduleType, ScheduleStatus, ScheduledExperiment
from .automation_templates import AutomationTemplates
from .experiment_models import (
    Experiment, ExperimentConfig, ExperimentResult,
    MetricCollection, PerformanceMetrics
)

__all__ = [
    'ExperimentManager',
    'ConfigurationComparator',
    'StatisticalAnalyzer',
    'ExperimentScheduler',
    'AutomationTemplates',
    'Experiment',
    'ExperimentConfig',
    'ExperimentResult',
    'ExperimentStatus',
    'ExperimentType',
    'ScheduleConfig',
    'ScheduleType',
    'ScheduleStatus',
    'ScheduledExperiment',
    'ConfigurationVariant',
    'StatisticalTest',
    'AnalysisResult',
    'MetricCollection',
    'PerformanceMetrics'
]