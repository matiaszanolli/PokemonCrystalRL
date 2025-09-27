"""
Experiment Models - Data structures for A/B testing framework

Defines the core data models used throughout the A/B testing system,
including experiments, configurations, results, and metrics.
"""

import uuid
import time
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Union, Tuple
from enum import Enum
import json


class ExperimentStatus(Enum):
    """Status of an A/B test experiment"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class ExperimentType(Enum):
    """Type of A/B test experiment"""
    PLUGIN_COMPARISON = "plugin_comparison"
    AGENT_COMPARISON = "agent_comparison"
    CONFIGURATION_COMPARISON = "configuration_comparison"
    STRATEGY_COMPARISON = "strategy_comparison"
    MULTI_FACTOR = "multi_factor"


class MetricType(Enum):
    """Types of metrics that can be collected"""
    REWARD = "reward"
    ACTIONS_PER_SECOND = "actions_per_second"
    SUCCESS_RATE = "success_rate"
    PROGRESS_RATE = "progress_rate"
    BATTLE_WIN_RATE = "battle_win_rate"
    EXPLORATION_COVERAGE = "exploration_coverage"
    LLM_DECISION_QUALITY = "llm_decision_quality"
    MEMORY_USAGE = "memory_usage"
    ERROR_RATE = "error_rate"
    CUSTOM = "custom"


@dataclass
class PerformanceMetrics:
    """Collection of performance metrics for a test variant"""
    total_reward: float = 0.0
    actions_per_second: float = 0.0
    success_rate: float = 0.0
    progress_rate: float = 0.0
    battle_win_rate: float = 0.0
    exploration_coverage: float = 0.0
    llm_decision_quality: float = 0.0
    memory_usage: float = 0.0
    error_rate: float = 0.0

    # Raw data for statistical analysis
    reward_samples: List[float] = field(default_factory=list)
    action_times: List[float] = field(default_factory=list)
    battle_results: List[bool] = field(default_factory=list)

    # Custom metrics
    custom_metrics: Dict[str, float] = field(default_factory=dict)

    def add_reward_sample(self, reward: float):
        """Add a reward sample to the collection"""
        self.reward_samples.append(reward)
        self.total_reward += reward

    def add_action_time(self, duration: float):
        """Add an action duration sample"""
        self.action_times.append(duration)
        if self.action_times:
            self.actions_per_second = len(self.action_times) / sum(self.action_times)

    def add_battle_result(self, won: bool):
        """Add a battle result"""
        self.battle_results.append(won)
        if self.battle_results:
            self.battle_win_rate = sum(self.battle_results) / len(self.battle_results)

    def add_custom_metric(self, name: str, value: float):
        """Add a custom metric value"""
        self.custom_metrics[name] = value

    def get_sample_count(self) -> int:
        """Get the total number of samples collected"""
        return len(self.reward_samples)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for serialization"""
        return {
            'total_reward': self.total_reward,
            'actions_per_second': self.actions_per_second,
            'success_rate': self.success_rate,
            'progress_rate': self.progress_rate,
            'battle_win_rate': self.battle_win_rate,
            'exploration_coverage': self.exploration_coverage,
            'llm_decision_quality': self.llm_decision_quality,
            'memory_usage': self.memory_usage,
            'error_rate': self.error_rate,
            'sample_count': self.get_sample_count(),
            'custom_metrics': self.custom_metrics
        }


@dataclass
class MetricCollection:
    """Collection of metrics over time for statistical analysis"""
    metrics: Dict[str, List[float]] = field(default_factory=dict)
    timestamps: List[float] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_metric(self, name: str, value: float, timestamp: float = None):
        """Add a metric value with timestamp"""
        if timestamp is None:
            timestamp = time.time()

        if name not in self.metrics:
            self.metrics[name] = []

        self.metrics[name].append(value)
        self.timestamps.append(timestamp)

    def get_latest(self, name: str) -> Optional[float]:
        """Get the latest value for a metric"""
        if name in self.metrics and self.metrics[name]:
            return self.metrics[name][-1]
        return None

    def get_average(self, name: str) -> Optional[float]:
        """Get the average value for a metric"""
        if name in self.metrics and self.metrics[name]:
            return sum(self.metrics[name]) / len(self.metrics[name])
        return None

    def get_trend(self, name: str, window_size: int = 10) -> List[float]:
        """Get recent trend for a metric"""
        if name in self.metrics:
            return self.metrics[name][-window_size:]
        return []


@dataclass
class ExperimentConfig:
    """Configuration for an A/B test experiment"""
    name: str
    experiment_type: ExperimentType
    description: str = ""

    # Test variants - each variant has a name and configuration
    variants: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Test parameters
    sample_size_per_variant: int = 100
    max_runtime_seconds: int = 3600  # 1 hour default
    confidence_level: float = 0.95
    minimum_effect_size: float = 0.1  # Minimum meaningful difference

    # Metrics to track
    primary_metrics: List[str] = field(default_factory=lambda: ['total_reward'])
    secondary_metrics: List[str] = field(default_factory=lambda: ['actions_per_second', 'battle_win_rate'])

    # Test conditions
    randomization_seed: Optional[int] = None
    parallel_execution: bool = True
    early_stopping: bool = True  # Stop early if significance is reached

    # Integration settings
    save_state_path: Optional[str] = None
    rom_path: Optional[str] = None
    max_actions_per_run: int = 1000

    def add_variant(self, name: str, config: Dict[str, Any]):
        """Add a test variant"""
        self.variants[name] = config

    def get_variant_names(self) -> List[str]:
        """Get all variant names"""
        return list(self.variants.keys())

    def validate(self) -> bool:
        """Validate experiment configuration"""
        if len(self.variants) < 2:
            return False
        if self.sample_size_per_variant < 10:
            return False
        if not self.primary_metrics:
            return False
        return True


@dataclass
class ExperimentResult:
    """Results from a completed A/B test experiment"""
    experiment_id: str
    experiment_name: str
    status: ExperimentStatus

    # Timing
    start_time: float
    end_time: Optional[float] = None
    duration_seconds: Optional[float] = None

    # Results per variant
    variant_results: Dict[str, PerformanceMetrics] = field(default_factory=dict)
    variant_sample_counts: Dict[str, int] = field(default_factory=dict)

    # Statistical analysis
    statistical_significance: Dict[str, bool] = field(default_factory=dict)  # per metric
    p_values: Dict[str, float] = field(default_factory=dict)  # per metric
    confidence_intervals: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    effect_sizes: Dict[str, float] = field(default_factory=dict)

    # Winner determination
    winning_variant: Optional[str] = None
    winning_confidence: float = 0.0

    # Detailed analysis
    analysis_summary: str = ""
    recommendations: List[str] = field(default_factory=list)

    # Raw data for further analysis
    raw_metrics: Dict[str, MetricCollection] = field(default_factory=dict)

    def add_variant_result(self, variant_name: str, metrics: PerformanceMetrics):
        """Add results for a variant"""
        self.variant_results[variant_name] = metrics
        self.variant_sample_counts[variant_name] = metrics.get_sample_count()

    def set_winner(self, variant_name: str, confidence: float):
        """Set the winning variant"""
        self.winning_variant = variant_name
        self.winning_confidence = confidence

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the experiment results"""
        if self.end_time:
            self.duration_seconds = self.end_time - self.start_time

        return {
            'experiment_id': self.experiment_id,
            'experiment_name': self.experiment_name,
            'status': self.status.value,
            'duration_seconds': self.duration_seconds,
            'winning_variant': self.winning_variant,
            'winning_confidence': self.winning_confidence,
            'variant_count': len(self.variant_results),
            'total_samples': sum(self.variant_sample_counts.values()),
            'significant_metrics': len([m for m, sig in self.statistical_significance.items() if sig]),
            'recommendations': self.recommendations
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization"""
        return {
            'experiment_id': self.experiment_id,
            'experiment_name': self.experiment_name,
            'status': self.status.value,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration_seconds': self.duration_seconds,
            'variant_results': {k: v.to_dict() for k, v in self.variant_results.items()},
            'variant_sample_counts': self.variant_sample_counts,
            'statistical_significance': self.statistical_significance,
            'p_values': self.p_values,
            'confidence_intervals': self.confidence_intervals,
            'effect_sizes': self.effect_sizes,
            'winning_variant': self.winning_variant,
            'winning_confidence': self.winning_confidence,
            'analysis_summary': self.analysis_summary,
            'recommendations': self.recommendations
        }


@dataclass
class Experiment:
    """Main experiment entity combining configuration and execution state"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    config: ExperimentConfig = field(default_factory=lambda: ExperimentConfig("", ExperimentType.PLUGIN_COMPARISON))
    status: ExperimentStatus = ExperimentStatus.PENDING
    result: Optional[ExperimentResult] = None

    # Execution state
    current_variant: Optional[str] = None
    current_run: int = 0
    total_runs: int = 0

    # Real-time metrics
    live_metrics: Dict[str, MetricCollection] = field(default_factory=dict)

    # Error handling
    error_message: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3

    def __post_init__(self):
        """Initialize experiment after creation"""
        if self.config.validate():
            self.total_runs = len(self.config.variants) * self.config.sample_size_per_variant
            # Initialize live metrics for each variant
            for variant_name in self.config.get_variant_names():
                self.live_metrics[variant_name] = MetricCollection()
        else:
            self.status = ExperimentStatus.FAILED
            self.error_message = "Invalid experiment configuration"

    def start(self):
        """Mark experiment as started"""
        if self.status == ExperimentStatus.PENDING:
            self.status = ExperimentStatus.RUNNING
            if not self.result:
                self.result = ExperimentResult(
                    experiment_id=self.id,
                    experiment_name=self.config.name,
                    status=self.status,
                    start_time=time.time()
                )

    def complete(self):
        """Mark experiment as completed"""
        self.status = ExperimentStatus.COMPLETED
        if self.result:
            self.result.status = self.status
            self.result.end_time = time.time()

    def fail(self, error_message: str):
        """Mark experiment as failed"""
        self.status = ExperimentStatus.FAILED
        self.error_message = error_message
        if self.result:
            self.result.status = self.status

    def get_progress(self) -> float:
        """Get experiment progress as percentage"""
        if self.total_runs == 0:
            return 0.0
        return min(100.0, (self.current_run / self.total_runs) * 100.0)

    def can_retry(self) -> bool:
        """Check if experiment can be retried"""
        return self.retry_count < self.max_retries

    def to_dict(self) -> Dict[str, Any]:
        """Convert experiment to dictionary for serialization"""
        return {
            'id': self.id,
            'config': {
                'name': self.config.name,
                'experiment_type': self.config.experiment_type.value,
                'description': self.config.description,
                'variants': self.config.variants,
                'sample_size_per_variant': self.config.sample_size_per_variant,
                'primary_metrics': self.config.primary_metrics,
                'secondary_metrics': self.config.secondary_metrics
            },
            'status': self.status.value,
            'current_variant': self.current_variant,
            'current_run': self.current_run,
            'total_runs': self.total_runs,
            'progress': self.get_progress(),
            'error_message': self.error_message,
            'result': self.result.to_dict() if self.result else None
        }