"""
A/B Testing REST API Models

Data models specifically for A/B testing REST API endpoints,
providing comprehensive experiment management capabilities.
"""

from dataclasses import dataclass, asdict, field
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from enum import Enum
import time

from core.ab_testing import ExperimentStatus, ExperimentType


class ExperimentAction(Enum):
    """Experiment control actions"""
    START = "start"
    STOP = "stop"
    PAUSE = "pause"
    RESUME = "resume"
    CANCEL = "cancel"


@dataclass
class ExperimentCreateRequest:
    """Request model for creating new experiments"""
    name: str
    experiment_type: str  # Will be converted to ExperimentType
    description: str = ""

    # Variants configuration
    variants: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Test parameters
    sample_size_per_variant: int = 30
    max_runtime_seconds: int = 3600
    confidence_level: float = 0.95
    minimum_effect_size: float = 0.1

    # Metrics configuration
    primary_metrics: List[str] = field(default_factory=lambda: ['total_reward'])
    secondary_metrics: List[str] = field(default_factory=lambda: ['actions_per_second', 'battle_win_rate'])

    # Execution settings
    randomization_seed: Optional[int] = None
    parallel_execution: bool = True
    early_stopping: bool = True

    # Integration settings
    save_state_path: Optional[str] = None
    rom_path: Optional[str] = None
    max_actions_per_run: int = 1000

    def to_experiment_config(self):
        """Convert to ExperimentConfig"""
        from core.ab_testing import ExperimentConfig, ExperimentType

        # Convert string to enum
        exp_type = ExperimentType(self.experiment_type)

        config = ExperimentConfig(
            name=self.name,
            experiment_type=exp_type,
            description=self.description,
            variants=self.variants,
            sample_size_per_variant=self.sample_size_per_variant,
            max_runtime_seconds=self.max_runtime_seconds,
            confidence_level=self.confidence_level,
            minimum_effect_size=self.minimum_effect_size,
            primary_metrics=self.primary_metrics,
            secondary_metrics=self.secondary_metrics,
            randomization_seed=self.randomization_seed,
            parallel_execution=self.parallel_execution,
            early_stopping=self.early_stopping,
            save_state_path=self.save_state_path,
            rom_path=self.rom_path,
            max_actions_per_run=self.max_actions_per_run
        )

        return config


@dataclass
class ExperimentControlRequest:
    """Request model for experiment control actions"""
    action: str  # Will be converted to ExperimentAction
    configuration: Optional[Dict[str, Any]] = None


@dataclass
class ExperimentSummaryModel:
    """Summary model for experiment display"""
    experiment_id: str
    name: str
    status: str
    experiment_type: str
    created_time: float
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    duration_seconds: Optional[float] = None

    # Progress information
    progress_percentage: float = 0.0
    current_variant: Optional[str] = None
    current_run: int = 0
    total_runs: int = 0

    # Results summary
    variant_count: int = 0
    total_samples: int = 0
    has_significant_results: bool = False
    winning_variant: Optional[str] = None
    winning_confidence: float = 0.0

    # Error information
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)


@dataclass
class ExperimentDetailModel:
    """Detailed model for experiment information"""
    experiment_id: str
    name: str
    description: str
    status: str
    experiment_type: str

    # Configuration
    sample_size_per_variant: int
    max_runtime_seconds: int
    confidence_level: float
    primary_metrics: List[str]
    secondary_metrics: List[str]

    # Variants
    variants: Dict[str, Dict[str, Any]]
    variant_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Execution state
    created_time: float = field(default_factory=time.time)
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    duration_seconds: Optional[float] = None

    # Progress
    progress_percentage: float = 0.0
    current_variant: Optional[str] = None
    current_run: int = 0
    total_runs: int = 0

    # Live metrics
    live_metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Results (populated when complete)
    statistical_analysis: Optional[Dict[str, Any]] = None
    recommendations: List[str] = field(default_factory=list)

    # Error handling
    error_message: Optional[str] = None
    retry_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)


@dataclass
class VariantMetricsModel:
    """Model for variant performance metrics"""
    variant_name: str
    sample_count: int

    # Primary metrics
    total_reward: float = 0.0
    average_reward: float = 0.0
    reward_std_dev: float = 0.0

    # Performance metrics
    actions_per_second: float = 0.0
    battle_win_rate: float = 0.0
    success_rate: float = 0.0

    # Custom metrics
    custom_metrics: Dict[str, float] = field(default_factory=dict)

    # Statistical data
    confidence_interval: Optional[tuple] = None
    effect_size: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)


@dataclass
class StatisticalAnalysisModel:
    """Model for statistical analysis results"""
    experiment_id: str

    # Primary metric results
    primary_metric_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    secondary_metric_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Overall conclusions
    has_significant_results: bool = False
    winning_variant: Optional[str] = None
    confidence_score: float = 0.0

    # Recommendations
    recommendations: List[str] = field(default_factory=list)
    summary: str = ""

    # Detailed results
    variant_comparisons: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)


@dataclass
class ExperimentTemplateModel:
    """Model for experiment templates"""
    template_id: str
    name: str
    description: str
    experiment_type: str

    # Pre-configured variants
    default_variants: Dict[str, Dict[str, Any]]

    # Suggested configuration
    suggested_sample_size: int = 30
    suggested_metrics: List[str] = field(default_factory=list)

    # Template metadata
    category: str = "general"
    difficulty_level: str = "beginner"  # beginner, intermediate, advanced
    estimated_runtime_minutes: int = 60

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)


@dataclass
class ExperimentListResponse:
    """Response model for experiment list endpoint"""
    experiments: List[ExperimentSummaryModel]
    total_count: int
    active_count: int
    completed_count: int
    failed_count: int

    # Pagination (for future use)
    page: int = 1
    per_page: int = 20
    has_next: bool = False
    has_prev: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'experiments': [exp.to_dict() for exp in self.experiments],
            'total_count': self.total_count,
            'active_count': self.active_count,
            'completed_count': self.completed_count,
            'failed_count': self.failed_count,
            'page': self.page,
            'per_page': self.per_page,
            'has_next': self.has_next,
            'has_prev': self.has_prev
        }


@dataclass
class ExperimentProgressModel:
    """Model for real-time experiment progress"""
    experiment_id: str
    status: str
    progress_percentage: float
    current_variant: Optional[str]
    current_run: int
    total_runs: int
    elapsed_seconds: float
    estimated_completion_seconds: Optional[float] = None

    # Live performance data
    latest_metrics: Dict[str, float] = field(default_factory=dict)
    variant_progress: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)


# Utility functions for model conversion
def experiment_to_summary_model(experiment) -> ExperimentSummaryModel:
    """Convert Experiment to ExperimentSummaryModel"""
    from core.ab_testing import ExperimentStatus

    # Calculate duration if available
    duration = None
    if experiment.result and experiment.result.start_time and experiment.result.end_time:
        duration = experiment.result.end_time - experiment.result.start_time

    # Count samples
    total_samples = 0
    if experiment.result and experiment.result.variant_sample_counts:
        total_samples = sum(experiment.result.variant_sample_counts.values())

    return ExperimentSummaryModel(
        experiment_id=experiment.id,
        name=experiment.config.name,
        status=experiment.status.value,
        experiment_type=experiment.config.experiment_type.value,
        created_time=experiment.result.start_time if experiment.result else time.time(),
        start_time=experiment.result.start_time if experiment.result else None,
        end_time=experiment.result.end_time if experiment.result else None,
        duration_seconds=duration,
        progress_percentage=experiment.get_progress(),
        current_variant=experiment.current_variant,
        current_run=experiment.current_run,
        total_runs=experiment.total_runs,
        variant_count=len(experiment.config.variants),
        total_samples=total_samples,
        has_significant_results=bool(experiment.result and experiment.result.winning_variant),
        winning_variant=experiment.result.winning_variant if experiment.result else None,
        winning_confidence=experiment.result.winning_confidence if experiment.result else 0.0,
        error_message=experiment.error_message
    )


def experiment_to_detail_model(experiment, analysis=None) -> ExperimentDetailModel:
    """Convert Experiment to ExperimentDetailModel with optional analysis"""

    # Build live metrics summary
    live_metrics = {}
    for variant_name, metrics in experiment.live_metrics.items():
        live_metrics[variant_name] = {
            'sample_count': len(metrics.metrics.get('reward', [])),
            'latest_reward': metrics.get_latest('reward'),
            'average_reward': metrics.get_average('reward'),
            'latest_actions_per_second': metrics.get_latest('actions_per_second'),
            'latest_battle_win_rate': metrics.get_latest('battle_win_rate')
        }

    # Build variant results if available
    variant_results = {}
    if experiment.result:
        for variant_name, metrics in experiment.result.variant_results.items():
            variant_results[variant_name] = metrics.to_dict()

    # Calculate duration
    duration = None
    if experiment.result and experiment.result.start_time and experiment.result.end_time:
        duration = experiment.result.end_time - experiment.result.start_time

    return ExperimentDetailModel(
        experiment_id=experiment.id,
        name=experiment.config.name,
        description=experiment.config.description,
        status=experiment.status.value,
        experiment_type=experiment.config.experiment_type.value,
        sample_size_per_variant=experiment.config.sample_size_per_variant,
        max_runtime_seconds=experiment.config.max_runtime_seconds,
        confidence_level=experiment.config.confidence_level,
        primary_metrics=experiment.config.primary_metrics,
        secondary_metrics=experiment.config.secondary_metrics,
        variants=experiment.config.variants,
        variant_results=variant_results,
        start_time=experiment.result.start_time if experiment.result else None,
        end_time=experiment.result.end_time if experiment.result else None,
        duration_seconds=duration,
        progress_percentage=experiment.get_progress(),
        current_variant=experiment.current_variant,
        current_run=experiment.current_run,
        total_runs=experiment.total_runs,
        live_metrics=live_metrics,
        statistical_analysis=analysis.to_dict() if analysis else None,
        recommendations=analysis.recommendations if analysis else [],
        error_message=experiment.error_message,
        retry_count=experiment.retry_count
    )