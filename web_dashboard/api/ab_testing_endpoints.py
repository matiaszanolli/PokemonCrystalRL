"""
A/B Testing REST API Endpoints

Comprehensive REST API providing full control over A/B testing experiments,
including creation, execution, monitoring, and analysis.
"""

import logging
import json
from typing import Dict, Any, Optional, List
from pathlib import Path
import time

from .ab_testing_models import (
    ExperimentCreateRequest, ExperimentControlRequest, ExperimentAction,
    ExperimentSummaryModel, ExperimentDetailModel, ExperimentListResponse,
    ExperimentProgressModel, StatisticalAnalysisModel,
    experiment_to_summary_model, experiment_to_detail_model
)
from .rest_models import RestApiResponse
from core.ab_testing import (
    ExperimentManager, ConfigurationComparator, StatisticalAnalyzer,
    ExperimentStatus, ExperimentType
)

logger = logging.getLogger(__name__)


class ABTestingApiEndpoints:
    """
    A/B Testing REST API endpoints for experiment management.

    Provides comprehensive endpoints for:
    - Creating and configuring experiments
    - Managing experiment lifecycle
    - Real-time progress monitoring
    - Statistical analysis and results
    """

    def __init__(self, trainer=None):
        """Initialize A/B Testing API with optional trainer reference."""
        self.trainer = trainer
        self.logger = logger

        # Initialize A/B testing components
        self.experiment_manager = ExperimentManager({
            'max_concurrent': 3,
            'results_dir': 'data/ab_test_results'
        })
        self.configuration_comparator = ConfigurationComparator()
        self.statistical_analyzer = StatisticalAnalyzer()

        self.logger.info("A/B Testing API endpoints initialized")

    def list_experiments(self, status_filter: Optional[str] = None) -> Dict[str, Any]:
        """
        List all experiments with optional status filter.

        GET /api/v1/experiments
        GET /api/v1/experiments?status=running
        """
        try:
            # Parse status filter
            status_enum = None
            if status_filter:
                try:
                    status_enum = ExperimentStatus(status_filter)
                except ValueError:
                    return RestApiResponse(
                        success=False,
                        error=f"Invalid status filter: {status_filter}"
                    ).to_dict()

            # Get experiments
            experiments = self.experiment_manager.list_experiments(status_enum)

            # Convert to summary models
            experiment_summaries = [
                experiment_to_summary_model(exp) for exp in experiments
            ]

            # Calculate counts
            total_count = len(experiments)
            active_count = len([e for e in experiments if e.status == ExperimentStatus.RUNNING])
            completed_count = len([e for e in experiments if e.status == ExperimentStatus.COMPLETED])
            failed_count = len([e for e in experiments if e.status == ExperimentStatus.FAILED])

            response = ExperimentListResponse(
                experiments=experiment_summaries,
                total_count=total_count,
                active_count=active_count,
                completed_count=completed_count,
                failed_count=failed_count
            )

            return RestApiResponse(
                success=True,
                data=response.to_dict(),
                message=f"Retrieved {total_count} experiments"
            ).to_dict()

        except Exception as e:
            self.logger.error(f"Failed to list experiments: {e}")
            return RestApiResponse(
                success=False,
                error=f"Failed to list experiments: {str(e)}"
            ).to_dict()

    def create_experiment(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a new A/B testing experiment.

        POST /api/v1/experiments
        """
        try:
            # Parse request
            create_request = ExperimentCreateRequest(**request_data)

            # Convert to experiment configuration
            experiment_config = create_request.to_experiment_config()

            # Create experiment
            experiment_id = self.experiment_manager.create_experiment(experiment_config)

            # Get created experiment
            experiment = self.experiment_manager.get_experiment(experiment_id)
            summary = experiment_to_summary_model(experiment)

            return RestApiResponse(
                success=True,
                data=summary.to_dict(),
                message=f"Experiment '{experiment_config.name}' created successfully"
            ).to_dict()

        except Exception as e:
            self.logger.error(f"Failed to create experiment: {e}")
            return RestApiResponse(
                success=False,
                error=f"Failed to create experiment: {str(e)}"
            ).to_dict()

    def get_experiment(self, experiment_id: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific experiment.

        GET /api/v1/experiments/{experiment_id}
        """
        try:
            experiment = self.experiment_manager.get_experiment(experiment_id)
            if not experiment:
                return RestApiResponse(
                    success=False,
                    error=f"Experiment {experiment_id} not found"
                ).to_dict()

            # Get statistical analysis if experiment is completed
            analysis = None
            if experiment.status == ExperimentStatus.COMPLETED and experiment.result:
                try:
                    analysis = self.statistical_analyzer.analyze_experiment(experiment.result)
                except Exception as e:
                    self.logger.warning(f"Failed to analyze experiment {experiment_id}: {e}")

            # Convert to detailed model
            detail_model = experiment_to_detail_model(experiment, analysis)

            return RestApiResponse(
                success=True,
                data=detail_model.to_dict(),
                message=f"Retrieved experiment {experiment_id}"
            ).to_dict()

        except Exception as e:
            self.logger.error(f"Failed to get experiment {experiment_id}: {e}")
            return RestApiResponse(
                success=False,
                error=f"Failed to get experiment: {str(e)}"
            ).to_dict()

    def control_experiment(self, experiment_id: str, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Control experiment lifecycle (start/stop/pause/resume).

        POST /api/v1/experiments/{experiment_id}/control
        """
        try:
            # Parse control request
            control_request = ExperimentControlRequest(**request_data)
            action = ExperimentAction(control_request.action)

            # Execute control action
            success = False
            message = ""

            if action == ExperimentAction.START:
                success = self.experiment_manager.start_experiment(experiment_id)
                message = f"Experiment {experiment_id} started" if success else "Failed to start experiment"

            elif action == ExperimentAction.STOP:
                success = self.experiment_manager.stop_experiment(experiment_id)
                message = f"Experiment {experiment_id} stopped" if success else "Failed to stop experiment"

            elif action == ExperimentAction.CANCEL:
                success = self.experiment_manager.stop_experiment(experiment_id)
                message = f"Experiment {experiment_id} cancelled" if success else "Failed to cancel experiment"

            else:
                return RestApiResponse(
                    success=False,
                    error=f"Unsupported action: {control_request.action}"
                ).to_dict()

            # Get updated experiment status
            status = self.experiment_manager.get_experiment_status(experiment_id)

            return RestApiResponse(
                success=success,
                data=status,
                message=message
            ).to_dict()

        except Exception as e:
            self.logger.error(f"Failed to control experiment {experiment_id}: {e}")
            return RestApiResponse(
                success=False,
                error=f"Failed to control experiment: {str(e)}"
            ).to_dict()

    def get_experiment_progress(self, experiment_id: str) -> Dict[str, Any]:
        """
        Get real-time progress information for an experiment.

        GET /api/v1/experiments/{experiment_id}/progress
        """
        try:
            experiment = self.experiment_manager.get_experiment(experiment_id)
            if not experiment:
                return RestApiResponse(
                    success=False,
                    error=f"Experiment {experiment_id} not found"
                ).to_dict()

            # Calculate elapsed time
            elapsed_seconds = 0.0
            if experiment.result and experiment.result.start_time:
                elapsed_seconds = time.time() - experiment.result.start_time

            # Estimate completion time
            estimated_completion = None
            if experiment.get_progress() > 0:
                total_estimated = elapsed_seconds / (experiment.get_progress() / 100.0)
                estimated_completion = total_estimated - elapsed_seconds

            # Get latest metrics for each variant
            latest_metrics = {}
            variant_progress = {}

            for variant_name, metrics in experiment.live_metrics.items():
                latest_metrics[f"{variant_name}_reward"] = metrics.get_latest('reward') or 0.0
                latest_metrics[f"{variant_name}_actions_per_second"] = metrics.get_latest('actions_per_second') or 0.0

                variant_progress[variant_name] = {
                    'sample_count': len(metrics.metrics.get('reward', [])),
                    'target_samples': experiment.config.sample_size_per_variant,
                    'completion_percentage': min(100.0,
                        (len(metrics.metrics.get('reward', [])) / experiment.config.sample_size_per_variant) * 100.0)
                }

            progress_model = ExperimentProgressModel(
                experiment_id=experiment_id,
                status=experiment.status.value,
                progress_percentage=experiment.get_progress(),
                current_variant=experiment.current_variant,
                current_run=experiment.current_run,
                total_runs=experiment.total_runs,
                elapsed_seconds=elapsed_seconds,
                estimated_completion_seconds=estimated_completion,
                latest_metrics=latest_metrics,
                variant_progress=variant_progress
            )

            return RestApiResponse(
                success=True,
                data=progress_model.to_dict()
            ).to_dict()

        except Exception as e:
            self.logger.error(f"Failed to get experiment progress {experiment_id}: {e}")
            return RestApiResponse(
                success=False,
                error=f"Failed to get experiment progress: {str(e)}"
            ).to_dict()

    def get_experiment_analysis(self, experiment_id: str) -> Dict[str, Any]:
        """
        Get statistical analysis results for a completed experiment.

        GET /api/v1/experiments/{experiment_id}/analysis
        """
        try:
            experiment = self.experiment_manager.get_experiment(experiment_id)
            if not experiment:
                return RestApiResponse(
                    success=False,
                    error=f"Experiment {experiment_id} not found"
                ).to_dict()

            if experiment.status != ExperimentStatus.COMPLETED:
                return RestApiResponse(
                    success=False,
                    error=f"Experiment {experiment_id} is not completed yet"
                ).to_dict()

            if not experiment.result:
                return RestApiResponse(
                    success=False,
                    error=f"Experiment {experiment_id} has no results"
                ).to_dict()

            # Perform statistical analysis
            analysis = self.statistical_analyzer.analyze_experiment(experiment.result)

            # Convert to API model
            analysis_model = StatisticalAnalysisModel(
                experiment_id=experiment_id,
                primary_metric_results={
                    metric: {
                        'test_type': result.test_type.value,
                        'p_value': result.p_value,
                        'is_significant': result.is_significant,
                        'effect_size': result.effect_size,
                        'confidence_interval': result.confidence_interval,
                        'interpretation': result.interpretation
                    }
                    for metric, result in analysis.primary_metric_results.items()
                },
                secondary_metric_results={
                    metric: {
                        'test_type': result.test_type.value,
                        'p_value': result.p_value,
                        'is_significant': result.is_significant,
                        'effect_size': result.effect_size
                    }
                    for metric, result in analysis.secondary_metric_results.items()
                },
                has_significant_results=analysis.has_significant_results,
                winning_variant=analysis.winning_variant,
                confidence_score=analysis.confidence_score,
                recommendations=analysis.recommendations,
                summary=analysis.summary
            )

            return RestApiResponse(
                success=True,
                data=analysis_model.to_dict(),
                message=f"Statistical analysis for experiment {experiment_id}"
            ).to_dict()

        except Exception as e:
            self.logger.error(f"Failed to analyze experiment {experiment_id}: {e}")
            return RestApiResponse(
                success=False,
                error=f"Failed to analyze experiment: {str(e)}"
            ).to_dict()

    def list_experiment_templates(self) -> Dict[str, Any]:
        """
        List available experiment templates.

        GET /api/v1/experiments/templates
        """
        try:
            # Define available templates
            templates = [
                {
                    'template_id': 'battle_strategy_comparison',
                    'name': 'Battle Strategy Comparison',
                    'description': 'Compare different battle strategies (aggressive, defensive, balanced)',
                    'experiment_type': 'plugin_comparison',
                    'category': 'battle',
                    'difficulty_level': 'beginner',
                    'estimated_runtime_minutes': 45,
                    'suggested_sample_size': 30,
                    'suggested_metrics': ['total_reward', 'battle_win_rate']
                },
                {
                    'template_id': 'exploration_pattern_comparison',
                    'name': 'Exploration Pattern Comparison',
                    'description': 'Compare exploration patterns (systematic, spiral, random, wall-following)',
                    'experiment_type': 'plugin_comparison',
                    'category': 'exploration',
                    'difficulty_level': 'beginner',
                    'estimated_runtime_minutes': 60,
                    'suggested_sample_size': 25,
                    'suggested_metrics': ['total_reward', 'exploration_coverage']
                },
                {
                    'template_id': 'multi_agent_comparison',
                    'name': 'Multi-Agent Strategy Comparison',
                    'description': 'Compare different multi-agent coordination strategies',
                    'experiment_type': 'agent_comparison',
                    'category': 'coordination',
                    'difficulty_level': 'intermediate',
                    'estimated_runtime_minutes': 75,
                    'suggested_sample_size': 20,
                    'suggested_metrics': ['total_reward', 'actions_per_second']
                },
                {
                    'template_id': 'hybrid_configuration_comparison',
                    'name': 'Hybrid Configuration Comparison',
                    'description': 'Compare combinations of plugins and agent strategies',
                    'experiment_type': 'multi_factor',
                    'category': 'advanced',
                    'difficulty_level': 'advanced',
                    'estimated_runtime_minutes': 90,
                    'suggested_sample_size': 35,
                    'suggested_metrics': ['total_reward', 'battle_win_rate', 'actions_per_second']
                }
            ]

            return RestApiResponse(
                success=True,
                data={'templates': templates},
                message=f"Retrieved {len(templates)} experiment templates"
            ).to_dict()

        except Exception as e:
            self.logger.error(f"Failed to list experiment templates: {e}")
            return RestApiResponse(
                success=False,
                error=f"Failed to list experiment templates: {str(e)}"
            ).to_dict()

    def create_experiment_from_template(self, template_id: str, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create experiment from a predefined template.

        POST /api/v1/experiments/templates/{template_id}
        """
        try:
            # Get template configuration
            if template_id == 'battle_strategy_comparison':
                experiment_config = self.configuration_comparator.create_battle_strategy_comparison()
            elif template_id == 'exploration_pattern_comparison':
                experiment_config = self.configuration_comparator.create_exploration_pattern_comparison()
            elif template_id == 'multi_agent_comparison':
                experiment_config = self.configuration_comparator.create_multi_agent_comparison()
            elif template_id == 'hybrid_configuration_comparison':
                experiment_config = self.configuration_comparator.create_hybrid_comparison()
            else:
                return RestApiResponse(
                    success=False,
                    error=f"Unknown template: {template_id}"
                ).to_dict()

            # Apply customizations from request
            if 'name' in request_data:
                experiment_config.name = request_data['name']
            if 'sample_size_per_variant' in request_data:
                experiment_config.sample_size_per_variant = request_data['sample_size_per_variant']
            if 'max_runtime_seconds' in request_data:
                experiment_config.max_runtime_seconds = request_data['max_runtime_seconds']

            # Create experiment
            experiment_id = self.experiment_manager.create_experiment(experiment_config)

            # Get created experiment
            experiment = self.experiment_manager.get_experiment(experiment_id)
            summary = experiment_to_summary_model(experiment)

            return RestApiResponse(
                success=True,
                data=summary.to_dict(),
                message=f"Experiment created from template '{template_id}'"
            ).to_dict()

        except Exception as e:
            self.logger.error(f"Failed to create experiment from template {template_id}: {e}")
            return RestApiResponse(
                success=False,
                error=f"Failed to create experiment from template: {str(e)}"
            ).to_dict()

    def get_manager_stats(self) -> Dict[str, Any]:
        """
        Get A/B testing manager statistics.

        GET /api/v1/experiments/stats
        """
        try:
            stats = self.experiment_manager.get_summary_stats()

            return RestApiResponse(
                success=True,
                data=stats,
                message="A/B testing manager statistics"
            ).to_dict()

        except Exception as e:
            self.logger.error(f"Failed to get manager stats: {e}")
            return RestApiResponse(
                success=False,
                error=f"Failed to get manager stats: {str(e)}"
            ).to_dict()

    def cleanup(self):
        """Clean up A/B testing resources"""
        try:
            if self.experiment_manager:
                self.experiment_manager.cleanup()
            self.logger.info("A/B Testing API cleanup completed")
        except Exception as e:
            self.logger.error(f"Failed to cleanup A/B Testing API: {e}")

    def get_api_documentation(self) -> Dict[str, Any]:
        """
        Get comprehensive API documentation for A/B testing endpoints.

        GET /api/v1/experiments/docs
        """
        documentation = {
            'title': 'A/B Testing API Documentation',
            'version': '1.0.0',
            'description': 'Comprehensive A/B testing API for Pokemon Crystal RL platform',
            'endpoints': [
                {
                    'path': '/api/v1/experiments',
                    'method': 'GET',
                    'description': 'List all experiments',
                    'parameters': {'status': 'Optional status filter (running, completed, failed, etc.)'}
                },
                {
                    'path': '/api/v1/experiments',
                    'method': 'POST',
                    'description': 'Create new experiment',
                    'body': 'ExperimentCreateRequest'
                },
                {
                    'path': '/api/v1/experiments/{id}',
                    'method': 'GET',
                    'description': 'Get experiment details'
                },
                {
                    'path': '/api/v1/experiments/{id}/control',
                    'method': 'POST',
                    'description': 'Control experiment (start/stop/pause)',
                    'body': 'ExperimentControlRequest'
                },
                {
                    'path': '/api/v1/experiments/{id}/progress',
                    'method': 'GET',
                    'description': 'Get real-time progress'
                },
                {
                    'path': '/api/v1/experiments/{id}/analysis',
                    'method': 'GET',
                    'description': 'Get statistical analysis (completed experiments only)'
                },
                {
                    'path': '/api/v1/experiments/templates',
                    'method': 'GET',
                    'description': 'List experiment templates'
                },
                {
                    'path': '/api/v1/experiments/templates/{template_id}',
                    'method': 'POST',
                    'description': 'Create experiment from template'
                },
                {
                    'path': '/api/v1/experiments/stats',
                    'method': 'GET',
                    'description': 'Get A/B testing manager statistics'
                }
            ]
        }

        return RestApiResponse(
            success=True,
            data=documentation,
            message="A/B Testing API documentation"
        ).to_dict()