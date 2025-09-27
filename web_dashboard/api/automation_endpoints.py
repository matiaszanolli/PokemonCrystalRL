"""
Automation REST API Endpoints

REST API endpoints for automated experiment scheduling, template management,
and workflow automation in the A/B testing framework.
"""

import logging
import json
from typing import Dict, Any, List, Optional
from datetime import datetime

from core.ab_testing.experiment_scheduler import ExperimentScheduler, ScheduleConfig, ScheduleType, ScheduleStatus
from core.ab_testing.automation_templates import AutomationTemplates
from core.ab_testing.experiment_models import ExperimentConfig, ExperimentType
from .ab_testing_models import ExperimentCreateRequest


class AutomationEndpoints:
    """
    REST API endpoints for experiment automation and scheduling.

    Provides endpoints for:
    - Experiment scheduling and queuing
    - Automation template management
    - Workflow creation and monitoring
    - Automated execution control
    """

    def __init__(self, experiment_scheduler: ExperimentScheduler):
        """Initialize automation endpoints"""
        self.experiment_scheduler = experiment_scheduler
        self.automation_templates = AutomationTemplates()
        self.logger = logging.getLogger("AutomationEndpoints")

    def schedule_experiment(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Schedule an experiment for automated execution.

        POST /api/v1/automation/schedule
        """
        try:
            # Extract experiment configuration
            experiment_data = request_data.get('experiment', {})
            schedule_data = request_data.get('schedule', {})

            # Create experiment configuration
            if 'experiment_type' in experiment_data:
                experiment_data['experiment_type'] = ExperimentType(experiment_data['experiment_type'])

            experiment_config = ExperimentConfig(**experiment_data)

            # Create schedule configuration
            schedule_type = ScheduleType(schedule_data.get('schedule_type', 'immediate'))
            schedule_config = ScheduleConfig(
                schedule_type=schedule_type,
                start_time=datetime.fromisoformat(schedule_data['start_time']) if schedule_data.get('start_time') else None,
                delay_seconds=schedule_data.get('delay_seconds'),
                interval_seconds=schedule_data.get('interval_seconds'),
                max_runs=schedule_data.get('max_runs'),
                condition_check=schedule_data.get('condition_check'),
                auto_analyze=schedule_data.get('auto_analyze', True),
                auto_archive=schedule_data.get('auto_archive', True),
                max_concurrent=schedule_data.get('max_concurrent', 1),
                retry_on_failure=schedule_data.get('retry_on_failure', True),
                max_retries=schedule_data.get('max_retries', 3),
                depends_on=schedule_data.get('depends_on', [])
            )

            # Schedule the experiment
            schedule_id = self.experiment_scheduler.schedule_experiment(experiment_config, schedule_config)

            return {
                'success': True,
                'data': {
                    'schedule_id': schedule_id,
                    'message': 'Experiment scheduled successfully'
                }
            }

        except Exception as e:
            self.logger.error(f"Failed to schedule experiment: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def cancel_scheduled_experiment(self, schedule_id: str) -> Dict[str, Any]:
        """
        Cancel a scheduled experiment.

        DELETE /api/v1/automation/schedule/{schedule_id}
        """
        try:
            success = self.experiment_scheduler.cancel_scheduled_experiment(schedule_id)

            if success:
                return {
                    'success': True,
                    'data': {
                        'message': 'Scheduled experiment cancelled successfully'
                    }
                }
            else:
                return {
                    'success': False,
                    'error': 'Failed to cancel scheduled experiment - may not exist or already running'
                }

        except Exception as e:
            self.logger.error(f"Failed to cancel scheduled experiment {schedule_id}: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def list_scheduled_experiments(self, status_filter: Optional[str] = None) -> Dict[str, Any]:
        """
        List all scheduled experiments with optional status filtering.

        GET /api/v1/automation/schedule
        """
        try:
            # Parse status filter
            status_enum = None
            if status_filter:
                try:
                    status_enum = ScheduleStatus(status_filter.lower())
                except ValueError:
                    return {
                        'success': False,
                        'error': f'Invalid status filter: {status_filter}'
                    }

            # Get scheduled experiments
            scheduled_experiments = self.experiment_scheduler.get_scheduled_experiments(status_enum)

            # Convert to API response format
            experiments_data = []
            for scheduled_exp in scheduled_experiments:
                exp_data = {
                    'schedule_id': scheduled_exp.schedule_id,
                    'experiment_name': scheduled_exp.experiment_config.name,
                    'experiment_type': scheduled_exp.experiment_config.experiment_type.value,
                    'status': scheduled_exp.status.value,
                    'schedule_type': scheduled_exp.schedule_config.schedule_type.value,
                    'created_time': scheduled_exp.created_time.isoformat(),
                    'scheduled_time': scheduled_exp.scheduled_time.isoformat() if scheduled_exp.scheduled_time else None,
                    'started_time': scheduled_exp.started_time.isoformat() if scheduled_exp.started_time else None,
                    'completed_time': scheduled_exp.completed_time.isoformat() if scheduled_exp.completed_time else None,
                    'attempt_count': scheduled_exp.attempt_count,
                    'auto_analyze': scheduled_exp.schedule_config.auto_analyze,
                    'auto_archive': scheduled_exp.schedule_config.auto_archive,
                    'analysis_complete': scheduled_exp.analysis_complete,
                    'results_archived': scheduled_exp.results_archived
                }

                if scheduled_exp.experiment_id:
                    exp_data['experiment_id'] = scheduled_exp.experiment_id

                if scheduled_exp.error_message:
                    exp_data['error_message'] = scheduled_exp.error_message

                experiments_data.append(exp_data)

            # Calculate summary stats
            total_count = len(experiments_data)
            status_counts = {}
            for exp in experiments_data:
                status = exp['status']
                status_counts[status] = status_counts.get(status, 0) + 1

            return {
                'success': True,
                'data': {
                    'scheduled_experiments': experiments_data,
                    'total_count': total_count,
                    'status_distribution': status_counts,
                    'automation_active': self.experiment_scheduler.running
                }
            }

        except Exception as e:
            self.logger.error(f"Failed to list scheduled experiments: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def get_schedule_status(self, schedule_id: str) -> Dict[str, Any]:
        """
        Get detailed status of a specific scheduled experiment.

        GET /api/v1/automation/schedule/{schedule_id}
        """
        try:
            status = self.experiment_scheduler.get_schedule_status(schedule_id)

            if status:
                return {
                    'success': True,
                    'data': status
                }
            else:
                return {
                    'success': False,
                    'error': 'Scheduled experiment not found'
                }

        except Exception as e:
            self.logger.error(f"Failed to get schedule status for {schedule_id}: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def get_automation_templates(self) -> Dict[str, Any]:
        """
        Get list of available automation templates.

        GET /api/v1/automation/templates
        """
        try:
            templates = self.automation_templates.get_available_templates()

            template_list = []
            for template_id, description in templates.items():
                parameters = self.automation_templates.get_template_parameters(template_id)

                template_list.append({
                    'template_id': template_id,
                    'name': template_id.replace('_', ' ').title(),
                    'description': description,
                    'parameters': parameters
                })

            return {
                'success': True,
                'data': {
                    'templates': template_list
                }
            }

        except Exception as e:
            self.logger.error(f"Failed to get automation templates: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def create_from_template(self, template_id: str, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create scheduled experiments from a template.

        POST /api/v1/automation/templates/{template_id}
        """
        try:
            # Get template parameters
            parameters = request_data.get('parameters', {})

            # Create workflow based on template
            if template_id == 'continuous_optimization':
                workflows = self.automation_templates.create_continuous_optimization_workflow(
                    base_name=parameters.get('base_name', 'Continuous Optimization'),
                    test_interval_hours=parameters.get('test_interval_hours', 6),
                    sample_size=parameters.get('sample_size', 50)
                )
            elif template_id == 'regression_testing':
                workflows = self.automation_templates.create_regression_testing_suite(
                    baseline_config=parameters.get('baseline_config'),
                    test_configs=parameters.get('test_configs'),
                    sample_size=parameters.get('sample_size', 30)
                )
            elif template_id == 'hyperparameter_sweep':
                workflows = self.automation_templates.create_hyperparameter_sweep(
                    parameter_ranges=parameters.get('parameter_ranges'),
                    base_config=parameters.get('base_config'),
                    sample_size=parameters.get('sample_size', 20)
                )
            elif template_id == 'performance_monitoring':
                workflows = self.automation_templates.create_performance_monitoring_workflow(
                    check_interval_hours=parameters.get('check_interval_hours', 12),
                    performance_threshold=parameters.get('performance_threshold', 0.95)
                )
            elif template_id == 'weekend_stress_test':
                workflow = self.automation_templates.create_weekend_stress_test(
                    stress_duration_hours=parameters.get('stress_duration_hours', 48),
                    sample_size=parameters.get('sample_size', 100)
                )
                workflows = [workflow]
            elif template_id == 'custom_workflow':
                workflows = self.automation_templates.create_custom_workflow(
                    workflow_name=parameters.get('workflow_name'),
                    experiments=parameters.get('experiments'),
                    dependencies=parameters.get('dependencies'),
                    intervals=parameters.get('intervals')
                )
            else:
                return {
                    'success': False,
                    'error': f'Unknown template: {template_id}'
                }

            # Schedule all workflows
            schedule_ids = []
            for workflow in workflows:
                schedule_id = self.experiment_scheduler.schedule_experiment(
                    workflow['experiment_config'],
                    workflow['schedule_config']
                )
                schedule_ids.append({
                    'schedule_id': schedule_id,
                    'name': workflow['name'],
                    'description': workflow['description']
                })

            return {
                'success': True,
                'data': {
                    'template_id': template_id,
                    'scheduled_experiments': schedule_ids,
                    'message': f'Created {len(schedule_ids)} scheduled experiments from template'
                }
            }

        except Exception as e:
            self.logger.error(f"Failed to create from template {template_id}: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def start_automation(self) -> Dict[str, Any]:
        """
        Start the automated experiment scheduler.

        POST /api/v1/automation/start
        """
        try:
            if not self.experiment_scheduler.running:
                self.experiment_scheduler.start()
                return {
                    'success': True,
                    'data': {
                        'message': 'Automation scheduler started successfully'
                    }
                }
            else:
                return {
                    'success': True,
                    'data': {
                        'message': 'Automation scheduler is already running'
                    }
                }

        except Exception as e:
            self.logger.error(f"Failed to start automation: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def stop_automation(self) -> Dict[str, Any]:
        """
        Stop the automated experiment scheduler.

        POST /api/v1/automation/stop
        """
        try:
            if self.experiment_scheduler.running:
                self.experiment_scheduler.stop()
                return {
                    'success': True,
                    'data': {
                        'message': 'Automation scheduler stopped successfully'
                    }
                }
            else:
                return {
                    'success': True,
                    'data': {
                        'message': 'Automation scheduler is already stopped'
                    }
                }

        except Exception as e:
            self.logger.error(f"Failed to stop automation: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def get_automation_stats(self) -> Dict[str, Any]:
        """
        Get automation system statistics.

        GET /api/v1/automation/stats
        """
        try:
            stats = self.experiment_scheduler.get_automation_stats()

            return {
                'success': True,
                'data': stats
            }

        except Exception as e:
            self.logger.error(f"Failed to get automation stats: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def get_queue_status(self) -> Dict[str, Any]:
        """
        Get current queue status and running experiments.

        GET /api/v1/automation/queue
        """
        try:
            stats = self.experiment_scheduler.get_automation_stats()

            # Get currently running experiments
            running_experiments = []
            for schedule_id, experiment_id in self.experiment_scheduler.running_experiments.items():
                schedule_status = self.experiment_scheduler.get_schedule_status(schedule_id)
                if schedule_status:
                    running_experiments.append({
                        'schedule_id': schedule_id,
                        'experiment_id': experiment_id,
                        'experiment_name': schedule_status.get('experiment_name'),
                        'started_time': schedule_status.get('started_time')
                    })

            return {
                'success': True,
                'data': {
                    'queue_size': stats['queue_size'],
                    'currently_running': stats['currently_running'],
                    'running_experiments': running_experiments,
                    'automation_active': stats['automation_active']
                }
            }

        except Exception as e:
            self.logger.error(f"Failed to get queue status: {e}")
            return {
                'success': False,
                'error': str(e)
            }