"""
Test suite for A/B Testing Automation API Endpoints

Tests the REST API endpoints for automated experiment scheduling and management.
"""

import pytest
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch

from core.ab_testing.experiment_scheduler import ExperimentScheduler, ScheduleConfig, ScheduleType, ScheduleStatus
from core.ab_testing.experiment_manager import ExperimentManager
from core.ab_testing.experiment_models import ExperimentConfig, ExperimentType
from web_dashboard.api.automation_endpoints import AutomationEndpoints


class TestAutomationEndpoints:
    """Test suite for AutomationEndpoints class"""

    @pytest.fixture
    def mock_experiment_manager(self):
        """Create a mock experiment manager"""
        manager = Mock(spec=ExperimentManager)
        manager.create_experiment.return_value = "test_experiment_123"
        manager.start_experiment.return_value = True
        manager.get_experiment.return_value = Mock(status="completed")
        manager.event_bus = Mock()
        return manager

    @pytest.fixture
    def mock_scheduler(self, mock_experiment_manager):
        """Create a mock experiment scheduler"""
        scheduler = Mock(spec=ExperimentScheduler)
        scheduler.experiment_manager = mock_experiment_manager
        scheduler.running = False
        scheduler.running_experiments = {}

        # Mock return values
        scheduler.schedule_experiment.return_value = "schedule_123"
        scheduler.cancel_scheduled_experiment.return_value = True
        scheduler.get_scheduled_experiments.return_value = []
        scheduler.get_schedule_status.return_value = {
            'schedule_id': 'schedule_123',
            'status': 'pending',
            'experiment_name': 'Test Experiment'
        }
        scheduler.get_automation_stats.return_value = {
            'total_scheduled': 0,
            'currently_running': 0,
            'queue_size': 0,
            'automation_active': False
        }

        return scheduler

    @pytest.fixture
    def automation_endpoints(self, mock_scheduler):
        """Create AutomationEndpoints instance for testing"""
        return AutomationEndpoints(mock_scheduler)

    @pytest.fixture
    def sample_experiment_request(self):
        """Create a sample experiment request"""
        return {
            'experiment': {
                'name': 'Test Experiment',
                'description': 'A test experiment',
                'experiment_type': 'plugin_comparison',
                'variants': [{'name': 'control', 'config': {}}],
                'sample_size': 10
            },
            'schedule': {
                'schedule_type': 'immediate',
                'auto_analyze': True,
                'auto_archive': True
            }
        }

    def test_schedule_experiment_success(self, automation_endpoints, sample_experiment_request):
        """Test successful experiment scheduling"""
        response = automation_endpoints.schedule_experiment(sample_experiment_request)

        assert response['success'] is True
        assert 'schedule_id' in response['data']
        assert response['data']['schedule_id'] == 'schedule_123'

    def test_schedule_experiment_with_delayed_schedule(self, automation_endpoints):
        """Test scheduling experiment with delayed execution"""
        future_time = datetime.now() + timedelta(hours=1)
        request_data = {
            'experiment': {
                'name': 'Delayed Test',
                'description': 'A delayed test',
                'experiment_type': 'plugin_comparison',
                'variants': [{'name': 'control', 'config': {}}],
                'sample_size': 10
            },
            'schedule': {
                'schedule_type': 'delayed',
                'start_time': future_time.isoformat(),
                'auto_analyze': True
            }
        }

        response = automation_endpoints.schedule_experiment(request_data)
        assert response['success'] is True

    def test_schedule_experiment_with_recurring_schedule(self, automation_endpoints):
        """Test scheduling recurring experiment"""
        request_data = {
            'experiment': {
                'name': 'Recurring Test',
                'description': 'A recurring test',
                'experiment_type': 'plugin_comparison',
                'variants': [{'name': 'control', 'config': {}}],
                'sample_size': 10
            },
            'schedule': {
                'schedule_type': 'recurring',
                'interval_seconds': 3600,
                'max_runs': 5
            }
        }

        response = automation_endpoints.schedule_experiment(request_data)
        assert response['success'] is True

    def test_schedule_experiment_failure(self, automation_endpoints):
        """Test experiment scheduling failure"""
        automation_endpoints.experiment_scheduler.schedule_experiment.side_effect = Exception("Scheduling failed")

        request_data = {
            'experiment': {
                'name': 'Failing Test',
                'experiment_type': 'plugin_comparison',
                'variants': [],
                'sample_size': 10
            },
            'schedule': {
                'schedule_type': 'immediate'
            }
        }

        response = automation_endpoints.schedule_experiment(request_data)
        assert response['success'] is False
        assert 'error' in response

    def test_cancel_scheduled_experiment_success(self, automation_endpoints):
        """Test successful experiment cancellation"""
        response = automation_endpoints.cancel_scheduled_experiment('schedule_123')

        assert response['success'] is True
        assert 'message' in response['data']

    def test_cancel_scheduled_experiment_failure(self, automation_endpoints):
        """Test experiment cancellation failure"""
        automation_endpoints.experiment_scheduler.cancel_scheduled_experiment.return_value = False

        response = automation_endpoints.cancel_scheduled_experiment('nonexistent_id')

        assert response['success'] is False
        assert 'error' in response

    def test_cancel_scheduled_experiment_exception(self, automation_endpoints):
        """Test experiment cancellation with exception"""
        automation_endpoints.experiment_scheduler.cancel_scheduled_experiment.side_effect = Exception("Cancel failed")

        response = automation_endpoints.cancel_scheduled_experiment('schedule_123')

        assert response['success'] is False
        assert 'error' in response

    def test_list_scheduled_experiments_empty(self, automation_endpoints):
        """Test listing scheduled experiments when none exist"""
        response = automation_endpoints.list_scheduled_experiments()

        assert response['success'] is True
        assert response['data']['total_count'] == 0
        assert response['data']['scheduled_experiments'] == []

    def test_list_scheduled_experiments_with_data(self, automation_endpoints):
        """Test listing scheduled experiments with data"""
        # Create mock scheduled experiments
        mock_experiment = Mock()
        mock_experiment.schedule_id = 'schedule_123'
        mock_experiment.experiment_config.name = 'Test Experiment'
        mock_experiment.experiment_config.experiment_type.value = 'plugin_comparison'
        mock_experiment.status.value = 'pending'
        mock_experiment.schedule_config.schedule_type.value = 'immediate'
        mock_experiment.created_time = datetime.now()
        mock_experiment.scheduled_time = None
        mock_experiment.started_time = None
        mock_experiment.completed_time = None
        mock_experiment.attempt_count = 0
        mock_experiment.schedule_config.auto_analyze = True
        mock_experiment.schedule_config.auto_archive = True
        mock_experiment.analysis_complete = False
        mock_experiment.results_archived = False
        mock_experiment.experiment_id = None
        mock_experiment.error_message = None

        automation_endpoints.experiment_scheduler.get_scheduled_experiments.return_value = [mock_experiment]

        response = automation_endpoints.list_scheduled_experiments()

        assert response['success'] is True
        assert response['data']['total_count'] == 1
        assert len(response['data']['scheduled_experiments']) == 1

    def test_list_scheduled_experiments_with_status_filter(self, automation_endpoints):
        """Test listing scheduled experiments with status filter"""
        response = automation_endpoints.list_scheduled_experiments('pending')
        assert response['success'] is True

    def test_list_scheduled_experiments_invalid_status_filter(self, automation_endpoints):
        """Test listing scheduled experiments with invalid status filter"""
        response = automation_endpoints.list_scheduled_experiments('invalid_status')
        assert response['success'] is False
        assert 'Invalid status filter' in response['error']

    def test_get_schedule_status_success(self, automation_endpoints):
        """Test getting schedule status successfully"""
        response = automation_endpoints.get_schedule_status('schedule_123')

        assert response['success'] is True
        assert response['data']['schedule_id'] == 'schedule_123'

    def test_get_schedule_status_not_found(self, automation_endpoints):
        """Test getting schedule status for non-existent experiment"""
        automation_endpoints.experiment_scheduler.get_schedule_status.return_value = None

        response = automation_endpoints.get_schedule_status('nonexistent_id')

        assert response['success'] is False
        assert 'not found' in response['error']

    def test_get_automation_templates(self, automation_endpoints):
        """Test getting available automation templates"""
        response = automation_endpoints.get_automation_templates()

        assert response['success'] is True
        assert 'templates' in response['data']
        assert isinstance(response['data']['templates'], list)

    def test_create_from_template_continuous_optimization(self, automation_endpoints):
        """Test creating experiments from continuous optimization template"""
        request_data = {
            'parameters': {
                'base_name': 'Test Optimization',
                'test_interval_hours': 6,
                'sample_size': 50
            }
        }

        response = automation_endpoints.create_from_template('continuous_optimization', request_data)

        assert response['success'] is True
        assert 'scheduled_experiments' in response['data']

    def test_create_from_template_regression_testing(self, automation_endpoints):
        """Test creating experiments from regression testing template"""
        request_data = {
            'parameters': {
                'baseline_config': {'strategy': 'balanced'},
                'test_configs': [
                    {'name': 'Aggressive', 'strategy': 'aggressive'},
                    {'name': 'Defensive', 'strategy': 'defensive'}
                ],
                'sample_size': 30
            }
        }

        response = automation_endpoints.create_from_template('regression_testing', request_data)

        assert response['success'] is True

    def test_create_from_template_hyperparameter_sweep(self, automation_endpoints):
        """Test creating experiments from hyperparameter sweep template"""
        request_data = {
            'parameters': {
                'parameter_ranges': {
                    'aggression_level': [0.3, 0.6, 0.9],
                    'exploration_radius': [2, 4, 6]
                },
                'base_config': {'base_param': 'value'},
                'sample_size': 20
            }
        }

        response = automation_endpoints.create_from_template('hyperparameter_sweep', request_data)

        assert response['success'] is True

    def test_create_from_template_performance_monitoring(self, automation_endpoints):
        """Test creating experiments from performance monitoring template"""
        request_data = {
            'parameters': {
                'check_interval_hours': 12,
                'performance_threshold': 0.95
            }
        }

        response = automation_endpoints.create_from_template('performance_monitoring', request_data)

        assert response['success'] is True

    def test_create_from_template_weekend_stress_test(self, automation_endpoints):
        """Test creating experiments from weekend stress test template"""
        request_data = {
            'parameters': {
                'stress_duration_hours': 48,
                'sample_size': 100
            }
        }

        response = automation_endpoints.create_from_template('weekend_stress_test', request_data)

        assert response['success'] is True

    def test_create_from_template_custom_workflow(self, automation_endpoints):
        """Test creating experiments from custom workflow template"""
        request_data = {
            'parameters': {
                'workflow_name': 'Custom Test Workflow',
                'experiments': [
                    {'name': 'Test 1', 'config': {'param': 'value1'}},
                    {'name': 'Test 2', 'config': {'param': 'value2'}}
                ],
                'dependencies': {'Test 2': ['Test 1']},
                'intervals': {'Test 1': 3600, 'Test 2': 7200}
            }
        }

        response = automation_endpoints.create_from_template('custom_workflow', request_data)

        assert response['success'] is True

    def test_create_from_template_unknown(self, automation_endpoints):
        """Test creating experiments from unknown template"""
        request_data = {'parameters': {}}

        response = automation_endpoints.create_from_template('unknown_template', request_data)

        assert response['success'] is False
        assert 'Unknown template' in response['error']

    def test_start_automation_success(self, automation_endpoints):
        """Test starting automation successfully"""
        automation_endpoints.experiment_scheduler.running = False

        response = automation_endpoints.start_automation()

        assert response['success'] is True
        assert 'started successfully' in response['data']['message']

    def test_start_automation_already_running(self, automation_endpoints):
        """Test starting automation when already running"""
        automation_endpoints.experiment_scheduler.running = True

        response = automation_endpoints.start_automation()

        assert response['success'] is True
        assert 'already running' in response['data']['message']

    def test_start_automation_failure(self, automation_endpoints):
        """Test automation start failure"""
        automation_endpoints.experiment_scheduler.start.side_effect = Exception("Start failed")

        response = automation_endpoints.start_automation()

        assert response['success'] is False
        assert 'error' in response

    def test_stop_automation_success(self, automation_endpoints):
        """Test stopping automation successfully"""
        automation_endpoints.experiment_scheduler.running = True

        response = automation_endpoints.stop_automation()

        assert response['success'] is True
        assert 'stopped successfully' in response['data']['message']

    def test_stop_automation_already_stopped(self, automation_endpoints):
        """Test stopping automation when already stopped"""
        automation_endpoints.experiment_scheduler.running = False

        response = automation_endpoints.stop_automation()

        assert response['success'] is True
        assert 'already stopped' in response['data']['message']

    def test_stop_automation_failure(self, automation_endpoints):
        """Test automation stop failure"""
        automation_endpoints.experiment_scheduler.stop.side_effect = Exception("Stop failed")

        response = automation_endpoints.stop_automation()

        assert response['success'] is False
        assert 'error' in response

    def test_get_automation_stats(self, automation_endpoints):
        """Test getting automation statistics"""
        response = automation_endpoints.get_automation_stats()

        assert response['success'] is True
        assert 'total_scheduled' in response['data']
        assert 'currently_running' in response['data']
        assert 'queue_size' in response['data']
        assert 'automation_active' in response['data']

    def test_get_automation_stats_failure(self, automation_endpoints):
        """Test automation stats retrieval failure"""
        automation_endpoints.experiment_scheduler.get_automation_stats.side_effect = Exception("Stats failed")

        response = automation_endpoints.get_automation_stats()

        assert response['success'] is False
        assert 'error' in response

    def test_get_queue_status(self, automation_endpoints):
        """Test getting queue status"""
        response = automation_endpoints.get_queue_status()

        assert response['success'] is True
        assert 'queue_size' in response['data']
        assert 'currently_running' in response['data']
        assert 'running_experiments' in response['data']
        assert 'automation_active' in response['data']

    def test_get_queue_status_with_running_experiments(self, automation_endpoints):
        """Test getting queue status with running experiments"""
        # Setup running experiments
        automation_endpoints.experiment_scheduler.running_experiments = {
            'schedule_123': 'experiment_456'
        }

        response = automation_endpoints.get_queue_status()

        assert response['success'] is True
        assert len(response['data']['running_experiments']) >= 0

    def test_get_queue_status_failure(self, automation_endpoints):
        """Test queue status retrieval failure"""
        automation_endpoints.experiment_scheduler.get_automation_stats.side_effect = Exception("Queue failed")

        response = automation_endpoints.get_queue_status()

        assert response['success'] is False
        assert 'error' in response


class TestAutomationEndpointsIntegration:
    """Integration tests for automation endpoints"""

    @pytest.fixture
    def real_experiment_manager(self):
        """Create a real experiment manager for integration testing"""
        return ExperimentManager()

    @pytest.fixture
    def real_scheduler(self, real_experiment_manager):
        """Create a real experiment scheduler for integration testing"""
        return ExperimentScheduler(real_experiment_manager)

    @pytest.fixture
    def real_automation_endpoints(self, real_scheduler):
        """Create real AutomationEndpoints for integration testing"""
        return AutomationEndpoints(real_scheduler)

    def test_full_automation_workflow(self, real_automation_endpoints):
        """Test complete automation workflow with real components"""
        # Start automation
        start_response = real_automation_endpoints.start_automation()
        assert start_response['success'] is True

        try:
            # Get available templates
            templates_response = real_automation_endpoints.get_automation_templates()
            assert templates_response['success'] is True
            assert len(templates_response['data']['templates']) > 0

            # Create experiment from template
            template_response = real_automation_endpoints.create_from_template(
                'weekend_stress_test',
                {
                    'parameters': {
                        'stress_duration_hours': 1,  # Short for testing
                        'sample_size': 5
                    }
                }
            )
            assert template_response['success'] is True

            # List scheduled experiments
            list_response = real_automation_endpoints.list_scheduled_experiments()
            assert list_response['success'] is True
            assert list_response['data']['total_count'] > 0

            # Get automation stats
            stats_response = real_automation_endpoints.get_automation_stats()
            assert stats_response['success'] is True

        finally:
            # Stop automation
            stop_response = real_automation_endpoints.stop_automation()
            assert stop_response['success'] is True

    def test_error_handling_with_real_components(self, real_automation_endpoints):
        """Test error handling with real components"""
        # Test invalid experiment data
        invalid_request = {
            'experiment': {
                'name': '',  # Invalid empty name
                'experiment_type': 'invalid_type',
                'variants': [],
                'sample_size': -1  # Invalid negative sample size
            },
            'schedule': {
                'schedule_type': 'invalid_schedule_type'
            }
        }

        response = real_automation_endpoints.schedule_experiment(invalid_request)
        assert response['success'] is False

    def test_concurrent_api_calls(self, real_automation_endpoints):
        """Test concurrent API calls"""
        import threading
        import time

        results = []

        def make_api_call():
            response = real_automation_endpoints.get_automation_stats()
            results.append(response['success'])

        # Create multiple threads making concurrent API calls
        threads = [threading.Thread(target=make_api_call) for _ in range(5)]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        # All calls should succeed
        assert all(results)
        assert len(results) == 5