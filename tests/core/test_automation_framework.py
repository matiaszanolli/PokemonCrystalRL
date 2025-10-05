"""
Test suite for A/B Testing Automation Framework

Tests the automated experiment scheduling, execution, and template systems.
"""

import pytest
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, Any
from unittest.mock import Mock, patch, MagicMock

from core.ab_testing.experiment_scheduler import (
    ExperimentScheduler, ScheduleConfig, ScheduleType, ScheduleStatus, ScheduledExperiment
)
from core.ab_testing.automation_templates import AutomationTemplates
from core.ab_testing.experiment_manager import ExperimentManager
from core.ab_testing.experiment_models import ExperimentConfig, ExperimentType


class TestExperimentScheduler:
    """Test suite for ExperimentScheduler class"""

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
    def scheduler(self, mock_experiment_manager):
        """Create an ExperimentScheduler instance for testing"""
        return ExperimentScheduler(mock_experiment_manager)

    @pytest.fixture
    def sample_experiment_config(self):
        """Create a sample experiment configuration"""
        return ExperimentConfig(
            name="Test Experiment",
            description="A test experiment",
            experiment_type=ExperimentType.PLUGIN_COMPARISON,
            variants={"control": {"config": {}}},
            sample_size_per_variant=10
        )

    @pytest.fixture
    def sample_schedule_config(self):
        """Create a sample schedule configuration"""
        return ScheduleConfig(
            schedule_type=ScheduleType.IMMEDIATE,
            auto_analyze=True,
            auto_archive=True
        )

    def test_scheduler_initialization(self, mock_experiment_manager):
        """Test ExperimentScheduler initialization"""
        scheduler = ExperimentScheduler(mock_experiment_manager)

        assert scheduler.experiment_manager == mock_experiment_manager
        assert not scheduler.running
        assert len(scheduler.scheduled_experiments) == 0
        assert len(scheduler.running_experiments) == 0

    def test_schedule_immediate_experiment(self, scheduler, sample_experiment_config, sample_schedule_config):
        """Test scheduling an immediate experiment"""
        schedule_id = scheduler.schedule_experiment(sample_experiment_config, sample_schedule_config)

        assert schedule_id is not None
        assert schedule_id in scheduler.scheduled_experiments

        scheduled_exp = scheduler.scheduled_experiments[schedule_id]
        assert scheduled_exp.experiment_config == sample_experiment_config
        assert scheduled_exp.schedule_config == sample_schedule_config
        assert scheduled_exp.status == ScheduleStatus.PENDING

    def test_schedule_delayed_experiment(self, scheduler, sample_experiment_config):
        """Test scheduling a delayed experiment"""
        future_time = datetime.now() + timedelta(hours=1)
        delayed_config = ScheduleConfig(
            schedule_type=ScheduleType.DELAYED,
            start_time=future_time
        )

        schedule_id = scheduler.schedule_experiment(sample_experiment_config, delayed_config)
        scheduled_exp = scheduler.scheduled_experiments[schedule_id]

        assert scheduled_exp.scheduled_time == future_time
        assert scheduled_exp.status == ScheduleStatus.PENDING

    def test_schedule_recurring_experiment(self, scheduler, sample_experiment_config):
        """Test scheduling a recurring experiment"""
        recurring_config = ScheduleConfig(
            schedule_type=ScheduleType.RECURRING,
            interval_seconds=3600,  # 1 hour
            max_runs=5
        )

        schedule_id = scheduler.schedule_experiment(sample_experiment_config, recurring_config)
        scheduled_exp = scheduler.scheduled_experiments[schedule_id]

        assert scheduled_exp.schedule_config.interval_seconds == 3600
        assert scheduled_exp.schedule_config.max_runs == 5

    def test_cancel_scheduled_experiment(self, scheduler, sample_experiment_config, sample_schedule_config):
        """Test canceling a scheduled experiment"""
        schedule_id = scheduler.schedule_experiment(sample_experiment_config, sample_schedule_config)

        result = scheduler.cancel_scheduled_experiment(schedule_id)
        assert result is True

        scheduled_exp = scheduler.scheduled_experiments[schedule_id]
        assert scheduled_exp.status == ScheduleStatus.CANCELLED

    def test_cancel_nonexistent_experiment(self, scheduler):
        """Test canceling a non-existent experiment"""
        result = scheduler.cancel_scheduled_experiment("nonexistent_id")
        assert result is False

    def test_get_scheduled_experiments(self, scheduler, sample_experiment_config, sample_schedule_config):
        """Test retrieving scheduled experiments"""
        # Schedule multiple experiments
        schedule_id1 = scheduler.schedule_experiment(sample_experiment_config, sample_schedule_config)
        schedule_id2 = scheduler.schedule_experiment(sample_experiment_config, sample_schedule_config)

        all_experiments = scheduler.get_scheduled_experiments()
        assert len(all_experiments) == 2

        pending_experiments = scheduler.get_scheduled_experiments(ScheduleStatus.PENDING)
        assert len(pending_experiments) == 2

    def test_get_schedule_status(self, scheduler, sample_experiment_config, sample_schedule_config):
        """Test getting schedule status"""
        schedule_id = scheduler.schedule_experiment(sample_experiment_config, sample_schedule_config)

        status = scheduler.get_schedule_status(schedule_id)
        assert status is not None
        assert status['schedule_id'] == schedule_id
        assert status['status'] == ScheduleStatus.PENDING.value

    def test_automation_stats(self, scheduler, sample_experiment_config, sample_schedule_config):
        """Test automation statistics"""
        # Schedule some experiments
        scheduler.schedule_experiment(sample_experiment_config, sample_schedule_config)
        scheduler.schedule_experiment(sample_experiment_config, sample_schedule_config)

        stats = scheduler.get_automation_stats()

        assert 'total_scheduled' in stats
        assert 'currently_running' in stats
        assert 'queue_size' in stats
        assert 'automation_active' in stats
        assert stats['total_scheduled'] == 2

    def test_scheduler_start_stop(self, scheduler):
        """Test scheduler start and stop functionality"""
        assert not scheduler.running

        scheduler.start()
        assert scheduler.running

        scheduler.stop()
        assert not scheduler.running

    def test_experiment_execution_flow(self, scheduler, sample_experiment_config, sample_schedule_config, mock_experiment_manager):
        """Test the complete experiment execution flow"""
        # Start the scheduler
        scheduler.start()

        # Schedule an immediate experiment
        schedule_id = scheduler.schedule_experiment(sample_experiment_config, sample_schedule_config)

        # Wait briefly for execution
        time.sleep(0.1)

        # Verify experiment manager was called
        mock_experiment_manager.create_experiment.assert_called()

        scheduler.stop()

    def test_scheduler_thread_safety(self, scheduler, sample_experiment_config, sample_schedule_config):
        """Test scheduler thread safety"""
        def schedule_experiment():
            for i in range(5):
                scheduler.schedule_experiment(sample_experiment_config, sample_schedule_config)

        # Create multiple threads scheduling experiments
        threads = [threading.Thread(target=schedule_experiment) for _ in range(3)]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        # Should have 15 scheduled experiments (3 threads × 5 experiments)
        assert len(scheduler.scheduled_experiments) == 15

    def test_conditional_experiment_scheduling(self, scheduler, sample_experiment_config):
        """Test conditional experiment scheduling"""
        conditional_config = ScheduleConfig(
            schedule_type=ScheduleType.CONDITIONAL,
            condition_check="test_condition"
        )

        schedule_id = scheduler.schedule_experiment(sample_experiment_config, conditional_config)
        scheduled_exp = scheduler.scheduled_experiments[schedule_id]

        assert scheduled_exp.schedule_config.condition_check == "test_condition"

    def test_experiment_retry_logic(self, scheduler, sample_experiment_config):
        """Test experiment retry logic on failure"""
        retry_config = ScheduleConfig(
            schedule_type=ScheduleType.IMMEDIATE,
            retry_on_failure=True,
            max_retries=3
        )

        schedule_id = scheduler.schedule_experiment(sample_experiment_config, retry_config)
        scheduled_exp = scheduler.scheduled_experiments[schedule_id]

        assert scheduled_exp.schedule_config.retry_on_failure is True
        assert scheduled_exp.schedule_config.max_retries == 3

    def test_dependency_management(self, scheduler, sample_experiment_config):
        """Test experiment dependency management"""
        # Schedule parent experiment
        parent_schedule_id = scheduler.schedule_experiment(sample_experiment_config, ScheduleConfig(schedule_type=ScheduleType.IMMEDIATE))

        # Schedule dependent experiment
        dependent_config = ScheduleConfig(
            schedule_type=ScheduleType.IMMEDIATE,
            depends_on=[parent_schedule_id]
        )

        dependent_schedule_id = scheduler.schedule_experiment(sample_experiment_config, dependent_config)
        dependent_exp = scheduler.scheduled_experiments[dependent_schedule_id]

        assert parent_schedule_id in dependent_exp.schedule_config.depends_on


class TestAutomationTemplates:
    """Test suite for AutomationTemplates class"""

    @pytest.fixture
    def templates(self):
        """Create an AutomationTemplates instance for testing"""
        return AutomationTemplates()

    def test_templates_initialization(self, templates):
        """Test AutomationTemplates initialization"""
        assert templates.config_comparator is not None

    def test_get_available_templates(self, templates):
        """Test getting available automation templates"""
        available_templates = templates.get_available_templates()

        assert isinstance(available_templates, dict)
        assert 'continuous_optimization' in available_templates
        assert 'regression_testing' in available_templates
        assert 'hyperparameter_sweep' in available_templates
        assert 'performance_monitoring' in available_templates
        assert 'weekend_stress_test' in available_templates
        assert 'custom_workflow' in available_templates

    def test_get_template_parameters(self, templates):
        """Test getting template parameters"""
        params = templates.get_template_parameters('continuous_optimization')

        assert isinstance(params, dict)
        assert 'base_name' in params
        assert 'test_interval_hours' in params
        assert 'sample_size' in params

    def test_continuous_optimization_workflow(self, templates):
        """Test continuous optimization workflow creation"""
        workflows = templates.create_continuous_optimization_workflow(
            base_name="Test Optimization",
            test_interval_hours=6,
            sample_size=50
        )

        assert isinstance(workflows, list)
        assert len(workflows) > 0

        # Check first workflow structure
        workflow = workflows[0]
        assert 'name' in workflow
        assert 'description' in workflow
        assert 'experiment_config' in workflow
        assert 'schedule_config' in workflow

        assert isinstance(workflow['experiment_config'], ExperimentConfig)
        assert isinstance(workflow['schedule_config'], ScheduleConfig)

    def test_regression_testing_suite(self, templates):
        """Test regression testing suite creation"""
        baseline_config = {'strategy': 'balanced'}
        test_configs = [
            {'name': 'Aggressive', 'strategy': 'aggressive'},
            {'name': 'Defensive', 'strategy': 'defensive'}
        ]

        workflows = templates.create_regression_testing_suite(
            baseline_config=baseline_config,
            test_configs=test_configs,
            sample_size=30
        )

        assert isinstance(workflows, list)
        assert len(workflows) == len(test_configs)

        for workflow in workflows:
            assert workflow['experiment_config'].experiment_type == ExperimentType.PLUGIN_COMPARISON

    def test_hyperparameter_sweep(self, templates):
        """Test hyperparameter sweep creation"""
        parameter_ranges = {
            'aggression_level': [0.3, 0.6, 0.9],
            'exploration_radius': [2, 4, 6]
        }
        base_config = {'base_param': 'value'}

        workflows = templates.create_hyperparameter_sweep(
            parameter_ranges=parameter_ranges,
            base_config=base_config,
            sample_size=20
        )

        assert isinstance(workflows, list)
        # Should create 3 × 3 = 9 parameter combinations
        assert len(workflows) == 9

    def test_performance_monitoring_workflow(self, templates):
        """Test performance monitoring workflow creation"""
        workflows = templates.create_performance_monitoring_workflow(
            check_interval_hours=12,
            performance_threshold=0.95
        )

        assert isinstance(workflows, list)
        assert len(workflows) > 0

        # Check that workflows have monitoring-specific configuration
        for workflow in workflows:
            assert 'monitoring' in workflow['name'].lower() or 'performance' in workflow['name'].lower()

    def test_weekend_stress_test(self, templates):
        """Test weekend stress test creation"""
        workflow = templates.create_weekend_stress_test(
            stress_duration_hours=48,
            sample_size=100
        )

        assert isinstance(workflow, dict)
        assert 'name' in workflow
        assert 'stress' in workflow['name'].lower()
        assert workflow['experiment_config'].sample_size_per_variant == 100

    def test_custom_workflow(self, templates):
        """Test custom workflow creation"""
        experiments = [
            {'name': 'Test 1', 'config': {'param': 'value1'}},
            {'name': 'Test 2', 'config': {'param': 'value2'}}
        ]
        dependencies = {'Test 2': ['Test 1']}
        intervals = {'Test 1': 3600, 'Test 2': 7200}

        workflows = templates.create_custom_workflow(
            workflow_name="Custom Test Workflow",
            experiments=experiments,
            dependencies=dependencies,
            intervals=intervals
        )

        assert isinstance(workflows, list)
        assert len(workflows) == len(experiments)

    def test_template_validation(self, templates):
        """Test template parameter validation"""
        # Test with invalid parameter ranges
        with pytest.raises((ValueError, TypeError)):
            templates.create_hyperparameter_sweep(
                parameter_ranges=None,  # Invalid
                base_config={},
                sample_size=10
            )

    def test_template_edge_cases(self, templates):
        """Test template edge cases"""
        # Test with minimal parameters
        workflows = templates.create_continuous_optimization_workflow(
            base_name="Minimal Test",
            test_interval_hours=1,
            sample_size=1
        )

        assert len(workflows) > 0
        assert workflows[0]['experiment_config'].sample_size_per_variant == 1

    def test_template_configuration_consistency(self, templates):
        """Test that template configurations are consistent"""
        workflows = templates.create_continuous_optimization_workflow()

        for workflow in workflows:
            # All workflows should have proper experiment configs
            config = workflow['experiment_config']
            assert config.name is not None
            assert config.description is not None
            assert config.experiment_type is not None
            assert config.sample_size_per_variant > 0

            # All workflows should have proper schedule configs
            schedule = workflow['schedule_config']
            assert schedule.schedule_type is not None


class TestScheduleConfig:
    """Test suite for ScheduleConfig class"""

    def test_immediate_schedule_config(self):
        """Test immediate schedule configuration"""
        config = ScheduleConfig(schedule_type=ScheduleType.IMMEDIATE)

        assert config.schedule_type == ScheduleType.IMMEDIATE
        assert config.start_time is None
        assert config.delay_seconds is None

    def test_delayed_schedule_config(self):
        """Test delayed schedule configuration"""
        start_time = datetime.now() + timedelta(hours=1)
        config = ScheduleConfig(
            schedule_type=ScheduleType.DELAYED,
            start_time=start_time
        )

        assert config.schedule_type == ScheduleType.DELAYED
        assert config.start_time == start_time

    def test_recurring_schedule_config(self):
        """Test recurring schedule configuration"""
        config = ScheduleConfig(
            schedule_type=ScheduleType.RECURRING,
            interval_seconds=3600,
            max_runs=10
        )

        assert config.schedule_type == ScheduleType.RECURRING
        assert config.interval_seconds == 3600
        assert config.max_runs == 10

    def test_schedule_config_defaults(self):
        """Test schedule configuration defaults"""
        config = ScheduleConfig(schedule_type=ScheduleType.IMMEDIATE)

        assert config.auto_analyze is True
        assert config.auto_archive is True
        assert config.max_concurrent == 1
        assert config.retry_on_failure is True
        assert config.max_retries == 3


class TestScheduledExperiment:
    """Test suite for ScheduledExperiment class"""

    @pytest.fixture
    def sample_experiment_config(self):
        """Create a sample experiment configuration"""
        return ExperimentConfig(
            name="Test Experiment",
            description="A test experiment",
            experiment_type=ExperimentType.PLUGIN_COMPARISON,
            variants={"control": {"config": {}}},
            sample_size_per_variant=10
        )

    @pytest.fixture
    def sample_schedule_config(self):
        """Create a sample schedule configuration"""
        return ScheduleConfig(schedule_type=ScheduleType.IMMEDIATE)

    def test_scheduled_experiment_creation(self, sample_experiment_config, sample_schedule_config):
        """Test ScheduledExperiment creation"""
        scheduled_exp = ScheduledExperiment(
            schedule_id="test_123",
            experiment_config=sample_experiment_config,
            schedule_config=sample_schedule_config
        )

        assert scheduled_exp.schedule_id == "test_123"
        assert scheduled_exp.experiment_config == sample_experiment_config
        assert scheduled_exp.schedule_config == sample_schedule_config
        assert scheduled_exp.status == ScheduleStatus.PENDING
        assert scheduled_exp.attempt_count == 0

    def test_scheduled_experiment_status_transitions(self, sample_experiment_config, sample_schedule_config):
        """Test ScheduledExperiment status transitions"""
        scheduled_exp = ScheduledExperiment(
            schedule_id="test_123",
            experiment_config=sample_experiment_config,
            schedule_config=sample_schedule_config
        )

        # Test status progression
        scheduled_exp.status = ScheduleStatus.RUNNING
        assert scheduled_exp.status == ScheduleStatus.RUNNING

        scheduled_exp.status = ScheduleStatus.COMPLETED
        assert scheduled_exp.status == ScheduleStatus.COMPLETED

    def test_scheduled_experiment_timing(self, sample_experiment_config, sample_schedule_config):
        """Test ScheduledExperiment timing fields"""
        scheduled_exp = ScheduledExperiment(
            schedule_id="test_123",
            experiment_config=sample_experiment_config,
            schedule_config=sample_schedule_config
        )

        # Initially no timing information
        assert scheduled_exp.started_time is None
        assert scheduled_exp.completed_time is None

        # Set start time
        start_time = datetime.now()
        scheduled_exp.started_time = start_time
        assert scheduled_exp.started_time == start_time


class TestAutomationIntegration:
    """Integration tests for automation components"""

    @pytest.fixture
    def mock_experiment_manager(self):
        """Create a mock experiment manager"""
        manager = Mock(spec=ExperimentManager)
        manager.create_experiment.return_value = "test_experiment_123"
        manager.start_experiment.return_value = True
        manager.get_experiment.return_value = Mock(status="completed")
        manager.event_bus = Mock()
        return manager

    def test_scheduler_template_integration(self, mock_experiment_manager):
        """Test integration between scheduler and templates"""
        scheduler = ExperimentScheduler(mock_experiment_manager)
        templates = AutomationTemplates()

        # Create a workflow from template
        workflows = templates.create_continuous_optimization_workflow(
            base_name="Integration Test",
            test_interval_hours=1,
            sample_size=5
        )

        # Schedule the workflows
        schedule_ids = []
        for workflow in workflows[:2]:  # Only test first 2
            schedule_id = scheduler.schedule_experiment(
                workflow['experiment_config'],
                workflow['schedule_config']
            )
            schedule_ids.append(schedule_id)

        assert len(schedule_ids) == 2
        assert len(scheduler.scheduled_experiments) == 2

    def test_end_to_end_automation_workflow(self, mock_experiment_manager):
        """Test complete end-to-end automation workflow"""
        scheduler = ExperimentScheduler(mock_experiment_manager)
        templates = AutomationTemplates()

        # Start scheduler
        scheduler.start()

        try:
            # Create and schedule a simple workflow
            workflow = templates.create_weekend_stress_test(
                stress_duration_hours=1,  # Short for testing
                sample_size=5
            )

            schedule_id = scheduler.schedule_experiment(
                workflow['experiment_config'],
                workflow['schedule_config']
            )

            # Wait briefly for execution
            time.sleep(0.1)

            # Verify the experiment was processed
            assert schedule_id in scheduler.scheduled_experiments

        finally:
            scheduler.stop()

    def test_automation_error_handling(self, mock_experiment_manager):
        """Test automation error handling"""
        # Configure manager to simulate failure
        mock_experiment_manager.create_experiment.side_effect = Exception("Test error")

        scheduler = ExperimentScheduler(mock_experiment_manager)
        templates = AutomationTemplates()

        workflow = templates.create_weekend_stress_test(sample_size=1)

        # This should not raise an exception
        schedule_id = scheduler.schedule_experiment(
            workflow['experiment_config'],
            workflow['schedule_config']
        )

        assert schedule_id in scheduler.scheduled_experiments

    def test_concurrent_automation_execution(self, mock_experiment_manager):
        """Test concurrent automation execution"""
        scheduler = ExperimentScheduler(mock_experiment_manager)
        templates = AutomationTemplates()

        # Create multiple workflows
        workflows = []
        workflows.extend(templates.create_continuous_optimization_workflow(sample_size=1))
        workflows.extend(templates.create_performance_monitoring_workflow())

        # Schedule all workflows concurrently
        schedule_ids = []
        for workflow in workflows[:5]:  # Limit to 5 for testing
            schedule_id = scheduler.schedule_experiment(
                workflow['experiment_config'],
                workflow['schedule_config']
            )
            schedule_ids.append(schedule_id)

        assert len(schedule_ids) == 5
        assert len(scheduler.scheduled_experiments) == 5