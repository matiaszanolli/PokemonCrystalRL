"""
Automation Templates - Pre-built workflows for automated A/B testing

This module provides pre-configured templates for common A/B testing
automation scenarios, making it easy to set up complex testing workflows.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from .experiment_scheduler import ScheduleConfig, ScheduleType
from .configuration_comparator import ConfigurationComparator
from .experiment_models import ExperimentConfig, ExperimentType


class AutomationTemplates:
    """
    Pre-built automation templates for common A/B testing scenarios.

    Provides templates for:
    - Continuous optimization workflows
    - Systematic plugin testing
    - Performance regression testing
    - Automated hyperparameter tuning
    """

    def __init__(self):
        self.logger = logging.getLogger("AutomationTemplates")
        self.config_comparator = ConfigurationComparator()

    def create_continuous_optimization_workflow(self,
                                               base_name: str = "Continuous Optimization",
                                               test_interval_hours: int = 6,
                                               sample_size: int = 50) -> List[Dict[str, Any]]:
        """
        Create a continuous optimization workflow that automatically tests
        different configurations at regular intervals.

        Args:
            base_name: Base name for experiments
            test_interval_hours: Hours between tests
            sample_size: Sample size per variant

        Returns:
            List of experiment and schedule configurations
        """
        workflows = []

        # 1. Battle Strategy Optimization (every 6 hours)
        battle_config = self.config_comparator.create_battle_strategy_comparison()
        battle_config.name = f"{base_name} - Battle Strategies"
        battle_config.sample_size_per_variant = sample_size
        battle_config.max_runtime_seconds = test_interval_hours * 3600 - 300  # Leave 5 min buffer

        battle_schedule = ScheduleConfig(
            schedule_type=ScheduleType.RECURRING,
            interval_seconds=test_interval_hours * 3600,
            auto_analyze=True,
            auto_archive=True,
            max_concurrent=1,
            retry_on_failure=True
        )

        workflows.append({
            'name': 'Battle Strategy Optimization',
            'experiment_config': battle_config,
            'schedule_config': battle_schedule,
            'description': f'Automatically test battle strategies every {test_interval_hours} hours'
        })

        # 2. Exploration Pattern Testing (offset by 2 hours)
        exploration_config = self.config_comparator.create_exploration_pattern_test()
        exploration_config.name = f"{base_name} - Exploration Patterns"
        exploration_config.sample_size_per_variant = sample_size // 2  # Faster testing

        exploration_schedule = ScheduleConfig(
            schedule_type=ScheduleType.DELAYED,
            delay_seconds=2 * 3600,  # Start 2 hours after deployment
            auto_analyze=True,
            auto_archive=True,
            max_concurrent=1
        )

        # Set up recurring after initial delay
        exploration_schedule_recurring = ScheduleConfig(
            schedule_type=ScheduleType.RECURRING,
            interval_seconds=test_interval_hours * 3600,
            auto_analyze=True,
            auto_archive=True,
            max_concurrent=1
        )

        workflows.append({
            'name': 'Exploration Pattern Testing',
            'experiment_config': exploration_config,
            'schedule_config': exploration_schedule,
            'description': f'Test exploration patterns every {test_interval_hours} hours (offset by 2h)'
        })

        # 3. Plugin Performance Comparison (daily)
        plugin_config = self.config_comparator.create_plugin_comparison()
        plugin_config.name = f"{base_name} - Plugin Performance"
        plugin_config.sample_size_per_variant = sample_size * 2  # More thorough testing

        plugin_schedule = ScheduleConfig(
            schedule_type=ScheduleType.RECURRING,
            interval_seconds=24 * 3600,  # Daily
            auto_analyze=True,
            auto_archive=True,
            max_concurrent=1,
            retry_on_failure=True,
            max_retries=2
        )

        workflows.append({
            'name': 'Daily Plugin Performance',
            'experiment_config': plugin_config,
            'schedule_config': plugin_schedule,
            'description': 'Comprehensive daily plugin performance comparison'
        })

        return workflows

    def create_regression_testing_suite(self,
                                       baseline_config: Dict[str, Any],
                                       test_configs: List[Dict[str, Any]],
                                       sample_size: int = 30) -> List[Dict[str, Any]]:
        """
        Create a regression testing suite that compares new configurations
        against a known baseline.

        Args:
            baseline_config: Baseline configuration to test against
            test_configs: List of new configurations to test
            sample_size: Sample size per variant

        Returns:
            List of regression test configurations
        """
        regression_tests = []

        for i, test_config in enumerate(test_configs):
            # Create A/B test comparing baseline vs new config
            experiment_config = ExperimentConfig(
                name=f"Regression Test {i+1} - {test_config.get('name', f'Config {i+1}')}",
                experiment_type=ExperimentType.AGENT_COMPARISON,
                description=f"Regression test comparing baseline against {test_config.get('name', 'new configuration')}",
                variants={
                    'baseline': baseline_config,
                    'candidate': test_config
                },
                sample_size_per_variant=sample_size,
                max_runtime_seconds=1800,  # 30 minutes max
                primary_metrics=['total_reward', 'actions_per_second'],
                secondary_metrics=['battle_win_rate'],
                confidence_level=0.95,
                minimum_effect_size=0.05  # Detect 5% changes
            )

            # Schedule immediately but with dependencies
            schedule_config = ScheduleConfig(
                schedule_type=ScheduleType.IMMEDIATE if i == 0 else ScheduleType.CONDITIONAL,
                condition_check=f"completed_experiments >= {i}" if i > 0 else None,
                auto_analyze=True,
                auto_archive=True,
                max_concurrent=1,
                retry_on_failure=True
            )

            regression_tests.append({
                'name': f'Regression Test {i+1}',
                'experiment_config': experiment_config,
                'schedule_config': schedule_config,
                'description': f'Test configuration {i+1} against baseline'
            })

        return regression_tests

    def create_hyperparameter_sweep(self,
                                   parameter_ranges: Dict[str, List[Any]],
                                   base_config: Dict[str, Any],
                                   sample_size: int = 20) -> List[Dict[str, Any]]:
        """
        Create a hyperparameter sweep that systematically tests
        different parameter combinations.

        Args:
            parameter_ranges: Dict mapping parameter names to lists of values to test
            base_config: Base configuration to modify
            sample_size: Sample size per test

        Returns:
            List of hyperparameter test configurations
        """
        sweep_tests = []

        # Generate all parameter combinations
        param_names = list(parameter_ranges.keys())
        param_values = list(parameter_ranges.values())

        # Simple combinatorial generation (could use itertools.product for complex cases)
        test_count = 0
        for param_name in param_names:
            for value in parameter_ranges[param_name]:
                test_config = base_config.copy()
                test_config[param_name] = value

                experiment_config = ExperimentConfig(
                    name=f"Hyperparameter Sweep - {param_name}={value}",
                    experiment_type=ExperimentType.PLUGIN_COMPARISON,
                    description=f"Testing {param_name} = {value}",
                    variants={
                        'baseline': base_config,
                        f'{param_name}_{value}': test_config
                    },
                    sample_size_per_variant=sample_size,
                    max_runtime_seconds=1200,  # 20 minutes
                    primary_metrics=['total_reward'],
                    confidence_level=0.90  # Slightly lower for exploratory testing
                )

                # Stagger execution to avoid overload
                schedule_config = ScheduleConfig(
                    schedule_type=ScheduleType.DELAYED,
                    delay_seconds=test_count * 300,  # 5 minute intervals
                    auto_analyze=True,
                    auto_archive=True,
                    max_concurrent=1
                )

                sweep_tests.append({
                    'name': f'Sweep {param_name}={value}',
                    'experiment_config': experiment_config,
                    'schedule_config': schedule_config,
                    'description': f'Test hyperparameter {param_name} = {value}'
                })

                test_count += 1

        return sweep_tests

    def create_performance_monitoring_workflow(self,
                                             check_interval_hours: int = 12,
                                             performance_threshold: float = 0.95) -> List[Dict[str, Any]]:
        """
        Create a performance monitoring workflow that automatically
        runs performance checks and alerts on degradation.

        Args:
            check_interval_hours: Hours between performance checks
            performance_threshold: Performance threshold (0.0-1.0)

        Returns:
            List of monitoring configurations
        """
        monitoring_tests = []

        # 1. Quick performance check
        quick_check_config = self.config_comparator.create_battle_strategy_comparison()
        quick_check_config.name = "Performance Monitor - Quick Check"
        quick_check_config.sample_size_per_variant = 10  # Small for quick feedback
        quick_check_config.max_runtime_seconds = 600    # 10 minutes

        quick_schedule = ScheduleConfig(
            schedule_type=ScheduleType.RECURRING,
            interval_seconds=check_interval_hours * 3600,
            auto_analyze=True,
            auto_archive=True,
            max_concurrent=1,
            condition_check=f"True"  # Always run
        )

        monitoring_tests.append({
            'name': 'Quick Performance Check',
            'experiment_config': quick_check_config,
            'schedule_config': quick_schedule,
            'description': f'Quick performance check every {check_interval_hours} hours'
        })

        # 2. Comprehensive performance validation (triggered by poor quick check results)
        comprehensive_config = self.config_comparator.create_plugin_comparison()
        comprehensive_config.name = "Performance Monitor - Comprehensive"
        comprehensive_config.sample_size_per_variant = 50
        comprehensive_config.max_runtime_seconds = 3600  # 1 hour

        comprehensive_schedule = ScheduleConfig(
            schedule_type=ScheduleType.CONDITIONAL,
            condition_check=f"running_experiments == 0",  # Only when no other experiments running
            auto_analyze=True,
            auto_archive=True,
            max_concurrent=1
        )

        monitoring_tests.append({
            'name': 'Comprehensive Performance Validation',
            'experiment_config': comprehensive_config,
            'schedule_config': comprehensive_schedule,
            'description': 'Triggered when performance drops below threshold'
        })

        return monitoring_tests

    def create_weekend_stress_test(self,
                                   stress_duration_hours: int = 48,
                                   sample_size: int = 100) -> Dict[str, Any]:
        """
        Create a weekend stress test that runs intensive testing
        when system usage is typically lower.

        Args:
            stress_duration_hours: Duration of stress test
            sample_size: Large sample size for thorough testing

        Returns:
            Stress test configuration
        """
        # Create comprehensive test configuration
        stress_config = ExperimentConfig(
            name="Weekend Stress Test - Comprehensive Evaluation",
            experiment_type=ExperimentType.MULTI_FACTOR,
            description=f"Comprehensive {stress_duration_hours}h stress test with multiple factors",
            variants={
                'aggressive_battle_exploration': {
                    'aggressive_battle_strategy': {'aggression_level': 0.9},
                    'spiral_exploration': {'spiral_radius': 3, 'max_distance': 10}
                },
                'defensive_battle_systematic': {
                    'defensive_battle_strategy': {'aggression_level': 0.3},
                    'systematic_exploration': {'grid_size': 5, 'coverage_threshold': 0.8}
                },
                'balanced_approach': {
                    'balanced_battle_strategy': {'aggression_level': 0.6},
                    'wall_following_exploration': {'wall_preference': 0.7}
                }
            },
            sample_size_per_variant=sample_size,
            max_runtime_seconds=stress_duration_hours * 3600,
            primary_metrics=['total_reward', 'battle_win_rate', 'actions_per_second'],
            secondary_metrics=['exploration_coverage', 'decision_confidence'],
            confidence_level=0.99,  # High confidence for important test
            minimum_effect_size=0.03  # Detect small differences
        )

        # Schedule for Friday evening
        friday_evening = datetime.now()
        # Find next Friday at 6 PM
        days_until_friday = (4 - friday_evening.weekday()) % 7
        if days_until_friday == 0 and friday_evening.hour >= 18:
            days_until_friday = 7  # Next Friday if it's already past 6 PM on Friday

        start_time = friday_evening.replace(hour=18, minute=0, second=0, microsecond=0)
        start_time += timedelta(days=days_until_friday)

        schedule_config = ScheduleConfig(
            schedule_type=ScheduleType.RECURRING,
            start_time=start_time,
            interval_seconds=7 * 24 * 3600,  # Weekly
            auto_analyze=True,
            auto_archive=True,
            max_concurrent=1,
            retry_on_failure=False,  # Don't retry stress tests
            condition_check="running_experiments == 0"  # Only when system is free
        )

        return {
            'name': 'Weekend Stress Test',
            'experiment_config': stress_config,
            'schedule_config': schedule_config,
            'description': f'Weekly {stress_duration_hours}h comprehensive stress test starting Friday evenings'
        }

    def create_custom_workflow(self,
                              workflow_name: str,
                              experiments: List[ExperimentConfig],
                              dependencies: List[List[int]] = None,
                              intervals: List[int] = None) -> List[Dict[str, Any]]:
        """
        Create a custom workflow with specified experiments and dependencies.

        Args:
            workflow_name: Name for the workflow
            experiments: List of experiment configurations
            dependencies: List of dependency lists (experiment indices)
            intervals: List of delay intervals in seconds

        Returns:
            Custom workflow configuration
        """
        if dependencies is None:
            dependencies = [[] for _ in experiments]
        if intervals is None:
            intervals = [0 for _ in experiments]

        workflow = []

        for i, (experiment_config, deps, interval) in enumerate(zip(experiments, dependencies, intervals)):
            experiment_config.name = f"{workflow_name} - Step {i+1}: {experiment_config.name}"

            # Determine schedule type based on dependencies and interval
            if deps or interval > 0:
                schedule_type = ScheduleType.CONDITIONAL if deps else ScheduleType.DELAYED
            else:
                schedule_type = ScheduleType.IMMEDIATE

            # Create condition for dependencies
            condition = None
            if deps:
                completed_deps = " and ".join([f"'{workflow_name} - Step {d+1}' in completed_experiments" for d in deps])
                condition = completed_deps

            schedule_config = ScheduleConfig(
                schedule_type=schedule_type,
                delay_seconds=interval if interval > 0 else None,
                condition_check=condition,
                auto_analyze=True,
                auto_archive=True,
                max_concurrent=1,
                retry_on_failure=True
            )

            workflow.append({
                'name': f'{workflow_name} Step {i+1}',
                'experiment_config': experiment_config,
                'schedule_config': schedule_config,
                'description': f'Step {i+1} of {workflow_name} workflow'
            })

        return workflow

    def get_available_templates(self) -> Dict[str, str]:
        """Get list of available automation templates"""
        return {
            'continuous_optimization': 'Continuous optimization workflow with regular testing',
            'regression_testing': 'Regression testing suite comparing against baseline',
            'hyperparameter_sweep': 'Systematic hyperparameter testing',
            'performance_monitoring': 'Automated performance monitoring and alerting',
            'weekend_stress_test': 'Comprehensive weekend stress testing',
            'custom_workflow': 'Custom workflow with dependencies and scheduling'
        }

    def get_template_parameters(self, template_name: str) -> Dict[str, Any]:
        """Get parameters for a specific template"""
        parameters = {
            'continuous_optimization': {
                'base_name': {'type': 'string', 'default': 'Continuous Optimization', 'description': 'Base name for experiments'},
                'test_interval_hours': {'type': 'integer', 'default': 6, 'description': 'Hours between tests'},
                'sample_size': {'type': 'integer', 'default': 50, 'description': 'Sample size per variant'}
            },
            'regression_testing': {
                'baseline_config': {'type': 'dict', 'required': True, 'description': 'Baseline configuration'},
                'test_configs': {'type': 'list', 'required': True, 'description': 'List of test configurations'},
                'sample_size': {'type': 'integer', 'default': 30, 'description': 'Sample size per variant'}
            },
            'hyperparameter_sweep': {
                'parameter_ranges': {'type': 'dict', 'required': True, 'description': 'Parameter ranges to test'},
                'base_config': {'type': 'dict', 'required': True, 'description': 'Base configuration'},
                'sample_size': {'type': 'integer', 'default': 20, 'description': 'Sample size per test'}
            },
            'performance_monitoring': {
                'check_interval_hours': {'type': 'integer', 'default': 12, 'description': 'Hours between checks'},
                'performance_threshold': {'type': 'float', 'default': 0.95, 'description': 'Performance threshold'}
            },
            'weekend_stress_test': {
                'stress_duration_hours': {'type': 'integer', 'default': 48, 'description': 'Duration in hours'},
                'sample_size': {'type': 'integer', 'default': 100, 'description': 'Sample size for stress test'}
            },
            'custom_workflow': {
                'workflow_name': {'type': 'string', 'required': True, 'description': 'Workflow name'},
                'experiments': {'type': 'list', 'required': True, 'description': 'List of experiments'},
                'dependencies': {'type': 'list', 'default': None, 'description': 'Dependency lists'},
                'intervals': {'type': 'list', 'default': None, 'description': 'Delay intervals'}
            }
        }

        return parameters.get(template_name, {})