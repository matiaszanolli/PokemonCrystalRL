"""
Comprehensive tests for A/B Testing Framework

Tests all components of the A/B testing system including:
- Experiment models and configuration
- Experiment manager and execution
- Configuration comparator
- Statistical analyzer
"""

import pytest
import time
import threading
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from core.ab_testing import (
    ExperimentManager, ConfigurationComparator, StatisticalAnalyzer,
    Experiment, ExperimentConfig, ExperimentResult, ExperimentStatus, ExperimentType,
    PerformanceMetrics, MetricCollection, AnalysisResult,
    ConfigurationVariant, ConfigurationType
)
from core.ab_testing.statistical_analyzer import StatisticalTest, EffectSizeMethod


class TestExperimentModels:
    """Test experiment data models"""

    def test_performance_metrics_initialization(self):
        """Test PerformanceMetrics initialization and methods"""
        metrics = PerformanceMetrics()

        assert metrics.total_reward == 0.0
        assert metrics.get_sample_count() == 0
        assert len(metrics.reward_samples) == 0

    def test_performance_metrics_add_samples(self):
        """Test adding samples to PerformanceMetrics"""
        metrics = PerformanceMetrics()

        # Add reward samples
        metrics.add_reward_sample(10.0)
        metrics.add_reward_sample(15.0)

        assert metrics.total_reward == 25.0
        assert metrics.get_sample_count() == 2
        assert metrics.reward_samples == [10.0, 15.0]

    def test_performance_metrics_action_times(self):
        """Test action time tracking"""
        metrics = PerformanceMetrics()

        metrics.add_action_time(1.0)
        metrics.add_action_time(2.0)

        assert len(metrics.action_times) == 2
        assert metrics.actions_per_second == 2.0 / 3.0  # 2 actions in 3 seconds total

    def test_performance_metrics_battle_results(self):
        """Test battle result tracking"""
        metrics = PerformanceMetrics()

        metrics.add_battle_result(True)
        metrics.add_battle_result(False)
        metrics.add_battle_result(True)

        assert len(metrics.battle_results) == 3
        assert metrics.battle_win_rate == 2.0 / 3.0

    def test_performance_metrics_custom_metrics(self):
        """Test custom metrics"""
        metrics = PerformanceMetrics()

        metrics.add_custom_metric("exploration_coverage", 0.75)

        assert metrics.custom_metrics["exploration_coverage"] == 0.75

    def test_performance_metrics_to_dict(self):
        """Test metrics serialization"""
        metrics = PerformanceMetrics()
        metrics.add_reward_sample(10.0)
        metrics.add_custom_metric("test_metric", 42.0)

        data = metrics.to_dict()

        assert data["total_reward"] == 10.0
        assert data["sample_count"] == 1
        assert data["custom_metrics"]["test_metric"] == 42.0

    def test_experiment_config_validation(self):
        """Test experiment configuration validation"""
        # Valid configuration
        config = ExperimentConfig(
            name="Test Experiment",
            experiment_type=ExperimentType.PLUGIN_COMPARISON
        )
        config.add_variant("control", {"plugin": "control_config"})
        config.add_variant("treatment", {"plugin": "treatment_config"})

        assert config.validate() == True

        # Invalid configuration - not enough variants
        config_invalid = ExperimentConfig(
            name="Invalid Experiment",
            experiment_type=ExperimentType.PLUGIN_COMPARISON
        )
        config_invalid.add_variant("control", {"plugin": "control_config"})

        assert config_invalid.validate() == False

    def test_experiment_config_variant_management(self):
        """Test variant management in experiment configuration"""
        config = ExperimentConfig(
            name="Test Experiment",
            experiment_type=ExperimentType.PLUGIN_COMPARISON
        )

        config.add_variant("control", {"setting": "A"})
        config.add_variant("treatment", {"setting": "B"})

        assert len(config.get_variant_names()) == 2
        assert "control" in config.get_variant_names()
        assert "treatment" in config.get_variant_names()

    def test_experiment_initialization(self):
        """Test experiment entity initialization"""
        config = ExperimentConfig(
            name="Test Experiment",
            experiment_type=ExperimentType.PLUGIN_COMPARISON
        )
        config.add_variant("control", {"plugin": "control"})
        config.add_variant("treatment", {"plugin": "treatment"})

        experiment = Experiment(config=config)

        assert experiment.status == ExperimentStatus.PENDING
        assert experiment.total_runs == 200  # 2 variants × 100 samples default
        assert len(experiment.live_metrics) == 2

    def test_experiment_lifecycle(self):
        """Test experiment lifecycle management"""
        config = ExperimentConfig(
            name="Test Experiment",
            experiment_type=ExperimentType.PLUGIN_COMPARISON
        )
        config.add_variant("control", {"plugin": "control"})
        config.add_variant("treatment", {"plugin": "treatment"})

        experiment = Experiment(config=config)

        # Start experiment
        experiment.start()
        assert experiment.status == ExperimentStatus.RUNNING
        assert experiment.result is not None
        assert experiment.result.start_time > 0

        # Complete experiment
        experiment.complete()
        assert experiment.status == ExperimentStatus.COMPLETED
        assert experiment.result.end_time is not None

    def test_experiment_progress_tracking(self):
        """Test experiment progress calculation"""
        config = ExperimentConfig(
            name="Test Experiment",
            experiment_type=ExperimentType.PLUGIN_COMPARISON,
            sample_size_per_variant=10
        )
        config.add_variant("control", {"plugin": "control"})
        config.add_variant("treatment", {"plugin": "treatment"})

        experiment = Experiment(config=config)

        assert experiment.get_progress() == 0.0

        experiment.current_run = 10
        progress = experiment.get_progress()
        assert progress == 50.0  # 10 out of 20 total runs


class TestConfigurationComparator:
    """Test configuration comparison functionality"""

    def setup_method(self):
        """Set up test fixtures"""
        self.comparator = ConfigurationComparator()

    def test_create_plugin_comparison(self):
        """Test creating plugin comparison experiment"""
        base_config = {"max_actions": 1000}
        plugin_variants = {
            "aggressive": {"aggressive_battle": {"aggression": 0.9}},
            "defensive": {"defensive_battle": {"aggression": 0.2}}
        }

        experiment_config = self.comparator.create_plugin_comparison(
            base_config, plugin_variants, "Test Plugin Comparison"
        )

        assert experiment_config.name == "Test Plugin Comparison"
        assert experiment_config.experiment_type == ExperimentType.PLUGIN_COMPARISON
        assert len(experiment_config.variants) == 2
        assert "aggressive" in experiment_config.variants
        assert "defensive" in experiment_config.variants

    def test_create_agent_comparison(self):
        """Test creating agent comparison experiment"""
        base_config = {"max_actions": 1000}
        agent_variants = {
            "battle_focused": {"coordination": "battle_priority"},
            "exploration_focused": {"coordination": "exploration_priority"}
        }

        experiment_config = self.comparator.create_agent_comparison(
            base_config, agent_variants, "Test Agent Comparison"
        )

        assert experiment_config.experiment_type == ExperimentType.AGENT_COMPARISON
        assert len(experiment_config.variants) == 2

    def test_create_battle_strategy_comparison(self):
        """Test pre-configured battle strategy comparison"""
        experiment_config = self.comparator.create_battle_strategy_comparison()

        assert experiment_config.experiment_type == ExperimentType.PLUGIN_COMPARISON
        assert len(experiment_config.variants) == 3  # aggressive, defensive, balanced
        assert "aggressive" in experiment_config.variants
        assert "defensive" in experiment_config.variants
        assert "balanced" in experiment_config.variants

    def test_create_exploration_pattern_comparison(self):
        """Test pre-configured exploration pattern comparison"""
        experiment_config = self.comparator.create_exploration_pattern_comparison()

        assert experiment_config.experiment_type == ExperimentType.PLUGIN_COMPARISON
        assert len(experiment_config.variants) == 4  # systematic, spiral, random, wall_following
        assert "systematic" in experiment_config.variants
        assert "spiral" in experiment_config.variants

    def test_create_multi_agent_comparison(self):
        """Test pre-configured multi-agent comparison"""
        experiment_config = self.comparator.create_multi_agent_comparison()

        assert experiment_config.experiment_type == ExperimentType.AGENT_COMPARISON
        assert len(experiment_config.variants) == 4  # battle_focused, exploration_focused, balanced, progression_focused

    def test_create_hybrid_comparison(self):
        """Test hybrid configuration comparison"""
        experiment_config = self.comparator.create_hybrid_comparison()

        assert experiment_config.experiment_type == ExperimentType.MULTI_FACTOR
        assert len(experiment_config.variants) == 3

    def test_configuration_variant_validation(self):
        """Test configuration variant validation"""
        # Valid plugin configuration
        plugin_variant = ConfigurationVariant(
            name="Test Plugin",
            description="Test plugin configuration",
            configuration={"plugins": {"test_plugin": {"param": "value"}}},
            configuration_type=ConfigurationType.PLUGIN_CONFIG
        )

        is_valid, error = plugin_variant.validate()
        assert is_valid == True
        assert error == ""

        # Invalid plugin configuration
        invalid_variant = ConfigurationVariant(
            name="Invalid Plugin",
            description="Invalid plugin configuration",
            configuration={"invalid_key": "value"},
            configuration_type=ConfigurationType.PLUGIN_CONFIG
        )

        is_valid, error = invalid_variant.validate()
        assert is_valid == False
        assert "plugins" in error.lower()

    def test_compare_configurations(self):
        """Test configuration comparison functionality"""
        config_a = {"param1": "value1", "param2": "value2"}
        config_b = {"param1": "modified_value1", "param3": "value3"}

        differences = self.comparator.compare_configurations(config_a, config_b)

        assert "param2" in differences["removed_keys"]
        assert "param3" in differences["added_keys"]
        assert "param1" in differences["modified_values"]
        assert differences["modified_values"]["param1"]["from"] == "value1"
        assert differences["modified_values"]["param1"]["to"] == "modified_value1"


class TestStatisticalAnalyzer:
    """Test statistical analysis functionality"""

    def setup_method(self):
        """Set up test fixtures"""
        self.analyzer = StatisticalAnalyzer()

    def test_cohens_d_calculation(self):
        """Test Cohen's d effect size calculation"""
        control_data = [1.0, 2.0, 3.0, 4.0, 5.0]
        treatment_data = [3.0, 4.0, 5.0, 6.0, 7.0]  # Mean difference of 2

        cohens_d = self.analyzer._calculate_cohens_d(control_data, treatment_data)

        # Expected Cohen's d should be around 1.265 (large effect)
        assert abs(cohens_d - 1.265) < 0.05

    def test_cliff_delta_calculation(self):
        """Test Cliff's delta effect size calculation"""
        control_data = [1.0, 2.0, 3.0]
        treatment_data = [4.0, 5.0, 6.0]  # All treatment values > control values

        cliff_delta = self.analyzer._calculate_cliff_delta(control_data, treatment_data)

        # Should be 1.0 since all treatment values are greater
        assert cliff_delta == 1.0

    def test_t_test_performance(self):
        """Test t-test execution"""
        control_data = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0]
        treatment_data = [20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0]  # +10 mean difference (larger effect)

        result = self.analyzer._perform_t_test(control_data, treatment_data)

        assert result.test_type == StatisticalTest.T_TEST
        assert result.p_value < 0.05  # Should be significant
        assert result.is_significant == True
        assert result.effect_size > 0  # Positive effect
        assert result.interpretation in ["small effect", "medium effect", "large effect"]

    def test_mann_whitney_test(self):
        """Test Mann-Whitney U test"""
        control_data = [1.0, 2.0, 3.0, 4.0, 5.0]
        treatment_data = [4.0, 5.0, 6.0, 7.0, 8.0]

        result = self.analyzer._perform_mann_whitney_test(control_data, treatment_data)

        assert result.test_type == StatisticalTest.MANN_WHITNEY_U
        assert result.effect_size_method == EffectSizeMethod.CLIFF_DELTA

    def test_insufficient_data_handling(self):
        """Test handling of insufficient data"""
        control_data = [1.0, 2.0]  # Too few samples
        treatment_data = [3.0, 4.0]

        result = self.analyzer._perform_statistical_test(control_data, treatment_data, "test")

        assert result.interpretation == "insufficient data"
        assert result.is_significant == False

    def test_experiment_analysis(self):
        """Test full experiment analysis"""
        # Create mock experiment result
        experiment_result = ExperimentResult(
            experiment_id="test_experiment",
            experiment_name="Test Analysis",
            status=ExperimentStatus.COMPLETED,
            start_time=time.time()
        )

        # Add variant results with different performance (larger effect for statistical significance)
        control_metrics = PerformanceMetrics()
        for reward in [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0]:
            control_metrics.add_reward_sample(reward)
            control_metrics.add_action_time(1.0)
            control_metrics.add_battle_result(reward > 15.0)

        treatment_metrics = PerformanceMetrics()
        for reward in [20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0]:
            treatment_metrics.add_reward_sample(reward)
            treatment_metrics.add_action_time(0.8)
            treatment_metrics.add_battle_result(reward > 20.0)

        experiment_result.add_variant_result("control", control_metrics)
        experiment_result.add_variant_result("treatment", treatment_metrics)

        # Analyze experiment
        analysis = self.analyzer.analyze_experiment(experiment_result)

        assert analysis.experiment_id == "test_experiment"
        assert len(analysis.primary_metric_results) > 0
        assert "total_reward" in analysis.primary_metric_results

        # Check if significance was detected (treatment has higher rewards)
        reward_result = analysis.primary_metric_results["total_reward"]
        assert reward_result.is_significant == True
        assert analysis.has_significant_results == True
        assert analysis.winning_variant == "treatment"
        assert len(analysis.recommendations) > 0


class TestExperimentManager:
    """Test experiment manager functionality"""

    def setup_method(self):
        """Set up test fixtures"""
        self.manager = ExperimentManager({'max_concurrent': 1})

    def teardown_method(self):
        """Clean up after tests"""
        self.manager.cleanup()

    def test_experiment_creation(self):
        """Test experiment creation"""
        config = ExperimentConfig(
            name="Test Experiment",
            experiment_type=ExperimentType.PLUGIN_COMPARISON,
            sample_size_per_variant=15  # Meets minimum requirement
        )
        config.add_variant("control", {"plugin": "control"})
        config.add_variant("treatment", {"plugin": "treatment"})

        experiment_id = self.manager.create_experiment(config)

        assert experiment_id is not None
        assert experiment_id in self.manager.experiments

        experiment = self.manager.get_experiment(experiment_id)
        assert experiment.config.name == "Test Experiment"
        assert experiment.status == ExperimentStatus.PENDING

    def test_invalid_experiment_creation(self):
        """Test handling of invalid experiment configuration"""
        # Configuration with only one variant (invalid)
        config = ExperimentConfig(
            name="Invalid Experiment",
            experiment_type=ExperimentType.PLUGIN_COMPARISON
        )
        config.add_variant("control", {"plugin": "control"})

        with pytest.raises(ValueError):
            self.manager.create_experiment(config)

    def test_experiment_status_tracking(self):
        """Test experiment status tracking"""
        config = ExperimentConfig(
            name="Status Test",
            experiment_type=ExperimentType.PLUGIN_COMPARISON,
            sample_size_per_variant=10  # Minimum required
        )
        config.add_variant("control", {"plugin": "control"})
        config.add_variant("treatment", {"plugin": "treatment"})

        experiment_id = self.manager.create_experiment(config)

        # Check initial status
        status = self.manager.get_experiment_status(experiment_id)
        assert status["status"] == "pending"
        assert status["progress"] == 0.0

    def test_experiment_listing(self):
        """Test experiment listing and filtering"""
        # Create multiple experiments
        for i in range(3):
            config = ExperimentConfig(
                name=f"Experiment {i}",
                experiment_type=ExperimentType.PLUGIN_COMPARISON
            )
            config.add_variant("control", {"plugin": "control"})
            config.add_variant("treatment", {"plugin": "treatment"})
            self.manager.create_experiment(config)

        # List all experiments
        all_experiments = self.manager.list_experiments()
        assert len(all_experiments) == 3

        # Filter by status
        pending_experiments = self.manager.list_experiments(ExperimentStatus.PENDING)
        assert len(pending_experiments) == 3

    def test_concurrent_experiment_limit(self):
        """Test concurrent experiment limits"""
        # Create first experiment
        config1 = ExperimentConfig(
            name="Experiment 1",
            experiment_type=ExperimentType.PLUGIN_COMPARISON,
            sample_size_per_variant=10  # Minimum required
        )
        config1.add_variant("control", {"plugin": "control"})
        config1.add_variant("treatment", {"plugin": "treatment"})

        experiment_id1 = self.manager.create_experiment(config1)

        # Create second experiment
        config2 = ExperimentConfig(
            name="Experiment 2",
            experiment_type=ExperimentType.PLUGIN_COMPARISON,
            sample_size_per_variant=10  # Minimum required
        )
        config2.add_variant("control", {"plugin": "control"})
        config2.add_variant("treatment", {"plugin": "treatment"})

        experiment_id2 = self.manager.create_experiment(config2)

        # Start first experiment (should succeed)
        success1 = self.manager.start_experiment(experiment_id1)
        assert success1 == True

        # Try to start second experiment (should fail due to limit)
        success2 = self.manager.start_experiment(experiment_id2)
        assert success2 == False

    def test_experiment_execution_simulation(self):
        """Test experiment execution with simulation"""
        config = ExperimentConfig(
            name="Simulation Test",
            experiment_type=ExperimentType.PLUGIN_COMPARISON,
            sample_size_per_variant=10,  # Minimum required
            max_runtime_seconds=10
        )
        config.add_variant("control", {"plugin": "control"})
        config.add_variant("treatment", {"plugin": "treatment"})

        experiment_id = self.manager.create_experiment(config)

        # Start experiment
        success = self.manager.start_experiment(experiment_id)
        assert success == True

        # Wait for completion (simulation should be quick)
        max_wait = 15  # seconds
        start_time = time.time()

        while time.time() - start_time < max_wait:
            experiment = self.manager.get_experiment(experiment_id)
            if experiment.status in [ExperimentStatus.COMPLETED, ExperimentStatus.FAILED]:
                break
            time.sleep(0.5)

        # Check final status
        experiment = self.manager.get_experiment(experiment_id)
        assert experiment.status == ExperimentStatus.COMPLETED
        assert experiment.result is not None
        assert len(experiment.result.variant_results) == 2

    @patch('core.ab_testing.experiment_manager.get_event_bus')
    def test_event_publishing(self, mock_event_bus):
        """Test experiment event publishing"""
        mock_bus = Mock()
        mock_event_bus.return_value = mock_bus

        manager = ExperimentManager()

        config = ExperimentConfig(
            name="Event Test",
            experiment_type=ExperimentType.PLUGIN_COMPARISON
        )
        config.add_variant("control", {"plugin": "control"})
        config.add_variant("treatment", {"plugin": "treatment"})

        experiment_id = manager.create_experiment(config)

        # Verify event was published
        mock_bus.publish.assert_called()

        manager.cleanup()

    def test_summary_statistics(self):
        """Test experiment summary statistics"""
        # Create some experiments
        for i in range(2):
            config = ExperimentConfig(
                name=f"Summary Test {i}",
                experiment_type=ExperimentType.PLUGIN_COMPARISON
            )
            config.add_variant("control", {"plugin": "control"})
            config.add_variant("treatment", {"plugin": "treatment"})
            self.manager.create_experiment(config)

        stats = self.manager.get_summary_stats()

        assert stats["total_experiments"] == 2
        assert stats["active_experiments"] == 0
        assert "pending" in stats["status_distribution"]
        assert stats["status_distribution"]["pending"] == 2


class TestIntegration:
    """Integration tests for the complete A/B testing workflow"""

    def test_complete_ab_testing_workflow(self):
        """Test complete A/B testing workflow from creation to analysis"""
        # Initialize components
        manager = ExperimentManager({'max_concurrent': 1})
        comparator = ConfigurationComparator()
        analyzer = StatisticalAnalyzer()

        try:
            # Create experiment configuration
            experiment_config = comparator.create_battle_strategy_comparison()
            experiment_config.sample_size_per_variant = 10  # Minimum required
            experiment_config.max_runtime_seconds = 10

            # Create and start experiment
            experiment_id = manager.create_experiment(experiment_config)
            success = manager.start_experiment(experiment_id)
            assert success == True

            # Wait for completion
            max_wait = 15
            start_time = time.time()

            while time.time() - start_time < max_wait:
                experiment = manager.get_experiment(experiment_id)
                if experiment.status in [ExperimentStatus.COMPLETED, ExperimentStatus.FAILED]:
                    break
                time.sleep(0.5)

            # Get experiment results
            experiment = manager.get_experiment(experiment_id)
            assert experiment.status == ExperimentStatus.COMPLETED
            assert experiment.result is not None

            # Perform statistical analysis
            analysis = analyzer.analyze_experiment(experiment.result)

            assert analysis.experiment_id == experiment_id
            assert len(analysis.primary_metric_results) > 0
            assert len(analysis.recommendations) > 0
            assert analysis.summary != ""

        finally:
            manager.cleanup()

    def test_plugin_configuration_experiment(self):
        """Test plugin-specific configuration experiment"""
        manager = ExperimentManager({'max_concurrent': 1})
        comparator = ConfigurationComparator()

        try:
            # Create plugin comparison
            base_config = {"max_actions": 500}
            plugin_variants = {
                "high_aggression": {"aggressive_battle": {"aggression": 0.9}},
                "low_aggression": {"aggressive_battle": {"aggression": 0.3}}
            }

            experiment_config = comparator.create_plugin_comparison(
                base_config, plugin_variants, "Aggression Test"
            )
            experiment_config.sample_size_per_variant = 10  # Minimum required
            experiment_config.max_runtime_seconds = 8

            # Execute experiment
            experiment_id = manager.create_experiment(experiment_config)
            manager.start_experiment(experiment_id)

            # Verify execution
            status = manager.get_experiment_status(experiment_id)
            assert status["name"] == "Aggression Test"
            assert len(status["live_metrics"]) == 2

        finally:
            manager.cleanup()


# Pytest markers for test categorization
pytestmark = [
    pytest.mark.unit,
    pytest.mark.ab_testing,
    pytest.mark.experimental
]