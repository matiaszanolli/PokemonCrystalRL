#!/usr/bin/env python3
"""
A/B Testing Framework Demo

This example demonstrates how to use the comprehensive A/B testing framework
to compare different plugin configurations, agent strategies, and training parameters.

Usage:
    python examples/ab_testing_demo.py
"""

import logging
import time
from core.ab_testing import (
    ExperimentManager, ConfigurationComparator, StatisticalAnalyzer,
    ExperimentConfig, ExperimentType, ExperimentStatus
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ABTestingDemo")


def demo_battle_strategy_comparison():
    """Demonstrate battle strategy A/B testing"""
    logger.info("=== Battle Strategy A/B Testing Demo ===")

    # Initialize components
    manager = ExperimentManager({'max_concurrent': 1})
    comparator = ConfigurationComparator()
    analyzer = StatisticalAnalyzer()

    try:
        # Create battle strategy comparison experiment
        experiment_config = comparator.create_battle_strategy_comparison()
        experiment_config.sample_size_per_variant = 20  # Small for demo
        experiment_config.max_runtime_seconds = 30

        logger.info(f"Created experiment: {experiment_config.name}")
        logger.info(f"Variants: {experiment_config.get_variant_names()}")

        # Create and start experiment
        experiment_id = manager.create_experiment(experiment_config)
        logger.info(f"Experiment ID: {experiment_id}")

        success = manager.start_experiment(experiment_id)
        if not success:
            logger.error("Failed to start experiment")
            return

        logger.info("Experiment started, monitoring progress...")

        # Monitor experiment progress
        while True:
            status = manager.get_experiment_status(experiment_id)
            if not status:
                break

            logger.info(f"Progress: {status['progress']:.1f}% - Status: {status['status']}")

            if status['status'] in ['completed', 'failed', 'cancelled']:
                break

            time.sleep(2)

        # Get final results
        experiment = manager.get_experiment(experiment_id)
        if experiment and experiment.result:
            logger.info("Experiment completed! Analyzing results...")

            # Perform statistical analysis
            analysis = analyzer.analyze_experiment(experiment.result)

            logger.info(f"Analysis Summary: {analysis.summary}")
            logger.info(f"Significant Results: {analysis.has_significant_results}")
            if analysis.winning_variant:
                logger.info(f"Winning Variant: {analysis.winning_variant} (confidence: {analysis.confidence_score:.1%})")

            logger.info("Recommendations:")
            for rec in analysis.recommendations:
                logger.info(f"  - {rec}")

    except Exception as e:
        logger.error(f"Demo failed: {str(e)}")
    finally:
        manager.cleanup()


def demo_plugin_comparison():
    """Demonstrate custom plugin configuration comparison"""
    logger.info("=== Custom Plugin Configuration Demo ===")

    manager = ExperimentManager({'max_concurrent': 1})
    comparator = ConfigurationComparator()

    try:
        # Define custom plugin variants
        base_config = {
            "max_actions": 500,
            "llm_interval": 10,
            "headless": True
        }

        plugin_variants = {
            "high_aggression": {
                "aggressive_battle": {
                    "aggression_level": 0.9,
                    "risk_tolerance": 0.8
                }
            },
            "medium_aggression": {
                "aggressive_battle": {
                    "aggression_level": 0.6,
                    "risk_tolerance": 0.5
                }
            },
            "low_aggression": {
                "aggressive_battle": {
                    "aggression_level": 0.3,
                    "risk_tolerance": 0.2
                }
            }
        }

        # Create experiment
        experiment_config = comparator.create_plugin_comparison(
            base_config, plugin_variants, "Aggression Level Test"
        )
        experiment_config.sample_size_per_variant = 15
        experiment_config.max_runtime_seconds = 25

        logger.info(f"Created custom experiment: {experiment_config.name}")

        # Quick execution demo
        experiment_id = manager.create_experiment(experiment_config)
        manager.start_experiment(experiment_id)

        # Monitor briefly
        for i in range(10):
            status = manager.get_experiment_status(experiment_id)
            if status['status'] in ['completed', 'failed']:
                break
            logger.info(f"Running... Progress: {status['progress']:.1f}%")
            time.sleep(2.5)

        logger.info("Custom plugin comparison completed!")

    finally:
        manager.cleanup()


def demo_multi_agent_comparison():
    """Demonstrate multi-agent strategy comparison"""
    logger.info("=== Multi-Agent Strategy Demo ===")

    manager = ExperimentManager({'max_concurrent': 1})
    comparator = ConfigurationComparator()

    try:
        # Create multi-agent comparison
        experiment_config = comparator.create_multi_agent_comparison()
        experiment_config.sample_size_per_variant = 12
        experiment_config.max_runtime_seconds = 20

        logger.info(f"Created experiment: {experiment_config.name}")
        logger.info(f"Testing {len(experiment_config.variants)} agent coordination strategies")

        # Execute
        experiment_id = manager.create_experiment(experiment_config)
        manager.start_experiment(experiment_id)

        # Brief monitoring
        for i in range(8):
            status = manager.get_experiment_status(experiment_id)
            if status['status'] in ['completed', 'failed']:
                break
            time.sleep(2.5)

        logger.info("Multi-agent comparison completed!")

    finally:
        manager.cleanup()


def demo_experiment_manager_features():
    """Demonstrate experiment manager capabilities"""
    logger.info("=== Experiment Manager Features Demo ===")

    manager = ExperimentManager({'max_concurrent': 2})
    comparator = ConfigurationComparator()

    try:
        # Create multiple experiments
        experiments = []

        # Experiment 1: Battle strategies
        exp1_config = comparator.create_battle_strategy_comparison()
        exp1_config.sample_size_per_variant = 10
        exp1_config.max_runtime_seconds = 15
        exp1_id = manager.create_experiment(exp1_config)
        experiments.append(exp1_id)

        # Experiment 2: Exploration patterns
        exp2_config = comparator.create_exploration_pattern_comparison()
        exp2_config.sample_size_per_variant = 10
        exp2_config.max_runtime_seconds = 15
        exp2_id = manager.create_experiment(exp2_config)
        experiments.append(exp2_id)

        logger.info(f"Created {len(experiments)} experiments")

        # List all experiments
        all_experiments = manager.list_experiments()
        logger.info(f"Total experiments in manager: {len(all_experiments)}")

        # Show summary statistics
        stats = manager.get_summary_stats()
        logger.info(f"Manager stats: {stats}")

        # Start first experiment
        success = manager.start_experiment(exp1_id)
        logger.info(f"Started experiment 1: {success}")

        # Try to start second (should work since max_concurrent=2)
        success = manager.start_experiment(exp2_id)
        logger.info(f"Started experiment 2: {success}")

        # Brief monitoring
        for i in range(6):
            time.sleep(2.5)
            for exp_id in experiments:
                status = manager.get_experiment_status(exp_id)
                if status:
                    logger.info(f"Exp {exp_id[:8]}: {status['progress']:.1f}% ({status['status']})")

        logger.info("Multiple experiment demo completed!")

    finally:
        manager.cleanup()


def main():
    """Run all A/B testing demos"""
    logger.info("Starting A/B Testing Framework Demonstration")
    logger.info("=" * 60)

    try:
        # Run demos
        demo_battle_strategy_comparison()
        print()

        demo_plugin_comparison()
        print()

        demo_multi_agent_comparison()
        print()

        demo_experiment_manager_features()

    except KeyboardInterrupt:
        logger.info("Demo interrupted by user")
    except Exception as e:
        logger.error(f"Demo failed: {str(e)}")

    logger.info("=" * 60)
    logger.info("A/B Testing Framework Demo Complete!")
    logger.info("Key Features Demonstrated:")
    logger.info("  ✓ Battle strategy comparison")
    logger.info("  ✓ Custom plugin configuration testing")
    logger.info("  ✓ Multi-agent strategy evaluation")
    logger.info("  ✓ Experiment management and monitoring")
    logger.info("  ✓ Statistical analysis and recommendations")


if __name__ == "__main__":
    main()