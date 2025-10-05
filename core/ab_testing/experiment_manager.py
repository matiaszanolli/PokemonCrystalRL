"""
Experiment Manager - Core orchestration for A/B testing experiments

This module handles the lifecycle of A/B testing experiments, including:
- Experiment creation and configuration
- Execution coordination with training system
- Real-time monitoring and data collection
- Integration with plugin and agent systems
"""

import logging
import threading
import time
import random
from typing import Dict, Any, List, Optional, Callable
from pathlib import Path
import json

from .experiment_models import (
    Experiment, ExperimentConfig, ExperimentResult, ExperimentStatus, ExperimentType,
    PerformanceMetrics, MetricCollection
)
from core.event_system import EventType, Event, EventSubscriber, get_event_bus
from core.plugin_system import get_plugin_registry, PluginType


class ExperimentManager(EventSubscriber):
    """
    Core experiment management system for A/B testing.

    Handles experiment lifecycle, coordinates with training systems,
    and provides real-time monitoring of experiment progress.
    """

    def __init__(self, config: Dict[str, Any] = None, websocket_handler=None):
        """Initialize the experiment manager"""
        self.config = config or {}
        self.logger = logging.getLogger("ExperimentManager")

        # Experiment storage
        self.experiments: Dict[str, Experiment] = {}
        self.active_experiments: List[str] = []
        self.experiment_lock = threading.Lock()

        # Integration components
        self.event_bus = get_event_bus()
        self.plugin_registry = get_plugin_registry()
        self.websocket_handler = websocket_handler

        # Execution management
        self.max_concurrent_experiments = self.config.get('max_concurrent', 2)
        self.experiment_threads: Dict[str, threading.Thread] = {}

        # Data persistence
        self.results_directory = Path(self.config.get('results_dir', 'data/ab_test_results'))
        self.results_directory.mkdir(parents=True, exist_ok=True)

        # Event subscription
        self.event_bus.subscribe(self)

        self.logger.info("ExperimentManager initialized")

    def create_experiment(self, config: ExperimentConfig) -> str:
        """
        Create a new A/B testing experiment.

        Args:
            config: Experiment configuration

        Returns:
            experiment_id: Unique identifier for the experiment
        """
        with self.experiment_lock:
            experiment = Experiment(config=config)

            if experiment.status == ExperimentStatus.FAILED:
                self.logger.error(f"Failed to create experiment: {experiment.error_message}")
                raise ValueError(f"Invalid experiment configuration: {experiment.error_message}")

            self.experiments[experiment.id] = experiment
            self.logger.info(f"Created experiment '{config.name}' with ID: {experiment.id}")

            # Publish experiment creation event
            self._publish_experiment_event("EXPERIMENT_CREATED", experiment)

            return experiment.id

    def start_experiment(self, experiment_id: str) -> bool:
        """
        Start an A/B testing experiment.

        Args:
            experiment_id: ID of the experiment to start

        Returns:
            success: True if experiment started successfully
        """
        with self.experiment_lock:
            if experiment_id not in self.experiments:
                self.logger.error(f"Experiment {experiment_id} not found")
                return False

            experiment = self.experiments[experiment_id]

            if experiment.status != ExperimentStatus.PENDING:
                self.logger.error(f"Experiment {experiment_id} is not in PENDING status")
                return False

            if len(self.active_experiments) >= self.max_concurrent_experiments:
                self.logger.warning(f"Maximum concurrent experiments reached ({self.max_concurrent_experiments})")
                return False

            # Start experiment
            experiment.start()
            self.active_experiments.append(experiment_id)

            # Start execution thread
            thread = threading.Thread(
                target=self._execute_experiment,
                args=(experiment_id,),
                name=f"Experiment-{experiment_id[:8]}"
            )
            thread.daemon = True
            self.experiment_threads[experiment_id] = thread
            thread.start()

            self.logger.info(f"Started experiment {experiment_id}")
            self._publish_experiment_event("EXPERIMENT_STARTED", experiment)

            # Notify WebSocket clients of status change
            self._sync_notify_websocket_update(experiment_id, 'status_change')

            return True

    def stop_experiment(self, experiment_id: str) -> bool:
        """
        Stop a running experiment.

        Args:
            experiment_id: ID of the experiment to stop

        Returns:
            success: True if experiment stopped successfully
        """
        with self.experiment_lock:
            if experiment_id not in self.experiments:
                return False

            experiment = self.experiments[experiment_id]

            if experiment.status == ExperimentStatus.RUNNING:
                experiment.status = ExperimentStatus.CANCELLED

                # Remove from active list
                if experiment_id in self.active_experiments:
                    self.active_experiments.remove(experiment_id)

                # Thread will check status and terminate
                self.logger.info(f"Stopped experiment {experiment_id}")
                self._publish_experiment_event("EXPERIMENT_STOPPED", experiment)

                # Notify WebSocket clients of status change
                self._sync_notify_websocket_update(experiment_id, 'status_change')

                return True

            return False

    def get_experiment(self, experiment_id: str) -> Optional[Experiment]:
        """Get experiment by ID"""
        return self.experiments.get(experiment_id)

    def list_experiments(self, status_filter: Optional[ExperimentStatus] = None) -> List[Experiment]:
        """
        List experiments, optionally filtered by status.

        Args:
            status_filter: Optional status to filter by

        Returns:
            experiments: List of experiments matching filter
        """
        experiments = list(self.experiments.values())

        if status_filter:
            experiments = [exp for exp in experiments if exp.status == status_filter]

        return experiments

    def get_experiment_status(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed status of an experiment.

        Args:
            experiment_id: ID of the experiment

        Returns:
            status: Detailed status information
        """
        experiment = self.get_experiment(experiment_id)
        if not experiment:
            return None

        return {
            'id': experiment.id,
            'name': experiment.config.name,
            'status': experiment.status.value,
            'progress': experiment.get_progress(),
            'current_variant': experiment.current_variant,
            'current_run': experiment.current_run,
            'total_runs': experiment.total_runs,
            'live_metrics': {
                variant: {
                    'sample_count': len(metrics.metrics.get('reward', [])),
                    'latest_reward': metrics.get_latest('reward'),
                    'avg_reward': metrics.get_average('reward')
                }
                for variant, metrics in experiment.live_metrics.items()
            },
            'error_message': experiment.error_message
        }

    def _execute_experiment(self, experiment_id: str):
        """
        Execute an experiment in a separate thread.

        Args:
            experiment_id: ID of the experiment to execute
        """
        try:
            experiment = self.experiments[experiment_id]
            self.logger.info(f"Executing experiment {experiment_id}: {experiment.config.name}")

            # Set random seed if specified
            if experiment.config.randomization_seed:
                random.seed(experiment.config.randomization_seed)

            # Execute each variant
            variant_names = experiment.config.get_variant_names()

            for run in range(experiment.config.sample_size_per_variant):
                if experiment.status != ExperimentStatus.RUNNING:
                    break

                for variant_name in variant_names:
                    if experiment.status != ExperimentStatus.RUNNING:
                        break

                    experiment.current_variant = variant_name
                    experiment.current_run = run * len(variant_names) + variant_names.index(variant_name) + 1

                    # Execute single run for this variant
                    self._execute_variant_run(experiment, variant_name, run)

                    # Notify WebSocket clients of progress update
                    self._sync_notify_websocket_update(experiment_id, 'progress')

                    # Small delay between runs
                    time.sleep(0.1)

            # Complete experiment if still running
            if experiment.status == ExperimentStatus.RUNNING:
                self._complete_experiment(experiment)

        except Exception as e:
            self.logger.error(f"Error executing experiment {experiment_id}: {str(e)}")
            experiment.fail(str(e))
            self._publish_experiment_event("EXPERIMENT_FAILED", experiment)

        finally:
            # Cleanup
            with self.experiment_lock:
                if experiment_id in self.active_experiments:
                    self.active_experiments.remove(experiment_id)
                if experiment_id in self.experiment_threads:
                    del self.experiment_threads[experiment_id]

    def _execute_variant_run(self, experiment: Experiment, variant_name: str, run_number: int):
        """
        Execute a single run for a variant.

        Args:
            experiment: The experiment being executed
            variant_name: Name of the variant to run
            run_number: Run number for this variant
        """
        variant_config = experiment.config.variants[variant_name]

        # Create mock metrics for this run (in real implementation, this would
        # integrate with the actual training system)
        metrics = self._simulate_training_run(variant_config, experiment.config)

        # Store metrics
        experiment.live_metrics[variant_name].add_metric('reward', metrics['reward'])
        experiment.live_metrics[variant_name].add_metric('actions_per_second', metrics['actions_per_second'])
        experiment.live_metrics[variant_name].add_metric('battle_win_rate', metrics['battle_win_rate'])

        # Publish progress event
        self._publish_experiment_event("EXPERIMENT_PROGRESS", experiment)

        self.logger.debug(
            f"Completed run {run_number} for variant {variant_name} in experiment {experiment.id}: "
            f"reward={metrics['reward']:.2f}"
        )

    def _simulate_training_run(self, variant_config: Dict[str, Any], experiment_config: ExperimentConfig) -> Dict[str, float]:
        """
        Simulate a training run for testing purposes.

        In production, this would integrate with the actual training system.

        Args:
            variant_config: Configuration for this variant
            experiment_config: Overall experiment configuration

        Returns:
            metrics: Simulated performance metrics
        """
        # Simulate different performance based on configuration
        base_reward = 50.0

        # Simulate plugin effects
        if 'plugins' in variant_config:
            for plugin_name, plugin_config in variant_config['plugins'].items():
                if 'aggressive' in plugin_name.lower():
                    base_reward += random.uniform(5, 15)  # Aggressive strategies get bonus
                elif 'defensive' in plugin_name.lower():
                    base_reward += random.uniform(-5, 10)  # More conservative

        # Add noise
        reward = base_reward + random.uniform(-10, 10)
        actions_per_second = random.uniform(1.5, 3.0)
        battle_win_rate = random.uniform(0.6, 0.9)

        # Simulate training time
        time.sleep(random.uniform(0.5, 1.5))

        return {
            'reward': reward,
            'actions_per_second': actions_per_second,
            'battle_win_rate': battle_win_rate
        }

    def _complete_experiment(self, experiment: Experiment):
        """
        Complete an experiment and generate results.

        Args:
            experiment: The experiment to complete
        """
        experiment.complete()

        # Generate final results
        result = experiment.result
        if result:
            # Aggregate metrics for each variant
            for variant_name, live_metrics in experiment.live_metrics.items():
                performance_metrics = PerformanceMetrics()

                # Calculate aggregated metrics
                reward_samples = live_metrics.metrics.get('reward', [])
                if reward_samples:
                    performance_metrics.reward_samples = reward_samples
                    performance_metrics.total_reward = sum(reward_samples)

                action_times = live_metrics.metrics.get('actions_per_second', [])
                if action_times:
                    performance_metrics.actions_per_second = sum(action_times) / len(action_times)

                battle_rates = live_metrics.metrics.get('battle_win_rate', [])
                if battle_rates:
                    performance_metrics.battle_win_rate = sum(battle_rates) / len(battle_rates)

                result.add_variant_result(variant_name, performance_metrics)

            # Save results
            self._save_experiment_result(experiment)

        self.logger.info(f"Completed experiment {experiment.id}")
        self._publish_experiment_event("EXPERIMENT_COMPLETED", experiment)

        # Notify WebSocket clients of completion
        self._sync_notify_websocket_update(experiment.id, 'status_change')

    def _save_experiment_result(self, experiment: Experiment):
        """
        Save experiment results to disk.

        Args:
            experiment: The experiment to save
        """
        try:
            result_file = self.results_directory / f"experiment_{experiment.id}.json"
            with open(result_file, 'w') as f:
                json.dump(experiment.to_dict(), f, indent=2)

            self.logger.info(f"Saved experiment results to {result_file}")
        except Exception as e:
            self.logger.error(f"Failed to save experiment results: {str(e)}")

    def _publish_experiment_event(self, event_type: str, experiment: Experiment):
        """
        Publish an experiment-related event.

        Args:
            event_type: Type of event
            experiment: Experiment related to the event
        """
        try:
            event = Event(
                type=getattr(EventType, event_type, EventType.SYSTEM_STATUS),
                data={
                    'experiment_id': experiment.id,
                    'experiment_name': experiment.config.name,
                    'status': experiment.status.value,
                    'progress': experiment.get_progress()
                },
                source='ExperimentManager'
            )
            self.event_bus.publish(event)
        except Exception as e:
            self.logger.error(f"Failed to publish experiment event: {str(e)}")

    def handle_event(self, event: Event):
        """
        Handle incoming events.

        Args:
            event: Event to handle
        """
        # Handle training events that might affect experiments
        if event.type == EventType.TRAINING_STARTED:
            # Log training start
            self.logger.debug("Training started during experiment execution")
        elif event.type == EventType.TRAINING_STOPPED:
            # Log training stop
            self.logger.debug("Training stopped during experiment execution")
        elif event.type == EventType.EPISODE_ENDED:
            # Potentially useful for experiment progress tracking
            pass
        elif event.type == EventType.SYSTEM_ERROR:
            # Handle system errors
            self.logger.warning(f"System error during experiments: {event.data}")

    def get_subscribed_events(self):
        """Return set of event types this subscriber is interested in"""
        from core.event_system import EventType
        return {
            EventType.TRAINING_STARTED,
            EventType.TRAINING_STOPPED,
            EventType.EPISODE_ENDED,
            EventType.SYSTEM_ERROR
        }

    def cleanup(self):
        """Clean up experiment manager resources"""
        # Stop all active experiments
        active_experiments = list(self.active_experiments)
        for experiment_id in active_experiments:
            self.stop_experiment(experiment_id)

        # Wait for threads to complete (make copy of dict values to avoid runtime modification issues)
        threads = list(self.experiment_threads.values())
        for thread in threads:
            if thread.is_alive():
                thread.join(timeout=5.0)

        self.logger.info("ExperimentManager cleanup completed")

    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics for all experiments"""
        total_experiments = len(self.experiments)
        status_counts = {}

        for experiment in self.experiments.values():
            status = experiment.status.value
            status_counts[status] = status_counts.get(status, 0) + 1

        return {
            'total_experiments': total_experiments,
            'active_experiments': len(self.active_experiments),
            'status_distribution': status_counts,
            'results_directory': str(self.results_directory)
        }

    def set_websocket_handler(self, websocket_handler):
        """Set the WebSocket handler for real-time updates"""
        self.websocket_handler = websocket_handler
        self.logger.info("WebSocket handler connected to ExperimentManager")

    async def _notify_websocket_update(self, experiment_id: str, update_type: str = 'progress'):
        """Send real-time update via WebSocket"""
        if self.websocket_handler:
            try:
                import asyncio
                # Create a new event loop task to broadcast the update
                loop = asyncio.get_event_loop()
                loop.create_task(
                    self.websocket_handler.broadcast_experiment_update(experiment_id, update_type)
                )
            except Exception as e:
                self.logger.warning(f"Failed to send WebSocket update: {e}")

    def _sync_notify_websocket_update(self, experiment_id: str, update_type: str = 'progress'):
        """Synchronous wrapper for WebSocket notifications"""
        if self.websocket_handler:
            try:
                import asyncio
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # If loop is running, schedule the coroutine
                        loop.create_task(
                            self.websocket_handler.broadcast_experiment_update(experiment_id, update_type)
                        )
                    else:
                        # If no loop is running, run it
                        loop.run_until_complete(
                            self.websocket_handler.broadcast_experiment_update(experiment_id, update_type)
                        )
                except RuntimeError:
                    # No event loop, run in new loop
                    asyncio.run(
                        self.websocket_handler.broadcast_experiment_update(experiment_id, update_type)
                    )
            except Exception as e:
                self.logger.warning(f"Failed to send WebSocket update: {e}")