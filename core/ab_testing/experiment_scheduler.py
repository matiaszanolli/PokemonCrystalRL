"""
Experiment Scheduler - Automated A/B Testing Execution

This module provides automated experiment scheduling, queuing, and execution
management for hands-free A/B testing workflows.
"""

import logging
import threading
import time
import queue
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum

from .experiment_models import ExperimentConfig, ExperimentStatus
from .experiment_manager import ExperimentManager
from .statistical_analyzer import StatisticalAnalyzer
from core.event_system import EventType, Event, EventSubscriber, get_event_bus


class ScheduleType(Enum):
    """Types of experiment scheduling"""
    IMMEDIATE = "immediate"
    DELAYED = "delayed"
    RECURRING = "recurring"
    CONDITIONAL = "conditional"


class ScheduleStatus(Enum):
    """Status of scheduled experiments"""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ScheduleConfig:
    """Configuration for experiment scheduling"""
    schedule_type: ScheduleType

    # Timing configuration
    start_time: Optional[datetime] = None
    delay_seconds: Optional[int] = None

    # Recurring schedule
    interval_seconds: Optional[int] = None
    max_runs: Optional[int] = None

    # Conditional execution
    condition_check: Optional[str] = None  # Python expression to evaluate

    # Execution settings
    auto_analyze: bool = True
    auto_archive: bool = True
    max_concurrent: int = 1
    retry_on_failure: bool = True
    max_retries: int = 3

    # Dependencies
    depends_on: List[str] = None  # List of experiment IDs that must complete first

    def __post_init__(self):
        if self.depends_on is None:
            self.depends_on = []


@dataclass
class ScheduledExperiment:
    """A scheduled experiment with automation settings"""
    schedule_id: str
    experiment_config: ExperimentConfig
    schedule_config: ScheduleConfig

    # State tracking
    status: ScheduleStatus = ScheduleStatus.PENDING
    created_time: datetime = None
    scheduled_time: Optional[datetime] = None
    started_time: Optional[datetime] = None
    completed_time: Optional[datetime] = None

    # Execution tracking
    attempt_count: int = 0
    experiment_id: Optional[str] = None
    error_message: Optional[str] = None

    # Results
    results_archived: bool = False
    analysis_complete: bool = False

    def __post_init__(self):
        if self.created_time is None:
            self.created_time = datetime.now()
        if self.schedule_id is None:
            self.schedule_id = str(uuid.uuid4())


class ExperimentScheduler(EventSubscriber):
    """
    Automated experiment scheduler and execution manager.

    Handles experiment queuing, scheduling, automatic execution,
    analysis, and result archiving for hands-free A/B testing.
    """

    def __init__(self, experiment_manager: ExperimentManager, config: Dict[str, Any] = None):
        """Initialize the experiment scheduler"""
        self.config = config or {}
        self.logger = logging.getLogger("ExperimentScheduler")

        # Core components
        self.experiment_manager = experiment_manager
        self.statistical_analyzer = StatisticalAnalyzer()
        self.event_bus = get_event_bus()

        # Scheduling state
        self.scheduled_experiments: Dict[str, ScheduledExperiment] = {}
        self.execution_queue = queue.PriorityQueue()
        self.running_experiments: Dict[str, str] = {}  # schedule_id -> experiment_id

        # Thread management
        self.scheduler_thread: Optional[threading.Thread] = None
        self.executor_thread: Optional[threading.Thread] = None
        self.running = False
        self.scheduler_lock = threading.Lock()

        # Configuration
        self.max_concurrent_scheduled = self.config.get('max_concurrent_scheduled', 3)
        self.check_interval = self.config.get('check_interval_seconds', 10)
        self.results_directory = Path(self.config.get('results_dir', 'data/automated_experiments'))
        self.results_directory.mkdir(parents=True, exist_ok=True)

        # Event subscription
        self.event_bus.subscribe(self)

        self.logger.info("ExperimentScheduler initialized")

    def get_subscribed_events(self):
        """Get events this scheduler subscribes to"""
        return {
            EventType.TRAINING_STOPPED,
            EventType.SYSTEM_ERROR
        }

    def handle_event(self, event: Event):
        """Handle system events"""
        if event.event_type == EventType.TRAINING_STOPPED:
            # Check if we can start queued experiments
            self._check_pending_experiments()
        elif event.event_type == EventType.SYSTEM_ERROR:
            # Handle system errors that might affect scheduled experiments
            self._handle_system_error(event)

    def start(self):
        """Start the automated scheduler"""
        if self.running:
            self.logger.warning("Scheduler already running")
            return

        self.running = True

        # Start scheduler thread
        self.scheduler_thread = threading.Thread(
            target=self._scheduler_loop,
            name="ExperimentScheduler",
            daemon=True
        )
        self.scheduler_thread.start()

        # Start executor thread
        self.executor_thread = threading.Thread(
            target=self._executor_loop,
            name="ExperimentExecutor",
            daemon=True
        )
        self.executor_thread.start()

        self.logger.info("Automated experiment scheduler started")

    def stop(self):
        """Stop the automated scheduler"""
        self.running = False

        if self.scheduler_thread and self.scheduler_thread.is_alive():
            self.scheduler_thread.join(timeout=5.0)

        if self.executor_thread and self.executor_thread.is_alive():
            self.executor_thread.join(timeout=5.0)

        self.logger.info("Automated experiment scheduler stopped")

    def schedule_experiment(self,
                          experiment_config: ExperimentConfig,
                          schedule_config: ScheduleConfig) -> str:
        """
        Schedule an experiment for automated execution.

        Args:
            experiment_config: The experiment configuration
            schedule_config: The scheduling configuration

        Returns:
            The schedule ID for tracking
        """
        scheduled_exp = ScheduledExperiment(
            schedule_id=str(uuid.uuid4()),
            experiment_config=experiment_config,
            schedule_config=schedule_config
        )

        # Calculate scheduled time
        if schedule_config.schedule_type == ScheduleType.IMMEDIATE:
            scheduled_exp.scheduled_time = datetime.now()
        elif schedule_config.schedule_type == ScheduleType.DELAYED:
            if schedule_config.delay_seconds:
                scheduled_exp.scheduled_time = datetime.now() + timedelta(seconds=schedule_config.delay_seconds)
            elif schedule_config.start_time:
                scheduled_exp.scheduled_time = schedule_config.start_time
        elif schedule_config.schedule_type == ScheduleType.RECURRING:
            scheduled_exp.scheduled_time = datetime.now()

        with self.scheduler_lock:
            self.scheduled_experiments[scheduled_exp.schedule_id] = scheduled_exp

        self.logger.info(f"Scheduled experiment: {experiment_config.name} (ID: {scheduled_exp.schedule_id[:8]}...)")

        # Save schedule to disk
        self._save_schedule(scheduled_exp)

        return scheduled_exp.schedule_id

    def cancel_scheduled_experiment(self, schedule_id: str) -> bool:
        """Cancel a scheduled experiment"""
        with self.scheduler_lock:
            if schedule_id in self.scheduled_experiments:
                scheduled_exp = self.scheduled_experiments[schedule_id]

                if scheduled_exp.status in [ScheduleStatus.PENDING, ScheduleStatus.QUEUED]:
                    scheduled_exp.status = ScheduleStatus.CANCELLED
                    self.logger.info(f"Cancelled scheduled experiment: {schedule_id[:8]}...")
                    return True
                elif scheduled_exp.status == ScheduleStatus.RUNNING:
                    # Try to stop the running experiment
                    if scheduled_exp.experiment_id:
                        if self.experiment_manager.stop_experiment(scheduled_exp.experiment_id):
                            scheduled_exp.status = ScheduleStatus.CANCELLED
                            self.logger.info(f"Cancelled running experiment: {schedule_id[:8]}...")
                            return True

        return False

    def get_scheduled_experiments(self, status_filter: Optional[ScheduleStatus] = None) -> List[ScheduledExperiment]:
        """Get list of scheduled experiments, optionally filtered by status"""
        with self.scheduler_lock:
            experiments = list(self.scheduled_experiments.values())

        if status_filter:
            experiments = [exp for exp in experiments if exp.status == status_filter]

        return sorted(experiments, key=lambda x: x.created_time, reverse=True)

    def get_schedule_status(self, schedule_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed status of a scheduled experiment"""
        with self.scheduler_lock:
            if schedule_id not in self.scheduled_experiments:
                return None

            scheduled_exp = self.scheduled_experiments[schedule_id]

        status = {
            'schedule_id': schedule_id,
            'experiment_name': scheduled_exp.experiment_config.name,
            'status': scheduled_exp.status.value,
            'schedule_type': scheduled_exp.schedule_config.schedule_type.value,
            'created_time': scheduled_exp.created_time.isoformat(),
            'scheduled_time': scheduled_exp.scheduled_time.isoformat() if scheduled_exp.scheduled_time else None,
            'started_time': scheduled_exp.started_time.isoformat() if scheduled_exp.started_time else None,
            'completed_time': scheduled_exp.completed_time.isoformat() if scheduled_exp.completed_time else None,
            'attempt_count': scheduled_exp.attempt_count,
            'experiment_id': scheduled_exp.experiment_id,
            'auto_analyze': scheduled_exp.schedule_config.auto_analyze,
            'auto_archive': scheduled_exp.schedule_config.auto_archive,
            'analysis_complete': scheduled_exp.analysis_complete,
            'results_archived': scheduled_exp.results_archived
        }

        if scheduled_exp.error_message:
            status['error_message'] = scheduled_exp.error_message

        return status

    def _scheduler_loop(self):
        """Main scheduler loop that checks for experiments to queue"""
        while self.running:
            try:
                self._check_pending_experiments()
                time.sleep(self.check_interval)
            except Exception as e:
                self.logger.error(f"Scheduler loop error: {e}")
                time.sleep(5)

    def _executor_loop(self):
        """Main executor loop that processes queued experiments"""
        while self.running:
            try:
                # Get next experiment from queue (with timeout)
                try:
                    priority, schedule_id = self.execution_queue.get(timeout=5.0)
                except queue.Empty:
                    continue

                # Execute the experiment
                self._execute_scheduled_experiment(schedule_id)
                self.execution_queue.task_done()

            except Exception as e:
                self.logger.error(f"Executor loop error: {e}")
                time.sleep(1)

    def _check_pending_experiments(self):
        """Check for experiments that are ready to be queued"""
        current_time = datetime.now()

        with self.scheduler_lock:
            experiments_to_queue = []

            for scheduled_exp in self.scheduled_experiments.values():
                if scheduled_exp.status != ScheduleStatus.PENDING:
                    continue

                # Check if it's time to queue this experiment
                should_queue = False

                if scheduled_exp.schedule_config.schedule_type == ScheduleType.IMMEDIATE:
                    should_queue = True
                elif scheduled_exp.schedule_config.schedule_type in [ScheduleType.DELAYED, ScheduleType.RECURRING]:
                    if scheduled_exp.scheduled_time and current_time >= scheduled_exp.scheduled_time:
                        should_queue = True
                elif scheduled_exp.schedule_config.schedule_type == ScheduleType.CONDITIONAL:
                    should_queue = self._evaluate_condition(scheduled_exp)

                # Check dependencies
                if should_queue and scheduled_exp.schedule_config.depends_on:
                    should_queue = self._check_dependencies(scheduled_exp)

                # Check concurrent limits
                if should_queue:
                    running_count = len([exp for exp in self.scheduled_experiments.values()
                                       if exp.status == ScheduleStatus.RUNNING])
                    if running_count >= self.max_concurrent_scheduled:
                        should_queue = False

                if should_queue:
                    experiments_to_queue.append(scheduled_exp)

        # Queue experiments
        for scheduled_exp in experiments_to_queue:
            self._queue_experiment(scheduled_exp)

    def _queue_experiment(self, scheduled_exp: ScheduledExperiment):
        """Queue an experiment for execution"""
        scheduled_exp.status = ScheduleStatus.QUEUED

        # Calculate priority (lower number = higher priority)
        priority = int(scheduled_exp.scheduled_time.timestamp()) if scheduled_exp.scheduled_time else int(time.time())

        self.execution_queue.put((priority, scheduled_exp.schedule_id))
        self.logger.info(f"Queued experiment: {scheduled_exp.experiment_config.name}")

    def _execute_scheduled_experiment(self, schedule_id: str):
        """Execute a scheduled experiment"""
        with self.scheduler_lock:
            if schedule_id not in self.scheduled_experiments:
                return

            scheduled_exp = self.scheduled_experiments[schedule_id]

        try:
            scheduled_exp.status = ScheduleStatus.RUNNING
            scheduled_exp.started_time = datetime.now()
            scheduled_exp.attempt_count += 1

            self.logger.info(f"Starting automated experiment: {scheduled_exp.experiment_config.name}")

            # Create and start the experiment
            experiment_id = self.experiment_manager.create_experiment(scheduled_exp.experiment_config)
            scheduled_exp.experiment_id = experiment_id

            with self.scheduler_lock:
                self.running_experiments[schedule_id] = experiment_id

            # Start the experiment
            if not self.experiment_manager.start_experiment(experiment_id):
                raise Exception("Failed to start experiment")

            # Wait for experiment completion
            self._wait_for_experiment_completion(scheduled_exp)

            # Post-processing
            self._post_process_experiment(scheduled_exp)

        except Exception as e:
            self.logger.error(f"Failed to execute scheduled experiment {schedule_id[:8]}...: {e}")
            scheduled_exp.status = ScheduleStatus.FAILED
            scheduled_exp.error_message = str(e)

            # Retry if configured
            if (scheduled_exp.schedule_config.retry_on_failure and
                scheduled_exp.attempt_count < scheduled_exp.schedule_config.max_retries):

                self.logger.info(f"Retrying experiment {schedule_id[:8]}... (attempt {scheduled_exp.attempt_count + 1})")
                scheduled_exp.status = ScheduleStatus.PENDING
                # Schedule retry with delay
                scheduled_exp.scheduled_time = datetime.now() + timedelta(seconds=30)

        finally:
            with self.scheduler_lock:
                if schedule_id in self.running_experiments:
                    del self.running_experiments[schedule_id]

    def _wait_for_experiment_completion(self, scheduled_exp: ScheduledExperiment):
        """Wait for experiment to complete"""
        experiment_id = scheduled_exp.experiment_id

        while self.running:
            experiment = self.experiment_manager.get_experiment(experiment_id)
            if not experiment:
                raise Exception("Experiment not found")

            if experiment.status == ExperimentStatus.COMPLETED:
                scheduled_exp.status = ScheduleStatus.COMPLETED
                scheduled_exp.completed_time = datetime.now()
                break
            elif experiment.status in [ExperimentStatus.FAILED, ExperimentStatus.CANCELLED]:
                raise Exception(f"Experiment {experiment.status.value}")

            time.sleep(5)  # Check every 5 seconds

    def _post_process_experiment(self, scheduled_exp: ScheduledExperiment):
        """Perform post-processing after experiment completion"""
        experiment_id = scheduled_exp.experiment_id
        experiment = self.experiment_manager.get_experiment(experiment_id)

        if not experiment:
            return

        # Automatic analysis
        if scheduled_exp.schedule_config.auto_analyze:
            try:
                analysis = self.statistical_analyzer.analyze_experiment_results(experiment.result)
                scheduled_exp.analysis_complete = True

                # Save analysis results
                self._save_analysis(scheduled_exp, analysis)

                self.logger.info(f"Completed automated analysis for {scheduled_exp.schedule_id[:8]}...")
            except Exception as e:
                self.logger.error(f"Failed to analyze experiment {scheduled_exp.schedule_id[:8]}...: {e}")

        # Automatic archiving
        if scheduled_exp.schedule_config.auto_archive:
            try:
                self._archive_results(scheduled_exp, experiment)
                scheduled_exp.results_archived = True

                self.logger.info(f"Archived results for {scheduled_exp.schedule_id[:8]}...")
            except Exception as e:
                self.logger.error(f"Failed to archive experiment {scheduled_exp.schedule_id[:8]}...: {e}")

        # Handle recurring experiments
        if scheduled_exp.schedule_config.schedule_type == ScheduleType.RECURRING:
            self._schedule_next_recurrence(scheduled_exp)

    def _schedule_next_recurrence(self, scheduled_exp: ScheduledExperiment):
        """Schedule the next recurrence of a recurring experiment"""
        if not scheduled_exp.schedule_config.interval_seconds:
            return

        # Check if we've reached max runs
        if (scheduled_exp.schedule_config.max_runs and
            scheduled_exp.attempt_count >= scheduled_exp.schedule_config.max_runs):
            return

        # Create new scheduled experiment for next run
        next_config = ScheduleConfig(
            schedule_type=ScheduleType.DELAYED,
            delay_seconds=scheduled_exp.schedule_config.interval_seconds,
            auto_analyze=scheduled_exp.schedule_config.auto_analyze,
            auto_archive=scheduled_exp.schedule_config.auto_archive,
            max_concurrent=scheduled_exp.schedule_config.max_concurrent,
            retry_on_failure=scheduled_exp.schedule_config.retry_on_failure,
            max_retries=scheduled_exp.schedule_config.max_retries
        )

        # Update experiment name to include run number
        next_experiment_config = scheduled_exp.experiment_config
        next_experiment_config.name = f"{scheduled_exp.experiment_config.name} (Run {scheduled_exp.attempt_count + 1})"

        self.schedule_experiment(next_experiment_config, next_config)

        self.logger.info(f"Scheduled next recurrence of {scheduled_exp.experiment_config.name}")

    def _evaluate_condition(self, scheduled_exp: ScheduledExperiment) -> bool:
        """Evaluate conditional scheduling criteria"""
        if not scheduled_exp.schedule_config.condition_check:
            return True

        try:
            # Create evaluation context
            context = {
                'datetime': datetime,
                'time': time,
                'experiment_manager': self.experiment_manager,
                'running_experiments': len(self.running_experiments),
                'completed_experiments': len([exp for exp in self.scheduled_experiments.values()
                                            if exp.status == ScheduleStatus.COMPLETED])
            }

            # Evaluate the condition
            result = eval(scheduled_exp.schedule_config.condition_check, {"__builtins__": {}}, context)
            return bool(result)

        except Exception as e:
            self.logger.error(f"Error evaluating condition for {scheduled_exp.schedule_id[:8]}...: {e}")
            return False

    def _check_dependencies(self, scheduled_exp: ScheduledExperiment) -> bool:
        """Check if all dependencies are satisfied"""
        for dep_id in scheduled_exp.schedule_config.depends_on:
            dep_exp = self.scheduled_experiments.get(dep_id)
            if not dep_exp or dep_exp.status != ScheduleStatus.COMPLETED:
                return False
        return True

    def _save_schedule(self, scheduled_exp: ScheduledExperiment):
        """Save schedule configuration to disk"""
        schedule_file = self.results_directory / f"schedule_{scheduled_exp.schedule_id}.json"

        schedule_data = {
            'schedule_id': scheduled_exp.schedule_id,
            'experiment_config': asdict(scheduled_exp.experiment_config),
            'schedule_config': asdict(scheduled_exp.schedule_config),
            'created_time': scheduled_exp.created_time.isoformat(),
            'status': scheduled_exp.status.value
        }

        with open(schedule_file, 'w') as f:
            json.dump(schedule_data, f, indent=2, default=str)

    def _save_analysis(self, scheduled_exp: ScheduledExperiment, analysis):
        """Save analysis results to disk"""
        analysis_file = self.results_directory / f"analysis_{scheduled_exp.schedule_id}.json"

        with open(analysis_file, 'w') as f:
            json.dump(asdict(analysis), f, indent=2, default=str)

    def _archive_results(self, scheduled_exp: ScheduledExperiment, experiment):
        """Archive experiment results"""
        archive_file = self.results_directory / f"results_{scheduled_exp.schedule_id}.json"

        results_data = {
            'schedule_id': scheduled_exp.schedule_id,
            'experiment_id': experiment.id,
            'experiment_config': asdict(experiment.config),
            'experiment_result': asdict(experiment.result) if experiment.result else None,
            'execution_summary': {
                'started_time': scheduled_exp.started_time.isoformat() if scheduled_exp.started_time else None,
                'completed_time': scheduled_exp.completed_time.isoformat() if scheduled_exp.completed_time else None,
                'attempt_count': scheduled_exp.attempt_count,
                'analysis_complete': scheduled_exp.analysis_complete,
                'auto_archived': True
            }
        }

        with open(archive_file, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)

    def _handle_system_error(self, event: Event):
        """Handle system errors that might affect scheduled experiments"""
        # Mark running experiments as failed if system error is critical
        if event.data and event.data.get('critical', False):
            with self.scheduler_lock:
                for schedule_id, experiment_id in list(self.running_experiments.items()):
                    if schedule_id in self.scheduled_experiments:
                        scheduled_exp = self.scheduled_experiments[schedule_id]
                        scheduled_exp.status = ScheduleStatus.FAILED
                        scheduled_exp.error_message = f"System error: {event.data.get('message', 'Unknown error')}"

                self.running_experiments.clear()

    def get_automation_stats(self) -> Dict[str, Any]:
        """Get automation statistics"""
        with self.scheduler_lock:
            total = len(self.scheduled_experiments)
            status_counts = {}

            for scheduled_exp in self.scheduled_experiments.values():
                status = scheduled_exp.status.value
                status_counts[status] = status_counts.get(status, 0) + 1

        return {
            'total_scheduled': total,
            'currently_running': len(self.running_experiments),
            'queue_size': self.execution_queue.qsize(),
            'status_distribution': status_counts,
            'automation_active': self.running,
            'results_directory': str(self.results_directory)
        }