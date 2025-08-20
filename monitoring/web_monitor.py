"""
Web monitor module for Pokemon Crystal RL.

This module coordinates all monitoring components, handling:
- Integration between web server and interface
- Training data processing and streaming
- State management and synchronization
- Real-time updates and event handling
- Dashboard coordination
- Data persistence and recovery
"""

import asyncio
import logging
import threading
from typing import Dict, Any, Optional, List, Set
from pathlib import Path
import json
import time
from datetime import datetime
import weakref
from dataclasses import dataclass, asdict
import signal
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
from collections import deque
from enum import Enum, auto
from contextlib import contextmanager


class TrainingState(Enum):
    """Training process states."""
    INITIALIZING = auto()
    RUNNING = auto()
    PAUSED = auto()
    STOPPED = auto()
    ERROR = auto()


class ClientMetrics:
    """Client-side metric collection and aggregation."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.metrics = {
            'episode_rewards': deque(maxlen=window_size),
            'episode_lengths': deque(maxlen=window_size),
            'step_rewards': deque(maxlen=window_size),
            'actions_per_second': deque(maxlen=window_size),
            'inference_times': deque(maxlen=window_size),
            'processing_times': deque(maxlen=window_size)
        }
        self.text_frequency = {}
        self.recent_actions = deque(maxlen=50)
        self.start_time = time.time()
        
        # Performance tracking
        self.action_count = 0
        self.last_action_time = time.time()
        self.fps_counter = 0
        self.fps_start_time = time.time()
    
    def update_metric(self, name: str, value: float) -> None:
        """Update a metric with a new value."""
        if name in self.metrics:
            self.metrics[name].append(value)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current metric statistics."""
        stats = {}
        for name, values in self.metrics.items():
            if values:
                stats[name] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'current': float(values[-1]),
                    'history': list(values)
                }
            else:
                stats[name] = {
                    'mean': 0.0, 'std': 0.0,
                    'min': 0.0, 'max': 0.0,
                    'current': 0.0, 'history': []
                }
        return stats
    
    def update_action(self, action: str, timestamp: float = None) -> None:
        """Record an action with timestamp."""
        self.recent_actions.append({
            'action': action,
            'timestamp': timestamp or time.time()
        })
        self.action_count += 1
    
    def update_text(self, text: str) -> None:
        """Update text frequency tracking."""
        if text and len(text.strip()) > 0:
            clean_text = text.strip().upper()
            self.text_frequency[clean_text] = self.text_frequency.get(clean_text, 0) + 1
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        current_time = time.time()
        time_diff = current_time - self.last_action_time
        
        return {
            'actions_per_second': self.action_count / max(time_diff, 1.0),
            'uptime': current_time - self.start_time,
            'action_count': self.action_count,
            'recent_actions': list(self.recent_actions),
            'text_frequency': dict(sorted(
                self.text_frequency.items(),
                key=lambda x: x[1],
                reverse=True
            )[:20])  # Top 20 most frequent texts
        }

from .web_server import WebServer, ServerConfig
from .web_interface import WebInterface
from .trainer_monitor_bridge import TrainerMonitorBridge
from .data_bus import DataType, get_data_bus
from .error_handler import ErrorHandler, ErrorCategory, ErrorSeverity
from .stats_collector import StatsCollector
from .text_logger import TextLogger
from .database import DatabaseManager


@dataclass
class MonitorConfig:
    """Configuration for the web monitor."""
    db_path: str = "monitoring.db"
    host: str = "localhost"
    port: int = 8080
    static_dir: str = "static"
    data_dir: str = "monitor_data"
    update_interval: float = 1.0
    snapshot_interval: float = 300.0  # 5 minutes
    max_events: int = 1000
    max_snapshots: int = 100
    debug: bool = False


class DataSnapshot:
    """Snapshot of monitoring data for persistence."""
    
    def __init__(self):
        self.timestamp = time.time()
        self.training_state = {}
        self.metrics = {}
        self.events = []
        self.system_stats = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert snapshot to dictionary."""
        return {
            'timestamp': self.timestamp,
            'training_state': self.training_state,
            'metrics': self.metrics,
            'events': self.events,
            'system_stats': self.system_stats
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DataSnapshot':
        """Create snapshot from dictionary."""
        snapshot = cls()
        snapshot.timestamp = data.get('timestamp', time.time())
        snapshot.training_state = data.get('training_state', {})
        snapshot.metrics = data.get('metrics', {})
        snapshot.events = data.get('events', [])
        snapshot.system_stats = data.get('system_stats', {})
        return snapshot


class WebMonitor:
    """Main coordinator for web-based monitoring system."""
    
    def __init__(self, config: Optional[MonitorConfig] = None, auto_connect: bool = True):
        self.config = config or MonitorConfig()
        
        # Initialize database
        self.db = DatabaseManager(self.config.db_path)
        
        # Training state
        self.training_state = TrainingState.INITIALIZING
        self.episode = 0
        self.total_steps = 0
        self.current_run_id = None
        
        # Metrics collection
        self.metrics = ClientMetrics()
        self.last_update_time = time.time()
        
        # Initialize components
        self.server_config
            host=self.config.host,
            port=self.config.port,
            static_dir=self.config.static_dir,
            debug=self.config.debug
        )
        
        self.server = WebServer(self.server_config)
        self.interface = WebInterface(self.config.static_dir)
        self.bridge = TrainerMonitorBridge()
        self.data_bus = get_data_bus()
        self.error_handler = ErrorHandler()
        self.stats_collector = StatsCollector()
        self.logger = TextLogger()
        
        # Data management
        self.data_dir = Path(self.config.data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.current_snapshot = DataSnapshot()
        self._snapshot_lock = threading.Lock()
        
        # Client management
        self.connected_clients: Set[str] = set()
        self._client_lock = threading.Lock()
        
        # Event processing
        self.event_queue = Queue(maxsize=1000)
        self.event_handlers = {
            'training_update': self._handle_training_update,
            'game_state': self._handle_game_state,
            'metric_update': self._handle_metric_update,
            'system_stats': self._handle_system_stats,
            'client_event': self._handle_client_event
        }
        
        # Processing state
        self.is_running = False
        self._update_thread: Optional[threading.Thread] = None
        self._snapshot_thread: Optional[threading.Thread] = None
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Setup handlers
        self._setup_data_bus_handlers()
        self._setup_signal_handlers()
        
        self.logger.info("🖥️ Web monitor initialized")
    
    def _setup_data_bus_handlers(self) -> None:
        """Setup handlers for data bus events."""
        if self.data_bus:
            # Training events
            self.data_bus.subscribe(
                DataType.TRAINING_METRICS,
                lambda data: self.event_queue.put(('metric_update', data))
            )
            self.data_bus.subscribe(
                DataType.TRAINING_STATE,
                lambda data: self.event_queue.put(('training_update', data))
            )
            self.data_bus.subscribe(
                DataType.GAME_SCREEN,
                lambda data: self.event_queue.put(('game_state', data))
            )
            
            # System events
            self.data_bus.subscribe(
                DataType.SYSTEM_INFO,
                lambda data: self.event_queue.put(('system_stats', data))
            )
            self.data_bus.subscribe(
                DataType.ERROR_NOTIFICATION,
                self._handle_error_notification
            )
    
    def _setup_signal_handlers(self) -> None:
        """Setup handlers for system signals."""
        signal.signal(signal.SIGTERM, lambda s, f: asyncio.create_task(self.shutdown()))
        signal.signal(signal.SIGINT, lambda s, f: asyncio.create_task(self.shutdown()))
    
    async def start(self) -> None:
        """Start the monitoring system."""
        try:
            # Build interface components
            self.interface.build_components()
            
            # Start components
            self.bridge.start_processing()
            self.stats_collector.start_collecting()
            await self.server.start()
            
            # Start processing threads
            self.is_running = True
            self._start_processing_threads()
            
            # Load latest snapshot if available
            self._load_latest_snapshot()
            
            self.logger.info("🚀 Web monitor started")
            
        except Exception as e:
            self.error_handler.handle_error(
                e,
                severity=ErrorSeverity.CRITICAL,
                category=ErrorCategory.SYSTEM,
                component="web_monitor"
            )
            await self.shutdown()
    
    async def shutdown(self) -> None:
        """Shutdown the monitoring system."""
        self.logger.info("Shutting down web monitor...")
        self.is_running = False
        
        # Stop processing threads
        if self._update_thread and self._update_thread.is_alive():
            self._update_thread.join(timeout=5.0)
        if self._snapshot_thread and self._snapshot_thread.is_alive():
            self._snapshot_thread.join(timeout=5.0)
        
        # Save final snapshot
        self._save_snapshot()
        
        # Shutdown components
        self.bridge.stop_processing()
        self.stats_collector.stop_collecting()
        await self.server.shutdown()
        
        # Cleanup
        self.executor.shutdown(wait=True)
        
        self.logger.info("👋 Web monitor stopped")
    
    def _start_processing_threads(self) -> None:
        """Start processing threads."""
        # Start update thread
        self._update_thread = threading.Thread(
            target=self._update_loop,
            daemon=True
        )
        self._update_thread.start()
        
        # Start snapshot thread
        self._snapshot_thread = threading.Thread(
            target=self._snapshot_loop,
            daemon=True
        )
        self._snapshot_thread.start()
    
    def _update_loop(self) -> None:
        """Main update loop for processing events."""
        while self.is_running:
            try:
                # Process events
                while not self.event_queue.empty():
                    event_type, data = self.event_queue.get_nowait()
                    handler = self.event_handlers.get(event_type)
                    if handler:
                        self.executor.submit(handler, data)
                
                # Broadcast updates
                self._broadcast_updates()
                
                time.sleep(self.config.update_interval)
                
            except Exception as e:
                self.error_handler.handle_error(
                    e,
                    severity=ErrorSeverity.ERROR,
                    category=ErrorCategory.SYSTEM,
                    component="web_monitor"
                )
                time.sleep(1)
    
    def _snapshot_loop(self) -> None:
        """Loop for periodic data snapshots."""
        while self.is_running:
            try:
                time.sleep(self.config.snapshot_interval)
                self._save_snapshot()
            except Exception as e:
                self.error_handler.handle_error(
                    e,
                    severity=ErrorSeverity.ERROR,
                    category=ErrorCategory.SYSTEM,
                    component="web_monitor"
                )
    
    def _save_snapshot(self) -> None:
        """Save current data snapshot."""
        try:
            with self._snapshot_lock:
                # Prepare snapshot path
                timestamp = int(time.time())
                snapshot_file = self.data_dir / f"snapshot_{timestamp}.json"
                
                # Save snapshot
                with open(snapshot_file, "w") as f:
                    json.dump(self.current_snapshot.to_dict(), f, indent=2)
                
                # Cleanup old snapshots
                self._cleanup_old_snapshots()
                
        except Exception as e:
            self.error_handler.handle_error(
                e,
                severity=ErrorSeverity.ERROR,
                category=ErrorCategory.SYSTEM,
                component="web_monitor"
            )
    
    def _load_latest_snapshot(self) -> None:
        """Load the latest available snapshot."""
        try:
            snapshots = sorted(self.data_dir.glob("snapshot_*.json"))
            if not snapshots:
                return
            
            latest = snapshots[-1]
            with open(latest, "r") as f:
                data = json.load(f)
                self.current_snapshot = DataSnapshot.from_dict(data)
                
            self.logger.info(f"📥 Loaded snapshot from {latest}")
            
        except Exception as e:
            self.error_handler.handle_error(
                e,
                severity=ErrorSeverity.ERROR,
                category=ErrorCategory.SYSTEM,
                component="web_monitor"
            )
    
    def _cleanup_old_snapshots(self) -> None:
        """Remove old snapshots exceeding the limit."""
        try:
            snapshots = sorted(self.data_dir.glob("snapshot_*.json"))
            while len(snapshots) > self.config.max_snapshots:
                oldest = snapshots.pop(0)
                oldest.unlink()
        except Exception as e:
            self.logger.error(f"Failed to cleanup snapshots: {e}")
    
    def _broadcast_updates(self) -> None:
        """Broadcast updates to connected clients."""
        if not self.connected_clients:
            return
        
        try:
            with self._snapshot_lock:
                update_data = {
                    'timestamp': time.time(),
                    'training_state': self.current_snapshot.training_state,
                    'metrics': self.current_snapshot.metrics,
                    'system_stats': self.current_snapshot.system_stats
                }
            
            asyncio.run(self.server.broadcast('state_update', update_data))
            
        except Exception as e:
            self.error_handler.handle_error(
                e,
                severity=ErrorSeverity.ERROR,
                category=ErrorCategory.SYSTEM,
                component="web_monitor"
            )
    
    def _handle_training_update(self, data: Dict[str, Any]) -> None:
        """Handle training state updates."""
        with self._snapshot_lock:
            self.current_snapshot.training_state.update(data)
    
    def _handle_game_state(self, data: Dict[str, Any]) -> None:
        """Handle game state updates."""
        with self._snapshot_lock:
            self.current_snapshot.training_state['game_state'] = data
    
    def _handle_metric_update(self, data: Dict[str, Any]) -> None:
        """Handle metric updates."""
        with self._snapshot_lock:
            self.current_snapshot.metrics.update(data)
    
    def _handle_system_stats(self, data: Dict[str, Any]) -> None:
        """Handle system statistics updates."""
        with self._snapshot_lock:
            self.current_snapshot.system_stats.update(data)
    
    def _handle_client_event(self, data: Dict[str, Any]) -> None:
        """Handle events from web clients."""
        event_type = data.get('type')
        if not event_type:
            return
        
        try:
            if event_type == 'connect':
                client_id = data.get('client_id')
                if client_id:
                    with self._client_lock:
                        self.connected_clients.add(client_id)
                    
            elif event_type == 'disconnect':
                client_id = data.get('client_id')
                if client_id:
                    with self._client_lock:
                        self.connected_clients.discard(client_id)
                    
            elif event_type == 'command':
                # Forward commands to training system
                if self.data_bus:
                    self.data_bus.publish(
                        DataType.TRAINING_CONTROL,
                        data,
                        'web_monitor'
                    )
        
        except Exception as e:
            self.error_handler.handle_error(
                e,
                severity=ErrorSeverity.ERROR,
                category=ErrorCategory.SYSTEM,
                component="web_monitor"
            )
    
    def _handle_error_notification(self, data: Dict[str, Any]) -> None:
        """Handle error notifications."""
        try:
            error_data = {
                'timestamp': time.time(),
                'type': 'error',
                'message': data.get('message', 'Unknown error'),
                'component': data.get('component', 'unknown'),
                'severity': data.get('severity', 'error')
            }
            
            # Add to events
            with self._snapshot_lock:
                self.current_snapshot.events.append(error_data)
                if len(self.current_snapshot.events) > self.config.max_events:
                    self.current_snapshot.events.pop(0)
            
            # Broadcast error
            asyncio.run(self.server.broadcast('error', error_data))
            
        except Exception as e:
            self.logger.error(f"Failed to handle error notification: {e}")
    
    def update_episode(self, episode: int, total_reward: float = 0.0,
                      steps: int = 0, success: bool = False,
                      metadata: Dict[str, Any] = None) -> None:
        """Update episode information."""
        self.episode = episode
        self.total_steps += steps
        
        # Record episode in database
        if self.current_run_id is not None:
            episode_id = self.db.record_episode(
                self.current_run_id,
                episode,
                steps,
                total_reward,
                success,
                metadata
            )
        
        # Update metrics
        self.metrics.update_metric('episode_rewards', total_reward)
        self.metrics.update_metric('episode_lengths', steps)
        
        # Publish event
        if self.data_bus:
            self.data_bus.publish(
                DataType.TRAINING_METRICS,
                {
                    'episode': episode,
                    'total_reward': total_reward,
                    'steps': steps,
                    'success': success,
                    'total_steps': self.total_steps,
                    'timestamp': time.time()
                },
                'web_monitor'
            )
    
    def update_step(self, step: int, reward: float = 0.0, action: str = None,
                    inference_time: float = None, game_state: Dict[str, Any] = None,
                    **kwargs) -> None:
        """Update step information."""
        # Record game state if provided
        if game_state and self.current_run_id is not None:
            self.db.record_game_state(
                self.current_run_id,
                state_type='step',
                map_id=game_state.get('map_id', 0),
                player_x=game_state.get('player_x', 0),
                player_y=game_state.get('player_y', 0),
                metadata=game_state
            )
        # Update metrics
        self.metrics.update_metric('step_rewards', reward)
        if inference_time is not None:
            self.metrics.update_metric('inference_times', inference_time)
        
        # Record action
        if action:
            self.metrics.update_action(action)
        
        # Calculate processing time
        current_time = time.time()
        processing_time = current_time - self.last_update_time
        self.metrics.update_metric('processing_times', processing_time)
        self.last_update_time = current_time
        
        # Publish event if enough time has passed
        if current_time - self.last_update_time >= 1.0:  # Update every second
            if self.data_bus:
                self.data_bus.publish(
                    DataType.TRAINING_METRICS,
                    {
                        'step': step,
                        'reward': reward,
                        'action': action,
                        'metrics': self.metrics.get_stats(),
                        'performance': self.metrics.get_performance_stats(),
                        **kwargs
                    },
                    'web_monitor'
                )
    
    def update_text(self, text: str, text_type: str = "dialogue") -> None:
        """Update detected text information."""
        self.metrics.update_text(text)
        
        if self.data_bus:
            self.data_bus.publish(
                DataType.TRAINING_METRICS,
                {
                    'text': text,
                    'type': text_type,
                    'frequency': self.metrics.text_frequency
                },
                'web_monitor'
            )
    
    @contextmanager
    def training_context(self, run_id: Optional[int] = None):
        """Context manager for training sessions."""
        try:
            self.start_training(run_id)
            yield self
        finally:
            self.stop_training()
    
    def start_training(self, run_id: Optional[int] = None, config: Dict[str, Any] = None) -> None:
        """Start or resume training."""
        if run_id is None:
            # Start new training run
            run_id = self.db.start_training_run(config)
        
        self.training_state = TrainingState.RUNNING
        self.current_run_id = run_id
        self.metrics = ClientMetrics()  # Reset metrics
        
        if self.data_bus:
            self.data_bus.publish(
                DataType.TRAINING_STATE,
                {
                    'state': 'running',
                    'run_id': run_id,
                    'timestamp': time.time()
                },
                'web_monitor'
            )
    
    def pause_training(self) -> None:
        """Pause training."""
        self.training_state = TrainingState.PAUSED
        
        if self.data_bus:
            self.data_bus.publish(
                DataType.TRAINING_STATE,
                {
                    'state': 'paused',
                    'run_id': self.current_run_id,
                    'timestamp': time.time()
                },
                'web_monitor'
            )
    
    def stop_training(self, final_reward: Optional[float] = None) -> None:
        """Stop training."""
        self.training_state = TrainingState.STOPPED
        
        # Record final state in database
        if self.current_run_id is not None:
            self.db.end_training_run(
                self.current_run_id,
                status='completed',
                final_reward=final_reward
            )
        
        if self.data_bus:
            self.data_bus.publish(
                DataType.TRAINING_STATE,
                {
                    'state': 'stopped',
                    'run_id': self.current_run_id,
                    'timestamp': time.time(),
                    'final_metrics': self.metrics.get_stats()
                },
                'web_monitor'
            )
    
    def start(self) -> None:
        """Run the monitor in the current thread."""
        loop = asyncio.get_event_loop()
        try:
            loop.run_until_complete(self.start())
            loop.run_forever()
        except KeyboardInterrupt:
            loop.run_until_complete(self.shutdown())
        finally:
            loop.close()
    
    def run_in_thread(self) -> threading.Thread:
        """Run the monitor in a separate thread."""
        thread = threading.Thread(target=self.run, daemon=True)
        thread.start()
        return thread
