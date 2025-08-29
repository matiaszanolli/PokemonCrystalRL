"""
System integration tests for the Pokemon Crystal RL monitoring system.

Tests the complete system functionality including:
- Training lifecycle
- Real-time monitoring
- Data collection
- Error handling
- System recovery
"""

import pytest
import time
import threading
import traceback
import numpy as np
import tempfile
import json
import asyncio
import psutil
import requests
import logging
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from queue import Queue
from concurrent.futures import ThreadPoolExecutor

from monitoring import (
    UnifiedMonitor,
    MonitorConfig,
    DatabaseManager,
    ErrorHandler,
    ErrorSeverity,
    RecoveryStrategy,
    ErrorCategory,
    ErrorEvent
)

class MockGameState:
    """Mock game state for testing."""
    def __init__(self):
        self.map_id = 1
        self.player_x = 0
        self.player_y = 0
        self.screen = np.random.randint(0, 255, (144, 160, 3), dtype=np.uint8)
        self.frame = 0
        self.inventory = ["POTION", "POKEBALL"]
        self.pokemon = [
            {"species": "PIKACHU", "level": 5, "hp": 20},
            {"species": "CHARMANDER", "level": 8, "hp": 30}
        ]
    
    def update(self):
        """Update game state."""
        self.frame += 1
        self.player_x = (self.player_x + 1) % 20
        self.player_y = (self.player_y + 1) % 20
        self.screen = np.random.randint(0, 255, (144, 160, 3), dtype=np.uint8)
        
        if self.frame % 100 == 0:
            self.map_id += 1
    
    def get_state(self):
        """Get current game state."""
        return {
            "map_id": self.map_id,
            "player_x": self.player_x,
            "player_y": self.player_y,
            "frame": self.frame,
            "inventory": self.inventory,
            "pokemon": self.pokemon
        }

class MockTrainer:
    """Mock trainer for testing."""
    def __init__(self, monitor):
        self.monitor = monitor
        self.game_state = MockGameState()
        self.episode = 0
        self.step = 0
        self.total_reward = 0.0
        self.running = False
        self.thread = None
        self.actions = ["UP", "DOWN", "LEFT", "RIGHT"]
        self.metrics = {
            "loss": 1.0,
            "accuracy": 0.0,
            "reward": 0.0
        }
    
    def start(self):
        """Start training loop."""
        self.running = True
        self.thread = threading.Thread(target=self._training_loop)
        self.thread.start()
    
    def stop(self):
        """Stop training loop."""
        self.running = False
        if self.thread:
            self.thread.join()
    
    def _training_loop(self):
        """Main training loop."""
        try:
            while self.running:
                # Update game state
                self.game_state.update()
                
                # Select random action
                action = np.random.choice(self.actions)
                
                # Update metrics
                self.metrics["loss"] *= 0.95  # Decrease loss
                self.metrics["accuracy"] = min(0.99, self.metrics["accuracy"] + 0.01)
                self.metrics["reward"] = np.random.normal(0.5, 0.1)
                
                # Update monitor metrics
                self.monitor.update_metrics({
                    "cpu_percent": psutil.cpu_percent(),
                    "memory_percent": psutil.virtual_memory().percent,
                    "disk_usage": psutil.disk_usage("/").percent,
                    **self.metrics
                })
                
                # Occasionally generate events
                if self.step % 50 == 0:
                    self.monitor.record_event(
                        event_type="progress",
                        event_data={
                            "episode": self.episode,
                            "step": self.step,
                            "reward": self.total_reward
                        }
                    )
                
                # Occasionally simulate errors
                if self.step % 200 == 0:
                    try:
                        raise Exception("Test error")
                    except Exception as e:
                        error_event = ErrorEvent(
                            timestamp=time.time(),
                            component="MockTrainer",
                            error_type=type(e).__name__,
                            message="Simulated error",
                            severity=ErrorSeverity.WARNING,
                            traceback=traceback.format_exc(),
                            recovery_strategy=RecoveryStrategy.RETRY
                        )
                        self.monitor.error_handler.handle_error(error_event)
                
                self.step += 1
                self.total_reward += self.metrics["reward"]
                
                # End episode
                if self.step % 100 == 0:
                    self.monitor.update_episode(
                        episode=self.episode,
                        total_reward=self.total_reward,
                        steps=100,
                        success=True,
                        metadata={
                            "completion": min(1.0, self.episode * 0.1),
                            "accuracy": self.metrics["accuracy"]
                        }
                    )
                    self.episode += 1
                    self.step = 0
                    self.total_reward = 0.0
                
                time.sleep(0.01)  # Simulate processing time
                
        except Exception as e:
            logging.error(f"Training loop error: {e}")
            error_event = ErrorEvent(
                timestamp=time.time(),
                component="MockTrainer",
                error_type=type(e).__name__,
                message="Training loop failed",
                severity=ErrorSeverity.CRITICAL,
                traceback=traceback.format_exc(),
                recovery_strategy=RecoveryStrategy.GRACEFUL_SHUTDOWN
            )
            self.monitor.error_handler.handle_error(error_event)

class TestSystemIntegration:
    """System integration tests."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    @pytest.fixture
    def test_config(self, temp_dir):
        """Create test configuration."""
        return MonitorConfig(
            db_path=str(temp_dir / "test.db"),
            data_dir=str(temp_dir / "data"),
            static_dir=str(temp_dir / "static"),
            web_port=8099,  # Use fixed port for tests
            update_interval=0.1,
            snapshot_interval=0.5,
            max_events=1000,
            max_snapshots=10,
            debug=True
        )
    
    @pytest.fixture
    def monitor(self, test_config):
        """Create monitor instance."""
        monitor = UnifiedMonitor(test_config)
        yield monitor
        monitor.stop_training()
    
    @pytest.fixture
    def trainer(self, monitor):
        """Create mock trainer."""
        trainer = MockTrainer(monitor)
        yield trainer
        trainer.stop()
    
    def test_complete_training_session(self, monitor, trainer, temp_dir):
        """Test complete training session with all components."""
        # Start training
        start_time = datetime.now()
        
        # Create a mock training session
        mock_training_session = Mock()
        mock_training_session.get_stats = Mock(return_value={
            "total_actions": 0,
            "actions_per_second": 0.0,
            "total_duration": 0,
        })
        monitor.training_session = mock_training_session
        
        # Add record_event method to monitor
        monitor.record_event = Mock()
        
        monitor.start_training(config={
            "test": True,
            "learning_rate": 0.001,
            "batch_size": 64
        })
        
        # Start mock trainer
        trainer.start()
        
        # Monitor training progress
        metrics_received = []
        events_received = []
        screens_received = []
        
        def collect_metrics():
            while trainer.running:
                try:
                    response = requests.get(
                        f"http://localhost:{monitor.config.web_port}/api/stats",
                        timeout=1.0
                    )
                    if response.status_code == 200:
                        metrics_data = response.json()
                        if 'stats' in metrics_data:
                            metrics_received.append(metrics_data['stats'])
                    time.sleep(0.1)
                except (requests.exceptions.RequestException, KeyError) as e:
                    print(f"\nDEBUG: Error collecting metrics: {e}")
        
        def collect_events():
            while trainer.running:
                response = requests.get(
                    f"http://localhost:{monitor.config.web_port}/api/events"
                )
                if response.status_code == 200:
                    events_received.extend(response.json()["events"])
                time.sleep(0.1)
        
        def collect_screens():
            while trainer.running:
                response = requests.get(
                    f"http://localhost:{monitor.config.web_port}/api/screen"
                )
                if response.status_code == 200:
                    screens_received.append(response.json())
                time.sleep(0.1)
        
        # Start collectors
        with ThreadPoolExecutor(max_workers=3) as executor:
            collectors = [
                executor.submit(collect_metrics),
                executor.submit(collect_events),
                executor.submit(collect_screens)
            ]
            
            # Let training run for a while
            time.sleep(5.0)  # Run for 5 seconds
            
            # Stop training
            trainer.stop()
            monitor.stop_training()
        
        # Verify training duration
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        assert duration >= 5.0
        
        # Verify data collection
        assert len(metrics_received) > 0
        assert len(events_received) > 0
        assert len(screens_received) > 0
        
        # Verify database state
        db = DatabaseManager(monitor.config.db_path)
        
        # Check training run
        run = db.get_run_summary(monitor.current_run_id)
        assert run["status"] == "completed"
        assert run["total_episodes"] > 0
        assert run["total_steps"] > 0
        
        # Check metrics
        metrics = db.get_run_metrics(monitor.current_run_id)
        assert not metrics.empty
        assert "loss" in metrics.columns
        assert "accuracy" in metrics.columns
        assert "reward" in metrics.columns
        
        # Verify metric trends
        loss_values = metrics["loss"].values
        assert loss_values[0] > loss_values[-1]  # Loss should decrease
        
        accuracy_values = metrics["accuracy"].values
        assert accuracy_values[0] < accuracy_values[-1]  # Accuracy should increase
        
        # Check events
        events = db.get_run_events(monitor.current_run_id)
        assert len(events) > 0
        
        # Verify event types
        event_types = set(e["event_type"] for e in events)
        assert "progress" in event_types
        assert "error" in event_types
        
        # Check snapshots
        snapshots = list(Path(monitor.config.data_dir).glob("snapshot_*.json"))
        assert len(snapshots) > 0
        
        # Verify snapshot content
        with open(snapshots[0]) as f:
            snapshot = json.load(f)
            assert "metrics" in snapshot
            assert "state" in snapshot
            assert "timestamp" in snapshot
        
        # Test data export
        export_dir = Path(monitor.config.data_dir) / "exports"
        export_dir.mkdir(exist_ok=True)
        
        export_path = monitor.export_run_data(
            run_id=monitor.current_run_id,
            output_dir=export_dir,
            include_snapshots=True
        )
        
        assert export_path.exists()
        assert export_path.suffix == ".zip"
        
        # Verify system metrics
        system_metrics = db.get_run_system_metrics(monitor.current_run_id)
        assert not system_metrics.empty
        assert "cpu_percent" in system_metrics.columns
        assert "memory_percent" in system_metrics.columns
        assert "disk_usage" in system_metrics.columns
        
        # Check error handling
        error_events = [e for e in events if e["event_type"] == "error"]
        assert len(error_events) > 0
        
        # Verify training recovered from errors
        assert run["status"] == "completed"  # Training completed despite errors
        
        # Test data cleanup
        db.cleanup_old_data(older_than=start_time - timedelta(hours=1))
        
        # Optimize database
        db.optimize_database()
        
        # Final status check
        response = requests.get(
            f"http://localhost:{monitor.config.web_port}/api/status"
        )
        assert response.status_code == 200
        status = response.json()
        assert status["status"] == "completed"

if __name__ == "__main__":
    pytest.main(["-v", __file__])
