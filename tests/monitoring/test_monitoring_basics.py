"""Basic functionality tests for unified monitoring system."""

import pytest
import tempfile
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch
from monitoring import (
    WebMonitor,
    MonitorConfig,
    DatabaseManager,
    ErrorHandler,
    ErrorSeverity,
    RecoveryStrategy,
    TrainingState,
    UnifiedMonitor
)

@pytest.fixture
def temp_dir():
    """Create temporary directory for test data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)

@pytest.fixture
def test_config(temp_dir):
    """Create test configuration."""
    return MonitorConfig(
        db_path=str(temp_dir / "test.db"),
        data_dir=str(temp_dir / "data"),
        static_dir=str(temp_dir / "static"),
        port=8099,  # Use non-standard port
        update_interval=0.1,
        snapshot_interval=0.5,
        max_events=1000,
        max_snapshots=10,
        debug=True
    )

@pytest.fixture
def monitor(test_config):
    """Create test monitor instance."""
    monitor = UnifiedMonitor(config=test_config)
    try:
        yield monitor
    finally:
        monitor.stop_training()

@pytest.mark.unit
class TestMonitorBasics:
    """Test basic monitor functionality."""
    
    def test_initialization(self, monitor, test_config):
        """Test monitor initialization."""
        assert monitor.config == test_config
        assert monitor.training_state == TrainingState.INITIALIZING
        assert monitor.current_run_id is None
        assert monitor.is_monitoring is False
        assert monitor.error_handler is not None
        assert monitor.db is not None
    
    def test_start_stop_training(self, monitor):
        """Test training start/stop."""
        # Start training
        run_id = monitor.start_training(config={"test": True})
        assert run_id is not None
        assert monitor.current_run_id == run_id
        assert monitor.training_state == TrainingState.RUNNING
        assert monitor.is_monitoring is True
        
        # Stop training
        monitor.stop_training(final_reward=100.0)
        assert monitor.training_state == TrainingState.COMPLETED
        assert monitor.is_monitoring is False
        
    def test_pause_resume_training(self, monitor):
        """Test pause/resume functionality."""
        # Start training
        monitor.start_training()
        assert monitor.training_state == TrainingState.RUNNING
        
        # Pause training
        monitor.pause_training()
        assert monitor.training_state == TrainingState.PAUSED
        
        # Resume training
        monitor.resume_training()
        assert monitor.training_state == TrainingState.RUNNING
    
    def test_metrics_update(self, monitor):
        """Test metric updates."""
        monitor.start_training()
        
        # Update metrics
        metrics = {
            "loss": 0.5,
            "accuracy": 0.8,
            "reward": 1.0
        }
        monitor.update_metrics(metrics)
        
        # Check metrics were stored in memory
        for name, value in metrics.items():
            assert name in monitor.performance_metrics
            assert value in monitor.performance_metrics[name]
        
        # Check metrics recorded in DB if available
        if monitor.db:
            run_metrics = monitor.db.get_run_metrics(monitor.current_run_id)
            assert not run_metrics.empty
            assert "loss" in run_metrics.columns
            assert run_metrics["loss"].iloc[-1] == 0.5
    
    def test_episode_update(self, monitor):
        """Test episode updates."""
        monitor.start_training()
        
        # Update episode
        episode_data = {
            'episode': 0,
            'total_reward': 10.0,
            'steps': 100,
            'success': True,
            'metadata': {"completion": 0.5}
        }
        monitor.update_episode(
            episode=episode_data['episode'],
            total_reward=episode_data['total_reward'],
            steps=episode_data['steps'],
            success=episode_data['success'],
            metadata=episode_data['metadata']
        )
        
        # Check episode was stored in memory
        assert len(monitor.episode_data) == 1
        stored_episode = monitor.episode_data[0]
        assert stored_episode['episode'] == episode_data['episode']
        assert stored_episode['total_reward'] == episode_data['total_reward']
        assert stored_episode['steps'] == episode_data['steps']
        assert stored_episode['success'] == episode_data['success']
        assert stored_episode['metadata'] == episode_data['metadata']
        
        # Check episode recorded in DB if available
        if monitor.db:
            episodes = monitor.db.get_run_episodes(monitor.current_run_id)
            assert len(episodes) == 1
            episode = episodes[0]
            assert episode["total_reward"] == 10.0
            assert episode["total_steps"] == 100
            assert episode["success"] == True  # SQLite stores boolean as int, so use == not is
            assert episode["metadata"]["completion"] == 0.5

    def test_game_state_tracking(self, monitor):
        """Test game state tracking."""
        monitor.start_training()
        
        # Update state
        game_state = {
            "map_id": 1,
            "player_x": 10,
            "player_y": 20,
            "pokemon": [
                {"species": "PIKACHU", "level": 5},
                {"species": "CHARMANDER", "level": 8}
            ]
        }
        
        monitor.update_step(
            step=0,
            reward=1.0,
            action="RIGHT",
            inference_time=0.01,
            game_state=game_state
        )
        
        # Check state recorded
        db = monitor.db
        states = db.get_run_states(monitor.current_run_id)
        assert len(states) == 1
        state = states[0]
        assert state["game_state"]["map_id"] == 1
        assert state["game_state"]["player_x"] == 10
        assert len(state["game_state"]["pokemon"]) == 2

    @pytest.mark.parametrize("severity,strategy", [
        (ErrorSeverity.MEDIUM, RecoveryStrategy.RETRY),
        (ErrorSeverity.HIGH, RecoveryStrategy.RESTART_COMPONENT),
        (ErrorSeverity.CRITICAL, RecoveryStrategy.GRACEFUL_SHUTDOWN)
    ])
    def test_error_handling(self, monitor, severity, strategy):
        """Test error handling at different severity levels."""
        monitor.start_training()
        
        # Generate test error
        error = Exception("Test error")
        monitor.error_handler.handle_error(
            error=error,
            message="Test error message",
            severity=severity,
            recovery_strategy=strategy,
            category="test",
            component="monitor"
        )
        
        # Check error recorded
        db = monitor.db
        events = db.get_run_events(
            monitor.current_run_id,
            event_type="error"
        )
        assert len(events) == 1
        event = events[0]
        assert event["event_data"]["severity"] == severity.value
        assert event["event_data"]["recovery_strategy"] == strategy.value
        
        # Check training state based on severity
        if severity == ErrorSeverity.CRITICAL:
            assert monitor.training_state != TrainingState.RUNNING
        else:
            assert monitor.training_state == TrainingState.RUNNING

    def test_screenshot_handling(self, monitor):
        """Test screenshot handling."""
        monitor.start_training()
        
        # Create test screenshot
        screenshot = np.random.randint(0, 255, (144, 160, 3), dtype=np.uint8)
        
        # Update screenshot
        monitor.update_screenshot(screenshot)
        
        # Check screenshot queue
        assert not monitor.screen_queue.empty()
        screenshot_data = monitor.screen_queue.get()
        assert "image" in screenshot_data
        assert screenshot_data["image"].startswith("data:image/png;base64,")
        assert "timestamp" in screenshot_data
        assert "dimensions" in screenshot_data

    def test_training_lifecycle(self, monitor):
        """Test complete training lifecycle."""
        # Test that monitor starts in correct state
        assert monitor.training_state == TrainingState.INITIALIZING
        assert not monitor.is_monitoring
        assert monitor.current_run_id is None
        
        # Start training and verify state
        config = {"learning_rate": 0.001, "batch_size": 32}
        run_id = monitor.start_training(config)
        assert monitor.training_state == TrainingState.RUNNING
        assert monitor.is_monitoring
        assert monitor.current_run_id == run_id
        
        # Verify cannot start another run while running
        with pytest.raises(RuntimeError):
            monitor.start_training()
        
        # Update training data
        monitor.update_metrics({"loss": 0.5, "accuracy": 0.9})
        monitor.update_episode(episode=1, total_reward=100.0, steps=500, success=True)
        
        # Pause training and verify state
        monitor.pause_training()
        assert monitor.training_state == TrainingState.PAUSED
        
        # Verify monitoring continues during pause
        assert monitor.is_monitoring
        
        # Resume and verify state
        monitor.resume_training()
        assert monitor.training_state == TrainingState.RUNNING
        
        # Update more training data
        monitor.update_metrics({"loss": 0.3, "accuracy": 0.95})
        monitor.update_episode(episode=2, total_reward=150.0, steps=450, success=True)
        
        # Stop training and verify final state
        final_reward = 250.0
        monitor.stop_training(final_reward=final_reward)
        assert monitor.training_state == TrainingState.COMPLETED
        assert not monitor.is_monitoring
