"""
Integration tests for the unified monitoring system.

Tests the integration between:
- Database operations
- Video streaming
- Client API
- Performance metrics
- Error handling
"""

import pytest
import json
import time
import threading
import tempfile
from pathlib import Path
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import socket
import psutil
import requests
from queue import Queue

from pokemon_crystal_rl.monitoring import (
    WebMonitor,
    MonitorConfig,
    DatabaseManager,
    ErrorHandler,
    ErrorSeverity,
    RecoveryStrategy,
    TrainingState
)

# Fixtures

@pytest.fixture
def temp_dir():
    """Create temporary directory for test data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)

@pytest.fixture
def free_port():
    """Get an available port for testing."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port

@pytest.fixture
def test_config(temp_dir, free_port):
    """Create test configuration."""
    return MonitorConfig(
        db_path=str(temp_dir / "test.db"),
        data_dir=str(temp_dir / "data"),
        static_dir=str(temp_dir / "static"),
        web_port=free_port,
        update_interval=0.1,  # Fast updates for testing
        snapshot_interval=0.5,
        max_events=10,
        max_snapshots=5,
        debug=True
    )

@pytest.fixture
def mock_monitor(test_config):
    """Create test monitor instance."""
    monitor = WebMonitor(test_config)
    yield monitor
    monitor.stop_training()


class TestUnifiedMonitoring:
    """Integration tests for the unified monitoring system."""
    
    def test_database_integration(self, mock_monitor, temp_dir):
        """Test database integration with monitoring."""
        # Start training run
        mock_monitor.start_training(config={"test": True})
        run_id = mock_monitor.current_run_id
        
        # Record metrics
        metrics = {
            "loss": 0.5,
            "accuracy": 0.8,
            "reward": 1.0
        }
        mock_monitor.update_metrics(metrics)
        
        # Record system metrics
        mock_monitor.update_system_metrics(
            cpu_percent=50.0,
            memory_percent=60.0,
            disk_usage=70.0
        )
        
        # Record events
        mock_monitor.record_event(
            event_type="achievement",
            event_data={"type": "badge", "value": 1}
        )
        
        # Stop training
        mock_monitor.stop_training(final_reward=100.0)
        
        # Verify database records
        db = DatabaseManager(mock_monitor.config.db_path)
        
        # Check run recorded
        run = db.get_run_summary(run_id)
        assert run["status"] == "completed"
        assert run["final_reward"] == 100.0
        
        # Check metrics recorded
        metrics = db.get_run_metrics(run_id)
        assert not metrics.empty
        assert "loss" in metrics.columns
        assert "accuracy" in metrics.columns
        
        # Check events recorded
        events = db.get_run_events(run_id)
        assert len(events) > 0
        assert events[0]["event_type"] == "achievement"
    
    def test_video_streaming(self, mock_monitor):
        """Test video streaming functionality."""
        # Create mock game screen
        screen = np.random.randint(0, 255, (144, 160, 3), dtype=np.uint8)
        frame_queue = Queue(maxsize=30)
        
        # Start training
        mock_monitor.start_training()
        
        # Mock screen capture
        def capture_screen():
            return screen
        
        with patch.object(mock_monitor, '_capture_screen', side_effect=capture_screen):
            # Queue some frames
            for i in range(5):
                mock_monitor._update_screen()
                time.sleep(0.05)
            
            # Test screen endpoint
            response = requests.get(
                f"http://localhost:{mock_monitor.config.web_port}/api/screen"
            )
            assert response.status_code == 200
            data = response.json()
            assert "screen" in data
            assert "timestamp" in data
            assert "frame" in data
            
            # Test screen stream
            received_frames = []
            stream_event = threading.Event()
            
            def stream_callback(frame):
                received_frames.append(frame)
                if len(received_frames) >= 3:
                    stream_event.set()
            
            mock_monitor.subscribe_to_stream(stream_callback)
            
            # Generate frames
            for i in range(5):
                mock_monitor._update_screen()
                time.sleep(0.05)
            
            # Verify stream reception
            assert stream_event.wait(timeout=1.0)
            assert len(received_frames) >= 3
            
            # Verify frame format
            frame = received_frames[0]
            assert "screen" in frame
            assert "timestamp" in frame
            assert "frame" in frame
    
    def test_client_api_integration(self, mock_monitor):
        """Test client API functionality."""
        # Start training
        mock_monitor.start_training(config={"test": True})
        
        # Test metric updates
        mock_monitor.update_metrics({
            "loss": 0.5,
            "accuracy": 0.8
        })
        
        # Test step updates
        mock_monitor.update_step(
            step=1,
            reward=1.0,
            action="RIGHT",
            inference_time=0.01,
            game_state={
                "map_id": 1,
                "player_x": 10,
                "player_y": 20
            }
        )
        
        # Test episode updates
        mock_monitor.update_episode(
            episode=0,
            total_reward=10.0,
            steps=100,
            success=True,
            metadata={"completion": 0.5}
        )
        
        # Verify API responses
        response = requests.get(
            f"http://localhost:{mock_monitor.config.web_port}/api/stats"
        )
        assert response.status_code == 200
        stats = response.json()
        assert "metrics" in stats
        assert stats["metrics"]["loss"] == 0.5
        
        response = requests.get(
            f"http://localhost:{mock_monitor.config.web_port}/api/state"
        )
        assert response.status_code == 200
        state = response.json()
        assert state["map_id"] == 1
        assert state["player_x"] == 10
    
    def test_performance_metrics(self, mock_monitor):
        """Test performance metric collection and analysis."""
        # Start training
        mock_monitor.start_training()
        
        # Record various metrics
        for i in range(10):
            mock_monitor.update_metrics({
                "loss": 1.0 - (i * 0.1),
                "accuracy": i * 0.1,
                "reward": i * 1.0
            })
            time.sleep(0.05)
        
        # Get metric history
        response = requests.get(
            f"http://localhost:{mock_monitor.config.web_port}/api/metrics/history",
            params={"metric": "loss", "minutes": 5}
        )
        assert response.status_code == 200
        history = response.json()
        
        # Verify metric data
        assert len(history["data"]) >= 5
        assert "mean" in history["statistics"]
        assert "min" in history["statistics"]
        assert "max" in history["statistics"]
        
        # Test metric aggregation
        response = requests.get(
            f"http://localhost:{mock_monitor.config.web_port}/api/metrics/summary"
        )
        assert response.status_code == 200
        summary = response.json()
        assert "loss" in summary
        assert "accuracy" in summary
        assert "reward" in summary
    
    def test_error_handling(self, mock_monitor):
        """Test error handling and recovery."""
        # Start training
        mock_monitor.start_training()
        
        # Record test errors
        errors = [
            (RecoveryStrategy.RETRY, ErrorSeverity.WARNING, "Test warning"),
            (RecoveryStrategy.RESET, ErrorSeverity.ERROR, "Test error"),
            (RecoveryStrategy.ABORT, ErrorSeverity.CRITICAL, "Test critical error")
        ]
        
        for category, severity, message in errors:
            error = Exception(message)
            mock_monitor.error_handler.handle_error(
                error=error,
                message=message,
                category=category,
                severity=severity
            )
        
        # Check error events recorded
        response = requests.get(
            f"http://localhost:{mock_monitor.config.web_port}/api/events",
            params={"type": "error"}
        )
        assert response.status_code == 200
        events = response.json()["events"]
        assert len(events) == len(errors)
        
        # Test error recovery
        with patch.object(mock_monitor, '_update_screen') as mock_update:
            # Simulate error
            mock_update.side_effect = Exception("Screen error")
            mock_monitor._update_screen()
            
            # Should still be running
            assert mock_monitor.training_state == TrainingState.RUNNING
            
            # Fix error and continue
            mock_update.side_effect = None
            mock_monitor._update_screen()
            
            # Should have recorded recovery
            response = requests.get(
                f"http://localhost:{mock_monitor.config.web_port}/api/events",
                params={"type": "recovery"}
            )
            assert response.status_code == 200
            recoveries = response.json()["events"]
            assert len(recoveries) > 0
    
    def test_data_persistence(self, mock_monitor, temp_dir):
        """Test data persistence and recovery."""
        # Generate test data
        mock_monitor.start_training(config={"test": True})
        run_id = mock_monitor.current_run_id
        
        for episode in range(2):
            mock_monitor.update_episode(
                episode=episode,
                total_reward=episode * 1.0,
                steps=10,
                success=True
            )
        
        # Create snapshot
        mock_monitor._save_snapshot()
        
        # Create new monitor instance
        new_monitor = WebMonitor(mock_monitor.config)
        
        # Verify data is recovered
        run_summary = new_monitor.db.get_run_summary(run_id)
        assert run_summary["total_episodes"] == 2
        
        # Check snapshots exist
        snapshots = list(Path(mock_monitor.config.data_dir).glob("snapshot_*.json"))
        assert len(snapshots) > 0
    
    def test_system_integration(self, mock_monitor):
        """Test complete system integration."""
        # Start training
        mock_monitor.start_training(config={
            "learning_rate": 0.001,
            "batch_size": 64
        })
        
        # Simulate training loop
        for episode in range(3):
            # Episode steps
            for step in range(10):
                mock_monitor.update_step(
                    step=step,
                    reward=0.5,
                    action="RIGHT",
                    inference_time=0.01,
                    game_state={
                        "map_id": 1,
                        "player_x": step,
                        "player_y": 5
                    }
                )
                
                # Update metrics
                mock_monitor.update_metrics({
                    "loss": 1.0 - (step * 0.1),
                    "accuracy": step * 0.1,
                    "reward": step * 0.5
                })
                
                # Record events
                if step % 5 == 0:
                    mock_monitor.record_event(
                        event_type="progress",
                        event_data={"step": step, "milestone": f"Step {step}"}
                    )
                
                time.sleep(0.05)
            
            # End episode
            mock_monitor.update_episode(
                episode=episode,
                total_reward=5.0,
                steps=10,
                success=True,
                metadata={"completion": episode * 0.3}
            )
        
        # Stop training
        mock_monitor.stop_training(final_reward=15.0)
        
        # Verify complete system state
        response = requests.get(
            f"http://localhost:{mock_monitor.config.web_port}/api/status"
        )
        assert response.status_code == 200
        status = response.json()
        
        assert status["status"] == "completed"
        assert status["total_episodes"] == 3
        assert status["total_steps"] == 30
        
        # Check all data recorded
        db = DatabaseManager(mock_monitor.config.db_path)
        
        metrics = db.get_run_metrics(mock_monitor.current_run_id)
        assert not metrics.empty
        assert len(metrics) > 20  # Should have multiple metrics per step
        
        events = db.get_run_events(mock_monitor.current_run_id)
        assert len(events) >= 6  # Should have progress events
        
        # Export run data
        export_dir = Path(mock_monitor.config.data_dir) / "exports"
        export_dir.mkdir(exist_ok=True)
        
        export_path = mock_monitor.export_run_data(
            run_id=mock_monitor.current_run_id,
            output_dir=export_dir,
            include_snapshots=True
        )
        
        assert export_path.exists()
        assert export_path.suffix == ".zip"

if __name__ == "__main__":
    pytest.main(["-v", __file__])
