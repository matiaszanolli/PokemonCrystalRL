"""Tests for web monitoring APIs.

This module tests the REST API endpoints in the web monitoring system:
- Training API: statistics and LLM decisions
- System API: status and resource monitoring
- Game API: game state and screen capture data
"""
import pytest
pytestmark = [pytest.mark.web, pytest.mark.web_monitoring]

import time
import pytest
from unittest.mock import Mock, MagicMock
from datetime import datetime, timezone
from monitoring.web.api import TrainingAPI, SystemAPI, GameAPI
from monitoring.web.api.training import TrainingMetrics


class TestTrainingAPI:
    """Tests for training statistics and control API."""
    
    def test_training_metrics_from_dict(self):
        """Test metrics creation from dictionary."""
        data = {
            'actions_taken': 100,
            'actions_per_second': 1.5,
            'llm_decision_count': 50,
            'total_reward': 75.5
        }
        metrics = TrainingMetrics.from_dict(data)
        
        assert metrics.total_actions == 100
        assert metrics.actions_per_second == 1.5
        assert metrics.llm_calls == 50
        assert metrics.total_reward == 75.5
    
    def test_get_training_stats_with_stats_attr(self):
        """Test getting training stats from trainer.stats."""
        mock_trainer = Mock()
        mock_trainer.stats = {
            'actions_taken': 200,
            'actions_per_second': 2.0,
            'llm_decision_count': 100,
            'total_reward': 150.0
        }
        
        api = TrainingAPI(trainer=mock_trainer)
        stats = api.get_training_stats()
        
        assert stats['total_actions'] == 200
        assert stats['actions_per_second'] == 2.0
        assert stats['llm_calls'] == 100
        assert stats['total_reward'] == 150.0
    
    def test_get_training_stats_with_get_current_stats(self):
        """Test getting training stats via get_current_stats method."""
        stats_data = {
            'actions_taken': 300,
            'actions_per_second': 2.5,
            'llm_decision_count': 150,
            'total_reward': 225.0
        }
        mock_trainer = MagicMock()
        mock_trainer.get_current_stats = Mock(return_value=stats_data)
        mock_trainer.stats = stats_data  # Backup stats path
        
        api = TrainingAPI(trainer=mock_trainer)
        stats = api.get_training_stats()
        
        assert stats['total_actions'] == 300
        assert stats['actions_per_second'] == 2.5
        assert stats['llm_calls'] == 150
        assert stats['total_reward'] == 225.0
    
    def test_get_training_stats_no_trainer(self):
        """Test getting training stats with no trainer."""
        api = TrainingAPI()
        stats = api.get_training_stats()
        
        assert stats['total_actions'] == 0
        assert stats['actions_per_second'] == 0.0
        assert stats['llm_calls'] == 0
        assert stats['total_reward'] == 0.0
    
    def test_get_llm_decisions_processing(self):
        """Test LLM decision processing and enhancement."""
        current_time = time.time()
        decisions = [
            {
                'timestamp': current_time - 10,
                'action': 4,
                'response_time': 0.5,
                'reasoning': 'Test reasoning'
            },
            {
                'timestamp': current_time - 20,
                'action': 2,
                'response_time_ms': 600,
                'reasoning': 'A' * 250  # Long reasoning
            }
        ]
        mock_trainer = MagicMock()
        mock_trainer.llm_decisions = decisions
        mock_trainer.stats = {'recent_llm_decisions': []}
        
        api = TrainingAPI(trainer=mock_trainer)
        decisions = api.get_llm_decisions()
        
        assert len(decisions['recent_decisions']) == 2
        assert decisions['total_decisions'] == 2
        assert decisions['decision_rate'] > 0
        assert 500 < decisions['average_response_time_ms'] < 600
        
        # Check first decision (most recent)
        first = decisions['recent_decisions'][0]
        assert first['action'] == 4
        assert first['action_name'] == 'A'
        assert first['age_seconds'] == pytest.approx(10, abs=1)
        assert first['response_time_ms'] == 500
        assert first['reasoning_truncated'] == first['reasoning_full']
        
        # Check second decision
        second = decisions['recent_decisions'][1]
        assert second['action'] == 2
        assert second['action_name'] == 'UP'
        assert second['age_seconds'] == pytest.approx(20, abs=1)
        assert second['response_time_ms'] == 600
        assert len(second['reasoning_truncated']) == 203  # 200 + "..."
        assert len(second['reasoning_full']) == 250


class TestSystemAPI:
    """Tests for system status and monitoring API."""
    
    def test_get_system_status_with_trainer(self):
        """Test getting system status with active trainer."""
        mock_trainer = Mock()
        start_time = datetime.now(timezone.utc)
        mock_trainer.stats = {'start_time': start_time.isoformat()}
        
        mock_screen = Mock()
        mock_screen.pyboy = Mock()
        mock_screen.capture_active = True
        
        api = SystemAPI(trainer=mock_trainer, screen_capture=mock_screen)
        status = api.get_system_status()
        
        assert status['status'] == 'running'
        assert status['uptime'] >= 0
        assert status['version'] == '1.0.0'
        assert status['screen_capture_active'] is True
    
    def test_get_system_status_no_trainer(self):
        """Test getting system status without trainer."""
        api = SystemAPI()
        status = api.get_system_status()
        
        assert status['status'] == 'running'
        assert status['uptime'] >= 0
        assert status['version'] == '1.0.0'
        assert status['screen_capture_active'] is False
    
    def test_screen_capture_status_checks(self):
        """Test various screen capture status conditions."""
        mock_screen = Mock()
        api = SystemAPI(screen_capture=mock_screen)
        
        # No PyBoy
        mock_screen.pyboy = None
        mock_screen.capture_active = True
        assert api.get_system_status()['screen_capture_active'] is False
        
        # Inactive capture
        mock_screen.pyboy = Mock()
        mock_screen.capture_active = False
        assert api.get_system_status()['screen_capture_active'] is False
        
        # Mock PyBoy (test environment)
        mock_screen.pyboy = Mock()
        mock_screen.pyboy._mock_name = 'MockPyBoy'
        mock_screen.capture_active = True
        assert api.get_system_status()['screen_capture_active'] is False
        
        # Active capture
        mock_screen.pyboy = Mock(spec=[])  # Real PyBoy-like object
        mock_screen.capture_active = True
        assert api.get_system_status()['screen_capture_active'] is True


class TestGameAPI:
    """Tests for game state and control API."""
    
    def test_get_memory_debug_no_reader(self):
        """Test memory debug when reader import fails."""
        mock_trainer = MagicMock()
        api = GameAPI(trainer=mock_trainer)
        
        # Initial state should trigger the import failed error
        result = api.get_memory_debug()
        assert 'error' in result
        assert 'Memory reader not available - import failed' in result['error']
        assert 'timestamp' in result
    
    def test_get_memory_debug_no_pyboy(self):
        """Test memory debug without PyBoy instance."""
        mock_trainer = MagicMock()
        mock_trainer.pyboy = None
        api = GameAPI(trainer=mock_trainer)
        
        # Initially we should get import failed
        result = api.get_memory_debug()
        assert 'error' in result
        assert 'Memory reader not available - import failed' in result['error']
    
    def test_get_screen_bytes(self):
        """Test getting screen capture bytes."""
        mock_screen = Mock()
        mock_screen.get_latest_screen_bytes.return_value = b'test_image_data'
        
        api = GameAPI(screen_capture=mock_screen)
        screen_bytes = api.get_screen_bytes()
        
        assert screen_bytes == b'test_image_data'
        mock_screen.get_latest_screen_bytes.assert_called_once()
    
    def test_get_screen_data(self):
        """Test getting screen metadata."""
        mock_screen = Mock()
        test_data = {'width': 160, 'height': 144, 'format': 'RGB'}
        mock_screen.get_latest_screen_data.return_value = test_data
        
        api = GameAPI(screen_capture=mock_screen)
        screen_data = api.get_screen_data()
        
        assert screen_data == test_data
        mock_screen.get_latest_screen_data.assert_called_once()
    
    def test_component_updates(self):
        """Test updating component references."""
        api = GameAPI()
        
        mock_trainer = Mock()
        mock_screen = Mock()
        
        api.update_trainer(mock_trainer)
        api.update_screen_capture(mock_screen)
        
        assert api.trainer == mock_trainer
        assert api.screen_capture == mock_screen
