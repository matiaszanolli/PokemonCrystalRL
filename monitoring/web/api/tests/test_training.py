"""Tests for training API."""

import pytest
import time
from unittest.mock import Mock, PropertyMock
from collections import deque

from ..training import TrainingAPI, TrainingMetrics


@pytest.fixture
def mock_trainer():
    """Create mock trainer with stats."""
    trainer = Mock()
    trainer.stats = {
        'actions_taken': 100,
        'actions_per_second': 2.5,
        'llm_decision_count': 50,
        'total_reward': 75.5
    }
    return trainer

@pytest.fixture
def mock_trainer_with_decisions(mock_trainer):
    """Create mock trainer with decisions."""
    # Add decision history
    decisions = deque(maxlen=100)
    current_time = time.time()
    for i in range(5):
        decisions.append({
            'action': i % 8,  # Use all action types
            'timestamp': current_time - (5 - i) * 60,  # Each 1 minute apart
            'response_time': 0.5,
            'reasoning': f'Decision {i} reasoning'
        })
    type(mock_trainer).llm_decisions = PropertyMock(return_value=decisions)
    
    # Add stats decisions
    mock_trainer.stats['recent_llm_decisions'] = [
        {
            'action': 0,
            'timestamp': current_time - 360,  # 6 minutes ago
            'response_time_ms': 500,
            'reasoning': 'Old decision'
        }
    ]
    return mock_trainer

@pytest.fixture
def api():
    """Create training API instance."""
    return TrainingAPI()

def test_training_metrics():
    """Test training metrics dataclass."""
    # Create from dictionary
    data = {
        'actions_taken': 100,
        'actions_per_second': 2.5,
        'llm_decision_count': 50,
        'total_reward': 75.5
    }
    metrics = TrainingMetrics.from_dict(data)
    assert metrics.total_actions == 100
    assert metrics.actions_per_second == 2.5
    assert metrics.llm_calls == 50
    assert metrics.total_reward == 75.5
    
    # Create with defaults
    metrics = TrainingMetrics()
    assert metrics.total_actions == 0
    assert metrics.actions_per_second == 0.0
    assert metrics.llm_calls == 0
    assert metrics.total_reward == 0.0

def test_get_training_stats_with_stats(api, mock_trainer):
    """Test getting training stats from trainer stats."""
    api.trainer = mock_trainer
    stats = api.get_training_stats()
    
    assert stats['total_actions'] == 100
    assert stats['actions_per_second'] == 2.5
    assert stats['llm_calls'] == 50
    assert stats['total_reward'] == 75.5

def test_get_training_stats_with_method(api):
    """Test getting training stats from get_current_stats method."""
    trainer = Mock()
    trainer.get_current_stats = Mock(return_value={
        'actions_taken': 100,
        'actions_per_second': 2.5,
        'llm_decision_count': 50,
        'total_reward': 75.5
    })
    api.trainer = trainer
    
    stats = api.get_training_stats()
    assert stats['total_actions'] == 100
    assert stats['actions_per_second'] == 2.5
    assert stats['llm_calls'] == 50
    assert stats['total_reward'] == 75.5

def test_get_training_stats_no_trainer(api):
    """Test getting training stats without trainer."""
    stats = api.get_training_stats()
    assert stats['total_actions'] == 0
    assert stats['actions_per_second'] == 0.0
    assert stats['llm_calls'] == 0
    assert stats['total_reward'] == 0.0

def test_get_llm_decisions(api, mock_trainer_with_decisions):
    """Test getting LLM decisions."""
    api.trainer = mock_trainer_with_decisions
    decisions = api.get_llm_decisions()
    
    # Check basic structure
    assert 'recent_decisions' in decisions
    assert 'total_decisions' in decisions
    assert 'decision_rate' in decisions
    assert 'average_response_time_ms' in decisions
    assert 'timestamp' in decisions
    
    # Check decision processing
    recent = decisions['recent_decisions']
    assert len(recent) == 6  # 5 from deque + 1 from stats
    assert recent[0]['action_name'] in TrainingAPI.ACTION_NAMES.values()
    assert 'age_seconds' in recent[0]
    assert 'timestamp_readable' in recent[0]
    assert 'reasoning_truncated' in recent[0]
    assert 'reasoning_full' in recent[0]
    
    # Check statistics
    assert decisions['total_decisions'] == 6
    assert decisions['decision_rate'] > 0
    assert decisions['average_response_time_ms'] > 0
    assert decisions['last_decision_age_seconds'] >= 0

def test_get_llm_decisions_no_trainer(api):
    """Test getting LLM decisions without trainer."""
    decisions = api.get_llm_decisions()
    assert len(decisions['recent_decisions']) == 0
    assert decisions['total_decisions'] == 0
    assert decisions['decision_rate'] == 0.0
    assert decisions['average_response_time_ms'] == 0.0
    assert decisions['last_decision_age_seconds'] is None

def test_get_action_name(api):
    """Test action name conversion."""
    # Test known actions
    for action_id, name in TrainingAPI.ACTION_NAMES.items():
        assert api._get_action_name(action_id) == name
    
    # Test unknown action
    assert api._get_action_name(99) == "ACTION_99"

def test_update_trainer(api, mock_trainer):
    """Test trainer reference update."""
    api.update_trainer(mock_trainer)
    assert api.trainer == mock_trainer
