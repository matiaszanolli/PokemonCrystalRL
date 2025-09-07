"""Tests for metrics processing service."""

import pytest
import time
from unittest.mock import Mock
from typing import Dict, Any

from .metrics import MetricsService, MetricsConfig, MetricHistory
from ..components.metrics import MetricsCollector


@pytest.fixture
def config():
    """Metrics service configuration for testing."""
    return MetricsConfig(
        history_size=100,
        update_interval=0.1,
        retention_hours=1.0
    )

@pytest.fixture
def metrics_service(config):
    """Create metrics service instance."""
    return MetricsService(config)

@pytest.fixture
def mock_collector():
    """Mock metrics collector component."""
    collector = Mock(spec=MetricsCollector)
    collector.get_metrics.return_value = {
        'episode': 1,
        'total_steps': 100,
        'total_reward': 50.0
    }
    return collector

def test_metrics_service_init(metrics_service, config):
    """Test metrics service initialization."""
    assert metrics_service.config == config
    assert not metrics_service._running
    assert metrics_service._total_recorded == 0
    assert len(metrics_service._active_metrics) == 0
    
    # Check history initialization
    assert 'reward' in metrics_service._history
    assert 'steps' in metrics_service._history
    assert 'pokemon_count' in metrics_service._history
    assert 'badge_count' in metrics_service._history
    assert 'money' in metrics_service._history
    assert 'experience' in metrics_service._history
    assert 'exploration' in metrics_service._history
    assert 'cpu_percent' in metrics_service._history
    assert 'memory_usage' in metrics_service._history
    assert 'network_bytes_sec' in metrics_service._history

def test_set_metrics_collector(metrics_service, mock_collector):
    """Test setting metrics collector component."""
    metrics_service.set_metrics_collector(mock_collector)
    assert metrics_service._collector == mock_collector

def test_update_training_metrics(metrics_service):
    """Test training metrics updates."""
    training_data = {
        'episode': 1,
        'total_steps': 100,
        'total_reward': 50.0,
        'experience': 75.5,
        'exploration': 0.25
    }
    
    metrics_service.update_training_metrics(training_data)
    assert metrics_service._total_recorded == len(training_data)
    assert len(metrics_service._active_metrics) == len(training_data)
    
    # Check history updates
    assert len(metrics_service._history['reward'].values) == 1
    assert metrics_service._history['reward'].values[0] == 50.0
    assert len(metrics_service._history['experience'].values) == 1
    assert metrics_service._history['experience'].values[0] == 75.5

def test_update_game_metrics(metrics_service):
    """Test game metrics updates."""
    game_data = {
        'party_count': 3,
        'badges_total': 2,
        'money': 1000
    }
    
    metrics_service.update_game_metrics(game_data)
    assert metrics_service._total_recorded == len(game_data)
    assert len(metrics_service._active_metrics) == len(game_data)
    
    # Check history updates
    assert len(metrics_service._history['pokemon_count'].values) == 1
    assert metrics_service._history['pokemon_count'].values[0] == 3
    assert len(metrics_service._history['badge_count'].values) == 1
    assert metrics_service._history['badge_count'].values[0] == 2

def test_update_resource_metrics(metrics_service):
    """Test resource metrics updates."""
    resource_data = {
        'cpu_percent': 25.5,
        'memory_usage_mb': 512,
        'network_bytes_sec': 1024
    }
    
    metrics_service.update_resource_metrics(resource_data)
    assert metrics_service._total_recorded == len(resource_data)
    assert len(metrics_service._active_metrics) == len(resource_data)
    
    # Check history updates
    assert len(metrics_service._history['cpu_percent'].values) == 1
    assert metrics_service._history['cpu_percent'].values[0] == 25.5
    assert len(metrics_service._history['memory_usage'].values) == 1
    assert metrics_service._history['memory_usage'].values[0] == 512

def test_get_metrics(metrics_service):
    """Test metrics retrieval."""
    # Add some metrics
    training_data = {'episode': 1, 'total_reward': 50.0}
    game_data = {'party_count': 3, 'money': 1000}
    resource_data = {'cpu_percent': 25.5}
    
    metrics_service.update_training_metrics(training_data)
    metrics_service.update_game_metrics(game_data)
    metrics_service.update_resource_metrics(resource_data)
    
    # Get all metrics
    metrics = metrics_service.get_metrics()
    assert len(metrics) == 5
    assert metrics['episode'] == 1
    assert metrics['total_reward'] == 50.0
    assert metrics['party_count'] == 3
    
    # Get specific metrics
    metrics = metrics_service.get_metrics(names=['episode', 'money'])
    assert len(metrics) == 2
    assert metrics['episode'] == 1
    assert metrics['money'] == 1000

def test_get_metrics_with_history(metrics_service):
    """Test metrics retrieval with history."""
    # Add metrics at different times
    start_time = time.time()
    
    metrics_service.update_training_metrics({
        'total_reward': 50.0,
        'experience': 75.5
    })
    
    time.sleep(0.1)
    metrics_service.update_training_metrics({
        'total_reward': 60.0,
        'experience': 80.5
    })
    
    # Get metrics with history
    metrics = metrics_service.get_metrics(
        names=['total_reward'],
        since=start_time
    )
    
    assert 'total_reward' in metrics
    assert 'total_reward_history' in metrics
    assert len(metrics['total_reward_history']) == 2
    assert metrics['total_reward_history'][0]['value'] == 50.0

def test_get_chart_data(metrics_service):
    """Test chart data formatting."""
    # Add some progress data
    metrics_service.update_training_metrics({
        'total_reward': 50.0,
        'experience': 75.5,
        'exploration': 0.25
    })
    metrics_service.update_training_metrics({
        'total_reward': 60.0,
        'experience': 80.5,
        'exploration': 0.20
    })
    
    chart_data = metrics_service.get_chart_data()
    assert 'reward_history' in chart_data
    assert 'progress' in chart_data
    
    assert len(chart_data['reward_history']) == 2
    assert len(chart_data['progress']) == 2
    assert chart_data['progress'][0]['experience'] == 75.5
    assert chart_data['progress'][0]['exploration'] == 0.25

def test_get_status(metrics_service):
    """Test status reporting."""
    # Add some metrics
    metrics_service.update_training_metrics({'episode': 1})
    metrics_service.update_game_metrics({'party_count': 3})
    
    status = metrics_service.get_status()
    assert 'running' in status
    assert 'active_metrics' in status
    assert 'total_recorded' in status
    assert status['active_metrics'] == 2
    assert status['total_recorded'] == 2

def test_clear(metrics_service):
    """Test metrics clearing."""
    # Add some metrics
    metrics_service.update_training_metrics({'episode': 1})
    metrics_service.update_game_metrics({'party_count': 3})
    assert metrics_service._total_recorded > 0
    
    # Clear metrics
    metrics_service.clear()
    assert metrics_service._total_recorded == 0
    assert len(metrics_service._active_metrics) == 0
    assert len(metrics_service._metrics) == 0
    
    # Check histories are cleared
    for history in metrics_service._history.values():
        assert len(history.values) == 0
        assert len(history.timestamps) == 0

def test_metric_history():
    """Test metric history functionality."""
    history = MetricHistory()
    
    # Add values
    start_time = time.time()
    history.add(1.0, start_time)
    history.add(2.0, start_time + 1)
    history.add(3.0, start_time + 2)
    
    # Get all values since start
    values = history.get_since(start_time)
    assert len(values) == 3
    assert values[0]['value'] == 1.0
    assert values[1]['value'] == 2.0
    assert values[2]['value'] == 3.0
    
    # Get values since middle
    values = history.get_since(start_time + 1)
    assert len(values) == 2
    assert values[0]['value'] == 2.0
    assert values[1]['value'] == 3.0
    
    # Clear history
    history.clear()
    assert len(history.values) == 0
    assert len(history.timestamps) == 0
