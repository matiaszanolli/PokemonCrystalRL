"""Tests for system API."""

import pytest
import time
from unittest.mock import Mock, PropertyMock
from datetime import datetime, timedelta

from ..system import SystemAPI, SystemStatus


@pytest.fixture
def mock_trainer():
    """Create mock trainer with stats."""
    trainer = Mock()
    start_time = datetime.now() - timedelta(hours=1)
    trainer.stats = {
        'start_time': start_time.isoformat()
    }
    return trainer

@pytest.fixture
def mock_screen_capture():
    """Create mock screen capture component."""
    capture = Mock()
    capture.capture_active = True
    capture.pyboy = Mock()  # Not a mock in _mock_name sense
    return capture

@pytest.fixture
def api():
    """Create system API instance."""
    return SystemAPI()

def test_system_status():
    """Test system status dataclass."""
    # Create with defaults
    status = SystemStatus()
    assert status.status == 'running'
    assert status.uptime == 0.0
    assert status.version == '1.0.0'
    assert not status.screen_capture_active
    
    # Create with values
    status = SystemStatus(
        status='paused',
        uptime=3600.0,
        version='2.0.0',
        screen_capture_active=True
    )
    data = status.to_dict()
    assert data['status'] == 'paused'
    assert data['uptime'] == 3600.0
    assert data['version'] == '2.0.0'
    assert data['screen_capture_active']

def test_get_system_status_with_trainer(api, mock_trainer):
    """Test getting system status with trainer."""
    api.trainer = mock_trainer
    status = api.get_system_status()
    
    assert status['status'] == 'running'
    assert status['uptime'] > 3500  # ~1 hour
    assert status['version'] == '1.0.0'
    assert not status['screen_capture_active']

def test_get_system_status_with_capture(api, mock_screen_capture):
    """Test getting system status with screen capture."""
    api.screen_capture = mock_screen_capture
    status = api.get_system_status()
    
    assert status['screen_capture_active']

def test_get_system_status_no_components(api):
    """Test getting system status without components."""
    status = api.get_system_status()
    
    assert status['status'] == 'running'
    assert status['uptime'] >= 0
    assert status['version'] == '1.0.0'
    assert not status['screen_capture_active']

def test_check_screen_capture_status(api, mock_screen_capture):
    """Test screen capture status checking."""
    # Active capture
    api.screen_capture = mock_screen_capture
    assert api._check_screen_capture_status()
    
    # Inactive capture
    mock_screen_capture.capture_active = False
    assert not api._check_screen_capture_status()
    
    # Mock PyBoy (in tests)
    type(mock_screen_capture.pyboy)._mock_name = PropertyMock(return_value='MockPyBoy')
    assert not api._check_screen_capture_status()
    
    # No capture
    api.screen_capture = None
    assert not api._check_screen_capture_status()

def test_invalid_start_time(api, mock_trainer):
    """Test handling invalid start time."""
    # Invalid format
    mock_trainer.stats['start_time'] = 'invalid'
    status = api.get_system_status()
    assert status['uptime'] >= 0  # Should use current time
    
    # Missing start time
    del mock_trainer.stats['start_time']
    status = api.get_system_status()
    assert status['uptime'] >= 0

def test_update_references(api, mock_trainer, mock_screen_capture):
    """Test updating component references."""
    api.update_trainer(mock_trainer)
    api.update_screen_capture(mock_screen_capture)
    
    assert api.trainer == mock_trainer
    assert api.screen_capture == mock_screen_capture
