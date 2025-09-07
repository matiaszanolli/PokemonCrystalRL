"""Tests for game API."""

import pytest
import time
from unittest.mock import Mock, patch

from ..game import GameAPI


@pytest.fixture
def mock_trainer():
    """Create mock trainer."""
    trainer = Mock()
    trainer.pyboy = Mock()
    trainer.memory_reader = Mock()
    trainer.memory_reader.read_game_state.return_value = {
        'map_id': 1,
        'player_x': 10,
        'player_y': 20
    }
    trainer.memory_reader.get_debug_info.return_value = {
        'debug_key': 'debug_value'
    }
    return trainer

@pytest.fixture
def mock_screen_capture():
    """Create mock screen capture component."""
    capture = Mock()
    capture.get_latest_screen_bytes.return_value = b'test_image_data'
    capture.get_latest_screen_data.return_value = {
        'width': 160,
        'height': 144,
        'format': 'RGB'
    }
    return capture

@pytest.fixture
def api():
    """Create game API instance."""
    return GameAPI()

def test_get_memory_debug_with_memory_reader(api, mock_trainer):
    """Test getting memory debug info with existing reader."""
    api.trainer = mock_trainer
    state = api.get_memory_debug()
    
    assert 'map_id' in state
    assert state['map_id'] == 1
    assert state['player_x'] == 10
    assert state['player_y'] == 20
    assert 'debug_info' in state
    assert state['debug_info']['debug_key'] == 'debug_value'

def test_get_memory_debug_initialize_reader(api, mock_trainer):
    """Test memory reader initialization."""
    api.trainer = mock_trainer
    api.trainer.memory_reader = None
    
    with patch('trainer.memory_reader.PokemonCrystalMemoryReader') as MockReader:
        MockReader.return_value = mock_trainer.memory_reader
        state = api.get_memory_debug()
        
        MockReader.assert_called_once_with(mock_trainer.pyboy)
        assert state['map_id'] == 1

def test_get_memory_debug_no_pyboy(api, mock_trainer):
    """Test memory debug without PyBoy instance."""
    api.trainer = mock_trainer
    api.trainer.memory_reader = None
    api.trainer.pyboy = None
    
    state = api.get_memory_debug()
    assert 'error' in state
    assert 'PyBoy instance not available' in state['error']
    assert 'timestamp' in state

def test_get_memory_debug_import_error(api):
    """Test memory debug with import error."""
    with patch.object(api, 'PokemonCrystalMemoryReader', None):
        state = api.get_memory_debug()
        assert 'error' in state
        assert 'Memory reader not available' in state['error']
        assert 'timestamp' in state

def test_get_screen_bytes(api, mock_screen_capture):
    """Test getting screen bytes."""
    api.screen_capture = mock_screen_capture
    screen_data = api.get_screen_bytes()
    
    assert screen_data == b'test_image_data'
    mock_screen_capture.get_latest_screen_bytes.assert_called_once()

def test_get_screen_bytes_no_capture(api):
    """Test getting screen bytes without capture component."""
    assert api.get_screen_bytes() is None

def test_get_screen_data(api, mock_screen_capture):
    """Test getting screen metadata."""
    api.screen_capture = mock_screen_capture
    screen_info = api.get_screen_data()
    
    assert screen_info['width'] == 160
    assert screen_info['height'] == 144
    assert screen_info['format'] == 'RGB'
    mock_screen_capture.get_latest_screen_data.assert_called_once()

def test_get_screen_data_no_capture(api):
    """Test getting screen data without capture component."""
    assert api.get_screen_data() is None

def test_make_error_response(api):
    """Test error response creation."""
    error = api._make_error_response('Test error')
    
    assert error['error'] == 'Test error'
    assert 'timestamp' in error
    assert isinstance(error['timestamp'], float)

def test_update_references(api, mock_trainer, mock_screen_capture):
    """Test updating component references."""
    api.update_trainer(mock_trainer)
    api.update_screen_capture(mock_screen_capture)
    
    assert api.trainer == mock_trainer
    assert api.screen_capture == mock_screen_capture
