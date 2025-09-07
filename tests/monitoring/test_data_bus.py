"""Test monitoring data bus integration."""

import time
import queue
import threading
import pytest
import numpy as np
from unittest.mock import Mock, patch

from training.trainer import TrainingConfig, PokemonTrainer
from monitoring.data_bus import DataBus, DataType
from config.config import TrainingMode

@pytest.fixture
def data_bus(test_config):
    """Create test data bus instance."""
    bus = DataBus()
    yield bus
    bus.shutdown()

@pytest.fixture(autouse=True)
def mock_data_bus_init(monkeypatch, data_bus):
    """Mock get_data_bus to return the test fixture."""
    def mock_get_data_bus():
        return data_bus
    monkeypatch.setattr('monitoring.data_bus.get_data_bus', mock_get_data_bus)

@pytest.fixture
def mock_pyboy():
    """Mock PyBoy for testing."""
    with patch('trainer.trainer.PyBoy') as mock:
        mock_instance = Mock()
        mock_instance.frame_count = 0
        mock_instance.screen.ndarray = np.random.randint(0, 255, (144, 160, 3))
        mock_instance.send_input = Mock()
        mock_instance.tick = Mock()
        mock.return_value = mock_instance
        yield mock

@pytest.mark.monitoring
@pytest.mark.integration
class TestDataBusIntegration:
    """Test data bus integration."""
    
    def test_data_bus_initialization(self, data_bus, mock_pyboy, test_config):
        """Test data bus initialization."""
        trainer = PokemonTrainer(test_config)
        assert trainer.data_bus is not None
        assert trainer.data_bus._running
        
        trainer._finalize_training()
    
    def test_data_publishing(self, data_bus, mock_pyboy, test_config):
        """Test data publishing to bus."""
        trainer = PokemonTrainer(test_config)
        
        # Subscribe to updates
        updates = []
        def callback(data):
            updates.append(data)
        
        data_bus.subscribe(DataType.TRAINING_STATS, "test_subscriber", callback)
        
        # Generate some updates
        for i in range(5):
            trainer.stats['total_actions'] = i
            trainer._update_stats()
        
        # Give time for updates to process
        time.sleep(0.1)
        
        assert len(updates) == 5
        assert updates[-1]['total_actions'] == 4
        
        trainer._finalize_training()
    
    def test_screen_data_publishing(self, data_bus, mock_pyboy, test_config):
        """Test screen data publishing."""
        trainer = PokemonTrainer(test_config)
        
        # Subscribe to screen updates
        screens = []
        def callback(data):
            screens.append(data)
            
        data_bus.subscribe(DataType.GAME_SCREEN, "test_subscriber", callback)
        
        # Generate some screens
        mock_screen = np.random.randint(0, 255, (144, 160, 3), dtype=np.uint8)
        screen_data = {
            "image": mock_screen,
            "timestamp": time.time(),
            "frame": 0
        }
        
        # Simulate screen capture events
        for _ in range(3):
            trainer.data_bus.publish(DataType.GAME_SCREEN, screen_data, "trainer")
            time.sleep(0.05)
        
        assert len(screens) >= 1
        assert 'image' in screens[0]
        assert isinstance(screens[0]['image'], np.ndarray)
        
        trainer._finalize_training()
    
    def test_data_bus_shutdown(self, data_bus, mock_pyboy, test_config):
        """Test clean data bus shutdown."""
        trainer = PokemonTrainer(test_config)
        assert trainer.data_bus._running
        
        trainer.data_bus.shutdown()
        assert not trainer.data_bus._running
    
    def test_data_bus_error_handling(self, data_bus, mock_pyboy, test_config):
        """Test data bus error handling."""
        trainer = PokemonTrainer(test_config)
        
        def failing_callback(data):
            raise Exception("Callback failed")
        
        # Add failing callback
        data_bus.subscribe(DataType.TRAINING_STATS, "test_subscriber", failing_callback)
        
        # Should continue despite callback failure
        trainer._update_stats()
        assert trainer.data_bus._running
        
        trainer._finalize_training()
