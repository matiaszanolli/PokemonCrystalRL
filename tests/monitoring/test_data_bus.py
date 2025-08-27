"""Test monitoring data bus integration."""

import time
import queue
import threading
import pytest
import numpy as np
from unittest.mock import Mock, patch

from trainer.trainer import TrainingConfig, PokemonTrainer
from monitoring.data_bus import DataBus, DataType

@pytest.fixture
def data_bus():
    """Create test data bus instance."""
    bus = DataBus()
    yield bus
    bus.stop()

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
    
    def test_data_bus_initialization(self, data_bus, mock_pyboy):
        """Test data bus initialization."""
        config = TrainingConfig(
            rom_path='test.gbc',
            mode=TrainingConfig.TrainingMode.FAST_MONITORED
        )
        
        trainer = PokemonTrainer(config)
        assert trainer.data_bus is not None
        assert trainer.data_bus._running
        
        trainer._finalize_training()
    
    def test_data_publishing(self, data_bus, mock_pyboy):
        """Test data publishing to bus."""
        config = TrainingConfig(
            rom_path='test.gbc',
            mode=TrainingConfig.TrainingMode.FAST_MONITORED
        )
        
        trainer = PokemonTrainer(config)
        
        # Subscribe to updates
        updates = []
        def callback(data):
            updates.append(data)
        
        data_bus.subscribe(DataType.TRAINING_STATS, callback)
        
        # Generate some updates
        for i in range(5):
            trainer.stats['total_actions'] = i
            trainer._update_stats()
        
        # Give time for updates to process
        time.sleep(0.1)
        
        assert len(updates) == 5
        assert updates[-1]['total_actions'] == 4
        
        trainer._finalize_training()
    
    def test_screen_data_publishing(self, data_bus, mock_pyboy):
        """Test screen data publishing."""
        config = TrainingConfig(
            rom_path='test.gbc',
            mode=TrainingConfig.TrainingMode.FAST_MONITORED,
            capture_screens=True
        )
        
        trainer = PokemonTrainer(config)
        
        # Subscribe to screen updates
        screens = []
        def callback(data):
            screens.append(data)
            
        data_bus.subscribe(DataType.GAME_SCREEN, callback)
        
        # Generate some screens
        for _ in range(3):
            trainer._capture_and_queue_screen()
            time.sleep(0.05)
        
        assert len(screens) >= 1
        assert 'screen' in screens[0]
        assert isinstance(screens[0]['screen'], np.ndarray)
        
        trainer._finalize_training()
    
    def test_data_bus_shutdown(self, data_bus, mock_pyboy):
        """Test clean data bus shutdown."""
        config = TrainingConfig(
            rom_path='test.gbc',
            mode=TrainingConfig.TrainingMode.FAST_MONITORED
        )
        
        trainer = PokemonTrainer(config)
        assert trainer.data_bus._running
        
        trainer._finalize_training()
        assert not trainer.data_bus._running
    
    def test_data_bus_error_handling(self, data_bus, mock_pyboy):
        """Test data bus error handling."""
        config = TrainingConfig(
            rom_path='test.gbc',
            mode=TrainingConfig.TrainingMode.FAST_MONITORED
        )
        
        trainer = PokemonTrainer(config)
        
        def failing_callback(data):
            raise Exception("Callback failed")
        
        # Add failing callback
        data_bus.subscribe(DataType.TRAINING_STATS, failing_callback)
        
        # Should continue despite callback failure
        trainer._update_stats()
        assert trainer.data_bus._running
        
        trainer._finalize_training()
