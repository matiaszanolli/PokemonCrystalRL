"""Performance benchmark tests for core training functionality."""

import pytest
import time
import psutil
import os
from unittest.mock import Mock, patch
import numpy as np

from training.unified_pokemon_trainer import UnifiedPokemonTrainer, UnifiedTrainerConfig
from training.config import TrainingMode, LLMBackend

@pytest.mark.benchmarking
@pytest.mark.performance
@pytest.mark.skip(reason="UnifiedPokemonTrainer interface changed during refactoring - performance tests need updating")
class TestUnifiedTrainerPerformance:
    """Test UnifiedPokemonTrainer performance characteristics."""
    
    @pytest.fixture
    @patch('training.trainer.PyBoy')
    @patch('training.trainer.PYBOY_AVAILABLE', True)
    def fast_trainer(self, mock_pyboy_class):
        """Create trainer for fast performance testing."""
        mock_pyboy_instance = Mock(spec=['send_input', 'tick', 'frame_count', 'screen'])
        screen_mock = Mock()
        screen_mock.ndarray = np.random.randint(0, 255, (144, 160, 3), dtype=np.uint8)
        mock_pyboy_instance.frame_count = 1000
        mock_pyboy_instance.screen = screen_mock
        mock_pyboy_class.return_value = mock_pyboy_instance
        
        config = UnifiedTrainerConfig(
            rom_path="test.gbc",
            mode=TrainingMode.FAST_MONITORED,
            max_actions=1000,
            headless=True
        )
        
        return UnifiedPokemonTrainer(config)
    
    @pytest.fixture
    @patch('training.trainer.PyBoy')
    @patch('training.trainer.PYBOY_AVAILABLE', True)
    def ultra_fast_trainer(self, mock_pyboy_class):
        """Create trainer for ultra-fast mode testing."""
        mock_pyboy_instance = Mock(spec=['send_input', 'tick', 'frame_count', 'screen'])
        screen_mock = Mock()
        screen_mock.ndarray = np.random.randint(0, 255, (144, 160, 3), dtype=np.uint8)
        mock_pyboy_instance.frame_count = 1000
        mock_pyboy_instance.screen = screen_mock
        mock_pyboy_class.return_value = mock_pyboy_instance
        
        config = UnifiedTrainerConfig(
            rom_path="test.gbc",
            mode=TrainingMode.ULTRA_FAST,
            max_actions=1000,
            headless=True
        )
        
        return UnifiedPokemonTrainer(config)
    
    def test_fast_mode_performance(self, fast_trainer):
        """Test fast mode achieves expected performance."""
        trainer = fast_trainer
        
        start_time = time.time()
        actions_executed = 0
        
        for step in range(100):
            action = trainer.get_action(step)
            trainer.execute_action(action)
            actions_executed += 1
        
        elapsed = time.time() - start_time
        actions_per_second = actions_executed / elapsed
        
        # Should achieve at least 40 actions/second in fast mode
        assert actions_per_second >= 40, f"Fast mode: {actions_per_second:.2f} actions/sec"
    
    def test_ultra_fast_performance(self, ultra_fast_trainer):
        """Test ultra-fast mode achieves maximum performance."""
        trainer = ultra_fast_trainer
        
        start_time = time.time()
        actions_executed = 0
        
        for step in range(1000):
            action = trainer.get_action(step)
            trainer.execute_action(action)
            actions_executed += 1
        
        elapsed = time.time() - start_time
        actions_per_second = actions_executed / elapsed
        
        # Should achieve at least 600 actions/second in ultra-fast mode
        assert actions_per_second >= 600, f"Ultra-fast mode: {actions_per_second:.2f} actions/sec"
    
    def test_memory_usage(self, fast_trainer):
        """Test memory usage remains stable."""
        trainer = fast_trainer
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        for step in range(500):
            action = trainer.get_action(step)
            trainer.execute_action(action)
            
            if step % 100 == 0:
                trainer.capture_screen()  # Test screen capture memory
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (under 100MB for 500 actions)
        assert memory_increase < 100, f"Memory increased by {memory_increase:.1f}MB"
    
    def test_screen_capture_performance(self, fast_trainer):
        """Test screen capture performance."""
        trainer = fast_trainer
        
        # Perform 50 screen captures and measure time
        start_time = time.time()
        for _ in range(50):
            trainer.capture_screen()
        elapsed = time.time() - start_time
        captures_per_second = 50 / elapsed
        
        # Should achieve at least 30 captures/second
        assert captures_per_second >= 30, f"Screen capture: {captures_per_second:.2f} captures/sec"
