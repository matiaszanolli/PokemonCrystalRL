#!/usr/bin/env python3
"""
test_adaptive_llm_intervals.py - Tests specifically for adaptive LLM interval functionality

This test file focuses on the LLMManager._track_llm_performance method and adaptive interval logic.
"""

import pytest
import time
from unittest.mock import Mock, patch
import numpy as np

# Import test system modules
import sys
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from trainer import (
    TrainingConfig,
    LLMBackend,
    UnifiedPokemonTrainer
)


@pytest.mark.llm
@pytest.mark.adaptive_intervals
class TestAdaptiveLLMIntervals:
    """Test the adaptive LLM interval system comprehensively"""
    
    @pytest.fixture
    @patch('trainer.trainer.PyBoy')  # Correct import path
    @patch('trainer.trainer.PYBOY_AVAILABLE', True)  # Correct import path
    def trainer_with_llm(self, mock_pyboy_class):
        """Create trainer with LLM backend for interval testing"""
        mock_pyboy_instance = Mock()
        mock_pyboy_instance.frame_count = 1000
        mock_pyboy_class.return_value = mock_pyboy_instance
        
        config = TrainingConfig(
            rom_path="test.gbc",
            llm_backend=LLMBackend.SMOLLM2,
            llm_interval=10,  # Start with default
            debug_mode=True,
            headless=True,
            capture_screens=False
        )
        
        return UnifiedPokemonTrainer(config)
    
    def test_adaptive_interval_initialization(self, trainer_with_llm):
        """Test that adaptive interval system is properly initialized"""
        trainer = trainer_with_llm
        
        # Check that LLM manager exists
        assert trainer.llm_manager is not None
        
        # Check all required attributes exist in TRAINER (not llm_manager)
        assert hasattr(trainer, 'llm_response_times')
        assert hasattr(trainer, 'adaptive_llm_interval')
        assert hasattr(trainer, '_track_llm_performance')
        
        # Check initial values
        assert trainer.llm_response_times == []
        assert trainer.adaptive_llm_interval == trainer.config.llm_interval
        assert 'llm_total_time' in trainer.stats
        assert 'llm_avg_time' in trainer.stats
        assert trainer.stats['llm_total_time'] == 0
        assert trainer.stats['llm_avg_time'] == 0

    def test_track_llm_performance_single_call(self, trainer_with_llm):
        """Test tracking a single LLM performance measurement"""
        trainer = trainer_with_llm
        
        # Simulate first LLM call - use TRAINER method, not llm_manager
        trainer._track_llm_performance(2.5)
        
        # Verify response times tracking
        assert len(trainer.llm_response_times) == 1
        assert trainer.llm_response_times[0] == 2.5
        
        # Verify stats update
        assert trainer.stats['llm_total_time'] == 2.5
        assert trainer.stats['llm_avg_time'] == 2.5
        
        # Interval should not change yet (need 10 calls)
        assert trainer.adaptive_llm_interval == trainer.config.llm_interval

    def test_track_llm_performance_multiple_calls(self, trainer_with_llm):
        """Test tracking multiple LLM performance measurements"""
        trainer = trainer_with_llm
        
        response_times = [1.0, 1.5, 2.0, 0.8, 1.2]
        
        for rt in response_times:
            trainer._track_llm_performance(rt)  # Use trainer method
        
        # Verify all response times are tracked
        assert len(trainer.llm_response_times) == 5
        assert trainer.llm_response_times == response_times
        
        # Verify cumulative stats
        expected_total = sum(response_times)
        expected_avg = expected_total / len(response_times)
        
        assert trainer.stats['llm_total_time'] == expected_total
        assert abs(trainer.stats['llm_avg_time'] - expected_avg) < 0.001

    
    def test_response_times_window_management(self, trainer_with_llm):
        """Test that response times window is properly managed (max 20 entries)"""
        trainer = trainer_with_llm
        llm_manager = trainer.llm_manager
        
        # Add 25 response times (more than the 20 limit)
        for i in range(25):
            llm_manager._track_llm_performance(1.0 + i * 0.1)
        
        # Should only keep last 20
        assert len(llm_manager.llm_response_times) == 20
        
        # Should have the most recent values (from call 6 to call 25)
        expected_first = 1.0 + 5 * 0.1  # 6th call (0-indexed)
        expected_last = 1.0 + 24 * 0.1   # 25th call
        
        assert abs(llm_manager.llm_response_times[0] - expected_first) < 0.001
        assert abs(llm_manager.llm_response_times[-1] - expected_last) < 0.001
    
    def test_adaptive_interval_slow_llm_increase(self, trainer_with_llm):
        """Test that interval increases when LLM calls are consistently slow"""
        trainer = trainer_with_llm
        llm_manager = trainer.llm_manager
        
        original_interval = llm_manager.adaptive_llm_interval
        
        # Add 10 slow response times (>3 seconds each)
        for i in range(10):
            llm_manager._track_llm_performance(4.0)  # Consistently slow
        
        # Interval should have increased
        assert llm_manager.adaptive_llm_interval > original_interval
        
        # Should not exceed maximum
        assert llm_manager.adaptive_llm_interval <= 50
        
        # Check that it scaled by expected factor (1.5x)
        expected_new_interval = min(50, int(original_interval * 1.5))
        assert llm_manager.adaptive_llm_interval == expected_new_interval
    
    def test_adaptive_interval_fast_llm_decrease(self, trainer_with_llm):
        """Test that interval decreases when LLM calls are consistently fast"""
        trainer = trainer_with_llm
        llm_manager = trainer.llm_manager
        
        # First, artificially increase the interval
        llm_manager.adaptive_llm_interval = 20
        
        # Add 10 fast response times (<1 second each)
        for i in range(10):
            llm_manager._track_llm_performance(0.5)  # Consistently fast
        
        # Interval should have decreased
        assert llm_manager.adaptive_llm_interval < 20
        
        # Should not go below original config
        assert llm_manager.adaptive_llm_interval >= trainer.config.llm_interval
        
        # Check that it scaled by expected factor (0.8x)
        expected_new_interval = max(trainer.config.llm_interval, int(20 * 0.8))
        assert llm_manager.adaptive_llm_interval == expected_new_interval
    
    def test_adaptive_interval_adjustment_timing(self, trainer_with_llm):
        """Test that interval adjustment only happens every 10 calls"""
        trainer = trainer_with_llm
        llm_manager = trainer.llm_manager
        
        original_interval = llm_manager.adaptive_llm_interval
        
        # Add 9 slow response times (should not trigger adjustment)
        for i in range(9):
            llm_manager._track_llm_performance(4.0)
        
        # Interval should not have changed yet
        assert llm_manager.adaptive_llm_interval == original_interval
        
        # Add 10th slow response time (should trigger adjustment)
        llm_manager._track_llm_performance(4.0)
        
        # Now interval should have increased
        assert llm_manager.adaptive_llm_interval > original_interval
    
    def test_adaptive_interval_mixed_performance(self, trainer_with_llm):
        """Test adaptive interval with mixed fast/slow performance"""
        trainer = trainer_with_llm
        llm_manager = trainer.llm_manager
        
        original_interval = llm_manager.adaptive_llm_interval
        
        # Add mixed response times (average around 2.0s - should not trigger changes)
        mixed_times = [1.0, 3.0, 1.5, 2.5, 2.0, 1.8, 2.2, 1.7, 2.3, 2.0]
        avg_time = sum(mixed_times) / len(mixed_times)
        
        for rt in mixed_times:
            llm_manager._track_llm_performance(rt)
        
        # Interval should remain unchanged (average is between 1.0 and 3.0)
        assert llm_manager.adaptive_llm_interval == original_interval
        
        # Verify the average is in expected range
        assert 1.0 < avg_time < 3.0
    
    def test_adaptive_interval_bounds_enforcement(self, trainer_with_llm):
        """Test that adaptive interval respects minimum and maximum bounds"""
        trainer = trainer_with_llm
        llm_manager = trainer.llm_manager
        
        # Test maximum bound (should not exceed 50)
        llm_manager.adaptive_llm_interval = 40  # Start close to max
        
        # Add very slow calls
        for i in range(10):
            llm_manager._track_llm_performance(10.0)  # Extremely slow
        
        # Should be capped at 50
        assert llm_manager.adaptive_llm_interval == 50
        
        # Test minimum bound (should not go below original config)
        llm_manager.adaptive_llm_interval = trainer.config.llm_interval + 2  # Start slightly above
        
        # Add very fast calls
        for i in range(10):
            llm_manager._track_llm_performance(0.1)  # Extremely fast
        
        # Should not go below original config value
        assert llm_manager.adaptive_llm_interval >= trainer.config.llm_interval
    
    def test_adaptive_interval_with_real_training_simulation(self, trainer_with_llm):
        """Test adaptive intervals in a realistic training simulation"""
        trainer = trainer_with_llm
        llm_manager = trainer.llm_manager
        
        with patch('trainer.llm_manager.ollama') as mock_ollama:
            initial_interval = llm_manager.adaptive_llm_interval
            
            # Phase 1: Slow performance - should increase interval
            for i in range(10):
                llm_manager._track_llm_performance(4.0)  # Consistently slow
            
            interval_after_slow = llm_manager.adaptive_llm_interval
            
            # Should have increased from initial
            assert interval_after_slow > initial_interval, f"Expected interval to increase from {initial_interval}, got {interval_after_slow}"
            
            # Phase 2: Reset to moderate interval and test fast performance
            llm_manager.adaptive_llm_interval = 20  # Set to moderate level
            
            for i in range(10):
                llm_manager._track_llm_performance(0.5)  # Consistently fast
            
            interval_after_fast = llm_manager.adaptive_llm_interval
            
            # Should have decreased from the moderate level
            assert interval_after_fast < 20, f"Expected interval to decrease from 20, got {interval_after_fast}"
            
            # Verify bounds are respected
            assert interval_after_fast >= trainer.config.llm_interval, "Interval should not go below config minimum"
            assert interval_after_slow <= 50, "Interval should not exceed maximum of 50"
            
            print(f"Interval adaptation test: {initial_interval} -> {interval_after_slow} (slow) -> {interval_after_fast} (fast)")
    
    def test_performance_tracking_overhead(self, trainer_with_llm):
        """Test that performance tracking itself has minimal overhead"""
        trainer = trainer_with_llm
        llm_manager = trainer.llm_manager
        
        # Measure overhead of tracking 1000 calls
        start_time = time.perf_counter()
        
        for i in range(1000):
            llm_manager._track_llm_performance(1.0 + (i % 10) * 0.1)
        
        elapsed = time.perf_counter() - start_time
        
        # Should be very fast (under 5ms for 1000 calls)
        assert elapsed < 0.005, f"Performance tracking too slow: {elapsed:.4f}s for 1000 calls"
        
        # Verify data integrity
        assert len(llm_manager.llm_response_times) == 20  # Should keep last 20
        assert llm_manager.stats['llm_total_time'] > 0
        assert llm_manager.stats['llm_avg_time'] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
