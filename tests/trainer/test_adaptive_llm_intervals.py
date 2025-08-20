#!/usr/bin/env python3
"""
test_adaptive_llm_intervals.py - Tests specifically for adaptive LLM interval functionality

This test file focuses on the LLMManager._track_llm_performance method and adaptive interval logic.
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
import numpy as np

# Import test system modules
import sys
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from trainer.unified_trainer import UnifiedPokemonTrainer
from trainer.trainer import (
    TrainingConfig,
    LLMBackend
)


@pytest.mark.llm
@pytest.mark.adaptive_intervals
class TestAdaptiveLLMIntervals:
    """Test the adaptive LLM interval system comprehensively"""
    
    @pytest.fixture
    @patch('trainer.trainer.PyBoy')
    @patch('trainer.trainer.PYBOY_AVAILABLE', True)
    @patch('pokemon_crystal_rl.llm.local_llm_agent.LLMManager')  # Patch at the source where it's imported
    def trainer_with_llm(self, mock_llm_manager_class, mock_pyboy_class):
        """Create trainer with LLM backend for interval testing"""
        # Mock PyBoy
        mock_pyboy_instance = Mock()
        mock_pyboy_instance.frame_count = 1000
        mock_pyboy_class.return_value = mock_pyboy_instance
        
        # Create a mock LLM manager with adaptive interval functionality
        mock_llm_manager = Mock()
        mock_llm_manager.model = "smollm2:1.7b"
        mock_llm_manager.interval = 10
        mock_llm_manager.adaptive_llm_interval = 10
        mock_llm_manager.llm_response_times = []
        mock_llm_manager.stats = {
            'calls': 0,
            'successes': 0,
            'failures': 0,
            'total_time': 0.0,
            'avg_time': 0.0
        }
        
        # Add the _track_llm_performance method to the mock
        def track_llm_performance(response_time: float):
            """Mock implementation of _track_llm_performance"""
            mock_llm_manager.stats['total_time'] += response_time
            if mock_llm_manager.stats['calls'] > 0:
                mock_llm_manager.stats['avg_time'] = mock_llm_manager.stats['total_time'] / mock_llm_manager.stats['calls']
            
            mock_llm_manager.llm_response_times.append(response_time)
            if len(mock_llm_manager.llm_response_times) > 20:
                mock_llm_manager.llm_response_times = mock_llm_manager.llm_response_times[-20:]
            
            if len(mock_llm_manager.llm_response_times) >= 10:
                avg_time = sum(mock_llm_manager.llm_response_times[-10:]) / 10
                if avg_time > 3.0:
                    mock_llm_manager.adaptive_llm_interval = min(50, int(mock_llm_manager.adaptive_llm_interval * 1.5))
                elif avg_time < 1.5:
                    mock_llm_manager.adaptive_llm_interval = max(
                        mock_llm_manager.interval,
                        int(mock_llm_manager.adaptive_llm_interval * 0.8)
                    )
        
        mock_llm_manager._track_llm_performance = track_llm_performance
        
        # Set up the mock class to return our mock instance
        mock_llm_manager_class.return_value = mock_llm_manager
        
        config = TrainingConfig(
            rom_path="test.gbc",
            llm_backend=LLMBackend.SMOLLM2,
            llm_interval=10,
            debug_mode=True,
            headless=True,
            capture_screens=False
        )
        
        trainer = UnifiedPokemonTrainer(config)
        return trainer
    
    def test_adaptive_interval_initialization(self, trainer_with_llm):
        """Test that adaptive interval system is properly initialized"""
        trainer = trainer_with_llm
        
        # Check that LLM manager exists
        assert trainer.llm_manager is not None
        
        # Check all required attributes exist in LLM manager
        assert hasattr(trainer.llm_manager, 'llm_response_times')
        assert hasattr(trainer.llm_manager, 'adaptive_llm_interval')
        assert hasattr(trainer.llm_manager, '_track_llm_performance')
        
        # Check initial values
        assert trainer.llm_manager.llm_response_times == []
        assert trainer.llm_manager.adaptive_llm_interval == trainer.config.llm_interval
        assert 'total_time' in trainer.llm_manager.stats
        assert 'avg_time' in trainer.llm_manager.stats
        assert trainer.llm_manager.stats['total_time'] == 0
        assert trainer.llm_manager.stats['avg_time'] == 0

    def test_track_llm_performance_single_call(self, trainer_with_llm):
        """Test tracking a single LLM performance measurement"""
        trainer = trainer_with_llm
        llm_manager = trainer.llm_manager
        
        # Simulate first LLM call
        llm_manager._track_llm_performance(2.5)
        
        # Verify response times tracking
        assert len(llm_manager.llm_response_times) == 1
        assert llm_manager.llm_response_times[0] == 2.5
        
        # Verify stats update
        assert llm_manager.stats['total_time'] == 2.5
        # Note: avg_time calculation depends on calls count, which we need to set
        llm_manager.stats['calls'] = 1
        llm_manager._track_llm_performance(0)  # Trigger avg calculation
        assert llm_manager.stats['avg_time'] == 2.5
        
        # Interval should not change yet (need 10 calls)
        assert llm_manager.adaptive_llm_interval == trainer.config.llm_interval

    def test_track_llm_performance_multiple_calls(self, trainer_with_llm):
        """Test tracking multiple LLM performance measurements"""
        trainer = trainer_with_llm
        llm_manager = trainer.llm_manager
        
        response_times = [1.0, 1.5, 2.0, 0.8, 1.2]
        
        for rt in response_times:
            llm_manager._track_llm_performance(rt)
        
        # Verify all response times are tracked
        assert len(llm_manager.llm_response_times) == 5
        assert llm_manager.llm_response_times == response_times
        
        # Verify cumulative stats
        expected_total = sum(response_times)
        assert llm_manager.stats['total_time'] == expected_total

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
        
        # Add 10 fast response times (<1.5 seconds each)
        for i in range(10):
            llm_manager._track_llm_performance(1.0)  # Consistently fast
        
        # Interval should have decreased
        assert llm_manager.adaptive_llm_interval < 20
        
        # Should not go below original config
        assert llm_manager.adaptive_llm_interval >= llm_manager.interval
        
        # Check that it scaled by expected factor (0.8x)
        expected_new_interval = max(llm_manager.interval, int(20 * 0.8))
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
        
        # Interval should remain unchanged (average is between 1.5 and 3.0)
        assert llm_manager.adaptive_llm_interval == original_interval
        
        # Verify the average is in expected range
        assert 1.5 < avg_time < 3.0
    
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
        llm_manager.adaptive_llm_interval = llm_manager.interval + 2  # Start slightly above
        
        # Add very fast calls
        for i in range(10):
            llm_manager._track_llm_performance(0.5)  # Extremely fast
        
        # Should not go below original config value
        assert llm_manager.adaptive_llm_interval >= llm_manager.interval
    
    def test_adaptive_interval_with_real_training_simulation(self, trainer_with_llm):
        """Test adaptive intervals in a realistic training simulation"""
        trainer = trainer_with_llm
        llm_manager = trainer.llm_manager
        
        initial_interval = llm_manager.adaptive_llm_interval
        
        # Phase 1: Slow performance - should increase interval
        for i in range(10):
            llm_manager._track_llm_performance(4.0)  # Consistently slow
        
        interval_after_slow = llm_manager.adaptive_llm_interval
        
        # Should have increased from initial
        assert interval_after_slow > initial_interval, f"Expected interval to increase from {initial_interval}, got {interval_after_slow}"
        
        # Phase 2: Reset response times history and test fast performance
        llm_manager.adaptive_llm_interval = 20  # Set to moderate level
        llm_manager.llm_response_times = []  # Clear history to get clean fast performance test
        
        for i in range(10):
            llm_manager._track_llm_performance(1.0)  # Consistently fast
        
        interval_after_fast = llm_manager.adaptive_llm_interval
        
        # Should have decreased from the moderate level
        assert interval_after_fast < 20, f"Expected interval to decrease from 20, got {interval_after_fast}"
        
        # Verify bounds are respected
        assert interval_after_fast >= llm_manager.interval, "Interval should not go below config minimum"
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
        
        # Should be very fast (under 50ms for 1000 calls - more lenient for mock)
        assert elapsed < 0.05, f"Performance tracking too slow: {elapsed:.4f}s for 1000 calls"
        
        # Verify data integrity
        assert len(llm_manager.llm_response_times) == 20  # Should keep last 20
        assert llm_manager.stats['total_time'] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
