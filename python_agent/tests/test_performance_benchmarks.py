#!/usr/bin/env python3
"""
test_performance_benchmarks.py - Performance Benchmarking Tests

Tests performance characteristics including:
- ~2.3 actions/second target performance
- LLM inference timing (25ms SmolLM2, 30ms Llama3.2-1B, 60ms Llama3.2-3B)
- Memory usage patterns
- Training mode performance comparison
- Real-world usage scenario benchmarks
"""

import pytest
import time
import psutil
import os
import gc
import threading
from statistics import mean, stdev
from unittest.mock import Mock, patch, MagicMock
import numpy as np

# Import test system modules
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# Import the enhanced trainer system
try:
    from pokemon_trainer import (
        UnifiedPokemonTrainer,
        TrainingConfig,
        TrainingMode,
        LLMBackend
    )
except ImportError:
    # Fallback import path
    from scripts.pokemon_trainer import (
        UnifiedPokemonTrainer,
        TrainingConfig,
        TrainingMode,
        LLMBackend
    )


@pytest.mark.benchmarking
@pytest.mark.performance
class TestActionPerformanceBenchmarks:
    """Test action execution performance benchmarks"""
    
    @pytest.fixture
    @patch('scripts.pokemon_trainer.PyBoy')
    @patch('trainer.trainer.PYBOY_AVAILABLE', True)
    def trainer_fast_monitored(self, mock_pyboy_class):
        """Create trainer optimized for fast monitored performance"""
        mock_pyboy_instance = Mock()
        mock_pyboy_instance.frame_count = 1000
        mock_pyboy_instance.screen.ndarray = np.random.randint(0, 255, (144, 160, 3), dtype=np.uint8)
        mock_pyboy_instance.send_input = Mock()
        mock_pyboy_instance.tick = Mock()
        mock_pyboy_class.return_value = mock_pyboy_instance
        
        config = TrainingConfig(
            rom_path="test.gbc",
            mode=TrainingMode.FAST_MONITORED,
            llm_backend=LLMBackend.SMOLLM2,
            llm_interval=3,
            max_actions=1000,
            headless=True,
            capture_screens=True,
            enable_web=False  # Disable web for pure performance testing
        )
        
        return UnifiedPokemonTrainer(config)
    
    @pytest.fixture
    @patch('scripts.pokemon_trainer.PyBoy')
    @patch('trainer.trainer.PYBOY_AVAILABLE', True)
    def trainer_ultra_fast(self, mock_pyboy_class):
        """Create trainer for ultra-fast performance testing"""
        mock_pyboy_instance = Mock()
        mock_pyboy_instance.frame_count = 1000
        mock_pyboy_instance.screen.ndarray = np.random.randint(0, 255, (144, 160, 3), dtype=np.uint8)
        mock_pyboy_instance.send_input = Mock()
        mock_pyboy_instance.tick = Mock()
        mock_pyboy_class.return_value = mock_pyboy_instance
        
        config = TrainingConfig(
            rom_path="test.gbc",
            mode=TrainingMode.ULTRA_FAST,
            llm_backend=LLMBackend.NONE,
            max_actions=1000,
            headless=True,
            capture_screens=False
        )
        
        return UnifiedPokemonTrainer(config)
    
    def test_target_actions_per_second_benchmark(self, trainer_fast_monitored):
        """Test that system achieves ~2.3 actions/second in fast monitored mode"""
        trainer = trainer_fast_monitored
        
        # Mock required dependencies
        with patch.object(trainer, '_simple_screenshot_capture') as mock_capture:
            mock_capture.return_value = np.random.randint(0, 255, (144, 160, 3), dtype=np.uint8)
            
            # Warm up
            for i in range(10):
                trainer._get_rule_based_action(i)
            
            # Benchmark
            start_time = time.time()
            action_count = 100
            
            for i in range(action_count):
                action = trainer._get_rule_based_action(i)
                trainer._execute_action(action)
            
            elapsed = time.time() - start_time
            actions_per_second = action_count / elapsed
            
            # Should achieve at least 2.0 actions/second (allow some variance)
            assert actions_per_second >= 2.0, f"Achieved {actions_per_second:.2f} actions/sec, expected >= 2.0"
            
            # Log the actual performance
            print(f"✅ Fast Monitored Performance: {actions_per_second:.2f} actions/second")
    
    def test_ultra_fast_performance_benchmark(self, trainer_ultra_fast):
        """Test ultra-fast mode performance (should be 600+ actions/sec)"""
        trainer = trainer_ultra_fast
        
        # Mock minimal dependencies for maximum speed
        # Can't mock tick directly as it's read-only, mock through strategy_manager
        with patch.object(trainer.strategy_manager, 'execute_action', return_value=None):
            # Warm up
            for i in range(20):
                trainer._get_rule_based_action(i)
            
            # Benchmark
            start_time = time.time()
            action_count = 1000
            
            for i in range(action_count):
                action = trainer._get_rule_based_action(i)
                trainer._execute_action(action)
            
            elapsed = time.time() - start_time
            actions_per_second = action_count / elapsed
            
            # Should achieve at least 300 actions/second (conservative target)
            assert actions_per_second >= 300, f"Ultra-fast achieved {actions_per_second:.2f} actions/sec, expected >= 300"
            
            print(f"✅ Ultra Fast Performance: {actions_per_second:.2f} actions/second")
    
    def test_performance_consistency(self, trainer_fast_monitored):
        """Test that performance is consistent across multiple runs"""
        trainer = trainer_fast_monitored
        
        with patch.object(trainer, '_simple_screenshot_capture') as mock_capture:
            mock_capture.return_value = np.random.randint(0, 255, (144, 160, 3), dtype=np.uint8)
            
            performance_samples = []
            
            for run in range(5):
                start_time = time.time()
                
                for i in range(50):
                    action = trainer._get_rule_based_action(i + run * 50)
                    trainer._execute_action(action)
                
                elapsed = time.time() - start_time
                actions_per_second = 50 / elapsed
                performance_samples.append(actions_per_second)
            
            # Calculate statistics
            avg_performance = mean(performance_samples)
            std_dev = stdev(performance_samples) if len(performance_samples) > 1 else 0
            
            # Performance should be consistent (low std deviation)
            coefficient_of_variation = std_dev / avg_performance if avg_performance > 0 else 1
            assert coefficient_of_variation < 0.3, f"Performance too variable: CV = {coefficient_of_variation:.3f}"
            
            # Average should still meet target
            assert avg_performance >= 2.0, f"Average performance {avg_performance:.2f} below target"
            
            print(f"✅ Consistent Performance: {avg_performance:.2f} ± {std_dev:.2f} actions/second")


@pytest.mark.benchmarking
@pytest.mark.llm
class TestLLMInferenceBenchmarks:
    """Test LLM inference timing benchmarks"""
    
    @pytest.fixture
    def performance_expectations(self):
        """Performance expectations for different LLM backends"""
        return {
            LLMBackend.SMOLLM2: {
                "target_ms": 25,
                "max_ms": 50,
                "model": "smollm2:1.7b"
            },
            LLMBackend.LLAMA32_1B: {
                "target_ms": 30,
                "max_ms": 60,
                "model": "llama3.2:1b"
            },
            LLMBackend.LLAMA32_3B: {
                "target_ms": 60,
                "max_ms": 120,
                "model": "llama3.2:3b"
            }
        }
    
    @pytest.mark.parametrize("backend", [LLMBackend.SMOLLM2, LLMBackend.LLAMA32_1B, LLMBackend.LLAMA32_3B])
    @patch('trainer.llm_manager.ollama')
    @patch('trainer.trainer.PYBOY_AVAILABLE', True)
    def test_llm_inference_timing(self, mock_pyboy_class, backend, performance_expectations):
        """Test LLM inference timing for each backend"""
        mock_pyboy_instance = Mock()
        mock_pyboy_class.return_value = mock_pyboy_instance
        
        config = TrainingConfig(
            rom_path="test.gbc",
            llm_backend=backend,
            headless=True
        )
        
        trainer = UnifiedPokemonTrainer(config)
        expectations = performance_expectations[backend]
        
        with patch('trainer.llm_manager.ollama') as mock_ollama:
            # Mock response with realistic delay
            def mock_generate(*args, **kwargs):
                time.sleep(expectations["target_ms"] / 1000)  # Convert to seconds
                return {'response': '5'}
            
            mock_ollama.generate.side_effect = mock_generate
            mock_ollama.show.return_value = {'model': expectations["model"]}
            
            # Warm up
            for _ in range(3):
                trainer._get_llm_action()
            
            # Benchmark
            timings = []
            for _ in range(10):
                start_time = time.perf_counter()
                action = trainer._get_llm_action()
                elapsed = time.perf_counter() - start_time
                
                if action is not None:  # Only count successful calls
                    timings.append(elapsed * 1000)  # Convert to ms
            
            if timings:
                avg_timing = mean(timings)
                max_timing = max(timings)
                
                # Should meet performance expectations
                assert avg_timing <= expectations["max_ms"], f"{backend.value} avg: {avg_timing:.1f}ms > {expectations['max_ms']}ms"
                assert max_timing <= expectations["max_ms"] * 1.5, f"{backend.value} max: {max_timing:.1f}ms too high"
                
                print(f"✅ {backend.value} Performance: {avg_timing:.1f}ms avg, {max_timing:.1f}ms max")
    
    @patch('trainer.llm_manager.ollama')
    @patch('trainer.trainer.PYBOY_AVAILABLE', True)
    def test_llm_fallback_performance(self, mock_pyboy_class):
        """Test performance when LLM fails and falls back to rule-based"""
        mock_pyboy_instance = Mock()
        mock_pyboy_class.return_value = mock_pyboy_instance
        
        config = TrainingConfig(
            rom_path="test.gbc",
            llm_backend=LLMBackend.SMOLLM2,
            headless=True
        )
        
        trainer = UnifiedPokemonTrainer(config)
        
        with patch('trainer.llm_manager.ollama') as mock_ollama:
            # Mock LLM failure
            mock_ollama.generate.side_effect = Exception("LLM unavailable")
            mock_ollama.show.side_effect = Exception("Model not found")
            
            # Benchmark fallback performance
            start_time = time.time()
            action_count = 100
            
            for i in range(action_count):
                action = trainer._get_llm_action()
                # Should still get valid actions via fallback
                if action is not None:
                    assert 1 <= action <= 8
            
            elapsed = time.time() - start_time
            
            # Fallback should be very fast (< 30ms total for 100 calls)
            assert elapsed < 0.03, f"LLM fallback too slow: {elapsed:.4f}s for {action_count} calls"
            
            print(f"✅ LLM Fallback Performance: {elapsed*1000:.2f}ms for {action_count} calls")
    
    @patch('trainer.llm_manager.ollama')
    @patch('trainer.trainer.PYBOY_AVAILABLE', True)
    def test_adaptive_llm_interval_performance(self, mock_pyboy_class):
        """Test adaptive LLM interval performance optimization"""
        mock_pyboy_instance = Mock()
        mock_pyboy_class.return_value = mock_pyboy_instance
        
        config = TrainingConfig(
            rom_path="test.gbc",
            llm_backend=LLMBackend.SMOLLM2,
            llm_interval=10,  # Start with default interval
            debug_mode=True,
            headless=True
        )
        
        trainer = UnifiedPokemonTrainer(config)
        
        with patch('trainer.llm_manager.ollama') as mock_ollama:
            # Mock slow LLM responses initially
            def slow_generate(*args, **kwargs):
                time.sleep(0.004)  # 4ms delay (simulated slow)
                return {'response': '5'}
            
            mock_ollama.generate.side_effect = slow_generate
            mock_ollama.show.return_value = {'model': 'smollm2:1.7b'}
            
            original_interval = trainer.adaptive_llm_interval
            
            # Trigger 10 slow LLM calls to activate adaptive interval
            for i in range(10):
                trainer.stats['llm_calls'] = i + 1
                trainer._track_llm_performance(4.5)  # Report slow response times
            
            # Interval should have increased
            assert trainer.adaptive_llm_interval > original_interval
            increased_interval = trainer.adaptive_llm_interval
            
            print(f"✅ Adaptive Interval Increased: {original_interval} → {increased_interval}")
            
            # Now mock fast responses - clear the window first with many fast responses
            def fast_generate(*args, **kwargs):
                time.sleep(0.001)  # 1ms delay (simulated fast)
                return {'response': '5'}
            
            mock_ollama.generate.side_effect = fast_generate
            
            # Add enough fast calls to fill the entire sliding window (25 calls)
            # This ensures the average drops below the 1.5s threshold
            for i in range(10, 35):
                trainer.stats['llm_calls'] = i + 1
                trainer._track_llm_performance(0.8)  # Report fast response times
            
            # Interval should have decreased
            assert trainer.adaptive_llm_interval < increased_interval
            final_interval = trainer.adaptive_llm_interval
            
            print(f"✅ Adaptive Interval Decreased: {increased_interval} → {final_interval}")
            
            # But should not go below original
            assert trainer.adaptive_llm_interval >= original_interval
    
    @patch('scripts.pokemon_trainer.PyBoy')
    @patch('trainer.trainer.PYBOY_AVAILABLE', True)
    def test_llm_performance_tracking_overhead(self, mock_pyboy_class):
        """Test that LLM performance tracking has minimal overhead"""
        mock_pyboy_instance = Mock()
        mock_pyboy_class.return_value = mock_pyboy_instance
        
        config = TrainingConfig(
            rom_path="test.gbc",
            llm_backend=LLMBackend.SMOLLM2,
            headless=True
        )
        
        trainer = UnifiedPokemonTrainer(config)
        
        # Test performance tracking overhead
        start_time = time.perf_counter()
        
        for i in range(1000):
            trainer.stats['llm_calls'] = i + 1
            trainer._track_llm_performance(1.0 + (i % 10) * 0.1)  # Vary response times
        
        elapsed = time.perf_counter() - start_time
        
        # Tracking 1000 calls should be very fast (<5ms)
        assert elapsed < 0.005, f"Performance tracking too slow: {elapsed:.4f}s for 1000 calls"
        
        # Verify data integrity
        assert len(trainer.llm_response_times) == 20  # Should keep last 20
        assert trainer.stats['llm_total_time'] > 0
        assert trainer.stats['llm_avg_time'] > 0
        
        print(f"✅ Performance Tracking Overhead: {elapsed*1000:.2f}ms for 1000 calls")


@pytest.mark.benchmarking
@pytest.mark.performance
class TestMemoryPerformanceBenchmarks:
    """Test memory usage and performance characteristics"""
    
    @pytest.fixture
    @patch('scripts.pokemon_trainer.PyBoy')
    @patch('trainer.trainer.PYBOY_AVAILABLE', True)
    def trainer(self, mock_pyboy_class):
        mock_pyboy_instance = Mock()
        mock_pyboy_instance.frame_count = 1000
        mock_pyboy_instance.screen.ndarray = np.random.randint(0, 255, (144, 160, 3), dtype=np.uint8)
        mock_pyboy_class.return_value = mock_pyboy_instance
        
        config = TrainingConfig(
            rom_path="test.gbc",
            mode=TrainingMode.FAST_MONITORED,
            max_actions=5000,
            headless=True,
            capture_screens=True
        )
        
        return UnifiedPokemonTrainer(config)
    
    def test_memory_usage_baseline(self, trainer):
        """Test baseline memory usage of trainer system"""
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Run training simulation
        with patch.object(trainer, '_simple_screenshot_capture') as mock_capture:
            mock_capture.return_value = np.random.randint(0, 255, (144, 160, 3), dtype=np.uint8)
            
            for i in range(1000):
                trainer._get_rule_based_action(i)
                
                # Simulate some screen captures
                if i % 10 == 0:
                    trainer._capture_and_queue_screen()
        
        gc.collect()
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (under 100MB)
        assert memory_increase < 100, f"Memory usage increased by {memory_increase:.1f}MB"
        
        print(f"✅ Memory Usage: +{memory_increase:.1f}MB after 1000 actions")
    
    def test_screen_capture_memory_efficiency(self, trainer):
        """Test screen capture memory efficiency"""
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Simulate heavy screen capture usage
        test_screen = np.random.randint(0, 255, (144, 160, 3), dtype=np.uint8)
        
        with patch.object(trainer, '_simple_screenshot_capture', return_value=test_screen):
            for i in range(500):
                trainer._capture_and_queue_screen()
                
                # Queue should not grow unbounded
                assert trainer.screen_queue.qsize() <= 30, "Screen queue should be bounded"
        
        gc.collect()
        final_memory = process.memory_info().rss
        memory_increase = (final_memory - initial_memory) / 1024 / 1024  # MB
        
        # Screen capture should not use excessive memory
        assert memory_increase < 50, f"Screen capture used {memory_increase:.1f}MB"
        
        print(f"✅ Screen Capture Memory: +{memory_increase:.1f}MB for 500 captures")
    
    def test_long_running_memory_stability(self, trainer):
        """Test memory stability over extended operation"""
        process = psutil.Process(os.getpid())
        memory_samples = []
        
        test_screen = np.random.randint(0, 255, (144, 160, 3), dtype=np.uint8)
        
        with patch.object(trainer, '_simple_screenshot_capture', return_value=test_screen):
            for i in range(2000):
                trainer._get_rule_based_action(i)
                
                # Sample memory usage periodically
                if i % 200 == 0:
                    current_memory = process.memory_info().rss / 1024 / 1024
                    memory_samples.append(current_memory)
                    
                    # Force GC occasionally
                    if i % 400 == 0:
                        gc.collect()
        
        # Memory usage should be stable (not continuously growing)
        if len(memory_samples) > 2:
            # Check that final memory isn't much higher than middle samples
            mid_point = len(memory_samples) // 2
            mid_memory = mean(memory_samples[mid_point:mid_point+2])
            final_memory = memory_samples[-1]
            
            memory_growth = final_memory - mid_memory
            assert memory_growth < 20, f"Memory grew by {memory_growth:.1f}MB during operation"
            
            print(f"✅ Long-term Memory Stability: {memory_growth:+.1f}MB growth")


@pytest.mark.benchmarking
@pytest.mark.integration
class TestRealWorldPerformanceBenchmarks:
    """Test performance in real-world usage scenarios"""
    
    @pytest.fixture
    @patch('trainer.llm_manager.ollama')
    @patch('trainer.trainer.PYBOY_AVAILABLE', True)
    def training_scenarios(self, mock_pyboy_class):
        """Create trainers for different real-world scenarios"""
        mock_pyboy_instance = Mock()
        mock_pyboy_instance.frame_count = 1000
        mock_pyboy_instance.screen.ndarray = np.random.randint(0, 255, (144, 160, 3), dtype=np.uint8)
        mock_pyboy_instance.send_input = Mock()
        mock_pyboy_instance.tick = Mock()
        mock_pyboy_class.return_value = mock_pyboy_instance
        
        scenarios = {
            "content_creation": TrainingConfig(
                rom_path="test.gbc",
                mode=TrainingMode.FAST_MONITORED,
                llm_backend=LLMBackend.SMOLLM2,
                llm_interval=3,
                enable_web=True,
                capture_screens=True,
                headless=False
            ),
            "research_training": TrainingConfig(
                rom_path="test.gbc",
                mode=TrainingMode.CURRICULUM,
                llm_backend=LLMBackend.LLAMA32_3B,
                llm_interval=5,
                enable_web=True,
                capture_screens=True,
                debug_mode=True
            ),
            "speed_testing": TrainingConfig(
                rom_path="test.gbc",
                mode=TrainingMode.ULTRA_FAST,
                llm_backend=LLMBackend.NONE,
                capture_screens=False,
                headless=True
            )
        }
        
        return {name: UnifiedPokemonTrainer(config) for name, config in scenarios.items()}
    
    def test_content_creation_performance(self, training_scenarios):
        """Test performance for content creation scenario"""
        trainer = training_scenarios["content_creation"]
        
        with patch('trainer.llm_manager.ollama') as mock_ollama:
            mock_ollama.generate.return_value = {'response': '5'}
            mock_ollama.show.return_value = {'model': 'smollm2:1.7b'}
            
            with patch.object(trainer, '_simple_screenshot_capture') as mock_capture:
                mock_capture.return_value = np.random.randint(0, 255, (144, 160, 3), dtype=np.uint8)
                
                start_time = time.time()
                actions_executed = 0
                llm_calls = 0
                
                for step in range(100):
                    if step % trainer.config.llm_interval == 0:
                        action = trainer._get_llm_action()
                        llm_calls += 1
                    else:
                        action = trainer._get_rule_based_action(step)
                    
                    if action:
                        trainer._execute_action(action)
                        actions_executed += 1
                    
                    # Simulate screen capture
                    trainer._capture_and_queue_screen()
                
                elapsed = time.time() - start_time
                actions_per_second = actions_executed / elapsed
                
                # Should achieve reasonable performance for content creation
                assert actions_per_second >= 1.5, f"Content creation: {actions_per_second:.2f} actions/sec"
                assert llm_calls > 0, "Should make LLM calls"
                
                print(f"✅ Content Creation: {actions_per_second:.2f} actions/sec, {llm_calls} LLM calls")
    
    def test_research_training_performance(self, training_scenarios):
        """Test performance for research training scenario"""
        trainer = training_scenarios["research_training"]
        
        with patch('trainer.llm_manager.ollama') as mock_ollama:
            # Simulate slower LLM (Llama3.2-3B)
            def slow_generate(*args, **kwargs):
                time.sleep(0.06)  # 60ms simulated inference
                return {'response': '5'}
            
            mock_ollama.generate.side_effect = slow_generate
            mock_ollama.show.return_value = {'model': 'llama3.2:3b'}
            
            with patch.object(trainer, '_simple_screenshot_capture') as mock_capture:
                mock_capture.return_value = np.random.randint(0, 255, (144, 160, 3), dtype=np.uint8)
                
                start_time = time.time()
                actions_executed = 0
                
                for step in range(50):  # Fewer steps due to slower LLM
                    if step % trainer.config.llm_interval == 0:
                        action = trainer._get_llm_action()
                    else:
                        action = trainer._get_rule_based_action(step)
                    
                    if action:
                        trainer._execute_action(action)
                        actions_executed += 1
                
                elapsed = time.time() - start_time
                actions_per_second = actions_executed / elapsed
                
                # Should achieve reasonable performance despite slower LLM
                assert actions_per_second >= 0.8, f"Research training: {actions_per_second:.2f} actions/sec"
                
                print(f"✅ Research Training: {actions_per_second:.2f} actions/sec (with slower LLM)")
    
    def test_speed_testing_performance(self, training_scenarios):
        """Test performance for speed testing scenario"""
        trainer = training_scenarios["speed_testing"]
        
        start_time = time.time()
        actions_executed = 0
        
        for step in range(2000):
            action = trainer._get_rule_based_action(step)
            if action:
                trainer._execute_action(action)
                actions_executed += 1
        
        elapsed = time.time() - start_time
        actions_per_second = actions_executed / elapsed
        
        # Should achieve very high performance in speed testing mode
        assert actions_per_second >= 500, f"Speed testing: {actions_per_second:.2f} actions/sec"
        
        print(f"✅ Speed Testing: {actions_per_second:.2f} actions/sec")
    
    def test_concurrent_performance(self, training_scenarios):
        """Test performance with concurrent operations (simulating real usage)"""
        trainer = training_scenarios["content_creation"]
        
        results = {"actions": 0, "captures": 0, "errors": 0}
        
        def action_thread():
            """Simulate action execution thread"""
            try:
                with patch.object(trainer, '_simple_screenshot_capture') as mock_capture:
                    mock_capture.return_value = np.random.randint(0, 255, (144, 160, 3), dtype=np.uint8)
                    
                    for i in range(100):
                        action = trainer._get_rule_based_action(i)
                        if action:
                            trainer._execute_action(action)
                            results["actions"] += 1
            except Exception:
                results["errors"] += 1
        
        def capture_thread():
            """Simulate screen capture thread"""
            try:
                with patch.object(trainer, '_simple_screenshot_capture') as mock_capture:
                    mock_capture.return_value = np.random.randint(0, 255, (144, 160, 3), dtype=np.uint8)
                    
                    for i in range(50):
                        trainer._capture_and_queue_screen()
                        results["captures"] += 1
                        time.sleep(0.001)  # Simulate capture interval
            except Exception:
                results["errors"] += 1
        
        # Run concurrent operations
        action_t = threading.Thread(target=action_thread)
        capture_t = threading.Thread(target=capture_thread)
        
        start_time = time.time()
        action_t.start()
        capture_t.start()
        
        action_t.join()
        capture_t.join()
        elapsed = time.time() - start_time
        
        # Should complete without errors and maintain performance
        assert results["errors"] == 0, f"Concurrent operations had {results['errors']} errors"
        assert results["actions"] > 90, "Should complete most actions"
        assert results["captures"] > 45, "Should complete most captures"
        
        actions_per_second = results["actions"] / elapsed
        assert actions_per_second >= 1.0, f"Concurrent performance: {actions_per_second:.2f} actions/sec"
        
        print(f"✅ Concurrent Performance: {actions_per_second:.2f} actions/sec, {results['captures']} captures")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-m", "benchmarking"])
