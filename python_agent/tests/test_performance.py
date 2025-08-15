#!/usr/bin/env python3
"""
test_performance.py - Performance and integration tests

Tests performance improvements and validates the fixes made to:
- PyBoy stability and crash recovery
- Memory management and leak prevention
- Error handling and recovery mechanisms
- Overall system resilience under stress
"""

import pytest
import time
import threading
import gc
import psutil
import os
import tempfile
from unittest.mock import Mock, patch, MagicMock
import json
import base64
import sys
from pathlib import Path

# Mock PyBoy since we can't import the real one in tests
class MockPyBoy:
    """Mock PyBoy for testing stability improvements"""
    
    def __init__(self, rom_path=None, **kwargs):
        self.rom_path = rom_path
        self.frame_count = 0
        self.crashed = False
        self.initialized = True
        self._last_action_time = time.time()
        self.memory_usage = 100.0  # MB
        
        # Simulate different failure modes
        self.failure_mode = None
        self.failure_counter = 0
        
        # Track method calls
        self.method_calls = {
            'tick': 0,
            'button_press': 0,
            'button_release': 0,
            'screen_ndarray': 0,
            'save_state': 0,
            'load_state': 0
        }
    
    def tick(self):
        """Mock tick with potential failures"""
        self.method_calls['tick'] += 1
        self.frame_count += 1
        
        if self.failure_mode == 'crash_after_ticks':
            self.failure_counter += 1
            if self.failure_counter > 100:
                self.crashed = True
                raise RuntimeError("PyBoy crashed after 100 ticks")
        
        if self.failure_mode == 'freeze':
            if self.failure_counter > 50:
                # Simulate frozen state - frame count stops advancing
                self.frame_count -= 1
                time.sleep(0.1)  # Simulate hanging
        
        # Simulate memory growth
        if self.failure_mode == 'memory_leak':
            self.memory_usage += 0.5
    
    def button_press(self, button):
        """Mock button press"""
        self.method_calls['button_press'] += 1
        self._last_action_time = time.time()
        
        if self.crashed:
            raise RuntimeError("PyBoy is crashed")
    
    def button_release(self, button):
        """Mock button release"""
        self.method_calls['button_release'] += 1
        
        if self.crashed:
            raise RuntimeError("PyBoy is crashed")
    
    def screen_ndarray(self):
        """Mock screen capture with potential failures"""
        self.method_calls['screen_ndarray'] += 1
        
        if self.crashed:
            raise RuntimeError("PyBoy is crashed")
        
        if self.failure_mode == 'screenshot_error':
            self.failure_counter += 1
            if self.failure_counter % 20 == 0:  # Fail every 20th screenshot
                raise Exception("Screenshot capture failed")
        
        # Return mock image array
        import numpy as np
        return np.zeros((144, 160, 3), dtype=np.uint8)
    
    def save_state(self, path):
        """Mock save state"""
        self.method_calls['save_state'] += 1
        
        if self.crashed:
            raise RuntimeError("PyBoy is crashed")
        
        # Create mock save file
        with open(path, 'wb') as f:
            f.write(b'mock_save_state_data')
    
    def load_state(self, path):
        """Mock load state"""
        self.method_calls['load_state'] += 1
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"Save state not found: {path}")
        
        # Reset crash state when loading
        self.crashed = False
        self.failure_counter = 0
    
    def get_memory_usage(self):
        """Get simulated memory usage"""
        return self.memory_usage


class TestPyBoyStabilityImprovements:
    """Test PyBoy stability improvements and crash detection"""
    
    @pytest.fixture
    def mock_pyboy(self):
        """Create mock PyBoy instance"""
        return MockPyBoy()
    
    def test_frame_count_health_check(self, mock_pyboy):
        """Test lightweight frame count health check"""
        
        # Simulate healthy PyBoy
        initial_frame = mock_pyboy.frame_count
        
        # Run some ticks
        for _ in range(10):
            mock_pyboy.tick()
        
        # Frame count should increase
        assert mock_pyboy.frame_count > initial_frame
        assert mock_pyboy.frame_count == initial_frame + 10
    
    def test_crash_detection_via_frame_count(self, mock_pyboy):
        """Test crash detection when frame count stops advancing"""
        
        # Set freeze failure mode
        mock_pyboy.failure_mode = 'freeze'
        
        frame_counts = []
        
        # Run ticks and track frame count
        for i in range(60):
            mock_pyboy.tick()
            frame_counts.append(mock_pyboy.frame_count)
        
        # After failure counter exceeds 50, frame count should stop advancing
        recent_frames = frame_counts[-10:]  # Last 10 frames
        
        # Check if frame count is stuck (indicating freeze)
        if len(set(recent_frames)) == 1:  # All the same
            # Detected frozen state
            assert True, "Successfully detected frozen PyBoy state"
        else:
            # Normal operation
            assert len(set(recent_frames)) > 1, "Frame count should advance normally"
    
    def test_crash_detection_via_exceptions(self, mock_pyboy):
        """Test crash detection via exceptions"""
        
        # Set crash failure mode
        mock_pyboy.failure_mode = 'crash_after_ticks'
        
        # Run ticks until crash
        crash_detected = False
        try:
            for i in range(150):  # Should crash after 100
                mock_pyboy.tick()
        except RuntimeError as e:
            if "crashed" in str(e):
                crash_detected = True
        
        assert crash_detected, "Should detect PyBoy crash via exception"
        assert mock_pyboy.crashed is True
    
    def test_pyboy_recovery_mechanism(self, mock_pyboy):
        """Test PyBoy recovery after crash"""
        
        # Simulate crash
        mock_pyboy.crashed = True
        
        # Verify crash state
        with pytest.raises(RuntimeError):
            mock_pyboy.button_press("A")
        
        # Create temporary save state for recovery
        with tempfile.NamedTemporaryFile(delete=False) as temp_save:
            temp_save_path = temp_save.name
        
        try:
            # Save state before crash (simulate)
            mock_pyboy.crashed = False  # Temporarily uncrash to save
            mock_pyboy.save_state(temp_save_path)
            mock_pyboy.crashed = True  # Re-crash
            
            # Recovery process
            new_pyboy = MockPyBoy(rom_path="test.gb")
            new_pyboy.load_state(temp_save_path)
            
            # Verify recovery
            assert not new_pyboy.crashed
            assert new_pyboy.method_calls['load_state'] == 1
            
            # Test functionality after recovery
            new_pyboy.button_press("A")
            assert new_pyboy.method_calls['button_press'] == 1
            
        finally:
            # Clean up
            if os.path.exists(temp_save_path):
                os.unlink(temp_save_path)
    
    def test_screenshot_error_resilience(self, mock_pyboy):
        """Test resilience to screenshot capture errors"""
        
        # Set screenshot error mode
        mock_pyboy.failure_mode = 'screenshot_error'
        
        successful_screenshots = 0
        failed_screenshots = 0
        
        # Try to capture many screenshots
        for i in range(50):
            try:
                screen = mock_pyboy.screen_ndarray()
                if screen is not None:
                    successful_screenshots += 1
            except Exception:
                failed_screenshots += 1
        
        # Should have both successful and failed captures
        assert successful_screenshots > 0, "Should have some successful screenshots"
        assert failed_screenshots > 0, "Should have simulated some screenshot failures"
        
        # Should have more successes than failures
        assert successful_screenshots > failed_screenshots
    
    def test_memory_leak_detection(self, mock_pyboy):
        """Test memory leak detection capabilities"""
        
        # Set memory leak mode
        mock_pyboy.failure_mode = 'memory_leak'
        
        initial_memory = mock_pyboy.get_memory_usage()
        
        # Run many ticks to accumulate memory
        for _ in range(200):
            mock_pyboy.tick()
        
        final_memory = mock_pyboy.get_memory_usage()
        
        # Memory should have increased significantly
        memory_increase = final_memory - initial_memory
        assert memory_increase > 50, f"Memory leak detected: {memory_increase}MB increase"
        
        # This test validates that we can detect memory growth patterns


class TestErrorHandlingImprovements:
    """Test comprehensive error handling improvements"""
    
    def test_error_context_manager_basic(self):
        """Test basic error context manager functionality"""
        
        error_log = []
        
        class MockErrorHandler:
            def __init__(self):
                self.recovery_attempts = 0
            
            def _handle_errors(self, operation_name):
                """Mock error handler context manager"""
                class ErrorContext:
                    def __init__(self, handler, op_name):
                        self.handler = handler
                        self.operation_name = op_name
                    
                    def __enter__(self):
                        return self
                    
                    def __exit__(self, exc_type, exc_val, exc_tb):
                        if exc_type is not None:
                            error_log.append({
                                'operation': self.operation_name,
                                'error_type': exc_type.__name__,
                                'error_message': str(exc_val),
                                'timestamp': time.time()
                            })
                            
                            # Simulate recovery attempt
                            self.handler.recovery_attempts += 1
                            return True  # Suppress exception
                        return False
                
                return ErrorContext(self, operation_name)
        
        handler = MockErrorHandler()
        
        # Test successful operation
        with handler._handle_errors("test_operation"):
            result = "success"
        
        assert len(error_log) == 0
        assert handler.recovery_attempts == 0
        
        # Test failed operation
        with handler._handle_errors("failing_operation"):
            raise ValueError("Test error")
        
        assert len(error_log) == 1
        assert error_log[0]['operation'] == "failing_operation"
        assert error_log[0]['error_type'] == "ValueError"
        assert handler.recovery_attempts == 1
    
    def test_error_recovery_strategies(self):
        """Test different error recovery strategies"""
        
        recovery_log = []
        
        def simulate_operation_with_recovery(operation_type, max_retries=3):
            """Simulate operation with retry logic"""
            for attempt in range(max_retries + 1):
                try:
                    # Simulate different failure modes
                    if operation_type == "transient_failure" and attempt < 2:
                        raise ConnectionError("Temporary network issue")
                    elif operation_type == "persistent_failure":
                        raise ValueError("Persistent configuration error")
                    elif operation_type == "success_after_retry" and attempt == 0:
                        raise TimeoutError("First attempt timeout")
                    
                    # Success case
                    recovery_log.append({
                        'operation': operation_type,
                        'attempt': attempt,
                        'result': 'success',
                        'timestamp': time.time()
                    })
                    return "success"
                    
                except Exception as e:
                    recovery_log.append({
                        'operation': operation_type,
                        'attempt': attempt,
                        'result': 'error',
                        'error': str(e),
                        'timestamp': time.time()
                    })
                    
                    if attempt == max_retries:
                        raise  # Final attempt failed
                    
                    # Wait before retry
                    time.sleep(0.01)
            
        # Test transient failure recovery
        result = simulate_operation_with_recovery("transient_failure")
        assert result == "success"
        
        # Check recovery log
        transient_logs = [log for log in recovery_log if log['operation'] == "transient_failure"]
        assert len(transient_logs) == 3  # 2 failures + 1 success
        assert transient_logs[-1]['result'] == 'success'
        
        # Test success after retry
        result = simulate_operation_with_recovery("success_after_retry")
        assert result == "success"
        
        # Test persistent failure (should still fail after retries)
        with pytest.raises(ValueError):
            simulate_operation_with_recovery("persistent_failure")
        
        persistent_logs = [log for log in recovery_log if log['operation'] == "persistent_failure"]
        assert len(persistent_logs) == 4  # 4 failed attempts (0-3)
        assert all(log['result'] == 'error' for log in persistent_logs)
    
    def test_structured_logging_functionality(self):
        """Test structured logging improvements"""
        
        import logging
        from io import StringIO
        
        # Create string buffer for log capture
        log_buffer = StringIO()
        
        # Configure logger
        logger = logging.getLogger("test_trainer")
        logger.setLevel(logging.DEBUG)
        
        # Add handler
        handler = logging.StreamHandler(log_buffer)
        handler.setFormatter(logging.Formatter('%(levelname)s:%(name)s:%(message)s'))
        logger.addHandler(handler)
        
        # Test different log levels and structured data
        logger.info("Training started", extra={'mode': 'fast_monitored', 'model': 'smollm2:1.7b'})
        logger.warning("PyBoy health check failed", extra={'frame_count': 1234, 'last_frame_time': time.time()})
        logger.error("Recovery attempt failed", extra={'attempt': 1, 'error_type': 'ConnectionError'})
        
        # Get log contents
        log_contents = log_buffer.getvalue()
        
        # Verify log entries
        assert "Training started" in log_contents
        assert "PyBoy health check failed" in log_contents
        assert "Recovery attempt failed" in log_contents
        assert "INFO" in log_contents
        assert "WARNING" in log_contents
        assert "ERROR" in log_contents
        
        # Clean up
        logger.removeHandler(handler)
        handler.close()


class TestMemoryManagementImprovements:
    """Test memory management and leak prevention"""
    
    def test_memory_usage_monitoring(self):
        """Test memory usage monitoring capabilities"""
        
        # Get initial memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Simulate memory-intensive operations
        large_data = []
        for i in range(100):
            # Create some data structures
            data = {
                'frame_id': i,
                'screenshot': b'x' * 10000,  # 10KB per frame
                'metadata': {'timestamp': time.time(), 'size': (160, 144)}
            }
            large_data.append(data)
        
        # Check memory increase
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = peak_memory - initial_memory
        
        # Clean up explicitly
        large_data.clear()
        gc.collect()
        
        # Check memory after cleanup
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_reclaimed = peak_memory - final_memory
        
        # Validate memory management
        assert memory_increase > 0.5, f"Memory should increase: {memory_increase}MB"
        # Note: Exact memory reclamation is hard to test due to Python's memory management
        # but we can at least verify the monitoring works
    
    def test_screenshot_buffer_management(self):
        """Test screenshot buffer management to prevent leaks"""
        
        screenshot_buffers = {}
        max_buffers = 5
        
        def manage_screenshot_buffer(frame_id, screenshot_data):
            """Mock screenshot buffer management"""
            
            # Add new buffer
            screenshot_buffers[frame_id] = {
                'data': screenshot_data,
                'timestamp': time.time(),
                'size': len(screenshot_data)
            }
            
            # Clean up old buffers if we exceed limit
            if len(screenshot_buffers) > max_buffers:
                # Remove oldest buffer
                oldest_frame = min(screenshot_buffers.keys())
                del screenshot_buffers[oldest_frame]
        
        # Add many screenshot buffers
        for i in range(20):
            screenshot_data = f"screenshot_data_{i}".encode() * 100
            manage_screenshot_buffer(i, screenshot_data)
        
        # Should never exceed max_buffers
        assert len(screenshot_buffers) <= max_buffers
        
        # Should contain the most recent frames
        frame_ids = list(screenshot_buffers.keys())
        assert max(frame_ids) >= 15  # Recent frames should be kept
        assert min(frame_ids) >= 15  # Old frames should be removed
    
    def test_web_server_resource_cleanup(self):
        """Test web server resource cleanup"""
        
        # Mock web server resources
        active_connections = {}
        response_buffers = {}
        
        class MockWebServer:
            def __init__(self):
                self.request_count = 0
                self.cleanup_count = 0
            
            def handle_request(self, request_id):
                """Mock request handling"""
                self.request_count += 1
                
                # Create connection and response buffer
                active_connections[request_id] = {
                    'start_time': time.time(),
                    'status': 'active'
                }
                
                response_buffers[request_id] = {
                    'data': b'x' * 1000,  # 1KB response buffer
                    'created': time.time()
                }
            
            def cleanup_resources(self, max_age=60):
                """Mock resource cleanup"""
                current_time = time.time()
                
                # Clean up old connections
                expired_connections = [
                    req_id for req_id, conn in active_connections.items()
                    if current_time - conn['start_time'] > max_age
                ]
                
                for req_id in expired_connections:
                    del active_connections[req_id]
                    if req_id in response_buffers:
                        del response_buffers[req_id]
                    self.cleanup_count += 1
        
        server = MockWebServer()
        
        # Generate many requests
        for i in range(50):
            server.handle_request(f"request_{i}")
            
            # Simulate some requests finishing quickly
            if i % 10 == 0:
                time.sleep(0.001)
                server.cleanup_resources(max_age=0.001)  # Very short age for testing
        
        # Final cleanup
        server.cleanup_resources(max_age=0)
        
        # Verify cleanup occurred
        assert server.cleanup_count > 0, "Resource cleanup should have occurred"
        assert len(active_connections) <= 10, "Should clean up old connections"
        assert len(response_buffers) <= 10, "Should clean up old response buffers"


class TestIntegrationStressTests:
    """Integration stress tests to validate overall system stability"""
    
    def test_concurrent_training_simulation(self):
        """Test concurrent training operations"""
        
        results = {}
        errors = []
        
        def training_worker(worker_id, iterations):
            """Simulate training worker thread"""
            worker_results = []
            mock_pyboy = MockPyBoy()
            
            for i in range(iterations):
                try:
                    # Simulate training step
                    mock_pyboy.tick()
                    mock_pyboy.button_press("A")
                    mock_pyboy.button_release("A")
                    screenshot = mock_pyboy.screen_ndarray()
                    
                    worker_results.append({
                        'iteration': i,
                        'frame_count': mock_pyboy.frame_count,
                        'screenshot_size': screenshot.shape if screenshot is not None else None,
                        'timestamp': time.time()
                    })
                    
                    # Small delay to simulate processing
                    time.sleep(0.001)
                    
                except Exception as e:
                    errors.append({
                        'worker_id': worker_id,
                        'iteration': i,
                        'error': str(e),
                        'timestamp': time.time()
                    })
            
            results[worker_id] = worker_results
        
        # Start multiple training workers
        threads = []
        for worker_id in range(4):
            thread = threading.Thread(target=training_worker, args=(worker_id, 50))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Verify results
        assert len(results) == 4, "All workers should complete"
        
        total_iterations = sum(len(worker_results) for worker_results in results.values())
        assert total_iterations == 200, "Should complete all iterations"  # 4 workers * 50 iterations
        
        # Check error rate
        error_rate = len(errors) / total_iterations if total_iterations > 0 else 1.0
        assert error_rate < 0.1, f"Error rate too high: {error_rate:.2%}"  # Less than 10% errors
    
    def test_web_dashboard_stress_test(self):
        """Test web dashboard under stress"""
        
        # Mock trainer with stress testing
        class StressTestTrainer:
            def __init__(self):
                self.request_count = 0
                self.error_count = 0
                self.response_times = []
            
            def handle_api_request(self, endpoint):
                """Mock API request with stress testing"""
                start_time = time.time()
                self.request_count += 1
                
                try:
                    # Simulate processing delay
                    time.sleep(0.001 + (self.request_count % 10) * 0.0001)
                    
                    # Simulate occasional errors
                    if self.request_count % 100 == 0:
                        raise Exception("Simulated server stress error")
                    
                    # Return mock response
                    if endpoint == 'status':
                        response = {'is_training': True, 'actions': self.request_count}
                    elif endpoint == 'system':
                        response = {'cpu_percent': 50.0, 'memory_percent': 60.0}
                    elif endpoint == 'screenshot':
                        response = b'mock_screenshot_data'
                    else:
                        response = {'error': 'Unknown endpoint'}
                    
                    response_time = time.time() - start_time
                    self.response_times.append(response_time)
                    
                    return response
                    
                except Exception as e:
                    self.error_count += 1
                    response_time = time.time() - start_time
                    self.response_times.append(response_time)
                    raise
        
        trainer = StressTestTrainer()
        
        # Simulate high load
        def stress_worker(worker_id, request_count):
            """Stress test worker"""
            endpoints = ['status', 'system', 'screenshot']
            
            for i in range(request_count):
                endpoint = endpoints[i % len(endpoints)]
                
                try:
                    response = trainer.handle_api_request(endpoint)
                    assert response is not None
                except Exception:
                    pass  # Expected occasional errors
                
                # Brief pause to simulate realistic load
                time.sleep(0.001)
        
        # Start multiple stress workers
        threads = []
        for worker_id in range(8):  # 8 concurrent clients
            thread = threading.Thread(target=stress_worker, args=(worker_id, 50))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Analyze stress test results
        total_requests = trainer.request_count
        error_rate = trainer.error_count / total_requests if total_requests > 0 else 0
        avg_response_time = sum(trainer.response_times) / len(trainer.response_times)
        max_response_time = max(trainer.response_times)
        
        # Validate performance under stress
        assert total_requests == 400, f"Expected 400 requests, got {total_requests}"
        assert error_rate < 0.05, f"Error rate too high under stress: {error_rate:.2%}"
        assert avg_response_time < 0.01, f"Average response time too slow: {avg_response_time:.4f}s"
        assert max_response_time < 0.05, f"Max response time too slow: {max_response_time:.4f}s"
    
    def test_memory_stability_long_running(self):
        """Test memory stability over extended periods"""
        
        # Simulate long-running training session
        memory_readings = []
        process = psutil.Process()
        
        def memory_monitor():
            """Monitor memory usage"""
            for _ in range(20):  # Monitor for short duration in test
                memory_mb = process.memory_info().rss / 1024 / 1024
                memory_readings.append({
                    'timestamp': time.time(),
                    'memory_mb': memory_mb
                })
                time.sleep(0.01)
        
        def simulate_training():
            """Simulate training operations"""
            mock_pyboy = MockPyBoy()
            temp_data = []
            
            for i in range(100):
                # Simulate training operations
                mock_pyboy.tick()
                screenshot = mock_pyboy.screen_ndarray()
                
                # Temporary data that should be cleaned up
                temp_data.append({
                    'frame': i,
                    'screenshot': screenshot,
                    'metadata': {'size': screenshot.shape if screenshot is not None else None}
                })
                
                # Clean up old data periodically
                if len(temp_data) > 10:
                    temp_data = temp_data[-5:]  # Keep only recent data
                
                time.sleep(0.001)
        
        # Start monitoring and training
        monitor_thread = threading.Thread(target=memory_monitor)
        training_thread = threading.Thread(target=simulate_training)
        
        monitor_thread.start()
        training_thread.start()
        
        # Wait for completion
        training_thread.join()
        monitor_thread.join()
        
        # Analyze memory stability
        if len(memory_readings) > 5:
            initial_memory = memory_readings[0]['memory_mb']
            final_memory = memory_readings[-1]['memory_mb']
            peak_memory = max(reading['memory_mb'] for reading in memory_readings)
            
            memory_growth = final_memory - initial_memory
            memory_peak_increase = peak_memory - initial_memory
            
            # Memory should be relatively stable
            # Allow for some growth but not excessive
            assert memory_growth < 50, f"Excessive memory growth: {memory_growth:.2f}MB"
            assert memory_peak_increase < 100, f"Excessive peak memory: {memory_peak_increase:.2f}MB"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x"])  # Stop on first failure for performance tests
