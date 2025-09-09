"""
Tests for core/error_handler.py

Tests comprehensive error handling system including severity levels,
recovery strategies, component health tracking, and memory monitoring.
"""

import pytest
import threading
import time
import gc
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from core.error_handler import (
    ErrorSeverity, RecoveryStrategy, ErrorEvent, MemoryMonitor,
    SafeOperation, error_boundary, ErrorHandler
)


class TestErrorSeverity:
    """Test ErrorSeverity enum"""
    
    def test_error_severity_values(self):
        """Test that error severity values are correct"""
        assert ErrorSeverity.LOW.value == "low"
        assert ErrorSeverity.MEDIUM.value == "medium"
        assert ErrorSeverity.HIGH.value == "high"
        assert ErrorSeverity.CRITICAL.value == "critical"


class TestRecoveryStrategy:
    """Test RecoveryStrategy enum"""
    
    def test_recovery_strategy_values(self):
        """Test that recovery strategy values are correct"""
        assert RecoveryStrategy.NONE.value == "none"
        assert RecoveryStrategy.RETRY.value == "retry"
        assert RecoveryStrategy.RESTART.value == "restart"
        assert RecoveryStrategy.RESET.value == "reset"
        assert RecoveryStrategy.FALLBACK.value == "fallback"


class TestErrorEvent:
    """Test ErrorEvent dataclass"""
    
    def test_error_event_creation(self):
        """Test ErrorEvent creation with all fields"""
        event = ErrorEvent(
            timestamp=1234567890.0,
            component="test_component",
            error_type="ValueError",
            message="Test error",
            severity=ErrorSeverity.HIGH,
            traceback="traceback info",
            recovery_strategy=RecoveryStrategy.RETRY
        )
        
        assert event.timestamp == 1234567890.0
        assert event.component == "test_component"
        assert event.error_type == "ValueError"
        assert event.message == "Test error"
        assert event.severity == ErrorSeverity.HIGH
        assert event.traceback == "traceback info"
        assert event.recovery_strategy == RecoveryStrategy.RETRY


class TestMemoryMonitor:
    """Test MemoryMonitor class"""
    
    def test_memory_monitor_init(self):
        """Test MemoryMonitor initialization"""
        monitor = MemoryMonitor(threshold_mb=256.0, check_interval=0.5)
        
        assert monitor.threshold_mb == 256.0
        assert monitor.check_interval == 0.5
        assert not monitor.monitoring_active
        assert monitor.monitoring_thread is None
    
    @patch('psutil.Process')
    def test_get_memory_info(self, mock_process):
        """Test memory info retrieval"""
        # Mock memory info
        mock_memory_info = Mock()
        mock_memory_info.rss = 1024 * 1024 * 100  # 100MB
        mock_memory_info.vms = 1024 * 1024 * 200  # 200MB
        
        mock_proc_instance = Mock()
        mock_proc_instance.memory_info.return_value = mock_memory_info
        mock_proc_instance.memory_percent.return_value = 15.5
        mock_process.return_value = mock_proc_instance
        
        monitor = MemoryMonitor()
        info = monitor.get_memory_info()
        
        assert info['rss_mb'] == 100.0
        assert info['vms_mb'] == 200.0
        assert info['percent'] == 15.5
    
    @patch('gc.get_objects')
    @patch('gc.collect')
    def test_trigger_garbage_collection(self, mock_collect, mock_get_objects):
        """Test garbage collection trigger"""
        # Mock object counts
        mock_get_objects.side_effect = [list(range(1000)), list(range(800))]  # Before and after
        mock_collect.return_value = None
        
        monitor = MemoryMonitor()
        stats = monitor.trigger_garbage_collection()
        
        assert stats['objects_before'] == 1000
        assert stats['objects_after'] == 800
        assert stats['objects_collected'] == 200
        mock_collect.assert_called_once()
    
    def test_start_stop_monitoring(self):
        """Test starting and stopping memory monitoring"""
        monitor = MemoryMonitor(check_interval=0.01)  # Very short interval for testing
        
        # Start monitoring
        monitor.start_monitoring()
        assert monitor.monitoring_active
        assert monitor.monitoring_thread is not None
        assert monitor.monitoring_thread.is_alive()
        
        # Stop monitoring
        monitor.stop_monitoring()
        assert not monitor.monitoring_active
        
        # Wait for thread to finish
        time.sleep(0.1)
        assert not monitor.monitoring_thread.is_alive()


class TestSafeOperation:
    """Test SafeOperation context manager"""
    
    def test_safe_operation_success(self):
        """Test SafeOperation with successful operation"""
        # Initialize error handler for test
        ErrorHandler.initialize()
        
        with SafeOperation("test_component", "test_operation"):
            result = "success"
        
        assert result == "success"
    
    def test_safe_operation_with_error(self):
        """Test SafeOperation with error"""
        # Initialize error handler for test
        handler = ErrorHandler.initialize()
        handler.register_component("test_component")
        
        with pytest.raises(ValueError):
            with SafeOperation("test_component", "test_operation"):
                raise ValueError("Test error")
        
        # Check that error was handled
        health = handler.get_component_health("test_component")
        assert health['error_count'] > 0


class TestErrorBoundary:
    """Test error_boundary decorator"""
    
    def test_error_boundary_success(self):
        """Test error_boundary with successful function"""
        ErrorHandler.initialize()
        
        @error_boundary("test_component")
        def successful_function():
            return "success"
        
        result = successful_function()
        assert result == "success"
    
    def test_error_boundary_with_retry(self):
        """Test error_boundary with retries"""
        ErrorHandler.initialize()
        call_count = 0
        
        @error_boundary("test_component", max_retries=2)
        def failing_function():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise ValueError("Test error")
            return "success after retries"
        
        result = failing_function()
        assert result == "success after retries"
        assert call_count == 3  # Initial + 2 retries
    
    def test_error_boundary_max_retries_exceeded(self):
        """Test error_boundary when max retries exceeded"""
        ErrorHandler.initialize()
        
        @error_boundary("test_component", max_retries=1)
        def always_failing_function():
            raise ValueError("Always fails")
        
        with pytest.raises(ValueError):
            always_failing_function()


class TestErrorHandler:
    """Test ErrorHandler singleton class"""
    
    def setup_method(self):
        """Reset singleton for each test"""
        ErrorHandler._instance = None
    
    def test_singleton_initialization(self):
        """Test ErrorHandler singleton pattern"""
        handler1 = ErrorHandler.initialize()
        handler2 = ErrorHandler.get_instance()
        
        assert handler1 is handler2
        assert isinstance(handler1, ErrorHandler)
    
    def test_get_instance_without_initialization(self):
        """Test getting instance before initialization raises error"""
        with pytest.raises(RuntimeError, match="ErrorHandler not initialized"):
            ErrorHandler.get_instance()
    
    def test_component_registration(self):
        """Test component registration"""
        handler = ErrorHandler.initialize()
        handler.register_component("test_component")
        
        health = handler.get_component_health("test_component")
        assert health['status'] == 'healthy'
        assert health['error_count'] == 0
        assert health['recovery_count'] == 0
    
    def test_recovery_callback_registration(self):
        """Test recovery callback registration"""
        handler = ErrorHandler.initialize()
        callback = Mock()
        
        handler.add_recovery_callback("test_component", callback)
        assert "test_component" in handler._recovery_callbacks
        assert callback in handler._recovery_callbacks["test_component"]
    
    def test_error_handling(self):
        """Test error event handling"""
        handler = ErrorHandler.initialize()
        handler.register_component("test_component")
        
        error_event = ErrorEvent(
            timestamp=time.time(),
            component="test_component",
            error_type="ValueError",
            message="Test error",
            severity=ErrorSeverity.MEDIUM,
            traceback="test traceback",
            recovery_strategy=RecoveryStrategy.RETRY
        )
        
        handler.handle_error(error_event)
        
        # Check error was recorded
        assert len(handler._error_history) == 1
        assert handler._error_history[0] == error_event
        
        # Check component status updated
        health = handler.get_component_health("test_component")
        assert health['error_count'] == 1
        assert health['last_error'] == error_event
    
    def test_component_status_degradation(self):
        """Test component status changes with multiple errors"""
        handler = ErrorHandler.initialize()
        handler.register_component("test_component")
        
        # Add multiple medium severity errors
        for i in range(6):
            error_event = ErrorEvent(
                timestamp=time.time(),
                component="test_component",
                error_type="ValueError",
                message=f"Test error {i}",
                severity=ErrorSeverity.MEDIUM,
                traceback="test traceback",
                recovery_strategy=RecoveryStrategy.NONE
            )
            handler.handle_error(error_event)
        
        health = handler.get_component_health("test_component")
        assert health['status'] == 'degraded'  # Should be degraded after >5 errors
    
    def test_critical_error_status(self):
        """Test component status with critical error"""
        handler = ErrorHandler.initialize()
        handler.register_component("test_component")
        
        critical_error = ErrorEvent(
            timestamp=time.time(),
            component="test_component",
            error_type="CriticalError",
            message="Critical failure",
            severity=ErrorSeverity.CRITICAL,
            traceback="critical traceback",
            recovery_strategy=RecoveryStrategy.RESET
        )
        
        handler.handle_error(critical_error)
        
        health = handler.get_component_health("test_component")
        assert health['status'] == 'failing'  # Should immediately be failing
    
    def test_recovery_attempt(self):
        """Test recovery attempt execution"""
        handler = ErrorHandler.initialize()
        handler.register_component("test_component")
        
        callback = Mock()
        handler.add_recovery_callback("test_component", callback)
        
        error_event = ErrorEvent(
            timestamp=time.time(),
            component="test_component",
            error_type="ValueError",
            message="Test error",
            severity=ErrorSeverity.MEDIUM,
            traceback="test traceback",
            recovery_strategy=RecoveryStrategy.RETRY
        )
        
        handler.handle_error(error_event)
        
        # Recovery callback should have been called
        callback.assert_called_once()
        
        # Recovery count should be updated
        health = handler.get_component_health("test_component")
        assert health['recovery_count'] == 1
    
    def test_force_component_recovery(self):
        """Test forced component recovery"""
        handler = ErrorHandler.initialize()
        handler.register_component("test_component")
        
        callback = Mock()
        handler.add_recovery_callback("test_component", callback)
        
        result = handler.force_component_recovery("test_component")
        assert result is True
        callback.assert_called_once()
    
    def test_force_recovery_nonexistent_component(self):
        """Test forced recovery on nonexistent component"""
        handler = ErrorHandler.initialize()
        
        result = handler.force_component_recovery("nonexistent")
        assert result is False
    
    def test_error_statistics(self):
        """Test error statistics generation"""
        handler = ErrorHandler.initialize()
        handler.register_component("comp1")
        handler.register_component("comp2")
        
        # Add various errors
        errors = [
            ErrorEvent(time.time(), "comp1", "ValueError", "Error 1", 
                      ErrorSeverity.LOW, None, RecoveryStrategy.NONE),
            ErrorEvent(time.time(), "comp1", "TypeError", "Error 2", 
                      ErrorSeverity.HIGH, None, RecoveryStrategy.NONE),
            ErrorEvent(time.time(), "comp2", "RuntimeError", "Error 3", 
                      ErrorSeverity.CRITICAL, None, RecoveryStrategy.NONE),
        ]
        
        for error in errors:
            handler.handle_error(error)
        
        stats = handler.get_error_statistics()
        
        assert stats['total_errors'] == 3
        assert stats['errors_by_component']['comp1'] == 2
        assert stats['errors_by_component']['comp2'] == 1
        assert stats['errors_by_severity']['low'] == 1
        assert stats['errors_by_severity']['high'] == 1
        assert stats['errors_by_severity']['critical'] == 1
        assert 'memory_info' in stats
        assert 'component_health' in stats
    
    def test_error_history_limit(self):
        """Test error history size limit"""
        handler = ErrorHandler.initialize(max_error_history=5)
        handler.register_component("test_component")
        
        # Add more errors than the limit
        for i in range(10):
            error_event = ErrorEvent(
                timestamp=time.time(),
                component="test_component",
                error_type="ValueError",
                message=f"Error {i}",
                severity=ErrorSeverity.LOW,
                traceback=None,
                recovery_strategy=RecoveryStrategy.NONE
            )
            handler.handle_error(error_event)
        
        # Should only keep the last 5 errors
        assert len(handler._error_history) == 5
        assert handler._error_history[-1].message == "Error 9"  # Most recent
    
    def test_unknown_component_health(self):
        """Test health check for unknown component"""
        handler = ErrorHandler.initialize()
        
        health = handler.get_component_health("unknown_component")
        assert health['status'] == 'unknown'
        assert health['error_count'] == 0
    
    def test_shutdown(self):
        """Test error handler shutdown"""
        handler = ErrorHandler.initialize()
        
        # Memory monitor should be running
        assert handler.memory_monitor.monitoring_active
        
        handler.shutdown()
        
        # Memory monitor should be stopped
        assert not handler.memory_monitor.monitoring_active