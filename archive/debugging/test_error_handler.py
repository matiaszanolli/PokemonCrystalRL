#!/usr/bin/env python3
"""
Test script for the comprehensive error handling and recovery system
"""

import os
import sys
import time
import threading

# Add paths for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from monitoring.error_handler import (
    ErrorHandler, ErrorSeverity, RecoveryStrategy, ErrorEvent,
    error_boundary, SafeOperation, MemoryMonitor
)

def test_error_handler_initialization():
    """Test error handler initialization and singleton pattern"""
    print("🧪 Testing Error Handler Initialization")
    print("-" * 50)
    
    # Initialize error handler
    error_handler1 = ErrorHandler.initialize(
        max_error_history=100,
        memory_threshold_mb=512.0
    )
    
    # Test singleton pattern
    error_handler2 = ErrorHandler.get_instance()
    
    if error_handler1 is error_handler2:
        print("✅ Singleton pattern working correctly")
    else:
        print("❌ Singleton pattern failed")
    
    # Test component registration
    error_handler1.register_component("test_component")
    health = error_handler1.get_component_health("test_component")
    
    if health and health['status'] == 'healthy':
        print("✅ Component registration working")
    else:
        print("❌ Component registration failed")
    
    return error_handler1

def test_error_boundary_decorator():
    """Test the error boundary decorator"""
    print("\n🧪 Testing Error Boundary Decorator")
    print("-" * 50)
    
    error_handler = ErrorHandler.get_instance()
    error_count_before = len(error_handler._error_history)
    
    @error_boundary("test_component", severity=ErrorSeverity.MEDIUM, max_retries=2)
    def failing_function():
        """Function that always fails"""
        raise ValueError("Test error for boundary testing")
    
    @error_boundary("test_component", severity=ErrorSeverity.LOW, max_retries=1)  
    def working_function():
        """Function that works"""
        return "success"
    
    # Test working function
    try:
        result = working_function()
        if result == "success":
            print("✅ Error boundary allows successful operations")
        else:
            print("❌ Error boundary interfered with successful operation")
    except Exception:
        print("❌ Error boundary failed on successful operation")
    
    # Test failing function
    try:
        failing_function()
        print("❌ Error boundary should have raised exception")
    except ValueError:
        print("✅ Error boundary correctly handled and re-raised exception")
    except Exception as e:
        print(f"❌ Error boundary raised wrong exception: {e}")
    
    # Check if errors were recorded
    error_count_after = len(error_handler._error_history)
    if error_count_after > error_count_before:
        print(f"✅ Error boundary recorded {error_count_after - error_count_before} errors")
    else:
        print("❌ Error boundary did not record errors")

def test_memory_monitoring():
    """Test memory monitoring functionality"""
    print("\n🧪 Testing Memory Monitoring")
    print("-" * 50)
    
    memory_monitor = MemoryMonitor(threshold_mb=1.0, check_interval=1.0)  # Very low threshold for testing
    
    # Test memory info
    memory_info = memory_monitor.get_memory_info()
    if memory_info and 'rss_mb' in memory_info:
        print(f"✅ Memory monitoring working: {memory_info['rss_mb']:.1f}MB used")
    else:
        print("❌ Memory monitoring failed to get info")
    
    # Test garbage collection
    gc_stats = memory_monitor.trigger_garbage_collection()
    if gc_stats and 'objects_collected' in gc_stats:
        print(f"✅ Garbage collection working: {gc_stats['objects_collected']} objects collected")
    else:
        print("❌ Garbage collection failed")
    
    # Test monitoring thread
    memory_monitor.start_monitoring()
    time.sleep(2.0)  # Let it run briefly
    memory_monitor.stop_monitoring()
    print("✅ Memory monitoring thread started and stopped successfully")

def test_component_health_tracking():
    """Test component health tracking"""
    print("\n🧪 Testing Component Health Tracking")
    print("-" * 50)
    
    error_handler = ErrorHandler.get_instance()
    
    # Register test components
    error_handler.register_component("healthy_component")
    error_handler.register_component("failing_component")
    
    # Simulate errors in failing component
    for i in range(3):
        error_event = ErrorEvent(
            timestamp=time.time(),
            component="failing_component",
            error_type="TestError",
            message=f"Test error {i+1}",
            severity=ErrorSeverity.MEDIUM,
            traceback=None,
            recovery_strategy=RecoveryStrategy.RETRY
        )
        error_handler.handle_error(error_event)
        time.sleep(0.1)
    
    # Check component health
    failing_health = error_handler.get_component_health("failing_component")
    healthy_health = error_handler.get_component_health("healthy_component")
    
    if failing_health['status'] == 'degraded' and failing_health['error_count'] == 3:
        print("✅ Failing component correctly marked as degraded")
    else:
        print(f"❌ Failing component status: {failing_health['status']}, errors: {failing_health['error_count']}")
    
    if healthy_health['status'] == 'healthy' and healthy_health['error_count'] == 0:
        print("✅ Healthy component status correct")
    else:
        print(f"❌ Healthy component status: {healthy_health['status']}, errors: {healthy_health['error_count']}")

def test_recovery_callbacks():
    """Test recovery callback functionality"""
    print("\n🧪 Testing Recovery Callbacks")
    print("-" * 50)
    
    error_handler = ErrorHandler.get_instance()
    
    # Track callback execution
    callback_executed = {'count': 0}
    
    def recovery_callback():
        callback_executed['count'] += 1
        print(f"   🔄 Recovery callback executed (count: {callback_executed['count']})")
    
    # Register callback
    error_handler.add_recovery_callback("test_recovery_component", recovery_callback)
    error_handler.register_component("test_recovery_component")
    
    # Force component recovery
    success = error_handler.force_component_recovery("test_recovery_component")
    
    if success and callback_executed['count'] > 0:
        print("✅ Recovery callback system working")
    else:
        print(f"❌ Recovery callback failed: success={success}, count={callback_executed['count']}")

def test_safe_operation_context_manager():
    """Test SafeOperation context manager"""
    print("\n🧪 Testing SafeOperation Context Manager")
    print("-" * 50)
    
    error_handler = ErrorHandler.get_instance()
    initial_error_count = len(error_handler._error_history)
    
    # Test successful operation
    with SafeOperation("test_context", "successful_operation"):
        result = "success"
    
    # Test failing operation
    try:
        with SafeOperation("test_context", "failing_operation"):
            raise RuntimeError("Test context manager error")
    except RuntimeError:
        pass  # Expected
    
    # Check if error was recorded
    final_error_count = len(error_handler._error_history)
    if final_error_count > initial_error_count:
        print("✅ SafeOperation context manager recorded errors")
    else:
        print("❌ SafeOperation context manager did not record errors")

def test_error_statistics():
    """Test error statistics and reporting"""
    print("\n🧪 Testing Error Statistics")
    print("-" * 50)
    
    error_handler = ErrorHandler.get_instance()
    
    # Get statistics
    stats = error_handler.get_error_statistics()
    
    required_fields = [
        'total_errors', 'recent_errors_1h', 'errors_by_component',
        'errors_by_severity', 'component_health', 'memory_info'
    ]
    
    missing_fields = [field for field in required_fields if field not in stats]
    
    if not missing_fields:
        print("✅ Error statistics contain all required fields")
        print(f"   Total errors: {stats['total_errors']}")
        print(f"   Recent errors (1h): {stats['recent_errors_1h']}")
        print(f"   Component health rate: {stats['component_health']['health_rate']:.2f}")
    else:
        print(f"❌ Missing fields in error statistics: {missing_fields}")

def test_cleanup():
    """Test system cleanup"""
    print("\n🧪 Testing System Cleanup")
    print("-" * 50)
    
    error_handler = ErrorHandler.get_instance()
    
    try:
        error_handler.shutdown()
        print("✅ Error handler shutdown completed successfully")
    except Exception as e:
        print(f"❌ Error handler shutdown failed: {e}")

if __name__ == "__main__":
    print("🔬 ERROR HANDLER COMPREHENSIVE TEST")
    print("=" * 70)
    
    try:
        error_handler = test_error_handler_initialization()
        test_error_boundary_decorator()
        test_memory_monitoring()
        test_component_health_tracking()
        test_recovery_callbacks()
        test_safe_operation_context_manager()
        test_error_statistics()
        test_cleanup()
        
        print(f"\n🎉 All error handler tests completed!")
        
    except KeyboardInterrupt:
        print(f"\n⏸️ Tests interrupted by user")
    except Exception as e:
        print(f"\n💥 Tests failed: {e}")
        import traceback
        traceback.print_exc()
