#!/usr/bin/env python3
"""
Test script for the StatsCollector system
"""

import os
import sys
import time
import threading

# Add paths for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from monitoring.stats_collector import StatsCollector, Timer
from monitoring.data_bus import get_data_bus, DataType

def test_basic_metrics():
    """Test basic metric recording and aggregation"""
    print("🧪 Testing Basic Metrics Collection")
    print("-" * 40)
    
    collector = StatsCollector(aggregation_window=1.0, enable_data_bus=False)
    collector.start_collection()
    
    try:
        # Record various metrics
        collector.record_counter("test.actions", 1.0)
        collector.record_counter("test.actions", 1.0) 
        collector.record_counter("test.actions", 1.0)
        
        collector.record_gauge("test.speed", 45.0)
        collector.record_gauge("test.speed", 50.0)
        
        collector.record_timer("test.response_time", 0.1)
        collector.record_timer("test.response_time", 0.15)
        collector.record_timer("test.response_time", 0.08)
        
        collector.record_histogram("test.values", 10.0)
        collector.record_histogram("test.values", 20.0)
        collector.record_histogram("test.values", 15.0)
        
        # Wait for aggregation (longer to ensure it happens)
        time.sleep(2.5)
        
        # Check results
        summary = collector.get_metrics_summary()
        print(f"📊 Collected metrics:")
        
        for name, metric in summary.items():
            print(f"  {name:20s}: type={metric['type']:8s} current={metric['current']:6.1f} count={metric['count']} avg={metric['avg']:6.2f}")
        
        # Validate specific metrics
        actions_metric = collector.get_metric("test.actions")
        if actions_metric and actions_metric.current_value == 3.0:
            print("✅ Counter metrics working correctly")
        else:
            print("❌ Counter metrics failed")
        
        speed_metric = collector.get_metric("test.speed")
        if speed_metric and speed_metric.current_value == 50.0:  # Most recent value
            print("✅ Gauge metrics working correctly")
        else:
            print("❌ Gauge metrics failed")
        
        timer_metric = collector.get_metric("test.response_time") 
        if timer_metric and timer_metric.count == 3:
            print("✅ Timer metrics working correctly")
        else:
            print("❌ Timer metrics failed")
            
    finally:
        collector.shutdown()

def test_data_bus_integration():
    """Test integration with the data bus"""
    print("\n🧪 Testing Data Bus Integration")
    print("-" * 40)
    
    collector = StatsCollector(aggregation_window=1.0, enable_data_bus=True)
    collector.start_collection()
    
    try:
        # Get data bus and publish some events
        data_bus = get_data_bus()
        
        # Publish training stats
        data_bus.publish(DataType.TRAINING_STATS, {
            "total_actions": 100,
            "actions_per_second": 45.0,
            "llm_calls": 10
        }, "test")
        
        # Publish action events
        data_bus.publish(DataType.ACTION_TAKEN, {
            "action_id": 5,
            "action_name": "A",
            "execution_time": 0.05
        }, "test")
        
        data_bus.publish(DataType.ACTION_TAKEN, {
            "action_id": 7, 
            "action_name": "START",
            "execution_time": 0.03
        }, "test")
        
        # Publish LLM decision
        data_bus.publish(DataType.LLM_DECISION, {
            "response_time": 0.12,
            "success": True
        }, "test")
        
        # Publish game state change
        data_bus.publish(DataType.GAME_STATE, {
            "state": "dialogue",
            "previous_state": "title_screen"
        }, "test")
        
        # Publish error event
        data_bus.publish(DataType.ERROR_EVENT, {
            "component": "test_component",
            "severity": "medium"
        }, "test")
        
        # Wait for collection and aggregation (longer)
        time.sleep(2.5)
        
        # Check collected metrics
        summary = collector.get_metrics_summary()
        
        print(f"📊 Data bus integration metrics:")
        for name, metric in sorted(summary.items()):
            if metric['count'] > 0:
                print(f"  {name:25s}: {metric['current']:6.1f} (count: {metric['count']})")
        
        # Validate key metrics
        if "training.total_actions" in summary:
            print("✅ Training stats integration working")
        else:
            print("❌ Training stats integration failed")
        
        if "actions.total" in summary:
            print("✅ Action tracking working")
        else:
            print("❌ Action tracking failed")
        
        if "llm.decisions" in summary:
            print("✅ LLM decision tracking working")
        else:
            print("❌ LLM decision tracking failed")
        
    finally:
        collector.shutdown()

def test_timer_context_manager():
    """Test the Timer context manager"""
    print("\n🧪 Testing Timer Context Manager")
    print("-" * 40)
    
    collector = StatsCollector(aggregation_window=1.0, enable_data_bus=False)
    collector.start_collection()
    
    try:
        # Use timer context manager
        with Timer(collector, "test.operation_time"):
            time.sleep(0.1)  # Simulate work
        
        with Timer(collector, "test.operation_time"):
            time.sleep(0.05)  # Simulate different work
        
        # Wait for aggregation
        time.sleep(2.0)
        
        # Check results
        timer_metric = collector.get_metric("test.operation_time")
        if timer_metric and timer_metric.count == 2:
            print(f"✅ Timer context manager working: avg={timer_metric.avg_value:.3f}s")
        else:
            print("❌ Timer context manager failed")
            
    finally:
        collector.shutdown()

def test_performance():
    """Test performance with high-frequency metrics"""
    print("\n🧪 Testing Performance")
    print("-" * 40)
    
    collector = StatsCollector(aggregation_window=1.0, enable_data_bus=False)
    collector.start_collection()
    
    try:
        start_time = time.time()
        num_metrics = 1000
        
        # Record many metrics quickly
        for i in range(num_metrics):
            collector.record_counter("perf.test")
            collector.record_gauge("perf.value", float(i))
            if i % 10 == 0:
                collector.record_timer("perf.interval", 0.01)
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"📊 Recorded {num_metrics} metrics in {duration:.3f}s")
        print(f"   Rate: {num_metrics/duration:.0f} metrics/second")
        
        # Wait for aggregation
        time.sleep(2.0)
        
        # Check performance stats
        perf_stats = collector.get_performance_stats()
        print(f"📈 Collector performance:")
        print(f"   Collection count: {perf_stats['collection_count']}")
        print(f"   Collection errors: {perf_stats['collection_errors']}")
        print(f"   Error rate: {perf_stats['error_rate']:.4f}")
        print(f"   Active metrics: {perf_stats['active_metrics']}")
        
        if perf_stats['collection_errors'] == 0:
            print("✅ High-frequency collection working without errors")
        else:
            print("❌ Collection errors occurred")
            
    finally:
        collector.shutdown()

if __name__ == "__main__":
    print("🔬 STATS COLLECTOR COMPREHENSIVE TEST")
    print("=" * 50)
    
    try:
        test_basic_metrics()
        test_data_bus_integration() 
        test_timer_context_manager()
        test_performance()
        
        print(f"\n🎉 All tests completed!")
        
    except KeyboardInterrupt:
        print(f"\n⏸️ Tests interrupted by user")
    except Exception as e:
        print(f"\n💥 Tests failed: {e}")
        import traceback
        traceback.print_exc()
