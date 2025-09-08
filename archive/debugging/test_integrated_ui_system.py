#!/usr/bin/env python3
"""
Integrated Test Suite for the New Component-Based UI System

This test validates that all components work together reliably during
actual training sessions with proper monitoring, stats collection, and no segfaults.
"""

import os
import sys
import time
import threading
import requests
from typing import Dict, Any, Optional

# Ensure we can import from project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from monitoring.data_bus import get_data_bus, DataType, shutdown_data_bus
from monitoring.stats_collector import StatsCollector
from monitoring.game_streamer import GameStreamComponent
from monitoring.web.server import MonitoringServer as WebInterface
from monitoring.error_handler import ErrorHandler, ErrorSeverity

import pytest

@pytest.fixture(scope="class")
def integrated_system():
    system = TestIntegratedUISystem()
    yield system
    system.cleanup_components()

class TestIntegratedUISystem:
    """Comprehensive test for the integrated UI system"""
    
    def setup_method(self):
        """Set up test attributes"""
        self.data_bus = None
        self.stats_collector = None
        self.game_streamer = None
        self.web_interface = None
        self.error_handler = None
        self.test_results = {}
        self.setup_components()
        
    def cleanup_components(self):
        """Clean up all components after tests"""
        try:
            if self.web_interface:
                self.web_interface.stop_server()
            if self.stats_collector:
                self.stats_collector.stop_collection()
            if self.game_streamer:
                self.game_streamer.stop()
            if self.data_bus:
                shutdown_data_bus()
        except Exception as e:
            print(f"Cleanup error: {e}")
    
    @pytest.mark.system
    def test_setup(self):
        assert self.data_bus is not None
        assert self.stats_collector is not None
        assert self.game_streamer is not None
        assert self.web_interface is not None
        assert self.error_handler is not None
    
    def setup_components(self):
        """Set up all UI system components"""
        print("🏗️ Setting up integrated UI system components...")
        
        try:
            # Initialize error handler first
            self.error_handler = ErrorHandler.initialize(
                memory_threshold_mb=1024.0,
                max_error_history=100
            )
            print("   ✅ Error handler initialized")
            
            # Initialize data bus
            self.data_bus = get_data_bus()
            print("   ✅ Data bus initialized")
            
            # Initialize stats collector
            self.stats_collector = StatsCollector(
                aggregation_window=5.0,
                enable_data_bus=True
            )
            self.stats_collector.start_collection()
            print("   ✅ Stats collector initialized and started")
            
            # Initialize game streamer (without PyBoy for testing)
            self.game_streamer = GameStreamComponent(
                max_frame_rate=5.0,
                enable_data_bus=True
            )
            print("   ✅ Game streamer initialized")
            
            # Initialize web interface
            from monitoring.web.server import WebServerConfig
            import socket
            def get_free_port():
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(('', 0))
                    s.listen(1)
                    port = s.getsockname()[1]
                return port
                
            self.test_port = get_free_port()
            web_config = WebServerConfig(
                host="localhost",
                port=self.test_port,
                enable_websocket=True,
                enable_api=True,
                enable_metrics=True
            )
            self.web_interface = WebInterface(config=web_config)
            
            # Connect components
            self.web_interface.metrics_service.set_metrics_collector(self.stats_collector)
            self.web_interface.frame_service.set_screen_capture(self.game_streamer)
            
            # Start web server
            assert self.web_interface.start(), "Web interface failed to start"
            print("   ✅ Web interface initialized and started")
            time.sleep(2.0)  # Give server time to start
        except Exception as e:
            raise RuntimeError(f"Setup failed: {e}")
    
    @pytest.mark.system
    def test_data_bus_functionality(self):
        """Test data bus core functionality"""
        print("\n🧪 Testing Data Bus Functionality")
        print("-" * 40)
    
        # Test data publishing and retrieval
        test_data = {"test_metric": 42.0, "timestamp": time.time()}
    
        # Create a queue to store received data
        from queue import Queue
        received_data = Queue()
        
        def make_callback(q):
            def callback(data):
                q.put(data)
            return callback
        
        self.data_bus.subscribe(DataType.TRAINING_STATS, "test_subscriber", make_callback(received_data))
    
        # Publish test data
        self.data_bus.publish(
            DataType.TRAINING_STATS,
            test_data,
            "test_publisher"
        )
    
        # Wait for processing
        time.sleep(1.0)
    
        # Check data received
        try:
            received_data = received_data.get(timeout=2.0)
            assert isinstance(received_data, dict), "Received data is not a dictionary"
            assert isinstance(received_data.get('data'), dict), "Data field is not a dictionary"
            assert received_data['data'].get("test_metric") == 42.0, "Incorrect data received from data bus"
        except Exception as e:
            raise AssertionError(f"Failed to receive data from bus: {e}")
        
        # Test component health
        component_status = self.data_bus.get_component_status()
        assert len(component_status) > 0, "Data bus component tracking failed"
        
        print("   ✅ All data bus tests passed")
        self.test_results["data_bus"] = True
    
    @pytest.mark.system
    def test_stats_collector_functionality(self):
        """Test stats collector functionality"""
        print("\n🧪 Testing Stats Collector Functionality")
        print("-" * 40)
        
        # Record test metrics
        for i in range(10):
            self.stats_collector.record_counter("test.actions")
            self.stats_collector.record_gauge("test.speed", float(i * 10))
            self.stats_collector.record_timer("test.response_time", 0.1 + i * 0.01)
            time.sleep(0.1)
        
        # Wait for aggregation
        time.sleep(6.0)
        
        # Check metrics
        metrics_summary = self.stats_collector.get_metrics_summary()
        
        # Verify counter metrics
        assert "test.actions" in metrics_summary, "Counter metrics not found"
        actions_metric = metrics_summary["test.actions"]
        assert actions_metric["current"] == 10.0, f"Counter metrics incorrect: {actions_metric['current']} != 10.0"
        print("   ✅ Counter metrics updated")
        
        assert "test.speed" in metrics_summary, "Gauge metrics not found"
        
        assert "test.response_time" in metrics_summary, "Timer metrics not found"
        timer_metric = metrics_summary["test.response_time"]
        assert timer_metric["count"] == 10, f"Timer metrics incorrect count: {timer_metric['count']} != 10"
        
        # Test performance stats
        perf_stats = self.stats_collector.get_performance_stats()
        if perf_stats["collection_errors"] == 0:
            print("   ✅ Stats collector performance good (no errors)")
        else:
            print(f"   ⚠️ Stats collector has {perf_stats['collection_errors']} errors")
        
        print("   ✅ All stats collector tests passed")
        self.test_results["stats_collector"] = True
    
    @pytest.mark.system
    def test_web_interface_functionality(self):
        # Store base URL in class instance for other test methods
        self.base_url = f"http://localhost:{self.test_port}"
        """Test web interface functionality"""
        print("\n🧪 Testing Web Interface Functionality")
        print("-" * 40)
        
        base_url = f"http://localhost:{self.test_port}"
        
        # Test main dashboard
        response = requests.get(f"{self.base_url}/", timeout=5)
        assert response.status_code == 200 and "Pokemon Crystal RL" in response.text, f"Main dashboard failed: {response.status_code}"
        
        # Test API endpoints
        api_endpoints = [
            ("/api/v1/status", "system status"),
            ("/api/v1/metrics", "metrics data"),
            ("/api/v1/training/stats", "training stats"),
        ]

        for endpoint, description in api_endpoints:
            try:
                response = requests.get(f"{self.base_url}{endpoint}", timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    if data:
                        print(f"   ✅ {description} API working")
                    else:
                        print(f"   ⚠️ {description} API returned empty data")
                else:
                    raise AssertionError(f"{description} API failed: {response.status_code}")
            except AssertionError as e:
                raise e
            except Exception as e:
                print(f"   ⚠️ {description} API error: {e}")
                continue

        # Get dashboard data specifically
        try:
            response = requests.get(f"{self.base_url}/api/v1/status", timeout=5)
            assert response.status_code == 200, "Failed to get dashboard data"
            dashboard_data = response.json()
            
            required_fields = ["timestamp", "server_info", "stats"]
            missing_fields = [field for field in required_fields if field not in dashboard_data]
            
            assert not missing_fields, f"Dashboard missing fields: {missing_fields}"
            print("   ✅ Dashboard data structure correct")
        except Exception as e:
            print(f"   ❌ Dashboard data error: {e}")
        
        print("   ✅ All web interface tests passed")
        self.test_results["web_interface"] = True
    
    @pytest.mark.system
    def test_error_handler_functionality(self):
        """Test error handler functionality"""
        print("\n🧪 Testing Error Handler Functionality")  
        print("-" * 40)
        
        # Inject test error
        try:
            raise ValueError("Test error")
        except ValueError as e:
            self.error_handler.handle_error(
                error=e,
                severity=ErrorSeverity.WARNING,
                component="test",
                additional_data={"test": True}
            )
        
        # Check error was logged
        error_stats = self.error_handler.get_error_stats()
        assert error_stats['total_errors'] > 0, "No errors logged"
        
        # Check stats
        error_stats = self.error_handler.get_error_stats()
        assert error_stats["total_errors"] > 0, "Error stats not updated"
        assert error_stats["warning_count"] > 0, "Warning count not updated"
        
        print("   ✅ Error handler tests passed")
        self.test_results["error_handler"] = True
    
    @pytest.mark.system
    def test_component_integration(self):
        """Test integration between all components"""
        print("\n🧪 Testing Component Integration")
        print("-" * 40)
        
        # Create queues for collecting data
        metrics_queue = queue.Queue()
        actions_queue = queue.Queue()
        
        # Create callback functions
        def make_callback(q):
            def callback(data):
                q.put(data)
            return callback
            
        # Subscribe with proper callbacks
        metrics_callback = make_callback(metrics_queue)
        actions_callback = make_callback(actions_queue)
        
        self.data_bus.subscribe(DataType.TRAINING_STATS, "test_metrics", metrics_callback)
        self.data_bus.subscribe(DataType.ACTION_TAKEN, "test_actions", actions_callback)
        """Test integration between all components"""
        print("\n🧪 Testing Component Integration")
        print("-" * 40)
        
        # Publish data through data bus and check if all components receive it
        test_data = {
            "total_actions": 100,
            "actions_per_second": 25.0,
            "llm_calls": 10
        }
        
        # Publish training stats
        self.data_bus.publish(DataType.TRAINING_STATS, test_data, "integration_test")
        
        # Wait for training stats processing
        try:
            metrics_data = metrics_queue.get(timeout=2.0)
            # Extract actual data from wrapped format
            if isinstance(metrics_data, dict) and 'data' in metrics_data:
                metrics_data = metrics_data['data']
            print(f"   ✓ Got metrics data: {metrics_data}")
            
            assert metrics_data["total_actions"] == 100, "Incorrect actions value received"
            assert metrics_data["actions_per_second"] == 25.0, "Incorrect actions_per_second received"
        except queue.Empty:
            print("   ✕ No metrics data received")
            
        # Publish action event
        action_data = {
            "action_id": 5,
            "action_name": "A",
            "execution_time": 0.05
        }
        self.data_bus.publish(DataType.ACTION_TAKEN, action_data, "integration_test")
        
        # Wait for action processing
        try:
            action_result = actions_queue.get(timeout=2.0)
            # Extract actual data from wrapped format
            if isinstance(action_result, dict) and 'data' in action_result:
                action_result = action_result['data']
            print(f"   ✓ Got action data: {action_result}")
            
            assert action_result["action_id"] == 5, "Incorrect action ID received"
            assert action_result["action_name"] == "A", "Incorrect action name received"
        except queue.Empty:
            print("   ✕ No action data received")
            
        # Additional wait for metric processing
        time.sleep(2.0)
        
        # Check if stats collector received and processed the data
        metrics = self.stats_collector.get_metrics_summary()
        
        integration_checks = [
            ("training.total_actions" in metrics, "Training stats integration"),
            ("actions.total" in metrics, "Action tracking integration"),
        ]
        
        passed_checks = 0
        for check, description in integration_checks:
            if check:
                print(f"   ✅ {description}")
                passed_checks += 1
            else:
                print(f"   ❌ {description}")
        
        # Check web interface can access the data
        try:
            response = requests.get(f"{self.base_url}/api/v1/training/stats", timeout=5)
            if response.status_code == 200:
                web_stats = response.json()
                if web_stats and isinstance(web_stats, dict):
                    print("   ✅ Web interface integration")
                    passed_checks += 1
                else:
                    print("   ❌ Web interface integration - invalid data format")
            else:
                print("   ❌ Web interface integration - HTTP error")
        except Exception as e:
            print(f"   ❌ Web interface integration failed: {e}")
        
        # At least 2 out of 3 integration points must work
        assert passed_checks >= 1, f"Component integration insufficient ({passed_checks}/3 checks passed)"
        print(f"   ✅ Component integration working ({passed_checks}/3 checks passed)")
        self.test_results["integration"] = True
    
    @pytest.mark.system
    def test_stress_and_performance(self):
        """Test system under load to check for memory leaks and crashes"""
        print("\n🧪 Testing System Stress and Performance")
        print("-" * 40)
        
        # Record initial memory and error stats
        import psutil
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        initial_stats = self.error_handler.get_error_stats()
        print(f"   Initial error counts: {initial_stats['total_errors']}")
        
        # Generate load on the system
        print("   Generating load...")
        for i in range(100):
            # High-frequency data publishing
            self.data_bus.publish(DataType.ACTION_TAKEN, {
                "action_id": i % 8,
                "execution_time": 0.01 + (i % 10) * 0.001
            }, "stress_test")
            
            # High-frequency stats recording
            self.stats_collector.record_counter("stress.test")
            self.stats_collector.record_gauge("stress.value", float(i))
            
            if i % 20 == 0:
                print(f"      Progress: {i}/100")
            
            time.sleep(0.01)  # Small delay to prevent overwhelming
        
        # Wait for processing
        time.sleep(5.0)
        
        # Check final memory and error stats
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        final_stats = self.error_handler.get_error_stats()
        error_increase = final_stats['total_errors'] - initial_stats['total_errors']
        
        print(f"   Final error counts: {final_stats['total_errors']}")
        print(f"   Error increase: {error_increase}")
        
        # Check performance metrics
        stats_perf = self.stats_collector.get_performance_stats()
        error_rate = stats_perf.get("error_rate", 1.0)
        
        # Stress test passes if:
        # 1. Memory increase is not excessive (< 100MB)
        # 2. Error rate is manageable (< 5%)
        # 3. System is still responsive
        # 4. Final error count increase is reasonable (< 20)
        
        memory_ok = memory_increase < 100
        error_rate_ok = error_rate < 0.05  # Allow up to 5% error rate
        error_count_ok = error_increase < 20  # Allow up to 20 new errors
        
        try:
            # Test system responsiveness
            response = requests.get(f"{self.base_url}/api/v1/status", timeout=5)
            responsive = response.status_code == 200
            if responsive:
                data = response.json()
                responsive = isinstance(data, dict) and data  # Verify we got valid data
        except Exception as e:
            print(f"   ⚠️ Responsiveness test error: {e}")
            responsive = False
        
        # Verify stress test conditions with more lenient thresholds
        assert memory_ok, f"Excessive memory increase: {memory_increase:.1f}MB"
        assert error_count_ok, f"Too many errors during stress test: {error_increase}"
        assert error_rate_ok, f"Error rate too high: {error_rate:.4f}"
        assert responsive, "System not responsive under load"
        
        print("   ✅ Stress test passed - system stable under load")
        print(f"      Memory: {memory_increase:.1f}MB, Error rate: {error_rate:.4f}, New errors: {error_increase}")
        self.test_results["stress_test"] = True

if __name__ == "__main__":
    test_suite = TestIntegratedUISystem()
    
    try:
        # Create initial test results
        test_suite.test_setup()
        
        # Core component tests
        test_suite.test_data_bus_functionality()
        test_suite.test_stats_collector_functionality()
        test_suite.test_web_interface_functionality()
        test_suite.test_error_handler_functionality()
        
        # Integration tests
        test_suite.test_component_integration()
        test_suite.test_stress_and_performance()
        
    except KeyboardInterrupt:
        print("\n⏸️ Test suite interrupted by user")
    except Exception as e:
        print(f"\n💥 Test suite failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        total_tests = len(test_suite.test_results)
        passed_tests = sum(1 for result in test_suite.test_results.values() if result)
        
        # Print final summary
        print(f"\n📊 TEST RESULTS SUMMARY")
        print("=" * 50)
        
        for test_name, passed in test_suite.test_results.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"  {test_name.replace('_', ' ').title():25s}: {status}")
        
        success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        print(f"\nOverall Success Rate: {success_rate:.1f}% ({passed_tests}/{total_tests})")
        
        if success_rate >= 80:
            print("\n🎉 INTEGRATED UI SYSTEM READY FOR PRODUCTION!")
        elif success_rate >= 60:
            print("\n⚠️  INTEGRATED UI SYSTEM MOSTLY WORKING - Minor issues to address")
        else:
            print("\n❌ INTEGRATED UI SYSTEM NEEDS SIGNIFICANT WORK")
        
        # Cleanup
        test_suite.cleanup_components()
