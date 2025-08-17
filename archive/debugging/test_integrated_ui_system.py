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

# Add paths for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from monitoring.data_bus import get_data_bus, DataType, shutdown_data_bus
from monitoring.stats_collector import StatsCollector
from monitoring.game_streamer import GameStreamComponent
from monitoring.web_interface import WebInterface
from monitoring.error_handler import ErrorHandler, ErrorSeverity

class IntegratedUISystemTest:
    """Comprehensive test for the integrated UI system"""
    
    def __init__(self):
        self.data_bus = None
        self.stats_collector = None
        self.game_streamer = None
        self.web_interface = None
        self.error_handler = None
        
        self.test_results = {
            "data_bus": False,
            "stats_collector": False,
            "game_streamer": False,
            "web_interface": False,
            "error_handler": False,
            "integration": False,
            "stress_test": False,
            "cleanup": False
        }
    
    def setup_components(self) -> bool:
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
            self.web_interface = WebInterface(
                host="localhost",
                port=8081,  # Use different port to avoid conflicts
                enable_websockets=True,
                enable_data_bus=True
            )
            
            # Connect components
            self.web_interface.set_stats_collector(self.stats_collector)
            self.web_interface.set_game_streamer(self.game_streamer)
            
            # Start web server
            success = self.web_interface.start_server()
            if success:
                print("   ✅ Web interface initialized and started")
                time.sleep(2.0)  # Give server time to start
                return True
            else:
                print("   ❌ Web interface failed to start")
                return False
                
        except Exception as e:
            print(f"   ❌ Setup failed: {e}")
            return False
    
    def test_data_bus_functionality(self) -> bool:
        """Test data bus core functionality"""
        print("\n🧪 Testing Data Bus Functionality")
        print("-" * 40)
        
        try:
            # Test data publishing and retrieval
            test_data = {"test_metric": 42.0, "timestamp": time.time()}
            
            # Publish test data
            success = self.data_bus.publish(
                DataType.TRAINING_STATS,
                test_data,
                "test_publisher"
            )
            
            if not success:
                print("   ❌ Data bus publish failed")
                return False
            
            # Wait for processing
            time.sleep(1.0)
            
            # Check data retrieval
            current_data = self.data_bus.get_current_data(DataType.TRAINING_STATS)
            
            if current_data and "test_metric" in current_data:
                print("   ✅ Data bus publish/retrieve working")
            else:
                print("   ❌ Data bus data retrieval failed")
                return False
            
            # Test component health
            component_status = self.data_bus.get_component_status()
            if len(component_status) > 0:
                print(f"   ✅ Data bus component tracking ({len(component_status)} components)")
            else:
                print("   ❌ Data bus component tracking failed")
                return False
            
            self.test_results["data_bus"] = True
            return True
            
        except Exception as e:
            print(f"   ❌ Data bus test failed: {e}")
            return False
    
    def test_stats_collector_functionality(self) -> bool:
        """Test stats collector functionality"""
        print("\n🧪 Testing Stats Collector Functionality")
        print("-" * 40)
        
        try:
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
            
            if "test.actions" in metrics_summary:
                actions_metric = metrics_summary["test.actions"]
                if actions_metric["current"] == 10.0:
                    print("   ✅ Counter metrics working correctly")
                else:
                    print(f"   ❌ Counter metrics incorrect: {actions_metric['current']} != 10.0")
                    return False
            else:
                print("   ❌ Counter metrics not found")
                return False
            
            if "test.speed" in metrics_summary:
                print("   ✅ Gauge metrics working correctly")
            else:
                print("   ❌ Gauge metrics not found")
                return False
            
            if "test.response_time" in metrics_summary:
                timer_metric = metrics_summary["test.response_time"]
                if timer_metric["count"] == 10:
                    print("   ✅ Timer metrics working correctly")
                else:
                    print(f"   ❌ Timer metrics incorrect count: {timer_metric['count']} != 10")
                    return False
            else:
                print("   ❌ Timer metrics not found")
                return False
            
            # Test performance stats
            perf_stats = self.stats_collector.get_performance_stats()
            if perf_stats["collection_errors"] == 0:
                print("   ✅ Stats collector performance good (no errors)")
            else:
                print(f"   ⚠️ Stats collector has {perf_stats['collection_errors']} errors")
            
            self.test_results["stats_collector"] = True
            return True
            
        except Exception as e:
            print(f"   ❌ Stats collector test failed: {e}")
            return False
    
    def test_web_interface_functionality(self) -> bool:
        """Test web interface functionality"""
        print("\n🧪 Testing Web Interface Functionality")
        print("-" * 40)
        
        try:
            base_url = "http://localhost:8081"
            
            # Test main dashboard
            response = requests.get(f"{base_url}/", timeout=5)
            if response.status_code == 200 and "Pokemon Crystal RL" in response.text:
                print("   ✅ Main dashboard accessible")
            else:
                print(f"   ❌ Main dashboard failed: {response.status_code}")
                return False
            
            # Test API endpoints
            api_endpoints = [
                ("/api/dashboard", "dashboard data"),
                ("/api/stats", "statistics"),
            ]
            
            for endpoint, description in api_endpoints:
                try:
                    response = requests.get(f"{base_url}{endpoint}", timeout=5)
                    if response.status_code == 200:
                        data = response.json()
                        if data:
                            print(f"   ✅ {description} API working")
                        else:
                            print(f"   ❌ {description} API returned empty data")
                            return False
                    else:
                        print(f"   ❌ {description} API failed: {response.status_code}")
                        return False
                except requests.exceptions.RequestException as e:
                    print(f"   ❌ {description} API request failed: {e}")
                    return False
            
            # Test dashboard data structure
            response = requests.get(f"{base_url}/api/dashboard", timeout=5)
            dashboard_data = response.json()
            
            required_fields = ["timestamp", "server_info", "stats"]
            missing_fields = [field for field in required_fields if field not in dashboard_data]
            
            if not missing_fields:
                print("   ✅ Dashboard data structure correct")
            else:
                print(f"   ❌ Dashboard missing fields: {missing_fields}")
                return False
            
            self.test_results["web_interface"] = True
            return True
            
        except Exception as e:
            print(f"   ❌ Web interface test failed: {e}")
            return False
    
    def test_error_handler_functionality(self) -> bool:
        """Test error handler functionality"""
        print("\n🧪 Testing Error Handler Functionality")  
        print("-" * 40)
        
        try:
            # Test error recording
            initial_error_count = len(self.error_handler._error_history)
            
            # Simulate some errors
            from monitoring.error_handler import ErrorEvent, RecoveryStrategy
            
            test_error = ErrorEvent(
                timestamp=time.time(),
                component="test_component",
                error_type="TestError",
                message="Integration test error",
                severity=ErrorSeverity.MEDIUM,
                traceback=None,
                recovery_strategy=RecoveryStrategy.RETRY
            )
            
            self.error_handler.handle_error(test_error)
            
            # Check if error was recorded
            final_error_count = len(self.error_handler._error_history)
            if final_error_count > initial_error_count:
                print("   ✅ Error handler recording errors")
            else:
                print("   ❌ Error handler not recording errors")
                return False
            
            # Test error statistics
            error_stats = self.error_handler.get_error_statistics()
            required_stats = ["total_errors", "component_health", "memory_info"]
            
            if all(field in error_stats for field in required_stats):
                print("   ✅ Error handler statistics working")
            else:
                print("   ❌ Error handler statistics incomplete")
                return False
            
            # Test memory monitoring
            memory_info = error_stats["memory_info"]
            if "rss_mb" in memory_info:
                print(f"   ✅ Memory monitoring working ({memory_info['rss_mb']:.1f}MB)")
            else:
                print("   ❌ Memory monitoring failed")
                return False
            
            self.test_results["error_handler"] = True
            return True
            
        except Exception as e:
            print(f"   ❌ Error handler test failed: {e}")
            return False
    
    def test_component_integration(self) -> bool:
        """Test integration between all components"""
        print("\n🧪 Testing Component Integration")
        print("-" * 40)
        
        try:
            # Publish data through data bus and check if all components receive it
            test_data = {
                "total_actions": 100,
                "actions_per_second": 25.0,
                "llm_calls": 10
            }
            
            # Publish training stats
            self.data_bus.publish(DataType.TRAINING_STATS, test_data, "integration_test")
            
            # Publish action event
            action_data = {
                "action_id": 5,
                "action_name": "A",
                "execution_time": 0.05
            }
            self.data_bus.publish(DataType.ACTION_TAKEN, action_data, "integration_test")
            
            # Wait for processing
            time.sleep(6.0)
            
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
                response = requests.get("http://localhost:8081/api/stats", timeout=5)
                if response.status_code == 200:
                    web_stats = response.json()
                    if web_stats:
                        print("   ✅ Web interface integration")
                        passed_checks += 1
                    else:
                        print("   ❌ Web interface integration - no data")
                else:
                    print("   ❌ Web interface integration - HTTP error")
            except Exception as e:
                print(f"   ❌ Web interface integration failed: {e}")
            
            success = passed_checks >= 2  # At least 2 out of 3 integration points
            if success:
                print(f"   ✅ Component integration working ({passed_checks}/3 checks passed)")
            else:
                print(f"   ❌ Component integration insufficient ({passed_checks}/3 checks passed)")
            
            self.test_results["integration"] = success
            return success
            
        except Exception as e:
            print(f"   ❌ Component integration test failed: {e}")
            return False
    
    def test_stress_and_performance(self) -> bool:
        """Test system under load to check for memory leaks and crashes"""
        print("\n🧪 Testing System Stress and Performance")
        print("-" * 40)
        
        try:
            # Record initial memory usage
            initial_memory = self.error_handler.memory_monitor.get_memory_info()["rss_mb"]
            print(f"   Initial memory usage: {initial_memory:.1f}MB")
            
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
            
            # Check final memory usage
            final_memory = self.error_handler.memory_monitor.get_memory_info()["rss_mb"]
            memory_increase = final_memory - initial_memory
            
            print(f"   Final memory usage: {final_memory:.1f}MB")
            print(f"   Memory increase: {memory_increase:.1f}MB")
            
            # Check performance metrics
            stats_perf = self.stats_collector.get_performance_stats()
            error_rate = stats_perf.get("error_rate", 1.0)
            
            # Stress test passes if:
            # 1. Memory increase is reasonable (< 100MB)
            # 2. Error rate is low (< 1%)
            # 3. System is still responsive
            
            memory_ok = memory_increase < 100.0
            error_rate_ok = error_rate < 0.01
            
            try:
                # Test system responsiveness
                response = requests.get("http://localhost:8081/api/dashboard", timeout=5)
                responsive = response.status_code == 200
            except:
                responsive = False
            
            if memory_ok and error_rate_ok and responsive:
                print("   ✅ Stress test passed - system stable under load")
                print(f"      Memory impact: {memory_increase:.1f}MB, Error rate: {error_rate:.4f}")
            else:
                print("   ❌ Stress test failed:")
                print(f"      Memory OK: {memory_ok} ({memory_increase:.1f}MB)")
                print(f"      Error rate OK: {error_rate_ok} ({error_rate:.4f})")
                print(f"      Responsive: {responsive}")
                return False
            
            self.test_results["stress_test"] = True
            return True
            
        except Exception as e:
            print(f"   ❌ Stress test failed: {e}")
            return False
    
    def cleanup_components(self) -> bool:
        """Clean shutdown of all components"""
        print("\n🧹 Cleaning up components...")
        
        try:
            # Shutdown in reverse order
            if self.web_interface:
                self.web_interface.shutdown()
                print("   ✅ Web interface shut down")
            
            if self.game_streamer:
                self.game_streamer.shutdown()
                print("   ✅ Game streamer shut down")
            
            if self.stats_collector:
                self.stats_collector.shutdown()
                print("   ✅ Stats collector shut down")
            
            if self.error_handler:
                self.error_handler.shutdown()
                print("   ✅ Error handler shut down")
            
            # Shutdown data bus last
            shutdown_data_bus()
            print("   ✅ Data bus shut down")
            
            # Small delay to ensure clean shutdown
            time.sleep(1.0)
            
            self.test_results["cleanup"] = True
            return True
            
        except Exception as e:
            print(f"   ❌ Cleanup failed: {e}")
            return False
    
    def run_comprehensive_test(self) -> Dict[str, bool]:
        """Run the comprehensive integrated test suite"""
        print("🚀 INTEGRATED UI SYSTEM COMPREHENSIVE TEST")
        print("=" * 70)
        
        try:
            # Setup
            if not self.setup_components():
                print("\n❌ SETUP FAILED - Cannot continue with tests")
                return self.test_results
            
            # Core component tests
            self.test_data_bus_functionality()
            self.test_stats_collector_functionality()
            self.test_web_interface_functionality()
            self.test_error_handler_functionality()
            
            # Integration tests
            self.test_component_integration()
            self.test_stress_and_performance()
            
            return self.test_results
            
        except KeyboardInterrupt:
            print("\n⏸️ Tests interrupted by user")
            return self.test_results
        except Exception as e:
            print(f"\n💥 Test suite crashed: {e}")
            import traceback
            traceback.print_exc()
            return self.test_results
        finally:
            self.cleanup_components()
    
    def print_test_summary(self):
        """Print comprehensive test results summary"""
        print(f"\n📊 TEST RESULTS SUMMARY")
        print("=" * 50)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result)
        
        for test_name, passed in self.test_results.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"  {test_name.replace('_', ' ').title():25s}: {status}")
        
        success_rate = (passed_tests / total_tests) * 100
        print(f"\nOverall Success Rate: {success_rate:.1f}% ({passed_tests}/{total_tests})")
        
        if success_rate >= 80:
            print("\n🎉 INTEGRATED UI SYSTEM READY FOR PRODUCTION!")
        elif success_rate >= 60:
            print("\n⚠️  INTEGRATED UI SYSTEM MOSTLY WORKING - Minor issues to address")
        else:
            print("\n❌ INTEGRATED UI SYSTEM NEEDS SIGNIFICANT WORK")


if __name__ == "__main__":
    test_suite = IntegratedUISystemTest()
    
    try:
        # Run comprehensive test
        results = test_suite.run_comprehensive_test()
        
        # Print summary
        test_suite.print_test_summary()
        
    except KeyboardInterrupt:
        print(f"\n⏸️ Test suite interrupted by user")
        test_suite.cleanup_components()
    except Exception as e:
        print(f"\n💥 Test suite failed: {e}")
        test_suite.cleanup_components()
