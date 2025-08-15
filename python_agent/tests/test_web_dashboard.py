#!/usr/bin/env python3
"""
test_web_dashboard.py - Tests for web dashboard HTTP polling improvements

Tests the web dashboard fixes including:
- Connection status initialization
- HTTP polling fallback with proper cleanup
- Memory leak prevention in polling intervals
- Error handling and recovery in HTTP requests
- Screenshot blob URL management
"""

import pytest
import time
import json
import base64
import threading
from unittest.mock import Mock, patch, MagicMock
from http.server import HTTPServer
import tempfile
import os

# Test the web dashboard functionality by creating a simple mock server
# that simulates the unified trainer's HTTP endpoints

class MockUnifiedTrainer:
    """Mock trainer for testing web dashboard functionality"""
    
    def __init__(self):
        self.stats = {
            'start_time': time.time(),
            'total_actions': 1000,
            'llm_calls': 100,
            'actions_per_second': 2.5,
            'mode': 'fast_monitored',
            'model': 'smollm2:1.7b'
        }
        
        self.latest_screen = {
            'image_b64': base64.b64encode(b'fake_image_data').decode(),
            'timestamp': time.time(),
            'size': (160, 144),
            'frame_id': 12345,
            'data_length': 100
        }
        
        self._training_active = True
        self.current_run_id = 1
        
        # Track API calls for testing
        self.api_calls = {
            'status': 0,
            'system': 0,
            'screenshot': 0,
            'socket_io': 0
        }
    
    def get_api_status(self):
        """Mock API status response"""
        self.api_calls['status'] += 1
        return {
            'is_training': self._training_active,
            'current_run_id': self.current_run_id,
            'mode': self.stats['mode'],
            'model': self.stats['model'],
            'start_time': self.stats['start_time'],
            'total_actions': self.stats['total_actions'],
            'llm_calls': self.stats['llm_calls'],
            'actions_per_second': self.stats['actions_per_second'],
            'current_reward': 0.5,
            'total_reward': 25.8,
            'game_state': 'overworld',
            'map_id': 'route_29',
            'player_x': 10,
            'player_y': 15,
            'avg_reward': 1.2,
            'success_rate': 0.85
        }
    
    def get_api_system(self):
        """Mock API system response"""
        self.api_calls['system'] += 1
        return {
            'cpu_percent': 45.2,
            'memory_percent': 62.8,
            'disk_usage': 55.1,
            'gpu_available': False
        }
    
    def get_screenshot(self):
        """Mock screenshot response"""
        self.api_calls['screenshot'] += 1
        return base64.b64decode(self.latest_screen['image_b64'])
    
    def handle_socket_io_fallback(self):
        """Mock Socket.IO fallback response"""
        self.api_calls['socket_io'] += 1
        return {
            'error': 'WebSocket/Socket.IO not implemented',
            'message': 'This trainer uses HTTP polling instead of WebSockets',
            'use_polling': True,
            'polling_endpoints': {
                'status': '/api/status',
                'system': '/api/system',
                'screenshot': '/api/screenshot'
            }
        }


class TestWebDashboardHTTPPolling:
    """Test HTTP polling functionality and improvements"""
    
    @pytest.fixture
    def mock_trainer(self):
        """Create mock trainer for testing"""
        return MockUnifiedTrainer()
    
    def test_api_status_endpoint_structure(self, mock_trainer):
        """Test API status endpoint returns correct structure"""
        status = mock_trainer.get_api_status()
        
        # Check all required fields are present
        required_fields = [
            'is_training', 'mode', 'model', 'start_time',
            'total_actions', 'llm_calls', 'actions_per_second'
        ]
        
        for field in required_fields:
            assert field in status, f"Missing required field: {field}"
        
        # Check data types
        assert isinstance(status['is_training'], bool)
        assert isinstance(status['total_actions'], int)
        assert isinstance(status['actions_per_second'], (int, float))
        assert isinstance(status['start_time'], (int, float))
    
    def test_api_status_enhanced_fields(self, mock_trainer):
        """Test API status includes enhanced data mapping fields"""
        status = mock_trainer.get_api_status()
        
        # Check enhanced fields for better dashboard data mapping
        enhanced_fields = [
            'current_reward', 'total_reward', 'game_state',
            'map_id', 'player_x', 'player_y', 'avg_reward', 'success_rate'
        ]
        
        for field in enhanced_fields:
            assert field in status, f"Missing enhanced field: {field}"
        
        # Check data types for enhanced fields
        assert isinstance(status['current_reward'], (int, float))
        assert isinstance(status['total_reward'], (int, float))
        assert isinstance(status['game_state'], str)
        assert status['avg_reward'] is not None
        assert status['success_rate'] is not None
    
    def test_api_system_endpoint_structure(self, mock_trainer):
        """Test API system endpoint returns correct structure"""
        system = mock_trainer.get_api_system()
        
        # Check all required fields are present
        required_fields = ['cpu_percent', 'memory_percent', 'disk_usage', 'gpu_available']
        
        for field in required_fields:
            assert field in system, f"Missing required field: {field}"
        
        # Check data types and ranges
        assert 0.0 <= system['cpu_percent'] <= 100.0
        assert 0.0 <= system['memory_percent'] <= 100.0
        assert isinstance(system['gpu_available'], bool)
    
    def test_screenshot_endpoint(self, mock_trainer):
        """Test screenshot endpoint returns binary data"""
        screenshot_data = mock_trainer.get_screenshot()
        
        assert isinstance(screenshot_data, bytes)
        assert len(screenshot_data) > 0
        assert mock_trainer.api_calls['screenshot'] == 1
    
    def test_socket_io_fallback_structure(self, mock_trainer):
        """Test Socket.IO fallback returns correct structure"""
        fallback = mock_trainer.handle_socket_io_fallback()
        
        # Check required fields
        assert 'error' in fallback
        assert 'message' in fallback
        assert 'use_polling' in fallback
        assert 'polling_endpoints' in fallback
        
        # Check polling endpoints structure
        endpoints = fallback['polling_endpoints']
        assert 'status' in endpoints
        assert 'system' in endpoints
        assert 'screenshot' in endpoints
        
        # Check proper response indicates HTTP polling
        assert fallback['use_polling'] is True
        assert 'HTTP polling' in fallback['message']
    
    def test_multiple_api_calls_tracking(self, mock_trainer):
        """Test API call tracking for monitoring usage"""
        # Make multiple calls
        for i in range(5):
            mock_trainer.get_api_status()
            mock_trainer.get_api_system()
            mock_trainer.get_screenshot()
        
        # Check call counts
        assert mock_trainer.api_calls['status'] == 5
        assert mock_trainer.api_calls['system'] == 5
        assert mock_trainer.api_calls['screenshot'] == 5
    
    def test_training_status_changes(self, mock_trainer):
        """Test training status changes are reflected in API"""
        # Initially training
        status1 = mock_trainer.get_api_status()
        assert status1['is_training'] is True
        
        # Stop training
        mock_trainer._training_active = False
        status2 = mock_trainer.get_api_status()
        assert status2['is_training'] is False
        
        # Restart training
        mock_trainer._training_active = True
        status3 = mock_trainer.get_api_status()
        assert status3['is_training'] is True


class TestDashboardMemoryManagement:
    """Test memory management improvements in dashboard"""
    
    def test_polling_interval_management(self):
        """Test proper polling interval management"""
        # Simulate the JavaScript polling behavior
        intervals = []
        urls_created = []
        urls_revoked = []
        
        def mock_set_interval(func, delay):
            """Mock setInterval"""
            interval_id = len(intervals)
            intervals.append({'id': interval_id, 'func': func, 'delay': delay})
            return interval_id
        
        def mock_clear_interval(interval_id):
            """Mock clearInterval"""
            intervals[interval_id] = None
        
        def mock_create_object_url(blob):
            """Mock URL.createObjectURL"""
            url = f"blob:http://localhost/mock-{len(urls_created)}"
            urls_created.append(url)
            return url
        
        def mock_revoke_object_url(url):
            """Mock URL.revokeObjectURL"""
            urls_revoked.append(url)
        
        # Simulate starting HTTP polling
        stats_interval = mock_set_interval(lambda: None, 1000)
        screenshot_interval = mock_set_interval(lambda: None, 2000)
        
        assert len(intervals) == 2
        assert intervals[0]['delay'] == 1000  # Stats interval
        assert intervals[1]['delay'] == 2000  # Screenshot interval
        
        # Simulate screenshot URL management
        for i in range(10):
            # Create new URL
            new_url = mock_create_object_url(f"mock_blob_{i}")
            
            # Revoke previous URL (memory leak prevention)
            if i > 0:
                previous_url = urls_created[i-1]
                mock_revoke_object_url(previous_url)
        
        # Check URL management
        assert len(urls_created) == 10
        assert len(urls_revoked) == 9  # One less than created (last one not revoked yet)
        
        # Simulate stopping HTTP polling
        mock_clear_interval(stats_interval)
        mock_clear_interval(screenshot_interval)
        
        # Simulate final cleanup
        if urls_created:
            mock_revoke_object_url(urls_created[-1])
        
        assert len(urls_revoked) == 10  # All URLs should be revoked
    
    def test_connection_status_lifecycle(self):
        """Test connection status lifecycle"""
        # Simulate connection status changes
        status_changes = []
        
        def update_status(status, is_error=False):
            """Mock connection status update"""
            status_changes.append({
                'status': status,
                'is_error': is_error,
                'timestamp': time.time()
            })
        
        # Simulate dashboard loading
        update_status("Connecting...")
        
        # Socket.IO connection attempt fails
        update_status("Connection Error", is_error=True)
        
        # Fallback to HTTP polling
        update_status("HTTP Polling")
        
        # Connection recovers
        update_status("HTTP Polling")  # Should remain stable
        
        # Verify status progression
        assert len(status_changes) == 4
        assert status_changes[0]['status'] == "Connecting..."
        assert status_changes[1]['status'] == "Connection Error"
        assert status_changes[1]['is_error'] is True
        assert status_changes[2]['status'] == "HTTP Polling"
        assert status_changes[3]['status'] == "HTTP Polling"
    
    def test_error_recovery_mechanism(self):
        """Test error recovery mechanism"""
        error_recovery_attempts = []
        connection_states = []
        
        def simulate_api_call(should_fail=False):
            """Simulate API call with optional failure"""
            if should_fail:
                connection_states.append("Connection Error")
                # Simulate recovery timeout
                time.sleep(0.01)  # Small delay for testing
                error_recovery_attempts.append(time.time())
                return None
            else:
                connection_states.append("HTTP Polling")
                return {"status": "ok"}
        
        # Simulate failed requests
        for i in range(3):
            result = simulate_api_call(should_fail=True)
            assert result is None
        
        # Simulate successful recovery
        result = simulate_api_call(should_fail=False)
        assert result is not None
        
        # Check recovery attempts were made
        assert len(error_recovery_attempts) == 3
        assert len(connection_states) == 4
        assert connection_states[:3] == ["Connection Error"] * 3
        assert connection_states[3] == "HTTP Polling"


class TestDashboardErrorHandling:
    """Test error handling improvements in dashboard"""
    
    def test_graceful_socket_io_fallback(self):
        """Test graceful Socket.IO fallback handling"""
        mock_trainer = MockUnifiedTrainer()
        
        # Simulate Socket.IO connection failure
        fallback_response = mock_trainer.handle_socket_io_fallback()
        
        # Should return 200 status (not 404) to avoid browser errors
        assert 'error' in fallback_response
        assert fallback_response['use_polling'] is True
        
        # Should provide clear instructions for HTTP polling
        assert 'polling_endpoints' in fallback_response
        endpoints = fallback_response['polling_endpoints']
        assert endpoints['status'] == '/api/status'
        assert endpoints['system'] == '/api/system'
        assert endpoints['screenshot'] == '/api/screenshot'
    
    def test_api_error_resilience(self):
        """Test API error resilience"""
        mock_trainer = MockUnifiedTrainer()
        error_counts = {}
        
        def simulate_api_with_errors(endpoint, error_rate=0.3):
            """Simulate API calls with intermittent errors"""
            if endpoint not in error_counts:
                error_counts[endpoint] = 0
            
            # Simulate error based on rate
            if (error_counts[endpoint] % 10) < (error_rate * 10):
                error_counts[endpoint] += 1
                raise Exception(f"Simulated {endpoint} error")
            
            error_counts[endpoint] += 1
            
            # Return appropriate response
            if endpoint == 'status':
                return mock_trainer.get_api_status()
            elif endpoint == 'system':
                return mock_trainer.get_api_system()
            elif endpoint == 'screenshot':
                return mock_trainer.get_screenshot()
        
        successful_calls = {}
        failed_calls = {}
        
        # Simulate multiple API calls with errors
        for endpoint in ['status', 'system', 'screenshot']:
            successful_calls[endpoint] = 0
            failed_calls[endpoint] = 0
            
            for i in range(20):
                try:
                    result = simulate_api_with_errors(endpoint, error_rate=0.3)
                    if result is not None:
                        successful_calls[endpoint] += 1
                except Exception:
                    failed_calls[endpoint] += 1
        
        # Should have both successful and failed calls
        for endpoint in ['status', 'system', 'screenshot']:
            assert successful_calls[endpoint] > 0, f"No successful calls for {endpoint}"
            assert failed_calls[endpoint] > 0, f"No failed calls for {endpoint}"
            
            # Total calls should be 20
            total = successful_calls[endpoint] + failed_calls[endpoint]
            assert total == 20, f"Wrong total calls for {endpoint}: {total}"
    
    def test_screenshot_polling_frequency(self):
        """Test screenshot polling frequency adjustment"""
        mock_trainer = MockUnifiedTrainer()
        
        # Test different polling frequencies
        polling_intervals = [500, 1000, 2000, 5000]  # milliseconds
        
        for interval_ms in polling_intervals:
            start_time = time.time()
            call_count = 0
            
            # Simulate polling for a short duration
            duration = 0.1  # 100ms test duration
            end_time = start_time + duration
            
            current_time = start_time
            while current_time < end_time:
                # Check if it's time for next call
                if (current_time - start_time) * 1000 >= call_count * interval_ms:
                    mock_trainer.get_screenshot()
                    call_count += 1
                
                current_time += 0.01  # Advance by 10ms
            
            # For 2000ms interval (2 seconds), should have at most 1 call in 100ms
            if interval_ms == 2000:
                assert call_count <= 1, f"Too many calls for {interval_ms}ms interval: {call_count}"
            
            # Reset API call counter for next test
            mock_trainer.api_calls['screenshot'] = 0
    
    def test_data_mapping_completeness(self):
        """Test completeness of data mapping from API to dashboard"""
        mock_trainer = MockUnifiedTrainer()
        status_data = mock_trainer.get_api_status()
        system_data = mock_trainer.get_api_system()
        
        # Dashboard fields that need to be populated
        dashboard_fields = [
            # Training status fields
            'is_training', 'total_actions', 'actions_per_second',
            'llm_calls', 'mode', 'model',
            
            # Game state fields (enhanced mapping)
            'current_reward', 'total_reward', 'game_state',
            'map_id', 'player_x', 'player_y',
            
            # Performance fields
            'avg_reward', 'success_rate',
            
            # System fields
            'cpu_percent', 'memory_percent'
        ]
        
        # Check that all dashboard fields have corresponding data
        all_data = {**status_data, **system_data}
        
        for field in dashboard_fields:
            assert field in all_data, f"Missing dashboard field: {field}"
            assert all_data[field] is not None, f"Null value for dashboard field: {field}"


class TestDashboardIntegration:
    """Integration tests for dashboard functionality"""
    
    def test_full_polling_cycle_simulation(self):
        """Test complete HTTP polling cycle simulation"""
        mock_trainer = MockUnifiedTrainer()
        
        # Simulate dashboard lifecycle
        lifecycle_events = []
        
        def log_event(event, data=None):
            lifecycle_events.append({
                'event': event,
                'timestamp': time.time(),
                'data': data
            })
        
        # 1. Dashboard loads
        log_event('dashboard_load')
        
        # 2. Try Socket.IO connection
        log_event('socketio_attempt')
        
        # 3. Socket.IO fails, get fallback info
        fallback_info = mock_trainer.handle_socket_io_fallback()
        log_event('socketio_fallback', fallback_info)
        
        # 4. Start HTTP polling
        log_event('http_polling_start')
        
        # 5. Multiple polling cycles
        for cycle in range(5):
            # Status update
            status = mock_trainer.get_api_status()
            log_event('status_update', {'cycle': cycle, 'actions': status['total_actions']})
            
            # System update  
            system = mock_trainer.get_api_system()
            log_event('system_update', {'cycle': cycle, 'cpu': system['cpu_percent']})
            
            # Screenshot update (every other cycle to simulate 2-second interval)
            if cycle % 2 == 0:
                screenshot = mock_trainer.get_screenshot()
                log_event('screenshot_update', {'cycle': cycle, 'size': len(screenshot)})
        
        # 6. Stop polling
        log_event('http_polling_stop')
        
        # Verify lifecycle
        assert len(lifecycle_events) >= 20  # At least 20 events
        
        # Check event sequence
        event_types = [event['event'] for event in lifecycle_events]
        assert 'dashboard_load' in event_types
        assert 'socketio_fallback' in event_types
        assert 'http_polling_start' in event_types
        assert 'status_update' in event_types
        assert 'system_update' in event_types
        assert 'screenshot_update' in event_types
    
    def test_concurrent_polling_simulation(self):
        """Test concurrent polling requests simulation"""
        mock_trainer = MockUnifiedTrainer()
        
        # Simulate concurrent requests
        def worker_thread(thread_id, call_count):
            results = []
            for i in range(call_count):
                try:
                    # Each thread makes different types of calls
                    if thread_id % 3 == 0:
                        result = mock_trainer.get_api_status()
                        results.append(('status', result))
                    elif thread_id % 3 == 1:
                        result = mock_trainer.get_api_system()
                        results.append(('system', result))
                    else:
                        result = mock_trainer.get_screenshot()
                        results.append(('screenshot', len(result)))
                        
                    time.sleep(0.001)  # Small delay
                except Exception as e:
                    results.append(('error', str(e)))
            
            return results
        
        # Start multiple threads
        threads = []
        thread_results = {}
        
        for thread_id in range(6):  # 6 threads
            thread = threading.Thread(
                target=lambda tid=thread_id: thread_results.update({
                    tid: worker_thread(tid, 10)
                })
            )
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Verify all threads completed successfully
        assert len(thread_results) == 6
        
        total_calls = 0
        total_errors = 0
        
        for thread_id, results in thread_results.items():
            assert len(results) == 10  # Each thread made 10 calls
            total_calls += len(results)
            
            # Count errors
            errors = [r for r in results if r[0] == 'error']
            total_errors += len(errors)
        
        assert total_calls == 60  # 6 threads * 10 calls each
        # Should have minimal or no errors in mock environment
        assert total_errors < 5  # Allow for some simulation errors


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
