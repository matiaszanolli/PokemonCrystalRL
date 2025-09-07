#!/usr/bin/env python3
"""
Standalone test to validate the mock fix for UnifiedMonitor.
"""

import sys
import tempfile
import os
from unittest.mock import Mock, patch

# Add the project root to path
sys.path.insert(0, '.')

def test_unified_monitor_mock_fix():
    """Test that the mock fix works for UnifiedMonitor initialization."""
    print("Testing UnifiedMonitor mock fix...")
    
    from config.config import MonitorConfig
    from monitoring.unified_monitor import TrainingState
    
    # Create a temporary directory
    tmp_path = tempfile.mkdtemp()
    
    config = MonitorConfig(
        db_path=os.path.join(tmp_path, 'test.db'),
        static_dir=os.path.join(tmp_path, 'static'),
        port=8000
    )
    
    # Mock Flask and SocketIO to avoid hanging in tests
    with patch('monitoring.unified_monitor.Flask') as mock_flask, \
         patch('monitoring.unified_monitor.SocketIO') as mock_socketio:
        
        mock_app = Mock()
        # Configure the mock app to have a proper config dictionary
        mock_app.config = {}
        mock_flask.return_value = mock_app
        mock_socketio.return_value = Mock()
        
        from monitoring.unified_monitor import UnifiedMonitor
        monitor = UnifiedMonitor(config=config)
        
        # Verify database initialization
        assert monitor.db is not None, 'Database should be initialized'
        assert monitor.current_run_id is None, 'Run ID should be None initially'
        assert monitor.training_state == TrainingState.INITIALIZING, 'Training state should be INITIALIZING'
        
        # Clean up
        monitor.stop_monitoring()
        
        print('✅ All assertions passed!')
        print(f'Database: {monitor.db}')
        print(f'Current run ID: {monitor.current_run_id}')
        print(f'Training state: {monitor.training_state}')
        
        return True

def test_unified_monitor_without_config():
    """Test UnifiedMonitor without config (original fixture behavior)."""
    print("Testing UnifiedMonitor without config...")
    
    # Mock Flask and SocketIO for all tests using this fixture
    with patch('monitoring.unified_monitor.Flask') as mock_flask, \
         patch('monitoring.unified_monitor.SocketIO') as mock_socketio:
        
        mock_app = Mock()
        mock_app.config = {}
        mock_flask.return_value = mock_app
        mock_socketio.return_value = Mock()
        
        from monitoring.unified_monitor import UnifiedMonitor
        monitor = UnifiedMonitor(host='127.0.0.1', port=5000)
        
        # Basic checks
        assert monitor is not None
        assert monitor.host == '127.0.0.1'
        assert monitor.port == 5000
        assert not monitor.is_monitoring
        
        monitor.stop_monitoring()
        
        print('✅ Basic monitor test passed!')
        return True

if __name__ == '__main__':
    try:
        print("=" * 60)
        print("TESTING UNIFIED MONITOR MOCK FIX")
        print("=" * 60)
        
        # Test 1: Database integration test
        test_unified_monitor_mock_fix()
        print()
        
        # Test 2: Basic monitor test
        test_unified_monitor_without_config()
        print()
        
        print("=" * 60)
        print("🎉 ALL TESTS PASSED! The mock fix is working correctly.")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)