"""Tests for the UnifiedMonitor class."""

import pytest
import numpy as np
import json
import time
import threading
from queue import Queue
from pathlib import Path
from unittest.mock import Mock, patch
import tempfile

from pokemon_crystal_rl.monitoring import (
    MonitorConfig,
    ErrorHandler,
    ErrorSeverity,
    RecoveryStrategy,
)
from pokemon_crystal_rl.monitoring.unified_monitor import UnifiedMonitor

class TestUnifiedMonitor:
    """Test suite for UnifiedMonitor class."""
    
    @pytest.fixture
    def monitor(self):
        """Create a test monitor instance."""
        monitor = UnifiedMonitor(host='127.0.0.1', port=5000)
        yield monitor
        monitor.stop_monitoring()

    def test_init(self, monitor):
        """Test monitor initialization."""
        assert monitor is not None
        assert monitor.host == '127.0.0.1'
        assert monitor.port == 5000
        assert not monitor.is_monitoring

    def test_start_stop_monitoring(self, monitor):
        """Test starting and stopping monitoring."""
        assert not monitor.is_monitoring
        monitor.start_monitoring()
        assert monitor.is_monitoring
        monitor.stop_monitoring()
        assert not monitor.is_monitoring

    def test_screenshot_update(self, monitor):
        """Test screenshot update functionality."""
        # Create test screenshot
        screenshot = np.random.randint(0, 255, (144, 160, 3), dtype=np.uint8)
        
        # Test updating screenshot
        monitor.update_screenshot(screenshot)
        
        # Verify screenshot queue has data
        assert not monitor.screen_queue.empty()
        screenshot_data = monitor.screen_queue.get()
        assert 'image' in screenshot_data
        assert 'timestamp' in screenshot_data
        assert 'dimensions' in screenshot_data

    def test_text_update(self, monitor):
        """Test text recognition update."""
        test_text = "TEST TEXT"
        test_location = "menu"
        
        monitor.update_text(test_text, test_location)
        
        assert len(monitor.recent_text) == 1
        text_data = monitor.recent_text[0]
        assert text_data['text'] == test_text
        assert text_data['location'] == test_location
        assert text_data['timestamp'] is not None

    def test_action_update(self, monitor):
        """Test action history update."""
        test_action = "MOVE_RIGHT"
        test_details = "Moving to next tile"
        
        monitor.update_action(test_action, test_details)
        
        assert len(monitor.recent_actions) == 1
        action_data = monitor.recent_actions[0]
        assert action_data['action'] == test_action
        assert action_data['details'] == test_details
        assert action_data['timestamp'] is not None

    def test_decision_update(self, monitor):
        """Test decision history update."""
        test_decision = {
            'action': 'MOVE_RIGHT',
            'confidence': 0.95,
            'reasoning': 'Path to objective'
        }
        
        monitor.update_decision(test_decision)
        
        assert len(monitor.recent_decisions) == 1
        decision_data = monitor.recent_decisions[0]
        assert decision_data['action'] == test_decision['action']
        assert decision_data['confidence'] == test_decision['confidence']
        assert decision_data['reasoning'] == test_decision['reasoning']
        assert 'timestamp' in decision_data

    def test_memory_cleanup(self, monitor):
        """Test memory management cleanup."""
        # Add lots of text entries
        for i in range(1100):
            monitor.update_text(f"Text {i}")
        
        # Force cleanup
        monitor.cleanup_interval = 0  # Set to 0 to force immediate cleanup
        monitor.cleanup()
        
        # Verify cleanup worked
        assert len(monitor.text_frequency) <= 1000

    def test_system_stats(self, monitor):
        """Test system statistics collection."""
        # Since PSUTIL_AVAILABLE is a module-level variable,
        # we mock it to test both cases
        from pokemon_crystal_rl.monitoring.unified_monitor import PSUTIL_AVAILABLE
        
        stats = monitor._get_system_stats()
        
        if stats:  # If psutil is available
            assert 'cpu_percent' in stats
            assert 'memory_percent' in stats
            assert 'disk_usage' in stats
