"""
Comprehensive tests for monitoring/data_bus.py to improve coverage

Tests all functionality including component registration, pub/sub patterns,
error handling, and threading behavior.
"""

import pytest
import threading
import time
import queue
from unittest.mock import Mock, patch, MagicMock
from monitoring.data_bus import (
    DataType, DataMessage, DataBus, get_data_bus, shutdown_data_bus
)


class TestDataType:
    """Test DataType enum"""
    
    def test_data_type_values(self):
        """Test all DataType enum values"""
        assert DataType.GAME_STATE.value == "game_state"
        assert DataType.TRAINING_STATS.value == "training_stats"
        assert DataType.TRAINING_STATE.value == "training_state"
        assert DataType.TRAINING_CONTROL.value == "training_control"
        assert DataType.TRAINING_METRICS.value == "training_metrics"
        assert DataType.ACTION_TAKEN.value == "action_taken"
        assert DataType.LLM_DECISION.value == "llm_decision"
        assert DataType.SYSTEM_INFO.value == "system_info"
        assert DataType.GAME_SCREEN.value == "game_screen"
        assert DataType.ERROR_EVENT.value == "error_event"
        assert DataType.ERROR_NOTIFICATION.value == "error_notification"
        assert DataType.COMPONENT_STATUS.value == "component_status"


class TestDataMessage:
    """Test DataMessage dataclass"""
    
    def test_data_message_creation(self):
        """Test DataMessage creation"""
        message = DataMessage(
            data_type=DataType.GAME_STATE,
            timestamp=1234567890.0,
            data={"level": 5},
            source_component="trainer",
            message_id="msg_123"
        )
        
        assert message.data_type == DataType.GAME_STATE
        assert message.timestamp == 1234567890.0
        assert message.data == {"level": 5}
        assert message.source_component == "trainer"
        assert message.message_id == "msg_123"


class TestDataBus:
    """Test DataBus class"""
    
    def setup_method(self):
        """Set up fresh data bus for each test"""
        self.data_bus = DataBus()
    
    def teardown_method(self):
        """Clean up after each test"""
        if self.data_bus:
            self.data_bus.shutdown()
    
    def test_initialization(self):
        """Test DataBus initialization"""
        assert self.data_bus._active is True
        assert self.data_bus._running is True
        assert isinstance(self.data_bus._subscribers, dict)
        assert isinstance(self.data_bus._components, dict)
        assert isinstance(self.data_bus._lock, type(threading.Lock()))
        assert self.data_bus._logger.name == "data_bus"
    
    def test_register_component(self):
        """Test component registration"""
        metadata = {"type": "trainer", "version": "1.0"}
        self.data_bus.register_component("trainer_1", metadata)
        
        # Check component was registered
        status = self.data_bus.get_component_status()
        assert "trainer_1" in status
        assert status["trainer_1"]["type"] == "trainer"
        assert status["trainer_1"]["version"] == "1.0"
        assert "last_seen" in status["trainer_1"]
    
    def test_unregister_component(self):
        """Test component unregistration"""
        # Register first
        metadata = {"type": "trainer"}
        self.data_bus.register_component("trainer_1", metadata)
        
        # Verify registration
        status = self.data_bus.get_component_status()
        assert "trainer_1" in status
        
        # Unregister
        self.data_bus.unregister_component("trainer_1")
        
        # Verify removal
        status = self.data_bus.get_component_status()
        assert "trainer_1" not in status
    
    def test_unregister_nonexistent_component(self):
        """Test unregistering nonexistent component doesn't crash"""
        # Should not raise exception
        self.data_bus.unregister_component("nonexistent")
    
    def test_unregister_removes_subscriptions(self):
        """Test that unregistering component removes its subscriptions"""
        # Register component
        self.data_bus.register_component("test_comp", {"type": "test"})
        
        # Subscribe to data type
        self.data_bus.subscribe(DataType.GAME_STATE, "test_comp", callback=Mock())
        
        # Verify subscription exists
        assert DataType.GAME_STATE in self.data_bus._subscribers
        assert len(self.data_bus._subscribers[DataType.GAME_STATE]) == 1
        
        # Unregister component
        self.data_bus.unregister_component("test_comp")
        
        # Verify subscriptions removed
        assert len(self.data_bus._subscribers[DataType.GAME_STATE]) == 0
    
    def test_subscribe_with_callback(self):
        """Test subscribing with callback function"""
        callback = Mock()
        queue_result = self.data_bus.subscribe(
            DataType.GAME_STATE, "test_comp", callback=callback
        )
        
        # Should return None when callback provided
        assert queue_result is None
        
        # Check subscription was added
        assert DataType.GAME_STATE in self.data_bus._subscribers
        assert len(self.data_bus._subscribers[DataType.GAME_STATE]) == 1
        
        subscription = self.data_bus._subscribers[DataType.GAME_STATE][0]
        assert subscription['component_id'] == "test_comp"
        assert subscription['callback'] == callback
        assert subscription['queue'] is None
    
    def test_subscribe_with_queue(self):
        """Test subscribing with queue (no callback)"""
        result_queue = self.data_bus.subscribe(
            DataType.GAME_STATE, "test_comp", queue_size=50
        )
        
        # Should return queue when no callback
        assert isinstance(result_queue, queue.Queue)
        assert result_queue.maxsize == 50
        
        # Check subscription was added
        subscription = self.data_bus._subscribers[DataType.GAME_STATE][0]
        assert subscription['component_id'] == "test_comp"
        assert subscription['callback'] is None
        assert subscription['queue'] == result_queue
    
    def test_publish_to_callback_subscriber(self):
        """Test publishing data to callback subscriber"""
        callback = Mock()
        self.data_bus.register_component("publisher", {"type": "test"})
        self.data_bus.subscribe(DataType.GAME_STATE, "subscriber", callback=callback)
        
        # Publish data
        test_data = {"level": 5, "hp": 100}
        self.data_bus.publish(DataType.GAME_STATE, test_data, "publisher")
        
        # Callback should have been called with data
        callback.assert_called_once_with(test_data)
    
    def test_publish_to_callback_subscriber_with_nested_data(self):
        """Test publishing data with nested 'data' key to callback subscriber"""
        callback = Mock()
        self.data_bus.subscribe(DataType.GAME_STATE, "subscriber", callback=callback)
        
        # Publish data with nested 'data' key
        test_data = {"data": {"level": 5, "hp": 100}, "meta": "info"}
        self.data_bus.publish(DataType.GAME_STATE, test_data, "publisher")
        
        # Callback should have been called with nested data
        callback.assert_called_once_with({"level": 5, "hp": 100})
    
    def test_publish_to_queue_subscriber(self):
        """Test publishing data to queue subscriber"""
        result_queue = self.data_bus.subscribe(
            DataType.GAME_STATE, "subscriber", queue_size=10
        )
        self.data_bus.register_component("publisher", {"type": "test"})
        
        # Publish data
        test_data = {"level": 5}
        self.data_bus.publish(DataType.GAME_STATE, test_data, "publisher")
        
        # Check queue received data
        assert not result_queue.empty()
        message = result_queue.get_nowait()
        assert message['data'] == test_data
        assert message['publisher'] == "publisher"
        assert 'timestamp' in message
    
    def test_publish_to_full_queue(self):
        """Test publishing when queue is full"""
        result_queue = self.data_bus.subscribe(
            DataType.GAME_STATE, "subscriber", queue_size=2
        )
        
        # Fill queue to capacity
        self.data_bus.publish(DataType.GAME_STATE, {"msg": 1}, "pub")
        self.data_bus.publish(DataType.GAME_STATE, {"msg": 2}, "pub")
        
        # Queue should be full
        assert result_queue.full()
        
        # Publish one more - should evict oldest
        self.data_bus.publish(DataType.GAME_STATE, {"msg": 3}, "pub")
        
        # Queue should still have 2 items, but first should be evicted
        assert result_queue.qsize() == 2
        first_msg = result_queue.get_nowait()
        assert first_msg['data']['msg'] == 2  # First message evicted
        second_msg = result_queue.get_nowait()
        assert second_msg['data']['msg'] == 3
    
    def test_publish_callback_exception(self):
        """Test that callback exceptions are handled gracefully"""
        # Create callback that raises exception
        def failing_callback(data):
            raise ValueError("Callback failed")
        
        self.data_bus.subscribe(DataType.GAME_STATE, "subscriber", callback=failing_callback)
        
        # Should not raise exception
        self.data_bus.publish(DataType.GAME_STATE, {"test": True}, "publisher")
    
    def test_publish_when_inactive(self):
        """Test that publishing when inactive does nothing"""
        callback = Mock()
        self.data_bus.subscribe(DataType.GAME_STATE, "subscriber", callback=callback)
        
        # Mark as inactive
        self.data_bus._active = False
        
        # Publish data
        self.data_bus.publish(DataType.GAME_STATE, {"test": True}, "publisher")
        
        # Callback should not have been called
        callback.assert_not_called()
    
    def test_publish_no_subscribers(self):
        """Test publishing when no subscribers exist"""
        # Should not raise exception
        self.data_bus.publish(DataType.GAME_STATE, {"test": True}, "publisher")
    
    def test_publish_updates_component_heartbeat(self):
        """Test that publishing updates component last seen time"""
        self.data_bus.register_component("publisher", {"type": "test"})
        
        # Subscribe to the data type so publish won't return early
        callback = Mock()
        self.data_bus.subscribe(DataType.GAME_STATE, "subscriber", callback=callback)
        
        # Wait a bit to let some time elapse
        time.sleep(0.05)
        
        # Get initial last seen time (time elapsed since last seen)
        status1 = self.data_bus.get_component_status()
        initial_last_seen = status1["publisher"]["last_seen"]
        
        # Publish (this should update the heartbeat)
        self.data_bus.publish(DataType.GAME_STATE, {"test": True}, "publisher")
        
        # Check last seen was updated (should be less time since last seen now)
        status2 = self.data_bus.get_component_status()
        new_last_seen = status2["publisher"]["last_seen"]
        # After publishing, elapsed time should be smaller than before
        assert new_last_seen < initial_last_seen
    
    def test_update_component_heartbeat(self):
        """Test manual component heartbeat update"""
        self.data_bus.register_component("test_comp", {"type": "test"})
        
        # Wait to let some time elapse
        time.sleep(0.05)
        
        # Get initial last seen (time elapsed since last seen)
        status1 = self.data_bus.get_component_status()
        initial_last_seen = status1["test_comp"]["last_seen"]
        
        # Update heartbeat (this should reset the last seen time)
        self.data_bus.update_component_heartbeat("test_comp")
        
        # Check update (elapsed time should now be smaller)
        status2 = self.data_bus.get_component_status()
        new_last_seen = status2["test_comp"]["last_seen"]
        # After heartbeat update, elapsed time should be smaller
        assert new_last_seen < initial_last_seen
    
    def test_update_heartbeat_nonexistent_component(self):
        """Test updating heartbeat for nonexistent component"""
        # Should not raise exception
        self.data_bus.update_component_heartbeat("nonexistent")
    
    def test_get_component_status_empty(self):
        """Test getting status when no components registered"""
        status = self.data_bus.get_component_status()
        assert status == {}
    
    def test_shutdown(self):
        """Test data bus shutdown"""
        # Register some components and subscriptions
        self.data_bus.register_component("comp1", {"type": "test"})
        self.data_bus.subscribe(DataType.GAME_STATE, "comp1", callback=Mock())
        
        # Shutdown
        self.data_bus.shutdown()
        
        # Check cleanup
        assert not self.data_bus._active
        assert not self.data_bus._running
        assert len(self.data_bus._components) == 0
        assert len(self.data_bus._subscribers) == 0
    
    def test_shutdown_with_remaining_components(self):
        """Test shutdown when components remain registered"""
        # Register components
        self.data_bus.register_component("comp1", {"type": "test"})
        self.data_bus.register_component("comp2", {"type": "test"})
        
        # Shutdown should clean up components
        self.data_bus.shutdown()
        
        assert len(self.data_bus._components) == 0


class TestDataBusSingleton:
    """Test global data bus singleton functions"""
    
    def setup_method(self):
        """Reset global data bus before each test"""
        import monitoring.data_bus
        monitoring.data_bus._global_data_bus = None
    
    def teardown_method(self):
        """Clean up after each test"""
        try:
            shutdown_data_bus()
        except:
            pass
    
    def test_get_data_bus_singleton(self):
        """Test that get_data_bus returns singleton"""
        bus1 = get_data_bus()
        bus2 = get_data_bus()
        
        assert bus1 is bus2
        assert isinstance(bus1, DataBus)
    
    def test_get_data_bus_thread_safety(self):
        """Test that get_data_bus is thread safe"""
        buses = []
        
        def get_bus():
            buses.append(get_data_bus())
        
        # Create multiple threads
        threads = [threading.Thread(target=get_bus) for _ in range(5)]
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # All should be the same instance
        assert len(set(id(bus) for bus in buses)) == 1
    
    @patch('threading.Timer')
    def test_shutdown_data_bus_no_instance(self, mock_timer):
        """Test shutting down when no instance exists"""
        # Should not raise exception
        shutdown_data_bus()
        
        # Timer should still be created and cancelled
        mock_timer.assert_called_once()
        mock_timer.return_value.cancel.assert_called_once()
    
    @patch('threading.Timer')
    def test_shutdown_data_bus_with_instance(self, mock_timer):
        """Test shutting down existing instance"""
        # Create instance
        bus = get_data_bus()
        bus.register_component("test_comp", {"type": "test"})
        
        # Shutdown
        shutdown_data_bus()
        
        # Timer should be created and cancelled
        mock_timer.assert_called_once()
        mock_timer.return_value.cancel.assert_called_once()
        
        # Global instance should be cleared
        import monitoring.data_bus
        assert monitoring.data_bus._global_data_bus is None
    
    @patch('threading.Timer')
    def test_shutdown_data_bus_watchdog_timeout(self, mock_timer):
        """Test watchdog timer functionality during shutdown"""
        # Create instance
        bus = get_data_bus()
        
        # Mock timer to simulate timeout
        mock_timer_instance = Mock()
        mock_timer.return_value = mock_timer_instance
        
        # Make shutdown hang to trigger watchdog
        original_shutdown = bus.shutdown
        def hanging_shutdown():
            time.sleep(0.1)  # Simulate hanging
            original_shutdown()
        
        with patch.object(bus, 'shutdown', hanging_shutdown):
            # Manually trigger watchdog function
            shutdown_data_bus()
            
            # Get the watchdog function from timer call
            watchdog_func = mock_timer.call_args[0][1]
            watchdog_func()  # Execute watchdog
        
        # Should have cleared global instance
        import monitoring.data_bus
        assert monitoring.data_bus._global_data_bus is None


class TestDataBusIntegration:
    """Integration tests for DataBus functionality"""
    
    def test_full_pubsub_workflow(self):
        """Test complete publish-subscribe workflow"""
        bus = DataBus()
        
        try:
            # Register components
            bus.register_component("publisher", {"type": "trainer"})
            bus.register_component("subscriber", {"type": "monitor"})
            
            # Create subscriber with both callback and queue
            received_data = []
            def callback(data):
                received_data.append(data)
            
            bus.subscribe(DataType.TRAINING_STATS, "subscriber", callback=callback)
            result_queue = bus.subscribe(DataType.TRAINING_STATS, "subscriber2")
            
            # Publish data
            stats_data = {
                "episode": 100,
                "reward": 250.5,
                "steps": 1000
            }
            bus.publish(DataType.TRAINING_STATS, stats_data, "publisher")
            
            # Check callback received data
            assert len(received_data) == 1
            assert received_data[0] == stats_data
            
            # Check queue received data
            assert not result_queue.empty()
            queue_message = result_queue.get_nowait()
            assert queue_message['data'] == stats_data
            assert queue_message['publisher'] == "publisher"
            
            # Check component status
            status = bus.get_component_status()
            assert "publisher" in status
            assert "subscriber" in status
            
        finally:
            bus.shutdown()
    
    def test_multiple_data_types(self):
        """Test handling multiple data types simultaneously"""
        bus = DataBus()
        
        try:
            game_data = []
            training_data = []
            
            def game_callback(data):
                game_data.append(data)
            
            def training_callback(data):
                training_data.append(data)
            
            # Subscribe to different data types
            bus.subscribe(DataType.GAME_STATE, "monitor", callback=game_callback)
            bus.subscribe(DataType.TRAINING_STATS, "monitor", callback=training_callback)
            
            # Publish to both types
            bus.publish(DataType.GAME_STATE, {"level": 5}, "game")
            bus.publish(DataType.TRAINING_STATS, {"reward": 100}, "trainer")
            
            # Check both received correct data
            assert len(game_data) == 1
            assert game_data[0] == {"level": 5}
            
            assert len(training_data) == 1
            assert training_data[0] == {"reward": 100}
            
        finally:
            bus.shutdown()


if __name__ == "__main__":
    pytest.main([__file__])