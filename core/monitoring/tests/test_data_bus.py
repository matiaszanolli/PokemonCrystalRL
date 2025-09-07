"""
Tests for Data Bus

This module provides comprehensive tests for the DataBus implementation,
including validation, error handling, and performance monitoring.
"""

import pytest
from unittest.mock import Mock, call
import threading
import time

from ..data.bus import DataBus, DataValidator, DataBusError

# Fixtures

@pytest.fixture
def data_bus():
    """Create fresh data bus instance."""
    return DataBus()

@pytest.fixture
def simple_validator():
    """Create simple validator function."""
    return lambda x: isinstance(x, dict) and 'value' in x

# Validation Tests

def test_validator_init():
    """Test validator initialization."""
    validator = DataValidator(
        "test_topic",
        validator=lambda x: isinstance(x, dict),
        schema={"value": int}
    )
    assert validator.topic == "test_topic"
    assert validator.validations == 0
    assert validator.failures == 0

def test_validator_custom_function():
    """Test custom validation function."""
    validator = DataValidator(
        "test_topic",
        validator=lambda x: isinstance(x, dict) and 'value' in x
    )
    
    assert validator.validate({"value": 42})
    assert validator.validations == 1
    assert validator.failures == 0
    
    assert not validator.validate({"wrong": 42})
    assert validator.validations == 2
    assert validator.failures == 1

def test_validator_schema():
    """Test schema validation."""
    validator = DataValidator(
        "test_topic",
        schema={"value": int, "name": str}
    )
    
    assert validator.validate({"value": 42, "name": "test"})
    assert not validator.validate({"value": "wrong", "name": "test"})
    assert validator.last_failure == "Invalid type for value"

# Subscribe Tests

def test_subscribe_basic(data_bus):
    """Test basic subscription."""
    callback = Mock()
    assert data_bus.subscribe("test_topic", callback)
    assert "test_topic" in data_bus.list_topics()

def test_subscribe_invalid(data_bus):
    """Test invalid subscription."""
    with pytest.raises(DataBusError):
        data_bus.subscribe("test_topic", "not_callable")

def test_unsubscribe(data_bus):
    """Test unsubscription."""
    callback = Mock()
    data_bus.subscribe("test_topic", callback)
    assert data_bus.unsubscribe("test_topic", callback)
    assert "test_topic" not in data_bus.list_topics()

def test_unsubscribe_nonexistent(data_bus):
    """Test unsubscribing nonexistent callback."""
    callback = Mock()
    assert not data_bus.unsubscribe("test_topic", callback)

# Publish Tests

def test_publish_basic(data_bus):
    """Test basic publish."""
    callback = Mock()
    data_bus.subscribe("test_topic", callback)
    
    assert data_bus.publish("test_topic", {"value": 42})
    callback.assert_called_once_with("test_topic", {"value": 42})

def test_publish_multiple_subscribers(data_bus):
    """Test publishing to multiple subscribers."""
    callbacks = [Mock(), Mock(), Mock()]
    for cb in callbacks:
        data_bus.subscribe("test_topic", cb)
    
    data_bus.publish("test_topic", {"value": 42})
    
    for cb in callbacks:
        cb.assert_called_once_with("test_topic", {"value": 42})

def test_publish_with_validation(data_bus, simple_validator):
    """Test publishing with validation."""
    data_bus.add_validator("test_topic", simple_validator)
    callback = Mock()
    data_bus.subscribe("test_topic", callback)
    
    # Valid data
    assert data_bus.publish("test_topic", {"value": 42})
    callback.assert_called_once_with("test_topic", {"value": 42})
    
    # Invalid data
    assert not data_bus.publish("test_topic", {"wrong": 42})
    assert callback.call_count == 1  # No additional calls

def test_publish_error_handling(data_bus):
    """Test error handling during publish."""
    def failing_callback(topic, data):
        raise ValueError("Test error")
    
    data_bus.subscribe("test_topic", failing_callback)
    
    # Should not raise, but log error
    assert data_bus.publish("test_topic", {"value": 42})
    assert data_bus._delivery_errors == 1

# Performance Tests

def test_publish_performance(data_bus):
    """Test publishing performance."""
    # Add several subscribers
    callbacks = [Mock() for _ in range(5)]
    for cb in callbacks:
        data_bus.subscribe("test_topic", cb)
    
    # Publish several messages
    start = time.time()
    for i in range(100):
        data_bus.publish("test_topic", {"value": i})
    end = time.time()
    
    # Check stats
    stats = data_bus.get_stats()
    assert stats["messages_published"] == 100
    assert stats["messages_delivered"] == 500  # 100 * 5 subscribers
    assert stats["avg_latency"] < 0.001  # Should be fast

def test_concurrent_publish(data_bus):
    """Test concurrent publishing."""
    results = []
    
    def subscriber(topic, data):
        results.append(data["value"])
    
    data_bus.subscribe("test_topic", subscriber)
    
    def publisher():
        for i in range(100):
            data_bus.publish("test_topic", {"value": i})
    
    # Start multiple publisher threads
    threads = [
        threading.Thread(target=publisher)
        for _ in range(3)
    ]
    
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    
    # Check results
    assert len(results) == 300  # 3 threads * 100 messages
    assert len(set(results)) == 100  # All values 0-99 should be present

# Integration Tests

def test_full_workflow(data_bus):
    """Test complete data bus workflow."""
    # Setup validators
    data_bus.add_validator(
        "metrics",
        schema={"value": float, "timestamp": float}
    )
    data_bus.add_validator(
        "events",
        validator=lambda x: isinstance(x, dict) and "type" in x
    )
    
    # Setup subscribers
    metrics = []
    events = []
    
    def metrics_handler(topic, data):
        metrics.append(data["value"])
    
    def events_handler(topic, data):
        events.append(data["type"])
    
    data_bus.subscribe("metrics", metrics_handler)
    data_bus.subscribe("events", events_handler)
    
    # Publish valid and invalid data
    data_bus.publish("metrics", {
        "value": 42.0,
        "timestamp": time.time()
    })
    data_bus.publish("metrics", {
        "value": "wrong",  # Invalid type
        "timestamp": time.time()
    })
    data_bus.publish("events", {"type": "start"})
    data_bus.publish("events", {"wrong": "data"})  # Invalid
    
    # Check results
    assert len(metrics) == 1
    assert metrics[0] == 42.0
    assert len(events) == 1
    assert events[0] == "start"
    
    # Check stats
    stats = data_bus.get_stats()
    assert stats["validation_errors"] == 2
    assert stats["messages_delivered"] == 2
    assert stats["active_topics"] == 2
    assert len(stats["validators"]) == 2
    
    # Clean up
    assert data_bus.unsubscribe("metrics", metrics_handler)
    assert data_bus.unsubscribe("events", events_handler)
    assert len(data_bus.list_topics()) == 0
