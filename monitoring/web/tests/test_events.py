"""Tests for WebSocket events system."""

import pytest
import time
from unittest.mock import Mock, patch
from typing import Dict, Any

from ..events import EventDispatcher, EventType, EventContext
from ..services.frame import FrameService, FrameConfig
from ..services.metrics import MetricsService, MetricsConfig


@pytest.fixture
def frame_service():
    """Create frame service for testing."""
    return Mock(spec=FrameService)

@pytest.fixture
def metrics_service():
    """Create metrics service for testing."""
    return Mock(spec=MetricsService)

@pytest.fixture
def socket():
    """Create mock socket for testing."""
    socket = Mock()
    socket.emit = Mock()
    return socket

@pytest.fixture
def dispatcher(frame_service, metrics_service):
    """Create event dispatcher for testing."""
    return EventDispatcher(frame_service, metrics_service)

def test_event_type_values():
    """Test event type enum values."""
    # Check connection events
    assert EventType.CONNECT.value == "connect"
    assert EventType.DISCONNECT.value == "disconnect"
    
    # Check frame events
    assert EventType.REQUEST_FRAME.value == "request_frame"
    assert EventType.FRAME.value == "frame"
    assert EventType.SET_FRAME_QUALITY.value == "set_quality"
    
    # Check metrics events
    assert EventType.REQUEST_METRICS.value == "subscribe_metrics"
    assert EventType.METRICS_UPDATE.value == "metrics"
    
    # Check status events
    assert EventType.REQUEST_STATUS.value == "request_status"
    assert EventType.STATUS_UPDATE.value == "status"

def test_event_context():
    """Test event context creation."""
    frame_service = Mock()
    metrics_service = Mock()
    socket = Mock()
    data = {'test': 'data'}
    
    # Create context without timestamp
    context = EventContext(frame_service, metrics_service, socket, data)
    assert context.frame_service == frame_service
    assert context.metrics_service == metrics_service
    assert context.socket == socket
    assert context.event_data == data
    assert isinstance(context.timestamp, float)
    
    # Create context with timestamp
    timestamp = time.time()
    context = EventContext(frame_service, metrics_service, socket, data, timestamp)
    assert context.timestamp == timestamp

def test_event_registration(dispatcher):
    """Test event handler registration."""
    # Create test handler
    handler = Mock()
    
    # Register handler
    dispatcher.register(EventType.CONNECT, handler)
    assert handler in dispatcher.handlers[EventType.CONNECT]
    
    # Register another handler for same event
    handler2 = Mock()
    dispatcher.register(EventType.CONNECT, handler2)
    assert len(dispatcher.handlers[EventType.CONNECT]) == 2
    assert handler2 in dispatcher.handlers[EventType.CONNECT]

def test_frame_request_handling(dispatcher, frame_service, socket):
    """Test frame request event handling."""
    # Dispatch frame request
    dispatcher.dispatch(EventType.REQUEST_FRAME.value, socket)
    frame_service.handle_frame_request.assert_called_once()

def test_set_quality_handling(dispatcher, frame_service, socket):
    """Test frame quality setting event handling."""
    # Test with valid quality
    quality = "high"
    dispatcher.dispatch(EventType.SET_FRAME_QUALITY.value, socket, {'quality': quality})
    frame_service.set_quality.assert_called_once_with(quality)
    
    # Test with missing quality
    frame_service.reset_mock()
    dispatcher.dispatch(EventType.SET_FRAME_QUALITY.value, socket, {})
    frame_service.set_quality.assert_not_called()
    socket.emit.assert_called_with(EventType.ERROR.value, {
        'error': 'Missing quality parameter',
        'timestamp': pytest.approx(time.time(), rel=1)
    })

def test_metrics_request_handling(dispatcher, metrics_service, socket):
    """Test metrics request event handling."""
    # Setup test data
    metrics_data = {'test': 'metrics'}
    metrics_service.get_metrics.return_value = metrics_data
    
    # Test with metrics names
    request_data = {'metrics': ['test'], 'since': 123.45}
    dispatcher.dispatch(EventType.REQUEST_METRICS.value, socket, request_data)
    metrics_service.get_metrics.assert_called_with(
        names=['test'],
        since=123.45
    )
    socket.emit.assert_called_with(EventType.METRICS_UPDATE.value, metrics_data)
    
    # Test without data
    metrics_service.reset_mock()
    socket.reset_mock()
    dispatcher.dispatch(EventType.REQUEST_METRICS.value, socket, None)
    metrics_service.get_metrics.assert_not_called()

def test_status_request_handling(dispatcher, frame_service, metrics_service, socket):
    """Test status request event handling."""
    # Setup mock status data
    frame_status = {'frame': 'status'}
    metrics_status = {'metrics': 'status'}
    frame_service.get_status.return_value = frame_status
    metrics_service.get_status.return_value = metrics_status
    
    # Request status
    dispatcher.dispatch(EventType.REQUEST_STATUS.value, socket)
    
    # Check service calls
    frame_service.get_status.assert_called_once()
    metrics_service.get_status.assert_called_once()
    
    # Check emitted status
    socket.emit.assert_called_once()
    args = socket.emit.call_args[0]
    assert args[0] == EventType.STATUS_UPDATE.value
    status_data = args[1]
    assert status_data['frame_service'] == frame_status
    assert status_data['metrics_service'] == metrics_status
    assert 'timestamp' in status_data

def test_error_handling(dispatcher, socket):
    """Test error handling during event processing."""
    # Register failing handler
    def failing_handler(context):
        raise ValueError("Test error")
    
    dispatcher.register(EventType.CONNECT, failing_handler)
    
    # Dispatch event
    dispatcher.dispatch(EventType.CONNECT.value, socket)
    
    # Check error emission
    socket.emit.assert_called_with(EventType.ERROR.value, {
        'error': 'Test error',
        'timestamp': pytest.approx(time.time(), rel=1)
    })

def test_unknown_event_handling(dispatcher, socket):
    """Test handling of unknown event types."""
    # Dispatch unknown event
    dispatcher.dispatch("unknown_event", socket)
    
    # No handlers should be called
    assert not socket.emit.called

def test_multiple_handlers(dispatcher, socket):
    """Test multiple handlers for same event."""
    # Create test handlers
    handler1 = Mock()
    handler2 = Mock()
    
    # Register both handlers
    dispatcher.register(EventType.CONNECT, handler1)
    dispatcher.register(EventType.CONNECT, handler2)
    
    # Dispatch event
    dispatcher.dispatch(EventType.CONNECT.value, socket)
    
    # Both handlers should be called
    handler1.assert_called_once()
    handler2.assert_called_once()
    
    # Check handler arguments
    context = handler1.call_args[0][0]
    assert isinstance(context, EventContext)
    assert context.socket == socket

def test_control_events(dispatcher, frame_service, socket):
    """Test control event handling."""
    # Test pause
    dispatcher.dispatch(EventType.PAUSE.value, socket)
    assert frame_service._running is False
    
    # Test resume
    dispatcher.dispatch(EventType.RESUME.value, socket)
    assert frame_service._running is True
    
    # Test interval setting
    interval = 1000
    dispatcher.dispatch(EventType.SET_UPDATE_INTERVAL.value, socket, {'interval': interval})
    assert dispatcher._update_interval == interval
