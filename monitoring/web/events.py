"""WebSocket events for monitoring system.

This module provides:
- Event type definitions
- Event data validation
- Event handlers and dispatching
"""

from enum import Enum, auto
from dataclasses import dataclass
from typing import Dict, Any, Optional, Callable, List
from datetime import datetime
import time
import logging
import json

from .services.frame import FrameService
from .services.metrics import MetricsService


class EventType(Enum):
    """WebSocket event types."""
    
    # Connection events
    CONNECT = "connect"
    DISCONNECT = "disconnect"
    
    # Frame events
    REQUEST_FRAME = "request_frame"
    FRAME = "frame"
    SET_FRAME_QUALITY = "set_quality"
    
    # Metrics events
    REQUEST_METRICS = "subscribe_metrics"
    METRICS_UPDATE = "metrics"
    
    # Status events
    REQUEST_STATUS = "request_status"
    STATUS_UPDATE = "status"
    
    # Control events
    SET_UPDATE_INTERVAL = "set_interval"
    PAUSE = "pause"
    RESUME = "resume"
    
    # Error events
    ERROR = "error"


@dataclass
class EventContext:
    """Event handling context."""
    frame_service: FrameService
    metrics_service: MetricsService
    socket: Any  # SocketIO instance
    event_data: Any
    timestamp: float = None
    
    def __post_init__(self):
        """Initialize timestamp if not provided."""
        if self.timestamp is None:
            self.timestamp = time.time()


class EventDispatcher:
    """WebSocket event dispatcher.
    
    Handles:
    - Event registration and routing
    - Event validation
    - Error handling and recovery
    - Event timing and metrics
    """
    
    def __init__(self, frame_service: FrameService, metrics_service: MetricsService):
        """Initialize event dispatcher.
        
        Args:
            frame_service: Frame processing service
            metrics_service: Metrics processing service
        """
        self.frame_service = frame_service
        self.metrics_service = metrics_service
        self.handlers: Dict[EventType, List[Callable]] = {event: [] for event in EventType}
        self.logger = logging.getLogger(__name__)
        
        # Register default handlers
        self._register_default_handlers()
    
    def _register_default_handlers(self):
        """Register default event handlers."""
        # Frame handlers
        self.register(EventType.REQUEST_FRAME, self._handle_frame_request)
        self.register(EventType.SET_FRAME_QUALITY, self._handle_set_quality)
        
        # Metrics handlers
        self.register(EventType.REQUEST_METRICS, self._handle_metrics_request)
        
        # Status handlers
        self.register(EventType.REQUEST_STATUS, self._handle_status_request)
        
        # Control handlers
        self.register(EventType.SET_UPDATE_INTERVAL, self._handle_set_interval)
        self.register(EventType.PAUSE, self._handle_pause)
        self.register(EventType.RESUME, self._handle_resume)
    
    def register(self, event_type: EventType, handler: Callable):
        """Register event handler.
        
        Args:
            event_type: Type of event to handle
            handler: Handler function
        """
        self.handlers[event_type].append(handler)
    
    def dispatch(self, event_type: str, socket: Any, event_data: Any = None):
        """Dispatch event to registered handlers.
        
        Args:
            event_type: Event type string
            socket: SocketIO instance
            event_data: Optional event data
        """
        try:
            # Convert string to enum
            try:
                event_enum = EventType(event_type)
            except ValueError:
                self.logger.warning(f"Unknown event type: {event_type}")
                return
            
            # Create event context
            context = EventContext(
                frame_service=self.frame_service,
                metrics_service=self.metrics_service,
                socket=socket,
                event_data=event_data
            )
            
            # Call handlers
            for handler in self.handlers[event_enum]:
                try:
                    handler(context)
                except Exception as e:
                    self.logger.error(f"Handler error for {event_type}: {e}")
                    self._handle_error(context, str(e))
        
        except Exception as e:
            self.logger.error(f"Dispatch error for {event_type}: {e}")
            if socket:
                socket.emit(EventType.ERROR.value, {
                    'error': str(e),
                    'timestamp': time.time()
                })
    
    def _handle_error(self, context: EventContext, error: str):
        """Handle errors during event processing.
        
        Args:
            context: Event context
            error: Error message
        """
        error_data = {
            'error': error,
            'timestamp': time.time()
        }
        context.socket.emit(EventType.ERROR.value, error_data)
    
    def _handle_frame_request(self, context: EventContext):
        """Handle frame request event."""
        self.frame_service.handle_frame_request()
    
    def _handle_set_quality(self, context: EventContext):
        """Handle frame quality setting event."""
        if not context.event_data or 'quality' not in context.event_data:
            raise ValueError("Missing quality parameter")
        quality = context.event_data['quality']
        self.frame_service.set_quality(quality)
    
    def _handle_metrics_request(self, context: EventContext):
        """Handle metrics subscription event."""
        if not context.event_data:
            return
            
        metrics = self.metrics_service.get_metrics(
            names=context.event_data.get('metrics'),
            since=context.event_data.get('since')
        )
        if metrics:
            context.socket.emit(EventType.METRICS_UPDATE.value, metrics)
    
    def _handle_status_request(self, context: EventContext):
        """Handle status request event."""
        frame_status = self.frame_service.get_status()
        metrics_status = self.metrics_service.get_status()
        
        status = {
            'frame_service': frame_status,
            'metrics_service': metrics_status,
            'timestamp': time.time()
        }
        context.socket.emit(EventType.STATUS_UPDATE.value, status)
    
    def _handle_set_interval(self, context: EventContext):
        """Handle update interval setting event."""
        if not context.event_data or 'interval' not in context.event_data:
            raise ValueError("Missing interval parameter")
        # Store interval for future use
        self._update_interval = context.event_data['interval']
    
    def _handle_pause(self, context: EventContext):
        """Handle pause event."""
        self.frame_service._running = False
    
    def _handle_resume(self, context: EventContext):
        """Handle resume event."""
        self.frame_service._running = True
