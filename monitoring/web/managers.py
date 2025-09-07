"""Status and event management for the monitoring system."""

import logging
import threading
import time
from typing import Any, Callable, Dict, Optional

import cv2
import numpy as np
from flask_socketio import SocketIO

logger = logging.getLogger(__name__)

class StatusManager:
    """Centralized status management and data broadcasting."""
    
    def __init__(self, socketio: SocketIO):
        """Initialize the status manager.
        
        Args:
            socketio: Flask-SocketIO instance for broadcasting
        """
        self.socketio = socketio
        self._last_update = 0.0
        self._last_frame_time = 0.0
        self._frame_count = 0
        self._current_fps = 0.0
        self._lock = threading.RLock()
        self._connected_clients = set()
        self._client_count = 0
        
    def broadcast_update(self, update_type: str, data: Dict[str, Any]) -> None:
        """Send a unified update to all clients.
        
        Args:
            update_type: Type of update (status, metrics, etc)
            data: Update data to broadcast
        """
        with self._lock:
            try:
                now = time.time()
                
                # Update performance metrics
                if update_type == 'frame':
                    self._frame_count += 1
                    time_diff = now - self._last_frame_time
                    if time_diff >= 1.0:
                        self._current_fps = self._frame_count / time_diff
                        self._frame_count = 0
                        self._last_frame_time = now
                
                # Send update
                self.socketio.emit(update_type, data)
                self._last_update = now
                
            except Exception as e:
                logger.error(f"Error broadcasting {update_type}: {e}")
                
    def process_frame(self, frame: np.ndarray, quality: int = 85) -> Optional[bytes]:
        """Process a frame for transmission.
        
        Args:
            frame: Raw frame data
            quality: JPEG quality (0-100)
            
        Returns:
            Processed frame data or None if processing failed
        """
        if frame is None:
            return None
            
        try:
            success, buffer = cv2.imencode(
                '.jpg',
                frame,
                [
                    cv2.IMWRITE_JPEG_QUALITY, quality,
                    cv2.IMWRITE_JPEG_OPTIMIZE, 1,
                    cv2.IMWRITE_JPEG_PROGRESSIVE, 0
                ]
            )
            return buffer.tobytes() if success else None
            
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            return None
            
    @property
    def current_fps(self) -> float:
        """Get the current frames per second rate."""
        return self._current_fps
        
    @property
    def last_update_time(self) -> float:
        """Get the timestamp of the last update."""
        return self._last_update
        
    def client_connected(self, client_id: str = None) -> None:
        """Handle client connection.
        
        Args:
            client_id: Optional client identifier
        """
        with self._lock:
            if client_id:
                self._connected_clients.add(client_id)
            self._client_count += 1
            
    def client_disconnected(self, client_id: str = None) -> None:
        """Handle client disconnection.
        
        Args:
            client_id: Optional client identifier
        """
        with self._lock:
            if client_id:
                self._connected_clients.discard(client_id)
            self._client_count = max(0, self._client_count - 1)
    
    def get_connected_clients(self) -> int:
        """Get number of connected clients.
        
        Returns:
            Number of connected clients
        """
        return self._client_count


class EventManager:
    """Centralized event handling system."""
    
    def __init__(self, status_manager: StatusManager):
        """Initialize the event manager.
        
        Args:
            status_manager: StatusManager instance for broadcasting
        """
        self.status_manager = status_manager
        self.handlers: Dict[str, Callable] = {}
        self._lock = threading.RLock()
        
    def register_handler(self, event_type: str, handler: Callable) -> None:
        """Register an event handler.
        
        Args:
            event_type: Type of event to handle
            handler: Handler function
        """
        with self._lock:
            if event_type in self.handlers:
                logger.warning(f"Overwriting existing handler for {event_type}")
            self.handlers[event_type] = handler
            
    def deregister_handler(self, event_type: str) -> None:
        """Remove an event handler.
        
        Args:
            event_type: Type of event to remove handler for
        """
        with self._lock:
            if event_type in self.handlers:
                del self.handlers[event_type]
                
    async def handle_event(self, event_type: str, data: Any = None) -> None:
        """Handle an incoming event.
        
        Args:
            event_type: Type of event to handle
            data: Event data
        """
        try:
            with self._lock:
                if event_type in self.handlers:
                    handler = self.handlers[event_type]
                    
            if handler:
                await handler(data)
                
        except Exception as e:
            logger.error(f"Error handling event {event_type}: {e}")
            # Broadcast error to clients
            self.status_manager.broadcast_update('error', {
                'message': f"Error handling {event_type}: {str(e)}",
                'type': 'event_error'
            })
