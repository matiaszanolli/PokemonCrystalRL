"""Frame processing service.

This service handles game frame capture, processing, and streaming:
- Frame compression and optimization
- Frame rate tracking
- Frame buffering and queuing
- Quality settings management
"""

import threading
import time
import logging
import cv2
import numpy as np
from queue import Queue
from dataclasses import dataclass
from typing import Optional, Tuple, Any
from flask_socketio import emit

from monitoring.components.capture import ScreenCapture


@dataclass
class FrameConfig:
    """Frame processing configuration."""
    buffer_size: int = 1  # Frame buffer size (1 for lowest latency)
    quality: int = 85    # JPEG quality (0-100)
    target_fps: int = 30  # Target frame rate
    optimize: bool = True  # Enable JPEG optimization
    progressive: bool = False  # Disable progressive scan for lower latency


class FrameService:
    """Handles game frame processing and streaming.
    
    Features:
    - Frame capture and buffering
    - JPEG compression with quality control
    - FPS tracking and metrics
    - Frame streaming via WebSocket
    """
    
    def __init__(self, config: Optional[FrameConfig] = None):
        """Initialize frame service.
        
        Args:
            config: Frame processing configuration
        """
        self.config = config or FrameConfig()
        
        # Component references
        self._screen_capture: Optional[ScreenCapture] = None
        
    @property
    def screen_capture(self) -> Optional[ScreenCapture]:
        """Get screen capture component."""
        return self._screen_capture
        
        
        # Frame buffer
        self._frame_queue = Queue(maxsize=self.config.buffer_size)
        
        # Performance tracking
        self.frames_captured = 0
        self.frames_sent = 0
        self.current_fps = 0.0
        self._fps_counter = 0
        self._fps_start_time = time.time()
        
        # State
        self._running = False
        self._lock = threading.RLock()
        self._app_context = None
        
        # Logging
        self.logger = logging.getLogger(__name__)
    
    def set_screen_capture(self, capture: ScreenCapture) -> None:
        """Set screen capture component.
        
        Args:
            capture: Screen capture component
        """
        self._screen_capture = capture
    
    def set_quality(self, quality: str) -> None:
        """Set frame quality.
        
        Args:
            quality: Quality level ('low', 'medium', 'high')
        """
        quality_levels = {
            'low': 50,
            'medium': 85,
            'high': 95
        }
        self.config.quality = quality_levels.get(quality, 85)
    
    def get_status(self) -> dict:
        """Get service status.
        
        Returns:
            Dictionary with status information
        """
        return {
            'running': self._running,
            'frames_captured': self.frames_captured,
            'frames_sent': self.frames_sent,
            'current_fps': round(self.current_fps, 1),
            'frame_queue_size': self._frame_queue.qsize(),
            'quality': self.config.quality
        }
    
    def process_frame(self, frame: np.ndarray) -> Optional[bytes]:
        """Process a single frame.
        
        Args:
            frame: Raw frame data
            
        Returns:
            Processed frame as JPEG bytes or None if processing failed
        """
        try:
            # Update FPS tracking
            with self._lock:
                self._fps_counter += 1
                current_time = time.time()
                elapsed = current_time - self._fps_start_time
                
                if elapsed >= 1.0:
                    self.current_fps = self._fps_counter / elapsed
                    self._fps_counter = 0
                    self._fps_start_time = current_time
            
            # Compress frame
            success, buffer = cv2.imencode(
                '.jpg',
                frame,
                [
                    cv2.IMWRITE_JPEG_QUALITY, self.config.quality,
                    cv2.IMWRITE_JPEG_OPTIMIZE, 1 if self.config.optimize else 0,
                    cv2.IMWRITE_JPEG_PROGRESSIVE, 1 if self.config.progressive else 0
                ]
            )
            
            if not success:
                return None
            
            with self._lock:
                self.frames_captured += 1
            
            return buffer.tobytes()
            
        except Exception as e:
            self.logger.error(f"Frame processing error: {e}")
            return None
    
    def send_frame(self, frame_data: bytes) -> bool:
        """Send processed frame to clients.
        
        Args:
            frame_data: Processed frame data
            
        Returns:
            True if frame was sent successfully
        """
        try:
            # Send frame via WebSocket
            if not self._app_context:
                from flask import current_app
                self._app_context = current_app.app_context()
            
            with self._app_context:
                emit('frame', frame_data, binary=True)
                
                with self._lock:
                    self.frames_sent += 1
                
                return True
            
        except Exception as e:
            self.logger.error(f"Frame send error: {e}")
            return False
    
    def handle_frame_request(self) -> None:
        """Handle client frame request."""
        if not self._screen_capture:
            return
        
        try:
            # Get latest frame
            frame = self._screen_capture.get_frame("raw")
            if frame is None:
                return
            
            # Process and send
            frame_data = self.process_frame(frame)
            if frame_data:
                self.send_frame(frame_data)
                
        except Exception as e:
            self.logger.error(f"Frame request error: {e}")
    
    def clear(self) -> None:
        """Clear internal state and counters."""
        with self._lock:
            while not self._frame_queue.empty():
                try:
                    self._frame_queue.get_nowait()
                except:
                    pass
            self.frames_captured = 0
            self.frames_sent = 0
            self.current_fps = 0.0
            self._fps_counter = 0
            self._fps_start_time = time.time()
