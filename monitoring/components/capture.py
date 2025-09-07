"""
Screen Capture Component

This module provides the screen capture implementation for monitoring PyBoy games.
It handles frame capture, buffering, and processing with efficient resource usage.
"""

import time
import threading
import collections
from typing import Optional, Dict, Any, Deque
import numpy as np
from PIL import Image
import io
import base64

from ..base import MonitorComponent, ScreenCaptureConfig, ComponentError
from ..base import DataPublisher

class ScreenCapture(MonitorComponent, DataPublisher):
    """Screen capture implementation with efficient buffering and processing.
    
    This component captures screens from PyBoy, processes them efficiently,
    and provides them in various formats with proper resource management.
    """
    
    def __init__(self, config: ScreenCaptureConfig):
        """Initialize screen capture component.
        
        Args:
            config: Configuration for screen capture
        """
        self.config = config
        self._pyboy = None
        self._frame_buffer: Deque[np.ndarray] = collections.deque(
            maxlen=config.buffer_size
        )
        
        # Thread management
        self._capture_thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.RLock()
        
        # Performance tracking
        self._frames_captured = 0
        self._frames_dropped = 0
        self._last_frame_time = 0.0
        self._frame_times: Deque[float] = collections.deque(maxlen=100)
        
    def set_pyboy(self, pyboy) -> None:
        """Set the PyBoy instance for screen capture.
        
        Args:
            pyboy: PyBoy instance to capture from
        """
        with self._lock:
            self._pyboy = pyboy
            self._frames_captured = 0
            self._frames_dropped = 0
            self._frame_buffer.clear()
    
    def start(self) -> bool:
        """Start screen capture thread.
        
        Returns:
            bool: True if started successfully
        """
        if not self._pyboy:
            raise ComponentError("PyBoy instance not set")
            
        with self._lock:
            if self._running:
                return True
                
            self._running = True
            self._capture_thread = threading.Thread(
                target=self._capture_loop,
                daemon=True,
                name="ScreenCapture"
            )
            self._capture_thread.start()
            
            # Wait for first frame
            timeout = 5.0
            start = time.time()
            while not self._frame_buffer and time.time() - start < timeout:
                time.sleep(0.1)
                
            return bool(self._frame_buffer)
    
    def stop(self) -> bool:
        """Stop screen capture thread.
        
        Returns:
            bool: True if stopped successfully
        """
        with self._lock:
            self._running = False
            
        if self._capture_thread and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=5.0)
            return not self._capture_thread.is_alive()
            
        return True
    
    def get_frame(self, format: str = "numpy") -> Optional[Any]:
        """Get the most recent frame in the specified format.
        
        Args:
            format: Output format ('numpy', 'pil', 'png', 'jpeg', 'base64')
            
        Returns:
            Frame in requested format, or None if no frame available
        """
        with self._lock:
            if not self._frame_buffer:
                return None
                
            # Get latest frame
            frame = self._frame_buffer[-1].copy()
            
        try:
            if format == "numpy":
                return frame
                
            # Convert to PIL
            if frame.shape[2] == 4:  # RGBA
                pil_image = Image.fromarray(frame, 'RGBA')
                if format != "pil":
                    pil_image = pil_image.convert('RGB')
            else:
                pil_image = Image.fromarray(frame)
            
            # Apply upscaling if configured
            if self.config.upscale_factor > 1:
                new_size = tuple(s * self.config.upscale_factor for s in pil_image.size)
                pil_image = pil_image.resize(new_size, Image.NEAREST)
            
            if format == "pil":
                return pil_image
                
            # Convert to bytes
            buffer = io.BytesIO()
            if format in ("png", "base64_png"):
                pil_image.save(buffer, format="PNG")
            elif format in ("jpeg", "base64_jpeg"):
                pil_image.save(buffer, format="JPEG",
                             quality=self.config.compression_quality)
            else:
                raise ValueError(f"Unsupported format: {format}")
                
            buffer.seek(0)
            
            # Return appropriate format
            if format in ("png", "jpeg"):
                return buffer.getvalue()
            else:  # base64
                return base64.b64encode(buffer.getvalue()).decode('utf-8')
                
        except Exception as e:
            raise ComponentError(f"Frame conversion error: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get component status.
        
        Returns:
            Dict containing status information
        """
        with self._lock:
            frame_times = list(self._frame_times)
            current_fps = 1.0 / (sum(frame_times) / len(frame_times)) if frame_times else 0
            
            return {
                "running": self._running,
                "frames_captured": self._frames_captured,
                "frames_dropped": self._frames_dropped,
                "buffer_size": len(self._frame_buffer),
                "current_fps": current_fps,
                "target_fps": self.config.frame_rate,
                "last_frame_time": self._last_frame_time
            }
    
    def _capture_loop(self) -> None:
        """Main capture loop."""
        target_interval = 1.0 / self.config.frame_rate
        
        while self._running:
            try:
                loop_start = time.time()
                
                # Capture frame
                with self._lock:
                    if not self._pyboy:
                        time.sleep(0.1)
                        continue
                        
                    screen = self._pyboy.screen.ndarray()
                    if screen is None:
                        self._frames_dropped += 1
                        continue
                        
                    # Process frame
                    frame = screen.copy()  # Defensive copy
                    self._frame_buffer.append(frame)
                    self._frames_captured += 1
                    
                    # Track timing
                    now = time.time()
                    if self._last_frame_time:
                        frame_time = now - self._last_frame_time
                        self._frame_times.append(frame_time)
                    self._last_frame_time = now
                    
                    # Publish frame if needed
                    self.publish("screen_frame", frame)
                    
                # Rate limiting
                elapsed = time.time() - loop_start
                sleep_time = max(0, target_interval - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    
            except Exception as e:
                self._frames_dropped += 1
                time.sleep(0.1)  # Prevent tight loop on errors
    
    def publish(self, topic: str, data: Any) -> bool:
        """Publish frame data.
        
        This is a minimal implementation - for full pub/sub, use the DataBus.
        
        Args:
            topic: Topic to publish to (e.g., "screen_frame")
            data: Frame data to publish
            
        Returns:
            bool: True if published successfully
        """
        # This simple implementation just returns True
        # The DataBus handles actual pub/sub
        return True
