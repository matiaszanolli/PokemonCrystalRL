#!/usr/bin/env python3
"""
GameStreamComponent - Reliable game screen capture and streaming

This module provides a robust, memory-safe game streaming system with
proper error recovery and multiple output formats.
"""

import time
import threading
import queue
import logging
import io
import base64
from typing import Optional, Dict, Any, Union
from PIL import Image
import numpy as np

try:
    from .data_bus import get_data_bus, DataType
except ImportError:
    # Fallback for testing
    from data_bus import get_data_bus, DataType


class GameStreamComponent:
    """
    Reliable game screen capture and streaming with memory leak prevention
    
    Features:
    - PyBoy screen capture with error recovery
    - Frame buffering and compression
    - Multiple output formats (PNG, JPEG, base64)
    - Memory leak prevention and cleanup
    - Performance monitoring
    """
    
    def __init__(self, 
                 buffer_size: int = 10,
                 compression_quality: int = 85,
                 max_frame_rate: float = 10.0,
                 enable_data_bus: bool = True):
        
        self.buffer_size = buffer_size
        self.compression_quality = compression_quality
        self.max_frame_rate = max_frame_rate
        self.enable_data_bus = enable_data_bus
        
        # Threading and synchronization
        self._lock = threading.RLock()
        self._capture_thread: Optional[threading.Thread] = None
        self._capture_active = False
        
        # Frame buffering
        self._frame_buffer: queue.Queue = queue.Queue(maxsize=buffer_size)
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_frame_data: Dict[str, Any] = {}
        
        # PyBoy integration
        self._pyboy = None
        
        # Performance tracking
        self._frames_captured = 0
        self._frames_served = 0
        self._capture_errors = 0
        self._last_capture_time = 0.0
        self._capture_times: list = []
        
        # Error recovery
        self._consecutive_errors = 0
        self._max_consecutive_errors = 5
        self._error_backoff_time = 1.0
        
        # Setup logging
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Data bus integration
        self.data_bus = get_data_bus() if enable_data_bus else None
        if self.data_bus:
            self.data_bus.register_component("game_streamer", {
                "type": "streaming",
                "buffer_size": buffer_size,
                "max_fps": max_frame_rate
            })
        
        self.logger.info("🎮 GameStreamComponent initialized")
    
    def set_pyboy_instance(self, pyboy) -> None:
        """Set the PyBoy instance for screen capture"""
        with self._lock:
            self._pyboy = pyboy
            self.logger.info("🎮 PyBoy instance connected to streamer")
    
    def start_streaming(self) -> bool:
        """Start the streaming process"""
        if self._capture_active:
            self.logger.warning("Streaming already active")
            return True
        
        if not self._pyboy:
            self.logger.error("Cannot start streaming - PyBoy instance not set")
            return False
        
        with self._lock:
            self._capture_active = True
            self._capture_thread = threading.Thread(
                target=self._capture_loop,
                daemon=True,
                name="GameStreamer"
            )
            self._capture_thread.start()
        
        self.logger.info("🚀 Game streaming started")
        return True
    
    def stop_streaming(self) -> None:
        """Stop the streaming process"""
        if not self._capture_active:
            return
        
        with self._lock:
            self._capture_active = False
        
        if self._capture_thread and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=5.0)
            if self._capture_thread.is_alive():
                self.logger.warning("Streaming thread did not shut down cleanly")
        
        self.logger.info("🛑 Game streaming stopped")
    
    def get_latest_frame(self, format: str = "numpy") -> Optional[Union[np.ndarray, bytes, str]]:
        """
        Get the latest captured frame in the specified format
        
        Args:
            format: Output format ('numpy', 'png', 'jpeg', 'base64_png', 'base64_jpeg')
            
        Returns:
            Frame data in the specified format, or None if no frame available
        """
        with self._lock:
            if self._latest_frame is None:
                return None
            
            frame = self._latest_frame.copy()  # Defensive copy
        
        try:
            self._frames_served += 1
            
            if format == "numpy":
                return frame
            
            # Convert to PIL Image for other formats
            if len(frame.shape) == 3 and frame.shape[2] == 4:  # RGBA
                pil_image = Image.fromarray(frame, 'RGBA')
                pil_image = pil_image.convert('RGB')  # Remove alpha for JPEG compatibility
            else:
                pil_image = Image.fromarray(frame)
            
            if format == "png":
                buffer = io.BytesIO()
                pil_image.save(buffer, format='PNG')
                return buffer.getvalue()
            
            elif format == "jpeg":
                buffer = io.BytesIO()
                pil_image.save(buffer, format='JPEG', quality=self.compression_quality)
                return buffer.getvalue()
            
            elif format == "base64_png":
                buffer = io.BytesIO()
                pil_image.save(buffer, format='PNG')
                return base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            elif format == "base64_jpeg":
                buffer = io.BytesIO()
                pil_image.save(buffer, format='JPEG', quality=self.compression_quality)
                return base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            else:
                self.logger.error(f"Unsupported format: {format}")
                return None
        
        except Exception as e:
            self.logger.error(f"Error converting frame to {format}: {e}")
            return None
