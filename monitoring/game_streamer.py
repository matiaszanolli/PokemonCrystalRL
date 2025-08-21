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

from monitoring.data_bus import get_data_bus, DataType


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
    
    def get_frame_info(self) -> Dict[str, Any]:
        """Get information about the latest frame"""
        with self._lock:
            return self._latest_frame_data.copy()
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for the streamer"""
        with self._lock:
            current_time = time.time()
            uptime = current_time - (self._last_capture_time or current_time)
            
            avg_capture_time = (
                sum(self._capture_times) / len(self._capture_times)
                if self._capture_times else 0.0
            )
            
            current_fps = (
                len(self._capture_times) / 30.0  # Based on last 30 seconds
                if len(self._capture_times) > 0 else 0.0
            )
            
            return {
                "frames_captured": self._frames_captured,
                "frames_served": self._frames_served,
                "capture_errors": self._capture_errors,
                "error_rate": self._capture_errors / max(self._frames_captured, 1),
                "current_fps": min(current_fps, self.max_frame_rate),
                "target_fps": self.max_frame_rate,
                "avg_capture_time_ms": avg_capture_time * 1000,
                "buffer_utilization": self._frame_buffer.qsize() / self.buffer_size,
                "is_streaming": self._capture_active,
                "consecutive_errors": self._consecutive_errors,
                "uptime_seconds": uptime
            }
    
    def is_healthy(self) -> bool:
        """Check if the streamer is healthy"""
        with self._lock:
            # Consider healthy if:
            # 1. Streaming is active
            # 2. No recent consecutive errors
            # 3. Recent frame capture activity
            
            recent_activity = (
                time.time() - self._last_capture_time < 10.0
                if self._last_capture_time > 0 else False
            )
            
            return (
                self._capture_active and
                self._consecutive_errors < self._max_consecutive_errors and
                recent_activity
            )
    
    def clear_buffer(self) -> None:
        """Clear the frame buffer"""
        with self._lock:
            while not self._frame_buffer.empty():
                try:
                    self._frame_buffer.get_nowait()
                except queue.Empty:
                    break
        
        self.logger.debug("Frame buffer cleared")
    
    def shutdown(self) -> None:
        """Clean shutdown of the streamer"""
        self.logger.info("🛑 Shutting down GameStreamComponent")
        
        # Stop streaming
        self.stop_streaming()
        
        # Clear resources
        with self._lock:
            self._latest_frame = None
            self._latest_frame_data.clear()
            self.clear_buffer()
            self._pyboy = None
        
        # Notify data bus
        if self.data_bus:
            self.data_bus.publish(
                DataType.COMPONENT_STATUS,
                {"component": "game_streamer", "status": "shutdown"},
                "game_streamer"
            )
        
        self.logger.info("✅ GameStreamComponent shutdown complete")
    
    def _capture_loop(self) -> None:
        """Main capture loop running in a separate thread"""
        self.logger.info("🔄 Game capture loop started")
        
        frame_interval = 1.0 / self.max_frame_rate
        last_frame_time = 0.0
        
        while self._capture_active:
            loop_start = time.time()
            
            try:
                # Rate limiting
                if (loop_start - last_frame_time) < frame_interval:
                    time.sleep(frame_interval - (loop_start - last_frame_time))
                    continue
                
                # Capture frame
                frame = self._capture_frame()
                if frame is not None:
                    self._process_frame(frame)
                    last_frame_time = loop_start
                    
                    # Reset error counter on success
                    self._consecutive_errors = 0
                
            except Exception as e:
                self._handle_capture_error(e)
            
            # Update heartbeat
            if self.data_bus:
                self.data_bus.update_component_heartbeat("game_streamer")
        
        self.logger.info("🔄 Game capture loop ended")
    
    def _capture_frame(self) -> Optional[np.ndarray]:
        """Capture a single frame from PyBoy"""
        if not self._pyboy:
            raise RuntimeError("PyBoy instance not available")
        
        capture_start = time.time()
        
        try:
            # Get screen from PyBoy
            screen = self._pyboy.screen.ndarray
            if screen is None or screen.size == 0:
                return None
            
            # Track performance
            capture_time = time.time() - capture_start
            self._capture_times.append(capture_time)
            if len(self._capture_times) > 100:  # Keep last 100 measurements
                self._capture_times.pop(0)
            
            self._frames_captured += 1
            self._last_capture_time = time.time()
            
            return screen.copy()  # Defensive copy
            
        except Exception as e:
            self.logger.error(f"Failed to capture frame from PyBoy: {e}")
            raise
    
    def _process_frame(self, frame: np.ndarray) -> None:
        """Process and store a captured frame"""
        with self._lock:
            # Update latest frame
            self._latest_frame = frame
            
            # Update frame metadata
            self._latest_frame_data = {
                "timestamp": time.time(),
                "shape": frame.shape,
                "dtype": str(frame.dtype),
                "size_bytes": frame.nbytes,
                "min_value": float(np.min(frame)),
                "max_value": float(np.max(frame)),
                "mean_value": float(np.mean(frame))
            }
            
            # Add to buffer (non-blocking)
            try:
                self._frame_buffer.put_nowait(frame)
            except queue.Full:
                # Remove oldest frame to make space
                try:
                    self._frame_buffer.get_nowait()
                    self._frame_buffer.put_nowait(frame)
                except queue.Empty:
                    pass
        
        # Publish to data bus
        if self.data_bus:
            try:
                # Don't publish the full frame data to avoid memory issues
                # Just publish metadata
                self.data_bus.publish(
                    DataType.GAME_SCREEN,
                    {
                        "frame_info": self._latest_frame_data,
                        "has_frame": True,
                        "buffer_size": self._frame_buffer.qsize()
                    },
                    "game_streamer"
                )
            except Exception as e:
                self.logger.warning(f"Failed to publish frame data to bus: {e}")
    
    def _handle_capture_error(self, error: Exception) -> None:
        """Handle capture errors with exponential backoff"""
        self._capture_errors += 1
        self._consecutive_errors += 1
        
        self.logger.error(f"Capture error #{self._consecutive_errors}: {error}")
        
        # Exponential backoff
        backoff_time = min(self._error_backoff_time * (2 ** self._consecutive_errors), 30.0)
        
        if self._consecutive_errors >= self._max_consecutive_errors:
            self.logger.error(f"Too many consecutive errors ({self._consecutive_errors}). Pausing for {backoff_time}s")
            
            # Notify data bus of error
            if self.data_bus:
                self.data_bus.publish(
                    DataType.ERROR_EVENT,
                    {
                        "component": "game_streamer",
                        "error": str(error),
                        "consecutive_errors": self._consecutive_errors,
                        "severity": "high" if self._consecutive_errors >= self._max_consecutive_errors else "medium"
                    },
                    "game_streamer"
                )
        
        # Sleep with error backoff
        if self._capture_active:
            time.sleep(min(backoff_time, 1.0))  # Cap at 1 second for responsiveness
