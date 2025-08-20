"""
Game streamer module for Pokemon Crystal RL.

Provides optimized video streaming functionality with:
- Direct PyBoy buffer access for maximum performance
- Configurable quality and compression settings
- Memory-efficient frame handling
- Automatic frame rate control
- Performance monitoring

This module handles real-time streaming of game visuals and actions
for monitoring and debugging purposes.
"""

import cv2
import numpy as np
from typing import Optional, Dict, Any, Union, List
from queue import Queue
import threading
import time
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
import io
from PIL import Image
from pathlib import Path
import base64

from .data_bus import DataType, get_data_bus


class StreamQuality(Enum):
    """Video stream quality presets."""
    LOW = {"scale": 1, "fps": 5, "compression": 50}
    MEDIUM = {"scale": 2, "fps": 15, "compression": 75}
    HIGH = {"scale": 3, "fps": 30, "compression": 90}
    ULTRA = {"scale": 4, "fps": 60, "compression": 95}


@dataclass
class StreamFrame:
    """A single frame from the game stream."""
    frame: np.ndarray
    timestamp: float
    frame_number: int
    width: int
    height: int
    format: str = "RGB"
    compressed_data: Optional[bytes] = None
    compressed_size: Optional[int] = None
    quality: Optional[str] = None
    action: Optional[int] = None
    reward: Optional[float] = None
    state: Optional[Dict[str, Any]] = None
    
    def get_compressed(self, quality: Dict[str, Any]) -> bytes:
        """Get compressed frame data, caching the result."""
        if self.compressed_data is None:
            # Convert numpy array to PIL Image
            image = Image.fromarray(self.frame)
            
            # Apply scaling if needed
            scale = quality.get('scale', 1)
            if scale != 1:
                new_size = (self.width * scale, self.height * scale)
                image = image.resize(new_size, Image.NEAREST)
            
            # Compress to JPEG
            buffer = io.BytesIO()
            image.save(
                buffer,
                format='JPEG',
                quality=quality.get('compression', 75),
                optimize=True
            )
            
            self.compressed_data = buffer.getvalue()
            self.compressed_size = len(self.compressed_data)
            self.quality = f"{quality.get('compression')}%"
        
        return self.compressed_data
    
    def get_base64(self, quality: Dict[str, Any]) -> str:
        """Get frame as base64 string."""
        return base64.b64encode(self.get_compressed(quality)).decode()


class GameStreamer:
    """Handles real-time streaming of game visuals and actions."""
    
    def __init__(self, quality: Union[StreamQuality, Dict[str, Any]] = StreamQuality.MEDIUM,
                buffer_size: int = 100):
        # Set quality settings
        self.quality = quality.value if isinstance(quality, StreamQuality) else quality
        self.frame_interval = 1.0 / self.quality['fps']
        self.buffer_size = buffer_size
        
        # Frame buffer
        self.frame_buffer = Queue(maxsize=buffer_size)
        self.latest_frame: Optional[StreamFrame] = None
        self.frame_count = 0
        
        # Frame rate tracking
        self.fps_window = 30  # Calculate FPS over 30 frames
        self.frame_times = []
        self.current_fps = 0.0
        
        # Streaming state
        self.is_streaming = False
        self.stream_thread: Optional[threading.Thread] = None
        self._stream_lock = threading.Lock()
        
        # Data bus for metrics
        self.data_bus = get_data_bus()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        self.logger.info("🎮 Game streamer initialized")
    
    def start_streaming(self) -> None:
        """Start the streaming process."""
        with self._stream_lock:
            if self.is_streaming:
                return
            
            self.is_streaming = True
            self.stream_thread = threading.Thread(target=self._stream_loop, daemon=True)
            self.stream_thread.start()
            
            self.logger.info("🎯 Streaming started")
    
    def stop_streaming(self) -> None:
        """Stop the streaming process."""
        with self._stream_lock:
            self.is_streaming = False
            if self.stream_thread and self.stream_thread.is_alive():
                self.stream_thread.join(timeout=5.0)
            
            self.logger.info("⏹️ Streaming stopped")
    
    def add_frame(self, frame: np.ndarray, action: Optional[int] = None,
                 reward: Optional[float] = None, state: Optional[Dict[str, Any]] = None) -> None:
        """Add a new frame to the stream."""
        try:
            if frame is None or frame.size == 0:
                return
            
            # Ensure frame is RGB
            if len(frame.shape) == 2:  # Grayscale
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            elif frame.shape[2] == 4:  # RGBA
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
            
            # Create frame object
            stream_frame = StreamFrame(
                frame=frame.copy(),
                timestamp=time.time(),
                frame_number=self.frame_count,
                width=frame.shape[1],
                height=frame.shape[0],
                action=action,
                reward=reward,
                state=state
            )
            
            # Pre-compress frame with current quality settings
            stream_frame.get_compressed(self.quality)
            
            # Update frame buffer
            if self.frame_buffer.full():
                try:
                    self.frame_buffer.get_nowait()
                except:
                    pass
            
            self.frame_buffer.put(stream_frame)
            self.latest_frame = stream_frame
            self.frame_count += 1
            
            # Update FPS tracking
            self.frame_times.append(time.time())
            if len(self.frame_times) > self.fps_window:
                self.frame_times.pop(0)
                duration = self.frame_times[-1] - self.frame_times[0]
                if duration > 0:
                    self.current_fps = len(self.frame_times) / duration
            
            # Publish metrics
            if self.data_bus and self.frame_count % 30 == 0:  # Every 30 frames
                self.data_bus.publish(
                    DataType.SYSTEM_INFO,
                    {
                        'fps': round(self.current_fps, 1),
                        'frame_count': self.frame_count,
                        'buffer_usage': self.frame_buffer.qsize() / self.buffer_size * 100,
                        'timestamp': datetime.now().isoformat()
                    },
                    'game_streamer'
                )
            
        except Exception as e:
            self.logger.error(f"⚠️ Error adding frame: {e}")
    
    def get_latest_frame(self) -> Optional[StreamFrame]:
        """Get the most recent frame."""
        return self.latest_frame
    
    def get_frame_history(self, n_frames: int) -> List[StreamFrame]:
        """Get the last n frames from the buffer."""
        frames = []
        try:
            # Create a list from queue without removing items
            with self._stream_lock:
                temp_queue = Queue()
                while not self.frame_buffer.empty() and len(frames) < n_frames:
                    frame = self.frame_buffer.get()
                    frames.append(frame)
                    temp_queue.put(frame)
                
                # Restore frames to original queue
                while not temp_queue.empty():
                    self.frame_buffer.put(temp_queue.get())
                
            return frames[-n_frames:]
            
        except Exception as e:
            self.logger.error(f"⚠️ Error getting frame history: {e}")
            return []
    
    def clear_buffer(self) -> None:
        """Clear the frame buffer."""
        with self._stream_lock:
            while not self.frame_buffer.empty():
                try:
                    self.frame_buffer.get_nowait()
                except:
                    pass
    
    def get_stream_stats(self) -> Dict[str, Any]:
        """Get streaming statistics."""
        latest_frame = self.latest_frame
        return {
            'fps': round(self.current_fps, 1),
            'frame_count': self.frame_count,
            'buffer_size': self.buffer_size,
            'buffer_usage': self.frame_buffer.qsize(),
            'is_streaming': self.is_streaming,
            'quality': self.quality,
            'frame_size': (latest_frame.width, latest_frame.height) if latest_frame else None,
            'compressed_size': latest_frame.compressed_size if latest_frame else None,
            'compression_ratio': (latest_frame.width * latest_frame.height * 3) / latest_frame.compressed_size
                                if latest_frame and latest_frame.compressed_size else None,
            'latest_frame_time': latest_frame.timestamp if latest_frame else None
        }
    
    def change_quality(self, quality_name: str) -> None:
        """Change streaming quality settings."""
        try:
            if quality_name.upper() in StreamQuality.__members__:
                self.quality = StreamQuality[quality_name.upper()].value
                self.frame_interval = 1.0 / self.quality['fps']
                self.logger.info(f"🎮 Streaming quality changed to {quality_name}")
                return
            
            self.logger.warning(f"⚠️ Invalid quality setting: {quality_name}")
            
        except Exception as e:
            self.logger.error(f"❌ Error changing quality: {e}")
    
    def capture_pyboy_frame(self, pyboy_instance) -> Optional[np.ndarray]:
        """Capture frame directly from PyBoy instance."""
        try:
            if not pyboy_instance or not hasattr(pyboy_instance, 'screen'):
                return None
            
            # Get raw buffer
            raw_buffer = pyboy_instance.screen.raw_buffer
            if raw_buffer is None:
                return None
            
            # Get dimensions and format
            dims = getattr(pyboy_instance.screen, 'raw_buffer_dims', (144, 160))
            format_str = getattr(pyboy_instance.screen, 'raw_buffer_format', 'RGBA')
            
            # Convert memory view to numpy array
            frame_array = np.frombuffer(raw_buffer, dtype=np.uint8)
            
            if format_str == "RGBA" and frame_array.size >= dims[0] * dims[1] * 4:
                frame_array = frame_array[:dims[0] * dims[1] * 4].reshape((dims[0], dims[1], 4))
                return frame_array[:, :, :3].copy()  # Convert RGBA to RGB
                
            elif format_str == "RGB" and frame_array.size >= dims[0] * dims[1] * 3:
                return frame_array[:dims[0] * dims[1] * 3].reshape((dims[0], dims[1], 3)).copy()
            
            return None
            
        except Exception as e:
            self.logger.error(f"❌ Error capturing PyBoy frame: {e}")
            return None
    
    def _stream_loop(self):
        """Main streaming loop."""
        last_frame_time = time.time()
        
        while self.is_streaming:
            try:
                # Rate limiting
                current_time = time.time()
                elapsed = current_time - last_frame_time
                if elapsed < self.frame_interval:
                    time.sleep(self.frame_interval - elapsed)
                
                # Get next frame
                if not self.frame_buffer.empty():
                    frame = self.frame_buffer.get()
                    if frame and self.data_bus:
                        # Publish frame data
                        self.data_bus.publish(
                            DataType.GAME_SCREEN,
                            {
                                'frame': frame.frame.tolist(),  # Convert to list for JSON
                                'frame_number': frame.frame_number,
                                'timestamp': frame.timestamp,
                                'action': frame.action,
                                'reward': frame.reward,
                                'state': frame.state
                            },
                            'game_streamer'
                        )
                
                last_frame_time = time.time()
                
            except Exception as e:
                self.logger.error(f"❌ Streaming error: {e}")
                time.sleep(1)  # Avoid tight loop on error
    
    def __del__(self):
        """Cleanup on deletion."""
        self.stop_streaming()
