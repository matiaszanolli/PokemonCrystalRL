"""
Core Video Streaming for Pokemon Crystal RL

Ultra-low latency video streaming using PyBoy's raw buffer access.
Provides efficient frame capture, compression, and streaming capabilities.
"""

import time
import threading
import queue
import numpy as np
from typing import Optional, Dict, Any, Callable
import base64
import io
from dataclasses import dataclass
from enum import Enum

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


class StreamQuality(Enum):
    """Video stream quality presets"""
    LOW = {"scale": 1, "fps": 5, "compression": 50}
    MEDIUM = {"scale": 2, "fps": 10, "compression": 75} 
    HIGH = {"scale": 3, "fps": 15, "compression": 90}
    ULTRA = {"scale": 4, "fps": 30, "compression": 95}


@dataclass
class StreamFrame:
    """Single video frame data"""
    frame_id: int
    timestamp: float
    data: bytes
    format: str
    width: int
    height: int
    compressed_size: int
    quality: str


class PyBoyVideoStreamer:
    """Ultra-low latency PyBoy video streamer using raw buffer access"""
    
    def __init__(self, pyboy_instance, quality: StreamQuality = StreamQuality.MEDIUM):
        if not PIL_AVAILABLE:
            raise RuntimeError("PIL required for image processing")
            
        self.pyboy = pyboy_instance
        self.quality = quality.value if isinstance(quality, StreamQuality) else quality
        self.running = False
        
        # Frame buffer management
        self.frame_queue = queue.Queue(maxsize=10)  # Small buffer for low latency
        self.current_frame = None
        self.frame_counter = 0
        
        # Threading
        self.capture_thread = None
        
        # Performance tracking
        self.stats = {
            'frames_captured': 0,
            'frames_streamed': 0,
            'frames_dropped': 0,
            'avg_capture_time_ms': 0.0,
            'avg_compression_time_ms': 0.0,
            'start_time': time.time()
        }
        
        # Frame timing
        self.target_frame_interval = 1.0 / self.quality['fps']
        self.last_capture_time = 0
    
    def start_streaming(self):
        """Start the video streaming system"""
        if self.running:
            return
            
        self.running = True
        self.stats['start_time'] = time.time()
        
        # Start capture thread
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
    
    def stop_streaming(self):
        """Stop the video streaming system"""
        self.running = False
        
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=1.0)
    
    def _capture_loop(self):
        """Main capture loop using PyBoy raw buffer"""
        while self.running:
            try:
                capture_start = time.time()
                
                # Check if we should capture this frame (FPS limiting)
                if capture_start - self.last_capture_time < self.target_frame_interval:
                    time.sleep(0.001)  # Small sleep to prevent busy waiting
                    continue
                
                # Get raw buffer from PyBoy
                frame_data = self._capture_raw_frame()
                
                if frame_data is not None:
                    # Process and compress frame
                    processed_frame = self._process_frame(frame_data)
                    
                    if processed_frame:
                        # Add to queue (drop oldest if full)
                        try:
                            self.frame_queue.put_nowait(processed_frame)
                            self.stats['frames_captured'] += 1
                        except queue.Full:
                            # Drop oldest frame to maintain low latency
                            try:
                                self.frame_queue.get_nowait()
                                self.frame_queue.put_nowait(processed_frame)
                                self.stats['frames_dropped'] += 1
                            except queue.Empty:
                                pass
                
                # Update timing stats
                capture_time = time.time() - capture_start
                self.stats['avg_capture_time_ms'] = (
                    self.stats['avg_capture_time_ms'] * 0.9 + capture_time * 1000 * 0.1
                )
                
                self.last_capture_time = capture_start
                
            except Exception as e:
                # Continue streaming despite errors
                time.sleep(0.01)
    
    def _capture_raw_frame(self) -> Optional[np.ndarray]:
        """Capture frame using PyBoy's raw buffer for maximum performance"""
        try:
            if not self.pyboy or not hasattr(self.pyboy, 'screen'):
                return None
            
            # Get raw buffer - this is the fastest method
            raw_buffer = self.pyboy.screen.raw_buffer
            
            if raw_buffer is None:
                return None
            
            # Get dimensions and format
            dims = getattr(self.pyboy.screen, 'raw_buffer_dims', (144, 160))
            format_str = getattr(self.pyboy.screen, 'raw_buffer_format', 'RGBA')
            
            # Convert memory view to numpy array efficiently
            if format_str == "RGBA":
                # Reshape raw buffer to RGBA format
                frame_array = np.frombuffer(raw_buffer, dtype=np.uint8)
                if frame_array.size >= dims[0] * dims[1] * 4:
                    frame_array = frame_array[:dims[0] * dims[1] * 4].reshape((dims[0], dims[1], 4))
                    # Convert RGBA to RGB (drop alpha channel)
                    return frame_array[:, :, :3].copy()
            
            elif format_str == "RGB":
                frame_array = np.frombuffer(raw_buffer, dtype=np.uint8)
                if frame_array.size >= dims[0] * dims[1] * 3:
                    return frame_array[:dims[0] * dims[1] * 3].reshape((dims[0], dims[1], 3)).copy()
            
            return None
                
        except Exception:
            return None
    
    def _process_frame(self, frame_data: np.ndarray) -> Optional[StreamFrame]:
        """Process and compress frame for streaming"""
        try:
            compress_start = time.time()
            
            # Convert numpy array to PIL Image
            pil_image = Image.fromarray(frame_data)
            
            # Apply scaling
            scale = self.quality['scale']
            if scale != 1:
                new_size = (pil_image.width * scale, pil_image.height * scale)
                pil_image = pil_image.resize(new_size, Image.NEAREST)
            
            # Compress to JPEG for streaming (better compression than PNG)
            buffer = io.BytesIO()
            pil_image.save(
                buffer, 
                format='JPEG', 
                quality=self.quality['compression'],
                optimize=True
            )
            
            compressed_data = buffer.getvalue()
            
            # Create frame object
            frame = StreamFrame(
                frame_id=self.frame_counter,
                timestamp=time.time(),
                data=compressed_data,
                format='JPEG',
                width=pil_image.width,
                height=pil_image.height,
                compressed_size=len(compressed_data),
                quality=f"{self.quality['compression']}%"
            )
            
            # Update timing stats
            compress_time = time.time() - compress_start
            self.stats['avg_compression_time_ms'] = (
                self.stats['avg_compression_time_ms'] * 0.9 + compress_time * 1000 * 0.1
            )
            
            self.frame_counter += 1
            return frame
            
        except Exception:
            return None
    
    def get_latest_frame(self) -> Optional[StreamFrame]:
        """Get the most recent frame for HTTP polling"""
        try:
            # Get latest frame without blocking
            latest_frame = None
            while not self.frame_queue.empty():
                try:
                    latest_frame = self.frame_queue.get_nowait()
                except queue.Empty:
                    break
            
            if latest_frame:
                self.current_frame = latest_frame
                self.stats['frames_streamed'] += 1
            
            return self.current_frame
            
        except Exception:
            return None
    
    def get_frame_as_base64(self) -> Optional[str]:
        """Get latest frame as base64 for web display"""
        frame = self.get_latest_frame()
        if frame:
            return base64.b64encode(frame.data).decode()
        return None
    
    def get_frame_as_bytes(self) -> Optional[bytes]:
        """Get latest frame as raw bytes for HTTP response"""
        frame = self.get_latest_frame()
        if frame:
            return frame.data
        return None
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get streaming performance statistics"""
        runtime = time.time() - self.stats['start_time']
        
        return {
            'method': 'raw_buffer_optimized',
            'capture_latency_ms': self.stats['avg_capture_time_ms'],
            'compression_latency_ms': self.stats['avg_compression_time_ms'],
            'total_latency_ms': self.stats['avg_capture_time_ms'] + self.stats['avg_compression_time_ms'],
            'frames_captured': self.stats['frames_captured'],
            'frames_streamed': self.stats['frames_streamed'],
            'frames_dropped': self.stats['frames_dropped'],
            'drop_rate_percent': (self.stats['frames_dropped'] / max(self.stats['frames_captured'], 1)) * 100,
            'target_fps': self.quality['fps'],
            'quality_settings': self.quality,
            'runtime_seconds': runtime,
            'streaming_efficiency': (self.stats['frames_streamed'] / max(self.stats['frames_captured'], 1)) * 100
        }
    
    def change_quality(self, quality_name: str):
        """Change streaming quality dynamically"""
        quality_map = {
            "low": StreamQuality.LOW,
            "medium": StreamQuality.MEDIUM, 
            "high": StreamQuality.HIGH,
            "ultra": StreamQuality.ULTRA
        }
        
        if quality_name.lower() in quality_map:
            new_quality = quality_map[quality_name.lower()]
            self.quality = new_quality.value
            self.target_frame_interval = 1.0 / self.quality['fps']


def create_video_streamer(pyboy_instance, quality: str = "medium") -> PyBoyVideoStreamer:
    """Factory function to create optimized video streamer"""
    quality_map = {
        "low": StreamQuality.LOW,
        "medium": StreamQuality.MEDIUM, 
        "high": StreamQuality.HIGH,
        "ultra": StreamQuality.ULTRA
    }
    
    selected_quality = quality_map.get(quality.lower(), StreamQuality.MEDIUM)
    return PyBoyVideoStreamer(pyboy_instance, selected_quality)
