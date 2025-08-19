"""
game_streamer.py - Game screen capture and streaming

Handles capturing and streaming game screenshots, as well as annotating them
with relevant game state and debugging information.
"""

import time
import threading
from typing import Dict, Any, Optional, Tuple
from queue import Queue, Empty
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2

from core.monitoring.data_bus import DataType, get_data_bus
from core.error_handler import SafeOperation, error_boundary


class GameStreamer:
    """Handles game screen capture and streaming"""
    
    def __init__(self, max_queue_size: int = 30, annotation_font_size: int = 12):
        self.data_bus = get_data_bus()
        self.frame_queue = Queue(maxsize=max_queue_size)
        self.running = False
        
        # Image processing
        self.last_frame: Optional[np.ndarray] = None
        self.last_frame_time = 0
        self.frame_count = 0
        
        # Annotation settings
        try:
            self.font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 
                                         annotation_font_size)
        except Exception:
            # Fallback to default font
            self.font = ImageFont.load_default()
            
        # Performance tracking
        self._processing_times: List[float] = []
        self._fps_window = deque(maxlen=100)
        
        # Subscribe to events
        self.data_bus.subscribe(DataType.SCREEN_CAPTURE, self._handle_screen_capture)
        
    def start(self) -> None:
        """Start screen capture processing"""
        if self.running:
            return
            
        self.running = True
        self._processing_thread = threading.Thread(
            target=self._process_frames,
            name="GameStreamer",
            daemon=True
        )
        self._processing_thread.start()
        
    def stop(self) -> None:
        """Stop screen capture processing"""
        self.running = False
        if hasattr(self, '_processing_thread'):
            self._processing_thread.join(timeout=1.0)
            
    @error_boundary("GameStreamer")
    def _handle_screen_capture(self, frame_data: Dict[str, Any]) -> None:
        """Handle new screen capture frame"""
        if 'frame' not in frame_data:
            return
            
        frame = frame_data['frame']
        if not isinstance(frame, np.ndarray):
            return
            
        try:
            self.frame_queue.put_nowait({
                'frame': frame,
                'timestamp': time.time(),
                'metadata': frame_data.get('metadata', {})
            })
        except Queue.Full:
            # Queue full, drop oldest frame
            try:
                self.frame_queue.get_nowait()
                self.frame_queue.put_nowait({
                    'frame': frame,
                    'timestamp': time.time(),
                    'metadata': frame_data.get('metadata', {})
                })
            except (Queue.Full, Queue.Empty):
                pass
                
    def _process_frames(self) -> None:
        """Main frame processing loop"""
        while self.running:
            try:
                frame_data = self.frame_queue.get(timeout=0.1)
                
                start_time = time.time()
                
                # Process and annotate frame
                annotated_frame = self._annotate_frame(
                    frame_data['frame'],
                    frame_data.get('metadata', {})
                )
                
                # Update stats
                process_time = time.time() - start_time
                self._processing_times.append(process_time)
                if len(self._processing_times) > 100:
                    self._processing_times.pop(0)
                    
                # Calculate FPS
                if self.last_frame_time > 0:
                    fps = 1.0 / (time.time() - self.last_frame_time)
                    self._fps_window.append(fps)
                    
                self.last_frame_time = time.time()
                self.last_frame = annotated_frame
                self.frame_count += 1
                
                # Publish processed frame
                self.data_bus.publish(
                    DataType.SCREEN_CAPTURE,
                    {
                        'frame': annotated_frame,
                        'fps': np.mean(self._fps_window) if self._fps_window else 0,
                        'frame_count': self.frame_count
                    },
                    component="GameStreamer"
                )
                
            except Empty:
                continue
            except Exception as e:
                print(f"Error processing frame: {e}")
                continue
                
    def _annotate_frame(self, frame: np.ndarray, metadata: Dict[str, Any]) -> np.ndarray:
        """Annotate frame with game state and debug info
        
        Args:
            frame: Raw game frame
            metadata: Additional information to annotate
            
        Returns:
            Annotated frame
        """
        # Convert to PIL for text drawing
        pil_image = Image.fromarray(frame)
        draw = ImageDraw.Draw(pil_image)
        
        # Add timestamp
        timestamp = time.strftime("%H:%M:%S")
        draw.text((10, 10), timestamp, font=self.font, fill=(255, 255, 255))
        
        # Add FPS
        fps = np.mean(self._fps_window) if self._fps_window else 0
        draw.text((10, 30), f"FPS: {fps:.1f}", font=self.font, fill=(255, 255, 255))
        
        # Add metadata annotations
        y = 50
        for key, value in metadata.items():
            if isinstance(value, (int, float, str)):
                draw.text((10, y), f"{key}: {value}", font=self.font, fill=(255, 255, 255))
                y += 20
                
        # Convert back to numpy array
        return np.array(pil_image)
        
    def get_stream_stats(self) -> Dict[str, Any]:
        """Get streaming performance statistics
        
        Returns:
            Dictionary of streaming metrics
        """
        stats = {
            'frame_count': self.frame_count,
            'queue_size': self.frame_queue.qsize(),
            'fps': np.mean(self._fps_window) if self._fps_window else 0,
            'processing_time': {
                'mean': np.mean(self._processing_times) if self._processing_times else 0,
                'max': np.max(self._processing_times) if self._processing_times else 0
            }
        }
        return stats
