"""Mock screen capture for testing."""

import time
import threading
import queue
import numpy as np
from dataclasses import dataclass
from typing import Optional, Any, Dict, List

@dataclass
class MockCaptureConfig:
    """Mock screen capture configuration."""
    enabled: bool = True
    fps: int = 10
    width: int = 160
    height: int = 144
    channels: int = 3
    queue_size: int = 30
    
    @classmethod
    def from_training_config(cls, config: Any) -> 'MockCaptureConfig':
        """Create from training config."""
        return cls(
            enabled=getattr(config, 'capture_screens', True),
            fps=getattr(config, 'capture_fps', 10),
            queue_size=30  # Fixed size for tests
        )

class MockScreenCapture:
    """Mock screen capture system for testing.
    
    Simulates screen capture without actual screen access or threads.
    """
    
    def __init__(self, config: MockCaptureConfig):
        """Initialize mock capture."""
        self.config = config
        self._running = False
        self.screen_queue = queue.Queue(maxsize=config.queue_size)
        self.latest_screen = None
        self.frame_count = 0
        
        # State tracking for tests
        self.capture_calls: List[float] = []  # Timestamps of capture calls
        self.error_count = 0
        self.mock_frames: List[np.ndarray] = []  # Pre-loaded frames for tests
        
    def start(self) -> bool:
        """Start mock capture."""
        if not self.config.enabled:
            return False
        self._running = True
        return True
        
    def stop(self) -> None:
        """Stop mock capture."""
        self._running = False
        
    def get_current_frame(self) -> Optional[np.ndarray]:
        """Get current mock frame.
        
        Returns a random frame if no mock frames are loaded.
        """
        if not self._running:
            return None
            
        # Record capture call
        self.capture_calls.append(time.time())
        
        # Return mock frame if available
        if self.mock_frames:
            frame = self.mock_frames[self.frame_count % len(self.mock_frames)]
            self.frame_count += 1
            return frame
            
        # Generate random frame
        return np.random.randint(
            0, 255,
            (self.config.height, self.config.width, self.config.channels),
            dtype=np.uint8
        )
        
    def queue_frame(self, frame_data: Dict[str, Any]) -> bool:
        """Queue a frame update."""
        if not self._running:
            return False
            
        try:
            # Remove oldest if full
            if self.screen_queue.full():
                try:
                    self.screen_queue.get_nowait()
                except queue.Empty:
                    pass
                    
            # Add new frame
            self.screen_queue.put_nowait(frame_data)
            self.latest_screen = frame_data
            return True
            
        except queue.Full:
            self.error_count += 1
            return False
            
    def load_mock_frames(self, frames: List[np.ndarray]) -> None:
        """Load pre-defined frames for testing."""
        self.mock_frames = frames
        self.frame_count = 0
        
    def clear_mock_frames(self) -> None:
        """Clear pre-defined frames."""
        self.mock_frames.clear()
        self.frame_count = 0
        
    def reset_state(self) -> None:
        """Reset all state for testing."""
        self.capture_calls.clear()
        self.error_count = 0
        self.clear_mock_frames()
        self.frame_count = 0
        while not self.screen_queue.empty():
            try:
                self.screen_queue.get_nowait()
            except queue.Empty:
                break
        self.latest_screen = None
