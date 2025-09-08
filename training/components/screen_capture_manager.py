"""
Screen Capture Manager - Screen capture and queue management

Extracted from PokemonTrainer to handle screen capture, queue management,
image encoding, and web display coordination.
"""

import time
import queue
import logging
import threading
import numpy as np
from typing import Dict, Any, Optional, List
from dataclasses import dataclass


@dataclass
class ScreenCaptureConfig:
    """Configuration for screen capture system."""
    enabled: bool = True
    fps: int = 10
    queue_size: int = 30
    resize_dimensions: tuple = (320, 288)
    jpeg_quality: int = 85
    auto_cleanup: bool = True
    capture_format: str = "rgb"  # "rgb", "bgr", or "grayscale"


class ScreenCaptureManager:
    """Manages screen capture, encoding, and queue management for monitoring."""
    
    def __init__(self, config: ScreenCaptureConfig, emulation_manager=None):
        self.config = config
        self.emulation_manager = emulation_manager
        self.logger = logging.getLogger("ScreenCaptureManager")
        
        # Queue management
        self.screen_queue = queue.Queue(maxsize=config.queue_size)
        self.screenshot_queue = []  # List for backward compatibility
        self.latest_screen = None
        
        # Capture control
        self.capture_active = False
        self.capture_thread = None
        self._shutdown_event = threading.Event()
        self._lock = threading.RLock()
        
        # Performance tracking
        self.capture_stats = {
            'frames_captured': 0,
            'frames_dropped': 0,
            'encoding_failures': 0,
            'queue_overflows': 0,
            'start_time': time.time()
        }
        
        # Screen state tracking
        self.last_screen_hash = None
        self.consecutive_same_screens = 0
        
        self.logger.info(f"Screen capture manager initialized (fps={config.fps}, queue_size={config.queue_size})")
    
    def start_capture(self) -> bool:
        """Start the screen capture thread.
        
        Returns:
            bool: True if capture started successfully
        """
        if not self.config.enabled:
            self.logger.info("Screen capture disabled in configuration")
            return False
        
        if self.capture_active:
            self.logger.warning("Screen capture already active")
            return True
        
        if not self.emulation_manager or not self.emulation_manager.is_alive():
            self.logger.error("Emulation manager not available or not alive")
            return False
        
        try:
            self.capture_active = True
            self._shutdown_event.clear()
            
            self.capture_thread = threading.Thread(
                target=self._capture_loop,
                daemon=True,
                name="ScreenCaptureThread"
            )
            self.capture_thread.start()
            
            self.logger.info("📸 Screen capture started")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start screen capture: {e}")
            self.capture_active = False
            return False
    
    def stop_capture(self):
        """Stop the screen capture thread."""
        if not self.capture_active:
            return
        
        self.logger.info("Stopping screen capture...")
        self.capture_active = False
        self._shutdown_event.set()
        
        # Wait for thread to stop
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2.0)
            if self.capture_thread.is_alive():
                self.logger.warning("Screen capture thread did not stop within timeout")
        
        # Clear queues
        self._clear_queues()
        
        self.logger.info("📸 Screen capture stopped")
    
    def _capture_loop(self):
        """Main screen capture loop running in separate thread."""
        capture_interval = 1.0 / self.config.fps if self.config.fps > 0 else 0.1
        
        self.logger.debug(f"Screen capture loop started (interval={capture_interval:.3f}s)")
        
        try:
            while self.capture_active and not self._shutdown_event.is_set():
                try:
                    start_time = time.time()
                    
                    # Capture screen from emulation
                    screen_data = self._capture_screen()
                    
                    if screen_data is not None:
                        # Process and queue the screen
                        processed_screen = self._process_screen(screen_data)
                        if processed_screen:
                            self._queue_screen(processed_screen)
                            self.capture_stats['frames_captured'] += 1
                        else:
                            self.capture_stats['encoding_failures'] += 1
                    else:
                        self.capture_stats['frames_dropped'] += 1
                    
                    # Maintain capture rate
                    elapsed = time.time() - start_time
                    sleep_time = max(0, capture_interval - elapsed)
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                        
                except Exception as e:
                    self.logger.error(f"Screen capture error: {e}")
                    # Brief pause on error to prevent tight error loops
                    time.sleep(0.1)
                    
        except Exception as e:
            self.logger.error(f"Screen capture loop failed: {e}")
        finally:
            self.capture_active = False
            self.logger.debug("Screen capture loop ended")
    
    def _capture_screen(self) -> Optional[np.ndarray]:
        """Capture screen from emulation manager.
        
        Returns:
            Optional[np.ndarray]: Screen data or None if unavailable
        """
        if not self.emulation_manager or not self.emulation_manager.is_alive():
            return None
        
        try:
            # Get screen array from emulation
            screen_array = self.emulation_manager.get_screen_array()
            
            if screen_array is not None:
                # Convert format if needed
                if self.config.capture_format == "grayscale" and len(screen_array.shape) == 3:
                    # Convert to grayscale
                    screen_array = np.mean(screen_array, axis=2).astype(np.uint8)
                elif self.config.capture_format == "bgr" and len(screen_array.shape) == 3:
                    # Convert RGB to BGR
                    screen_array = screen_array[:, :, ::-1]
                
                return screen_array
            
        except Exception as e:
            self.logger.debug(f"Screen capture failed: {e}")
        
        return None
    
    def _process_screen(self, screen_data: np.ndarray) -> Optional[Dict[str, Any]]:
        """Process captured screen data.
        
        Args:
            screen_data: Raw screen data from emulation
            
        Returns:
            Optional[Dict]: Processed screen data or None if processing failed
        """
        try:
            # Resize if configured
            processed_screen = screen_data
            if (self.config.resize_dimensions and 
                screen_data.shape[:2] != self.config.resize_dimensions):
                processed_screen = self._resize_screen(screen_data)
            
            # Generate hash for change detection
            screen_hash = self._generate_screen_hash(processed_screen)
            
            # Check for duplicate screens
            is_duplicate = (screen_hash == self.last_screen_hash)
            if is_duplicate:
                self.consecutive_same_screens += 1
            else:
                self.consecutive_same_screens = 0
                self.last_screen_hash = screen_hash
            
            # Encode image for web display
            encoded_image = self._encode_image(processed_screen)
            
            # Create screen data package
            screen_package = {
                'image': processed_screen,
                'timestamp': time.time(),
                'frame_id': getattr(self.emulation_manager, 'get_frame_count', lambda: 0)(),
                'hash': screen_hash,
                'is_duplicate': is_duplicate,
                'consecutive_duplicates': self.consecutive_same_screens,
                'data_length': len(encoded_image) if encoded_image else 0
            }
            
            # Add encoded image if available
            if encoded_image:
                screen_package['image_b64'] = encoded_image
            
            return screen_package
            
        except Exception as e:
            self.logger.error(f"Screen processing failed: {e}")
            return None
    
    def _resize_screen(self, screen_data: np.ndarray) -> np.ndarray:
        """Resize screen data to configured dimensions."""
        try:
            # Try OpenCV first (faster)
            import cv2
            return cv2.resize(
                screen_data, 
                self.config.resize_dimensions, 
                interpolation=cv2.INTER_NEAREST
            )
        except ImportError:
            # Fallback to PIL
            try:
                from PIL import Image
                if len(screen_data.shape) == 3:
                    img = Image.fromarray(screen_data)
                else:
                    img = Image.fromarray(screen_data, mode='L')
                
                resized_img = img.resize(self.config.resize_dimensions, Image.NEAREST)
                return np.array(resized_img)
            except ImportError:
                self.logger.warning("No resize library available, using original size")
                return screen_data
    
    def _generate_screen_hash(self, screen_data: np.ndarray) -> int:
        """Generate hash for screen change detection."""
        try:
            # Sample every 4th pixel for performance
            sampled = screen_data[::4, ::4]
            return hash(sampled.tobytes())
        except Exception:
            return hash(str(screen_data.shape))
    
    def _encode_image(self, screen_data: np.ndarray) -> Optional[str]:
        """Encode screen image as base64 JPEG.
        
        Args:
            screen_data: Screen image data
            
        Returns:
            Optional[str]: Base64 encoded JPEG or None if encoding failed
        """
        try:
            # Try PIL first (most reliable)
            from PIL import Image
            import io
            import base64
            
            # Convert to PIL Image
            if len(screen_data.shape) == 3:
                pil_image = Image.fromarray(screen_data.astype('uint8'), mode='RGB')
            else:
                pil_image = Image.fromarray(screen_data.astype('uint8'), mode='L')
            
            # Encode as JPEG
            buf = io.BytesIO()
            pil_image.save(buf, format='JPEG', quality=self.config.jpeg_quality)
            buf.seek(0)
            
            # Encode to base64
            encoded = base64.b64encode(buf.getvalue()).decode('utf-8')
            return encoded
            
        except ImportError:
            # Fallback to OpenCV
            try:
                import cv2
                import base64
                
                # Ensure correct color format for OpenCV
                if len(screen_data.shape) == 3 and screen_data.shape[2] == 3:
                    # Convert RGB to BGR for OpenCV
                    bgr_image = cv2.cvtColor(screen_data, cv2.COLOR_RGB2BGR)
                else:
                    bgr_image = screen_data
                
                # Encode as JPEG
                ret, jpg_data = cv2.imencode(
                    '.jpg', 
                    bgr_image, 
                    [cv2.IMWRITE_JPEG_QUALITY, self.config.jpeg_quality]
                )
                
                if ret and jpg_data is not None:
                    encoded = base64.b64encode(jpg_data).decode('utf-8')
                    return encoded
                    
            except ImportError:
                pass
        
        except Exception as e:
            self.logger.debug(f"Image encoding failed: {e}")
        
        return None
    
    def _queue_screen(self, screen_data: Dict[str, Any]) -> bool:
        """Queue processed screen data.
        
        Args:
            screen_data: Processed screen data package
            
        Returns:
            bool: True if queued successfully
        """
        try:
            with self._lock:
                # Handle queue overflow
                while self.screen_queue.full():
                    try:
                        self.screen_queue.get_nowait()
                        self.capture_stats['queue_overflows'] += 1
                    except queue.Empty:
                        break
                
                # Queue new screen
                self.screen_queue.put_nowait(screen_data)
                self.latest_screen = screen_data
                
                # Mirror to list queue for backward compatibility
                self.screenshot_queue.append(screen_data)
                if len(self.screenshot_queue) > self.config.queue_size:
                    self.screenshot_queue = self.screenshot_queue[-self.config.queue_size:]
                
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to queue screen: {e}")
            return False
    
    def get_latest_screen(self) -> Optional[Dict[str, Any]]:
        """Get the most recent screen capture.
        
        Returns:
            Optional[Dict]: Latest screen data or None if no captures
        """
        with self._lock:
            return self.latest_screen.copy() if self.latest_screen else None
    
    def get_screen_queue_size(self) -> int:
        """Get current number of items in screen queue.
        
        Returns:
            int: Number of queued screens
        """
        return self.screen_queue.qsize()
    
    def get_capture_stats(self) -> Dict[str, Any]:
        """Get capture performance statistics.
        
        Returns:
            Dict: Capture statistics
        """
        elapsed = time.time() - self.capture_stats['start_time']
        
        stats = self.capture_stats.copy()
        stats.update({
            'capture_active': self.capture_active,
            'queue_size': self.screen_queue.qsize(),
            'elapsed_time': elapsed,
            'fps_actual': self.capture_stats['frames_captured'] / max(elapsed, 1),
            'drop_rate': (self.capture_stats['frames_dropped'] / 
                         max(self.capture_stats['frames_captured'] + self.capture_stats['frames_dropped'], 1)),
            'consecutive_same_screens': self.consecutive_same_screens
        })
        
        return stats
    
    def _clear_queues(self):
        """Clear all queues and reset state."""
        with self._lock:
            # Clear screen queue
            try:
                while not self.screen_queue.empty():
                    self.screen_queue.get_nowait()
            except queue.Empty:
                pass
            
            # Clear list queue
            self.screenshot_queue.clear()
            
            # Reset state
            self.latest_screen = None
            self.last_screen_hash = None
            self.consecutive_same_screens = 0
    
    def is_active(self) -> bool:
        """Check if capture is currently active.
        
        Returns:
            bool: True if capture is running
        """
        return self.capture_active
    
    def force_capture(self) -> Optional[Dict[str, Any]]:
        """Force immediate screen capture outside normal loop.
        
        Returns:
            Optional[Dict]: Captured screen data or None if failed
        """
        if not self.emulation_manager or not self.emulation_manager.is_alive():
            return None
        
        try:
            screen_data = self._capture_screen()
            if screen_data is not None:
                return self._process_screen(screen_data)
        except Exception as e:
            self.logger.error(f"Force capture failed: {e}")
        
        return None
    
    def __enter__(self):
        """Context manager entry."""
        self.start_capture()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_capture()