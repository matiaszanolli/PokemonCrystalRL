#!/usr/bin/env python3
"""
Screen Capture Module for Pokemon Crystal RL

Handles game screen capture with error recovery and WebSocket streaming.
Extracted from web_monitor.py for better organization.
"""

import os
import threading
import queue
import base64
import json
import time
import io
import logging
import numpy as np
from PIL import Image
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class ScreenCapture:
    """Handles game screen capture with error recovery"""
    
    def __init__(self, pyboy=None):
        self.pyboy = pyboy
        self.latest_screen = None
        self.latest_frame = None  # For WebSocket streaming
        self.capture_thread = None
        self.capture_active = False
        self.capture_queue = queue.Queue(maxsize=10)
        self.stats = {
            'frames_captured': 0,
            'frames_served': 0,
            'capture_errors': 0
        }
        self._lock = threading.Lock()
        self.ws_clients = None  # Will be set by WebMonitor
    
    def start_capture(self):
        """Start screen capture thread"""
        if self.capture_active or not self.pyboy:
            return
        
        self.capture_active = True
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        logger.info("📸 Screen capture started")
    
    def stop_capture(self):
        """Stop screen capture"""
        self.capture_active = False
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2.0)
        logger.info("📸 Screen capture stopped")
    
    def _capture_loop(self):
        """Main capture loop with improved error handling"""
        capture_interval = 0.2  # 5 FPS
        error_count = 0
        max_consecutive_errors = 5
        
        while self.capture_active and self.pyboy:
            try:
                # Get screen from PyBoy with timeout protection
                screen_array = None
                try:
                    screen_array = self.pyboy.screen.ndarray
                except Exception as screen_e:
                    logger.warning(f"PyBoy screen access error: {screen_e}")
                    error_count += 1
                    if error_count >= max_consecutive_errors:
                        logger.error("Too many screen access errors, stopping capture")
                        break
                    time.sleep(capture_interval * 2)  # Longer wait on error
                    continue
                
                if screen_array is not None:
                    try:
                        # Debug: Log screen capture success
                        if self.stats['frames_captured'] % 50 == 0:  # Log every 50 frames
                            logger.info(f"📸 Screen captured: shape={screen_array.shape}, frame={self.stats['frames_captured']}")
                        
                        # Convert to PIL Image
                        if len(screen_array.shape) == 3 and screen_array.shape[2] >= 3:
                            # RGB/RGBA
                            rgb_screen = screen_array[:, :, :3].astype(np.uint8)
                        else:
                            rgb_screen = screen_array.astype(np.uint8)
                        
                        # Create PIL image and resize for web
                        pil_image = Image.fromarray(rgb_screen)
                        resized = pil_image.resize((320, 288), Image.NEAREST)
                        
                        # Convert to base64 for web transfer
                        buffer = io.BytesIO()
                        resized.save(buffer, format='PNG', optimize=True)
                        img_b64 = base64.b64encode(buffer.getvalue()).decode()
                        
                        # Update latest screen with timeout (non-blocking)
                        if self._lock.acquire(timeout=0.05):  # 50ms timeout
                            try:
                                self.latest_screen = {
                                    'image_b64': img_b64,
                                    'timestamp': time.time(),
                                    'size': resized.size,
                                    'frame_id': self.stats['frames_captured'],
                                    'data_length': len(img_b64)
                                }
                                # Store raw PNG bytes for WebSocket streaming
                                self.latest_frame = buffer.getvalue()
                                self.stats['frames_captured'] += 1
                                error_count = 0  # Reset on success
                                
                                # Broadcast to WebSocket clients (non-blocking)
                                self._broadcast_to_websockets()
                            finally:
                                self._lock.release()
                        else:
                            logger.debug("Screen update skipped due to lock timeout")
                            
                    except Exception as process_e:
                        logger.warning(f"Screen processing error: {process_e}")
                        error_count += 1
            
            except Exception as e:
                self.stats['capture_errors'] += 1
                error_count += 1
                logger.warning(f"Screen capture error ({error_count}/{max_consecutive_errors}): {e}")
                
                if error_count >= max_consecutive_errors:
                    logger.error("Too many capture errors, stopping")
                    break
            
            # Wait with exponential backoff on errors
            if error_count > 0:
                time.sleep(min(capture_interval * (1.5 ** error_count), 2.0))
            else:
                time.sleep(capture_interval)
    
    def _broadcast_to_websockets(self):
        """Broadcast latest frame to WebSocket clients (non-blocking)"""
        if not self.ws_clients or not self.latest_frame:
            return
        
        # Store frame in a simple queue for WebSocket handler to pickup
        # This avoids complex async synchronization from capture thread
        if not hasattr(self, '_frame_queue'):
            self._frame_queue = queue.Queue(maxsize=5)
        
        try:
            # Add latest frame to queue (non-blocking)
            self._frame_queue.put_nowait(self.latest_frame)
        except queue.Full:
            # Queue full, drop oldest frame
            try:
                self._frame_queue.get_nowait()
                self._frame_queue.put_nowait(self.latest_frame)
            except queue.Empty:
                pass
    
    def get_latest_screen_bytes(self):
        """Get latest screen as PNG bytes with timeout protection"""
        if self._lock.acquire(timeout=0.1):  # 100ms timeout
            try:
                if not self.latest_screen:
                    return None
                
                img_b64 = self.latest_screen['image_b64']
                img_bytes = base64.b64decode(img_b64)
                self.stats['frames_served'] += 1
                return img_bytes
            except Exception as e:
                logger.warning(f"Error getting screen bytes: {e}")
                return None
            finally:
                self._lock.release()
        else:
            logger.debug("Screen access skipped due to lock timeout")
            return None
    
    def get_latest_screen_data(self):
        """Get latest screen metadata with timeout protection"""
        if self._lock.acquire(timeout=0.1):  # 100ms timeout
            try:
                return self.latest_screen.copy() if self.latest_screen else None
            finally:
                self._lock.release()
        else:
            return None

    def get_stats(self) -> Dict[str, Any]:
        """Get capture statistics"""
        return self.stats.copy()
    
    def start(self):
        """Start screen capture - alias for start_capture()"""
        return self.start_capture()
    
    def stop(self):
        """Stop screen capture - alias for stop_capture()"""
        return self.stop_capture()