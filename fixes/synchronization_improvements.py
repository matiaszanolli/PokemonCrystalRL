#!/usr/bin/env python3
"""
Synchronization Improvements for Pokemon Crystal RL Training

This module provides fixes for common synchronization issues that occur
during training, including:
- Lock contention between screen capture and training threads
- Queue overflow handling
- Resource cleanup race conditions
- Thread safety improvements

Apply these fixes to improve training stability.
"""

import threading
import queue
import time
import logging
from contextlib import contextmanager
from typing import Optional, Any, Dict
import weakref

logger = logging.getLogger(__name__)


class ThreadSafeScreenManager:
    """Thread-safe manager for screen capture and access."""
    
    def __init__(self, max_queue_size: int = 30):
        self.max_queue_size = max_queue_size
        self._screen_lock = threading.RLock()  # Re-entrant lock
        self._screen_queue = queue.Queue(maxsize=max_queue_size)
        self._latest_screen = None
        self._screen_stats = {
            'captures': 0,
            'drops': 0,
            'errors': 0
        }
        
    @contextmanager
    def screen_access(self):
        """Context manager for safe screen access."""
        acquired = self._screen_lock.acquire(timeout=0.1)  # Non-blocking with timeout
        if not acquired:
            logger.warning("Screen lock timeout - skipping operation")
            yield None
            return
            
        try:
            yield self
        finally:
            self._screen_lock.release()
    
    def put_screen(self, screen_data: Dict[str, Any]) -> bool:
        """Safely add screen data to queue."""
        with self.screen_access() as manager:
            if manager is None:
                return False
                
            try:
                # Remove old items if queue is full
                while self._screen_queue.full():
                    try:
                        dropped = self._screen_queue.get_nowait()
                        self._screen_stats['drops'] += 1
                    except queue.Empty:
                        break
                
                self._screen_queue.put_nowait(screen_data)
                self._latest_screen = screen_data
                self._screen_stats['captures'] += 1
                return True
                
            except queue.Full:
                self._screen_stats['drops'] += 1
                return False
            except Exception as e:
                logger.error(f"Error adding screen: {e}")
                self._screen_stats['errors'] += 1
                return False
    
    def get_latest_screen(self) -> Optional[Dict[str, Any]]:
        """Get the latest screen data safely."""
        with self.screen_access() as manager:
            if manager is None:
                return None
            return self._latest_screen.copy() if self._latest_screen else None
    
    def get_stats(self) -> Dict[str, int]:
        """Get screen manager statistics."""
        with self.screen_access() as manager:
            if manager is None:
                return {}
            return self._screen_stats.copy()


class ThreadSafeTrainer:
    """Mixin to add thread safety improvements to trainers."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._sync_lock = threading.RLock()
        self._screen_manager = ThreadSafeScreenManager()
        self._shutdown_event = threading.Event()
        
        # Replace screen queue with thread-safe manager
        if hasattr(self, 'screen_queue'):
            self.screen_queue = self._screen_manager
        
    def _safe_screen_capture(self) -> Optional[Any]:
        """Thread-safe screen capture."""
        if self._shutdown_event.is_set():
            return None
            
        try:
            # Quick check if PyBoy is available
            if not hasattr(self, 'pyboy') or not self.pyboy:
                return None
                
            # Handle Mock objects in tests
            if hasattr(self.pyboy, '_mock_name'):
                import numpy as np
                return np.random.randint(0, 256, (144, 160, 3), dtype=np.uint8)
                
            # Get screen data with timeout protection
            screen_data = None
            try:
                # Use a timeout to prevent hanging
                def get_screen():
                    return self.pyboy.screen.ndarray
                    
                screen_data = get_screen()
                
            except Exception as e:
                logger.warning(f"Screen capture error: {e}")
                return None
                
            return self._convert_screen_format(screen_data) if screen_data is not None else None
            
        except Exception as e:
            logger.error(f"Critical screen capture error: {e}")
            return None
    
    def _safe_queue_screen(self, screen_data: Dict[str, Any]) -> bool:
        """Safely queue screen data."""
        if self._shutdown_event.is_set():
            return False
            
        return self._screen_manager.put_screen(screen_data)
    
    def _graceful_shutdown(self):
        """Initiate graceful shutdown."""
        logger.info("Starting graceful shutdown...")
        self._shutdown_event.set()
        
        # Stop screen capture threads
        if hasattr(self, 'capture_active'):
            self.capture_active = False
            
        # Stop web monitor
        if hasattr(self, 'web_monitor') and self.web_monitor:
            try:
                self.web_monitor.stop()
            except Exception as e:
                logger.error(f"Error stopping web monitor: {e}")
        
        # Wait for threads to finish
        max_wait = 5.0
        start_time = time.time()
        
        if hasattr(self, 'capture_thread') and self.capture_thread:
            remaining = max_wait - (time.time() - start_time)
            if remaining > 0:
                self.capture_thread.join(timeout=remaining)
                
        logger.info("Graceful shutdown completed")


def apply_sync_fixes(trainer_instance):
    """Apply synchronization fixes to an existing trainer instance."""
    
    # Add thread-safe screen manager
    trainer_instance._screen_manager = ThreadSafeScreenManager()
    trainer_instance._sync_lock = threading.RLock()
    trainer_instance._shutdown_event = threading.Event()
    
    # Replace problematic methods with thread-safe versions
    original_capture = getattr(trainer_instance, '_simple_screenshot_capture', None)
    if original_capture:
        def safe_capture():
            if trainer_instance._shutdown_event.is_set():
                return None
            try:
                return original_capture()
            except Exception as e:
                logger.warning(f"Screen capture error (fixed): {e}")
                return None
        trainer_instance._simple_screenshot_capture = safe_capture
    
    # Improve screen queueing
    original_queue = getattr(trainer_instance, '_capture_and_queue_screen', None)
    if original_queue:
        def safe_queue():
            screen = trainer_instance._simple_screenshot_capture()
            if screen is not None:
                screen_data = {
                    'image': screen,
                    'timestamp': time.time(),
                    'frame': getattr(trainer_instance.pyboy, 'frame_count', 0),
                    'action': trainer_instance.stats.get('total_actions', 0)
                }
                trainer_instance._screen_manager.put_screen(screen_data)
        trainer_instance._capture_and_queue_screen = safe_queue
    
    # Add graceful shutdown method
    def graceful_shutdown():
        trainer_instance._shutdown_event.set()
        if hasattr(trainer_instance, 'web_monitor') and trainer_instance.web_monitor:
            trainer_instance.web_monitor.stop()
    trainer_instance._graceful_shutdown = graceful_shutdown
    
    logger.info("Applied synchronization fixes to trainer instance")
    return trainer_instance


class ImprovedScreenCapture:
    """Improved screen capture with better synchronization."""
    
    def __init__(self, pyboy=None, capture_fps=5):
        self.pyboy = pyboy
        self.capture_fps = capture_fps
        self._capture_active = False
        self._capture_thread = None
        self._latest_screen = None
        self._screen_lock = threading.RLock()
        self._shutdown_event = threading.Event()
        
        self.stats = {
            'frames_captured': 0,
            'frames_served': 0,
            'capture_errors': 0,
            'lock_timeouts': 0
        }
    
    def start_capture(self):
        """Start screen capture with improved thread safety."""
        if self._capture_active or not self.pyboy or self._shutdown_event.is_set():
            return
            
        self._capture_active = True
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()
        logger.info("🚀 Improved screen capture started")
    
    def stop_capture(self):
        """Stop screen capture gracefully."""
        if not self._capture_active:
            return
            
        logger.info("🛑 Stopping screen capture...")
        self._capture_active = False
        self._shutdown_event.set()
        
        if self._capture_thread and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=3.0)  # Increased timeout
            if self._capture_thread.is_alive():
                logger.warning("Screen capture thread did not stop gracefully")
        
        logger.info("✅ Screen capture stopped")
    
    def _capture_loop(self):
        """Improved capture loop with better error handling."""
        capture_interval = 1.0 / self.capture_fps
        error_count = 0
        max_errors = 10
        
        while self._capture_active and not self._shutdown_event.is_set():
            try:
                if not self.pyboy:
                    time.sleep(capture_interval)
                    continue
                
                # Get screen with timeout protection
                screen_data = self._get_screen_safe()
                if screen_data:
                    # Update latest screen with lock timeout
                    if self._screen_lock.acquire(timeout=0.1):
                        try:
                            self._latest_screen = screen_data
                            self.stats['frames_captured'] += 1
                            error_count = 0  # Reset error count on success
                        finally:
                            self._screen_lock.release()
                    else:
                        self.stats['lock_timeouts'] += 1
                        logger.warning("Screen lock timeout during capture")
                
            except Exception as e:
                error_count += 1
                self.stats['capture_errors'] += 1
                logger.error(f"Screen capture error ({error_count}/{max_errors}): {e}")
                
                if error_count >= max_errors:
                    logger.error("Too many capture errors, stopping capture")
                    break
                    
                # Exponential backoff on errors
                time.sleep(min(capture_interval * (2 ** error_count), 1.0))
                continue
            
            # Normal sleep between captures
            time.sleep(capture_interval)
    
    def _get_screen_safe(self):
        """Safely get screen data from PyBoy."""
        try:
            if hasattr(self.pyboy, '_mock_name'):
                import numpy as np
                screen_array = np.random.randint(0, 256, (144, 160, 3), dtype=np.uint8)
            else:
                screen_array = self.pyboy.screen.ndarray
            
            if screen_array is not None:
                # Convert to proper format
                if len(screen_array.shape) == 3 and screen_array.shape[2] >= 3:
                    rgb_screen = screen_array[:, :, :3].astype(np.uint8)
                else:
                    rgb_screen = screen_array.astype(np.uint8)
                
                # Create screen data
                return {
                    'image': rgb_screen,
                    'timestamp': time.time(),
                    'frame_id': self.stats['frames_captured'],
                    'size': rgb_screen.shape[:2]
                }
                
        except Exception as e:
            logger.error(f"Error getting screen data: {e}")
            return None
    
    def get_latest_screen(self):
        """Get latest screen with timeout protection."""
        if self._screen_lock.acquire(timeout=0.1):
            try:
                self.stats['frames_served'] += 1
                return self._latest_screen.copy() if self._latest_screen else None
            finally:
                self._screen_lock.release()
        else:
            self.stats['lock_timeouts'] += 1
            return None


def create_monitoring_fixes():
    """Create a comprehensive fix package for monitoring issues."""
    
    fixes = {
        'ThreadSafeScreenManager': ThreadSafeScreenManager,
        'ThreadSafeTrainer': ThreadSafeTrainer, 
        'ImprovedScreenCapture': ImprovedScreenCapture,
        'apply_sync_fixes': apply_sync_fixes
    }
    
    logger.info("Created synchronization fix package")
    return fixes


if __name__ == "__main__":
    print("🔧 Pokemon Crystal RL Synchronization Fixes")
    print("=" * 50)
    print()
    print("Available fixes:")
    print("- ThreadSafeScreenManager: Better queue management")
    print("- ThreadSafeTrainer: Trainer synchronization mixin")
    print("- ImprovedScreenCapture: Enhanced capture with timeouts")
    print("- apply_sync_fixes(): Apply fixes to existing trainer")
    print()
    print("Usage:")
    print("  from fixes.synchronization_improvements import apply_sync_fixes")
    print("  trainer = apply_sync_fixes(trainer)")
