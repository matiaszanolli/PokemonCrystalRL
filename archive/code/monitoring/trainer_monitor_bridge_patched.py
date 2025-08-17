#!/usr/bin/env python3
"""
PATCHED - Bridge to connect UnifiedPokemonTrainer with PokemonRLWebMonitor.

This patched version includes fixes for:
- Socket connection stability 
- Blank screen issues
- Frame validation
- Better error handling
"""

import time
import threading
import numpy as np
import base64
import io
from typing import Optional, Dict, Any
from PIL import Image
from datetime import datetime

from .web_monitor import PokemonRLWebMonitor


class TrainerWebMonitorBridge:
    """
    PATCHED Bridge class with enhanced reliability.
    """
    
    def __init__(self, trainer, web_monitor: PokemonRLWebMonitor):
        self.trainer = trainer
        self.web_monitor = web_monitor
        self.bridge_active = False
        self.bridge_thread = None
        
        # Enhanced configuration for stability
        self.screenshot_update_interval = 1.0  # Slower for stability
        self.stats_update_interval = 3.0       # Less frequent
        self.bridge_fps = 5                    # Lower FPS for stability
        
        # Enhanced state tracking
        self.last_screenshot_update = 0
        self.last_stats_update = 0
        self.screenshot_count = 0
        self.error_count = 0
        self.last_frame_id = None
        self.last_timestamp = 0
        self._drop_count = 0
        
        print("🌉 PATCHED TrainerWebMonitorBridge initialized")
    
    def start_bridge(self):
        """Start the bridge thread with enhanced monitoring"""
        if self.bridge_active:
            print("⚠️ Bridge already active")
            return
        
        self.bridge_active = True
        self.bridge_thread = threading.Thread(target=self._enhanced_bridge_loop, daemon=True)
        self.bridge_thread.start()
        print("🌉 PATCHED bridge started with enhanced stability")
        
        # Start web monitor monitoring if not already started
        if not self.web_monitor.is_monitoring:
            self.web_monitor.start_monitoring()
    
    def _enhanced_bridge_loop(self):
        """Enhanced bridge loop with better error handling"""
        print("🔄 PATCHED bridge loop started")
        consecutive_errors = 0
        max_consecutive_errors = 10  # More tolerant
        
        while self.bridge_active:
            try:
                current_time = time.time()
                
                # Transfer screenshots with enhanced validation
                if current_time - self.last_screenshot_update >= self.screenshot_update_interval:
                    success = self._enhanced_transfer_screenshot()
                    if success:
                        consecutive_errors = 0
                        self.last_screenshot_update = current_time
                    else:
                        consecutive_errors += 1
                
                # Transfer training statistics
                if current_time - self.last_stats_update >= self.stats_update_interval:
                    self._transfer_training_stats()
                    self.last_stats_update = current_time
                
                # Sleep with error-based backoff
                sleep_time = 1.0 / self.bridge_fps
                if consecutive_errors > 0:
                    sleep_time *= (1 + consecutive_errors * 0.5)  # Backoff
                
                time.sleep(min(sleep_time, 2.0))  # Max 2 second delay
                
            except Exception as e:
                consecutive_errors += 1
                self.error_count += 1
                
                if consecutive_errors <= 3:
                    print(f"🌉 PATCHED bridge error #{consecutive_errors}: {e}")
                elif consecutive_errors == 4:
                    print(f"🌉 Suppressing further error messages (count: {consecutive_errors})")
                
                if consecutive_errors >= max_consecutive_errors:
                    print(f"❌ Too many consecutive errors ({consecutive_errors}), stopping bridge")
                    self.bridge_active = False
                    break
                
                # Progressive backoff
                error_sleep = min(5.0, 0.2 * (1.5 ** min(consecutive_errors, 15)))
                time.sleep(error_sleep)
        
        print("🔄 PATCHED bridge loop ended")
    
    def _enhanced_transfer_screenshot(self) -> bool:
        """Enhanced screenshot transfer with validation"""
        try:
            # Check trainer availability
            if not hasattr(self.trainer, 'latest_screen') or not self.trainer.latest_screen:
                return False
            
            screen_data = self.trainer.latest_screen
            
            # Enhanced validation
            if not self._validate_screenshot_data(screen_data):
                return False
            
            # Check for new frame
            current_frame_id = screen_data.get('frame_id', 0)
            current_timestamp = screen_data.get('timestamp', 0)
            
            # Skip duplicate frames
            if (current_frame_id == self.last_frame_id and 
                current_timestamp <= self.last_timestamp):
                return False
            
            self.last_frame_id = current_frame_id
            self.last_timestamp = current_timestamp
            
            # Convert with enhanced validation
            screenshot = self._enhanced_convert_screenshot(screen_data)
            if screenshot is None:
                return False
            
            # Final content validation
            if not self._validate_screenshot_content(screenshot):
                return False
            
            # Send to web monitor
            self.web_monitor.update_screenshot(screenshot)
            self.screenshot_count += 1
            
            # Progress logging
            if self.screenshot_count % 5 == 0:
                variance = screen_data.get('variance', 0)
                print(f"🖼️ PATCHED bridge: {self.screenshot_count} screenshots (variance: {variance:.1f})")
            
            return True
            
        except Exception as e:
            if self.error_count <= 10:
                print(f"⚠️ PATCHED screenshot transfer error: {e}")
            return False
    
    def _validate_screenshot_data(self, screen_data) -> bool:
        """Validate screenshot data structure"""
        try:
            if not isinstance(screen_data, dict):
                return False
            
            if 'image_b64' not in screen_data:
                return False
            
            img_b64 = screen_data['image_b64']
            if not img_b64 or len(img_b64) < 100:
                return False
            
            # Test base64 decode
            try:
                decoded = base64.b64decode(img_b64)
                if len(decoded) < 50:  # Too small
                    return False
            except Exception:
                return False
                
            return True
                
        except Exception:
            return False
    
    def _validate_screenshot_content(self, screenshot: np.ndarray) -> bool:
        """Enhanced screenshot content validation"""
        try:
            if screenshot is None or screenshot.size == 0:
                return False
            
            # Shape validation
            if len(screenshot.shape) != 3 or screenshot.shape[2] != 3:
                return False
            
            height, width = screenshot.shape[:2]
            if height < 50 or width < 50:
                return False
            
            # Content validation
            variance = np.var(screenshot.astype(np.float32))
            
            # Very strict blank detection
            if variance < 1.0:
                print(f"⚠️ PATCHED bridge: Screenshot appears blank (variance: {variance:.3f})")
                return False
            
            # Uniform color detection
            if np.all(screenshot == screenshot[0, 0]):
                print("⚠️ PATCHED bridge: Screenshot is uniform color")
                return False
            
            return True
            
        except Exception as e:
            print(f"⚠️ PATCHED content validation error: {e}")
            return False
    
    def _enhanced_convert_screenshot(self, screen_data: Dict[str, Any]) -> Optional[np.ndarray]:
        """Enhanced screenshot conversion with better error handling"""
        try:
            img_b64 = screen_data.get('image_b64')
            if not img_b64:
                return None
            
            # Decode with validation
            try:
                img_data = base64.b64decode(img_b64)
                if len(img_data) < 100:
                    return None
            except Exception as e:
                print(f"⚠️ Base64 decode error: {e}")
                return None
            
            # PIL processing with validation
            try:
                img = Image.open(io.BytesIO(img_data))
                
                # Validate image
                if img.size[0] < 50 or img.size[1] < 50:
                    return None
                
                # Convert to RGB if needed
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Convert to numpy
                screenshot = np.array(img, dtype=np.uint8)
                
                return screenshot
                
            except Exception as e:
                print(f"⚠️ PIL processing error: {e}")
                return None
            
        except Exception as e:
            print(f"⚠️ PATCHED screenshot conversion error: {e}")
            return None
    
    def stop_bridge(self):
        """Enhanced bridge shutdown"""
        if not self.bridge_active:
            return
            
        self.bridge_active = False
        print("⏹️ Stopping PATCHED bridge...")
        
        if self.bridge_thread and self.bridge_thread.is_alive():
            self.bridge_thread.join(timeout=10)  # Longer timeout
            if self.bridge_thread.is_alive():
                print("⚠️ PATCHED bridge thread did not stop gracefully")
            else:
                print("✅ PATCHED bridge thread stopped")
        
        # Enhanced statistics
        print(f"📊 PATCHED Bridge Statistics:")
        print(f"   Screenshots transferred: {self.screenshot_count}")
        print(f"   Errors encountered: {self.error_count}")
        print(f"   Frames dropped: {getattr(self, '_drop_count', 0)}")
        
        if self.screenshot_count > 0:
            success_rate = (self.screenshot_count / (self.screenshot_count + self.error_count)) * 100
            print(f"   Success rate: {success_rate:.1f}%")
    
    def _transfer_training_stats(self):
        """Transfer training statistics (unchanged from original)"""
        try:
            if not hasattr(self.trainer, 'stats'):
                return
            
            stats = self.trainer.stats
            
            # Create mock training session if needed
            if not hasattr(self.web_monitor, 'training_session') or not self.web_monitor.training_session:
                trainer_ref = self.trainer
                
                class MockEnv:
                    def get_game_state(self):
                        return {
                            'player': {
                                'x': getattr(trainer_ref, '_player_x', 10),
                                'y': getattr(trainer_ref, '_player_y', 8), 
                                'map': getattr(trainer_ref, '_current_map', 1),
                                'money': 0,
                                'badges': 0
                            },
                            'party': []
                        }
                
                class MockTrainingSession:
                    def __init__(self):
                        self.training_stats = {}
                        self.env = MockEnv()
                
                self.web_monitor.training_session = MockTrainingSession()
            
            # Update training stats
            training_stats = {
                'total_steps': stats.get('total_actions', 0),
                'episodes': stats.get('total_episodes', 0),
                'decisions_made': stats.get('llm_calls', 0),
                'visual_analyses': max(1, stats.get('total_actions', 0) // 5)
            }
            
            if hasattr(self.web_monitor.training_session, 'training_stats'):
                self.web_monitor.training_session.training_stats.update(training_stats)
        
        except Exception as e:
            if self.error_count <= 5:
                print(f"⚠️ PATCHED stats transfer error: {e}")
    
    def get_bridge_stats(self) -> Dict[str, Any]:
        """Get enhanced bridge performance statistics"""
        return {
            'is_active': self.bridge_active,
            'screenshots_transferred': self.screenshot_count,
            'total_errors': self.error_count,
            'frames_dropped': getattr(self, '_drop_count', 0),
            'last_screenshot_update': self.last_screenshot_update,
            'last_stats_update': self.last_stats_update,
            'screenshot_fps': 1.0 / self.screenshot_update_interval,
            'bridge_fps': self.bridge_fps,
            'success_rate': (self.screenshot_count / max(1, self.screenshot_count + self.error_count)) * 100
        }


def create_integrated_monitoring_system(trainer, host='127.0.0.1', port=5001):
    """
    PATCHED factory function with enhanced monitoring.
    """
    print("🏗️ Creating PATCHED integrated monitoring system...")
    
    # Create web monitor
    web_monitor = PokemonRLWebMonitor()
    
    # Create enhanced bridge
    bridge = TrainerWebMonitorBridge(trainer, web_monitor)
    
    # Start web monitor in background thread
    def run_monitor():
        web_monitor.run(host=host, port=port, debug=False)
    
    monitor_thread = threading.Thread(target=run_monitor, daemon=True)
    monitor_thread.start()
    
    # Wait for server to start
    time.sleep(2)  # Longer wait for stability
    
    print(f"🚀 PATCHED integrated monitoring system created:")
    print(f"   🌐 Web monitor: http://{host}:{port}")
    print(f"   🌉 Bridge: Enhanced reliability mode")
    print(f"   📊 Features: Frame validation, error recovery, progress tracking")
    
    return web_monitor, bridge, monitor_thread
