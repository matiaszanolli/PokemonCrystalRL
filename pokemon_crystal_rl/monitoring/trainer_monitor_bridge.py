"""
Bridge to connect UnifiedPokemonTrainer with PokemonRLWebMonitor.

This module solves the blank screen issue by connecting the trainer's screenshot
capture system with the web monitor's WebSocket streaming system.
"""

import time
import threading
import numpy as np
import base64
import io
from typing import Optional, Dict, Any
from PIL import Image

from .web_monitor import PokemonRLWebMonitor


class TrainerWebMonitorBridge:
    """
    Bridge class that connects UnifiedPokemonTrainer with PokemonRLWebMonitor.
    
    Solves the blank screen issue by:
    1. Monitoring the trainer's latest_screen data
    2. Converting it to the format expected by the web monitor
    3. Calling update_screenshot() on the web monitor
    4. Transferring training statistics between systems
    """
    
    def __init__(self, trainer, web_monitor: PokemonRLWebMonitor):
        """
        Initialize the bridge.
        
        Args:
            trainer: UnifiedPokemonTrainer instance
            web_monitor: PokemonRLWebMonitor instance
        """
        self.trainer = trainer
        self.web_monitor = web_monitor
        self.bridge_active = False
        self.bridge_thread = None
        
        # Bridge configuration
        self.screenshot_update_interval = 0.5  # Update screenshots every 500ms
        self.stats_update_interval = 2.0       # Update stats every 2 seconds
        self.bridge_fps = 10                   # Bridge update rate (10 FPS)
        
        # Internal state tracking
        self.last_screenshot_update = 0
        self.last_stats_update = 0
        self.screenshot_count = 0
        self.error_count = 0
        self.last_frame_id = None
        
        print("🌉 TrainerWebMonitorBridge initialized")
    
    def start_bridge(self):
        """Start the bridge thread to connect trainer and web monitor."""
        if self.bridge_active:
            print("⚠️ Bridge already active")
            return
        
        self.bridge_active = True
        self.bridge_thread = threading.Thread(target=self._bridge_loop, daemon=True)
        self.bridge_thread.start()
        print("🌉 Trainer-WebMonitor bridge started")
        
        # Start web monitor monitoring if not already started
        if not self.web_monitor.is_monitoring:
            self.web_monitor.start_monitoring()
    
    def stop_bridge(self):
        """Stop the bridge thread."""
        if not self.bridge_active:
            return
            
        self.bridge_active = False
        print("⏹️ Stopping bridge...")
        
        if self.bridge_thread and self.bridge_thread.is_alive():
            self.bridge_thread.join(timeout=5)
            if self.bridge_thread.is_alive():
                print("⚠️ Bridge thread did not stop gracefully")
            else:
                print("✅ Bridge thread stopped")
        
        # Print bridge statistics
        print(f"📊 Bridge Statistics:")
        print(f"   Screenshots transferred: {self.screenshot_count}")
        print(f"   Errors encountered: {self.error_count}")
    
    def _bridge_loop(self):
        """
        Main bridge loop that runs in a separate thread.
        
        Transfers data from trainer to web monitor at regular intervals.
        """
        print("🔄 Bridge loop started")
        consecutive_errors = 0
        max_consecutive_errors = 20
        
        while self.bridge_active:
            try:
                current_time = time.time()
                
                # Transfer screenshots
                if current_time - self.last_screenshot_update >= self.screenshot_update_interval:
                    if self._transfer_screenshot():
                        consecutive_errors = 0
                        self.last_screenshot_update = current_time
                
                # Transfer training statistics
                if current_time - self.last_stats_update >= self.stats_update_interval:
                    self._transfer_training_stats()
                    self.last_stats_update = current_time
                
                # Sleep based on bridge FPS
                time.sleep(1.0 / self.bridge_fps)
                
            except Exception as e:
                consecutive_errors += 1
                self.error_count += 1
                
                if consecutive_errors <= 3:
                    print(f"🌉 Bridge error #{consecutive_errors}: {e}")
                elif consecutive_errors == 4:
                    print(f"🌉 Bridge error #{consecutive_errors}: {e} (suppressing further error messages)")
                
                if consecutive_errors >= max_consecutive_errors:
                    print(f"❌ Too many consecutive bridge errors ({consecutive_errors}), stopping bridge")
                    self.bridge_active = False
                    break
                
                # Exponential backoff for errors
                error_sleep = min(2.0, 0.1 * (2 ** min(consecutive_errors, 10)))
                time.sleep(error_sleep)
        
        print("🔄 Bridge loop ended")
    
    def _transfer_screenshot(self) -> bool:
        """
        Transfer screenshot from trainer to web monitor.
        
        Returns:
            bool: True if screenshot was transferred successfully
        """
        try:
            # Check if trainer has a screenshot available
            if not hasattr(self.trainer, 'latest_screen') or not self.trainer.latest_screen:
                return False
            
            screen_data = self.trainer.latest_screen
            
            # Check if this is a new frame
            current_frame_id = screen_data.get('frame_id')
            if current_frame_id == self.last_frame_id:
                return False  # Same frame, skip
            
            self.last_frame_id = current_frame_id
            
            # Convert trainer's base64 image back to numpy array
            screenshot = self._convert_trainer_screenshot(screen_data)
            if screenshot is None:
                return False
            
            # Send to web monitor
            self.web_monitor.update_screenshot(screenshot)
            self.screenshot_count += 1
            
            if self.screenshot_count % 20 == 0:  # Log every 20 screenshots
                print(f"🖼️ Bridge transferred {self.screenshot_count} screenshots")
            
            return True
            
        except Exception as e:
            if self.error_count <= 5:  # Only log first few errors
                print(f"⚠️ Screenshot transfer error: {e}")
            return False
    
    def _convert_trainer_screenshot(self, screen_data: Dict[str, Any]) -> Optional[np.ndarray]:
        """
        Convert trainer's screenshot format to numpy array for web monitor.
        
        Args:
            screen_data: Dictionary containing image_b64 and metadata
            
        Returns:
            numpy array of the screenshot, or None if conversion failed
        """
        try:
            if 'image_b64' not in screen_data:
                return None
            
            # Decode base64 image
            img_data = base64.b64decode(screen_data['image_b64'])
            img = Image.open(io.BytesIO(img_data))
            
            # Convert to RGB numpy array
            screenshot = np.array(img)
            
            # Ensure RGB format (remove alpha channel if present)
            if len(screenshot.shape) == 3 and screenshot.shape[2] == 4:
                screenshot = screenshot[:, :, :3]
            
            return screenshot.astype(np.uint8)
            
        except Exception as e:
            if self.error_count <= 5:
                print(f"⚠️ Screenshot conversion error: {e}")
            return None
    
    def _transfer_training_stats(self):
        """Transfer training statistics from trainer to web monitor."""
        try:
            if not hasattr(self.trainer, 'stats'):
                return
            
            stats = self.trainer.stats
            
            # Create mock training session if needed
            if not hasattr(self.web_monitor, 'training_session') or not self.web_monitor.training_session:
                # Create a mock training session object
                trainer_ref = self.trainer  # Capture trainer reference for closure
                
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
                        self.env = MockEnv()  # Property, not method
                
                self.web_monitor.training_session = MockTrainingSession()
            
            # Update training stats
            training_stats = {
                'total_steps': stats.get('total_actions', 0),
                'episodes': stats.get('total_episodes', 0),
                'decisions_made': stats.get('llm_calls', 0),
                'visual_analyses': max(1, stats.get('total_actions', 0) // 5)  # Approximate OCR calls
            }
            
            if hasattr(self.web_monitor.training_session, 'training_stats'):
                self.web_monitor.training_session.training_stats.update(training_stats)
            
            # Transfer actions if available
            if hasattr(self.trainer, 'last_action') and hasattr(self.trainer, 'last_action_reasoning'):
                self.web_monitor.update_action(
                    str(self.trainer.last_action),
                    self.trainer.last_action_reasoning
                )
        
        except Exception as e:
            if self.error_count <= 5:
                print(f"⚠️ Stats transfer error: {e}")
    
    def get_bridge_stats(self) -> Dict[str, Any]:
        """
        Get bridge performance statistics.
        
        Returns:
            Dictionary containing bridge performance metrics
        """
        return {
            'is_active': self.bridge_active,
            'screenshots_transferred': self.screenshot_count,
            'total_errors': self.error_count,
            'last_screenshot_update': self.last_screenshot_update,
            'last_stats_update': self.last_stats_update,
            'success_rate': (self.screenshot_count / max(1, self.screenshot_count + self.error_count)) * 100,
            'bridge_fps': self.bridge_fps
        }


def create_integrated_monitoring_system(trainer, host='127.0.0.1', port=5001):
    """
    Factory function to create a fully integrated monitoring system.
    
    Args:
        trainer: UnifiedPokemonTrainer instance
        host: Host for web monitor
        port: Port for web monitor
        
    Returns:
        Tuple of (web_monitor, bridge, monitor_thread)
    """
    print("🏗️ Creating integrated monitoring system...")
    
    # Create web monitor
    web_monitor = PokemonRLWebMonitor()
    
    # Create bridge
    bridge = TrainerWebMonitorBridge(trainer, web_monitor)
    
    # Start web monitor in background thread
    def run_monitor():
        web_monitor.run(host=host, port=port, debug=False)
    
    monitor_thread = threading.Thread(target=run_monitor, daemon=True)
    monitor_thread.start()
    
    # Wait for server to start
    time.sleep(1)
    
    print(f"🚀 Integrated monitoring system created:")
    print(f"   🌐 Web monitor: http://{host}:{port}")
    print(f"   🌉 Bridge: Ready to connect trainer")
    
    return web_monitor, bridge, monitor_thread
