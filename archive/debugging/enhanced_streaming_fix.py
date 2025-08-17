#!/usr/bin/env python3
"""
Enhanced streaming fix with better error detection and game state advancement.
"""

import os
import sys
import time
import numpy as np
import threading
import signal
from PIL import Image

# Add paths for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from trainer.trainer import UnifiedPokemonTrainer
from trainer.config import TrainingConfig, TrainingMode, LLMBackend
from monitoring.trainer_monitor_bridge import TrainerWebMonitorBridge, create_integrated_monitoring_system


def analyze_screen_content(screen_array):
    """Analyze screen content to determine game state"""
    if screen_array is None or screen_array.size == 0:
        return "blank", 0.0
    
    # Calculate variance for content detection
    variance = np.var(screen_array.astype(np.float32))
    
    # Convert to RGB if RGBA
    if len(screen_array.shape) == 3 and screen_array.shape[2] == 4:
        rgb_screen = screen_array[:, :, :3]
    else:
        rgb_screen = screen_array
    
    # Check for specific patterns
    if variance < 1.0:
        return "blank", variance
    elif variance < 100:
        return "minimal_content", variance  # Might be loading or transition
    elif variance < 1000:
        return "menu_or_dialogue", variance  # Likely menu or text
    else:
        return "gameplay", variance  # Rich gameplay content
    
    return "unknown", variance


def advanced_game_start_sequence(trainer, max_attempts=200):
    """Advanced game starting sequence with better detection"""
    print("🎮 Starting advanced game initialization...")
    
    attempts = 0
    last_variance = 0
    stable_count = 0
    
    # Button sequence to get past various screens
    button_sequence = [
        (7, "START"),     # Start button
        (5, "A"),         # A button  
        (8, "B"),         # B button (skip/back)
        (1, "DOWN"),      # Navigate menus
        (0, "UP"),        # Navigate menus
        (2, "LEFT"),      # Navigate menus
        (3, "RIGHT")      # Navigate menus
    ]
    
    button_index = 0
    presses_per_button = 0
    
    print("🔄 Advancing through game states...")
    
    while attempts < max_attempts:
        # Get current screen
        screen = trainer.pyboy.screen.ndarray
        state_type, variance = analyze_screen_content(screen)
        
        if attempts % 20 == 0:  # Log every 20 attempts
            print(f"   Attempt {attempts}: {state_type} (variance: {variance:.1f})")
        
        # Check if we've reached gameplay
        if state_type == "gameplay" and variance > 2000:
            print(f"✅ Reached gameplay! Final variance: {variance:.1f}")
            return True
        
        # Check for stability (screen not changing much)
        if abs(variance - last_variance) < 10:
            stable_count += 1
        else:
            stable_count = 0
        
        # If screen is stable, try pressing buttons
        if stable_count >= 5 or attempts % 10 == 0:
            action, button_name = button_sequence[button_index]
            trainer.strategy_manager.execute_action(action)
            
            if attempts % 20 == 0:
                print(f"   Pressing {button_name} (action {action})")
            
            presses_per_button += 1
            
            # Cycle through buttons, spending more time on START and A
            if button_name in ["START", "A"] and presses_per_button >= 3:
                button_index = (button_index + 1) % len(button_sequence)
                presses_per_button = 0
            elif presses_per_button >= 1:
                button_index = (button_index + 1) % len(button_sequence)  
                presses_per_button = 0
        
        # Always advance at least one frame
        trainer.pyboy.tick()
        
        last_variance = variance
        attempts += 1
        
        # Brief pause for stability
        time.sleep(0.01)
    
    print(f"⚠️ Max attempts reached. Final state: {state_type} (variance: {variance:.1f})")
    return variance > 100  # Accept if we have some content


def enhanced_streaming_diagnostics(trainer, bridge):
    """Run enhanced diagnostics on streaming system"""
    print("\n🔍 Enhanced Streaming Diagnostics")
    print("-" * 50)
    
    # Test 1: Check trainer screenshot capture
    print("1️⃣ Testing trainer screenshot capture...")
    screenshot = trainer._simple_screenshot_capture()
    if screenshot is not None:
        state_type, variance = analyze_screen_content(screenshot)
        print(f"   ✅ Trainer capture: {state_type} (variance: {variance:.1f})")
        
        if variance < 10:
            print("   ⚠️ Screenshot appears blank or nearly blank")
            return False
    else:
        print("   ❌ Trainer capture failed")
        return False
    
    # Test 2: Check if trainer has latest_screen data
    print("2️⃣ Testing trainer latest_screen data...")
    if hasattr(trainer, 'latest_screen') and trainer.latest_screen:
        screen_data = trainer.latest_screen
        print(f"   ✅ Latest screen: {screen_data.get('data_length', 0)} bytes")
        print(f"   Frame ID: {screen_data.get('frame_id', 'N/A')}")
    else:
        print("   ❌ No latest_screen data available")
        return False
    
    # Test 3: Test bridge conversion
    print("3️⃣ Testing bridge conversion...")
    converted = bridge._convert_trainer_screenshot(screen_data)
    if converted is not None:
        bridge_state, bridge_variance = analyze_screen_content(converted)
        print(f"   ✅ Bridge conversion: {bridge_state} (variance: {bridge_variance:.1f})")
    else:
        print("   ❌ Bridge conversion failed")
        return False
    
    return True


def create_enhanced_launch_script():
    """Create enhanced launch script with better monitoring"""
    
    def launch_with_enhanced_monitoring():
        print("🚀 Enhanced Pokemon Crystal RL Training")
        print("=" * 60)
        
        # Configuration with aggressive screen capture
        config = TrainingConfig(
            mode=TrainingMode.FAST_MONITORED,
            rom_path="../roms/pokemon_crystal.gbc",
            
            # Enhanced capture settings
            capture_screens=True,
            capture_fps=3,  # Higher FPS for better streaming
            screen_resize=(160, 144),
            
            # Training settings
            max_actions=5000,  # More actions for longer session
            llm_backend=LLMBackend.NONE,
            frames_per_action=3,  # Faster action execution
            
            # Web monitoring
            enable_web=True,
            web_host="127.0.0.1",
            web_port=5001,
            
            # Debug settings
            headless=True,
            debug_mode=True,
            log_level="INFO"
        )
        
        print(f"📋 Enhanced Configuration:")
        print(f"   Capture FPS: {config.capture_fps}")
        print(f"   Frames per action: {config.frames_per_action}")
        print(f"   Max actions: {config.max_actions}")
        print()
        
        # Create trainer
        print("🏗️ Creating trainer...")
        trainer = UnifiedPokemonTrainer(config)
        
        # Enhanced game initialization
        if not advanced_game_start_sequence(trainer):
            print("❌ Failed to start game properly")
            return False
        
        # Start screen capture with validation
        print("\n📸 Starting enhanced screen capture...")
        trainer._start_screen_capture()
        time.sleep(2)  # Let it capture frames
        
        # Validate capture is working
        if not trainer.latest_screen:
            print("❌ Screen capture not working")
            trainer._finalize_training()
            return False
        
        # Create monitoring system
        print("🌐 Creating enhanced monitoring system...")
        web_monitor, bridge, monitor_thread = create_integrated_monitoring_system(
            trainer, host=config.web_host, port=config.web_port
        )
        
        # Enhanced bridge with diagnostics
        enhanced_bridge = EnhancedBridge(trainer, web_monitor)
        enhanced_bridge.start_bridge()
        
        print(f"✅ Enhanced system running at: http://{config.web_host}:{config.web_port}")
        print("📊 Monitor streaming status in web dashboard")
        print("🔄 Press Ctrl+C to stop")
        print()
        
        # Graceful shutdown
        def signal_handler(sig, frame):
            print("\n⏸️ Shutting down enhanced system...")
            enhanced_bridge.stop_bridge()
            web_monitor.stop_monitoring()
            trainer._finalize_training()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        
        try:
            trainer.start_training()
        except KeyboardInterrupt:
            signal_handler(signal.SIGINT, None)
        
        return True
    
    return launch_with_enhanced_monitoring


class EnhancedBridge(TrainerWebMonitorBridge):
    """Enhanced bridge with better error detection and recovery"""
    
    def __init__(self, trainer, web_monitor):
        super().__init__(trainer, web_monitor)
        self.consecutive_blank_frames = 0
        self.total_frames_processed = 0
        self.last_successful_transfer = time.time()
        
        # Enhanced settings
        self.screenshot_update_interval = 0.3  # More frequent updates
        self.max_consecutive_blanks = 10
        self.blank_frame_recovery_attempts = 0
        
        print("🌉 Enhanced bridge initialized with error recovery")
    
    def _transfer_screenshot(self) -> bool:
        """Enhanced screenshot transfer with error detection"""
        success = super()._transfer_screenshot()
        
        if success:
            # Check if the transferred screenshot is meaningful
            if hasattr(self.trainer, 'latest_screen') and self.trainer.latest_screen:
                screen_data = self.trainer.latest_screen
                
                # Try to decode and analyze the screenshot
                try:
                    converted = self._convert_trainer_screenshot(screen_data)
                    if converted is not None:
                        state_type, variance = analyze_screen_content(converted)
                        
                        if variance < 10:  # Blank frame
                            self.consecutive_blank_frames += 1
                            
                            if self.consecutive_blank_frames >= self.max_consecutive_blanks:
                                print(f"⚠️ Bridge: {self.consecutive_blank_frames} consecutive blank frames detected")
                                self._attempt_recovery()
                        else:
                            self.consecutive_blank_frames = 0
                            self.last_successful_transfer = time.time()
                            
                            if self.total_frames_processed % 50 == 0:  # Log periodically
                                print(f"📊 Bridge: {self.total_frames_processed} frames, state: {state_type}")
                        
                        self.total_frames_processed += 1
                
                except Exception as e:
                    print(f"⚠️ Bridge analysis error: {e}")
        
        return success
    
    def _attempt_recovery(self):
        """Attempt to recover from blank frames"""
        print("🩹 Attempting bridge recovery...")
        self.blank_frame_recovery_attempts += 1
        
        # Try to advance the game state
        if hasattr(self.trainer, 'strategy_manager'):
            for i in range(5):
                self.trainer.strategy_manager.execute_action(7)  # START
                time.sleep(0.1)
                self.trainer.strategy_manager.execute_action(5)  # A
                time.sleep(0.1)
        
        self.consecutive_blank_frames = 0
        print(f"🩹 Recovery attempt #{self.blank_frame_recovery_attempts} completed")
    
    def get_enhanced_stats(self):
        """Get enhanced bridge statistics"""
        base_stats = self.get_bridge_stats()
        
        # Add enhanced metrics
        base_stats.update({
            'consecutive_blank_frames': self.consecutive_blank_frames,
            'total_frames_processed': self.total_frames_processed,
            'recovery_attempts': self.blank_frame_recovery_attempts,
            'time_since_last_success': time.time() - self.last_successful_transfer
        })
        
        return base_stats


if __name__ == "__main__":
    print("🔧 Enhanced Streaming Fix")
    print("=" * 40)
    
    launch_func = create_enhanced_launch_script()
    success = launch_func()
    
    if success:
        print("✅ Enhanced launch completed successfully")
    else:
        print("❌ Enhanced launch failed")
