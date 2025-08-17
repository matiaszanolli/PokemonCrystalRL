#!/usr/bin/env python3
"""
Launch Pokemon Crystal RL training with socket streaming fix applied.
"""

import os
import sys
import time
import numpy as np
import threading
import signal

# Add paths for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from trainer.trainer import UnifiedPokemonTrainer
from trainer.config import TrainingConfig, TrainingMode, LLMBackend
from monitoring.trainer_monitor_bridge import TrainerWebMonitorBridge, create_integrated_monitoring_system


def fix_blank_screen_streaming(trainer):
    """
    Fix for blank screen streaming - advance PyBoy to show game content.
    """
    print("🎮 Applying PyBoy blank screen fix...")
    
    # Step 1: Advance PyBoy past initial blank frames
    print("⏳ Advancing PyBoy frames to show game content...")
    
    for i in range(100):
        trainer.pyboy.tick()
        if i % 20 == 0:
            screen = trainer.pyboy.screen.ndarray
            variance = np.var(screen.astype(np.float32))
            print(f"   Frame {i}: variance = {variance:.1f}")
            
            if variance > 10:  # Found content
                print(f"✅ Game content found at frame {i}")
                break
    
    # Step 2: Press START to begin game (optional but helpful)
    print("🕹️ Pressing START to begin game...")
    
    for i in range(10):  # Less aggressive than debug version
        trainer.strategy_manager.execute_action(7)  # START button
        screen = trainer.pyboy.screen.ndarray
        variance = np.var(screen.astype(np.float32))
        
        if i % 3 == 0:  # Log every few presses
            print(f"   START press {i+1}: variance = {variance:.1f}")
        
        if variance > 100:  # Game started
            print(f"✅ Game started successfully")
            break
        
        time.sleep(0.1)  # Brief pause between presses
    
    print("✅ PyBoy fix applied - emulator showing game content")


def launch_training():
    """Launch training with all fixes applied"""
    print("🚀 Launching Pokemon Crystal RL Training")
    print("=" * 60)
    print()
    
    # Create training configuration
    config = TrainingConfig(
        mode=TrainingMode.FAST_MONITORED,
        rom_path="../roms/pokemon_crystal.gbc",
        
        # Core training settings
        max_actions=2000,
        llm_backend=LLMBackend.NONE,  # Start with rule-based for testing
        llm_interval=20,
        
        # Screen capture and monitoring
        capture_screens=True,
        capture_fps=2,  # 2 FPS for stable streaming
        screen_resize=(160, 144),  # Game Boy resolution
        
        # Web monitoring
        enable_web=True,
        web_host="127.0.0.1", 
        web_port=5001,
        
        # Performance settings
        headless=True,
        debug_mode=True,
        frames_per_action=4,
        
        # Logging
        log_level="INFO"
    )
    
    print(f"📋 Training Configuration:")
    print(f"   Mode: {config.mode.value}")
    print(f"   Max actions: {config.max_actions}")
    print(f"   LLM backend: {config.llm_backend.value}")
    print(f"   Screen capture: {config.capture_screens} ({config.capture_fps} FPS)")
    print(f"   Web monitoring: {config.enable_web} (http://{config.web_host}:{config.web_port})")
    print()
    
    # Create trainer
    print("🏗️ Creating trainer...")
    trainer = UnifiedPokemonTrainer(config)
    
    # Apply PyBoy blank screen fix BEFORE starting monitoring
    fix_blank_screen_streaming(trainer)
    
    # Create integrated monitoring system
    print("\n🌐 Creating integrated web monitoring system...")
    web_monitor, bridge, monitor_thread = create_integrated_monitoring_system(
        trainer, 
        host=config.web_host, 
        port=config.web_port
    )
    
    print(f"✅ Web monitor available at: http://{config.web_host}:{config.web_port}")
    print()
    
    # Set up graceful shutdown
    def signal_handler(sig, frame):
        print("\n⏸️ Graceful shutdown requested...")
        
        if bridge:
            print("🔄 Stopping bridge...")
            bridge.stop_bridge()
        
        if web_monitor:
            print("🌐 Stopping web monitor...")  
            web_monitor.stop_monitoring()
        
        if trainer:
            print("🎮 Stopping trainer...")
            trainer._finalize_training()
        
        print("✅ Shutdown complete")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print("🎯 Starting training...")
    print(f"💡 Open http://{config.web_host}:{config.web_port} to monitor progress")
    print("🔄 Press Ctrl+C to stop training gracefully")
    print("=" * 60)
    print()
    
    try:
        # Start training (this will also start screen capture)
        trainer.start_training()
        
    except KeyboardInterrupt:
        signal_handler(signal.SIGINT, None)
    
    except Exception as e:
        print(f"❌ Training error: {e}")
        import traceback
        traceback.print_exc()
        
        # Cleanup
        if bridge:
            bridge.stop_bridge()
        if trainer:
            trainer._finalize_training()


def test_streaming_first():
    """Quick test of streaming before full training"""
    print("🧪 Quick Streaming Test")
    print("=" * 40)
    
    # Minimal config for testing
    config = TrainingConfig(
        mode=TrainingMode.FAST_MONITORED,
        rom_path="../roms/pokemon_crystal.gbc",
        capture_screens=True,
        capture_fps=2,
        enable_web=False,  # No web for quick test
        headless=True,
        debug_mode=True,
        max_actions=50  # Very short test
    )
    
    trainer = UnifiedPokemonTrainer(config)
    
    # Apply fix
    fix_blank_screen_streaming(trainer)
    
    # Start screen capture
    trainer._start_screen_capture()
    time.sleep(3)  # Let it capture a few frames
    
    # Check results
    if trainer.latest_screen:
        screen_data = trainer.latest_screen
        print(f"✅ Screenshot captured:")
        print(f"   Frame ID: {screen_data.get('frame_id')}")
        print(f"   Data length: {screen_data.get('data_length')} bytes")
        print(f"   Size: {screen_data.get('size')}")
        
        # Quick bridge test
        from monitoring.web_monitor import PokemonRLWebMonitor
        web_monitor = PokemonRLWebMonitor()
        bridge = TrainerWebMonitorBridge(trainer, web_monitor)
        bridge.start_bridge()
        
        time.sleep(5)  # Test for 5 seconds
        
        stats = bridge.get_bridge_stats()
        print(f"📊 Bridge test results:")
        print(f"   Screenshots transferred: {stats['screenshots_transferred']}")
        print(f"   Success rate: {stats['success_rate']:.1f}%")
        
        bridge.stop_bridge()
        trainer._finalize_training()
        
        if stats['screenshots_transferred'] > 0:
            print("✅ Streaming test PASSED - ready for full training")
            return True
        else:
            print("❌ Streaming test FAILED")
            return False
    else:
        print("❌ No screenshots captured")
        trainer._finalize_training()
        return False


if __name__ == "__main__":
    print("🎮 Pokemon Crystal RL Training Launcher")
    print("=" * 60)
    
    # Check if we should run a quick test first
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        success = test_streaming_first()
        if success:
            print("\n✅ Test passed! Run without --test to start full training")
        else:
            print("\n❌ Test failed! Check the configuration")
    else:
        # Run quick test first, then full training if it passes
        print("Running quick streaming test first...")
        if test_streaming_first():
            print("\n" + "="*60)
            time.sleep(2)
            launch_training()
        else:
            print("\n❌ Streaming test failed - not starting full training")
            print("💡 Try running with --test flag to diagnose issues")
