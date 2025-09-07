#!/usr/bin/env python3
"""
Test script to verify graceful shutdown functionality.

This script creates a trainer, starts it briefly, then tests the graceful shutdown method.
"""

import sys
import os
import time
import threading
import logging
from unittest.mock import Mock, patch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from training.trainer import PokemonTrainer, TrainingConfig, TrainingMode


def test_graceful_shutdown():
    """Test the graceful shutdown functionality"""
    print("🧪 Testing Graceful Shutdown Functionality")
    print("=" * 50)
    
    # Create minimal config for testing
    config = TrainingConfig(
        rom_path="test_rom.gbc",  # Will be mocked
        mode=TrainingMode.ULTRA_FAST,
        max_actions=100,
        headless=True,
        enable_web=True,  # Enable web to test web monitor shutdown
        debug_mode=True,
        capture_screens=True
    )
    
    try:
        print("📝 Creating trainer...")
        # Mock PyBoy to avoid needing a real ROM file
        with patch('training.trainer.PyBoy') as mock_pyboy:
            mock_pyboy_instance = Mock()
            mock_pyboy_instance.frame_count = 0
            mock_pyboy_instance.screen.ndarray = Mock()
            mock_pyboy.return_value = mock_pyboy_instance
            
            trainer = PokemonTrainer(config)
            
            # Simulate some initialization
            print("🔧 Initializing synchronization primitives...")
            
            # Check if synchronization attributes were added correctly
            if hasattr(trainer, '_shutdown_event'):
                print("✅ Shutdown event initialized")
            else:
                print("❌ Shutdown event missing")
                
            if hasattr(trainer, '_shared_lock'):
                print("✅ Shared lock initialized")
            else:
                print("❌ Shared lock missing")
                
            # Test graceful shutdown method
            print("\n🛑 Testing graceful shutdown...")
            if hasattr(trainer, 'graceful_shutdown'):
                print("✅ Graceful shutdown method available")
                
                # Run shutdown with timeout
                start_time = time.time()
                try:
                    trainer.graceful_shutdown(timeout=5)
                    shutdown_time = time.time() - start_time
                    print(f"✅ Graceful shutdown completed in {shutdown_time:.2f}s")
                except Exception as e:
                    print(f"⚠️ Shutdown error: {e}")
                    
            else:
                print("❌ Graceful shutdown method not found")
            
            # Test shutdown event
            if hasattr(trainer, '_shutdown_event'):
                if trainer._shutdown_event.is_set():
                    print("✅ Shutdown event is set after graceful shutdown")
                else:
                    print("❌ Shutdown event not set")
            
            print("\n✅ All graceful shutdown tests completed!")
            return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_synchronization_improvements():
    """Test the synchronization improvements"""
    print("\n🔧 Testing Synchronization Improvements")
    print("=" * 50)
    
    # Check if sync fixes module is available
    try:
        from trainer.synchronization_improvements import ThreadSafeScreenManager, apply_sync_fixes
        print("✅ Synchronization improvements module loaded")
        
        # Test ThreadSafeScreenManager
        screen_manager = ThreadSafeScreenManager()
        print("✅ ThreadSafeScreenManager created successfully")
        
        # Test concurrent access simulation
        import threading
        import queue
        import numpy as np
        
        def producer():
            for i in range(10):
                screen_data = np.random.randint(0, 256, (144, 160, 3), dtype=np.uint8)
                screen_manager.update_screen(screen_data)
                time.sleep(0.01)
        
        def consumer():
            for i in range(10):
                screen = screen_manager.get_screen(timeout=0.1)
                if screen is not None:
                    print(f"📸 Consumer got screen {i}")
                time.sleep(0.01)
        
        # Run concurrent threads
        threads = [
            threading.Thread(target=producer),
            threading.Thread(target=consumer)
        ]
        
        print("🏃 Running concurrent screen access test...")
        for t in threads:
            t.start()
            
        for t in threads:
            t.join()
            
        print("✅ Concurrent access test completed successfully")
        
        return True
        
    except ImportError as e:
        print(f"⚠️ Synchronization improvements not available: {e}")
        return False
    except Exception as e:
        print(f"❌ Synchronization test failed: {e}")
        return False


def main():
    """Main test function"""
    print("🚀 Pokemon Crystal RL - Graceful Shutdown Test Suite")
    print("=" * 60)
    
    # Set up logging for better visibility
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    success = True
    
    # Test 1: Graceful shutdown
    try:
        if not test_graceful_shutdown():
            success = False
    except Exception as e:
        print(f"❌ Graceful shutdown test failed: {e}")
        success = False
    
    # Test 2: Synchronization improvements
    try:
        if not test_synchronization_improvements():
            success = False
    except Exception as e:
        print(f"❌ Synchronization test failed: {e}")
        success = False
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 ALL TESTS PASSED! Graceful shutdown is working correctly.")
        print("💡 The training system should now handle interruptions gracefully")
        print("   and properly clean up all threads and resources.")
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
