#!/usr/bin/env python3
"""
Segfault debugging script to test trainer stability and identify crash sources.
"""

import sys
import time
import signal
import traceback
import gc
from contextlib import contextmanager

def setup_debug_handlers():
    """Setup signal handlers for debugging"""
    def segfault_handler(signum, frame):
        print("\n=== SEGMENTATION FAULT DETECTED ===")
        print(f"Signal: {signum}")
        print("Stack trace:")
        traceback.print_stack(frame)
        print("=== END SEGFAULT INFO ===")
        sys.exit(139)
    
    def abort_handler(signum, frame):
        print("\n=== ABORT SIGNAL DETECTED ===")
        print(f"Signal: {signum}")
        print("Stack trace:")
        traceback.print_stack(frame)
        print("=== END ABORT INFO ===")
        sys.exit(134)
    
    signal.signal(signal.SIGSEGV, segfault_handler)
    signal.signal(signal.SIGABRT, abort_handler)

@contextmanager
def memory_monitor(test_name):
    """Monitor memory usage during test"""
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    
    start_memory = process.memory_info().rss / 1024 / 1024  # MB
    print(f"🧪 Starting test: {test_name}")
    print(f"📊 Initial memory: {start_memory:.1f} MB")
    
    try:
        yield
    finally:
        # Force garbage collection
        gc.collect()
        end_memory = process.memory_info().rss / 1024 / 1024  # MB
        print(f"📊 Final memory: {end_memory:.1f} MB")
        print(f"📈 Memory change: {end_memory - start_memory:+.1f} MB")
        print(f"✅ Test completed: {test_name}")
        print("-" * 50)

def test_basic_imports():
    """Test 1: Basic imports"""
    with memory_monitor("Basic Imports"):
        import numpy as np
        from pyboy import PyBoy
        import threading
        import queue
        from PIL import Image
        print("✅ All imports successful")

def test_pyboy_creation():
    """Test 2: PyBoy creation and cleanup"""
    with memory_monitor("PyBoy Creation"):
        from pyboy import PyBoy
        
        pyboy = PyBoy('../test_gameboy.gbc', window='null', debug=False)
        print("✅ PyBoy created")
        
        # Test basic operations
        screen = pyboy.screen.ndarray
        print(f"✅ Screen access OK: {screen.shape}")
        
        # Test ticking
        for i in range(10):
            pyboy.tick()
        print("✅ Ticking test passed")
        
        # Cleanup
        pyboy.stop()
        pyboy = None
        print("✅ PyBoy cleaned up")

def test_threading_with_pyboy():
    """Test 3: Threading with PyBoy"""
    with memory_monitor("PyBoy Threading"):
        import threading
        import queue
        import numpy as np
        from pyboy import PyBoy
        
        pyboy = None
        capture_active = True
        screen_queue = queue.Queue(maxsize=5)
        errors = []
        
        def capture_thread():
            nonlocal pyboy, capture_active, errors
            try:
                while capture_active and pyboy:
                    try:
                        screen = pyboy.screen.ndarray
                        if screen is not None:
                            screen_copy = screen.copy()
                            if not screen_queue.full():
                                screen_queue.put(screen_copy)
                            del screen_copy
                    except Exception as e:
                        errors.append(f"Capture error: {e}")
                    time.sleep(0.02)
            except Exception as e:
                errors.append(f"Thread error: {e}")
        
        # Start PyBoy
        pyboy = PyBoy('../test_gameboy.gbc', window='null', debug=False)
        
        # Start capture thread
        thread = threading.Thread(target=capture_thread, daemon=True)
        thread.start()
        
        # Main loop
        for i in range(50):
            pyboy.tick()
            if i % 10 == 0:
                print(f"Main tick {i}")
            time.sleep(0.01)
        
        # Cleanup
        capture_active = False
        time.sleep(0.1)
        pyboy.stop()
        pyboy = None
        
        if errors:
            print(f"⚠️ Threading errors: {errors}")
        else:
            print("✅ Threading test passed")

def test_trainer_initialization():
    """Test 4: Trainer initialization"""
    with memory_monitor("Trainer Init"):
        from trainer.config import TrainingConfig, TrainingMode
        from trainer.trainer import UnifiedPokemonTrainer
        
        # Create minimal config
        config = TrainingConfig(
            rom_path='../test_gameboy.gbc',
            mode=TrainingMode.ULTRA_FAST,
            max_actions=10,
            max_episodes=1,
            headless=True,
            capture_screens=False,
            enable_web=False,
            debug_mode=False,
            llm_backend=None
        )
        
        # Create trainer
        trainer = UnifiedPokemonTrainer(config)
        print("✅ Trainer initialized")
        
        # Cleanup
        try:
            trainer._finalize_training()
        except:
            pass
        print("✅ Trainer cleaned up")

def test_screen_capture():
    """Test 5: Screen capture with PIL processing"""
    with memory_monitor("Screen Capture"):
        from pyboy import PyBoy
        from PIL import Image
        import numpy as np
        import io
        import base64
        
        pyboy = PyBoy('../test_gameboy.gbc', window='null', debug=False)
        
        # Capture and process 20 screens
        for i in range(20):
            try:
                # Get screen
                screen = pyboy.screen.ndarray
                if screen is not None:
                    # Process like the trainer does
                    screen_safe = np.ascontiguousarray(screen.astype(np.uint8))
                    screen_pil = Image.fromarray(screen_safe)
                    screen_resized = screen_pil.resize((160, 144), Image.NEAREST)
                    
                    # Convert to JPEG
                    buffer = io.BytesIO()
                    screen_resized.save(buffer, format='JPEG', quality=85)
                    
                    # Explicit cleanup
                    buffer.close()
                    screen_pil.close()
                    screen_resized.close()
                    del screen_safe, screen_pil, screen_resized, buffer
                
                pyboy.tick()
                
                if i % 5 == 0:
                    print(f"Processed screen {i}")
                    
            except Exception as e:
                print(f"❌ Screen processing error at {i}: {e}")
                break
        
        # Cleanup
        pyboy.stop()
        pyboy = None
        print("✅ Screen capture test passed")

def test_full_trainer_run():
    """Test 6: Full trainer run (short)"""
    with memory_monitor("Full Trainer Run"):
        from trainer.config import TrainingConfig, TrainingMode
        from trainer.trainer import UnifiedPokemonTrainer
        
        # Create config for short run
        config = TrainingConfig(
            rom_path='../test_gameboy.gbc',
            mode=TrainingMode.ULTRA_FAST,
            max_actions=20,
            max_episodes=1,
            headless=True,
            capture_screens=False,  # Disable capture to isolate issue
            enable_web=False,
            debug_mode=False,
            llm_backend=None
        )
        
        # Run trainer
        try:
            trainer = UnifiedPokemonTrainer(config)
            trainer.start_training()
            print("✅ Full trainer run successful")
        except Exception as e:
            print(f"❌ Trainer run failed: {e}")
            raise

def main():
    """Run all debug tests"""
    setup_debug_handlers()
    
    print("🔍 SEGFAULT DEBUGGING SUITE")
    print("=" * 50)
    
    tests = [
        test_basic_imports,
        test_pyboy_creation,
        test_threading_with_pyboy,
        test_trainer_initialization,
        test_screen_capture,
        test_full_trainer_run,
    ]
    
    for i, test in enumerate(tests, 1):
        try:
            print(f"\n🧪 Running Test {i}/{len(tests)}: {test.__name__}")
            test()
        except KeyboardInterrupt:
            print("\n⚠️ Tests interrupted by user")
            break
        except Exception as e:
            print(f"\n❌ Test {i} failed: {e}")
            traceback.print_exc()
            print(f"⚠️ Continuing with remaining tests...")
    
    print("\n🎯 Debug suite completed")
    print("If no segfault occurred, the fixes are working!")

if __name__ == "__main__":
    main()
