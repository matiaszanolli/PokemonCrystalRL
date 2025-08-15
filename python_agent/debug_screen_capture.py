#!/usr/bin/env python3
"""
debug_screen_capture.py - Debug screen capture issues

This script tests PyBoy screen capture functionality to diagnose 
why the web UI game screen is not working.
"""

import sys
import os
import time
import base64
import io
from PIL import Image
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_pyboy_screen_capture():
    """Test PyBoy screen capture functionality"""
    print("🔍 Testing PyBoy screen capture...")
    
    try:
        from pyboy import PyBoy
        from pyboy.utils import WindowEvent
    except ImportError:
        print("❌ PyBoy not available")
        return False
    
    # ROM path
    rom_path = "/mnt/data/src/pokemon_crystal_rl/roms/pokemon_crystal.gbc"
    
    if not os.path.exists(rom_path):
        print(f"❌ ROM not found at {rom_path}")
        return False
    
    print(f"✅ ROM found: {rom_path}")
    
    # Initialize PyBoy
    try:
        pyboy = PyBoy(
            rom_path,
            window="null",  # Headless
            debug=False
        )
        print("✅ PyBoy initialized successfully")
    except Exception as e:
        print(f"❌ PyBoy initialization failed: {e}")
        return False
    
    # Let the game run for a few ticks
    for _ in range(60):
        pyboy.tick()
    
    print("✅ Game running, testing screen capture methods...")
    
    # Test Method 1: screen.ndarray
    try:
        screen_array = pyboy.screen.ndarray
        if screen_array is not None:
            print(f"✅ Method 1 (screen.ndarray): Success - Shape: {screen_array.shape}, Type: {screen_array.dtype}")
            
            # Try to convert to image
            if len(screen_array.shape) == 3:
                # Convert to PIL Image and save for testing
                pil_image = Image.fromarray(screen_array.astype(np.uint8))
                test_path = "/mnt/data/src/pokemon_crystal_rl/python_agent/test_screen_method1.png"
                pil_image.save(test_path)
                print(f"✅ Saved test image to: {test_path}")
                
                # Test base64 encoding
                buffer = io.BytesIO()
                pil_image.save(buffer, format='PNG')
                b64_data = base64.b64encode(buffer.getvalue()).decode()
                print(f"✅ Base64 encoding successful - Length: {len(b64_data)} chars")
                
                # Cleanup
                pyboy.stop()
                return True
            else:
                print(f"⚠️ Unexpected array shape: {screen_array.shape}")
        else:
            print("❌ Method 1: screen.ndarray returned None")
    except Exception as e:
        print(f"❌ Method 1 failed: {e}")
    
    # Test Method 2: screen.image
    try:
        if hasattr(pyboy.screen, 'image'):
            screen_image = pyboy.screen.image
            if screen_image is not None:
                print(f"✅ Method 2 (screen.image): Success - Type: {type(screen_image)}")
                
                # Save for testing
                test_path = "/mnt/data/src/pokemon_crystal_rl/python_agent/test_screen_method2.png"
                screen_image.save(test_path)
                print(f"✅ Saved test image to: {test_path}")
                
                # Test base64 encoding
                buffer = io.BytesIO()
                screen_image.save(buffer, format='PNG')
                b64_data = base64.b64encode(buffer.getvalue()).decode()
                print(f"✅ Base64 encoding successful - Length: {len(b64_data)} chars")
                
                # Cleanup
                pyboy.stop()
                return True
            else:
                print("❌ Method 2: screen.image returned None")
        else:
            print("❌ Method 2: No screen.image attribute")
    except Exception as e:
        print(f"❌ Method 2 failed: {e}")
    
    # Test Method 3: screen_image() function (deprecated but might work)
    try:
        if hasattr(pyboy, 'screen_image'):
            screen_image = pyboy.screen_image()
            if screen_image is not None:
                print(f"✅ Method 3 (screen_image()): Success - Type: {type(screen_image)}")
                
                # Save for testing
                test_path = "/mnt/data/src/pokemon_crystal_rl/python_agent/test_screen_method3.png"
                screen_image.save(test_path)
                print(f"✅ Saved test image to: {test_path}")
                
                # Cleanup
                pyboy.stop()
                return True
            else:
                print("❌ Method 3: screen_image() returned None")
        else:
            print("❌ Method 3: No screen_image() method")
    except Exception as e:
        print(f"❌ Method 3 failed: {e}")
    
    # Cleanup
    pyboy.stop()
    print("❌ All screen capture methods failed")
    return False

def test_screen_data_format():
    """Test what the actual screen data looks like"""
    print("\n🔍 Testing screen data format...")
    
    try:
        from pyboy import PyBoy
    except ImportError:
        print("❌ PyBoy not available")
        return
    
    rom_path = "/mnt/data/src/pokemon_crystal_rl/roms/pokemon_crystal.gbc"
    
    try:
        pyboy = PyBoy(rom_path, window="null", debug=False)
        
        # Let game initialize
        for _ in range(60):
            pyboy.tick()
        
        # Examine screen object
        print(f"📊 PyBoy version info: {hasattr(pyboy, '__version__')}")
        print(f"📊 Screen object type: {type(pyboy.screen)}")
        print(f"📊 Screen attributes: {[attr for attr in dir(pyboy.screen) if not attr.startswith('_')]}")
        
        # Try to get raw screen data
        try:
            screen_data = pyboy.screen.ndarray
            print(f"📊 Screen ndarray shape: {screen_data.shape}")
            print(f"📊 Screen ndarray dtype: {screen_data.dtype}")
            print(f"📊 Screen ndarray min/max: {screen_data.min()}/{screen_data.max()}")
            print(f"📊 Screen ndarray unique values: {len(np.unique(screen_data))}")
            
            # Check if it's palette indices or RGB
            if screen_data.max() <= 255 and len(screen_data.shape) == 2:
                print("📊 Appears to be palette-indexed data")
            elif len(screen_data.shape) == 3:
                print(f"📊 Appears to be RGB data with {screen_data.shape[2]} channels")
            
        except Exception as e:
            print(f"❌ Could not analyze screen data: {e}")
        
        pyboy.stop()
        
    except Exception as e:
        print(f"❌ Failed to test screen data: {e}")

def main():
    """Main diagnostic function"""
    print("🔬 PyBoy Screen Capture Diagnostic Tool")
    print("=" * 50)
    
    # Test basic screen capture
    capture_success = test_pyboy_screen_capture()
    
    # Test screen data format
    test_screen_data_format()
    
    print("\n" + "=" * 50)
    if capture_success:
        print("✅ Screen capture is working - the issue is likely in the trainer web server")
    else:
        print("❌ Screen capture is broken - need to fix the capture method")
    
    print("\n💡 Next steps:")
    if capture_success:
        print("   - Check web server screen serving logic")
        print("   - Verify screen capture thread in trainer")
        print("   - Test web UI image loading")
    else:
        print("   - Update screen capture method in trainer")
        print("   - Handle new PyBoy API correctly")
        print("   - Add proper error handling")

if __name__ == "__main__":
    main()
