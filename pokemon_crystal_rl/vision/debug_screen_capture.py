#!/usr/bin/env python3
"""
debug_screen_capture.py - Debug script to test PyBoy screen capture methods
"""

import numpy as np
import time
import traceback
from PIL import Image
import base64
import io

try:
    from pyboy import PyBoy
    PYBOY_AVAILABLE = True
except ImportError:
    PYBOY_AVAILABLE = False
    print("❌ PyBoy not available")

def _test_pyboy_screen_methods(rom_path):
    """Test different PyBoy screen capture methods"""
    if not PYBOY_AVAILABLE:
        print("❌ PyBoy not available")
        return
    
    print("🎮 Initializing PyBoy...")
    pyboy = PyBoy(rom_path, window="null", debug=False)
    
    # Let game boot for a few frames
    for _ in range(10):
        pyboy.tick()
    
    print("🔍 Testing screen capture methods...")
    
    # Method 1: screen.ndarray
    print("\\n1. Testing pyboy.screen.ndarray:")
    try:
        if hasattr(pyboy, 'screen') and hasattr(pyboy.screen, 'ndarray'):
            screen_data = pyboy.screen.ndarray
            print(f"   ✅ Success: shape={screen_data.shape}, dtype={screen_data.dtype}")
            print(f"   📊 Min: {screen_data.min()}, Max: {screen_data.max()}")
            print(f"   📊 Sample values: {screen_data.flatten()[:10]}")
            
            # Test if we can convert to PIL and base64
            try:
                screen_pil = Image.fromarray(screen_data)
                print(f"   🖼️ PIL conversion: {screen_pil.size}, mode={screen_pil.mode}")
                
                # Test resizing
                resized = screen_pil.resize((240, 216), Image.NEAREST)
                print(f"   📏 Resized: {resized.size}")
                
                # Test base64 encoding
                buffer = io.BytesIO()
                resized.save(buffer, format='PNG')
                b64_data = base64.b64encode(buffer.getvalue()).decode()
                print(f"   🔐 Base64 length: {len(b64_data)} chars")
                
            except Exception as e:
                print(f"   ❌ PIL/Base64 error: {e}")
                traceback.print_exc()
        else:
            print("   ❌ Method not available")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        traceback.print_exc()
    
    # Method 2: screen.image
    print("\\n2. Testing pyboy.screen.image:")
    try:
        if hasattr(pyboy, 'screen') and hasattr(pyboy.screen, 'image'):
            screen_image = pyboy.screen.image()
            screen_array = np.array(screen_image)
            print(f"   ✅ Success: shape={screen_array.shape}, dtype={screen_array.dtype}")
            print(f"   🖼️ PIL image: {screen_image.size}, mode={screen_image.mode}")
        else:
            print("   ❌ Method not available")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        traceback.print_exc()
    
    # Method 3: Check available screen methods
    print("\\n3. Available screen methods:")
    try:
        if hasattr(pyboy, 'screen'):
            methods = [attr for attr in dir(pyboy.screen) if not attr.startswith('_')]
            print(f"   📋 Methods: {methods}")
        else:
            print("   ❌ No screen attribute")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Method 4: Test continuous capture
    print("\\n4. Testing continuous capture (10 frames):")
    successful_captures = 0
    failed_captures = 0
    
    for i in range(10):
        try:
            pyboy.tick()  # Advance game
            
            # Try to capture screen
            if hasattr(pyboy.screen, 'ndarray'):
                screen_data = pyboy.screen.ndarray.copy()
                screen_pil = Image.fromarray(screen_data)
                resized = screen_pil.resize((240, 216), Image.NEAREST)
                
                buffer = io.BytesIO()
                resized.save(buffer, format='PNG', optimize=True)
                b64_data = base64.b64encode(buffer.getvalue()).decode()
                
                if len(b64_data) > 0:
                    successful_captures += 1
                    if i % 3 == 0:  # Print every 3rd frame
                        print(f"   ✅ Frame {i}: OK (b64: {len(b64_data)} chars)")
                else:
                    failed_captures += 1
                    print(f"   ❌ Frame {i}: Empty b64")
            else:
                failed_captures += 1
                print(f"   ❌ Frame {i}: No ndarray method")
                
        except Exception as e:
            failed_captures += 1
            print(f"   ❌ Frame {i}: {e}")
        
        time.sleep(0.1)
    
    print(f"\\n📊 Capture Results: {successful_captures} successful, {failed_captures} failed")
    
    pyboy.stop()
    print("🛑 PyBoy stopped")

def main():
    """Main function"""
    rom_path = "/mnt/data/src/pokemon_crystal_rl/roms/pokemon_crystal.gbc"
    
    print("🔬 PyBoy Screen Capture Debug Tool")
    print("=" * 50)
    
    try:
        _test_pyboy_screen_methods(rom_path)
    except Exception as e:
        print(f"❌ Test failed: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
