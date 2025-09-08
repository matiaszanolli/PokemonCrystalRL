"""
test_pyboy_screen.py - Test PyBoy screen capture methods

This script tests different methods to capture screen data from PyBoy
to determine the correct API for the current PyBoy version.
"""

import os
import sys
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from environments.pyboy_env import PyBoyPokemonCrystalEnv

def test_pyboy_screen_methods():
    """Test PyBoy screen capture methods"""
    print("🧪 Testing PyBoy screen capture methods...")
    
    try:
        # Create environment with debug mode enabled
        env = PyBoyPokemonCrystalEnv(
            rom_path="/mnt/data/src/pokemon_crystal_rl/roms/pokemon_crystal.gbc",
            save_state_path=None,
            debug_mode=True,
            headless=True
        )
        
        print("✅ Environment created successfully")
        
        # Try to reset the environment to initialize PyBoy
        try:
            obs, info = env.reset()
            print("✅ Environment reset successfully")
            
            # Check if PyBoy instance exists
            if env.pyboy:
                print("✅ PyBoy instance exists")
                
                # Check screen attribute
                if hasattr(env.pyboy, 'screen'):
                    screen = env.pyboy.screen
                    print(f"✅ Screen attribute found: {type(screen)}")
                    print(f"Screen methods: {[m for m in dir(screen) if not m.startswith('_')]}")
                    
                    # Test screen array access
                    if hasattr(screen, 'ndarray'):
                        try:
                            screen_data = screen.ndarray
                            print(f"✅ Screen ndarray shape: {screen_data.shape}")
                            print(f"✅ Screen ndarray dtype: {screen_data.dtype}")
                        except Exception as e:
                            print(f"❌ Screen ndarray error: {e}")
                    
                    # Test PIL image access
                    if hasattr(screen, 'pil_image'):
                        try:
                            pil_img = screen.pil_image()
                            print(f"✅ PIL image size: {pil_img.size}")
                            print(f"✅ PIL image mode: {pil_img.mode}")
                        except Exception as e:
                            print(f"❌ PIL image error: {e}")
                
                else:
                    print("❌ No screen attribute found")
                    print(f"Available PyBoy attributes: {[attr for attr in dir(env.pyboy) if not attr.startswith('_')]}")
                
                # Test screenshot capture
                print("\n🧪 Testing screenshot capture...")
                try:
                    screenshot = env.get_screenshot()
                    print(f"✅ Screenshot captured: {screenshot.shape}, dtype: {screenshot.dtype}")
                    print(f"Screenshot stats: min={screenshot.min()}, max={screenshot.max()}")
                except Exception as e:
                    print(f"❌ Screenshot capture failed: {e}")
                
            else:
                print("❌ PyBoy instance not created")
                
        except Exception as e:
            print(f"❌ Environment reset failed: {e}")
            print("This is expected without a valid ROM file")
            
            # Still try to check PyBoy attributes if possible
            if hasattr(env, 'pyboy') and env.pyboy:
                print("Checking PyBoy attributes anyway...")
                print(f"PyBoy attributes: {[attr for attr in dir(env.pyboy) if not attr.startswith('_')]}")
        
        finally:
            try:
                env.close()
            except:
                pass
                
    except Exception as e:
        print(f"❌ Environment creation failed: {e}")
        import traceback
        traceback.print_exc()

def test_minimal_screen_capture():
    """Test minimal screen capture approach"""
    print("\n🧪 Testing minimal screen capture approach...")
    
    # Test the environment's screen capture without ROM loading
    env = PyBoyPokemonCrystalEnv(
        rom_path="/mnt/data/src/pokemon_crystal_rl/roms/pokemon_crystal.gbc",
        save_state_path=None,
        debug_mode=True
    )
    
    # Test screenshot method without initialization
    screenshot = env.get_screenshot()
    print(f"Screenshot without PyBoy init: {screenshot.shape}, dtype: {screenshot.dtype}")
    print(f"Screenshot is empty: {np.all(screenshot == 0)}")

def main():
    """Run all screen capture tests"""
    print("🔍 PyBoy Screen Capture Method Testing")
    print("=" * 50)
    
    test_pyboy_screen_methods()
    test_minimal_screen_capture()
    
    print("\n📋 RECOMMENDATIONS")
    print("-" * 30)
    print("Based on the test results above:")
    print("1. If 'Screen ndarray' works - use screen.ndarray")
    print("2. If 'PIL image' works - use screen.pil_image()")
    print("3. If neither works - implement fallback with empty array")

if __name__ == "__main__":
    main()
