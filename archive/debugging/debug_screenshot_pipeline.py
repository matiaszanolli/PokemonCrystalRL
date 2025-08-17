#!/usr/bin/env python3
"""
Deep dive diagnosis of the screenshot processing pipeline.
"""

import os
import sys
import time
import base64
import io
import numpy as np
from PIL import Image

# Add the parent directory to the Python path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from trainer.trainer import UnifiedPokemonTrainer
from trainer.config import TrainingConfig, TrainingMode


def analyze_screenshot_pipeline():
    """Analyze each step of the screenshot pipeline"""
    print("🔍 Deep Diagnosis of Screenshot Processing Pipeline")
    print("=" * 70)
    
    # Create trainer
    config = TrainingConfig(
        mode=TrainingMode.FAST_MONITORED,
        rom_path="../roms/pokemon_crystal.gbc",
        capture_screens=True,
        headless=True,
        debug_mode=True
    )
    trainer = UnifiedPokemonTrainer(config)
    
    print("\n1️⃣ TESTING PYBOY DIRECT ACCESS")
    print("-" * 50)
    
    # Test PyBoy direct access
    try:
        pyboy_screen = trainer.pyboy.screen.ndarray
        print(f"✅ PyBoy screen shape: {pyboy_screen.shape}")
        print(f"✅ PyBoy screen dtype: {pyboy_screen.dtype}")
        print(f"✅ PyBoy screen range: {pyboy_screen.min()} - {pyboy_screen.max()}")
        
        # Calculate variance
        variance = np.var(pyboy_screen.astype(np.float32))
        print(f"✅ PyBoy variance: {variance:.3f}")
        
        # Check if it's blank
        if variance < 1.0:
            print("⚠️ PyBoy screen appears blank")
        else:
            print("✅ PyBoy screen has content")
            
        # Save PyBoy screenshot for inspection
        if len(pyboy_screen.shape) == 3 and pyboy_screen.shape[2] == 4:
            # RGBA - convert to RGB
            rgb_screen = pyboy_screen[:, :, :3]
        else:
            rgb_screen = pyboy_screen
            
        pyboy_img = Image.fromarray(rgb_screen.astype(np.uint8))
        pyboy_img.save("debug_pyboy_direct.png")
        print("💾 Saved PyBoy direct screenshot: debug_pyboy_direct.png")
        
    except Exception as e:
        print(f"❌ PyBoy direct access failed: {e}")
        return
    
    print("\n2️⃣ TESTING TRAINER SIMPLE CAPTURE")
    print("-" * 50)
    
    # Test trainer's simple screenshot capture
    try:
        trainer_screen = trainer._simple_screenshot_capture()
        if trainer_screen is not None:
            print(f"✅ Trainer screen shape: {trainer_screen.shape}")
            print(f"✅ Trainer screen dtype: {trainer_screen.dtype}")
            print(f"✅ Trainer screen range: {trainer_screen.min()} - {trainer_screen.max()}")
            
            # Calculate variance
            trainer_variance = np.var(trainer_screen.astype(np.float32))
            print(f"✅ Trainer variance: {trainer_variance:.3f}")
            
            if trainer_variance < 1.0:
                print("⚠️ Trainer screen appears blank")
            else:
                print("✅ Trainer screen has content")
                
            # Save trainer screenshot
            trainer_img = Image.fromarray(trainer_screen.astype(np.uint8))
            trainer_img.save("debug_trainer_simple.png")
            print("💾 Saved trainer simple screenshot: debug_trainer_simple.png")
            
        else:
            print("❌ Trainer simple capture returned None")
            return
            
    except Exception as e:
        print(f"❌ Trainer simple capture failed: {e}")
        return
    
    print("\n3️⃣ TESTING TRAINER PROCESSING PIPELINE")
    print("-" * 50)
    
    # Start screen capture and wait for processing
    trainer._start_screen_capture()
    time.sleep(2)  # Wait for capture to start
    
    if trainer.latest_screen:
        screen_data = trainer.latest_screen
        print(f"✅ Processed screen data keys: {list(screen_data.keys())}")
        print(f"✅ Image data length: {screen_data.get('data_length', 0)} bytes")
        print(f"✅ Target size: {screen_data.get('size', 'N/A')}")
        
        # Decode the base64 image
        try:
            img_b64 = screen_data['image_b64']
            img_data = base64.b64decode(img_b64)
            
            # Load with PIL
            processed_img = Image.open(io.BytesIO(img_data))
            print(f"✅ Processed image size: {processed_img.size}")
            print(f"✅ Processed image mode: {processed_img.mode}")
            
            # Convert to numpy and check variance
            processed_array = np.array(processed_img)
            processed_variance = np.var(processed_array.astype(np.float32))
            print(f"✅ Processed variance: {processed_variance:.3f}")
            
            if processed_variance < 1.0:
                print("⚠️ Processed screen appears blank")
            else:
                print("✅ Processed screen has content")
            
            # Save processed image
            processed_img.save("debug_trainer_processed.jpg")
            print("💾 Saved trainer processed screenshot: debug_trainer_processed.jpg")
            
            # Check if it's all the same color
            unique_colors = len(np.unique(processed_array.reshape(-1, processed_array.shape[-1]), axis=0))
            print(f"✅ Unique colors in processed image: {unique_colors}")
            
            if unique_colors == 1:
                print("⚠️ Image has only one color (completely uniform)")
                dominant_color = processed_array[0, 0]
                print(f"   Dominant color: {dominant_color}")
            
        except Exception as e:
            print(f"❌ Failed to decode processed image: {e}")
            return
    else:
        print("❌ No processed screen available")
        return
    
    print("\n4️⃣ TESTING BRIDGE CONVERSION")
    print("-" * 50)
    
    # Test how the bridge would convert the screen data
    try:
        # Simulate bridge conversion
        img_b64 = screen_data['image_b64']
        img_data = base64.b64decode(img_b64)
        bridge_img = Image.open(io.BytesIO(img_data))
        
        # Convert to RGB if needed
        if bridge_img.mode != 'RGB':
            bridge_img = bridge_img.convert('RGB')
        
        # Convert to numpy
        bridge_array = np.array(bridge_img, dtype=np.uint8)
        print(f"✅ Bridge array shape: {bridge_array.shape}")
        print(f"✅ Bridge array dtype: {bridge_array.dtype}")
        print(f"✅ Bridge array range: {bridge_array.min()} - {bridge_array.max()}")
        
        # Calculate variance like the bridge does
        bridge_variance = np.var(bridge_array.astype(np.float32))
        print(f"✅ Bridge variance calculation: {bridge_variance:.3f}")
        
        # Test the bridge's validation logic
        if bridge_variance < 1.0:
            print("❌ Bridge would reject this as blank")
        else:
            print("✅ Bridge would accept this screenshot")
            
        # Check for uniform color
        if np.all(bridge_array == bridge_array[0, 0]):
            print("❌ Bridge would detect uniform color")
        else:
            print("✅ Bridge would not detect uniform color")
            
    except Exception as e:
        print(f"❌ Bridge conversion test failed: {e}")
    
    print("\n5️⃣ COMPARISON AND ANALYSIS")  
    print("-" * 50)
    
    # Compare the images at each step
    try:
        # Load all saved images
        pyboy_img = Image.open("debug_pyboy_direct.png")
        trainer_img = Image.open("debug_trainer_simple.png") 
        processed_img = Image.open("debug_trainer_processed.jpg")
        
        pyboy_array = np.array(pyboy_img)
        trainer_array = np.array(trainer_img)
        processed_array = np.array(processed_img)
        
        print(f"PyBoy -> Trainer conversion:")
        print(f"  Shape: {pyboy_array.shape} -> {trainer_array.shape}")
        print(f"  Variance: {np.var(pyboy_array.astype(np.float32)):.3f} -> {np.var(trainer_array.astype(np.float32)):.3f}")
        
        print(f"Trainer -> Processed conversion:")
        print(f"  Shape: {trainer_array.shape} -> {processed_array.shape}")
        print(f"  Variance: {np.var(trainer_array.astype(np.float32)):.3f} -> {np.var(processed_array.astype(np.float32)):.3f}")
        
        # Check if images are identical at key points
        if np.array_equal(pyboy_array[:, :, :3], trainer_array):
            print("✅ PyBoy and Trainer arrays are identical")
        else:
            print("⚠️ PyBoy and Trainer arrays differ")
            
    except Exception as e:
        print(f"⚠️ Could not compare images: {e}")
    
    # Cleanup
    trainer._finalize_training()
    
    print("\n🎯 DIAGNOSIS COMPLETE")
    print("=" * 70)
    print("Check the saved debug images:")
    print("  - debug_pyboy_direct.png")
    print("  - debug_trainer_simple.png") 
    print("  - debug_trainer_processed.jpg")


if __name__ == "__main__":
    analyze_screenshot_pipeline()
