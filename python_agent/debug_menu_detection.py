#!/usr/bin/env python3
"""
Debug script to analyze menu detection logic
"""

import numpy as np
import sys
import os

# Add the parent directory to the path so we can import the trainer
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scripts.pokemon_trainer import UnifiedPokemonTrainer, TrainingConfig, TrainingMode

def analyze_menu_screen():
    """Analyze the test menu screen that's failing detection"""
    
    # Create the exact same screen as the test
    menu_screen = np.ones((144, 160, 3), dtype=np.uint8) * 120
    menu_screen[20:60, 20:140] = 200  # Menu box area
    
    print("🔍 Analyzing Menu Screen Detection")
    print("=" * 50)
    
    # Calculate key metrics
    height, width = menu_screen.shape[:2]
    mean_brightness = np.mean(menu_screen)
    overall_variance = np.var(menu_screen)
    
    print(f"Screen dimensions: {height} x {width}")
    print(f"Mean brightness: {mean_brightness:.2f}")
    print(f"Overall variance: {overall_variance:.2f}")
    
    # Test the individual detection components
    print("\n🧪 Detection Component Analysis:")
    
    # Menu detection components
    center_region = menu_screen[height//4:3*height//4, width//4:3*width//4]
    center_std = np.std(center_region)
    print(f"Center region std: {center_std:.2f}")
    
    primary_menu = 20 < center_std < 60
    secondary_menu = 60 < center_std < 100
    tertiary_menu = (120 < mean_brightness < 180 and 700 < overall_variance < 2500)
    quaternary_menu = (mean_brightness > 120 and center_std > 40 and overall_variance > 800)
    
    print(f"Primary menu (20 < {center_std:.2f} < 60): {primary_menu}")
    print(f"Secondary menu (60 < {center_std:.2f} < 100): {secondary_menu}")
    print(f"Tertiary menu (120 < {mean_brightness:.2f} < 180 AND 700 < {overall_variance:.2f} < 2500): {tertiary_menu}")
    print(f"Quaternary menu ({mean_brightness:.2f} > 120 AND {center_std:.2f} > 40 AND {overall_variance:.2f} > 800): {quaternary_menu}")
    
    menu_detected = primary_menu or secondary_menu or tertiary_menu or quaternary_menu
    print(f"Overall menu detection: {menu_detected}")
    
    # Overworld detection components
    print(f"\n🌍 Overworld Detection Analysis:")
    color_variance = np.var(menu_screen)
    
    primary_overworld = color_variance > 1500 and 50 < mean_brightness < 200
    secondary_overworld = color_variance > 800 and 80 < mean_brightness < 180
    tertiary_overworld = color_variance > 400 and 60 < mean_brightness < 200
    
    print(f"Primary overworld ({color_variance:.2f} > 1500 AND 50 < {mean_brightness:.2f} < 200): {primary_overworld}")
    print(f"Secondary overworld ({color_variance:.2f} > 800 AND 80 < {mean_brightness:.2f} < 180): {secondary_overworld}")
    print(f"Tertiary overworld ({color_variance:.2f} > 400 AND 60 < {mean_brightness:.2f} < 200): {tertiary_overworld}")
    
    overworld_detected = primary_overworld or secondary_overworld or tertiary_overworld
    print(f"Overall overworld detection: {overworld_detected}")
    
    # Test full detection logic
    print(f"\n🎯 Full Detection Logic:")
    
    # Create a dummy trainer to use the detection method
    config = TrainingConfig(rom_path="dummy.gbc", headless=True)
    
    class MockPyBoy:
        frame_count = 1000
    
    # Mock the trainer without initializing PyBoy
    trainer = object.__new__(UnifiedPokemonTrainer)
    trainer.pyboy = MockPyBoy()
    
    # Test the individual pattern methods directly
    menu_pattern = trainer._has_menu_pattern_fast(menu_screen)
    overworld_pattern = trainer._has_overworld_pattern_fast(menu_screen, mean_brightness)
    
    print(f"Menu pattern detected: {menu_pattern}")
    print(f"Overworld pattern detected: {overworld_pattern}")
    
    # Simulate the detection priority logic
    if menu_pattern:
        if overworld_pattern:
            # If both match, use brightness and variance to distinguish
            color_variance = np.var(menu_screen)
            if color_variance < 1200 and 140 < mean_brightness < 180:
                final_state = "menu"
            else:
                final_state = "overworld"
            print(f"Both patterns match. Variance: {color_variance:.2f}, Brightness: {mean_brightness:.2f}")
            print(f"Variance < 1200 AND 140 < brightness < 180: {color_variance < 1200 and 140 < mean_brightness < 180}")
        else:
            final_state = "menu"
    elif overworld_pattern:
        final_state = "overworld"
    else:
        final_state = "unknown"
    
    print(f"Final detected state: {final_state}")
    
    print(f"\n💡 Recommendations:")
    if not menu_pattern:
        print("- Menu pattern not detected. Consider adjusting thresholds.")
        if center_std <= 20:
            print(f"  - Center std ({center_std:.2f}) too low for primary detection")
        if center_std >= 100:
            print(f"  - Center std ({center_std:.2f}) too high for secondary detection")
        if not (700 < overall_variance < 2500):
            print(f"  - Overall variance ({overall_variance:.2f}) outside tertiary range")
        if not quaternary_menu:
            print(f"  - Quaternary conditions not met")

if __name__ == "__main__":
    analyze_menu_screen()
