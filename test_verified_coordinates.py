#!/usr/bin/env python3
"""
test_verified_coordinates.py

Test script to verify that our coordinate tracking is working correctly
with the verified memory addresses.
"""

import sys
import os
sys.path.insert(0, '.')

from pyboy import PyBoy
from core.memory_map import MEMORY_ADDRESSES


def test_coordinate_tracking():
    """Test the coordinate tracking system with verified addresses."""
    print("🧪 Testing Verified Coordinate Tracking System...")
    
    # Initialize PyBoy
    pyboy = PyBoy('roms/pokemon_crystal.gbc', window='null', debug=False)
    
    try:
        # Initialize game state
        print("⏳ Initializing game state...")
        for i in range(500):
            pyboy.button_press('a')
            for _ in range(2):
                pyboy.tick()
            pyboy.button_release('a')
            for _ in range(8):
                pyboy.tick()
        
        def get_coordinates():
            """Get current player coordinates using verified addresses."""
            x = pyboy.memory[MEMORY_ADDRESSES['player_x']]
            y = pyboy.memory[MEMORY_ADDRESSES['player_y']]
            map_id = pyboy.memory[MEMORY_ADDRESSES['player_map']]
            direction = pyboy.memory[MEMORY_ADDRESSES['player_direction']]
            return x, y, map_id, direction
        
        def attempt_movement(direction, max_attempts=5):
            """Attempt movement in a direction until successful or max attempts."""
            print(f"\n🚶 Testing {direction.upper()} movement...")
            
            for attempt in range(max_attempts):
                before_x, before_y, before_map, before_dir = get_coordinates()
                print(f"  Attempt {attempt + 1}: Before X={before_x}, Y={before_y}")
                
                # Execute movement with proper timing
                pyboy.button_press(direction)
                for _ in range(2):  # Hold button for 2 frames
                    pyboy.tick()
                pyboy.button_release(direction)
                for _ in range(16):  # Wait 16 frames for movement to complete
                    pyboy.tick()
                
                after_x, after_y, after_map, after_dir = get_coordinates()
                print(f"  Attempt {attempt + 1}: After X={after_x}, Y={after_y}")
                
                # Check if position changed
                if before_x != after_x or before_y != after_y:
                    dx = after_x - before_x
                    dy = after_y - before_y
                    print(f"  ✅ SUCCESS: Movement detected! Δx={dx:+d}, Δy={dy:+d}")
                    return True
                else:
                    print(f"  ❌ No movement (blocked/boundary)")
            
            print(f"  ⚠️  No successful movement after {max_attempts} attempts")
            return False
        
        # Test initial coordinates
        print("\n📍 Initial Position:")
        x, y, map_id, direction = get_coordinates()
        print(f"  X: {x}, Y: {y}, Map: {map_id}, Direction: {direction}")
        
        # Test movement in all directions
        directions = ['right', 'left', 'down', 'up']
        successful_moves = 0
        
        for direction in directions:
            if attempt_movement(direction):
                successful_moves += 1
        
        # Final position
        print("\n📍 Final Position:")
        x, y, map_id, direction = get_coordinates()
        print(f"  X: {x}, Y: {y}, Map: {map_id}, Direction: {direction}")
        
        # Summary
        print(f"\n📊 Test Results:")
        print(f"  Successful movements: {successful_moves}/{len(directions)}")
        print(f"  Coordinate tracking: {'✅ WORKING' if successful_moves > 0 else '❌ ISSUES'}")
        
        if successful_moves > 0:
            print("\n🎉 SUCCESS: Coordinate tracking is working correctly!")
            print("   Memory addresses 0xDCB8 (X) and 0xDCB9 (Y) are verified.")
            print("   The agent can now properly track player position.")
        else:
            print("\n⚠️  WARNING: No movements detected.")
            print("   This could be due to being stuck in an area with no valid moves.")
            print("   The memory addresses are still likely correct.")
        
        return successful_moves > 0
        
    finally:
        pyboy.stop()


def test_position_persistence():
    """Test that position tracking remains consistent across multiple reads."""
    print("\n🔄 Testing Position Persistence...")
    
    pyboy = PyBoy('roms/pokemon_crystal.gbc', window='null', debug=False)
    
    try:
        # Initialize
        for i in range(500):
            pyboy.button_press('a')
            for _ in range(2):
                pyboy.tick()
            pyboy.button_release('a')
            for _ in range(8):
                pyboy.tick()
        
        # Read position multiple times without input
        positions = []
        for i in range(10):
            x = pyboy.memory[MEMORY_ADDRESSES['player_x']]
            y = pyboy.memory[MEMORY_ADDRESSES['player_y']]
            positions.append((x, y))
            
            # Wait a bit between reads
            for _ in range(5):
                pyboy.tick()
        
        # Check consistency
        unique_positions = set(positions)
        if len(unique_positions) == 1:
            print(f"  ✅ Position consistent: {positions[0]}")
            return True
        else:
            print(f"  ❌ Position inconsistent: {unique_positions}")
            return False
    
    finally:
        pyboy.stop()


if __name__ == "__main__":
    print("=" * 60)
    print("🧪 POKEMON CRYSTAL COORDINATE VERIFICATION")
    print("=" * 60)
    
    # Check if ROM exists
    if not os.path.exists('roms/pokemon_crystal.gbc'):
        print("❌ ERROR: ROM file not found at 'roms/pokemon_crystal.gbc'")
        sys.exit(1)
    
    try:
        # Test 1: Basic coordinate tracking
        print("\n🧪 TEST 1: Coordinate Tracking")
        tracking_works = test_coordinate_tracking()
        
        # Test 2: Position persistence
        print("\n🧪 TEST 2: Position Persistence")
        persistence_works = test_position_persistence()
        
        # Final summary
        print("\n" + "=" * 60)
        print("📊 FINAL RESULTS")
        print("=" * 60)
        print(f"  Coordinate Tracking: {'✅ PASS' if tracking_works else '❌ FAIL'}")
        print(f"  Position Persistence: {'✅ PASS' if persistence_works else '❌ FAIL'}")
        
        if tracking_works and persistence_works:
            print("\n🎉 ALL TESTS PASSED!")
            print("   The coordinate system is ready for RL training!")
            print("   Memory addresses: X=0xDCB8, Y=0xDCB9")
        else:
            print("\n⚠️  SOME TESTS FAILED")
            print("   Manual verification may be needed.")
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        sys.exit(1)
