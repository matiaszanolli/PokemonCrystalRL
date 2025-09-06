#!/usr/bin/env python3
"""
Memory Address Debugging Tool for Pokemon Crystal

This script helps find the correct memory addresses by scanning
for patterns and known values in the Pokemon Crystal ROM.
"""

import sys
import os
from pyboy import PyBoy
import numpy as np
import time

# Add the core module to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core.memory_map import MEMORY_ADDRESSES

def scan_memory_range(pyboy, start_addr, end_addr, description=""):
    """Scan a range of memory addresses and show non-zero values"""
    print(f"\n🔍 Scanning {description} (0x{start_addr:04X} - 0x{end_addr:04X}):")
    
    non_zero_values = []
    for addr in range(start_addr, end_addr + 1):
        try:
            value = pyboy.memory[addr]
            if value != 0:
                non_zero_values.append((addr, value))
        except:
            pass
    
    if non_zero_values:
        print(f"Found {len(non_zero_values)} non-zero values:")
        for addr, value in non_zero_values[:20]:  # Show first 20
            print(f"  0x{addr:04X}: {value:3d} (0x{value:02X}) {chr(value) if 32 <= value <= 126 else ''}")
        if len(non_zero_values) > 20:
            print(f"  ... and {len(non_zero_values) - 20} more")
    else:
        print("  No non-zero values found")

def check_known_patterns(pyboy):
    """Check for known Pokemon Crystal patterns"""
    print("\n🎯 Looking for Known Patterns:")
    
    # Look for text patterns (Pokemon names, items, etc.)
    text_ranges = [
        (0xD000, 0xDFFF, "Work RAM"),
        (0xC000, 0xCFFF, "Video RAM / Work RAM"), 
        (0xFF80, 0xFFFE, "High RAM")
    ]
    
    for start, end, name in text_ranges:
        scan_memory_range(pyboy, start, end, name)

def test_current_addresses(pyboy):
    """Test our current memory addresses"""
    print("\n📋 Testing Current Address Mappings:")
    
    for name, addr in MEMORY_ADDRESSES.items():
        try:
            value = pyboy.memory[addr]
            print(f"  {name:20s} (0x{addr:04X}): {value:3d} (0x{value:02X})")
        except Exception as e:
            print(f"  {name:20s} (0x{addr:04X}): ERROR - {e}")

def monitor_memory_changes(pyboy, seconds=10):
    """Monitor memory for changes over time"""
    print(f"\n⏱️ Monitoring memory changes for {seconds} seconds...")
    print("Press some buttons during this time!")
    
    # Take initial snapshot
    initial_memory = {}
    for addr in range(0xC000, 0xE000):  # Work RAM area
        try:
            initial_memory[addr] = pyboy.memory[addr]
        except:
            pass
    
    time.sleep(seconds)
    
    # Compare with current state
    changed_addresses = []
    for addr, initial_value in initial_memory.items():
        try:
            current_value = pyboy.memory[addr]
            if current_value != initial_value:
                changed_addresses.append((addr, initial_value, current_value))
        except:
            pass
    
    if changed_addresses:
        print(f"Found {len(changed_addresses)} changed addresses:")
        for addr, old_val, new_val in changed_addresses[:30]:  # Show first 30
            print(f"  0x{addr:04X}: {old_val:3d} -> {new_val:3d} (diff: {new_val - old_val:+d})")
        if len(changed_addresses) > 30:
            print(f"  ... and {len(changed_addresses) - 30} more")
    else:
        print("No memory changes detected")

def find_player_position(pyboy):
    """Try to find player position by looking for coordinate patterns"""
    print("\n🎯 Looking for Player Position (coordinates 0-255):")
    
    # Player coordinates should be in a reasonable range and close together
    position_candidates = []
    
    for addr in range(0xD000, 0xE000):
        try:
            x_val = pyboy.memory[addr]
            y_val = pyboy.memory[addr + 1]
            
            # Look for reasonable coordinate pairs
            if 0 <= x_val <= 255 and 0 <= y_val <= 255 and (x_val + y_val) > 0:
                position_candidates.append((addr, x_val, y_val))
        except:
            pass
    
    print(f"Found {len(position_candidates)} potential coordinate pairs:")
    for addr, x, y in position_candidates[:20]:
        print(f"  0x{addr:04X}: X={x:3d}, Y={y:3d}")

def main():
    rom_path = "roms/pokemon_crystal.gbc"
    if not os.path.exists(rom_path):
        print(f"❌ ROM file not found: {rom_path}")
        return
    
    print("🔧 Pokemon Crystal Memory Debugging Tool")
    print("=" * 50)
    
    # Initialize PyBoy
    pyboy = PyBoy(rom_path, window="null", debug=False)
    print("✅ PyBoy initialized")
    
    # Let the game run for a few frames to stabilize
    for _ in range(100):
        pyboy.tick()
    
    try:
        # Test current addresses
        test_current_addresses(pyboy)
        
        # Scan memory ranges
        check_known_patterns(pyboy)
        
        # Look for player position specifically
        find_player_position(pyboy)
        
        # Monitor changes (commented out for non-interactive mode)
        # print("\n⚠️ Skipping interactive monitoring (uncomment to enable)")
        # monitor_memory_changes(pyboy, 5)
        
    finally:
        pyboy.stop()
    
    print("\n🎯 Debugging complete!")
    print("\nNext steps:")
    print("1. Look for patterns in the non-zero values")
    print("2. Cross-reference with known Pokemon Crystal memory maps")
    print("3. Test specific addresses by making changes in-game")

if __name__ == "__main__":
    main()
