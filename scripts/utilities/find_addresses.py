#!/usr/bin/env python3
"""
Advanced Pokemon Crystal Address Finder

This script uses known patterns and heuristics to find the correct
memory addresses for different versions of Pokemon Crystal.
"""

import sys
import os
from pyboy import PyBoy
import numpy as np
import time

def test_potential_addresses(pyboy):
    """Test potential addresses based on common Pokemon Crystal memory layouts"""
    
    print("🔍 Testing Common Pokemon Crystal Memory Layouts:")
    
    # Common Work RAM ranges for Pokemon games
    potential_ranges = [
        (0xD100, 0xD200, "Player Data Range 1"),
        (0xD200, 0xD300, "Player Data Range 2"), 
        (0xD300, 0xD400, "Party Data Range"),
        (0xD400, 0xD500, "Game State Range"),
        (0xD500, 0xD600, "Items/Money Range"),
        (0xD800, 0xD900, "Progress/Badges Range"),
    ]
    
    for start, end, name in potential_ranges:
        print(f"\n🎯 {name} (0x{start:04X}-0x{end:04X}):")
        non_zero = []
        for addr in range(start, end):
            try:
                value = pyboy.memory[addr]
                if value != 0:
                    non_zero.append((addr, value))
            except:
                pass
        
        if non_zero:
            print(f"  Found {len(non_zero)} non-zero values:")
            for addr, value in non_zero[:10]:  # Show first 10
                char = chr(value) if 32 <= value <= 126 else ''
                print(f"    0x{addr:04X}: {value:3d} (0x{value:02X}) '{char}'")
            if len(non_zero) > 10:
                print(f"    ... and {len(non_zero) - 10} more")
        else:
            print("  No non-zero values found")

def find_coordinates_pattern(pyboy):
    """Look for coordinate-like patterns in memory"""
    print("\n🎯 Advanced Coordinate Pattern Search:")
    
    # Look for typical coordinate patterns
    for base_addr in range(0xD000, 0xE000, 4):  # Check every 4 bytes
        try:
            # Check for coordinate-like values (0-255, reasonable for game maps)
            vals = [pyboy.memory[base_addr + i] for i in range(8)]
            
            # Look for patterns that might be X, Y coordinates
            if (0 <= vals[0] <= 50 and 0 <= vals[1] <= 50 and  # Reasonable coordinates
                (vals[0] > 0 or vals[1] > 0) and  # At least one non-zero
                vals[2] == 0 and vals[3] == 0):  # Some padding/unused bytes
                
                print(f"  0x{base_addr:04X}: X={vals[0]:2d}, Y={vals[1]:2d} (pattern: {vals[:4]})")
                
        except:
            pass

def scan_for_text_patterns(pyboy):
    """Look for text/name patterns that might indicate player or Pokemon names"""
    print("\n📝 Looking for Text Patterns:")
    
    # Look for potential text in typical ranges
    text_ranges = [(0xD000, 0xD100), (0xD400, 0xD500)]
    
    for start, end in text_ranges:
        print(f"\n  Scanning 0x{start:04X}-0x{end:04X} for text:")
        text_candidates = []
        
        for addr in range(start, end - 10):
            try:
                # Read 10 bytes and check if they look like text
                chars = []
                valid_text = True
                
                for i in range(10):
                    byte = pyboy.memory[addr + i]
                    if byte == 0:  # String terminator
                        break
                    elif 32 <= byte <= 126:  # Printable ASCII
                        chars.append(chr(byte))
                    elif byte == 0x50:  # Pokemon uses 0x50 for some chars
                        chars.append('?')
                    else:
                        valid_text = False
                        break
                
                if valid_text and len(chars) >= 3:
                    text = ''.join(chars)
                    text_candidates.append((addr, text))
                    
            except:
                pass
        
        if text_candidates:
            for addr, text in text_candidates[:5]:  # Show first 5
                print(f"    0x{addr:04X}: '{text}'")

def test_money_addresses(pyboy):
    """Look for money values (BCD format typically)"""
    print("\n💰 Looking for Money Values (BCD format):")
    
    # Money in Pokemon is often stored in BCD (Binary Coded Decimal)
    # Starting money is usually around 3000 (0x30 0x00 in BCD)
    for addr in range(0xD800, 0xDA00):
        try:
            byte1 = pyboy.memory[addr]
            byte2 = pyboy.memory[addr + 1] 
            byte3 = pyboy.memory[addr + 2]
            
            # Check if this looks like BCD money format
            if (0 <= byte1 <= 0x99 and 0 <= byte2 <= 0x99 and 0 <= byte3 <= 0x99):
                # Convert from BCD to decimal
                def bcd_to_decimal(byte):
                    high = (byte >> 4) & 0xF
                    low = byte & 0xF
                    if high <= 9 and low <= 9:
                        return high * 10 + low
                    return 0
                
                money = bcd_to_decimal(byte1) * 10000 + bcd_to_decimal(byte2) * 100 + bcd_to_decimal(byte3)
                if 0 < money <= 999999:  # Reasonable money range
                    print(f"  0x{addr:04X}: ¥{money} (BCD: {byte1:02X} {byte2:02X} {byte3:02X})")
                    
        except:
            pass

def comprehensive_analysis(pyboy):
    """Run all analysis functions"""
    test_potential_addresses(pyboy)
    find_coordinates_pattern(pyboy)
    scan_for_text_patterns(pyboy)
    test_money_addresses(pyboy)

def main():
    rom_path = "roms/pokemon_crystal.gbc"
    if not os.path.exists(rom_path):
        print(f"❌ ROM file not found: {rom_path}")
        return
    
    print("🔍 Advanced Pokemon Crystal Address Finder")
    print("=" * 60)
    
    # Initialize PyBoy
    pyboy = PyBoy(rom_path, window="null", debug=False)
    print("✅ PyBoy initialized")
    
    # Let the game boot and stabilize
    print("⏳ Letting game boot...")
    for _ in range(500):  # More boot time
        pyboy.tick()
    
    # Execute some actions to get into gameplay
    print("🎮 Pressing some buttons to get into gameplay...")
    actions = ['a', 'a', 'a', 'start', 'a', 'a', 'a']
    
    for action in actions:
        pyboy.button_press(action)
        for _ in range(10):
            pyboy.tick()
        pyboy.button_release(action)
        for _ in range(10):
            pyboy.tick()
    
    try:
        comprehensive_analysis(pyboy)
        
    finally:
        pyboy.stop()
    
    print("\n🎯 Analysis complete!")
    print("\nRecommendations:")
    print("1. Look for consistent patterns in the coordinate and text data")
    print("2. Try the addresses that show reasonable values")
    print("3. Test by making changes in-game and watching memory")

if __name__ == "__main__":
    main()
