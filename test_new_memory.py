#!/usr/bin/env python3
"""
Test script for updated memory addresses
"""

import sys
import os
from pyboy import PyBoy

# Add the core module to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core.memory_map_new import MEMORY_ADDRESSES

def test_new_addresses():
    rom_path = "roms/pokemon_crystal.gbc"
    
    print("🧪 Testing Updated Memory Addresses")
    print("=" * 50)
    
    # Initialize PyBoy
    pyboy = PyBoy(rom_path, window="null", debug=False)
    
    # Let game stabilize
    for _ in range(300):
        pyboy.tick()
    
    print("\n📋 Testing Updated Address Mappings:")
    
    # Test specific addresses we're confident about
    key_addresses = [
        'money', 'time_of_day', 'alt_x', 'alt_y', 
        'player_x', 'player_y', 'party_count', 'player_level'
    ]
    
    for name in key_addresses:
        if name in MEMORY_ADDRESSES:
            addr = MEMORY_ADDRESSES[name]
            try:
                value = pyboy.memory[addr]
                print(f"  {name:15s} (0x{addr:04X}): {value:3d} (0x{value:02X})")
            except Exception as e:
                print(f"  {name:15s} (0x{addr:04X}): ERROR - {e}")
    
    # Test money calculation (BCD format)
    print(f"\n💰 Money Calculation Test:")
    try:
        money_addr = MEMORY_ADDRESSES['money']
        byte1 = pyboy.memory[money_addr]
        byte2 = pyboy.memory[money_addr + 1] 
        byte3 = pyboy.memory[money_addr + 2]
        
        def bcd_to_decimal(byte):
            high = (byte >> 4) & 0xF
            low = byte & 0xF
            return high * 10 + low if high <= 9 and low <= 9 else 0
        
        money = bcd_to_decimal(byte1) * 10000 + bcd_to_decimal(byte2) * 100 + bcd_to_decimal(byte3)
        print(f"  Raw bytes: {byte1:02X} {byte2:02X} {byte3:02X}")
        print(f"  Calculated money: ¥{money}")
        
    except Exception as e:
        print(f"  Money calculation error: {e}")
    
    pyboy.stop()
    print("\n✅ Test complete!")

if __name__ == "__main__":
    test_new_addresses()
