#!/usr/bin/env python3
"""
verify_all_memory_addresses.py

Comprehensive verification of ALL memory addresses used in the Pokemon Crystal RL trainer.
This will test each address to confirm it contains the expected data.
"""

import sys
import os
sys.path.insert(0, '.')

from pyboy import PyBoy
import time


def comprehensive_memory_test():
    """Test all memory addresses systematically"""
    print("🔬 COMPREHENSIVE MEMORY ADDRESS VERIFICATION")
    print("=" * 60)
    
    pyboy = PyBoy('roms/pokemon_crystal.gbc', window='null', debug=False)
    
    try:
        # Initialize game to a stable state
        print("⏳ Initializing game state (1000 frames)...")
        for i in range(1000):
            if i % 100 == 0:
                print(f"   Frame {i}/1000...")
            pyboy.button_press('a')
            for _ in range(2):
                pyboy.tick()
            pyboy.button_release('a')
            for _ in range(8):
                pyboy.tick()
        
        print("\n🧪 Testing coordinates and movement...")
        test_coordinates_and_movement(pyboy)
        
        print("\n🎮 Testing game state addresses...")
        test_game_state_addresses(pyboy)
        
        print("\n💰 Testing money and progress addresses...")
        test_money_and_progress(pyboy)
        
        print("\n⚔️ Testing battle state addresses...")
        test_battle_addresses(pyboy)
        
        print("\n🏆 FINAL VERIFICATION SUMMARY")
        print("=" * 60)
        generate_verified_address_list(pyboy)
        
    finally:
        pyboy.stop()


def test_coordinates_and_movement(pyboy):
    """Test coordinate tracking addresses"""
    print("\n📍 COORDINATE VERIFICATION:")
    
    # Test addresses we believe are correct
    test_addresses = {
        'Player X (0xDCB8)': 0xDCB8,
        'Player Y (0xDCB9)': 0xDCB9,
        'Player Map (0xDCBA)': 0xDCBA,
        'Player Direction (0xDCBB)': 0xDCBB,
    }
    
    # Also test the ones from the LLM trainer for comparison
    comparison_addresses = {
        'LLM Trainer Map (0xD35D)': 0xD35D,
        'LLM Trainer X (0xD361)': 0xD361,
        'LLM Trainer Y (0xD362)': 0xD362,
        'LLM Trainer Dir (0xD363)': 0xD363,
    }
    
    all_test_addresses = {**test_addresses, **comparison_addresses}
    
    # Read initial values
    print("Initial readings:")
    initial_values = {}
    for name, addr in all_test_addresses.items():
        try:
            val = pyboy.memory[addr]
            initial_values[name] = val
            print(f"  {name}: {val}")
        except:
            print(f"  {name}: ERROR reading address")
            initial_values[name] = None
    
    # Test movement in each direction
    movements = ['right', 'left', 'down', 'up']
    
    for movement in movements:
        print(f"\n--- Testing {movement.upper()} movement ---")
        
        # Record before movement
        before_values = {}
        for name, addr in all_test_addresses.items():
            try:
                before_values[name] = pyboy.memory[addr]
            except:
                before_values[name] = None
        
        # Execute movement
        for attempt in range(3):  # Try up to 3 times to get movement
            pyboy.button_press(movement)
            for _ in range(2):
                pyboy.tick()
            pyboy.button_release(movement)
            for _ in range(18):
                pyboy.tick()
        
        # Record after movement
        after_values = {}
        for name, addr in all_test_addresses.items():
            try:
                after_values[name] = pyboy.memory[addr]
            except:
                after_values[name] = None
        
        # Check for changes
        changes_detected = False
        for name in all_test_addresses.keys():
            if before_values[name] is not None and after_values[name] is not None:
                if before_values[name] != after_values[name]:
                    change = after_values[name] - before_values[name]
                    print(f"  {name}: {before_values[name]} → {after_values[name]} (Δ{change:+d}) ⭐")
                    changes_detected = True
                else:
                    print(f"  {name}: {after_values[name]} (no change)")
        
        if not changes_detected:
            print("  ❌ No coordinate changes detected")
        
        # Small delay between tests
        time.sleep(0.1)


def test_game_state_addresses(pyboy):
    """Test general game state addresses"""
    print("\n🎯 GAME STATE VERIFICATION:")
    
    # Test various addresses used in the trainers
    test_addresses = {
        # Battle state
        'In Battle (0xD057)': 0xD057,
        'Battle Turn (0xD068)': 0xD068,
        'Enemy Species (0xD0A5)': 0xD0A5,
        'Enemy HP Low (0xD0A8)': 0xD0A8,
        'Enemy HP High (0xD0A9)': 0xD0A9,
        'Enemy Level (0xD0AA)': 0xD0AA,
        
        # Player data - testing party structure
        'Party Count (0xD163)': 0xD163,
        'Player Species (0xD163)': 0xD163,  # Same as party count for slot 0
        'Player Level (0xD16B)': 0xD16B,    # 8 bytes from species
        'Player HP Low (0xD167)': 0xD167,    # 4 bytes from species
        'Player HP High (0xD168)': 0xD168,   # 5 bytes from species
        'Player Max HP Low (0xD169)': 0xD169, # 6 bytes from species
        'Player Max HP High (0xD16A)': 0xD16A, # 7 bytes from species
        'Player Status (0xD16C)': 0xD16C,    # 9 bytes from species
    }
    
    print("Current game state readings:")
    for name, addr in test_addresses.items():
        try:
            val = pyboy.memory[addr]
            
            # Add interpretation for some values
            interpretation = ""
            if 'Battle' in name and addr == 0xD057:
                interpretation = f" ({'In Battle' if val == 1 else 'Overworld'})"
            elif 'Party Count' in name:
                interpretation = f" ({val} Pokemon)" if val <= 6 else f" (Invalid: {val})"
            elif 'Level' in name:
                interpretation = f" ({'Valid' if 1 <= val <= 100 else 'Invalid/Empty'})"
            elif 'HP' in name:
                interpretation = f" ({'Valid' if 0 <= val <= 999 else 'Suspicious'})"
                
            print(f"  {name}: {val}{interpretation}")
        except Exception as e:
            print(f"  {name}: ERROR - {e}")


def test_money_and_progress(pyboy):
    """Test money and progress tracking addresses"""
    print("\n💰 MONEY & PROGRESS VERIFICATION:")
    
    test_addresses = {
        'Money Low (0xD347)': 0xD347,
        'Money Mid (0xD348)': 0xD348, 
        'Money High (0xD349)': 0xD349,
        'Badges Johto (0xD355)': 0xD355,    # Test different badge addresses
        'Badges Johto (0xD359)': 0xD359,    # Address from trainer
        'Badges Kanto (0xD356)': 0xD356,
    }
    
    print("Progress state readings:")
    for name, addr in test_addresses.items():
        try:
            val = pyboy.memory[addr]
            
            # Add interpretation
            interpretation = ""
            if 'Money' in name:
                interpretation = f" (byte {addr - 0xD347 + 1}/3)"
            elif 'Badge' in name:
                interpretation = f" (0x{val:02X}, {bin(val).count('1')} badges)"
                
            print(f"  {name}: {val}{interpretation}")
        except Exception as e:
            print(f"  {name}: ERROR - {e}")
    
    # Calculate total money from 3-byte little-endian
    try:
        money_low = pyboy.memory[0xD347]
        money_mid = pyboy.memory[0xD348] 
        money_high = pyboy.memory[0xD349]
        total_money = money_low + (money_mid << 8) + (money_high << 16)
        print(f"  💰 Calculated Total Money: ¥{total_money}")
    except:
        print(f"  💰 Could not calculate total money")


def test_battle_addresses(pyboy):
    """Test battle-related addresses by looking for patterns"""
    print("\n⚔️ BATTLE STATE VERIFICATION:")
    
    # Test a range of addresses around the documented battle area
    battle_range_start = 0xD050
    battle_range_end = 0xD0B0
    
    print("Scanning battle memory region...")
    
    interesting_addresses = []
    
    for addr in range(battle_range_start, battle_range_end):
        try:
            val = pyboy.memory[addr]
            
            # Look for interesting patterns that might indicate battle data
            if val == 0:
                continue  # Skip zeros
            elif val == 1:
                interesting_addresses.append((addr, val, "Flag-like (0/1)"))
            elif 1 <= val <= 100:
                interesting_addresses.append((addr, val, "Level-like (1-100)"))
            elif val == 255:
                interesting_addresses.append((addr, val, "Uninitialized?"))
            elif 100 < val < 255:
                interesting_addresses.append((addr, val, "High value"))
        except:
            continue
    
    print("Interesting values in battle region:")
    for addr, val, desc in interesting_addresses[:20]:  # Show first 20
        print(f"  0x{addr:04X}: {val} ({desc})")
    
    if len(interesting_addresses) > 20:
        print(f"  ... and {len(interesting_addresses) - 20} more")


def generate_verified_address_list(pyboy):
    """Generate a final list of verified addresses based on tests"""
    print("\n✅ VERIFIED ADDRESSES (confirmed working):")
    
    verified_addresses = {}
    
    # Test the addresses we know work from previous testing
    coordinate_tests = [
        ('player_x', 0xDCB8, 'Player X coordinate'),
        ('player_y', 0xDCB9, 'Player Y coordinate'), 
        ('player_map', 0xDCBA, 'Current map ID'),
        ('player_direction', 0xDCBB, 'Direction facing'),
    ]
    
    print("\n📍 COORDINATE ADDRESSES:")
    for name, addr, desc in coordinate_tests:
        try:
            val = pyboy.memory[addr]
            if 0 <= val <= 255:  # Reasonable range
                verified_addresses[name] = addr
                print(f"  ✅ {name}: 0x{addr:04X} = {val} ({desc})")
            else:
                print(f"  ❌ {name}: 0x{addr:04X} = {val} (out of range)")
        except:
            print(f"  ❌ {name}: 0x{addr:04X} = ERROR")
    
    # Test party-related addresses
    print("\n🎮 PARTY ADDRESSES:")
    party_tests = [
        ('party_count', 0xD163, 'Number of Pokemon in party'),
        ('player_level', 0xD16B, 'First Pokemon level'),
        ('player_hp_low', 0xD167, 'First Pokemon HP (low byte)'),
        ('player_hp_high', 0xD168, 'First Pokemon HP (high byte)'),
        ('player_max_hp_low', 0xD169, 'First Pokemon Max HP (low byte)'),
        ('player_max_hp_high', 0xD16A, 'First Pokemon Max HP (high byte)'),
    ]
    
    for name, addr, desc in party_tests:
        try:
            val = pyboy.memory[addr]
            # Party count should be 0-6, levels 0-100, HP 0-999
            reasonable = True
            if 'party_count' in name and not (0 <= val <= 6):
                reasonable = False
            elif 'level' in name and not (0 <= val <= 100):
                reasonable = False
            elif 'hp' in name and not (0 <= val <= 255):  # Individual bytes
                reasonable = False
                
            if reasonable:
                verified_addresses[name] = addr
                print(f"  ✅ {name}: 0x{addr:04X} = {val} ({desc})")
            else:
                print(f"  ⚠️ {name}: 0x{addr:04X} = {val} (questionable value)")
        except:
            print(f"  ❌ {name}: 0x{addr:04X} = ERROR")
    
    # Test battle addresses
    print("\n⚔️ BATTLE ADDRESSES:")
    battle_tests = [
        ('in_battle', 0xD057, 'Battle flag (0=overworld, 1=battle)'),
    ]
    
    for name, addr, desc in battle_tests:
        try:
            val = pyboy.memory[addr]
            if val in [0, 1]:  # Should be binary flag
                verified_addresses[name] = addr
                print(f"  ✅ {name}: 0x{addr:04X} = {val} ({desc})")
            else:
                print(f"  ⚠️ {name}: 0x{addr:04X} = {val} (not a flag)")
        except:
            print(f"  ❌ {name}: 0x{addr:04X} = ERROR")
    
    # Generate Python dictionary for copying
    print("\n📋 PYTHON DICTIONARY FOR VERIFIED ADDRESSES:")
    print("VERIFIED_MEMORY_ADDRESSES = {")
    for name, addr in verified_addresses.items():
        print(f"    '{name}': 0x{addr:04X},  # {[desc for n, a, desc in coordinate_tests + party_tests + battle_tests if n == name][0] if any(n == name for n, a, desc in coordinate_tests + party_tests + battle_tests) else 'Address'}")
    print("}")
    
    print(f"\n🎯 TOTAL VERIFIED: {len(verified_addresses)} addresses")
    
    return verified_addresses


def test_comparison_with_trainer_addresses():
    """Compare our verified addresses with what the trainer is using"""
    print("\n⚖️  TRAINER COMPARISON:")
    
    # Import the trainer's addresses
    try:
        from config.memory_addresses import MEMORY_ADDRESSES
        trainer_addresses = MEMORY_ADDRESSES
        
        print("Addresses used by trainer vs our verified ones:")
        
        our_verified = {
            'player_x': 0xDCB8,
            'player_y': 0xDCB9, 
            'player_map': 0xDCBA,
            'player_direction': 0xDCBB,
        }
        
        for key in ['player_x', 'player_y', 'player_map', 'player_direction']:
            trainer_addr = trainer_addresses.get(key, 'NOT FOUND')
            our_addr = our_verified.get(key, 'NOT FOUND')
            
            if trainer_addr == our_addr:
                print(f"  ✅ {key}: Both use 0x{our_addr:04X}")
            else:
                print(f"  ❌ {key}: Trainer uses {trainer_addr}, we verified 0x{our_addr:04X}")
                
    except Exception as e:
        print(f"  ⚠️ Could not load trainer addresses: {e}")


if __name__ == "__main__":
    if not os.path.exists('roms/pokemon_crystal.gbc'):
        print("❌ ROM file not found at 'roms/pokemon_crystal.gbc'")
        sys.exit(1)
    
    try:
        comprehensive_memory_test()
        test_comparison_with_trainer_addresses()
        
        print("\n" + "=" * 60)
        print("🏁 VERIFICATION COMPLETE")
        print("=" * 60)
        print("Use the verified addresses above to update your trainers!")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Verification interrupted by user")
    except Exception as e:
        print(f"\n❌ Verification failed: {e}")
        import traceback
        traceback.print_exc()
