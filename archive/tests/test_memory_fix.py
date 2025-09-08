#!/usr/bin/env python3
"""
test_memory_fix.py - Quick test to verify money reading works correctly
"""

import os
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from environments.pyboy_env import PyBoyPokemonCrystalEnv

def test_money_reading():
    """Test if money reading is working correctly"""
    print("🧪 Testing Pokemon Crystal money reading...")
    
    # Initialize environment (without save state to avoid version issues)
    env = PyBoyPokemonCrystalEnv(
        rom_path="../roms/pokemon_crystal.gbc",
        save_state_path=None,  # Don't load save state
        headless=True,
        debug_mode=True
    )
    
    try:
        # Reset environment
        obs, info = env.reset()
        
        # Let game run for a few steps to get past intro
        for i in range(50):
            action = 0  # No-op
            obs, reward, terminated, truncated, info = env.step(action)
            
            if i % 10 == 0:
                game_state = env.get_game_state()
                player = game_state.get('player', {})
                
                print(f"Step {i}:")
                print(f"  💰 Money: ${player.get('money', 0)}")
                print(f"  🏆 Badges: {player.get('badges', 0)}")
                print(f"  📍 Location: Map {player.get('map', 0)} at ({player.get('x', 0)}, {player.get('y', 0)})")
                print(f"  🎪 Party size: {len(game_state.get('party', []))}")
                
                # Debug: Read raw memory values
                if env.pyboy:
                    money_addr = env.memory_addresses['money']
                    raw_bytes = []
                    for j in range(3):
                        raw_bytes.append(env.pyboy.memory[money_addr + j])
                    print(f"  🔍 Raw money bytes: {[hex(b) for b in raw_bytes]}")
                
                print()
        
    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        env.close()
    
    print("✅ Memory test completed!")

if __name__ == "__main__":
    test_money_reading()
