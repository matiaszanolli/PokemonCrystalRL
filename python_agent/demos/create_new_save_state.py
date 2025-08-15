#!/usr/bin/env python3
"""
create_new_save_state.py - Create a new save state at the game start with $3000
"""

import os
import sys
sys.path.append('.')

from pyboy_env import PyBoyPokemonCrystalEnv

def create_fresh_save_state():
    """Create a save state at the beginning of the game with proper initialization"""
    print("🎮 Creating fresh Pokemon Crystal save state...")
    
    # Initialize environment without save state
    env = PyBoyPokemonCrystalEnv(
        rom_path="../roms/pokemon_crystal.gbc",
        save_state_path=None,
        headless=False,  # Show window to see what's happening
        debug_mode=True
    )
    
    try:
        print("📥 Resetting environment...")
        obs, info = env.reset()
        
        print("⏭️ Running game to get past intro and initialize data...")
        
        # Run the game for many frames to get past the intro sequence
        # and initialize the game properly
        step = 0
        money_found = False
        
        while step < 2000 and not money_found:  # Maximum 2000 steps
            # Occasionally press A to advance through intro screens
            if step % 60 == 0:  # Press A every 60 frames
                action = 5  # A button
            else:
                action = 0  # No-op
            
            obs, reward, terminated, truncated, info = env.step(action)
            step += 1
            
            if step % 100 == 0:
                game_state = env.get_game_state()
                player = game_state.get('player', {})
                money = player.get('money', 0)
                
                print(f"Step {step}: Money = ${money}")
                
                # Check if we've reached the point where money is initialized to 3000
                if money == 3000:
                    print(f"🎉 Found the starting money of $3000 at step {step}!")
                    money_found = True
                    break
                elif money > 0 and money != 3000:
                    print(f"📊 Found money: ${money} (not the expected $3000)")
        
        if money_found:
            # Save the state
            save_path = "../pokecrystal_fresh_start.ss1"
            env.save_state(save_path)
            print(f"💾 Save state created at: {save_path}")
            
            # Verify the save state
            final_state = env.get_game_state()
            player = final_state.get('player', {})
            print(f"✅ Final state saved with:")
            print(f"   💰 Money: ${player.get('money', 0)}")
            print(f"   🏆 Badges: {player.get('badges', 0)}")
            print(f"   📍 Location: Map {player.get('map', 0)} at ({player.get('x', 0)}, {player.get('y', 0)})")
            print(f"   🎪 Party size: {len(final_state.get('party', []))}")
        else:
            print("❌ Could not find the expected starting money of $3000")
            print("💡 The game might need manual input to start properly")
            
            # Save anyway for debugging
            save_path = "../pokecrystal_debug.ss1"
            env.save_state(save_path)
            print(f"💾 Debug save state created at: {save_path}")
        
    except Exception as e:
        print(f"❌ Error during save state creation: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        env.close()
    
    print("✅ Save state creation completed!")

if __name__ == "__main__":
    create_fresh_save_state()
