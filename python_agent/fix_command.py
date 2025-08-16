#!/usr/bin/env python3
"""
Fix for the file not found error.

The user tried: python enhanced_monitored_training.py --rom ../roms/pokemon_crystal.gbc --episodes 5
The correct command is: python run_pokemon_trainer.py --rom ../roms/pokemon_crystal.gbc --mode fast_monitored --episodes 5 --web

This script shows the correction and can run the fixed command.
"""

import subprocess
import sys
import os

def show_fix():
    print("🔧 File Not Found Error - FIXED!")
    print("=" * 40)
    print()
    print("❌ You tried:")
    print("   python enhanced_monitored_training.py --rom ../roms/pokemon_crystal.gbc --episodes 5")
    print()
    print("✅ Correct command:")
    print("   python run_pokemon_trainer.py --rom ../roms/pokemon_crystal.gbc --mode fast_monitored --episodes 5 --web")
    print()
    print("📝 Key changes:")
    print("   • File: 'enhanced_monitored_training.py' → 'run_pokemon_trainer.py'")
    print("   • Added: '--mode fast_monitored' (required parameter)")
    print("   • Added: '--web' (enables monitoring dashboard)")
    print()

def run_fixed_command():
    """Run the corrected command"""
    print("🚀 Running the corrected command...")
    
    # Check if ROM exists
    rom_path = "../roms/pokemon_crystal.gbc"
    if not os.path.exists(rom_path):
        print(f"❌ ROM file not found: {rom_path}")
        print("   Make sure the ROM file exists in the correct location.")
        return False
    
    # Run the corrected command
    cmd = [
        sys.executable, 
        "run_pokemon_trainer.py", 
        "--rom", rom_path,
        "--mode", "fast_monitored",
        "--episodes", "5",
        "--web",
        "--debug"
    ]
    
    print(f"   Command: {' '.join(cmd)}")
    print()
    
    try:
        subprocess.run(cmd)
        return True
    except KeyboardInterrupt:
        print("\n⏸️ Training interrupted by user")
        return True
    except Exception as e:
        print(f"❌ Error running command: {e}")
        return False

if __name__ == "__main__":
    show_fix()
    
    response = input("Would you like to run the corrected command now? (y/n): ").strip().lower()
    if response in ['y', 'yes']:
        run_fixed_command()
    else:
        print("\n📋 To run manually, copy and paste this command:")
        print("   python run_pokemon_trainer.py --rom ../roms/pokemon_crystal.gbc --mode fast_monitored --episodes 5 --web")
