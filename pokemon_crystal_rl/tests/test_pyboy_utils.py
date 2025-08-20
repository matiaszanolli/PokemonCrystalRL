"""
Test utilities for PyBoy tests
"""

import os

def get_rom_path():
    """Get the absolute path to the test ROM."""
    test_rom_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "roms", "pokemon_crystal.gbc")
    return test_rom_path
