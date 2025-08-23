#!/usr/bin/env python3
"""
Create a minimal valid Game Boy ROM for testing purposes.
This ROM will have a proper header with valid checksum.
"""

import struct

def calculate_header_checksum(header_data):
    """Calculate the Game Boy cartridge header checksum."""
    checksum = 0
    # Checksum is calculated over bytes 0x134-0x14C (title, manufacturer, etc.)
    for i in range(0x34, 0x4D):  # 0x134-0x14C relative to header start (0x100)
        checksum = (checksum - header_data[i] - 1) & 0xFF
    return checksum

def create_minimal_gameboy_rom(filename):
    """Create a minimal valid Game Boy ROM file."""
    # Create a 32KB ROM (minimum size)
    rom_data = bytearray(32768)
    
    # Fill with NOP instructions (0x00)
    for i in range(len(rom_data)):
        rom_data[i] = 0x00
    
    # Add entry point at 0x100 (jump to 0x150)
    rom_data[0x100] = 0x00  # NOP
    rom_data[0x101] = 0xC3  # JP instruction
    rom_data[0x102] = 0x50  # Low byte of 0x150
    rom_data[0x103] = 0x01  # High byte of 0x150
    
    # Nintendo logo (required at 0x104-0x133)
    nintendo_logo = [
        0xCE, 0xED, 0x66, 0x66, 0xCC, 0x0D, 0x00, 0x0B, 0x03, 0x73, 0x00, 0x83,
        0x00, 0x0C, 0x00, 0x0D, 0x00, 0x08, 0x11, 0x1F, 0x88, 0x89, 0x00, 0x0E,
        0xDC, 0xCC, 0x6E, 0xE6, 0xDD, 0xDD, 0xD9, 0x99, 0xBB, 0xBB, 0x67, 0x63,
        0x6E, 0x0E, 0xEC, 0xCC, 0xDD, 0xDC, 0x99, 0x9F, 0xBB, 0xB9, 0x33, 0x3E
    ]
    
    for i, byte in enumerate(nintendo_logo):
        rom_data[0x104 + i] = byte
    
    # Game title (0x134-0x143) - "TEST ROM" padded with nulls
    title = "TEST ROM"
    for i, char in enumerate(title):
        if i < 16:  # Max 16 characters
            rom_data[0x134 + i] = ord(char)
    
    # Manufacturer code (0x13F-0x142) - leave as 0x00
    
    # CGB flag (0x143) - 0x00 for original Game Boy
    rom_data[0x143] = 0x00
    
    # New licensee code (0x144-0x145)
    rom_data[0x144] = 0x00
    rom_data[0x145] = 0x00
    
    # SGB flag (0x146) - 0x00 for no SGB support
    rom_data[0x146] = 0x00
    
    # Cartridge type (0x147) - 0x00 for ROM only
    rom_data[0x147] = 0x00
    
    # ROM size (0x148) - 0x00 for 32KB
    rom_data[0x148] = 0x00
    
    # RAM size (0x149) - 0x00 for no RAM
    rom_data[0x149] = 0x00
    
    # Destination code (0x14A) - 0x01 for non-Japanese
    rom_data[0x14A] = 0x01
    
    # Old licensee code (0x14B) - 0x33 for new licensee
    rom_data[0x14B] = 0x33
    
    # Mask ROM version (0x14C) - 0x00
    rom_data[0x14C] = 0x00
    
    # Calculate and set header checksum (0x14D)
    header_checksum = calculate_header_checksum(rom_data[0x100:])
    rom_data[0x14D] = header_checksum
    
    # Global checksum (0x14E-0x14F) - calculate over entire ROM except these bytes
    global_checksum = 0
    for i in range(len(rom_data)):
        if i != 0x14E and i != 0x14F:
            global_checksum = (global_checksum + rom_data[i]) & 0xFFFF
    
    rom_data[0x14E] = (global_checksum >> 8) & 0xFF  # High byte
    rom_data[0x14F] = global_checksum & 0xFF         # Low byte
    
    # Add some code at 0x150 (where we jump to)
    rom_data[0x150] = 0x76  # HALT instruction
    
    # Write the ROM file
    with open(filename, 'wb') as f:
        f.write(rom_data)
    
    print(f"Created minimal Game Boy ROM: {filename}")
    print(f"ROM size: {len(rom_data)} bytes")
    print(f"Header checksum: 0x{header_checksum:02X}")
    print(f"Global checksum: 0x{global_checksum:04X}")

if __name__ == "__main__":
    create_minimal_gameboy_rom("test_valid.gbc")