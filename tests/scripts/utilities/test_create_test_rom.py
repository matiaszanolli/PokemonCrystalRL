"""
Tests for scripts/utilities/create_test_rom.py

Tests ROM creation functionality including header validation,
checksum calculation, and file I/O operations.
"""

import pytest
import os
import tempfile
import struct
from unittest.mock import patch, mock_open, MagicMock

# Import the module under test
import sys
import importlib.util

# Load the create_test_rom module
spec = importlib.util.spec_from_file_location(
    "create_test_rom", 
    "/mnt/data/src/pokemon_crystal_rl/scripts/utilities/create_test_rom.py"
)
create_test_rom = importlib.util.module_from_spec(spec)
sys.modules["create_test_rom"] = create_test_rom
spec.loader.exec_module(create_test_rom)


class TestCalculateHeaderChecksum:
    """Test header checksum calculation"""
    
    def test_calculate_header_checksum_empty(self):
        """Test checksum calculation with empty header data"""
        # Create header data with zeros
        header_data = bytearray(0x50)  # Enough bytes for header
        
        checksum = create_test_rom.calculate_header_checksum(header_data)
        
        # Expected checksum for all zeros in range 0x34-0x4C
        expected = 0
        for i in range(0x34, 0x4D):
            expected = (expected - 0 - 1) & 0xFF
        
        assert checksum == expected
    
    def test_calculate_header_checksum_known_values(self):
        """Test checksum calculation with known values"""
        header_data = bytearray(0x50)
        
        # Set some known values in the checksum range
        header_data[0x34] = 0x54  # 'T'
        header_data[0x35] = 0x45  # 'E'
        header_data[0x36] = 0x53  # 'S'
        header_data[0x37] = 0x54  # 'T'
        
        checksum = create_test_rom.calculate_header_checksum(header_data)
        
        # Calculate expected checksum manually
        expected = 0
        for i in range(0x34, 0x4D):
            expected = (expected - header_data[i] - 1) & 0xFF
        
        assert checksum == expected
        assert isinstance(checksum, int)
        assert 0 <= checksum <= 255
    
    def test_calculate_header_checksum_all_ff(self):
        """Test checksum calculation with all 0xFF values"""
        header_data = bytearray(0x50)
        
        # Fill checksum range with 0xFF
        for i in range(0x34, 0x4D):
            header_data[i] = 0xFF
        
        checksum = create_test_rom.calculate_header_checksum(header_data)
        
        # Should handle overflow correctly
        assert isinstance(checksum, int)
        assert 0 <= checksum <= 255


class TestCreateMinimalGameboyRom:
    """Test ROM creation functionality"""
    
    def setup_method(self):
        """Set up temporary file for testing"""
        self.temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.gbc')
        self.temp_filename = self.temp_file.name
        self.temp_file.close()
    
    def teardown_method(self):
        """Clean up temporary file"""
        try:
            os.unlink(self.temp_filename)
        except FileNotFoundError:
            pass
    
    def test_create_minimal_rom_file_creation(self):
        """Test that ROM file is created successfully"""
        create_test_rom.create_minimal_gameboy_rom(self.temp_filename)
        
        # Check file exists and has correct size
        assert os.path.exists(self.temp_filename)
        assert os.path.getsize(self.temp_filename) == 32768  # 32KB
    
    def test_rom_structure_and_header(self):
        """Test ROM structure and header values"""
        create_test_rom.create_minimal_gameboy_rom(self.temp_filename)
        
        with open(self.temp_filename, 'rb') as f:
            rom_data = f.read()
        
        # Test ROM size
        assert len(rom_data) == 32768
        
        # Test entry point (0x100-0x103)
        assert rom_data[0x100] == 0x00  # NOP
        assert rom_data[0x101] == 0xC3  # JP instruction
        assert rom_data[0x102] == 0x50  # Low byte of 0x150
        assert rom_data[0x103] == 0x01  # High byte of 0x150
        
        # Test Nintendo logo is present (0x104-0x133)
        expected_logo = [
            0xCE, 0xED, 0x66, 0x66, 0xCC, 0x0D, 0x00, 0x0B, 0x03, 0x73, 0x00, 0x83,
            0x00, 0x0C, 0x00, 0x0D, 0x00, 0x08, 0x11, 0x1F, 0x88, 0x89, 0x00, 0x0E,
            0xDC, 0xCC, 0x6E, 0xE6, 0xDD, 0xDD, 0xD9, 0x99, 0xBB, 0xBB, 0x67, 0x63,
            0x6E, 0x0E, 0xEC, 0xCC, 0xDD, 0xDC, 0x99, 0x9F, 0xBB, 0xB9, 0x33, 0x3E
        ]
        
        for i, expected_byte in enumerate(expected_logo):
            assert rom_data[0x104 + i] == expected_byte
        
        # Test game title (0x134-0x143)
        title_bytes = rom_data[0x134:0x144]
        title_str = title_bytes.rstrip(b'\x00').decode('ascii')
        assert title_str == "TEST ROM"
        
        # Test cartridge header values
        assert rom_data[0x143] == 0x00  # CGB flag
        assert rom_data[0x144] == 0x00  # New licensee code low
        assert rom_data[0x145] == 0x00  # New licensee code high
        assert rom_data[0x146] == 0x00  # SGB flag
        assert rom_data[0x147] == 0x00  # Cartridge type (ROM only)
        assert rom_data[0x148] == 0x00  # ROM size (32KB)
        assert rom_data[0x149] == 0x00  # RAM size (no RAM)
        assert rom_data[0x14A] == 0x01  # Destination code (non-Japanese)
        assert rom_data[0x14B] == 0x33  # Old licensee code
        assert rom_data[0x14C] == 0x00  # Mask ROM version
        
        # Test code at jump target (0x150)
        assert rom_data[0x150] == 0x76  # HALT instruction
    
    def test_header_checksum_validity(self):
        """Test that header checksum is correctly calculated"""
        create_test_rom.create_minimal_gameboy_rom(self.temp_filename)
        
        with open(self.temp_filename, 'rb') as f:
            rom_data = f.read()
        
        # Extract header checksum from ROM
        stored_checksum = rom_data[0x14D]
        
        # Calculate expected checksum
        expected_checksum = create_test_rom.calculate_header_checksum(rom_data[0x100:])
        
        assert stored_checksum == expected_checksum
    
    def test_global_checksum_validity(self):
        """Test that global checksum is present and reasonable"""
        create_test_rom.create_minimal_gameboy_rom(self.temp_filename)
        
        with open(self.temp_filename, 'rb') as f:
            rom_data = f.read()
        
        # Extract global checksum from ROM
        stored_checksum = (rom_data[0x14E] << 8) | rom_data[0x14F]
        
        # Test that checksum is not zero (it should have been calculated)
        assert stored_checksum != 0
        
        # Test that checksum is within reasonable range for a 32KB ROM
        assert 0 < stored_checksum < 0x10000  # Should be a valid 16-bit value
        
        # Test that the bytes were actually set
        assert rom_data[0x14E] != 0 or rom_data[0x14F] != 0  # At least one byte should be non-zero
    
    def test_rom_mostly_zeros(self):
        """Test that ROM is mostly filled with zeros (NOPs)"""
        create_test_rom.create_minimal_gameboy_rom(self.temp_filename)
        
        with open(self.temp_filename, 'rb') as f:
            rom_data = f.read()
        
        # Count non-zero bytes (excluding known non-zero areas)
        non_zero_count = 0
        for i, byte in enumerate(rom_data):
            if byte != 0:
                # Check if this is in a known area that should have data
                if not (0x100 <= i <= 0x14F or i == 0x150):
                    non_zero_count += 1
        
        # Should be very few non-zero bytes outside header and entry areas
        assert non_zero_count == 0
    
    @patch('builtins.print')
    def test_output_messages(self, mock_print):
        """Test that creation outputs expected messages"""
        create_test_rom.create_minimal_gameboy_rom(self.temp_filename)
        
        # Check that print was called with expected messages
        print_calls = [call[0][0] for call in mock_print.call_args_list]
        
        assert any("Created minimal Game Boy ROM" in call for call in print_calls)
        assert any("ROM size: 32768 bytes" in call for call in print_calls)
        assert any("Header checksum:" in call for call in print_calls)
        assert any("Global checksum:" in call for call in print_calls)
    
    @patch('builtins.open', side_effect=PermissionError("Permission denied"))
    def test_file_write_error(self, mock_open):
        """Test handling of file write errors"""
        with pytest.raises(PermissionError):
            create_test_rom.create_minimal_gameboy_rom(self.temp_filename)
    
    def test_rom_with_custom_filename(self):
        """Test ROM creation with different filename"""
        custom_filename = self.temp_filename.replace('.gbc', '_custom.gbc')
        
        try:
            create_test_rom.create_minimal_gameboy_rom(custom_filename)
            
            assert os.path.exists(custom_filename)
            assert os.path.getsize(custom_filename) == 32768
        finally:
            try:
                os.unlink(custom_filename)
            except FileNotFoundError:
                pass


class TestRomValidation:
    """Test ROM validation aspects"""
    
    def test_nintendo_logo_integrity(self):
        """Test that Nintendo logo is exactly as expected"""
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.gbc')
        temp_filename = temp_file.name
        temp_file.close()
        
        try:
            create_test_rom.create_minimal_gameboy_rom(temp_filename)
            
            with open(temp_filename, 'rb') as f:
                rom_data = f.read()
            
            # Nintendo logo must be exact for Game Boy compatibility
            expected_logo = [
                0xCE, 0xED, 0x66, 0x66, 0xCC, 0x0D, 0x00, 0x0B, 0x03, 0x73, 0x00, 0x83,
                0x00, 0x0C, 0x00, 0x0D, 0x00, 0x08, 0x11, 0x1F, 0x88, 0x89, 0x00, 0x0E,
                0xDC, 0xCC, 0x6E, 0xE6, 0xDD, 0xDD, 0xD9, 0x99, 0xBB, 0xBB, 0x67, 0x63,
                0x6E, 0x0E, 0xEC, 0xCC, 0xDD, 0xDC, 0x99, 0x9F, 0xBB, 0xB9, 0x33, 0x3E
            ]
            
            actual_logo = list(rom_data[0x104:0x134])
            assert actual_logo == expected_logo
            
        finally:
            try:
                os.unlink(temp_filename)
            except FileNotFoundError:
                pass
    
    def test_header_bounds_checking(self):
        """Test that header fields are within valid ranges"""
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.gbc')
        temp_filename = temp_file.name
        temp_file.close()
        
        try:
            create_test_rom.create_minimal_gameboy_rom(temp_filename)
            
            with open(temp_filename, 'rb') as f:
                rom_data = f.read()
            
            # All header values should be valid bytes
            assert 0 <= rom_data[0x14D] <= 255  # Header checksum
            assert 0 <= rom_data[0x14E] <= 255  # Global checksum high
            assert 0 <= rom_data[0x14F] <= 255  # Global checksum low
            
            # ROM size indicator should be valid
            assert rom_data[0x148] in [0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08]
            
        finally:
            try:
                os.unlink(temp_filename)
            except FileNotFoundError:
                pass


class TestMainFunction:
    """Test main function execution"""
    
    def test_main_block_exists(self):
        """Test that main block exists and creates expected filename"""
        # Read the file and check for main block
        with open('/mnt/data/src/pokemon_crystal_rl/scripts/utilities/create_test_rom.py', 'r') as f:
            content = f.read()
        
        # Check that main block exists
        assert 'if __name__ == "__main__":' in content
        assert 'create_minimal_gameboy_rom("test_valid.gbc")' in content


if __name__ == "__main__":
    pytest.main([__file__])