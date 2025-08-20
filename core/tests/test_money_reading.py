"""Tests for Pokemon Crystal money reading functionality."""

from unittest.mock import MagicMock
import pytest
from core.pyboy_env import PyBoyPokemonCrystalEnv

class MemoryMock:
    """Mock memory class for testing."""
    def __init__(self, data):
        self.data = data
    
    def __getitem__(self, addr):
        return self.data.get(addr, 0)

@pytest.fixture
def money_env():
    """Create a test environment with mocked memory."""
    test_env = PyBoyPokemonCrystalEnv(
        rom_path="roms/pokemon_crystal.gbc",
        save_state_path=None,
        headless=True,
        debug_mode=True,
        enable_monitoring=False
    )
    # Mock PyBoy
    test_env.pyboy = MagicMock()
    yield test_env
    test_env.close()

def test_bcd_money_empty(money_env):
    """Test reading empty/zero money value."""
    # Set up memory mock to return 0 for all bytes
    money_env.pyboy.memory = MemoryMock({})
    
    assert money_env._read_bcd_money() == 0

def test_bcd_money_100(money_env):
    """Test reading ¥100 value."""
    # Set up memory mock to return BCD for 100 (0x00, 0x01, 0x00)
    money_addr = money_env.memory_addresses['money']
    money_env.pyboy.memory = MemoryMock({
        money_addr: 0x00,      # First byte
        money_addr + 1: 0x01,  # Second byte
        money_addr + 2: 0x00   # Third byte
    })
    
    assert money_env._read_bcd_money() == 100

def test_bcd_money_large_value(money_env):
    """Test reading large money value (¥999999)."""
    # Set up memory mock to return BCD for 999999 (0x99, 0x99, 0x99)
    money_addr = money_env.memory_addresses['money']
    money_env.pyboy.memory = MemoryMock({
        money_addr: 0x99,      # First byte
        money_addr + 1: 0x99,  # Second byte
        money_addr + 2: 0x99   # Third byte
    })
    
    assert money_env._read_bcd_money() == 999999

def test_bcd_money_invalid_digits(money_env):
    """Test handling invalid BCD digits (>9)."""
    # Set up memory mock to return invalid BCD (0xAA)
    money_addr = money_env.memory_addresses['money']
    money_env.pyboy.memory = MemoryMock({
        money_addr: 0xAA,      # Invalid BCD
        money_addr + 1: 0xAA,  # Invalid BCD
        money_addr + 2: 0xAA   # Invalid BCD
    })
    
    assert money_env._read_bcd_money() == 0  # Should return 0 for invalid BCD

def test_bcd_money_typical_value(money_env):
    """Test reading typical money value (¥1234)."""
    # Set up memory mock to return BCD for 1234 (0x00, 0x12, 0x34)
    money_addr = money_env.memory_addresses['money']
    money_env.pyboy.memory = MemoryMock({
        money_addr: 0x00,      # First byte
        money_addr + 1: 0x12,  # Second byte
        money_addr + 2: 0x34   # Third byte
    })
    
    assert money_env._read_bcd_money() == 1234

def test_bcd_money_starting_value(money_env):
    """Test reading starting money value (¥3000)."""
    # Set up memory mock to return BCD for 3000 (0x00, 0x30, 0x00)
    money_addr = money_env.memory_addresses['money']
    money_env.pyboy.memory = MemoryMock({
        money_addr: 0x00,      # First byte
        money_addr + 1: 0x30,  # Second byte
        money_addr + 2: 0x00   # Third byte
    })
    
    assert money_env._read_bcd_money() == 3000
