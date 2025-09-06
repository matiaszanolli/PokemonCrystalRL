"""Test suite for PyBoy environment."""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from core.pyboy_env import PyBoyPokemonCrystalEnv
from core.state.machine import PyBoyGameState


@pytest.fixture
def mock_pyboy():
    """Create a mock PyBoy instance."""
    mock = Mock()
    # Set up memory with test values
    mock.memory = {
        0xDCB8: 10,  # player_x
        0xDCB9: 20,  # player_y
        0xDCB5: 1,   # player_map
        0xDCB6: 1,   # player_direction
        0xD84E: 0x01,  # money (BCD format)
        0xD84F: 0x00,
        0xD850: 0x00,
        0xD855: 0b00000111,  # 3 Johto badges
        0xD856: 0b00000001,  # 1 Kanto badge
        0xDCD7: 2,  # party_count
    }
    
    # Set up party Pokémon data
    party_start = 0xDCDF
    pokemon1_data = {
        party_start: 25,       # species (Pikachu)
        party_start + 31: 50,  # level
        party_start + 34: 0,   # hp high byte
        party_start + 35: 100, # hp low byte
        party_start + 36: 0,   # max hp high byte
        party_start + 37: 150, # max hp low byte
        party_start + 32: 0,   # status
        party_start + 8: 0,    # exp high byte
        party_start + 9: 1,    # exp mid byte
        party_start + 10: 0,   # exp low byte
    }
    pokemon2_data = {
        party_start + 48: 6,      # species (Charizard)
        party_start + 79: 45,     # level
        party_start + 82: 0,      # hp high byte
        party_start + 83: 200,    # hp low byte
        party_start + 84: 0,      # max hp high byte
        party_start + 85: 250,    # max hp low byte
        party_start + 80: 0,      # status
        party_start + 56: 0,      # exp high byte
        party_start + 57: 2,      # exp mid byte
        party_start + 58: 0,      # exp low byte
    }
    mock.memory.update(pokemon1_data)
    mock.memory.update(pokemon2_data)

    # Add button_press method for action testing
    mock.button_press = Mock()

    # Add screen_image method that returns the mock screen
    def screen_image():
        return mock.screen.ndarray
    mock.screen_image = screen_image
    
    # Mock get_memory_value to use our mock memory dict
    def get_memory_value(addr):
        return mock.memory.get(addr, 0)
    mock.get_memory_value = get_memory_value
    
    # Mock screen methods
    mock_screen = Mock()
    mock_screen.ndarray = np.zeros((144, 160, 3), dtype=np.uint8)
    mock.screen = mock_screen
    
    return mock


@pytest.fixture
def env(mock_pyboy):
    """Create a test environment instance."""
    with patch('core.pyboy_env.PyBoy', return_value=mock_pyboy):
        env = PyBoyPokemonCrystalEnv(
            rom_path="test.gbc",
            headless=True,
            debug_mode=True,
            enable_monitoring=False
        )
        yield env
        env.close()


def test_init(env):
    """Test environment initialization."""
    assert env.action_space.n == 9  # Including no-op
    assert env.observation_space.shape == (20,)
    assert env.max_steps == 10000
    assert env.headless is True
    assert env.debug_mode is True


def test_reset(env):
    """Test environment reset."""
    obs, info = env.reset()
    
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (20,)
    assert isinstance(info, dict)
    assert env.step_count == 0
    assert env.episode_reward == 0


def test_step(env):
    """Test environment step."""
    env.reset()
    action = 1  # Up
    obs, reward, terminated, truncated, info = env.step(action)
    
    assert isinstance(obs, np.ndarray)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)
    assert env.step_count == 1


def test_read_party_data(env):
    """Test reading party data."""
    env.reset()
    party = env._read_party_data()
    
    assert len(party) == 2
    
    # Check first Pokémon (Pikachu)
    assert party[0]['species'] == 25
    assert party[0]['level'] == 50
    assert party[0]['hp'] == 100
    assert party[0]['max_hp'] == 150
    
    # Check second Pokémon (Charizard)
    assert party[1]['species'] == 6
    assert party[1]['level'] == 45
    assert party[1]['hp'] == 200
    assert party[1]['max_hp'] == 250


def test_read_badges(env):
    """Test reading badge count."""
    env.reset()
    badges = env._read_badges()
    # 3 Johto badges (0b00000111) + 1 Kanto badge (0b00000001)
    assert badges == 4


def test_read_bcd_money(env):
    """Test reading BCD format money. In Pokémon Crystal (USA),
    money is stored in Ᵽ (Pokédollar) format using BCD encoding.
    
    Memory layout example for Ᵽ100:
    Byte 0: 0x00 (00)
    Byte 1: 0x01 (01)
    Byte 2: 0x00 (00)
    """
    env.reset()
    
    # Test case 1: Ᵽ100 (encoded as 00 01 00)
    mock_memory = {
        env.memory_addresses['money']: 0x00,      # First byte
        env.memory_addresses['money'] + 1: 0x01,  # Second byte
        env.memory_addresses['money'] + 2: 0x00   # Third byte
    }
    env.pyboy.memory.update(mock_memory)
    assert env._read_bcd_money() == 100
    
    # Test case 2: Ᵽ3000 (encoded as 03 00 00)
    mock_memory = {
        env.memory_addresses['money']: 0x03,      # First byte
        env.memory_addresses['money'] + 1: 0x00,  # Second byte
        env.memory_addresses['money'] + 2: 0x00   # Third byte
    }
    env.pyboy.memory.update(mock_memory)
    assert env._read_bcd_money() == 30000
    
    # Test case 3: Invalid BCD digits should return 0
    mock_memory = {
        env.memory_addresses['money']: 0xAB,      # Invalid BCD
        env.memory_addresses['money'] + 1: 0x00,  # Second byte
        env.memory_addresses['money'] + 2: 0x00   # Third byte
    }
    env.pyboy.memory.update(mock_memory)
    assert env._read_bcd_money() == 0


def test_render(env):
    """Test environment rendering."""
    env.reset()
    
    # Test rgb_array mode
    env.render_mode = "rgb_array"
    frame = env.render()
    assert isinstance(frame, np.ndarray)
    assert frame.shape == (144, 160, 3)
    
    # Test human mode
    env.render_mode = "human"
    frame = env.render()
    assert frame is None


def test_get_info(env):
    """Test getting environment info."""
    env.reset()
    info = env._get_info()
    
    assert 'step_count' in info
    assert 'episode_reward' in info
    assert 'raw_state' in info
    assert 'player_position' in info
    assert 'player_map' in info
    assert 'player_money' in info
    assert 'badges' in info
    assert 'party_size' in info


def test_is_terminated(env):
    """Test termination conditions."""
    env.reset()
    
    # Test with healthy party
    terminated = env._is_terminated()
    assert not terminated
    
    # Modify party HP to test termination condition
    env.current_state = {
        'party': [
            {'hp': 0, 'species': 25, 'level': 50},
            {'hp': 0, 'species': 6, 'level': 45}
        ]
    }
    terminated = env._is_terminated()
    assert terminated


def test_action_execution(env):
    """Test action execution."""
    env.reset()
    
    # Test each action
    for action in range(env.action_space.n):
        env._execute_action(action)
        if action == 0:  # no-op
            continue
        
        # Verify button press was called
        if action == 1:
            env.pyboy.button_press.assert_called_with("up")
        elif action == 2:
            env.pyboy.button_press.assert_called_with("down")
        elif action == 3:
            env.pyboy.button_press.assert_called_with("left")
        elif action == 4:
            env.pyboy.button_press.assert_called_with("right")
        elif action == 5:
            env.pyboy.button_press.assert_called_with("a")
        elif action == 6:
            env.pyboy.button_press.assert_called_with("b")
        elif action == 7:
            env.pyboy.button_press.assert_called_with("start")
        elif action == 8:
            env.pyboy.button_press.assert_called_with("select")


def test_stuck_detection(env):
    """Test stuck detection through screen hashing."""
    env.reset()
    
    # Same screen multiple times
    mock_screen = np.ones((144, 160, 3), dtype=np.uint8) * 128
    for _ in range(5):
        screen_hash = env._get_screen_hash(mock_screen)
        assert isinstance(screen_hash, int)
        
    # Different screen
    mock_screen_2 = np.ones((144, 160, 3), dtype=np.uint8) * 200
    screen_hash_2 = env._get_screen_hash(mock_screen_2)
    assert screen_hash_2 != screen_hash
