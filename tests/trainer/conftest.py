"""Common test fixtures and mocks."""

import pytest
from unittest.mock import Mock, patch
from pokemon_crystal_rl.trainer.trainer import TrainingConfig, TrainingMode, LLMBackend, UnifiedPokemonTrainer

@pytest.fixture
def base_config():
    """Create a base training config for tests."""
    return TrainingConfig(
        rom_path="test.gbc",
        mode=TrainingMode.FAST_MONITORED,
        llm_backend=LLMBackend.SMOLLM2,
        headless=True,
        debug_mode=True
    )

@pytest.fixture
def trainer_fast_monitored(base_config):
    """Create a trainer fixture with fast monitored mode."""
    with patch('pokemon_crystal_rl.trainer.trainer.PyBoy') as mock_pyboy_class:
        mock_pyboy_instance = Mock()
        mock_pyboy_class.return_value = mock_pyboy_instance
        trainer = UnifiedPokemonTrainer(base_config)
        return trainer

@pytest.fixture
def trainer_ultra_fast(base_config):
    """Create a trainer fixture with ultra fast mode."""
    config = base_config
    config.mode = TrainingMode.ULTRA_FAST
    config.max_actions = 10
    config.capture_screens = False

    with patch('pokemon_crystal_rl.trainer.trainer.PyBoy') as mock_pyboy_class:
        mock_pyboy_instance = Mock()
        mock_pyboy_class.return_value = mock_pyboy_instance
        trainer = UnifiedPokemonTrainer(config)
        return trainer

@pytest.fixture(autouse=True)
def mock_llm_manager(monkeypatch):
    """Mock LocalLLMPokemonAgent to avoid import errors."""
    mock_agent = Mock()
    mock_agent.return_value = Mock()
    monkeypatch.setattr('pokemon_crystal_rl.llm.local_llm_agent.LLMManager', mock_agent)
