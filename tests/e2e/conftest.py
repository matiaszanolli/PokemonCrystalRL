"""
E2E Test Fixtures

Provides fixtures for end-to-end testing including:
- ROM and save state verification
- Isolated training environments
- Resource cleanup
- Performance monitoring
"""

import pytest
import os
import tempfile
import gc
import psutil
from pathlib import Path
from unittest.mock import patch

# Test data paths
TEST_ROM_PATH = "roms/pokemon_crystal.gbc"
TEST_SAVE_STATE = "roms/pokemon_crystal.gbc.state"


@pytest.fixture
def verify_test_rom():
    """Verify test ROM exists or skip test."""
    if not os.path.exists(TEST_ROM_PATH):
        pytest.skip(f"Test ROM not available at {TEST_ROM_PATH}")
    return TEST_ROM_PATH


@pytest.fixture
def verify_save_state():
    """Verify save state exists or skip test."""
    if not os.path.exists(TEST_SAVE_STATE):
        pytest.skip(f"Save state not available at {TEST_SAVE_STATE}")
    return TEST_SAVE_STATE


@pytest.fixture
def isolated_training_env(tmp_path):
    """Create isolated environment for training tests."""
    # Create isolated directories
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    db_dir = tmp_path / "databases"
    db_dir.mkdir()

    log_dir = tmp_path / "logs"
    log_dir.mkdir()

    return {
        'output_dir': str(output_dir),
        'db_dir': str(db_dir),
        'log_dir': str(log_dir),
        'tmp_path': tmp_path
    }


@pytest.fixture
def memory_monitor():
    """Monitor memory usage during test."""
    gc.collect()
    process = psutil.Process()

    initial_memory = process.memory_info().rss / 1024 / 1024  # MB

    monitor = {
        'initial_mb': initial_memory,
        'process': process
    }

    yield monitor

    # Final measurement
    gc.collect()
    final_memory = process.memory_info().rss / 1024 / 1024  # MB
    monitor['final_mb'] = final_memory
    monitor['growth_mb'] = final_memory - initial_memory


@pytest.fixture(autouse=True)
def cleanup_event_bus():
    """Ensure event bus is cleaned between E2E tests."""
    from core.event_system import get_event_bus

    event_bus = get_event_bus()
    event_bus.subscribers.clear()

    yield

    event_bus.subscribers.clear()


@pytest.fixture
def mock_ollama_for_testing():
    """Mock Ollama for tests that don't need real LLM."""
    with patch('training.components.llm_manager.requests.post') as mock_post:
        # Mock successful LLM response
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {
            'response': 'up'  # Simple action
        }
        yield mock_post
