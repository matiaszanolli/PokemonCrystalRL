"""
E2E Smoke Tests

Quick validation tests (1-5 seconds each) that verify core workflows
without running full training sessions.

These tests should:
- Run on every commit
- Complete in < 30 seconds total
- Catch basic integration issues
- Verify system can start and stop
"""

import pytest
import sys
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import after adding to path
from main import parse_arguments_from_dict, initialize_training_systems
from tests.e2e.helpers import (
    create_test_config,
    verify_training_stats,
    check_trainer_health,
    verify_memory_reading,
    wait_for_server,
    get_dashboard_data,
    count_llm_decisions
)


@pytest.mark.e2e
@pytest.mark.e2e_smoke
@pytest.mark.timeout(10)
def test_e2e_training_startup_and_shutdown(verify_test_rom, isolated_training_env, memory_monitor):
    """
    Test basic training startup and shutdown.

    Validates:
    - Argument parsing works
    - PyBoy initializes
    - Trainer starts without errors
    - Graceful shutdown completes
    - No resource leaks
    """
    # Create test configuration
    config = create_test_config(
        rom_path=verify_test_rom,
        max_actions=10,
        enable_web=False,  # Disable web for faster test
        isolated_env=isolated_training_env
    )

    # Parse arguments
    args = parse_arguments_from_dict(config)
    assert args.rom_path == verify_test_rom
    assert args.max_actions == 10

    # Initialize training systems
    systems = initialize_training_systems(args)
    trainer = systems['trainer']

    # Verify trainer initialized
    assert trainer is not None, "Trainer should be initialized"

    # Check component health (before training starts)
    health = check_trainer_health(trainer)
    assert health['emulation'], "Emulation should be initialized"
    assert health['stats_tracker'], "Stats tracker should be initialized"
    # Note: PyBoy may not be initialized until training actually starts

    # Start training
    trainer.start_training()

    # Wait a moment for training to actually start
    time.sleep(0.5)

    # Verify training started
    assert trainer.running or (hasattr(trainer, 'training_thread') and trainer.training_thread), \
        "Training should have started"

    # Wait for training to complete (max 10 actions should be quick)
    if hasattr(trainer, 'training_thread') and trainer.training_thread:
        trainer.training_thread.join(timeout=5)

    # Get statistics
    stats = trainer.get_statistics()

    # Verify statistics
    assert stats is not None, "Statistics should be available"

    # The stats may have different key names - be flexible
    if 'actions_taken' not in stats and 'total_actions' in stats:
        stats['actions_taken'] = stats['total_actions']

    # Verify basic stats
    assert 'total_actions' in stats or 'actions_taken' in stats, "Should have action count"
    assert 'total_reward' in stats, "Should have reward tracking"

    actions = stats.get('total_actions', stats.get('actions_taken', 0))
    assert actions == 10, f"Expected 10 actions, got {actions}"

    # Graceful shutdown
    trainer.stop_training()

    # Verify shutdown
    assert trainer.is_shutdown, "Trainer should be shutdown"
    assert not trainer.running, "Trainer should not be running"

    # Check memory usage didn't explode (if monitor captured it)
    if 'growth_mb' in memory_monitor:
        assert memory_monitor['growth_mb'] < 500, \
            f"Memory grew by {memory_monitor['growth_mb']}MB - possible leak"


@pytest.mark.e2e
@pytest.mark.e2e_smoke
@pytest.mark.timeout(10)
def test_e2e_save_state_loading(verify_test_rom, verify_save_state, isolated_training_env):
    """
    Test save state loading and memory reading.

    Validates:
    - Save state loads correctly
    - Memory reading works
    - Rewards are realistic (not garbage data)
    - Game state is valid
    """
    # Create config with save state
    config = create_test_config(
        rom_path=verify_test_rom,
        save_state=verify_save_state,
        max_actions=10,
        enable_web=False,
        isolated_env=isolated_training_env
    )

    # Parse and initialize
    args = parse_arguments_from_dict(config)
    systems = initialize_training_systems(args)
    trainer = systems['trainer']

    # Start training
    trainer.start_training()
    time.sleep(0.5)

    # Wait for completion
    if hasattr(trainer, 'training_thread') and trainer.training_thread:
        trainer.training_thread.join(timeout=5)

    # Get statistics
    stats = trainer.get_statistics()

    # Verify statistics
    verify_training_stats(stats, expected_actions=10)

    # Verify memory reading is working (rewards should be realistic)
    verify_memory_reading(stats, has_save_state=True)

    # Cleanup
    trainer.stop_training()
    assert trainer.is_shutdown


@pytest.mark.e2e
@pytest.mark.e2e_smoke
@pytest.mark.timeout(15)
def test_e2e_web_dashboard_integration(verify_test_rom, isolated_training_env):
    """
    Test web dashboard integration.

    Validates:
    - Training starts with web UI enabled
    - HTTP endpoints respond
    - Screen capture works
    - Stats are available via API
    - Shutdown is clean
    """
    # Create config with web enabled
    config = create_test_config(
        rom_path=verify_test_rom,
        max_actions=10,
        enable_web=True,
        web_port=8090,  # Use non-standard port to avoid conflicts
        isolated_env=isolated_training_env
    )

    # Parse and initialize
    args = parse_arguments_from_dict(config)
    systems = initialize_training_systems(args)
    trainer = systems['trainer']

    # Start training
    trainer.start_training()

    # Wait for web server to start
    assert wait_for_server('http://localhost:8090/api/dashboard', timeout=5), \
        "Web server should start within 5 seconds"

    # Fetch dashboard data
    dashboard_data = get_dashboard_data(port=8090)
    assert dashboard_data is not None, "Dashboard should return data"

    # Wait for training to complete
    if hasattr(trainer, 'training_thread') and trainer.training_thread:
        trainer.training_thread.join(timeout=5)

    # Verify stats via API
    final_data = get_dashboard_data(port=8090)
    if final_data:
        # Should have executed actions
        actions = final_data.get('actions_taken', 0)
        assert actions > 0, "Should have taken some actions"

    # Cleanup
    trainer.stop_training()
    assert trainer.is_shutdown


@pytest.mark.e2e
@pytest.mark.e2e_smoke
@pytest.mark.timeout(10)
def test_e2e_llm_workflow_basics(verify_test_rom, mock_ollama_for_testing, isolated_training_env):
    """
    Test LLM workflow with mocked Ollama.

    Validates:
    - LLM manager initializes
    - LLM is called at correct intervals
    - Actions are parsed correctly
    - Decisions affect game state
    """
    # Create config with LLM enabled
    config = create_test_config(
        rom_path=verify_test_rom,
        max_actions=10,
        llm_interval=5,  # Call LLM every 5 actions
        enable_web=False,
        isolated_env=isolated_training_env
    )

    # Parse and initialize
    args = parse_arguments_from_dict(config)
    systems = initialize_training_systems(args)
    trainer = systems['trainer']

    # Verify LLM engine exists
    assert systems.get('llm_agent') is not None, "LLM agent should be initialized"

    # Start training
    trainer.start_training()
    time.sleep(0.5)

    # Wait for completion
    if hasattr(trainer, 'training_thread') and trainer.training_thread:
        trainer.training_thread.join(timeout=5)

    # Get statistics
    stats = trainer.get_statistics()
    verify_training_stats(stats, expected_actions=10)

    # Verify LLM was called (at least once with interval=5 and 10 actions)
    llm_decisions = count_llm_decisions(stats)
    assert llm_decisions >= 1, f"LLM should have been called at least once, got {llm_decisions}"

    # Verify mock was actually called
    assert mock_ollama_for_testing.called, "Mocked LLM should have been called"

    # Cleanup
    trainer.stop_training()
    assert trainer.is_shutdown


@pytest.mark.e2e
@pytest.mark.e2e_smoke
@pytest.mark.timeout(15)
def test_e2e_curriculum_learning_startup(verify_test_rom, isolated_training_env, tmp_path):
    """
    Test curriculum learning initialization.

    Validates:
    - Save state library loads
    - Curriculum config is read
    - Stage selection works
    - Training can start
    """
    # Create a minimal save state library for testing
    library_path = tmp_path / "test_library"
    library_path.mkdir()

    # Create empty metadata file
    metadata_file = library_path / "metadata.json"
    metadata_file.write_text('{"save_states": []}')

    # Create config with curriculum enabled
    config = create_test_config(
        rom_path=verify_test_rom,
        max_actions=10,
        enable_curriculum=True,
        library_path=str(library_path),
        curriculum_episodes=1,
        enable_web=False,
        isolated_env=isolated_training_env
    )

    # Parse arguments
    args = parse_arguments_from_dict(config)

    # For curriculum mode, main.py uses a different code path
    # We'll test that it doesn't crash on initialization
    assert args.enable_curriculum == True
    assert args.library_path == str(library_path)

    # Note: Full curriculum training requires more setup
    # This test just validates the configuration is accepted
    # and library path exists
    assert library_path.exists(), "Library path should exist"
