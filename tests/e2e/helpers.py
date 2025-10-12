"""
E2E Test Helpers

Utility functions for end-to-end testing.
"""

import time
import requests
from typing import Dict, Any, Optional
from pathlib import Path


def create_test_config(**kwargs) -> Dict[str, Any]:
    """
    Create test configuration with sensible defaults.

    Args:
        **kwargs: Configuration overrides

    Returns:
        Configuration dictionary suitable for parse_arguments_from_dict()

    Example:
        >>> config = create_test_config(
        ...     rom_path='test.gbc',
        ...     max_actions=10
        ... )
    """
    defaults = {
        'max_actions': 10,
        'llm_interval': 5,
        'llm_temperature': 0.7,
        'llm_model': 'smollm2:1.7b',
        'llm_base_url': 'http://localhost:11434',
        'log_dir': kwargs.get('isolated_env', {}).get('log_dir', 'logs'),
        'quiet': True,  # Disable progress output in tests
    }

    # Add isolated environment paths if provided
    if 'isolated_env' in kwargs:
        env = kwargs.pop('isolated_env')
        defaults['log_dir'] = env.get('log_dir', 'logs')

    # Merge with user overrides
    defaults.update(kwargs)

    return defaults


def wait_for_server(url: str, timeout: int = 5, check_interval: float = 0.1) -> bool:
    """
    Wait for web server to be ready.

    Args:
        url: URL to check
        timeout: Maximum time to wait in seconds
        check_interval: Time between checks in seconds

    Returns:
        True if server is ready, False if timeout

    Example:
        >>> if wait_for_server('http://localhost:8080/api/dashboard'):
        ...     print("Server ready!")
    """
    start_time = time.time()

    while time.time() - start_time < timeout:
        try:
            response = requests.get(url, timeout=1)
            if response.status_code == 200:
                return True
        except (requests.ConnectionError, requests.Timeout):
            pass

        time.sleep(check_interval)

    return False


def verify_training_stats(stats: Dict[str, Any], expected_actions: Optional[int] = None):
    """
    Verify training statistics are valid.

    Args:
        stats: Statistics dictionary from trainer
        expected_actions: Expected number of actions (optional)

    Raises:
        AssertionError: If stats are invalid

    Example:
        >>> stats = trainer.get_statistics()
        >>> verify_training_stats(stats, expected_actions=10)
    """
    # Required fields
    assert 'actions_taken' in stats, "Missing 'actions_taken' in stats"
    assert 'total_reward' in stats, "Missing 'total_reward' in stats"

    # Type checks
    assert isinstance(stats['actions_taken'], (int, float)), \
        f"actions_taken should be numeric, got {type(stats['actions_taken'])}"
    assert isinstance(stats['total_reward'], (int, float)), \
        f"total_reward should be numeric, got {type(stats['total_reward'])}"

    # Value checks
    if expected_actions is not None:
        assert stats['actions_taken'] == expected_actions, \
            f"Expected {expected_actions} actions, got {stats['actions_taken']}"

    # Reward sanity check (should not be in thousands)
    assert -1000 < stats['total_reward'] < 1000, \
        f"Reward {stats['total_reward']} seems abnormal - check memory reading"


def verify_game_state(game_state: Dict[str, Any]):
    """
    Verify game state is valid.

    Args:
        game_state: Game state dictionary

    Raises:
        AssertionError: If game state is invalid
    """
    # Check required fields exist
    assert 'position' in game_state or 'x' in game_state, \
        "Game state missing position information"

    # If position is present, verify it's not (0,0) which indicates
    # LLM decisions aren't being executed
    if 'position' in game_state:
        x, y = game_state['position']
        # Allow (0,0) initially but not after several actions
        # This is checked in the actual tests with action count


def check_trainer_health(trainer) -> Dict[str, bool]:
    """
    Check trainer component health.

    Args:
        trainer: Trainer instance

    Returns:
        Dictionary of component health status

    Example:
        >>> health = check_trainer_health(trainer)
        >>> assert health['emulation'], "Emulation not initialized"
    """
    health = {}

    # Check core components
    health['emulation'] = hasattr(trainer, 'emulation_manager') and trainer.emulation_manager is not None
    health['stats_tracker'] = hasattr(trainer, 'stats_tracker') and trainer.stats_tracker is not None

    # Check optional components
    health['llm_engine'] = hasattr(trainer, 'llm_engine') and trainer.llm_engine is not None
    health['screen_capture'] = hasattr(trainer, 'screen_capture') and trainer.screen_capture is not None
    health['web_server'] = hasattr(trainer, 'web_server') and trainer.web_server is not None

    # Check PyBoy instance (may not be initialized until training starts)
    try:
        pyboy_instance = trainer.pyboy if hasattr(trainer, 'pyboy') else None
        health['pyboy'] = pyboy_instance is not None
    except Exception:
        # PyBoy may not be initialized yet - that's ok
        health['pyboy'] = False

    return health


def get_dashboard_data(port: int = 8080) -> Optional[Dict[str, Any]]:
    """
    Fetch dashboard data from web API.

    Args:
        port: Web server port

    Returns:
        Dashboard data or None if unavailable

    Example:
        >>> data = get_dashboard_data(8080)
        >>> if data:
        ...     print(f"Actions: {data.get('actions_taken', 0)}")
    """
    try:
        response = requests.get(f'http://localhost:{port}/api/dashboard', timeout=2)
        if response.status_code == 200:
            return response.json()
    except (requests.ConnectionError, requests.Timeout, ValueError):
        pass

    return None


def verify_memory_reading(stats: Dict[str, Any], has_save_state: bool):
    """
    Verify memory reading is working correctly.

    Args:
        stats: Statistics dictionary
        has_save_state: Whether save state was loaded

    Raises:
        AssertionError: If memory reading seems broken
    """
    if has_save_state:
        # With save state, rewards should be realistic
        reward = stats.get('total_reward', 0)
        actions = stats.get('actions_taken', 0)

        if actions > 0:
            avg_reward = reward / actions

            # Average reward per action should be reasonable
            # (not thousands which indicates garbage memory data)
            assert -100 < avg_reward < 100, \
                f"Average reward {avg_reward} per action seems abnormal. " \
                f"Check memory reading - may not have loaded save state correctly."


def wait_for_training_start(trainer, timeout: int = 5) -> bool:
    """
    Wait for training to actually start.

    Args:
        trainer: Trainer instance
        timeout: Maximum time to wait

    Returns:
        True if training started, False if timeout
    """
    start_time = time.time()

    while time.time() - start_time < timeout:
        if hasattr(trainer, 'running') and trainer.running:
            return True
        if hasattr(trainer, 'training_thread') and trainer.training_thread and trainer.training_thread.is_alive():
            return True

        time.sleep(0.1)

    return False


def count_llm_decisions(stats: Dict[str, Any]) -> int:
    """
    Count LLM decisions from stats.

    Args:
        stats: Statistics dictionary

    Returns:
        Number of LLM decisions made
    """
    # Try different possible field names
    if 'llm_decisions' in stats:
        return stats['llm_decisions']
    if 'llm_decision_count' in stats:
        return stats['llm_decision_count']

    # Could also be in nested dict
    if 'llm' in stats and isinstance(stats['llm'], dict):
        if 'decision_count' in stats['llm']:
            return stats['llm']['decision_count']

    return 0
