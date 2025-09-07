"""Configure pytest for the Pokemon Crystal RL test suite."""

import pytest
from training.trainer import TrainingMode, TrainingConfig

@pytest.fixture
def test_config() -> TrainingConfig:
    """Creates a test-optimized TrainingConfig instance.
    
    This config:
    - Disables web server
    - Uses FAST mode for data bus tests
    - Sets a test ROM path
    """
    return TrainingConfig(
        mode=TrainingMode.FAST_MONITORED,
        max_episodes=10,  # Short tests
        max_actions=100,  # Short tests
        test_mode=True  # For test isolation
    )


def pytest_configure(config):
    """Register custom pytest markers."""
    custom_markers = [
        "dialogue",  # Dialogue system tests
        "semantic",  # Semantic context system tests
        "database",  # Database operation tests
        "choice",   # Choice processing tests
        "benchmarking",  # Performance benchmarking tests
        "adaptive_intervals",  # LLM interval tests
        "enhanced_prompting",  # Enhanced LLM prompting tests
        "temperature",  # LLM temperature tuning tests
        "slow",  # Slow running tests
    ]
    
    for marker in custom_markers:
        config.addinivalue_line(
            "markers",
            f"{marker}: mark test as {marker} test"
        )
