"""Configure pytest for the Pokemon Crystal RL test suite."""

import pytest


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
