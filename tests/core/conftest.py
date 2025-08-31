"""
Configuration file for core tests.
"""

import pytest
import warnings


@pytest.fixture(autouse=True)
def suppress_sdl2_warnings():
    """Suppress SDL2 warnings during tests."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="sdl2")
        warnings.filterwarnings("ignore", message="Using SDL2 binaries")
        yield


def pytest_configure(config):
    """Configure pytest to suppress certain warnings globally."""
    warnings.filterwarnings("ignore", category=UserWarning, module="sdl2")
    warnings.filterwarnings("ignore", message="Using SDL2 binaries")
