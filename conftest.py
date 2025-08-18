"""Test configuration file for pytest."""

import os
import sys
import pytest

# Add the project root directory to PYTHONPATH
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
