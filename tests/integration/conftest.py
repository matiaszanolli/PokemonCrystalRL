"""Configure pytest fixtures for integration tests."""

import pytest
import tempfile
import os
import sqlite3
from pathlib import Path


@pytest.fixture
def temp_db():
    """Create a temporary database for testing semantic and dialogue systems."""
    # Create a temporary database file
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        temp_db_path = f.name

    # Initialize the schema
    with sqlite3.connect(temp_db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE dialogue_sessions (
                session_start TEXT NOT NULL,
                npc_type TEXT,
                location_map INTEGER,
                total_exchanges INTEGER DEFAULT 0,
                choices_made INTEGER DEFAULT 0,
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER UNIQUE
            )
        """)
        cursor.execute("""
            CREATE TABLE dialogue_choices (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER,
                choice_text TEXT,
                timestamp FLOAT,
                FOREIGN KEY (session_id) REFERENCES dialogue_sessions (session_id)
            )
        """)
        cursor.execute("""
            CREATE TABLE npc_interactions (
                npc_type TEXT PRIMARY KEY,
                interaction_count INTEGER,
                last_interaction FLOAT
            )
        """)
        conn.commit()

    yield temp_db_path

    # Clean up
    try:
        os.unlink(temp_db_path)
    except OSError:
        pass  # File might already be deleted

# Create mock classes for performance tests
class MockPyBoy:
    """Mock PyBoy class for testing without actual emulator"""
    def __init__(self):
        self.frame_count = 0
        self.tick_value = 0
        self.metadata = {"version": "test"}

    def tick(self):
        """Update game state"""
        self.tick_value += 1
        return True

    def get_memory_value(self, addr):
        """Mock memory read"""
        return 0

    def send_input(self, button):
        """Mock input"""
        pass

    def save_state(self, file=None):
        """Mock state save"""
        pass

    def load_state(self, file=None):
        """Mock state load"""
        pass

    def get_tile(self, *args):
        """Mock tile accessor"""
        return 0

    @property
    def screen_image(self):
        """Mock screen"""
        import numpy as np
        return np.zeros((144, 160, 3), dtype=np.uint8)

@pytest.fixture
def mock_pyboy_class():
    """Fixture providing a mock PyBoy implementation."""
    return MockPyBoy


@pytest.fixture
def temp_choice_db():
    """Create a temporary database for testing choice recognition system."""
    # Create a temporary database file
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        temp_db_path = f.name

    # Initialize the schema
    with sqlite3.connect(temp_db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE choice_recognitions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                dialogue_text TEXT,
                choice_text TEXT,
                choice_type TEXT,
                confidence REAL,
                priority INTEGER,
                outcome TEXT
            )
        """)
        cursor.execute("""
            CREATE TABLE choice_actions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                recognition_id INTEGER,
                action TEXT NOT NULL,
                success BOOLEAN,
                FOREIGN KEY (recognition_id) REFERENCES choice_recognitions (id)
            )
        """)
        conn.commit()

    yield temp_db_path

    # Clean up
    try:
        os.unlink(temp_db_path)
    except OSError:
        pass  # File might already be deleted
