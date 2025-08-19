"""
Mock LLM manager for testing
"""

class MockLLMManager:
    """Mock LLM manager that always returns fixed actions."""
    
    def __init__(self, model=None, interval=10):
        self.model = model
        self.interval = interval
        self.call_count = 0
        self.memory_db = ":memory:"  # Use in-memory SQLite for testing

    def get_action(self):
        """Return a fixed action for testing."""
        self.call_count += 1
        return 5  # Always return A button
