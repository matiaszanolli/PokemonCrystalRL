"""
Training state enumeration.
"""

from enum import Enum

class TrainingState(Enum):
    """Training state enumeration."""
    
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    
    def __str__(self):
        return self.value
