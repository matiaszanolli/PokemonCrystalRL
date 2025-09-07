"""
Training Modes Configuration

This module defines the different training modes available for the Pokemon Crystal RL system.
"""

from enum import Enum

class TrainingMode(Enum):
    """Available training modes for the Pokemon Crystal RL system."""
    
    LLM_ONLY = "llm_only"
    RL_ONLY = "rl_only" 
    HYBRID = "hybrid"
    EVALUATION = "evaluation"
    DEBUG = "debug"

__all__ = ['TrainingMode']