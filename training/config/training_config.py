"""
Training configuration classes and enums for Pokemon Crystal RL
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional


# Import centralized enums
from .training_modes import TrainingMode, LLMBackend


@dataclass
class TrainingConfig:
    """Configuration for Pokemon trainer."""
    rom_path: str = ""  # Make rom_path optional with empty string default
    mode: TrainingMode = TrainingMode.FAST_MONITORED
    llm_backend: Optional[LLMBackend] = LLMBackend.NONE  # Changed from SMOLLM2 to NONE
    max_actions: int = 10000  # Changed from 1000 to 10000
    max_episodes: int = 10
    llm_interval: int = 10
    frames_per_action: int = 24
    
    # PyBoy configuration
    headless: bool = True
    debug_mode: bool = False
    save_state_path: Optional[str] = None
    
    # Web monitoring
    enable_web: bool = True
    web_port: int = 8080
    web_host: str = "localhost"
    
    # Additional training parameters
    action_frequency: float = 1.0
    enable_render: bool = False
    sync_mode: bool = False
    screenshot_buffer_size: int = 100
    
    # Performance settings
    skip_frames: int = 0
    fast_mode: bool = False
    
    # Debugging and monitoring
    log_level: str = "INFO"
    enable_memory_monitoring: bool = False
    
    # Testing
    test_mode: bool = False
    mock_pyboy: bool = False
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.max_actions <= 0:
            raise ValueError("max_actions must be positive")
        if self.max_episodes <= 0:
            raise ValueError("max_episodes must be positive")
        if self.llm_interval <= 0:
            raise ValueError("llm_interval must be positive")
        if self.frames_per_action <= 0:
            raise ValueError("frames_per_action must be positive")