"""
Configuration classes and enums for Pokemon Crystal RL Trainer
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional


class TrainingMode(Enum):
    """Available training modes"""
    FAST_MONITORED = "fast_monitored"   # Fast training with comprehensive monitoring
    CURRICULUM = "curriculum"           # Progressive skill-based training (legacy)
    ULTRA_FAST = "ultra_fast"          # Rule-based maximum speed (legacy)
    CUSTOM = "custom"                  # User-defined configuration


class LLMBackend(Enum):
    """Available LLM backends"""
    SMOLLM2 = "smollm2:1.7b"          # Ultra-fast, optimized
    LLAMA32_1B = "llama3.2:1b"        # Fastest Llama
    LLAMA32_3B = "llama3.2:3b"        # Balanced speed/quality
    QWEN25_3B = "qwen2.5:3b"          # Alternative fast option
    NONE = None                        # Rule-based only


@dataclass
class TrainingConfig:
    """Unified training configuration"""
    # Core settings
    rom_path: str
    mode: TrainingMode = TrainingMode.FAST_MONITORED
    llm_backend: LLMBackend = LLMBackend.SMOLLM2
    
    # Training parameters
    max_actions: int = 1000
    max_episodes: int = 10
    llm_interval: int = 10             # Actions between LLM calls
    
    # Performance settings
    headless: bool = True
    debug_mode: bool = False
    save_state_path: Optional[str] = None
    
    # Frame timing (Game Boy runs at 60 FPS)
    frames_per_action: int = 24         # Standard RL timing: 24 frames = 400ms = 2.5 actions/sec
    
    # Web interface
    enable_web: bool = False
    web_port: int = 8080
    web_host: str = "localhost"
    
    # Screen capture
    capture_screens: bool = True
    capture_fps: int = 5               # Reduced FPS for stability
    screen_resize: tuple = (240, 216)  # Full resolution for CV/OCR, scaled in UI
    
    # Curriculum settings (for curriculum mode)
    curriculum_stages: int = 5
    stage_mastery_threshold: float = 0.7
    min_stage_episodes: int = 5
    max_stage_episodes: int = 20
    
    # Output settings
    save_stats: bool = True
    stats_file: str = "training_stats.json"
    log_level: str = "INFO"            # DEBUG, INFO, WARNING, ERROR


# Action mapping constants
GAME_ACTIONS = {
    1: "PRESS_ARROW_UP",
    2: "PRESS_ARROW_DOWN", 
    3: "PRESS_ARROW_LEFT",
    4: "PRESS_ARROW_RIGHT",
    5: "PRESS_BUTTON_A",
    6: "PRESS_BUTTON_B",
    7: "PRESS_BUTTON_START",
    8: "PRESS_BUTTON_SELECT",
    0: None
}

# Human readable action names
ACTION_NAMES = {
    1: 'UP', 
    2: 'DOWN', 
    3: 'LEFT', 
    4: 'RIGHT', 
    5: 'A', 
    6: 'B', 
    7: 'START', 
    8: 'SELECT'
}

# State-specific temperature settings for LLM
STATE_TEMPERATURES = {
    "dialogue": 0.8,
    "menu": 0.6,
    "battle": 0.8,
    "overworld": 0.7,
    "title_screen": 0.5,
    "intro_sequence": 0.4,
    "unknown": 0.6
}

# State-specific guidance for LLM
STATE_GUIDANCE = {
    "title_screen": "Press 7=START to begin, then 5=A to select menu options",
    "intro_sequence": "Press 5=A rapidly to skip text, try 7=START to skip faster", 
    "new_game_menu": "Use 1=UP/2=DOWN to navigate, 5=A to select, 6=B to go back",
    "dialogue": "Press 5=A to advance text, wait between presses",
    "overworld": "Move with 1=UP/2=DOWN/3=LEFT/4=RIGHT, 5=A to interact with objects/NPCs",
    "menu": "Use 1=UP/2=DOWN to navigate options, 5=A to select, 6=B to exit",
    "loading": "Press 5=A or 7=START if screen seems stuck",
    "unknown": "Try 5=A to interact, or movement keys 1/2/3/4, use 6=B to exit menus"
}
