"""Vision-enhanced training session for Pokemon Crystal RL.

This module provides a training session class that uses computer vision to enhance
the training process with screen text recognition and image analysis.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

class VisionEnhancedTrainingSession:
    """Training session with enhanced vision capabilities."""
    
    def __init__(self,
                 rom_path: str,
                 save_state_path: Optional[str] = None,
                 model_name: str = "smollm2:1.7b",
                 max_steps_per_episode: int = 1000,
                 screenshot_interval: int = 5,
                 capture_screens: bool = True,
                 debug_mode: bool = False):
        """Initialize the vision-enhanced training session.
        
        Args:
            rom_path: Path to the Pokemon Crystal ROM
            save_state_path: Optional path to a save state
            model_name: Name of the LLM model to use
            max_steps_per_episode: Maximum steps per training episode
            screenshot_interval: How often to capture screenshots
            capture_screens: Whether to enable screen capture
            debug_mode: Enable debug output
        """
        self.rom_path = rom_path
        self.save_state_path = save_state_path
        self.model_name = model_name
        self.max_steps = max_steps_per_episode
        self.screenshot_interval = screenshot_interval
        self.capture_screens = capture_screens
        self.debug_mode = debug_mode
        
        # Session state
        self.step_count = 0
        self.episode_count = 0
        self.total_reward = 0.0
        self.is_training = False
        
        # Vision components will be initialized on demand
        self._font_decoder = None
        self._vision_processor = None
