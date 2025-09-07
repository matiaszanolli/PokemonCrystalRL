"""Compatibility layer for web monitor.

This module provides backward compatibility for the old WebAPI interface
while using the new domain-specific APIs internally.
"""

from typing import Dict, Any, Optional
from monitoring.web.api import TrainingAPI, SystemAPI, GameAPI


class WebAPICompat:
    """Compatibility wrapper for old WebAPI interface."""
    
    def __init__(self, trainer=None, screen_capture=None):
        """Initialize compatibility APIs.
        
        Args:
            trainer: The Pokemon trainer instance
            screen_capture: The screen capture component
        """
        self.training_api = TrainingAPI(trainer)
        self.system_api = SystemAPI(trainer, screen_capture)
        self.game_api = GameAPI(trainer, screen_capture)
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics."""
        return self.training_api.get_training_stats()
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status information."""
        return self.system_api.get_system_status()
    
    def get_llm_decisions(self) -> Dict[str, Any]:
        """Get LLM decisions with enhanced information."""
        return self.training_api.get_llm_decisions()
    
    def get_memory_debug(self) -> Dict[str, Any]:
        """Get memory debug information."""
        return self.game_api.get_memory_debug()
    
    def get_screen_bytes(self) -> Optional[bytes]:
        """Get current screen as PNG bytes."""
        return self.game_api.get_screen_bytes()
    
    def get_screen_data(self) -> Optional[Dict[str, Any]]:
        """Get current screen metadata."""
        return self.game_api.get_screen_data()
    
    def update_trainer(self, trainer) -> None:
        """Update trainer reference."""
        self.training_api.update_trainer(trainer)
        self.system_api.update_trainer(trainer)
        self.game_api.update_trainer(trainer)
    
    def update_screen_capture(self, screen_capture) -> None:
        """Update screen capture reference."""
        self.system_api.update_screen_capture(screen_capture)
        self.game_api.update_screen_capture(screen_capture)
