"""Game API for Pokemon Crystal RL.

This API provides endpoints for game-related data:
- Game state and memory access
- Screen capture and frames
- Game metrics and status
"""

import time
from typing import Dict, Any, Optional
from dataclasses import dataclass

# Import memory reader for game state debugging
try:
    from trainer.memory_reader import PokemonCrystalMemoryReader
except ImportError:
    PokemonCrystalMemoryReader = None


class GameAPI:
    """Game state and control API."""
    
    def __init__(self, trainer=None, screen_capture=None):
        """Initialize game API.
        
        Args:
            trainer: The Pokemon trainer instance
            screen_capture: The screen capture component
        """
        self.trainer = trainer
        self.screen_capture = screen_capture
    
    def get_memory_debug(self) -> Dict[str, Any]:
        """Get memory debug information.
        
        Returns:
            Dictionary with memory state data
        """
        # Check memory reader availability
        if PokemonCrystalMemoryReader is None:
            return self._make_error_response(
                'Memory reader not available - import failed'
            )
        
        # Initialize memory reader if needed
        if not hasattr(self.trainer, 'memory_reader') or self.trainer.memory_reader is None:
            # Try unified trainer structure first
            pyboy_instance = None
            if hasattr(self.trainer, 'emulation_manager') and self.trainer.emulation_manager:
                pyboy_instance = self.trainer.emulation_manager.get_instance()
            elif hasattr(self.trainer, 'pyboy') and self.trainer.pyboy is not None:
                pyboy_instance = self.trainer.pyboy

            if pyboy_instance is not None:
                self.trainer.memory_reader = PokemonCrystalMemoryReader(pyboy_instance)
            else:
                return self._make_error_response('PyBoy instance not available')
        
        # Get memory state
        if hasattr(self.trainer, 'memory_reader') and self.trainer.memory_reader is not None:
            try:
                memory_state = self.trainer.memory_reader.read_game_state()
                # Add debug info
                memory_state['debug_info'] = self.trainer.memory_reader.get_debug_info()
                return memory_state
            except Exception as e:
                return self._make_error_response(f'Failed to read memory: {e}')
        
        return self._make_error_response('Memory reader initialization failed')
    
    def get_screen_bytes(self) -> Optional[bytes]:
        """Get current screen as PNG bytes.
        
        Returns:
            Screen image data as bytes or None if not available
        """
        if self.screen_capture:
            return self.screen_capture.get_latest_screen_bytes()
        return None
    
    def get_screen_data(self) -> Optional[Dict[str, Any]]:
        """Get current screen metadata.
        
        Returns:
            Screen metadata dictionary or None if not available
        """
        if self.screen_capture:
            return self.screen_capture.get_latest_screen_data()
        return None
    
    def _make_error_response(self, message: str) -> Dict[str, Any]:
        """Create error response dictionary.
        
        Args:
            message: Error message
            
        Returns:
            Error response with timestamp
        """
        return {
            'error': message,
            'timestamp': time.time()
        }
    
    def update_trainer(self, trainer) -> None:
        """Update trainer reference."""
        self.trainer = trainer
    
    def update_screen_capture(self, screen_capture) -> None:
        """Update screen capture reference."""
        self.screen_capture = screen_capture
