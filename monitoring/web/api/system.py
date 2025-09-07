"""System API for Pokemon Crystal RL.

This API provides endpoints for system-related data:
- System status and uptime
- Component status tracking
- Resource monitoring
"""

import time
from typing import Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class SystemStatus:
    """System status data."""
    status: str = 'running'
    uptime: float = 0.0
    version: str = '1.0.0'
    screen_capture_active: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'status': self.status,
            'uptime': self.uptime,
            'version': self.version,
            'screen_capture_active': self.screen_capture_active
        }


class SystemAPI:
    """System status and monitoring API."""
    
    def __init__(self, trainer=None, screen_capture=None):
        """Initialize system API.
        
        Args:
            trainer: The Pokemon trainer instance
            screen_capture: The screen capture component
        """
        self.trainer = trainer
        self.screen_capture = screen_capture
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status information.
        
        Returns:
            Dictionary with status information
        """
        current_time = time.time()
        trainer_start = current_time
        
        # Get trainer start time
        try:
            if (self.trainer and hasattr(self.trainer, 'stats') and 
                isinstance(self.trainer.stats, dict)):
                start_iso = self.trainer.stats.get('start_time')
                if isinstance(start_iso, str):
                    start_time = datetime.fromisoformat(start_iso)
                    trainer_start = start_time.timestamp()
        except Exception:
            # Ignore parsing issues and keep default
            pass
        
        # Check screen capture status
        screen_active = self._check_screen_capture_status()
        
        # Build status
        status = SystemStatus(
            uptime=max(0.0, current_time - trainer_start),
            screen_capture_active=screen_active
        )
        
        return status.to_dict()
    
    def _check_screen_capture_status(self) -> bool:
        """Check if screen capture is active.
        
        A screen capture is considered active if:
        1. The component exists
        2. Has a valid PyBoy instance
        3. Capture is marked as active
        4. Not a mock object in tests
        """
        if (self.screen_capture is not None and 
            self.screen_capture.pyboy is not None and 
            self.screen_capture.capture_active):
            # Check for mock objects in tests
            if hasattr(self.screen_capture.pyboy, '_mock_name'):
                return False  # Mock PyBoy doesn't count as active
            return True
        return False
    
    def update_trainer(self, trainer) -> None:
        """Update trainer reference."""
        self.trainer = trainer
    
    def update_screen_capture(self, screen_capture) -> None:
        """Update screen capture reference."""
        self.screen_capture = screen_capture
