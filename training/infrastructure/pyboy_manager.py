"""
PyBoy emulator management for Pokemon Crystal RL training.
"""

import os
import logging
from typing import Optional, Any
from ..config import TrainingConfig

# Defer resolving PyBoy until runtime so test patches work reliably
PyBoy = None  # Will be resolved dynamically
PYBOY_AVAILABLE = True


class PyBoyManager:
    """Manages PyBoy emulator instance and lifecycle."""
    
    def __init__(self, config: TrainingConfig, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.pyboy: Optional[Any] = None
        self._mock_pyboy_instance: Optional[Any] = None
    
    def _resolve_PyBoy(self):
        """Resolve the PyBoy class, honoring test patches on either trainer.trainer.PyBoy or pyboy.PyBoy."""
        # Prefer a patched symbol on this module if present
        global PyBoy
        if PyBoy is not None:
            return PyBoy
        try:
            from pyboy import PyBoy as PyBoyClass
            return PyBoyClass
        except Exception as e:
            raise
    
    def set_mock_instance(self, mock_instance: Any):
        """Set a mock PyBoy instance for testing."""
        self._mock_pyboy_instance = mock_instance
    
    def setup_pyboy(self) -> bool:
        """Setup PyBoy emulator.
        
        Returns:
            bool: True if setup successful, False otherwise
        """
        self.logger.debug("Setting up PyBoy...")
        try:
            self.logger.debug("Checking PyBoy prerequisites...")
            if not PYBOY_AVAILABLE:
                self.logger.debug("PyBoy not available, skipping setup")
                return False
            
            self.logger.debug(f"Validating config rom_path={self.config.rom_path}")
            if not self.config.rom_path:
                self.logger.debug("No ROM path specified")
                return False
                
            self.logger.debug(f"Creating PyBoy instance with parameters:")
            self.logger.debug(f"  - rom_path: {self.config.rom_path}")
            self.logger.debug(f"  - window: {'null' if self.config.headless else 'SDL2'}")
            self.logger.debug(f"  - debug: {self.config.debug_mode}")
            
            self.logger.debug("Attempting PyBoy instantiation...")
            
            # Special handling for mock objects in tests
            if self._mock_pyboy_instance is not None:
                self.pyboy = self._mock_pyboy_instance
                self.logger.debug("Using mock PyBoy instance")
            else:
                PyBoyClass = self._resolve_PyBoy()
                self.pyboy = PyBoyClass(
                    self.config.rom_path,
                    window="null" if self.config.headless else "SDL2",
                    debug=self.config.debug_mode
                )
            self.logger.debug("PyBoy instance created successfully")
            
            # Load save state if provided
            if self.config.save_state_path and os.path.exists(self.config.save_state_path):
                self.logger.debug(f"Loading save state from {self.config.save_state_path}")
                with open(self.config.save_state_path, 'rb') as f:
                    self.pyboy.load_state(f)
                self.logger.debug("Save state loaded successfully")
            else:
                self.logger.debug("No save state to load")
            
            return True

        except Exception as e:
            self.logger.error(f"PyBoy initialization failed with error: {str(e)}")
            self.logger.error(f"Failed to initialize PyBoy: {e}")
            raise
    
    def get_pyboy(self) -> Optional[Any]:
        """Get the PyBoy instance."""
        return self.pyboy
    
    def is_initialized(self) -> bool:
        """Check if PyBoy is initialized."""
        return self.pyboy is not None
    
    def cleanup(self):
        """Clean up PyBoy resources."""
        if self.pyboy is not None:
            try:
                if hasattr(self.pyboy, 'stop'):
                    self.pyboy.stop()
            except Exception as e:
                self.logger.warning(f"Error during PyBoy cleanup: {e}")
            finally:
                self.pyboy = None