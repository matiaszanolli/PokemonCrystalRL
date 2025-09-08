"""
Emulation Manager - PyBoy emulation setup and management

Extracted from LLMTrainer to handle PyBoy instance lifecycle,
ROM loading, save states, and emulation control.
"""

import os
import time
import logging
import threading
from typing import Optional, Any
from dataclasses import dataclass

try:
    from pyboy import PyBoy
except ImportError:
    print("⚠️  PyBoy not available")
    PyBoy = None


@dataclass
class EmulationConfig:
    """Configuration for PyBoy emulation setup."""
    rom_path: str
    save_state_path: Optional[str] = None
    headless: bool = True
    debug_mode: bool = False
    speed_factor: int = 0  # 0 = unlimited
    sound_enabled: bool = False
    window_type: str = "null"  # "null", "SDL2", or "OpenGL"


class EmulationManager:
    """Manages PyBoy emulation instance and lifecycle."""
    
    def __init__(self, config: EmulationConfig):
        self.config = config
        self.logger = logging.getLogger("EmulationManager")
        self.pyboy: Optional[PyBoy] = None
        self._lock = threading.RLock()
        
        # Validation
        if not self.config.rom_path or not os.path.exists(self.config.rom_path):
            raise ValueError(f"ROM file not found: {self.config.rom_path}")
    
    def initialize(self) -> bool:
        """Initialize PyBoy emulation instance.
        
        Returns:
            bool: True if initialization successful
        """
        with self._lock:
            if self.pyboy is not None:
                self.logger.warning("PyBoy already initialized")
                return True
            
            if PyBoy is None:
                self.logger.error("PyBoy not available - install with: pip install pyboy")
                return False
            
            try:
                self.logger.info(f"Initializing PyBoy with ROM: {self.config.rom_path}")
                
                # Create PyBoy instance
                self.pyboy = PyBoy(
                    self.config.rom_path,
                    window=self.config.window_type if not self.config.headless else "null",
                    debug=self.config.debug_mode,
                    sound=self.config.sound_enabled
                )
                
                # Set speed if configured
                if self.config.speed_factor > 0:
                    self.pyboy.set_emulation_speed(self.config.speed_factor)
                else:
                    self.pyboy.set_emulation_speed(0)  # Unlimited
                
                # Load save state if provided
                if self.config.save_state_path and os.path.exists(self.config.save_state_path):
                    with open(self.config.save_state_path, 'rb') as f:
                        self.pyboy.load_state(f)
                    self.logger.info(f"Loaded save state: {self.config.save_state_path}")
                
                self.logger.info("PyBoy initialization successful")
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to initialize PyBoy: {e}")
                self.pyboy = None
                return False
    
    def is_alive(self) -> bool:
        """Check if PyBoy instance is alive and functioning.
        
        Returns:
            bool: True if PyBoy is active
        """
        with self._lock:
            if self.pyboy is None:
                return False
            
            try:
                # Test basic functionality
                frame_count = self.pyboy.frame_count
                return isinstance(frame_count, int) and frame_count >= 0
            except Exception:
                return False
    
    def get_instance(self) -> Optional[PyBoy]:
        """Get the PyBoy instance.
        
        Returns:
            Optional[PyBoy]: PyBoy instance or None if not initialized
        """
        return self.pyboy
    
    def execute_action(self, action: int, frames: int = 1) -> bool:
        """Execute an action on the emulation.
        
        Args:
            action: Action code (0-8)
            frames: Number of frames to execute action
            
        Returns:
            bool: True if action executed successfully
        """
        with self._lock:
            if not self.is_alive():
                return False
            
            try:
                for _ in range(frames):
                    self.pyboy.send_input(action)
                    self.pyboy.tick()
                return True
            except Exception as e:
                self.logger.error(f"Error executing action {action}: {e}")
                return False
    
    def get_screen_array(self) -> Optional[Any]:
        """Get current screen as numpy array.
        
        Returns:
            Optional[np.ndarray]: Screen data or None if unavailable
        """
        with self._lock:
            if not self.is_alive():
                return None
            
            try:
                return self.pyboy.screen.ndarray.copy()
            except Exception as e:
                self.logger.warning(f"Failed to get screen array: {e}")
                return None
    
    def get_screen_image(self) -> Optional[Any]:
        """Get current screen as PIL Image.
        
        Returns:
            Optional[PIL.Image]: Screen image or None if unavailable
        """
        with self._lock:
            if not self.is_alive():
                return None
            
            try:
                return self.pyboy.screen_image()
            except Exception as e:
                self.logger.warning(f"Failed to get screen image: {e}")
                return None
    
    def get_frame_count(self) -> int:
        """Get current frame count.
        
        Returns:
            int: Frame count or 0 if unavailable
        """
        with self._lock:
            if not self.is_alive():
                return 0
            
            try:
                return self.pyboy.frame_count
            except Exception:
                return 0
    
    def save_state(self, file_path: str) -> bool:
        """Save current emulation state.
        
        Args:
            file_path: Path to save state file
            
        Returns:
            bool: True if save successful
        """
        with self._lock:
            if not self.is_alive():
                return False
            
            try:
                with open(file_path, 'wb') as f:
                    self.pyboy.save_state(f)
                self.logger.info(f"Saved state to: {file_path}")
                return True
            except Exception as e:
                self.logger.error(f"Failed to save state: {e}")
                return False
    
    def load_state(self, file_path: str) -> bool:
        """Load emulation state from file.
        
        Args:
            file_path: Path to state file
            
        Returns:
            bool: True if load successful
        """
        with self._lock:
            if not self.is_alive() or not os.path.exists(file_path):
                return False
            
            try:
                with open(file_path, 'rb') as f:
                    self.pyboy.load_state(f)
                self.logger.info(f"Loaded state from: {file_path}")
                return True
            except Exception as e:
                self.logger.error(f"Failed to load state: {e}")
                return False
    
    def shutdown(self) -> None:
        """Shutdown PyBoy instance and cleanup resources."""
        with self._lock:
            if self.pyboy is not None:
                try:
                    self.pyboy.stop()
                    self.logger.info("PyBoy shutdown complete")
                except Exception as e:
                    self.logger.error(f"Error during PyBoy shutdown: {e}")
                finally:
                    self.pyboy = None
    
    def __enter__(self):
        """Context manager entry."""
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()