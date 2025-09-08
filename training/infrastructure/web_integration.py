"""
Web monitoring integration for Pokemon Crystal RL training.
"""

import logging
from typing import Optional, Any
from ..config import TrainingConfig


class WebIntegrationManager:
    """Manages web monitoring integration for training."""
    
    def __init__(self, config: TrainingConfig, trainer: Any, logger: Optional[logging.Logger] = None):
        self.config = config
        self.trainer = trainer
        self.logger = logger or logging.getLogger(__name__)
        self.web_monitor: Optional[Any] = None
        # Backward compatibility attributes for tests
        self.web_server: Optional[Any] = None
        self.web_thread: Optional[Any] = None
    
    def setup_web_server(self) -> bool:
        """Setup web monitoring server if enabled.
        
        Returns:
            bool: True if setup successful, False otherwise
        """
        # Ensure attribute exists for tests even when disabled
        self.web_monitor = None
        if not self.config.enable_web:
            return False
            
        try:
            from core.web_monitor import WebMonitor
            self.web_monitor = WebMonitor(
                trainer=self.trainer,
                port=self.config.web_port,
                host=self.config.web_host
            )
            
            if self.web_monitor.start():
                self.logger.info(f"Web monitor started at {self.web_monitor.get_url()}")
                # If PyBoy is already initialized, update the web monitor
                try:
                    pyboy_manager = getattr(self.trainer, 'pyboy_manager', None)
                    if pyboy_manager and pyboy_manager.is_initialized():
                        self.web_monitor.update_pyboy(pyboy_manager.get_pyboy())
                except Exception:
                    pass
                return True
            else:
                self.logger.error("Failed to start web monitor")
                self.web_monitor = None
                return False
        except Exception as e:
            self.logger.error(f"Failed to initialize web monitor: {e}")
            self.web_monitor = None
            return False
    
    def setup_legacy_web_server(self):
        """Legacy web server setup for backward compatibility.
        
        Note: Web server functionality has been consolidated into core.web_monitor.WebMonitor
        in the main llm_trainer.py file. This method remains for compatibility with tests.
        """
        if not self.config.enable_web:
            self.web_server = None
            self.web_thread = None
            return
        
        # For backward compatibility with tests, create mock objects
        self.web_server = None
        self.web_thread = None
        
        # Log that web functionality has moved
        self.logger.info("Web server functionality consolidated into core.web_monitor.WebMonitor")
    
    def get_web_monitor(self) -> Optional[Any]:
        """Get the web monitor instance."""
        return self.web_monitor
    
    def is_web_enabled(self) -> bool:
        """Check if web monitoring is enabled and running."""
        return self.web_monitor is not None
    
    def update_pyboy_reference(self, pyboy: Any):
        """Update PyBoy reference in web monitor."""
        if self.web_monitor is not None:
            try:
                self.web_monitor.update_pyboy(pyboy)
            except Exception as e:
                self.logger.warning(f"Failed to update PyBoy reference in web monitor: {e}")
    
    def cleanup(self):
        """Clean up web monitoring resources."""
        if self.web_monitor is not None:
            try:
                if hasattr(self.web_monitor, 'stop'):
                    self.web_monitor.stop()
            except Exception as e:
                self.logger.warning(f"Error during web monitor cleanup: {e}")
            finally:
                self.web_monitor = None