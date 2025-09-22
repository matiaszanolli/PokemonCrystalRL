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
            from web_dashboard import create_web_server
            self.web_monitor = create_web_server(
                trainer=self.trainer,
                host=self.config.web_host,
                http_port=self.config.web_port,
                ws_port=self.config.web_port + 1
            )
            
            if self.web_monitor.start():
                url = f"http://{self.config.web_host}:{self.config.web_port}"
                self.logger.info(f"Web monitor started at {url}")
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
                # Unified web dashboard doesn't require explicit PyBoy updates
                # It gets PyBoy instance from the trainer automatically
                pass
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