"""Web monitor compatibility layer.

This module provides backward compatibility for the old web monitoring system
while using the new modular architecture internally.
"""

import os
import threading
from dataclasses import dataclass
from typing import Dict, Any, Optional

from monitoring.web.server import MonitoringServer, WebServerConfig
from monitoring.web.services.frame import FrameService
from monitoring.web.services.metrics import MetricsService
from monitoring.web.api import TrainingAPI, SystemAPI, GameAPI


@dataclass
class MonitorConfig:
    """Backwards compatible monitor config."""
    host: str = "localhost"
    port: int = 8080
    web_port: Optional[int] = None  # Legacy compat
    static_dir: str = "static"
    debug: bool = False


class WebMonitorCompat:
    """Compatibility wrapper for old WebMonitor interface."""
    
    def __init__(self, trainer=None, port=8080, host="localhost"):
        """Initialize web monitor with new architecture.
        
        Args:
            trainer: The Pokemon trainer instance
            port: Port for web server
            host: Host for web server
        """
        self.trainer = trainer
        self.port = port
        self.host = host
        
        # Create base config
        self.config = MonitorConfig(
            host=host,
            port=port,
            static_dir=os.path.join(os.path.dirname(__file__), "static")
        )
        
        # Initialize server with services
        server_config = WebServerConfig(
            host=host,
            port=port,
            template_dir="templates",
            static_dir=self.config.static_dir
        )
        self.server = MonitoringServer(server_config)
        
        # Create APIs
        self.training_api = TrainingAPI(trainer)
        self.system_api = SystemAPI(trainer)
        self.game_api = GameAPI(trainer)
        
        self.running = False
        self._server_thread = None
    
    def start(self) -> bool:
        """Start the web monitor.
        
        Returns:
            bool: True if started successfully
        """
        if self.running:
            return True
        
        try:
            # Start server
            success = self.server.start()
            if not success:
                return False
            
            self.running = True
            
            return True
            
        except Exception as e:
            print(f"Failed to start web monitor: {e}")
            return False
    
    def stop(self) -> bool:
        """Stop the web monitor.
        
        Returns:
            bool: True if stopped successfully
        """
        if not self.running:
            return True
        
        try:
            success = self.server.stop()
            if success:
                self.running = False
            return success
            
        except Exception as e:
            print(f"Failed to stop web monitor: {e}")
            return False
    
    def update_pyboy(self, pyboy):
        """Update PyBoy instance."""
        self.trainer.pyboy = pyboy
        self.system_api.update_trainer(self.trainer)
        self.game_api.update_trainer(self.trainer)
        self.training_api.update_trainer(self.trainer)
    
    def update_screenshot(self, screenshot_data):
        """Update screenshot data."""
        pass  # Screenshot updates now handled by FrameService
    
    def update_stats(self, stats: Dict[str, Any]):
        """Update monitoring statistics."""
        # Convert old stats format to new metrics
        if stats:
            self.server.metrics_service.update_training_metrics({
                'total_actions': stats.get('actions_taken', 0),
                'total_reward': stats.get('total_reward', 0.0),
                'episode': stats.get('current_episode', 0),
                'badges_total': stats.get('badges_total', 0),
                'player_level': stats.get('player_level', 0)
            })
    
    def update_game_state(self, state: Dict[str, Any]):
        """Update game state information."""
        if state:
            self.server.metrics_service.update_game_metrics(state)
    
    def get_url(self) -> str:
        """Get web monitor URL."""
        return f"http://{self.host}:{self.port}"
