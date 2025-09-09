"""
Unified Monitor - Compatibility Wrapper

This module provides backward compatibility for the UnifiedMonitor interface
while using the modern modular MonitoringServer architecture.
"""

from .web.server import MonitoringServer, WebServerConfig
from .base import MonitoringConfig

# Backward compatibility alias
UnifiedMonitor = MonitoringServer

# Backward compatibility configuration
class TrainingState:
    """Training state enum for backward compatibility."""
    IDLE = "idle"
    RUNNING = "running" 
    PAUSED = "paused"
    STOPPED = "stopped"

def create_unified_monitor(training_session=None, host='127.0.0.1', port=5000, config=None):
    """Factory function to create UnifiedMonitor with backward compatibility."""
    if config is None:
        config = WebServerConfig(
            host=host,
            port=port,
            debug=False
        )
    elif isinstance(config, MonitoringConfig):
        # Convert MonitorConfig to WebServerConfig
        config = WebServerConfig(
            host=config.host,
            port=config.port,
            debug=config.debug,
            enable_api=True,
            enable_websocket=True,
            enable_metrics=True,
            template_dir="templates",
            static_dir=config.static_dir,
            api_prefix="/api/v1",
            frame_buffer_size=1,
            frame_quality=85,
            update_interval=config.update_interval,
            metrics_interval=1.0
        )
    
    return MonitoringServer(config)

# For tests and legacy code
PSUTIL_AVAILABLE = True

__all__ = ['UnifiedMonitor', 'TrainingState', 'create_unified_monitor', 'PSUTIL_AVAILABLE']