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
    
    return MonitoringServer(config)

# For tests and legacy code
PSUTIL_AVAILABLE = True

__all__ = ['UnifiedMonitor', 'TrainingState', 'create_unified_monitor', 'PSUTIL_AVAILABLE']