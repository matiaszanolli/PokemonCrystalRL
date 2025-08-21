"""
Core Monitoring Package - Unified monitoring functionality for Pokemon Crystal RL

This package provides unified monitoring capabilities including:
- DataBus for component communication
- TrainingWebServer for web-based monitoring
- Bridge components for trainer integration
"""

from monitoring.data_bus import (
    DataBus, DataType, DataMessage, get_data_bus, init_data_bus, shutdown_data_bus, TrainingDataBus
)


from .web_server import (
    TrainingWebServer,
    MonitoringRequestHandler
)

try:
    from .bridge import *
except ImportError:
    pass  # Bridge module might not exist

__all__ = [
    # Data Bus
    'DataBus',
    'DataType', 
    'DataMessage',
    'get_data_bus',
    'init_data_bus',
    'shutdown_data_bus',
    'TrainingDataBus',  # Backward compatibility
    
    # Web Server
    'TrainingWebServer',
    'MonitoringRequestHandler',
]
