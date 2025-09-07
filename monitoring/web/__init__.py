"""
Monitoring Web Interface Package

This package provides web interface functionality:
- HTTP server and request handling
- WebSocket communication
- Domain-specific APIs
- Services and managers
"""

from .http_handler import HttpHandler
from .api import TrainingAPI, SystemAPI, GameAPI
from .services.frame import FrameService
from .services.metrics import MetricsService
from .managers import StatusManager, EventManager

__all__ = [
    'HttpHandler',
    'TrainingAPI',
    'SystemAPI',
    'GameAPI',
    'FrameService',
    'MetricsService',
    'StatusManager',
    'EventManager'
]
