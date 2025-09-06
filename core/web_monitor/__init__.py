"""
Web Monitor Package

Refactored web monitoring system split into focused modules:
- screen_capture: Screen capture functionality
- http_handler: HTTP request handling  
- web_api: Clean web API layer
- monitor: Main WebMonitor orchestrator

This package maintains backward compatibility while providing 
better separation of concerns.
"""

from .screen_capture import ScreenCapture
from .http_handler import WebMonitorHandler  
from .web_api import WebAPI
from .monitor import WebMonitor

# Maintain backward compatibility
__all__ = ['WebMonitor', 'WebMonitorHandler', 'ScreenCapture', 'WebAPI']