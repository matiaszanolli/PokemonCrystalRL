#!/usr/bin/env python3
"""
Web Monitor Module - REFACTORED COMPATIBILITY WRAPPER

### REFACTORING NOTICE ###
This module has been refactored into a modular package structure for better
maintainability and separation of concerns. The functionality is now split across:

- core.web_monitor.screen_capture: Screen capture functionality
- core.web_monitor.http_handler: HTTP request handling  
- core.web_monitor.web_api: Clean web API layer
- core.web_monitor.monitor: Main WebMonitor orchestrator

This file now serves as a compatibility wrapper to maintain backward compatibility.
All functionality has been preserved while providing cleaner architecture.
"""

# Import the new modular components
from .web_monitor import WebMonitor, ScreenCapture, WebMonitorHandler

# Legacy compatibility - re-export everything that was previously available
__all__ = ['WebMonitor', 'ScreenCapture', 'WebMonitorHandler']