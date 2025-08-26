"""
Web server module for core.monitoring package.

This module provides access to web server components from the main monitoring package.
"""

try:
    from monitoring.web_server import WebServer, TrainingWebServer, TrainingHandler
except ImportError:
    # Fallback mock classes
    class MockWebServer:
        def __init__(self, *args, **kwargs):
            pass
        def start(self):
            pass
        def shutdown(self):
            pass
    
    WebServer = MockWebServer
    TrainingWebServer = MockWebServer
    TrainingHandler = MockWebServer

__all__ = ['WebServer', 'TrainingWebServer', 'TrainingHandler']