"""
Web server module for core.monitoring package.

This module provides access to web server components from the main monitoring package.
"""

from dataclasses import dataclass

@dataclass
class ServerConfig:
    """Web server configuration."""
    host: str = "localhost"
    port: int = 8080
    static_dir: str = "monitoring/static"
    web_host: str = "localhost"  # For compatibility
    web_port: int = 8080  # For compatibility

    @classmethod
    def from_training_config(cls, training_config):
        """Create ServerConfig from TrainingConfig"""
        return cls(
            host=getattr(training_config, 'web_host', 'localhost'),
            port=getattr(training_config, 'web_port', 8080),
            web_host=getattr(training_config, 'web_host', 'localhost'),
            web_port=getattr(training_config, 'web_port', 8080),
            static_dir=getattr(training_config, 'static_dir', 'monitoring/static')
        )

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

__all__ = ['WebServer', 'TrainingWebServer', 'TrainingHandler', 'ServerConfig']