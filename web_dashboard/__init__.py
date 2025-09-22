"""
Unified Web Dashboard for Pokemon Crystal RL Training.

This package provides a complete, consolidated web dashboard solution that
replaces multiple fragmented implementations with a single, well-documented,
and maintainable system.

Key Features:
- Unified API with proper data models
- Real-time WebSocket updates
- Modern, responsive UI
- Comprehensive error handling
- Performance monitoring
- Memory debugging capabilities
- LLM decision tracking

Usage:
    from web_dashboard import UnifiedWebServer

    server = UnifiedWebServer(trainer=your_trainer)
    server.start()

    # Dashboard available at http://localhost:8080
    # WebSocket streaming at ws://localhost:8081

Architecture:
- api/: API endpoints and data models
- server.py: HTTP server implementation
- websocket_handler.py: Real-time updates
- static/: Frontend assets (HTML, CSS, JS)
"""

from .server import UnifiedWebServer, UnifiedHttpHandler
from .websocket_handler import WebSocketHandler
from .api import (
    UnifiedApiEndpoints,
    GameStateModel,
    TrainingStatsModel,
    MemoryDebugModel,
    LLMDecisionModel,
    SystemStatusModel,
    UnifiedDashboardModel,
    ApiResponseModel
)

__version__ = "1.0.0"
__author__ = "Pokemon Crystal RL Team"

__all__ = [
    'UnifiedWebServer',
    'UnifiedHttpHandler',
    'WebSocketHandler',
    'UnifiedApiEndpoints',
    'GameStateModel',
    'TrainingStatsModel',
    'MemoryDebugModel',
    'LLMDecisionModel',
    'SystemStatusModel',
    'UnifiedDashboardModel',
    'ApiResponseModel'
]


def create_web_server(trainer, host='localhost', http_port=8080, ws_port=8081):
    """
    Factory function to create a unified web server instance.

    Args:
        trainer: The Pokemon trainer instance
        host: Server host (default: localhost)
        http_port: HTTP server port (default: 8080)
        ws_port: WebSocket server port (default: 8081)

    Returns:
        UnifiedWebServer: Configured web server instance
    """
    return UnifiedWebServer(
        trainer=trainer,
        host=host,
        http_port=http_port,
        ws_port=ws_port
    )