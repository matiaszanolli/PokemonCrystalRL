"""
Web Dashboard HTTP Request Handlers

This package contains modular request handlers for the web dashboard:
- http_handler.py - Main HTTP request routing
- api_router.py - REST API endpoint routing
- static_handler.py - Static file serving
- dashboard_handler.py - Dashboard HTML serving
"""

from .http_handler import UnifiedHttpHandler
from .api_router import ApiRouter
from .static_handler import StaticFileHandler
from .dashboard_handler import DashboardHandler

__all__ = [
    'UnifiedHttpHandler',
    'ApiRouter',
    'StaticFileHandler',
    'DashboardHandler',
]
