"""
Unified HTTP Request Handler for Web Dashboard

This module provides the main HTTP request handler that delegates to specialized handlers.
"""

import http.server
import json
import logging
import time
from urllib.parse import urlparse, parse_qs
from typing import Optional

from .dashboard_handler import DashboardHandler
from .static_handler import StaticFileHandler

logger = logging.getLogger(__name__)


class UnifiedHttpHandler(http.server.BaseHTTPRequestHandler):
    """HTTP request handler for the unified web dashboard.

    This handler delegates to specialized handlers for different types of requests:
    - DashboardHandler for dashboard HTML pages
    - StaticFileHandler for static assets (CSS, JS, images)
    - Direct API handling for JSON endpoints
    """

    def __init__(self, *args, api_endpoints=None, rest_api=None, ab_testing_api=None,
                 automation_api=None, advanced_api=None, websocket_handler=None, **kwargs):
        """Initialize HTTP handler with API endpoints and handlers.

        Args:
            api_endpoints: Unified API endpoints instance
            rest_api: REST API endpoints instance
            ab_testing_api: A/B testing API instance
            automation_api: Automation API instance
            advanced_api: Advanced analytics/debugging API instance
            websocket_handler: WebSocket handler for screen capture
        """
        self.api_endpoints = api_endpoints
        self.rest_api = rest_api
        self.ab_testing_api = ab_testing_api
        self.automation_api = automation_api
        self.advanced_api = advanced_api
        self.websocket_handler = websocket_handler

        # Initialize specialized handlers
        self.dashboard_handler = DashboardHandler()
        self.static_handler = StaticFileHandler()

        super().__init__(*args, **kwargs)

    def do_GET(self):
        """Handle GET requests."""
        try:
            parsed_path = urlparse(self.path)
            path = parsed_path.path

            # Dashboard HTML pages
            if path == '/' or path == '/dashboard':
                self.dashboard_handler.serve_main_dashboard(self)
            elif path == '/hybrid' or path == '/hybrid-dashboard':
                self.dashboard_handler.serve_hybrid_dashboard(self)
            elif path == '/advanced' or path == '/advanced-dashboard':
                self.dashboard_handler.serve_advanced_dashboard(self)

            # Screen capture
            elif path == '/api/screen':
                self.dashboard_handler.serve_screen_capture(self, self.websocket_handler)

            # API endpoints (JSON)
            elif path == '/api/dashboard':
                self._serve_api_response(self.api_endpoints.get_dashboard_data())
            elif path == '/api/game_state':
                self._serve_api_response(self.api_endpoints.get_game_state())
            elif path == '/api/training_stats':
                self._serve_api_response(self.api_endpoints.get_training_stats())
            elif path == '/api/memory_debug':
                self._serve_api_response(self.api_endpoints.get_memory_debug())
            elif path == '/api/llm_decisions':
                self._serve_api_response(self.api_endpoints.get_llm_decisions())
            elif path == '/api/system_status':
                self._serve_api_response(self.api_endpoints.get_system_status())
            elif path == '/api/visualization_data':
                self._serve_api_response(self.api_endpoints.get_visualization_data())

            # REST API v1 endpoints (GET)
            elif path.startswith('/api/v1/'):
                self._handle_rest_api_request(path, {})

            # Static assets
            elif path.startswith('/static/'):
                self.static_handler.serve_static_file(self, path)

            # Health check
            elif path == '/health':
                self._serve_health()

            # Favicon - return empty response to prevent errors
            elif path == '/favicon.ico':
                self.send_response(204)  # No Content
                self.end_headers()

            else:
                self.send_error(404, "Endpoint not found")

        except BrokenPipeError:
            # Client disconnected, don't log as error
            logger.debug("Client disconnected (broken pipe)")
            return
        except Exception as e:
            logger.error(f"GET request error: {e}")
            try:
                self.send_error(500, f"Internal server error: {str(e)}")
            except BrokenPipeError:
                logger.debug("Client disconnected while sending error response")
                return

    def do_POST(self):
        """Handle POST requests for REST API."""
        try:
            parsed_path = urlparse(self.path)
            path = parsed_path.path

            # Get request body
            content_length = int(self.headers.get('Content-Length', 0))
            if content_length > 0:
                body_data = self.rfile.read(content_length).decode('utf-8')
                try:
                    request_data = json.loads(body_data) if body_data else {}
                except json.JSONDecodeError:
                    request_data = {}
            else:
                request_data = {}

            # REST API v1 endpoints
            if path.startswith('/api/v1/'):
                self._handle_rest_api_request(path, request_data)
            else:
                self.send_error(404, "REST API endpoint not found")

        except BrokenPipeError:
            logger.debug("Client disconnected during POST request")
            return
        except Exception as e:
            logger.error(f"POST request error: {e}")
            try:
                self.send_error(500, f"Internal server error: {str(e)}")
            except BrokenPipeError:
                logger.debug("Client disconnected while sending error response")
                return

    def do_OPTIONS(self):
        """Handle CORS preflight requests."""
        self.send_response(200)
        self._set_cors_headers()
        self.end_headers()

    def _serve_api_response(self, response_data: dict):
        """Serve a JSON API response.

        Args:
            response_data: Dictionary to serialize as JSON
        """
        try:
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self._set_cors_headers()
            self.end_headers()

            json_response = json.dumps(response_data, indent=2)
            self.wfile.write(json_response.encode('utf-8'))

        except Exception as e:
            logger.error(f"API response error: {e}")
            self.send_error(500, f"API error: {str(e)}")

    def _serve_health(self):
        """Serve health check endpoint."""
        health_data = {
            "status": "healthy",
            "timestamp": time.time(),
            "api_available": self.api_endpoints is not None,
            "websocket_available": self.websocket_handler is not None
        }

        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self._set_cors_headers()
        self.end_headers()
        self.wfile.write(json.dumps(health_data).encode('utf-8'))

    def _handle_rest_api_request(self, path: str, request_data: dict):
        """Handle REST API v1 requests - imports the original routing logic.

        Args:
            path: Request path
            request_data: Request body data

        Note:
            This method imports the routing logic from the original server.
            In a future refactoring, this could be further modularized.
        """
        # Import the API router module which contains the routing logic
        from ..api_router_logic import handle_rest_api_routing
        handle_rest_api_routing(self, path, request_data)

    def _set_cors_headers(self):
        """Set CORS headers for cross-origin requests."""
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')

    def log_message(self, format, *args):
        """Override to use Python logging instead of stderr."""
        logger.debug(format % args)
