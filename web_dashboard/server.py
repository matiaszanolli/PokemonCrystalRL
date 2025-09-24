"""
Unified Web Server for Pokemon Crystal RL Dashboard.

This server consolidates all web dashboard functionality into a single,
clean implementation with proper API endpoints and WebSocket support.
"""

import http.server
import socketserver
import json
import logging
import threading
import time
from urllib.parse import urlparse, parse_qs
from typing import Optional, Any
import websockets
import asyncio
import base64
from io import BytesIO

from .api import UnifiedApiEndpoints
from .websocket_handler import WebSocketHandler

logger = logging.getLogger(__name__)


class UnifiedHttpHandler(http.server.BaseHTTPRequestHandler):
    """HTTP request handler for the unified web dashboard."""

    def __init__(self, *args, api_endpoints=None, websocket_handler=None, **kwargs):
        self.api_endpoints = api_endpoints
        self.websocket_handler = websocket_handler
        super().__init__(*args, **kwargs)

    def do_GET(self):
        """Handle GET requests."""
        try:
            parsed_path = urlparse(self.path)
            path = parsed_path.path

            # Serve dashboard HTML
            if path == '/' or path == '/dashboard':
                self._serve_dashboard()

            # API endpoints
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

            # Screen capture
            elif path == '/api/screen':
                self._serve_screen()

            # Static assets
            elif path.startswith('/static/'):
                self._serve_static_file(path)

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
                # Client disconnected while sending error response
                logger.debug("Client disconnected while sending error response")
                return

    def do_OPTIONS(self):
        """Handle CORS preflight requests."""
        self.send_response(200)
        self._set_cors_headers()
        self.end_headers()

    def _serve_dashboard(self):
        """Serve the main dashboard HTML."""
        try:
            dashboard_path = "/mnt/data/src/pokemon_crystal_rl/web_dashboard/static/dashboard.html"
            with open(dashboard_path, 'r', encoding='utf-8') as f:
                content = f.read()

            self.send_response(200)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self._set_cors_headers()
            self.end_headers()
            self.wfile.write(content.encode('utf-8'))

        except FileNotFoundError:
            self.send_error(404, "Dashboard template not found")
        except Exception as e:
            logger.error(f"Dashboard serve error: {e}")
            self.send_error(500, f"Failed to serve dashboard: {str(e)}")

    def _serve_api_response(self, response_data: dict):
        """Serve a JSON API response."""
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

    def _serve_screen(self):
        """Serve current game screen capture."""
        try:
            if self.websocket_handler:
                # Update screen data first
                if hasattr(self.websocket_handler, 'update_screen_for_http'):
                    self.websocket_handler.update_screen_for_http()

                # Get the latest screen data
                if hasattr(self.websocket_handler, 'get_latest_screen'):
                    screen_data = self.websocket_handler.get_latest_screen()
                    if screen_data:
                        self.send_response(200)
                        self.send_header('Content-type', 'image/png')
                        self._set_cors_headers()
                        self.end_headers()
                        self.wfile.write(screen_data)
                        return

            # Fallback: return placeholder image
            self._serve_placeholder_image()

        except Exception as e:
            logger.error(f"Screen serve error: {e}")
            self._serve_placeholder_image()

    def _serve_placeholder_image(self):
        """Serve a placeholder image when screen capture is unavailable."""
        placeholder_data = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
        )
        self.send_response(200)
        self.send_header('Content-type', 'image/png')
        self._set_cors_headers()
        self.end_headers()
        self.wfile.write(placeholder_data)

    def _serve_static_file(self, path: str):
        """Serve static files (CSS, JS)."""
        try:
            # Remove /static/ prefix and get file extension
            file_path = path[8:]  # Remove '/static/'
            base_path = "/mnt/data/src/pokemon_crystal_rl/web_dashboard/static"
            full_path = f"{base_path}/{file_path}"

            with open(full_path, 'rb') as f:
                content = f.read()

            # Determine content type
            if file_path.endswith('.css'):
                content_type = 'text/css'
            elif file_path.endswith('.js'):
                content_type = 'application/javascript'
            elif file_path.endswith('.png'):
                content_type = 'image/png'
            elif file_path.endswith('.jpg') or file_path.endswith('.jpeg'):
                content_type = 'image/jpeg'
            else:
                content_type = 'application/octet-stream'

            self.send_response(200)
            self.send_header('Content-type', content_type)
            self._set_cors_headers()
            self.end_headers()
            self.wfile.write(content)

        except FileNotFoundError:
            self.send_error(404, "Static file not found")
        except Exception as e:
            logger.error(f"Static file serve error: {e}")
            self.send_error(500, f"Failed to serve static file: {str(e)}")

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

        json_response = json.dumps(health_data)
        self.wfile.write(json_response.encode('utf-8'))

    def _set_cors_headers(self):
        """Set CORS headers for browser compatibility."""
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')

    def log_message(self, format, *args):
        """Override to use our logger instead of stderr."""
        logger.info(f"{self.address_string()} - {format % args}")


class UnifiedWebServer:
    """
    Unified web server for Pokemon Crystal RL dashboard.

    Combines HTTP API endpoints with WebSocket screen streaming.
    """

    def __init__(self, trainer=None, host='localhost', http_port=8080, ws_port=8081):
        """Initialize the unified web server."""
        self.trainer = trainer
        self.host = host
        self.http_port = http_port
        self.ws_port = ws_port

        # Initialize API endpoints
        self.api_endpoints = UnifiedApiEndpoints(trainer)

        # Initialize WebSocket handler
        self.websocket_handler = WebSocketHandler(trainer)

        # Server instances
        self.http_server = None
        self.websocket_server = None

        # Control flags
        self.running = False
        self.threads = []

        logger.info(f"Unified web server initialized - HTTP: {host}:{http_port}, WS: {host}:{ws_port}")

    def start(self):
        """Start both HTTP and WebSocket servers."""
        if self.running:
            logger.warning("Web server already running")
            return True

        self.running = True

        try:
            # Start HTTP server
            self._start_http_server()

            # Start WebSocket server
            self._start_websocket_server()

            logger.info(f"🚀 Unified web server started successfully")
            logger.info(f"📊 Dashboard: http://{self.host}:{self.http_port}")
            logger.info(f"📡 WebSocket: ws://{self.host}:{self.ws_port}")

            return True

        except Exception as e:
            logger.error(f"Failed to start web server: {e}")
            self.stop()
            return False

    def stop(self):
        """Stop both HTTP and WebSocket servers."""
        if not self.running:
            return

        self.running = False

        try:
            # Stop HTTP server
            if self.http_server:
                self.http_server.shutdown()
                self.http_server.server_close()
                logger.info("HTTP server stopped")

            # Stop WebSocket server
            if self.websocket_server:
                self.websocket_server.close()
                logger.info("WebSocket server stopped")

            # Wait for threads to finish
            for thread in self.threads:
                thread.join(timeout=1.0)

            logger.info("✅ Unified web server stopped")

        except Exception as e:
            logger.error(f"Error stopping web server: {e}")

    def _start_http_server(self):
        """Start the HTTP server in a separate thread."""
        def run_http_server():
            handler = lambda *args, **kwargs: UnifiedHttpHandler(
                *args,
                api_endpoints=self.api_endpoints,
                websocket_handler=self.websocket_handler,
                **kwargs
            )

            self.http_server = socketserver.TCPServer((self.host, self.http_port), handler)
            logger.info(f"HTTP server listening on {self.host}:{self.http_port}")
            self.http_server.serve_forever()

        thread = threading.Thread(target=run_http_server, daemon=True)
        thread.start()
        self.threads.append(thread)

    def _start_websocket_server(self):
        """Start the WebSocket server in a separate thread."""
        def run_websocket_server():
            # Create new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            async def websocket_main():
                # Create wrapper to properly handle method call
                async def websocket_wrapper(websocket):
                    path = websocket.path if hasattr(websocket, 'path') else '/'
                    await self.websocket_handler.handle_connection(websocket, path)

                self.websocket_server = await websockets.serve(
                    websocket_wrapper,
                    self.host,
                    self.ws_port
                )
                logger.info(f"WebSocket server listening on {self.host}:{self.ws_port}")

                # Keep server running
                await self.websocket_server.wait_closed()

            loop.run_until_complete(websocket_main())

        thread = threading.Thread(target=run_websocket_server, daemon=True)
        thread.start()
        self.threads.append(thread)

    def is_running(self) -> bool:
        """Check if the web server is running."""
        return self.running