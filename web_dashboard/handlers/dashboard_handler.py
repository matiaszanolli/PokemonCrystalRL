"""
Dashboard HTML Handler for Web Dashboard

Handles serving the different dashboard HTML pages.
"""

import logging
import os
import base64

logger = logging.getLogger(__name__)


class DashboardHandler:
    """Handler for serving dashboard HTML pages."""

    def __init__(self, static_dir: str = "/mnt/data/src/pokemon_crystal_rl/web_dashboard/static"):
        """Initialize dashboard handler.

        Args:
            static_dir: Base directory for dashboard HTML files
        """
        self.static_dir = static_dir

    def serve_main_dashboard(self, request_handler):
        """Serve the main dashboard HTML.

        Args:
            request_handler: HTTP request handler instance
        """
        self._serve_dashboard_file(request_handler, "dashboard.html", "Dashboard")

    def serve_hybrid_dashboard(self, request_handler):
        """Serve the hybrid LLM-RL training dashboard HTML.

        Args:
            request_handler: HTTP request handler instance
        """
        self._serve_dashboard_file(request_handler, "hybrid_dashboard.html", "Hybrid dashboard")

    def serve_advanced_dashboard(self, request_handler):
        """Serve the advanced analytics & debugging dashboard HTML.

        Args:
            request_handler: HTTP request handler instance
        """
        self._serve_dashboard_file(request_handler, "advanced_dashboard.html", "Advanced dashboard")

    def serve_screen_capture(self, request_handler, websocket_handler=None):
        """Serve current game screen capture.

        Args:
            request_handler: HTTP request handler instance
            websocket_handler: WebSocket handler for screen data
        """
        try:
            if websocket_handler:
                # Update screen data first
                if hasattr(websocket_handler, 'update_screen_for_http'):
                    websocket_handler.update_screen_for_http()

                # Get the latest screen data
                if hasattr(websocket_handler, 'get_latest_screen'):
                    screen_data = websocket_handler.get_latest_screen()
                    if screen_data:
                        request_handler.send_response(200)
                        request_handler.send_header('Content-type', 'image/png')
                        request_handler.send_header('Access-Control-Allow-Origin', '*')
                        request_handler.end_headers()
                        request_handler.wfile.write(screen_data)
                        return

            # Fallback: return placeholder image
            self._serve_placeholder_image(request_handler)

        except Exception as e:
            logger.error(f"Screen serve error: {e}")
            self._serve_placeholder_image(request_handler)

    def _serve_dashboard_file(self, request_handler, filename: str, dashboard_name: str):
        """Serve a dashboard HTML file.

        Args:
            request_handler: HTTP request handler instance
            filename: Dashboard filename
            dashboard_name: Human-readable dashboard name for error messages
        """
        try:
            dashboard_path = os.path.join(self.static_dir, filename)
            with open(dashboard_path, 'r', encoding='utf-8') as f:
                content = f.read()

            request_handler.send_response(200)
            request_handler.send_header('Content-type', 'text/html; charset=utf-8')
            request_handler.send_header('Access-Control-Allow-Origin', '*')
            request_handler.end_headers()
            request_handler.wfile.write(content.encode('utf-8'))

        except FileNotFoundError:
            request_handler.send_error(404, f"{dashboard_name} template not found")
        except Exception as e:
            logger.error(f"{dashboard_name} serve error: {e}")
            request_handler.send_error(500, f"Failed to serve {dashboard_name.lower()}: {str(e)}")

    def _serve_placeholder_image(self, request_handler):
        """Serve a placeholder image when screen capture is unavailable.

        Args:
            request_handler: HTTP request handler instance
        """
        # 1x1 transparent PNG
        placeholder_data = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
        )
        request_handler.send_response(200)
        request_handler.send_header('Content-type', 'image/png')
        request_handler.send_header('Access-Control-Allow-Origin', '*')
        request_handler.end_headers()
        request_handler.wfile.write(placeholder_data)
