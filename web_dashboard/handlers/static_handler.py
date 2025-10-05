"""
Static File Handler for Web Dashboard

Handles serving static assets (CSS, JS, images) from the static/ directory.
"""

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


class StaticFileHandler:
    """Handler for serving static files."""

    def __init__(self, static_dir: str = "/mnt/data/src/pokemon_crystal_rl/web_dashboard/static"):
        """Initialize static file handler.

        Args:
            static_dir: Base directory for static files
        """
        self.static_dir = static_dir

    def serve_static_file(self, request_handler, path: str):
        """Serve static files (CSS, JS, images).

        Args:
            request_handler: HTTP request handler instance
            path: Request path (e.g., '/static/style.css')
        """
        try:
            # Remove /static/ prefix
            file_path = path[8:]  # Remove '/static/'
            full_path = os.path.join(self.static_dir, file_path)

            # Security check: prevent directory traversal
            if not os.path.abspath(full_path).startswith(os.path.abspath(self.static_dir)):
                request_handler.send_error(403, "Access denied")
                return

            with open(full_path, 'rb') as f:
                content = f.read()

            # Determine content type
            content_type = self._get_content_type(file_path)

            request_handler.send_response(200)
            request_handler.send_header('Content-type', content_type)
            request_handler.send_header('Access-Control-Allow-Origin', '*')
            request_handler.end_headers()
            request_handler.wfile.write(content)

        except FileNotFoundError:
            request_handler.send_error(404, "Static file not found")
        except Exception as e:
            logger.error(f"Static file serve error: {e}")
            request_handler.send_error(500, f"Failed to serve static file: {str(e)}")

    def _get_content_type(self, file_path: str) -> str:
        """Determine content type based on file extension.

        Args:
            file_path: File path

        Returns:
            Content-Type header value
        """
        if file_path.endswith('.css'):
            return 'text/css'
        elif file_path.endswith('.js'):
            return 'application/javascript'
        elif file_path.endswith('.png'):
            return 'image/png'
        elif file_path.endswith('.jpg') or file_path.endswith('.jpeg'):
            return 'image/jpeg'
        elif file_path.endswith('.svg'):
            return 'image/svg+xml'
        elif file_path.endswith('.html'):
            return 'text/html'
        else:
            return 'application/octet-stream'
