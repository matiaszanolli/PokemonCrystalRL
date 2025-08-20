"""
Web server module for Pokemon Crystal RL monitoring
"""

import os
import json
import time
import logging
import threading
from http.server import HTTPServer, SimpleHTTPRequestHandler
from typing import Optional, Dict, Any

from core.monitoring.data_bus import DataBus, DataType, get_data_bus


class MonitoringRequestHandler(SimpleHTTPRequestHandler):
    """HTTP request handler for monitoring interface."""

    def __init__(self, *args, server_state=None, **kwargs):
        """Initialize request handler with server state."""
        self.server_state = server_state
        super().__init__(*args, **kwargs)

    def _send_json_response(self, data: Dict[str, Any]) -> None:
        """Send JSON response.
        
        Args:
            data: Data to send as JSON
        """
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def do_GET(self):
        """Handle GET requests."""
        if self.path == '/api/status':
            self._send_json_response({
                'status': 'active',
                'timestamp': time.time(),
                'components': self.server_state.get('components', {})
            })
            return

        if self.path == '/api/stats':
            self._send_json_response({
                'stats': self.server_state.get('stats', {}),
                'timestamp': time.time()
            })
            return

        if self.path == '/api/screen':
            screen_data = self.server_state.get('latest_screen', {})
            self._send_json_response({
                'screen': screen_data.get('image', None),
                'timestamp': screen_data.get('timestamp', time.time())
            })
            return

        # Serve static files
        if self.path == '/':
            self.path = '/index.html'
            
        try:
            # Get the directory containing the web assets
            web_dir = os.path.join(os.path.dirname(__file__), 'web')
            
            # Construct full file path
            file_path = os.path.join(web_dir, self.path.lstrip('/'))
            
            # Only serve files from web directory
            if not os.path.commonprefix([web_dir, file_path]) == web_dir:
                self.send_error(403)
                return
                
            return SimpleHTTPRequestHandler.do_GET(self)
        except Exception:
            self.send_error(404)


class TrainingWebServer:
    """Web server for monitoring training progress."""

    def __init__(self, config, trainer):
        """Initialize web server with config and trainer instance."""
        self.config = config
        self.trainer = trainer
        self.logger = logging.getLogger('web_server')
        
        self.server = None
        self.server_thread = None
        self._server_state = {
            'components': {},
            'stats': {},
            'latest_screen': None
        }
        
        # Set up data bus subscriptions
        self.data_bus = get_data_bus()
        if self.data_bus:
            self.data_bus.register_component(
                'web_server',
                {
                    'type': 'monitoring',
                    'host': self.config.web_host,
                    'port': self.config.web_port
                }
            )
            
            # Subscribe to relevant data types
            self.data_bus.subscribe(
                DataType.GAME_SCREEN,
                'web_server',
                self._handle_screen_update
            )
            self.data_bus.subscribe(
                DataType.TRAINING_STATS,
                'web_server', 
                self._handle_stats_update
            )
            
    def start(self) -> None:
        """Start the web server."""
        try:
            # Create server instance
            handler = lambda *args: MonitoringRequestHandler(
                *args, 
                server_state=self._server_state
            )
            self.server = HTTPServer(
                (self.config.web_host, self.config.web_port),
                handler
            )
            
            # Start server in separate thread
            self.server_thread = threading.Thread(
                target=self.server.serve_forever,
                daemon=True
            )
            self.server_thread.start()
            
            self.logger.info(
                f"Web server started on http://{self.config.web_host}:{self.config.web_port}"
            )
            
        except Exception as e:
            self.logger.error(f"Failed to start web server: {e}")
            self.server = None
            self.server_thread = None
            
    def shutdown(self) -> None:
        """Shutdown the web server."""
        if self.server:
            try:
                self.server.shutdown()
                self.server.server_close()
                self.server = None
                self.server_thread = None
            except Exception as e:
                self.logger.error(f"Error shutting down web server: {e}")
                
        # Unregister from data bus
        if self.data_bus:
            self.data_bus.unregister_component('web_server')
            
    def _handle_screen_update(self, data: Dict[str, Any], 
                            publisher_id: str) -> None:
        """Handle screen update from data bus.
        
        Args:
            data: Screen data dict
            publisher_id: ID of publishing component
        """
        self._server_state['latest_screen'] = data
        
    def _handle_stats_update(self, data: Dict[str, Any], 
                           publisher_id: str) -> None:
        """Handle stats update from data bus.
        
        Args:
            data: Stats data dict
            publisher_id: ID of publishing component
        """
        self._server_state['stats'] = data
        
    def _update_component_status(self) -> None:
        """Update component status information."""
        if self.data_bus:
            self._server_state['components'] = self.data_bus.get_component_status()
