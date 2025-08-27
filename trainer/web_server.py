"""Legacy HTTP server for Pokemon Trainer interface"""

import base64
from dataclasses import dataclass
from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import logging
import socket
import threading
import time


@dataclass
class ServerConfig:
    """Web server configuration."""
    host: str = "localhost"
    port: int = 8080

    @classmethod
    def from_training_config(cls, training_config):
        """Create ServerConfig from TrainingConfig"""
        return cls(
            host=training_config.web_host,
            port=training_config.web_port
        )


def find_available_port(start_port: int = 8080) -> int:
    """Find an available port starting from the specified port."""
    for port in range(start_port, start_port + 10):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(('localhost', port))
                return port
        except OSError:
            continue
    raise RuntimeError(f"Could not find available port starting from {start_port}")


class WebServer:
    ServerConfig = ServerConfig
    
    def __init__(self, config: ServerConfig = None, trainer = None):
        self.config = config or ServerConfig()
        self.trainer = trainer
        self.server = None
        self._running = False
        self.logger = logging.getLogger("web_server")
        self.port = find_available_port(self.config.port)
        
    def start(self):
        """Start the HTTP server"""
        # Create handler factory that captures trainer reference
        def handler_factory(*args):
            handler = WebHandler(*args)
            handler.trainer = self.trainer
            return handler
            
        # Create server
        try:
            self.server = HTTPServer(
                (self.config.host, self.port),
                handler_factory
            )
            self._running = True
            return self.server
        except Exception as e:
            self.logger.error(f"Failed to start web server: {e}")
            return None
    
    def stop(self):
        """Stop the HTTP server"""
        if self.server:
            self.server.shutdown()
            self.server.server_close()
            self._running = False
    
    def run_in_thread(self):
        """Run server in a separate thread"""
        if not self.server:
            self.start()
        try:
            self.server.serve_forever()
        except Exception as e:
            self.logger.error(f"Server thread error: {e}")
            raise
        finally:
            self.stop()
    

class WebHandler(BaseHTTPRequestHandler):
    """HTTP request handler for web interface"""
    trainer = None  # Set by factory function

    def do_GET(self):
        """Handle GET requests"""
        try:
            if self.path == "/":
                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                self.wfile.write(b"<h1>Pokemon Trainer Web Server</h1>")
                
            elif self.path == "/screen":
                if self.trainer and hasattr(self.trainer, 'latest_screen'):
                    screen_data = self.trainer.latest_screen
                    if screen_data and 'image_b64' in screen_data:
                        image_data = base64.b64decode(screen_data['image_b64'])
                        self.send_response(200)
                        self.send_header('Content-type', 'image/png')
                        self.end_headers()
                        self.wfile.write(image_data)
                    else:
                        self.send_error(404)
                else:
                    self.send_error(404)
            
            elif self.path == "/stats":
                if self.trainer:
                    stats = {}
                    if hasattr(self.trainer, 'get_current_stats'):
                        stats = self.trainer.get_current_stats()
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps(stats).encode('utf-8'))
                else:
                    self.send_error(404)
                    
            else:
                self.send_error(404)
                
        except Exception as e:
            self.send_error(500)
