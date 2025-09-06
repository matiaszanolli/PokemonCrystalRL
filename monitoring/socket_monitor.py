"""Improved socket monitoring implementation with better cleanup and port management."""
import threading
import socket
import time
from flask import Flask
from flask_socketio import SocketIO
import logging

class SocketMonitor:
    """Socket monitor with improved port handling and cleanup."""
    
    def __init__(self, port_range=(8000, 8100)):
        """Initialize the socket monitor.
        
        Args:
            port_range (tuple): Range of ports to try (min, max)
        """
        self.port_range = port_range
        self.app = Flask(__name__)
        self.socketio = SocketIO(self.app)
        self.server_thread = None
        self._port = None
        self._running = False
        self._shutting_down = False
        
        # Add basic error handler
        @self.app.errorhandler(Exception)
        def handle_error(e):
            return str(e), 500

    def _get_available_port(self):
        """Find first available port in range."""
        for port in range(self.port_range[0], self.port_range[1]):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(('', port))
                    return port
            except OSError:
                continue
        raise RuntimeError(f"No ports available in range {self.port_range}")

    def start(self):
        """Start the socket monitor server."""
        if self._running:
            return
            
        self._port = self._get_available_port()
        self._running = True
        self._shutting_down = False
        
        def run_server():
            try:
                self.socketio.run(
                    self.app,
                    host='localhost',
                    port=self._port,
                    debug=False,
                    use_reloader=False,
                    allow_unsafe_werkzeug=True
                )
            except Exception as e:
                if not self._shutting_down:
                    logging.error(f"Socket server error: {e}")

        self.server_thread = threading.Thread(target=run_server)
        self.server_thread.daemon = True
        self.server_thread.start()
        
        # Wait for server to start
        start_time = time.time()
        while time.time() - start_time < 5:  # 5 second timeout
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.connect(('localhost', self._port))
                    return
            except:
                time.sleep(0.1)
        raise RuntimeError("Server failed to start")

    def stop(self):
        """Stop the socket monitor server."""
        if not self._running:
            return
            
        self._shutting_down = True
        self._running = False
        
        try:
            self.socketio.stop()
        except:
            pass
            
        if self.server_thread:
            self.server_thread.join(timeout=1.0)
            self.server_thread = None
            
        self._port = None

    @property
    def port(self):
        """Get current server port."""
        return self._port

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()

if __name__ == '__main__':
    # Example usage
    with SocketMonitor() as monitor:
        print(f"Server running on port {monitor.port}")
        time.sleep(5)  # Keep alive for 5 seconds
