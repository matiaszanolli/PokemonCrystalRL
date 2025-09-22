"""
Web Server Component

This module provides the base web server implementation for monitoring.
The full implementation will be expanded in Phase 4.2.
"""

import threading
from typing import Dict, Any, Optional
from flask import Flask, render_template
from flask_socketio import SocketIO, emit

from ..base import MonitorComponent, ComponentError

class WebServer(MonitorComponent):
    """Base web server implementation.
    
    This provides the core web server functionality. The full implementation
    with comprehensive API and real-time updates will be added in Phase 4.2.
    """
    
    def __init__(self, host: str = "localhost", port: int = 5000):
        """Initialize web server.
        
        Args:
            host: Host to bind to
            port: Port to listen on
        """
        self.host = host
        self.port = port
        
        # Flask setup
        self.app = Flask(__name__)
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        
        # State tracking
        self._running = False
        self._server_thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()
        
        # Performance tracking
        self.requests_handled = 0
        self.ws_messages_sent = 0
        self.ws_clients = 0
        
        self._setup_routes()
        self._setup_socketio()
    
    def start(self) -> bool:
        """Start the web server.
        
        Returns:
            bool: True if started successfully
        """
        with self._lock:
            if self._running:
                return True
                
            self._running = True
            
            # Start server in thread
            self._server_thread = threading.Thread(
                target=self._run_server,
                daemon=True,
                name="WebServer"
            )
            self._server_thread.start()
            
            return True
    
    def stop(self) -> bool:
        """Stop the web server.
        
        Returns:
            bool: True if stopped successfully
        """
        with self._lock:
            self._running = False
            
        if self._server_thread and self._server_thread.is_alive():
            self._server_thread.join(timeout=5.0)
            return not self._server_thread.is_alive()
            
        return True
    
    def get_status(self) -> Dict[str, Any]:
        """Get server status.
        
        Returns:
            Dict with status information
        """
        with self._lock:
            return {
                "running": self._running,
                "host": self.host,
                "port": self.port,
                "requests_handled": self.requests_handled,
                "ws_messages_sent": self.ws_messages_sent,
                "ws_clients": self.ws_clients
            }
    
    def _setup_routes(self) -> None:
        """Set up HTTP routes."""
        
        @self.app.route("/")
        def index():
            """Main dashboard page."""
            self.requests_handled += 1
            return render_template("dashboard.html")
            
        @self.app.route("/api/v1/status")
        def status():
            """Get server status."""
            self.requests_handled += 1
            return self.get_status()
    
    def _setup_socketio(self) -> None:
        """Set up WebSocket event handlers."""
        
        @self.socketio.on("connect")
        def handle_connect():
            """Handle client connection."""
            self.ws_clients += 1
            emit("status", {"connected": True})
            
        @self.socketio.on("disconnect")
        def handle_disconnect():
            """Handle client disconnection."""
            self.ws_clients -= 1
    
    def _run_server(self) -> None:
        """Run the web server."""
        try:
            self.socketio.run(
                self.app,
                host=self.host,
                port=self.port,
                debug=False,
                use_reloader=False
            )
        except Exception as e:
            raise ComponentError(f"Web server error: {e}")
    
    def publish_update(self, event: str, data: Any) -> bool:
        """Publish an update to connected clients.
        
        Args:
            event: Event name
            data: Update data
            
        Returns:
            bool: True if published successfully
        """
        try:
            self.socketio.emit(event, data)
            self.ws_messages_sent += 1
            return True
        except Exception:
            return False
