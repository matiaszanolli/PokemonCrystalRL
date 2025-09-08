"""
Monitoring Web Server

This module provides the main web server implementation for the monitoring system,
including REST API, WebSocket communication, and real-time updates.
"""

import os
import json
import threading
import time
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict

from flask import Flask, render_template, jsonify, request, Response
from flask_socketio import SocketIO

from .managers import StatusManager, EventManager
from .services.frame import FrameService, FrameConfig
from .services.metrics import MetricsService, MetricsConfig
from .api.training import TrainingAPI
from .api.system import SystemAPI
from .api.game import GameAPI

from monitoring.base import MonitorComponent, ComponentError
from monitoring.components.capture import ScreenCapture
from monitoring.components.metrics import MetricsCollector
from monitoring.data_bus import get_data_bus

@dataclass
class WebServerConfig:
    """Web server configuration."""
    host: str = "localhost"
    port: int = 8080
    debug: bool = False
    enable_api: bool = True
    enable_websocket: bool = True
    enable_metrics: bool = True
    template_dir: str = "templates"
    static_dir: str = "static"
    api_prefix: str = "/api/v1"
    frame_buffer_size: int = 1  # Only buffer 1 frame for lowest latency
    frame_quality: int = 85    # JPEG quality (0-100)
    update_interval: float = 0.033  # ~30fps for status updates
    metrics_interval: float = 1.0   # 1s for metrics
    metrics_retention: float = 24.0  # hours

class MonitoringServer(MonitorComponent):
    """Main monitoring web server.
    
    This class provides:
    - Real-time monitoring dashboard
    - REST API endpoints
    - WebSocket updates
    - Training metrics visualization
    - Screen capture streaming
    """
    
    def __init__(self, config: Optional[WebServerConfig] = None):
        """Initialize monitoring server.
        
        Args:
            config: Server configuration
        """
        self.config = config or WebServerConfig()
        
        # Setup Flask and SocketIO
        self.app = Flask(
            __name__,
            template_folder=self.config.template_dir,
            static_folder=self.config.static_dir
        )
        self.socketio = SocketIO(
            self.app,
            cors_allowed_origins=[],  # Local only
            async_mode="threading",
            ping_timeout=2,
            ping_interval=1,
            max_http_buffer_size=5 * 1024 * 1024  # 5MB for frame data
        )
        
        # Initialize managers
        self.status_manager = StatusManager(self.socketio)
        self.event_manager = EventManager(self.status_manager)
        
        # Initialize services
        self.frame_service = FrameService(
            FrameConfig(
                buffer_size=self.config.frame_buffer_size,
                quality=self.config.frame_quality
            )
        )
        self.metrics_service = MetricsService(
            MetricsConfig(
                update_interval=self.config.metrics_interval,
                retention_hours=self.config.metrics_retention
            )
        )
        
        # Initialize APIs
        self.training_api = TrainingAPI()
        self.system_api = SystemAPI()
        self.game_api = GameAPI()
        
        # Components
        self.screen_capture = None
        self.metrics_collector = None
        self.data_bus = get_data_bus()
        
        # State tracking
        self._running = False
        self._server_thread = None
        self._update_thread = None
        self._lock = threading.RLock()
        self._last_update = 0.0
        
        # Setup routes and handlers
        self._setup_routes()
        self._setup_socketio()
        self._setup_api()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
    def set_screen_capture(self, capture: ScreenCapture) -> None:
        """Set screen capture component.
        
        Args:
            capture: Screen capture component
        """
        self.screen_capture = capture
        self.frame_service.set_screen_capture(capture)
        self.system_api.update_screen_capture(capture)
        self.game_api.update_screen_capture(capture)
        
    def set_metrics_collector(self, collector: MetricsCollector) -> None:
        """Set metrics collector component.
        
        Args:
            collector: Metrics collector component
        """
        self.metrics_collector = collector
        self.metrics_service.set_metrics_collector(collector)
        
    def start(self) -> bool:
        """Start monitoring server.
        
        Returns:
            bool: True if started successfully
        """
        with self._lock:
            if self._running:
                return True
                
            self._running = True
            
            # Start update thread
            self._update_thread = threading.Thread(
                target=self._update_loop,
                daemon=True,
                name="MonitorUpdates"
            )
            self._update_thread.start()
            
            # Start server thread
            self._server_thread = threading.Thread(
                target=self._run_server,
                daemon=True,
                name="MonitorServer"
            )
            self._server_thread.start()
            
            return True
            
    def stop(self) -> bool:
        """Stop monitoring server.
        
        Returns:
            bool: True if stopped successfully
        """
        with self._lock:
            if not self._running:
                return True
                
            self._running = False
            
            try:
                # First try to shutdown Socket.IO
                self.socketio.stop()
                
                # Give a brief delay for cleanup
                time.sleep(0.1)
                
                # Now shutdown the Flask server
                if self.app:
                    try:
                        from werkzeug.serving import shutdown_server
                        shutdown_server()
                    except:
                        pass
            except:
                pass
            
        # Join threads with timeout
        if self._update_thread and self._update_thread.is_alive():
            self._update_thread.join(timeout=2.0)
            
        if self._server_thread and self._server_thread.is_alive():
            self._server_thread.join(timeout=2.0)
            
        is_shutdown = not (
            (self._update_thread and self._update_thread.is_alive()) or
            (self._server_thread and self._server_thread.is_alive())
        )
        
        return is_shutdown
            
    def get_status(self) -> Dict[str, Any]:
        """Get server status.
        
        Returns:
            Dict with status information
        """
        with self._lock:
            frame_status = self.frame_service.get_status()
            metrics_status = self.metrics_service.get_status()
            
            return {
                "running": self._running,
                "frame_service": frame_status,
                "metrics_service": metrics_status,
                "last_update": self._last_update
            }
            
    def _setup_routes(self) -> None:
        """Set up HTTP routes."""
        
        @self.app.route("/")
        def index():
            """Main dashboard page."""
            return render_template(
                "dashboard.html",
                config=asdict(self.config)
            )
            
        @self.app.route("/status")
        def status():
            """Server status page."""
            return render_template(
                "status.html",
                status=self.get_status()
            )
            
    def _setup_socketio(self) -> None:
        """Set up WebSocket handlers."""
        
        @self.socketio.on("connect")
        def handle_connect(auth):
            """Handle client connection.
            
            Args:
                auth: Authentication data from client (required by Flask-SocketIO)
            """
            self.status_manager.client_connected()
            
        @self.socketio.on("disconnect")
        def handle_disconnect():
            """Handle client disconnection."""
            self.status_manager.client_disconnected()
            
        @self.socketio.on("request_frame")
        def handle_frame_request():
            """Handle frame request."""
            self.frame_service.handle_frame_request()
            
        @self.socketio.on("subscribe_metrics")
        def handle_metrics_subscribe(data):
            """Handle metrics subscription."""
            metrics = self.metrics_service.get_metrics(
                names=data.get("metrics"),
                since=data.get("since")
            )
            if metrics:
                with self.app.app_context():
                    self.socketio.emit("metrics", metrics)
            
    def _setup_api(self) -> None:
        """Set up REST API endpoints."""
        prefix = self.config.api_prefix
        
        @self.app.route(f"{prefix}/status")
        def api_status():
            """Get server status."""
            return jsonify(self.get_status())
            
        @self.app.route(f"{prefix}/training/toggle", methods=['POST'])
        def api_toggle_training():
            """Toggle training state."""
            if not self.data_bus:
                return jsonify({'error': 'Data bus not initialized'}), 500
            
            try:
                # Publish toggle command to data bus
                self.data_bus.publish('training.command', {'action': 'toggle'})
                return jsonify({'success': True})
            except Exception as e:
                return jsonify({'error': str(e)}), 500
                
        @self.app.route(f"{prefix}/environment/reset", methods=['POST'])
        def api_reset_environment():
            """Reset the game environment."""
            if not self.data_bus:
                return jsonify({'error': 'Data bus not initialized'}), 500
            
            try:
                # Publish reset command to data bus
                self.data_bus.publish('environment.command', {'action': 'reset'})
                return jsonify({'success': True})
            except Exception as e:
                return jsonify({'error': str(e)}), 500
            
        @self.app.route(f"{prefix}/metrics")
        def api_metrics():
            """Get current metrics."""
            if not self.metrics_service:
                return jsonify({'error': 'Metrics service not initialized'}), 500
                
            try:
                metrics = self.metrics_service.get_metrics(
                    names=request.args.getlist("metrics"),
                    since=request.args.get("since", type=float)
                )
                return jsonify(metrics)
            except Exception as e:
                return jsonify({'error': str(e)}), 500
            
        @self.app.route(f"{prefix}/training/stats")
        def api_training_stats():
            """Get training statistics."""
            return jsonify(self.training_api.get_training_stats())
            
        @self.app.route(f"{prefix}/training/decisions")
        def api_training_decisions():
            """Get LLM decisions."""
            return jsonify(self.training_api.get_llm_decisions())
            
        @self.app.route(f"{prefix}/system/status")
        def api_system_status():
            """Get system status."""
            return jsonify(self.system_api.get_system_status())
            
        @self.app.route(f"{prefix}/game/frame")
        def api_frame():
            """Get current frame."""
            format = request.args.get("format", "png")
            if format in ["png", "jpeg"]:
                frame = self.game_api.get_screen_bytes()
                if frame:
                    return Response(frame, mimetype=f"image/{format}")
            return jsonify({"error": "Frame not available"})
            
        @self.app.route(f"{prefix}/game/debug")
        def api_game_debug():
            """Get game debug info."""
            return jsonify(self.game_api.get_memory_debug())
                
    def _run_server(self) -> None:
        """Run the web server."""
        try:
            # Run server with greater concurrency in testing
            self.socketio.run(
                self.app,
                host=self.config.host,
                port=self.config.port,
                debug=self.config.debug,
                use_reloader=False,
                allow_unsafe_werkzeug=True,  # Allow Werkzeug server in test environment
                log_output=False,            # Silence logs in testing
                max_size=5,                  # Allow up to 5 workers
                num_threads=3                # Use 3 worker threads
            )
        except Exception as e:
            self.logger.error(f"Server error: {e}")
            raise ComponentError(f"Web server error: {e}")
            
    def _update_loop(self) -> None:
        """Background update loop."""
        while self._running:
            try:
                now = time.time()
                
                # Check update interval
                if now - self._last_update < self.config.update_interval:
                    time.sleep(0.1)
                    continue
                    
                self._send_updates()
                self._last_update = now
                
            except Exception as e:
                self.logger.error(f"Update error: {e}")
                time.sleep(1.0)
                
    def _send_updates(self) -> None:
        """Send updates to connected clients."""
        try:
            # Get and send metrics update
            metrics = self.metrics_service.get_metrics()
            if metrics:
                with self.app.app_context():
                    self.socketio.emit("metrics", metrics)
                
            # Get and send status update
            status = self.get_status()
            with self.app.app_context():
                self.socketio.emit("status", status)
            
            # Send error events if available
            if hasattr(self.metrics_service, 'get_pending_errors'):
                errors = self.metrics_service.get_pending_errors()
                if errors:
                    with self.app.app_context():
                        self.socketio.emit("errors", errors)
            
            # Send frame data if available
            if hasattr(self.frame_service, 'get_latest_frame'):
                frame = self.frame_service.get_latest_frame()
                if frame:
                    with self.app.app_context():
                        self.socketio.emit("frame", frame)
                
        except Exception as e:
            self.logger.error(f"Update error: {e}")
            
    def _handle_callback_error(self, e: Exception, context: str) -> None:
        """Handle callback error.
        
        Args:
            e: Exception that occurred
            context: Error context
        """
        self.logger.error(f"Callback error in {context}: {e}")
        self.data_bus.publish("monitoring.error", {
            "component": "web_server",
            "context": context,
            "error": str(e),
            "timestamp": time.time()
        })
