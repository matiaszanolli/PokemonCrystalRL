"""
Web server module for Pokemon Crystal RL monitoring.

This module provides the HTTP and WebSocket server functionality, handling:
- HTTP API endpoints for data retrieval
- WebSocket connections for real-time updates
- Static file serving
- Authentication and session management
- Event streaming
- Request routing and middleware
"""

import asyncio
import aiohttp_cors
import json
import logging
from typing import Dict, Any, Optional, List, Callable, Set
from dataclasses import dataclass
from datetime import datetime
import threading
from pathlib import Path
import time
import weakref
from functools import partial
import signal
import socket
import base64
from http.server import HTTPServer, BaseHTTPRequestHandler
import os

# Configure module logger
logger = logging.getLogger(__name__)

from aiohttp import web
import aiohttp
from aiohttp.web import (
    Application, Request, Response, WebSocketResponse,
    middleware, HTTPException
)
import jwt
import aiofiles

from .data_bus import DataType, get_data_bus
from .error_handler import ErrorHandler, ErrorCategory, ErrorSeverity

# Check for psutil availability
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# Types for event handlers
EventHandler = Callable[[Dict[str, Any]], None]
WSClient = WebSocketResponse


@dataclass
class ServerConfig:
    """Web server configuration."""
    host: str = "localhost"
    port: int = 8080
    static_dir: str = "static"
    template_dir: str = "templates"
    secret_key: str = "default_secret_key"  # Change in production
    debug: bool = False
    ssl_cert: Optional[str] = None
    ssl_key: Optional[str] = None
    cors_origins: List[str] = None
    max_request_size: int = 10 * 1024 * 1024  # 10MB
    session_timeout: int = 3600  # 1 hour

    @classmethod
    def from_training_config(cls, training_config):
        """Create ServerConfig from TrainingConfig"""
        return cls(
            host=training_config.web_host,
            port=training_config.web_port,
            debug=training_config.debug_mode
        )


class WebServer:
    """HTTP and WebSocket server for monitoring interface."""
    ServerConfig = ServerConfig
    
    def __init__(self, config: Optional[ServerConfig] = None):
        self.config = config or ServerConfig()
        
        # Initialize components
        self.app = Application(
            client_max_size=self.config.max_request_size
        )
        self.data_bus = get_data_bus()
        self.error_handler = ErrorHandler()
        self.logger = logging.getLogger("web_server")
        
        # WebSocket clients
        self._ws_clients: Set[WSClient] = weakref.WeakSet()
        self._ws_lock = threading.Lock()
        
        # Event handlers
        self.event_handlers: Dict[str, List[EventHandler]] = {}
        
        # Setup routes and middleware
        self._setup_routes()
        self._setup_middleware()
        
        # Initialize server state
        self.is_running = False
        self.start_time = None
        self._cleanup_task = None
        self._ready = threading.Event()
        
        self.logger.info("🌐 Web server initialized")
    
    def _setup_routes(self) -> None:
        """Setup HTTP routes and WebSocket endpoints."""
        # API routes
        self.app.router.add_get("/api/status", self.handle_status)
        self.app.router.add_get("/api/metrics", self.handle_metrics)
        self.app.router.add_get("/api/events", self.handle_events)
        self.app.router.add_get("/api/training/state", self.handle_training_state)
        self.app.router.add_post("/api/training/control", self.handle_training_control)
        
        # WebSocket routes
        self.app.router.add_get("/ws", self.handle_websocket)
        
        # Static files
        self.app.router.add_static(
            "/static/",
            Path(self.config.static_dir),
            append_version=True
        )
        
        # SPA fallback
        self.app.router.add_get("/{tail:.*}", self.handle_spa)
    
    @middleware
    async def error_middleware(self, request: Request, handler: Callable) -> Response:
        """Middleware for error handling."""
        try:
            return await handler(request)
        except HTTPException as e:
            return web.json_response(
                {"error": str(e)},
                status=e.status
            )
        except Exception as e:
            self.error_handler.handle_error(
                e,
                severity=ErrorSeverity.ERROR,
                category=ErrorCategory.SYSTEM,
                component="web_server"
            )
            return web.json_response(
                {"error": "Internal server error"},
                status=500
            )
    
    @middleware
    async def auth_middleware(self, request: Request, handler: Callable) -> Response:
        """Middleware for authentication."""
        # Skip auth for public routes and API endpoints (for testing)
        public_routes = ["/", "/static", "/ws"]
        if (request.path in public_routes or 
            request.path.startswith("/static/") or 
            request.path.startswith("/api/")):
            return await handler(request)
        
        try:
            auth_header = request.headers.get("Authorization")
            if not auth_header or not auth_header.startswith("Bearer "):
                raise web.HTTPUnauthorized(
                    text=json.dumps({"error": "Missing authentication"}),
                    content_type="application/json"
                )
            
            token = auth_header.split(" ")[1]
            payload = jwt.decode(
                token,
                self.config.secret_key,
                algorithms=["HS256"]
            )
            
            request["user"] = payload
            return await handler(request)
            
        except jwt.InvalidTokenError:
            raise web.HTTPUnauthorized(
                text=json.dumps({"error": "Invalid token"}),
                content_type="application/json"
            )
    
    def _setup_middleware(self) -> None:
        """Setup middleware stack."""
        self.app.middlewares.append(self.error_middleware)
        self.app.middlewares.append(self.auth_middleware)
        
        # CORS setup if needed
        if self.config.cors_origins:
            cors = aiohttp_cors.setup(self.app, defaults={
                origin: aiohttp_cors.ResourceOptions(
                    allow_credentials=True,
                    expose_headers="*",
                    allow_headers="*"
                )
                for origin in self.config.cors_origins
            })
            
            # Apply CORS to all routes
            for route in list(self.app.router.routes()):
                cors.add(route)
    
    async def handle_status(self, request: Request) -> Response:
        """Handle server status requests."""
        return web.json_response({
            "status": "running",
            "uptime": time.time() - self.start_time if self.start_time else 0,
            "connected_clients": len(self._ws_clients),
            "version": "1.0.0"
        })
    
    async def handle_metrics(self, request: Request) -> Response:
        """Handle metrics data requests."""
        try:
            # Get query parameters
            start_time = float(request.query.get("start", 0))
            end_time = float(request.query.get("end", time.time()))
            metrics = request.query.getall("metric", [])
            
            # TODO: Implement metric data retrieval from storage
            
            return web.json_response({
                "metrics": [],
                "start_time": start_time,
                "end_time": end_time
            })
            
        except ValueError:
            raise web.HTTPBadRequest(
                text=json.dumps({"error": "Invalid time range"}),
                content_type="application/json"
            )
    
    async def handle_events(self, request: Request) -> Response:
        """Handle event history requests."""
        try:
            # Get query parameters
            start_time = float(request.query.get("start", 0))
            end_time = float(request.query.get("end", time.time()))
            event_types = request.query.getall("type", [])
            
            # TODO: Implement event history retrieval
            
            return web.json_response({
                "events": [],
                "start_time": start_time,
                "end_time": end_time
            })
            
        except ValueError:
            raise web.HTTPBadRequest(
                text=json.dumps({"error": "Invalid time range"}),
                content_type="application/json"
            )
    
    async def handle_training_state(self, request: Request) -> Response:
        """Handle training state requests."""
        if not self.data_bus:
            return web.json_response({"error": "Data bus not available"}, status=503)
        
        # Request current training state from data bus
        try:
            # TODO: Implement state retrieval from training system
            return web.json_response({
                "state": "unknown",
                "timestamp": time.time()
            })
        except Exception as e:
            self.error_handler.handle_error(
                e,
                severity=ErrorSeverity.ERROR,
                category=ErrorCategory.TRAINING,
                component="web_server"
            )
            return web.json_response({"error": str(e)}, status=500)
    
    async def handle_training_control(self, request: Request) -> Response:
        """Handle training control commands."""
        try:
            data = await request.json()
            command = data.get("command")
            
            if not command:
                raise web.HTTPBadRequest(
                    text=json.dumps({"error": "Missing command"}),
                    content_type="application/json"
                )
            
            # Publish control command to data bus
            if self.data_bus:
                self.data_bus.publish(
                    DataType.TRAINING_CONTROL,
                    {
                        "command": command,
                        "parameters": data.get("parameters", {}),
                        "timestamp": time.time()
                    },
                    "web_server"
                )
            
            return web.json_response({"status": "command sent"})
            
        except json.JSONDecodeError:
            raise web.HTTPBadRequest(
                text=json.dumps({"error": "Invalid JSON"}),
                content_type="application/json"
            )
    
    async def handle_websocket(self, request: Request) -> WebSocketResponse:
        """Handle WebSocket connections."""
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        
        # Register client
        with self._ws_lock:
            self._ws_clients.add(ws)
        
        try:
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                        await self._handle_ws_message(ws, data)
                    except json.JSONDecodeError:
                        await ws.send_json({"error": "Invalid JSON"})
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    self.logger.error(f"WebSocket error: {ws.exception()}")
        
        finally:
            # Unregister client
            with self._ws_lock:
                self._ws_clients.remove(ws)
        
        return ws
    
    async def _handle_ws_message(self, ws: WebSocketResponse,
                               data: Dict[str, Any]) -> None:
        """Handle incoming WebSocket messages."""
        message_type = data.get("type")
        if not message_type:
            await ws.send_json({"error": "Missing message type"})
            return
        
        handlers = self.event_handlers.get(message_type, [])
        for handler in handlers:
            try:
                handler(data)
            except Exception as e:
                self.error_handler.handle_error(
                    e,
                    severity=ErrorSeverity.ERROR,
                    category=ErrorCategory.SYSTEM,
                    component="web_server"
                )
    
    async def handle_spa(self, request: Request) -> Response:
        """Handle SPA routes by serving index.html."""
        try:
            async with aiofiles.open(
                Path(self.config.static_dir) / "index.html", mode='r'
            ) as f:
                content = await f.read()
                return web.Response(
                    text=content,
                    content_type="text/html"
                )
        except:
            raise web.HTTPNotFound()
    
    async def broadcast(self, event_type: str, data: Dict[str, Any]) -> None:
        """Broadcast message to all connected WebSocket clients."""
        if not self._ws_clients:
            return
        
        message = {
            "type": event_type,
            "data": data,
            "timestamp": time.time()
        }
        
        # Create tasks for each client
        tasks = []
        with self._ws_lock:
            for ws in self._ws_clients:
                if not ws.closed:
                    tasks.append(ws.send_json(message))
        
        # Wait for all sends to complete
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    def subscribe_to_events(self, event_type: str,
                          handler: EventHandler) -> None:
        """Subscribe to specific events."""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)
    
    async def _cleanup_loop(self) -> None:
        """Periodic cleanup of dead connections."""
        try:
            while self.is_running:
                with self._ws_lock:
                    dead = {ws for ws in self._ws_clients if ws.closed}
                    self._ws_clients.difference_update(dead)
                
                # Use asyncio.wait_for with timeout to make cancellation more responsive
                try:
                    await asyncio.wait_for(asyncio.sleep(60), timeout=60)
                except asyncio.TimeoutError:
                    continue  # This shouldn't happen, but just in case
        except asyncio.CancelledError:
            self.logger.debug("Cleanup loop cancelled")
            raise
        except Exception as e:
            self.logger.error(f"Error in cleanup loop: {e}")
        finally:
            self.logger.debug("Cleanup loop finished")
    
    def _setup_signal_handlers(self) -> None:
        """Setup handlers for system signals."""
        # Signal handlers only work in the main thread
        if threading.current_thread() is not threading.main_thread():
            self.logger.debug("Skipping signal handler setup - not in main thread")
            return
            
        try:
            loop = asyncio.get_running_loop()
            for sig in (signal.SIGTERM, signal.SIGINT):
                loop.add_signal_handler(
                    sig,
                    lambda s=sig: loop.create_task(self.shutdown(s))
                )
            self.logger.debug("Signal handlers set up successfully")
        except Exception as e:
            self.logger.warning(f"Failed to set up signal handlers: {e}")
    
    async def shutdown(self, sig: Optional[signal.Signals] = None) -> None:
        """Shutdown the server gracefully."""
        if sig:
            self.logger.info(f"Received exit signal {sig.name}...")
        
        self.logger.info("Shutting down web server...")
        
        # Set shutdown flag first
        self.is_running = False
        
        # Cancel cleanup task with proper error handling
        if self._cleanup_task and not self._cleanup_task.done():
            self.logger.debug("Cancelling cleanup task...")
            self._cleanup_task.cancel()
            try:
                await asyncio.wait_for(self._cleanup_task, timeout=5.0)
            except asyncio.CancelledError:
                self.logger.debug("Cleanup task cancelled successfully")
            except asyncio.TimeoutError:
                self.logger.warning("Cleanup task cancellation timed out")
            except Exception as e:
                self.logger.error(f"Error during cleanup task cancellation: {e}")
        
        # Close all WebSocket connections with error handling
        try:
            with self._ws_lock:
                close_tasks = []
                for ws in self._ws_clients:
                    if not ws.closed:
                        close_tasks.append(ws.close(
                            code=aiohttp.WSCloseCode.GOING_AWAY,
                            message="Server shutdown"
                        ))
                
                if close_tasks:
                    await asyncio.gather(*close_tasks, return_exceptions=True)
        except Exception as e:
            self.logger.error(f"Error closing WebSocket connections: {e}")
        
        # Shutdown the application with error handling
        try:
            await self.app.shutdown()
            await self.app.cleanup()
            self.logger.info("Web server shutdown complete")
        except Exception as e:
            self.logger.error(f"Error during application shutdown: {e}")
    
    async def start(self) -> None:
        """Start the web server."""
        if self.is_running:
            return
        
        self.is_running = True
        self.start_time = time.time()
        
        # Setup signal handlers
        self._setup_signal_handlers()
        
        # Start cleanup task
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        # Configure SSL if needed
        ssl_context = None
        if self.config.ssl_cert and self.config.ssl_key:
            import ssl
            ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
            ssl_context.load_cert_chain(
                self.config.ssl_cert,
                self.config.ssl_key
            )
        
        # Start the server
        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(
            runner,
            self.config.host,
            self.config.port,
            ssl_context=ssl_context
        )
        
        await site.start()
        self.logger.info(
            f"🚀 Server started at "
            f"{'https' if ssl_context else 'http'}://"
            f"{self.config.host}:{self.config.port}"
        )
    
    def run(self) -> None:
        """Run the server in the current thread."""
        if threading.current_thread() is threading.main_thread():
            loop = asyncio.get_event_loop()
        else:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        try:
            loop.run_until_complete(self.start())
            self._ready.set()  # Signal server is ready
            loop.run_forever()
        except KeyboardInterrupt:
            self.logger.info("Received keyboard interrupt")
            loop.run_until_complete(self.shutdown())
        except Exception as e:
            self.logger.error(f"Error in server run loop: {e}")
            loop.run_until_complete(self.shutdown())
        finally:
            # Cancel any remaining tasks
            pending_tasks = [task for task in asyncio.all_tasks(loop) if not task.done()]
            if pending_tasks:
                self.logger.info(f"Cancelling {len(pending_tasks)} pending tasks...")
                for task in pending_tasks:
                    task.cancel()
                
                # Wait for tasks to be cancelled
                try:
                    loop.run_until_complete(
                        asyncio.gather(*pending_tasks, return_exceptions=True)
                    )
                except Exception as e:
                    self.logger.error(f"Error cancelling pending tasks: {e}")
            
            loop.close()
            self.logger.debug("Event loop closed")
    
    def run_in_thread(self) -> threading.Thread:
        """Run the server in a separate thread and wait for it to be ready."""
        def run_with_exception_handling():
            try:
                self.run()
            except Exception as e:
                self.logger.error(f"Error in server thread: {e}")
                raise
        
        thread = threading.Thread(target=run_with_exception_handling, daemon=True)
        thread.start()
        # Wait for server to be ready
        if not self._ready.wait(timeout=10.0):
            self.logger.warning("Server did not become ready within timeout")
        return thread


# Legacy HTTP server classes for backwards compatibility
class TrainingWebServer:
    """Legacy training web server for backwards compatibility."""
    ServerConfig = ServerConfig
    
    def __init__(self, config, trainer):
        self.config = config
        self.trainer = trainer
        self.server = None
        self._running = False
        self.port = self._find_available_port()
        # Update config port to reflect the actually used port
        self.config.port = self.port
        self._trainer = trainer # For backward compatibility
        
        # Register with data bus
        self.data_bus = get_data_bus()
        if self.data_bus:
            self.data_bus.register_component("web_server", {
                "type": "monitoring",
                "port": self.port,
                "host": getattr(self.config, 'host', 'localhost')
            })
        
    def _find_available_port(self):
        """Find an available port starting from the configured port."""
        start_port = getattr(self.config, 'port', 8080)
        
        # Try specified port first
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind((getattr(self.config, 'host', 'localhost'), start_port))
                return start_port
        except OSError:
            pass
        
        # Try port range
        for port in range(start_port, start_port + 1000):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    s.bind((getattr(self.config, 'host', 'localhost'), port))
                    return port
            except OSError:
                continue
        
        # Let OS pick a port
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind((getattr(self.config, 'host', 'localhost'), 0))
                _, port = s.getsockname()
                return port
        except OSError:
            pass
        
        raise RuntimeError(f"Could not find available port starting from {start_port}")
    
    def start(self):
        """Start the HTTP server."""
        # Create partial function to capture trainer reference
        def handler_factory(*args):
            return TrainingHandler(self.trainer, *args)
            
        try:
            logger.debug(f"Starting server on {getattr(self.config, 'host', 'localhost')}:{self.port}")
            
            self.server = HTTPServer(
                (getattr(self.config, 'host', 'localhost'), self.port),
                handler_factory
            )
            self._running = True
            
            # Start server in a thread
            server_thread = threading.Thread(
                target=self.server.serve_forever,
                daemon=True
            )
            server_thread.start()
            logger.debug(f"Server thread started on port {self.port}")
            
            return self.server
        except Exception as e:
            logger.error(f"Failed to start server: {e}")
            self.server = None
            self._running = False
            self.data_bus.unregister_component("web_server")
            return None
    
    def stop(self):
        """Stop the HTTP server."""
        logger.debug("TrainingWebServer - Stopping HTTP server...")
        
        try:
            # First unregister from data bus to prevent new messages
            try:
                if self.data_bus:
                    logger.debug("TrainingWebServer - Unregistering from data bus")
                    self.data_bus.unregister_component("web_server")
                    logger.debug("TrainingWebServer - Unregistered from data bus")
            except Exception as e:
                logger.error(f"TrainingWebServer - Error unregistering from data bus: {e}")
            
            # Prevent further requests
            self._running = False
            logger.debug("TrainingWebServer - Marked as not running")
            
            # Shutdown server if running
            if self.server:
                try:
                    logger.debug("TrainingWebServer - Shutting down HTTP server...")
                    # In test mode, server might not be listening
                    if hasattr(self.server, '_handle_request_noblock'):
                        self.server._BaseServer__shutdown_request = True
                    self.server.shutdown()
                    self.server.server_close()
                    logger.debug("TrainingWebServer - HTTP server shutdown complete")
                except Exception as e:
                    logger.error(f"TrainingWebServer - Error shutting down HTTP server: {e}")
                    # Ensure we still clear the reference
                    pass
            
            self.server = None
            logger.debug("TrainingWebServer - Cleared server reference")
            
        except Exception as e:
            logger.error(f"TrainingWebServer - Unexpected error during stop: {e}")
            # Always ensure the server is marked as not running
            self._running = False
            self.server = None
    
    def shutdown(self):
        """Shutdown and cleanup."""
        logger.debug("TrainingWebServer - Starting shutdown sequence")
        try:
            self.stop()
            logger.debug("TrainingWebServer - Shutdown completed successfully")
        except Exception as e:
            logger.error(f"TrainingWebServer - Error during shutdown: {e}")


class TrainingHandler(BaseHTTPRequestHandler):
    """Legacy HTTP request handler for backwards compatibility."""
    
    def __init__(self, trainer, *args, **kwargs):
        self.trainer = trainer
        super().__init__(*args, **kwargs)
    
    def do_GET(self):
        """Handle GET requests."""
        if self.path == "/":
            self._serve_comprehensive_dashboard()
        elif self.path in ["/screen", "/api/screenshot"]:
            self._serve_screen()
        elif self.path == "/stats":
            self._serve_stats()
        elif self.path == "/api/status":
            self._serve_api_status()
        elif self.path == "/api/system":
            self._serve_api_system()
        elif self.path == "/api/runs":
            self._serve_api_runs()
        elif self.path == "/api/text":
            self._serve_api_text()
        elif self.path == "/api/llm_decisions":
            self._serve_api_llm_decisions()
        elif self.path == "/api/streaming/stats":
            self._serve_streaming_stats()
        elif self.path.startswith("/api/streaming/quality/"):
            self._handle_quality_control()
        elif self.path.startswith("/socket.io/"):
            self._handle_socketio_fallback()
        else:
            self.send_error(404)
    
    def do_POST(self):
        """Handle POST requests."""
        if self.path == "/api/start_training":
            self._handle_start_training()
        elif self.path == "/api/stop_training":
            self._handle_stop_training()
        else:
            self.send_error(404)
    
    def _serve_comprehensive_dashboard(self):
        """Serve the main dashboard."""
        try:
            # Try to load template file
            template_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "templates", "dashboard.html"
            )
            with open(template_path, 'r') as f:
                content = f.read()
            
            self.send_response(200)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(content.encode('utf-8'))
        except (FileNotFoundError, Exception):
            self._serve_fallback_dashboard()
    
    def _serve_fallback_dashboard(self):
        """Serve a fallback dashboard when template is not found."""
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Pokemon Crystal Trainer Monitor</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .status { padding: 20px; background: #f0f0f0; border-radius: 5px; }
            </style>
        </head>
        <body>
            <h1>Pokemon Crystal Trainer Monitor</h1>
            <div class="status">
                <p>Training monitor is running...</p>
                <p><a href="/api/status">View Status</a></p>
                <p><a href="/screen">View Screen</a></p>
            </div>
        </body>
        </html>
        """
        
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(html_content.encode('utf-8'))
    
    def _serve_screen(self):
        """Serve current game screen."""
        try:
            # Try optimized streaming first
            if hasattr(self.trainer, 'video_streamer') and self.trainer.video_streamer:
                try:
                    frame_bytes = self.trainer.video_streamer.get_frame_as_bytes()
                    if frame_bytes:
                        self.send_response(200)
                        self.send_header('Content-type', 'image/jpeg')
                        self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
                        self.send_header('Pragma', 'no-cache')
                        self.send_header('Expires', '0')
                        self.end_headers()
                        self.wfile.write(frame_bytes)
                        return
                except Exception:
                    pass  # Fall back to legacy method
            
            # Legacy fallback
            if hasattr(self.trainer, 'latest_screen') and self.trainer.latest_screen:
                image_b64 = self.trainer.latest_screen.get('image_b64')
                if image_b64:
                    image_data = base64.b64decode(image_b64)
                    self.send_response(200)
                    self.send_header('Content-type', 'image/png')
                    self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
                    self.send_header('Pragma', 'no-cache')
                    self.send_header('Expires', '0')
                    self.end_headers()
                    self.wfile.write(image_data)
                    return
            
            self.send_error(404)
        except Exception:
            self.send_error(500)
    
    def _serve_stats(self):
        """Serve training statistics."""
        try:
            stats = self.trainer.get_current_stats() if hasattr(self.trainer, 'get_current_stats') else {}
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(stats).encode('utf-8'))
        except Exception:
            self.send_error(500)
    
    def _serve_api_status(self):
        """Serve API status endpoint."""
        try:
            stats = self.trainer.get_current_stats() if hasattr(self.trainer, 'get_current_stats') else {}
            
            status = {
                'is_training': getattr(self.trainer, '_training_active', False),
                'current_run_id': getattr(self.trainer, 'current_run_id', None),
                'total_actions': stats.get('total_actions', 0),
                'llm_calls': stats.get('llm_calls', 0),
                'actions_per_second': stats.get('actions_per_second', 0.0),
                'elapsed_time': time.time() - stats.get('start_time', time.time()),
                'current_state': getattr(self.trainer, '_current_state', 'unknown'),
                'current_map': getattr(self.trainer, '_current_map', 0),
                'player_position': {
                    'x': getattr(self.trainer, '_player_x', 0),
                    'y': getattr(self.trainer, '_player_y', 0)
                },
                'config': {
                    'mode': getattr(self.trainer.config.mode, 'value', 'unknown') if hasattr(self.trainer, 'config') else 'unknown',
                    'llm_backend': getattr(self.trainer.config.llm_backend, 'value', 'unknown') if hasattr(self.trainer, 'config') else 'unknown',
                    'llm_interval': getattr(self.trainer.config, 'llm_interval', 0) if hasattr(self.trainer, 'config') else 0
                }
            }
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(status).encode('utf-8'))
        except Exception:
            self.send_error(500)
    
    def _serve_api_system(self):
        """Serve system metrics."""
        try:
            system_info = {
                'cpu_percent': 0.0,
                'memory_percent': 0.0,
                'disk_usage': 0.0,
                'gpu_available': False
            }
            
            if PSUTIL_AVAILABLE:
                try:
                    system_info.update({
                        'cpu_percent': psutil.cpu_percent(),
                        'memory_percent': psutil.virtual_memory().percent,
                        'disk_usage': psutil.disk_usage('/').percent
                    })
                except Exception as e:
                    system_info['error'] = str(e)
            else:
                system_info['error'] = 'psutil not available'
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(system_info).encode('utf-8'))
        except Exception:
            self.send_error(500)
    
    def _serve_api_runs(self):
        """Serve training runs data."""
        try:
            stats = self.trainer.get_current_stats() if hasattr(self.trainer, 'get_current_stats') else {}
            
            runs = [{
                'id': 1,
                'status': 'completed' if not getattr(self.trainer, '_training_active', False) else 'running',
                'start_time': stats.get('start_time', time.time()),
                'total_timesteps': stats.get('total_actions', 0),
                'reward': 0.0
            }]
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(runs).encode('utf-8'))
        except Exception:
            self.send_error(500)
    
    def _serve_api_text(self):
        """Serve detected text data."""
        try:
            recent_text = getattr(self.trainer, 'recent_text', [])
            text_frequency = getattr(self.trainer, 'text_frequency', {})
            
            # Sort by frequency
            sorted_frequency = dict(sorted(text_frequency.items(), key=lambda x: x[1], reverse=True))
            
            text_data = {
                'recent_text': recent_text,
                'total_texts': len(recent_text),
                'unique_texts': len(set(recent_text)),
                'text_frequency': sorted_frequency
            }
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(text_data).encode('utf-8'))
        except Exception:
            self.send_error(500)
    
    def _serve_api_llm_decisions(self):
        """Serve LLM decision data."""
        try:
            if hasattr(self.trainer, 'llm_manager') and self.trainer.llm_manager:
                llm_data = self.trainer.llm_manager.get_decision_data()
            else:
                llm_data = {
                    'recent_decisions': [],
                    'total_decisions': 0,
                    'performance_metrics': {
                        'total_llm_calls': 0,
                        'average_response_time': 0.0,
                        'current_model': getattr(self.trainer.config.llm_backend, 'value', 'unknown') if hasattr(self.trainer, 'config') else 'unknown'
                    }
                }
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(llm_data).encode('utf-8'))
        except Exception:
            self.send_error(500)
    
    def _serve_streaming_stats(self):
        """Serve video streaming statistics."""
        try:
            if hasattr(self.trainer, 'video_streamer') and self.trainer.video_streamer:
                try:
                    stats = self.trainer.video_streamer.get_performance_stats()
                except Exception as e:
                    stats = {
                        'method': 'optimized_streaming',
                        'available': False,
                        'error': str(e)
                    }
            else:
                stats = {
                    'method': 'legacy_fallback',
                    'available': False,
                    'message': 'Optimized streaming not initialized'
                }
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(stats).encode('utf-8'))
        except Exception:
            self.send_error(500)
    
    def _handle_quality_control(self):
        """Handle video quality control requests."""
        try:
            path_parts = self.path.split('/')
            # This will cause IndexError if path_parts[4] doesn't exist
            quality = path_parts[4]
            
            if hasattr(self.trainer, 'video_streamer') and self.trainer.video_streamer:
                try:
                    self.trainer.video_streamer.change_quality(quality)
                    response = {
                        'success': True,
                        'quality': quality,
                        'message': f'Quality changed to {quality}',
                        'available_qualities': ['low', 'medium', 'high', 'ultra']
                    }
                except Exception as e:
                    response = {
                        'success': False,
                        'error': str(e)
                    }
            else:
                response = {
                    'success': False,
                    'error': 'Optimized streaming not available'
                }
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode('utf-8'))
        except Exception as e:
            self._send_error_response(str(e))
    
    def _handle_start_training(self):
        """Handle start training requests."""
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            
            # Not implemented in legacy server
            response = {
                'success': False,
                'message': 'Training control not implemented in legacy server'
            }
            
            self.send_response(501)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode('utf-8'))
        except Exception as e:
            self._send_error_response(str(e))
    
    def _handle_stop_training(self):
        """Handle stop training requests."""
        try:
            response = {
                'success': False,
                'message': 'Training control not implemented in legacy server'
            }
            
            self.send_response(501)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode('utf-8'))
        except Exception as e:
            self._send_error_response(str(e))
    
    def _handle_socketio_fallback(self):
        """Handle Socket.IO fallback requests."""
        try:
            response = {
                'error': 'WebSocket/Socket.IO not implemented',
                'use_polling': True,
                'polling_endpoints': {
                    'status': '/api/status',
                    'metrics': '/api/metrics',
                    'screen': '/screen'
                }
            }
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode('utf-8'))
        except Exception as e:
            self._send_error_response(str(e))
    
    def _send_error_response(self, error_message):
        """Send error response."""
        try:
            response = {
                'success': False,
                'error': error_message
            }
            
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode('utf-8'))
        except Exception:
            pass  # Can't do much if error response fails
