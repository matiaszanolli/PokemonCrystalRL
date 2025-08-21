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
        # Skip auth for public routes
        if request.path in ["/", "/static", "/ws"] or request.path.startswith("/static/"):
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
        while self.is_running:
            with self._ws_lock:
                dead = {ws for ws in self._ws_clients if ws.closed}
                self._ws_clients.difference_update(dead)
            
            await asyncio.sleep(60)  # Run every minute
    
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
        self.is_running = False
        
        # Close all WebSocket connections
        with self._ws_lock:
            for ws in self._ws_clients:
                if not ws.closed:
                    await ws.close(code=aiohttp.WSCloseCode.GOING_AWAY,
                                message="Server shutdown")
        
        # Cancel cleanup task
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Shutdown the application
        await self.app.shutdown()
        await self.app.cleanup()
    
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
            loop.run_until_complete(self.shutdown())
        finally:
            loop.close()
    
    def run_in_thread(self) -> threading.Thread:
        """Run the server in a separate thread and wait for it to be ready."""
        thread = threading.Thread(target=self.run, daemon=True)
        thread.start()
        # Wait for server to be ready
        self._ready.wait(timeout=5.0)
        return thread
