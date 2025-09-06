"""
Main WebMonitor orchestrator

Coordinates all web monitoring components and implements the WebMonitorInterface.
This is the main entry point that users will interact with.
"""

import os
import threading
import socket
import logging
import asyncio
import websockets
import queue
from http.server import HTTPServer
from typing import Dict, Any, Optional, Set

from interfaces.monitoring import WebMonitorInterface, MonitoringStats
from .screen_capture import ScreenCapture
from .http_handler import WebMonitorHandler
from .web_api import WebAPI

logger = logging.getLogger(__name__)


def get_available_port(start_port=8080):
    """Find an available port starting from start_port"""
    for port in range(start_port, start_port + 100):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(('localhost', port))
                return port
        except OSError:
            continue
    return None


async def websocket_handler(websocket, path, web_monitor):
    """WebSocket handler for streaming game screen"""
    web_monitor.ws_clients.add(websocket)
    logger.info(f"📡 WebSocket client connected: {websocket.remote_address} (path: {path})")
    logger.info(f"📡 Total WebSocket clients: {len(web_monitor.ws_clients)}")
    
    try:
        # Send initial frame immediately
        if web_monitor.screen_capture.latest_frame:
            await websocket.send(web_monitor.screen_capture.latest_frame)
        
        # Start frame streaming task
        async def stream_frames():
            while True:
                try:
                    # Check for new frames in queue (non-blocking)
                    if hasattr(web_monitor.screen_capture, '_frame_queue'):
                        try:
                            frame = web_monitor.screen_capture._frame_queue.get_nowait()
                            await websocket.send(frame)
                        except queue.Empty:
                            pass
                    
                    # Wait before checking again (60 FPS max)
                    await asyncio.sleep(1/60)
                except websockets.exceptions.ConnectionClosed:
                    break
                except Exception as e:
                    logger.debug(f"Frame streaming error: {e}")
                    break
        
        # Start streaming and handle incoming messages
        streaming_task = asyncio.create_task(stream_frames())
        
        try:
            # Handle incoming messages
            async for message in websocket:
                if message == "ping":
                    await websocket.send("pong")
        finally:
            streaming_task.cancel()
            
    except websockets.exceptions.ConnectionClosed:
        pass
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        web_monitor.ws_clients.discard(websocket)
        logger.info(f"📡 WebSocket client disconnected")


class WebMonitor(WebMonitorInterface):
    """Web monitoring system for Pokemon Crystal RL training"""
    
    def __init__(self, trainer, port=8080, host='localhost'):
        self.trainer = trainer
        self.host = host
        self.port = self._find_available_port(port)
        self.ws_port = self._find_available_port(port + 1)
        self.server = None
        self.server_thread = None
        self.ws_server = None
        self.ws_thread = None
        self.running = False
        self.ws_clients: Set = set()
        
        # Initialize components
        self.screen_capture = ScreenCapture(getattr(trainer, 'pyboy', None))
        self.web_api = WebAPI(trainer, self.screen_capture)
        
        # Set up HTTP handler references
        WebMonitorHandler.trainer = trainer
        WebMonitorHandler.screen_capture = self.screen_capture
        WebMonitorHandler.web_api = self.web_api
        
        # Link WebSocket clients to screen capture
        self.screen_capture.ws_clients = self.ws_clients
        
        logger.info(f"🌐 Web monitor initialized on {self.host}:{self.port}")
    
    def _find_available_port(self, start_port):
        """Find an available port starting from start_port"""
        port = get_available_port(start_port=start_port)
        if port is None:
            raise RuntimeError(f"Could not find available port starting from {start_port}")
        return port
    
    def start(self) -> bool:
        """Start the monitoring component."""
        if self.running:
            return True
        
        try:
            # Start screen capture if we have a PyBoy instance
            if getattr(self.trainer, 'pyboy', None):
                self.screen_capture.pyboy = self.trainer.pyboy
                self.screen_capture.start()
            
            # Create server
            self.server = HTTPServer((self.host, self.port), WebMonitorHandler)
            self.running = True
            
            # Start HTTP server in daemon thread
            self.server_thread = threading.Thread(
                target=self.server.serve_forever,
                daemon=True
            )
            self.server_thread.start()
            
            # Start WebSocket server in daemon thread
            self.ws_thread = threading.Thread(
                target=self._start_websocket_server,
                daemon=True
            )
            self.ws_thread.start()
            
            logger.info(f"🚀 Web monitor started at http://{self.host}:{self.port}")
            logger.info(f"📡 WebSocket streaming at ws://{self.host}:{self.ws_port}/stream")
            return True
            
        except OSError as e:
            if "Address already in use" in str(e):
                logger.error(f"Port {self.port} already in use. Trying to find alternative...")
                try:
                    # Try to find another port
                    new_port = get_available_port(self.port + 1)
                    if new_port:
                        self.port = new_port
                        self.server = HTTPServer((self.host, self.port), WebMonitorHandler)
                        self.running = True
                        self.server_thread = threading.Thread(
                            target=self.server.serve_forever,
                            daemon=True
                        )
                        self.server_thread.start()
                        logger.info(f"🚀 Web monitor started at http://{self.host}:{self.port}")
                        return True
                except Exception as inner_e:
                    logger.error(f"Failed to start web monitor on alternative port: {inner_e}")
            else:
                logger.error(f"Failed to start web monitor - OS Error: {e}")
            self.running = False
            return False
        except Exception as e:
            logger.error(f"Failed to start web monitor - Unexpected error: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            self.running = False
            return False
    
    def stop(self) -> bool:
        """Stop the monitoring component."""
        if not self.running:
            return True
        
        logger.info("🛑 Stopping web monitor...")
        self.running = False
        
        # Stop screen capture
        self.screen_capture.stop()
        
        # Stop server
        if self.server:
            self.server.shutdown()
            self.server.server_close()
        
        # Wait for thread to finish
        if self.server_thread and self.server_thread.is_alive():
            self.server_thread.join(timeout=5.0)
        
        logger.info("✅ Web monitor stopped")
        return True
    
    def update_stats(self, stats: MonitoringStats) -> None:
        """Update component statistics."""
        # WebMonitor collects stats from trainer directly
        # This can be used to receive stats from external sources if needed
        pass
    
    def get_port(self) -> int:
        """Get the port number the web monitor is running on."""
        return self.port
    
    def broadcast_update(self, update_type: str, data: Dict[str, Any]) -> None:
        """Broadcast an update to connected clients."""
        # For future implementation of real-time updates
        logger.debug(f"Broadcast update: {update_type} - {len(self.ws_clients)} clients")
    
    def add_endpoint(self, endpoint: str, handler_func: callable) -> None:
        """Add a new HTTP endpoint."""
        # For future extensibility - would require modifying WebMonitorHandler
        logger.warning(f"Custom endpoint registration not yet implemented: {endpoint}")
    
    def update_pyboy(self, pyboy):
        """Update PyBoy instance for screen capture"""
        if self.screen_capture:
            # Stop screen capture if it's active
            if self.screen_capture.capture_active:
                self.screen_capture.stop()
            
            # Update PyBoy instance and restart capture
            self.screen_capture.pyboy = pyboy
            self.screen_capture.start()
            
            logger.info("📸 PyBoy instance updated for screen capture")
        
        # Update web API trainer reference
        self.web_api.update_trainer(self.trainer)
    
    def _start_websocket_server(self):
        """Start WebSocket server for streaming"""
        async def ws_handler(websocket):
            # In newer websockets, path is accessible via websocket.path
            path = getattr(websocket, 'path', '/stream')
            return await websocket_handler(websocket, path, self)
        
        try:
            logger.info(f"📡 Starting WebSocket server on {self.host}:{self.ws_port}")
            
            # Check for existing event loop
            try:
                loop = asyncio.get_running_loop()
                logger.warning("Event loop already running, creating new one")
                loop = None
            except RuntimeError:
                pass  # No running loop, which is what we want
            
            # Create new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Define async function to start server
            async def start_ws_server():
                # For websockets 15.x, we need to use the newer API
                import websockets.asyncio.server
                server = await websockets.asyncio.server.serve(
                    ws_handler, 
                    self.host, 
                    self.ws_port
                )
                logger.info(f"📡 WebSocket server started successfully on port {self.ws_port}")
                return server
            
            # Run the server setup and keep loop running
            self.ws_server = loop.run_until_complete(start_ws_server())
            loop.run_forever()
            
        except Exception as e:
            logger.error(f"WebSocket server error: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
    
    def get_url(self):
        """Get the web monitor URL"""
        return f"http://{self.host}:{self.port}"
    
    def get_stats(self):
        """Get web monitor statistics"""
        return {
            'running': self.running,
            'port': self.port,
            'host': self.host,
            'url': self.get_url(),
            'screen_capture_stats': self.screen_capture.stats if self.screen_capture else {}
        }