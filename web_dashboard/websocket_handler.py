"""
WebSocket handler for real-time screen streaming and live updates.

This module handles WebSocket connections for streaming game screens
and pushing real-time updates to connected clients.
"""

import json
import logging
import asyncio
import time
import base64
from typing import Set, Optional, Any
import websockets
from PIL import Image
from io import BytesIO

logger = logging.getLogger(__name__)


class WebSocketHandler:
    """Handle WebSocket connections for real-time dashboard updates."""

    def __init__(self, trainer=None):
        """Initialize WebSocket handler."""
        self.trainer = trainer
        self.connected_clients: Set[websockets.WebSocketServerProtocol] = set()
        self.latest_screen_data: Optional[bytes] = None
        self.running = False

        logger.info("WebSocket handler initialized")

    async def handle_connection(self, websocket, path):
        """Handle new WebSocket connection."""
        try:
            logger.info(f"New WebSocket connection from {websocket.remote_address}")
            self.connected_clients.add(websocket)

            # Send initial data
            await self._send_initial_data(websocket)

            # Handle messages from client
            async for message in websocket:
                await self._handle_client_message(websocket, message)

        except websockets.exceptions.ConnectionClosed:
            logger.info(f"WebSocket connection closed: {websocket.remote_address}")
        except Exception as e:
            logger.error(f"WebSocket connection error: {e}")
        finally:
            self.connected_clients.discard(websocket)

    async def _handle_client_message(self, websocket, message):
        """Handle incoming message from client."""
        try:
            data = json.loads(message)
            message_type = data.get('type')

            if message_type == 'request_screen':
                await self._send_screen_update(websocket)
            elif message_type == 'request_stats':
                await self._send_stats_update(websocket)
            elif message_type == 'ping':
                await self._send_pong(websocket)
            else:
                logger.warning(f"Unknown message type: {message_type}")

        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON message from {websocket.remote_address}")
        except Exception as e:
            logger.error(f"Error handling client message: {e}")

    async def _send_initial_data(self, websocket):
        """Send initial data to newly connected client."""
        try:
            # Send current screen
            await self._send_screen_update(websocket)

            # Send current stats
            await self._send_stats_update(websocket)

            # Send connection confirmation
            await websocket.send(json.dumps({
                'type': 'connection_established',
                'timestamp': time.time()
            }))

        except Exception as e:
            logger.error(f"Error sending initial data: {e}")

    async def _send_screen_update(self, websocket):
        """Send screen update to client."""
        try:
            screen_data = self._get_current_screen()
            if screen_data:
                message = {
                    'type': 'screen_update',
                    'data': screen_data,
                    'timestamp': time.time()
                }
                await websocket.send(json.dumps(message))

        except Exception as e:
            logger.error(f"Error sending screen update: {e}")

    async def _send_stats_update(self, websocket):
        """Send stats update to client."""
        try:
            # Try both possible attribute names for compatibility
            tracker = None
            if self.trainer:
                if hasattr(self.trainer, 'stats_tracker') and self.trainer.stats_tracker:
                    tracker = self.trainer.stats_tracker
                elif hasattr(self.trainer, 'statistics_tracker') and self.trainer.statistics_tracker:
                    tracker = self.trainer.statistics_tracker

            if tracker:
                stats = tracker.get_current_stats()
                message = {
                    'type': 'stats_update',
                    'data': stats,
                    'timestamp': time.time()
                }
                await websocket.send(json.dumps(message))

        except Exception as e:
            logger.error(f"Error sending stats update: {e}")

    async def _send_pong(self, websocket):
        """Respond to ping with pong."""
        try:
            await websocket.send(json.dumps({
                'type': 'pong',
                'timestamp': time.time()
            }))
        except Exception as e:
            logger.error(f"Error sending pong: {e}")

    def _get_current_screen(self) -> Optional[str]:
        """Get current game screen as base64 encoded image."""
        try:
            if not self.trainer:
                return None

            # Try to get screen from emulation manager
            pyboy_instance = None
            if hasattr(self.trainer, 'emulation_manager') and self.trainer.emulation_manager:
                pyboy_instance = self.trainer.emulation_manager.get_instance()
            elif hasattr(self.trainer, 'pyboy') and self.trainer.pyboy:
                pyboy_instance = self.trainer.pyboy

            if pyboy_instance:
                # Get screen buffer using the correct PyBoy method
                screen_array = pyboy_instance.screen.ndarray

                # Convert to PIL Image
                if screen_array.shape[-1] == 4:  # RGBA
                    image = Image.fromarray(screen_array, 'RGBA')
                    image = image.convert('RGB')  # Convert to RGB for better compatibility
                else:
                    image = Image.fromarray(screen_array, 'RGB')

                # Scale up for better visibility (optional)
                image = image.resize((320, 288), Image.NEAREST)

                # Convert to base64
                buffer = BytesIO()
                image.save(buffer, format='PNG')
                image_data = base64.b64encode(buffer.getvalue()).decode('utf-8')

                # Store for HTTP endpoint
                self.latest_screen_data = buffer.getvalue()

                return f"data:image/png;base64,{image_data}"

        except Exception as e:
            logger.warning(f"Could not capture screen: {e}")

        return None

    def update_screen_for_http(self):
        """Update screen data for HTTP API endpoint (called externally)."""
        try:
            self._get_current_screen()
        except Exception as e:
            logger.error(f"Error updating screen for HTTP: {e}")

    def get_latest_screen(self) -> Optional[bytes]:
        """Get latest screen data for HTTP endpoint."""
        return self.latest_screen_data

    async def broadcast_update(self, message_type: str, data: Any):
        """Broadcast update to all connected clients."""
        if not self.connected_clients:
            return

        message = {
            'type': message_type,
            'data': data,
            'timestamp': time.time()
        }

        # Send to all connected clients
        disconnected_clients = set()
        for client in self.connected_clients:
            try:
                await client.send(json.dumps(message))
            except websockets.exceptions.ConnectionClosed:
                disconnected_clients.add(client)
            except Exception as e:
                logger.error(f"Error broadcasting to client: {e}")
                disconnected_clients.add(client)

        # Remove disconnected clients
        self.connected_clients -= disconnected_clients

    def get_connection_count(self) -> int:
        """Get number of connected clients."""
        return len(self.connected_clients)

    def start_background_updates(self, screen_interval: float = 0.1, stats_interval: float = 2.0):
        """Start background tasks for regular updates."""
        async def screen_update_task():
            while self.running:
                try:
                    screen_data = self._get_current_screen()
                    if screen_data and self.connected_clients:
                        await self.broadcast_update('screen_update', screen_data)
                    await asyncio.sleep(screen_interval)
                except Exception as e:
                    logger.error(f"Screen update task error: {e}")
                    await asyncio.sleep(1.0)

        async def stats_update_task():
            while self.running:
                try:
                    # Try both possible attribute names for compatibility
                    tracker = None
                    if self.trainer:
                        if hasattr(self.trainer, 'stats_tracker') and self.trainer.stats_tracker:
                            tracker = self.trainer.stats_tracker
                        elif hasattr(self.trainer, 'statistics_tracker') and self.trainer.statistics_tracker:
                            tracker = self.trainer.statistics_tracker

                    if tracker and self.connected_clients:
                        stats = tracker.get_current_stats()
                        await self.broadcast_update('stats_update', stats)
                    await asyncio.sleep(stats_interval)
                except Exception as e:
                    logger.error(f"Stats update task error: {e}")
                    await asyncio.sleep(1.0)

        self.running = True

        # Start background tasks
        asyncio.create_task(screen_update_task())
        asyncio.create_task(stats_update_task())

        logger.info("Background update tasks started")

    def stop_background_updates(self):
        """Stop background update tasks."""
        self.running = False
        logger.info("Background update tasks stopped")