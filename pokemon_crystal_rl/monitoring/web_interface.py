#!/usr/bin/env python3
"""
Modern Web Interface with WebSocket Support

This module provides a responsive, real-time web UI for monitoring training
sessions with proper error handling and fallback mechanisms.
"""

import asyncio
import json
import time
import threading
import logging
import os
from typing import Dict, Any, List, Optional, Set
from datetime import datetime
from dataclasses import asdict

from fastapi import FastAPI, WebSocketDisconnect, HTTPException
from fastapi.websockets import WebSocket
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

try:
    import websockets
    from websockets.server import WebSocketServerProtocol
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    WebSocketServerProtocol = None

try:
    from .data_bus import get_data_bus, DataType, DataMessage
    from .stats_collector import StatsCollector
    from .game_streamer import GameStreamComponent
except ImportError:
    from data_bus import get_data_bus, DataType, DataMessage
    from stats_collector import StatsCollector
    from game_streamer import GameStreamComponent


class WebInterface:
    """
    Modern web interface with real-time updates
    
    Features:
    - FastAPI backend with WebSocket support
    - Real-time data streaming
    - Game screen streaming
    - Statistics dashboard
    - Error handling and fallbacks
    """
    
    def __init__(self, 
                 host: str = "localhost",
                 port: int = 8080,
                 enable_websockets: bool = True,
                 enable_data_bus: bool = True):
        
        self.host = host
        self.port = port
        self.enable_websockets = enable_websockets and WEBSOCKETS_AVAILABLE
        self.enable_data_bus = enable_data_bus
        
        # FastAPI app
        self.app = FastAPI(title="Pokemon Crystal RL Training Monitor")
        self.server = None
        self.server_thread: Optional[threading.Thread] = None
        self.server_active = False
        
        # WebSocket connections
        self.websocket_connections: Set[WebSocket] = set()
        self.connection_lock = threading.Lock()
        
        # Component references
        self.stats_collector: Optional[StatsCollector] = None
        self.game_streamer: Optional[GameStreamComponent] = None
        self.data_bus = get_data_bus() if enable_data_bus else None
        
        # Setup logging
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Initialize routes if FastAPI is available
        self._setup_routes()
        
        # Register with data bus
        if self.data_bus:
            self.data_bus.register_component("web_interface", {
                "type": "web_ui",
                "host": host,
                "port": port,
                "websockets_enabled": self.enable_websockets
            })
            
            # Subscribe to all relevant data types
            self.data_bus.subscribe(DataType.TRAINING_STATS, self._handle_training_stats, "web_interface")
            self.data_bus.subscribe(DataType.GAME_STATE, self._handle_game_state, "web_interface")
            self.data_bus.subscribe(DataType.ACTION_TAKEN, self._handle_action_taken, "web_interface")
            self.data_bus.subscribe(DataType.LLM_DECISION, self._handle_llm_decision, "web_interface")
            self.data_bus.subscribe(DataType.ERROR_EVENT, self._handle_error_event, "web_interface")
            self.data_bus.subscribe(DataType.GAME_SCREEN, self._handle_game_screen, "web_interface")
        
        self.logger.info("🌐 WebInterface initialized")
    
    def set_stats_collector(self, stats_collector: StatsCollector) -> None:
        """Set the stats collector reference"""
        self.stats_collector = stats_collector
        self.logger.info("📊 Stats collector connected")
    
    def set_game_streamer(self, game_streamer: GameStreamComponent) -> None:
        """Set the game streamer reference"""
        self.game_streamer = game_streamer
        self.logger.info("🎮 Game streamer connected")
    
    def start_server(self) -> bool:
        """Start the web server"""
        if self.server_active:
            self.logger.warning("Web server already active")
            return True
        
        try:
            self.server_active = True
            self.server_thread = threading.Thread(
                target=self._run_server,
                daemon=True,
                name="WebInterface"
            )
            self.server_thread.start()
            
            # Wait a moment to ensure server starts
            time.sleep(1.0)
            
            self.logger.info(f"🚀 Web server started at http://{self.host}:{self.port}")
            return True
            
        except Exception as e:
            self.server_active = False
            self.logger.error(f"Failed to start web server: {e}")
            return False
    
    def stop_server(self) -> None:
        """Stop the web server"""
        if not self.server_active:
            return
        
        self.server_active = False
        
        # Close all WebSocket connections
        with self.connection_lock:
            for connection in self.websocket_connections.copy():
                try:
                    asyncio.create_task(connection.close())
                except Exception:
                    pass
            self.websocket_connections.clear()
        
        # Stop server thread
        if self.server_thread and self.server_thread.is_alive():
            self.server_thread.join(timeout=5.0)
            if self.server_thread.is_alive():
                self.logger.warning("Web server thread did not shut down cleanly")
        
        self.logger.info("🛑 Web server stopped")
    
    def broadcast_to_websockets(self, message: Dict[str, Any]) -> None:
        """Broadcast a message to all connected WebSocket clients"""
        if not self.enable_websockets:
            return
        
        with self.connection_lock:
            disconnected_connections = set()
            
            for connection in self.websocket_connections.copy():
                try:
                    # Create async task to send message
                    asyncio.create_task(connection.send_text(json.dumps(message)))
                except Exception as e:
                    self.logger.warning(f"Failed to send WebSocket message: {e}")
                    disconnected_connections.add(connection)
            
            # Remove disconnected connections
            self.websocket_connections -= disconnected_connections
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data"""
        current_time = time.time()
        
        # Base dashboard data
        dashboard_data = {
            "timestamp": current_time,
            "server_info": {
                "host": self.host,
                "port": self.port,
                "websockets_enabled": self.enable_websockets,
                "active_connections": len(self.websocket_connections)
            }
        }
        
        # Add stats if available
        if self.stats_collector:
            try:
                dashboard_data["stats"] = self.stats_collector.get_metrics_summary()
                dashboard_data["stats_performance"] = self.stats_collector.get_performance_stats()
            except Exception as e:
                self.logger.warning(f"Failed to get stats data: {e}")
                dashboard_data["stats"] = {}
        
        # Add game stream info if available
        if self.game_streamer:
            try:
                dashboard_data["game_stream"] = {
                    "is_streaming": self.game_streamer.is_healthy(),
                    "performance": self.game_streamer.get_performance_stats(),
                    "frame_info": self.game_streamer.get_frame_info()
                }
            except Exception as e:
                self.logger.warning(f"Failed to get game stream data: {e}")
                dashboard_data["game_stream"] = {}
        
        # Add data bus info if available
        if self.data_bus:
            try:
                dashboard_data["data_bus"] = {
                    "performance": self.data_bus.get_performance_stats(),
                    "components": self.data_bus.get_component_status()
                }
            except Exception as e:
                self.logger.warning(f"Failed to get data bus info: {e}")
                dashboard_data["data_bus"] = {}
        
        return dashboard_data
    
    def shutdown(self) -> None:
        """Clean shutdown of the web interface"""
        self.logger.info("🛑 Shutting down WebInterface")
        
        # Stop server
        self.stop_server()
        
        # Notify data bus
        if self.data_bus:
            self.data_bus.publish(
                DataType.COMPONENT_STATUS,
                {"component": "web_interface", "status": "shutdown"},
                "web_interface"
            )
        
        self.logger.info("✅ WebInterface shutdown complete")
    
    # Internal methods
    
    def _setup_routes(self) -> None:
        """Setup FastAPI routes"""
        
        @self.app.get("/", response_class=HTMLResponse)
        async def dashboard():
            """Serve the main dashboard"""
            return self._get_dashboard_html()
        
        @self.app.get("/api/dashboard")
        async def api_dashboard():
            """API endpoint for dashboard data"""
            return self.get_dashboard_data()
        
        @self.app.get("/api/stats")
        async def api_stats():
            """API endpoint for statistics"""
            if self.stats_collector:
                return self.stats_collector.get_metrics_summary()
            else:
                raise HTTPException(status_code=503, detail="Stats collector not available")
        
        @self.app.get("/api/game_screen")
        async def api_game_screen():
            """API endpoint for game screen"""
            if self.game_streamer:
                try:
                    frame_data = self.game_streamer.get_latest_frame("base64_jpeg")
                    if frame_data:
                        return {"frame": frame_data, "timestamp": time.time()}
                    else:
                        raise HTTPException(status_code=503, detail="No game frame available")
                except Exception as e:
                    raise HTTPException(status_code=500, detail=str(e))
            else:
                raise HTTPException(status_code=503, detail="Game streamer not available")
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time updates"""
            await self._handle_websocket_connection(websocket)
    
    def _run_server(self) -> None:
        """Run the server in a separate thread"""
        try:
            # Configure uvicorn to be quiet
            config = uvicorn.Config(
                app=self.app,
                host=self.host,
                port=self.port,
                log_level="warning",
                access_log=False
            )
            server = uvicorn.Server(config)
            
            # Run server
            asyncio.run(server.serve())
            
        except Exception as e:
            self.logger.error(f"Web server error: {e}")
        finally:
            self.server_active = False
    
    async def _handle_websocket_connection(self, websocket: WebSocket) -> None:
        """Handle a WebSocket connection"""
        await websocket.accept()
        
        with self.connection_lock:
            self.websocket_connections.add(websocket)
        
        self.logger.info(f"WebSocket connection established (total: {len(self.websocket_connections)})")
        
        try:
            # Send initial dashboard data
            dashboard_data = self.get_dashboard_data()
            await websocket.send_text(json.dumps({
                "type": "dashboard_data",
                "data": dashboard_data
            }))
            
            # Keep connection alive and handle incoming messages
            while self.server_active:
                try:
                    # Wait for messages with timeout
                    message = await asyncio.wait_for(websocket.receive_text(), timeout=1.0)
                    
                    # Handle incoming message
                    try:
                        msg_data = json.loads(message)
                        await self._handle_websocket_message(websocket, msg_data)
                    except json.JSONDecodeError:
                        await websocket.send_text(json.dumps({
                            "type": "error",
                            "message": "Invalid JSON message"
                        }))
                        
                except asyncio.TimeoutError:
                    # Send periodic heartbeat
                    await websocket.send_text(json.dumps({
                        "type": "heartbeat",
                        "timestamp": time.time()
                    }))
                    
                except WebSocketDisconnect:
                    break
                    
        except Exception as e:
            self.logger.warning(f"WebSocket error: {e}")
        
        finally:
            with self.connection_lock:
                self.websocket_connections.discard(websocket)
            
            self.logger.info(f"WebSocket connection closed (remaining: {len(self.websocket_connections)})")
    
    async def _handle_websocket_message(self, websocket: WebSocket, message: Dict[str, Any]) -> None:
        """Handle incoming WebSocket messages"""
        msg_type = message.get("type")
        
        if msg_type == "get_dashboard":
            # Send current dashboard data
            dashboard_data = self.get_dashboard_data()
            await websocket.send_text(json.dumps({
                "type": "dashboard_data",
                "data": dashboard_data
            }))
        
        elif msg_type == "get_game_screen":
            # Send current game screen
            if self.game_streamer:
                try:
                    frame_data = self.game_streamer.get_latest_frame("base64_jpeg")
                    await websocket.send_text(json.dumps({
                        "type": "game_screen",
                        "frame": frame_data,
                        "timestamp": time.time()
                    }))
                except Exception as e:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": f"Failed to get game screen: {e}"
                    }))
        
        else:
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": f"Unknown message type: {msg_type}"
            }))
    
    def _get_dashboard_html(self) -> str:
        """Generate dashboard HTML"""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Pokemon Crystal RL - Training Monitor</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #1a1a1a;
            color: #ffffff;
            line-height: 1.6;
        }
        
        .header {
            text-align: center;
            margin-bottom: 30px;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 10px;
        }
        
        .header h1 {
            margin: 0;
            font-size: 2.5em;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        .status {
            display: inline-block;
            padding: 10px 20px;
            margin-top: 10px;
            border-radius: 25px;
            font-weight: bold;
        }
        
        .status.online { background-color: #4CAF50; }
        .status.offline { background-color: #f44336; }
        
        .dashboard {
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 20px;
            max-width: 1600px;
            margin: 0 auto;
        }
        
        .main-panel {
            display: grid;
            grid-template-rows: auto 1fr;
            gap: 20px;
        }
        
        .game-screen {
            background-color: #2a2a2a;
            border-radius: 10px;
            padding: 20px;
            text-align: center;
            min-height: 300px;
        }
        
        .game-screen img {
            max-width: 100%;
            border: 3px solid #555;
            border-radius: 5px;
            image-rendering: pixelated;
        }
        
        .stats-panel {
            background-color: #2a2a2a;
            border-radius: 10px;
            padding: 20px;
        }
        
        .stat-group {
            margin-bottom: 25px;
            padding: 15px;
            background-color: #333;
            border-radius: 8px;
        }
        
        .stat-group h3 {
            margin: 0 0 15px 0;
            color: #4CAF50;
            border-bottom: 2px solid #4CAF50;
            padding-bottom: 5px;
        }
        
        .stat-item {
            display: flex;
            justify-content: space-between;
            margin: 10px 0;
            padding: 8px;
            background-color: #444;
            border-radius: 5px;
        }
        
        .connection-status {
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 10px 15px;
            border-radius: 5px;
            font-weight: bold;
            z-index: 1000;
        }
        
        .connection-status.connected {
            background-color: #4CAF50;
            color: white;
        }
        
        .connection-status.disconnected {
            background-color: #f44336;
            color: white;
        }
        
        .error {
            color: #ff6b6b;
            font-style: italic;
        }
    </style>
</head>
<body>
    <div class="connection-status" id="connectionStatus">Connecting...</div>
    
    <div class="header">
        <h1>🎮 Pokemon Crystal RL Training Monitor</h1>
        <span class="status" id="trainingStatus">Loading...</span>
    </div>
    
    <div class="dashboard">
        <div class="main-panel">
            <div class="game-screen">
                <h2>🎬 Live Game Screen</h2>
                <div id="gameScreenContainer">
                    <img id="gameScreen" src="" alt="Game Screen" style="display: none;">
                    <div id="gameScreenPlaceholder">Waiting for game data...</div>
                </div>
            </div>
        </div>
        
        <div class="stats-panel">
            <div class="stat-group">
                <h3>📊 Training Stats</h3>
                <div id="trainingStats">Loading...</div>
            </div>
            
            <div class="stat-group">
                <h3>🎮 Game State</h3>
                <div id="gameStats">Loading...</div>
            </div>
            
            <div class="stat-group">
                <h3>⚡ Performance</h3>
                <div id="performanceStats">Loading...</div>
            </div>
        </div>
    </div>

    <script>
        let websocket = null;
        let reconnectInterval = null;
        
        function connectWebSocket() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}/ws`;
            
            try {
                websocket = new WebSocket(wsUrl);
                
                websocket.onopen = function(event) {
                    console.log('WebSocket connected');
                    document.getElementById('connectionStatus').textContent = 'Connected';
                    document.getElementById('connectionStatus').className = 'connection-status connected';
                    
                    if (reconnectInterval) {
                        clearInterval(reconnectInterval);
                        reconnectInterval = null;
                    }
                    
                    // Request initial data
                    websocket.send(JSON.stringify({type: 'get_dashboard'}));
                };
                
                websocket.onmessage = function(event) {
                    try {
                        const message = JSON.parse(event.data);
                        handleWebSocketMessage(message);
                    } catch (e) {
                        console.error('Failed to parse WebSocket message:', e);
                    }
                };
                
                websocket.onclose = function(event) {
                    console.log('WebSocket disconnected');
                    document.getElementById('connectionStatus').textContent = 'Disconnected';
                    document.getElementById('connectionStatus').className = 'connection-status disconnected';
                    
                    // Try to reconnect
                    if (!reconnectInterval) {
                        reconnectInterval = setInterval(() => {
                            console.log('Attempting to reconnect...');
                            connectWebSocket();
                        }, 5000);
                    }
                };
                
                websocket.onerror = function(error) {
                    console.error('WebSocket error:', error);
                };
                
            } catch (e) {
                console.error('Failed to create WebSocket connection:', e);
                // Fallback to HTTP polling
                startHttpPolling();
            }
        }
        
        function handleWebSocketMessage(message) {
            if (message.type === 'dashboard_data') {
                updateDashboard(message.data);
            } else if (message.type === 'game_screen') {
                updateGameScreen(message.frame);
            } else if (message.type === 'heartbeat') {
                // Keep connection alive
            } else if (message.type === 'error') {
                console.error('Server error:', message.message);
            }
        }
        
        function updateDashboard(data) {
            // Update training stats
            const trainingStats = document.getElementById('trainingStats');
            if (data.stats) {
                let statsHtml = '';
                for (const [name, metric] of Object.entries(data.stats)) {
                    if (name.startsWith('training.')) {
                        const displayName = name.replace('training.', '');
                        statsHtml += `<div class="stat-item">
                            <span>${displayName}</span>
                            <span>${metric.current.toFixed(1)}</span>
                        </div>`;
                    }
                }
                trainingStats.innerHTML = statsHtml || 'No training stats available';
            } else {
                trainingStats.innerHTML = '<span class="error">Stats not available</span>';
            }
            
            // Update performance stats
            const performanceStats = document.getElementById('performanceStats');
            if (data.stats_performance) {
                const perf = data.stats_performance;
                performanceStats.innerHTML = `
                    <div class="stat-item">
                        <span>Active Metrics</span>
                        <span>${perf.active_metrics || 0}</span>
                    </div>
                    <div class="stat-item">
                        <span>Collection Rate</span>
                        <span>${((perf.collection_count || 0) / 60).toFixed(1)}/min</span>
                    </div>
                    <div class="stat-item">
                        <span>Error Rate</span>
                        <span>${((perf.error_rate || 0) * 100).toFixed(2)}%</span>
                    </div>
                `;
            } else {
                performanceStats.innerHTML = '<span class="error">Performance data not available</span>';
            }
        }
        
        function updateGameScreen(frameData) {
            if (frameData) {
                const gameScreen = document.getElementById('gameScreen');
                const placeholder = document.getElementById('gameScreenPlaceholder');
                
                gameScreen.src = `data:image/jpeg;base64,${frameData}`;
                gameScreen.style.display = 'block';
                placeholder.style.display = 'none';
            }
        }
        
        function startHttpPolling() {
            console.log('Starting HTTP polling fallback');
            setInterval(async () => {
                try {
                    const response = await fetch('/api/dashboard');
                    const data = await response.json();
                    updateDashboard(data);
                    
                    // Also try to get game screen
                    try {
                        const screenResponse = await fetch('/api/game_screen');
                        const screenData = await screenResponse.json();
                        updateGameScreen(screenData.frame);
                    } catch (e) {
                        // Game screen not available
                    }
                    
                } catch (e) {
                    console.error('HTTP polling error:', e);
                }
            }, 2000);
        }
        
        // Initialize connection
        connectWebSocket();
        
        // Periodic game screen updates via WebSocket
        setInterval(() => {
            if (websocket && websocket.readyState === WebSocket.OPEN) {
                websocket.send(JSON.stringify({type: 'get_game_screen'}));
            }
        }, 1000);
    </script>
</body>
</html>
        """
    
    # Data bus event handlers
    
    def _handle_training_stats(self, message: DataMessage) -> None:
        """Handle training statistics updates"""
        self.broadcast_to_websockets({
            "type": "training_update",
            "data": message.data,
            "timestamp": message.timestamp
        })
    
    def _handle_game_state(self, message: DataMessage) -> None:
        """Handle game state changes"""
        self.broadcast_to_websockets({
            "type": "game_state_change",
            "data": message.data,
            "timestamp": message.timestamp
        })
    
    def _handle_action_taken(self, message: DataMessage) -> None:
        """Handle action events"""
        self.broadcast_to_websockets({
            "type": "action_taken",
            "data": message.data,
            "timestamp": message.timestamp
        })
    
    def _handle_llm_decision(self, message: DataMessage) -> None:
        """Handle LLM decision events"""
        self.broadcast_to_websockets({
            "type": "llm_decision",
            "data": message.data,
            "timestamp": message.timestamp
        })
    
    def _handle_error_event(self, message: DataMessage) -> None:
        """Handle error events"""
        self.broadcast_to_websockets({
            "type": "error_event",
            "data": message.data,
            "timestamp": message.timestamp
        })
    
    def _handle_game_screen(self, message: DataMessage) -> None:
        """Handle game screen updates"""
        # Don't broadcast screen data via WebSocket (too much data)
        # Let clients request it when needed
        pass
