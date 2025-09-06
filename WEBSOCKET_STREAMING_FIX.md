# WebSocket Live Streaming Fix - Implementation Plan

## 🎯 **Current Issues**

### **Grey Screen Problem:**
- ❌ Live game screen displays grey instead of actual game footage
- ❌ WebSocket screen streaming not working properly
- ❌ Image data not being transmitted or processed correctly

### **Game State Stats Not Updating:**
- ❌ Statistics panel not receiving real-time updates via WebSocket
- ❌ Game state data (coordinates, items, etc.) not streaming
- ❌ Memory debug information static

## 🔍 **Diagnostic Plan**

### **1. WebSocket Connection Status**
First, let's verify the WebSocket connection is working:

```bash
# Check if WebSocket server is running
netstat -tulpn | grep 8080

# Test WebSocket connection with wscat (if available)
wscat -c ws://localhost:8080/ws

# Alternative: Test with curl for WebSocket upgrade
curl -v -H "Upgrade: websocket" -H "Connection: Upgrade" \
     -H "Sec-WebSocket-Key: x3JJHMbDL1EzLkh9GBhXDw==" \
     -H "Sec-WebSocket-Version: 13" \
     http://localhost:8080/ws
```

### **2. WebSocket Message Flow Analysis**
Check the message flow in browser developer tools:
```javascript
// Console commands to test WebSocket in browser
const ws = new WebSocket('ws://localhost:8080/ws');
ws.onopen = () => console.log('WebSocket connected');
ws.onmessage = (event) => console.log('Received:', event.data);
ws.onerror = (error) => console.log('WebSocket error:', error);
ws.onclose = (event) => console.log('WebSocket closed:', event.code, event.reason);
```

## 🛠 **Implementation Fixes**

### **Phase 1: WebSocket Server-Side Fixes**

#### **1.1 WebSocket Handler Implementation**
**File:** `core/web_monitor.py`

```python
import websockets
import asyncio
import json
import base64
from threading import Thread
import time

class WebSocketHandler:
    def __init__(self, trainer, port=8081):
        self.trainer = trainer
        self.port = port
        self.connected_clients = set()
        self.server = None
        self.server_thread = None
        
    async def handle_websocket(self, websocket, path):
        """Handle WebSocket connections"""
        self.connected_clients.add(websocket)
        logger.info(f"WebSocket client connected: {websocket.remote_address}")
        
        try:
            await websocket.send(json.dumps({
                'type': 'connection',
                'status': 'connected',
                'message': 'Welcome to Pokemon Crystal RL Training Stream'
            }))
            
            # Keep connection alive and handle incoming messages
            async for message in websocket:
                try:
                    data = json.loads(message)
                    await self.handle_client_message(websocket, data)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON from client: {message}")
                    
        except websockets.exceptions.ConnectionClosed:
            logger.info("WebSocket client disconnected")
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
        finally:
            self.connected_clients.discard(websocket)
            
    async def handle_client_message(self, websocket, data):
        """Handle messages from clients"""
        msg_type = data.get('type')
        
        if msg_type == 'request_screen':
            await self.send_current_screen(websocket)
        elif msg_type == 'request_stats':
            await self.send_current_stats(websocket)
        elif msg_type == 'ping':
            await websocket.send(json.dumps({'type': 'pong'}))
            
    async def send_current_screen(self, websocket=None):
        """Send current screen to client(s)"""
        try:
            # Get screen from trainer
            screen_data = self.get_latest_screen_data()
            
            if screen_data:
                message = {
                    'type': 'screen_update',
                    'timestamp': time.time(),
                    'image_data': screen_data['image_b64'],
                    'frame_id': screen_data.get('frame_id', 0)
                }
                
                if websocket:
                    await websocket.send(json.dumps(message))
                else:
                    await self.broadcast(message)
            else:
                # Send placeholder/error screen
                await self.send_placeholder_screen(websocket)
                
        except Exception as e:
            logger.error(f"Error sending screen: {e}")
            
    def get_latest_screen_data(self):
        """Get latest screen from trainer with proper format conversion"""
        try:
            if not self.trainer or not hasattr(self.trainer, 'latest_screen'):
                return None
                
            latest = self.trainer.latest_screen
            if not latest:
                return None
                
            # Check if we have base64 encoded image
            if 'image_b64' in latest:
                return latest
                
            # If we only have raw image, encode it
            if 'image' in latest:
                import cv2
                import numpy as np
                
                image = latest['image']
                if isinstance(image, np.ndarray):
                    # Convert to BGR for cv2
                    if len(image.shape) == 3 and image.shape[2] == 3:
                        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    else:
                        image_bgr = image
                        
                    # Encode as JPEG
                    ret, jpg_data = cv2.imencode('.jpg', image_bgr, 
                                               [cv2.IMWRITE_JPEG_QUALITY, 85])
                    
                    if ret and jpg_data is not None:
                        image_b64 = base64.b64encode(jpg_data).decode('utf-8')
                        latest['image_b64'] = image_b64
                        return latest
                        
        except Exception as e:
            logger.error(f"Screen data conversion error: {e}")
            
        return None
        
    async def send_placeholder_screen(self, websocket=None):
        """Send a placeholder screen when no game screen available"""
        try:
            from PIL import Image, ImageDraw, ImageFont
            import io
            
            # Create placeholder image
            img = Image.new('RGB', (320, 288), color='gray')
            draw = ImageDraw.Draw(img)
            
            # Add text
            try:
                font = ImageFont.load_default()
            except:
                font = None
                
            text = "No Game Screen Available"
            text_bbox = draw.textbbox((0, 0), text, font=font) if font else (0, 0, 200, 20)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            x = (320 - text_width) // 2
            y = (288 - text_height) // 2
            
            draw.text((x, y), text, fill='white', font=font)
            
            # Convert to base64
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=85)
            image_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            message = {
                'type': 'screen_update',
                'timestamp': time.time(),
                'image_data': image_b64,
                'frame_id': -1,
                'placeholder': True
            }
            
            if websocket:
                await websocket.send(json.dumps(message))
            else:
                await self.broadcast(message)
                
        except Exception as e:
            logger.error(f"Error creating placeholder screen: {e}")
            
    async def send_current_stats(self, websocket=None):
        """Send current game statistics"""
        try:
            stats = self.get_trainer_stats()
            game_state = self.get_game_state()
            
            message = {
                'type': 'stats_update',
                'timestamp': time.time(),
                'stats': stats,
                'game_state': game_state
            }
            
            if websocket:
                await websocket.send(json.dumps(message))
            else:
                await self.broadcast(message)
                
        except Exception as e:
            logger.error(f"Error sending stats: {e}")
            
    def get_trainer_stats(self):
        """Get comprehensive trainer statistics"""
        try:
            if not self.trainer:
                return {}
                
            stats = getattr(self.trainer, 'stats', {})
            
            # Ensure all expected fields are present
            default_stats = {
                'total_actions': 0,
                'actions_per_second': 0.0,
                'uptime_seconds': 0.0,
                'llm_calls': 0,
                'llm_avg_time': 0.0,
                'total_reward': 0.0,
                'queue_size': 0,
                'capture_active': False
            }
            
            default_stats.update(stats)
            
            # Add real-time data
            if hasattr(self.trainer, 'screen_queue'):
                default_stats['queue_size'] = self.trainer.screen_queue.qsize()
                
            if hasattr(self.trainer, 'capture_active'):
                default_stats['capture_active'] = self.trainer.capture_active
                
            return default_stats
            
        except Exception as e:
            logger.error(f"Error getting trainer stats: {e}")
            return {}
            
    def get_game_state(self):
        """Get current Pokemon Crystal game state"""
        try:
            if not self.trainer or not hasattr(self.trainer, 'pyboy'):
                return {}
                
            # Try to get game state from memory reader if available
            if hasattr(self.trainer, 'memory_reader'):
                return self.trainer.memory_reader.read_game_state()
                
            # Fallback: basic state info
            pyboy = self.trainer.pyboy
            if pyboy:
                return {
                    'frame_count': getattr(pyboy, 'frame_count', 0),
                    'running': True
                }
                
        except Exception as e:
            logger.error(f"Error getting game state: {e}")
            
        return {'running': False}
        
    async def broadcast(self, message):
        """Broadcast message to all connected clients"""
        if not self.connected_clients:
            return
            
        # Convert message to JSON if it's not already a string
        if isinstance(message, dict):
            message = json.dumps(message)
            
        disconnected = set()
        
        for client in self.connected_clients:
            try:
                await client.send(message)
            except websockets.exceptions.ConnectionClosed:
                disconnected.add(client)
            except Exception as e:
                logger.error(f"Error broadcasting to client: {e}")
                disconnected.add(client)
                
        # Remove disconnected clients
        self.connected_clients -= disconnected
        
    def start_server(self):
        """Start the WebSocket server"""
        def run_server():
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                self.server = websockets.serve(
                    self.handle_websocket,
                    "localhost",
                    self.port,
                    ping_interval=30,
                    ping_timeout=10
                )
                
                logger.info(f"WebSocket server starting on ws://localhost:{self.port}")
                
                loop.run_until_complete(self.server)
                loop.run_forever()
                
            except Exception as e:
                logger.error(f"WebSocket server error: {e}")
                
        self.server_thread = Thread(target=run_server, daemon=True)
        self.server_thread.start()
        
        # Start periodic broadcasting
        self.start_periodic_updates()
        
    def start_periodic_updates(self):
        """Start periodic screen and stats updates"""
        def periodic_update():
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                async def update_loop():
                    while True:
                        try:
                            if self.connected_clients:
                                # Send screen update every 100ms (10 FPS)
                                await self.send_current_screen()
                                
                                # Send stats update every 1 second
                                if int(time.time()) % 1 == 0:
                                    await self.send_current_stats()
                                    
                            await asyncio.sleep(0.1)
                            
                        except Exception as e:
                            logger.error(f"Periodic update error: {e}")
                            await asyncio.sleep(1)
                            
                loop.run_until_complete(update_loop())
                
            except Exception as e:
                logger.error(f"Periodic update thread error: {e}")
                
        update_thread = Thread(target=periodic_update, daemon=True)
        update_thread.start()
        
    def stop_server(self):
        """Stop the WebSocket server"""
        try:
            if self.server:
                self.server.close()
                
        except Exception as e:
            logger.error(f"Error stopping WebSocket server: {e}")
```

#### **1.2 Integration with Existing WebMonitor**
**File:** `core/web_monitor.py` (modifications)

```python
class WebMonitor:
    def __init__(self, trainer, port=8080, host='localhost'):
        # ... existing init code ...
        
        # Add WebSocket handler
        self.websocket_handler = WebSocketHandler(trainer, port + 1)  # Use port 8081
        
    def start(self):
        """Start both HTTP and WebSocket servers"""
        # Start existing HTTP server
        success = self._start_http_server()
        
        # Start WebSocket server
        if success:
            self.websocket_handler.start_server()
            logger.info(f"WebSocket server started on ws://localhost:{self.websocket_handler.port}")
            
        return success
        
    def stop(self):
        """Stop both HTTP and WebSocket servers"""
        # Stop WebSocket server
        if hasattr(self, 'websocket_handler'):
            self.websocket_handler.stop_server()
            
        # Stop existing HTTP server
        self._stop_http_server()
```

### **Phase 2: Frontend WebSocket Client Fixes**

#### **2.1 JavaScript WebSocket Implementation**
**File:** `core/web_monitor.py` (in dashboard HTML)

```javascript
class WebSocketClient {
    constructor() {
        this.ws = null;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 10;
        this.reconnectDelay = 1000;
        this.isConnected = false;
        
        this.connect();
    }
    
    connect() {
        try {
            const wsUrl = `ws://${window.location.hostname}:8081/ws`;
            console.log('Connecting to WebSocket:', wsUrl);
            
            this.ws = new WebSocket(wsUrl);
            
            this.ws.onopen = (event) => {
                console.log('WebSocket connected');
                this.isConnected = true;
                this.reconnectAttempts = 0;
                this.updateConnectionStatus('Connected');
                
                // Request initial data
                this.requestScreen();
                this.requestStats();
            };
            
            this.ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    this.handleMessage(data);
                } catch (error) {
                    console.error('Error parsing WebSocket message:', error);
                }
            };
            
            this.ws.onclose = (event) => {
                console.log('WebSocket closed:', event.code, event.reason);
                this.isConnected = false;
                this.updateConnectionStatus('Disconnected');
                
                // Attempt to reconnect
                this.attemptReconnect();
            };
            
            this.ws.onerror = (error) => {
                console.error('WebSocket error:', error);
                this.updateConnectionStatus('Error');
            };
            
        } catch (error) {
            console.error('Error creating WebSocket connection:', error);
            this.attemptReconnect();
        }
    }
    
    handleMessage(data) {
        switch (data.type) {
            case 'screen_update':
                this.updateGameScreen(data);
                break;
                
            case 'stats_update':
                this.updateStats(data.stats);
                this.updateGameState(data.game_state);
                break;
                
            case 'connection':
                console.log('Connection message:', data.message);
                break;
                
            case 'pong':
                // Handle ping response
                break;
                
            default:
                console.log('Unknown message type:', data.type);
        }
    }
    
    updateGameScreen(data) {
        try {
            const screenImg = document.getElementById('game-screen-img');
            
            if (!screenImg) {
                console.error('Game screen image element not found');
                return;
            }
            
            if (data.image_data) {
                // Update screen with new image
                screenImg.src = `data:image/jpeg;base64,${data.image_data}`;
                screenImg.style.display = 'block';
                
                // Update frame info if available
                if (data.frame_id >= 0) {
                    const frameInfo = document.getElementById('frame-info');
                    if (frameInfo) {
                        frameInfo.textContent = `Frame: ${data.frame_id}`;
                    }
                }
                
                // Update timestamp
                const timestamp = document.getElementById('screen-timestamp');
                if (timestamp) {
                    timestamp.textContent = new Date(data.timestamp * 1000).toLocaleTimeString();
                }
                
            } else {
                console.warn('No image data in screen update');
            }
            
        } catch (error) {
            console.error('Error updating game screen:', error);
        }
    }
    
    updateStats(stats) {
        try {
            // Update statistics display
            const statElements = {
                'total-actions': stats.total_actions || 0,
                'actions-per-second': (stats.actions_per_second || 0).toFixed(1),
                'llm-decisions': stats.llm_calls || 0,
                'system-reward': (stats.total_reward || 0).toFixed(2),
                'uptime': this.formatUptime(stats.uptime_seconds || 0),
                'queue-size': stats.queue_size || 0
            };
            
            for (const [elementId, value] of Object.entries(statElements)) {
                const element = document.getElementById(elementId);
                if (element) {
                    element.textContent = value;
                }
            }
            
        } catch (error) {
            console.error('Error updating stats:', error);
        }
    }
    
    updateGameState(gameState) {
        try {
            // Update game state information
            const stateElements = {
                'player-x': gameState.PLAYER_X || 0,
                'player-y': gameState.PLAYER_Y || 0,
                'player-map': gameState.PLAYER_MAP || 0,
                'player-level': gameState.PLAYER_LEVEL || 1,
                'badges': gameState.BADGES || 0,
                'money': gameState.MONEY || 0
            };
            
            for (const [elementId, value] of Object.entries(stateElements)) {
                const element = document.getElementById(elementId);
                if (element) {
                    element.textContent = value;
                }
            }
            
        } catch (error) {
            console.error('Error updating game state:', error);
        }
    }
    
    formatUptime(seconds) {
        const hours = Math.floor(seconds / 3600);
        const minutes = Math.floor((seconds % 3600) / 60);
        const secs = Math.floor(seconds % 60);
        
        return `${hours}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    }
    
    updateConnectionStatus(status) {
        const statusElement = document.getElementById('connection-status');
        if (statusElement) {
            statusElement.textContent = status;
            statusElement.className = `status-${status.toLowerCase()}`;
        }
    }
    
    requestScreen() {
        if (this.isConnected) {
            this.send({ type: 'request_screen' });
        }
    }
    
    requestStats() {
        if (this.isConnected) {
            this.send({ type: 'request_stats' });
        }
    }
    
    send(data) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify(data));
        }
    }
    
    attemptReconnect() {
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
            this.reconnectAttempts++;
            console.log(`Attempting to reconnect (${this.reconnectAttempts}/${this.maxReconnectAttempts})...`);
            
            setTimeout(() => {
                this.connect();
            }, this.reconnectDelay * this.reconnectAttempts);
        } else {
            console.error('Max reconnect attempts reached');
            this.updateConnectionStatus('Failed');
        }
    }
    
    startHeartbeat() {
        setInterval(() => {
            if (this.isConnected) {
                this.send({ type: 'ping' });
            }
        }, 30000); // Ping every 30 seconds
    }
}

// Initialize WebSocket client when page loads
let wsClient;
document.addEventListener('DOMContentLoaded', () => {
    wsClient = new WebSocketClient();
    wsClient.startHeartbeat();
});
```

#### **2.2 HTML Structure Updates**
**File:** `core/web_monitor.py` (dashboard HTML modifications)

```html
<!-- Update game screen section -->
<div class="panel">
    <h2>🎮 Live Game Screen</h2>
    <div class="game-screen">
        <img id="game-screen-img" alt="Game screen" style="display: none;" />
        <div id="screen-status">Connecting...</div>
        <div class="screen-info">
            <span id="frame-info">Frame: --</span>
            <span id="screen-timestamp">--:--:--</span>
        </div>
    </div>
</div>

<!-- Add connection status -->
<div class="panel">
    <h2>📡 Connection Status</h2>
    <div class="connection-info">
        <span class="status-indicator" id="connection-status">Connecting...</span>
    </div>
</div>

<!-- Update stats section with specific IDs -->
<div class="panel">
    <h2>📊 Training Statistics</h2>
    <div class="stat-grid">
        <div class="stat-item">
            <div class="stat-label">Total Actions</div>
            <div class="stat-value" id="total-actions">0</div>
        </div>
        <div class="stat-item">
            <div class="stat-label">Actions/Second</div>
            <div class="stat-value" id="actions-per-second">0.0</div>
        </div>
        <div class="stat-item">
            <div class="stat-label">LLM Decisions</div>
            <div class="stat-value" id="llm-decisions">0</div>
        </div>
        <div class="stat-item">
            <div class="stat-label">System Reward</div>
            <div class="stat-value" id="system-reward">0.00</div>
        </div>
    </div>
</div>

<!-- Game state section -->
<div class="panel">
    <h2>🎯 Game State</h2>
    <div class="game-state-info">
        <div class="state-item">
            <span class="state-label">Position:</span>
            <span id="player-x">0</span>, <span id="player-y">0</span>
        </div>
        <div class="state-item">
            <span class="state-label">Map:</span>
            <span id="player-map">0</span>
        </div>
        <div class="state-item">
            <span class="state-label">Level:</span>
            <span id="player-level">1</span>
        </div>
        <div class="state-item">
            <span class="state-label">Badges:</span>
            <span id="badges">0</span>/16
        </div>
    </div>
</div>
```

## 🧪 **Testing & Validation**

### **Phase 3: Debugging and Testing**

#### **3.1 WebSocket Connection Testing**
```bash
# Test WebSocket endpoint
echo "Testing WebSocket connection..."
wscat -c ws://localhost:8081/ws

# Or use Python test script
python3 -c "
import asyncio
import websockets
import json

async def test_websocket():
    try:
        uri = 'ws://localhost:8081/ws'
        async with websockets.connect(uri) as websocket:
            print('Connected to WebSocket')
            
            # Request screen
            await websocket.send(json.dumps({'type': 'request_screen'}))
            
            # Listen for messages
            for i in range(5):
                message = await websocket.recv()
                data = json.loads(message)
                print(f'Received: {data[\"type\"]}')
                
    except Exception as e:
        print(f'Error: {e}')

asyncio.run(test_websocket())
"
```

#### **3.2 Screen Data Pipeline Testing**
```python
# Test screen capture pipeline
def test_screen_pipeline(trainer):
    print("Testing screen capture pipeline...")
    
    # Check PyBoy availability
    if not trainer.pyboy:
        print("❌ PyBoy not available")
        return
        
    # Test screen capture
    screen = trainer._simple_screenshot_capture()
    if screen is None:
        print("❌ Screen capture failed")
        return
    
    print(f"✅ Screen captured: {screen.shape}")
    
    # Test latest_screen data
    if hasattr(trainer, 'latest_screen') and trainer.latest_screen:
        latest = trainer.latest_screen
        print(f"✅ Latest screen available: {list(latest.keys())}")
        
        if 'image_b64' in latest:
            print(f"✅ Base64 image available: {len(latest['image_b64'])} chars")
        else:
            print("❌ No base64 image in latest_screen")
    else:
        print("❌ No latest_screen data")
```

## 🎯 **Expected Results**

After implementing these fixes:

✅ **Live Game Screen**: Real Pokemon Crystal game footage streaming at 10 FPS  
✅ **Real-Time Stats**: Training statistics updating every second  
✅ **Game State Data**: Player position, level, badges, etc. updating live  
✅ **WebSocket Connection**: Stable bidirectional communication  
✅ **Error Handling**: Graceful connection recovery and error reporting  
✅ **Performance**: Smooth streaming without browser lag  

## 📊 **Implementation Priority**

1. **Critical**: WebSocket server implementation and screen data pipeline
2. **High**: Frontend JavaScript WebSocket client and message handling  
3. **Medium**: Game state data integration and error handling
4. **Low**: Connection monitoring and diagnostics

This focused plan should resolve the grey screen issue and restore live game state updates via WebSocket streaming!
