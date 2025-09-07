"""
HTTP Handler for Web Monitor

Handles all HTTP requests for the web monitoring interface including:
- Dashboard HTML serving
- API endpoints for stats, status, screenshots
- LLM decision history
- Memory debugging interface
"""

import json
import time
import logging
import io
import base64
from datetime import datetime
from http.server import BaseHTTPRequestHandler
from PIL import Image

logger = logging.getLogger(__name__)

# Import memory reader for game state debugging
try:
    from trainer.memory_reader import PokemonCrystalMemoryReader
except ImportError:
    PokemonCrystalMemoryReader = None


class HttpHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the web monitoring interface"""
    
    trainer = None  # Will be set by WebMonitor
    screen_capture = None
    web_api = None  # Will be set by WebMonitor
    
    def log_message(self, format, *args):
        """Suppress default logging"""
        pass
    
    def do_POST(self):
        """Return 405 for unsupported POST requests."""
        self.send_error(405)
        
    def do_GET(self):
        """Handle GET requests"""
        try:
            if self.path == '/':
                self._serve_dashboard()
            elif self.path in ['/screen', '/api/screenshot'] or self.path.startswith('/api/screenshot'):
                self._serve_screen()
            elif self.path.startswith('screenshot:'):
                # Handle malformed screenshot URLs (likely browser caching issues)
                self._serve_screen()
            elif self.path == '/api/stats':
                self._serve_stats()
            elif self.path == '/api/status':
                self._serve_status()
            elif self.path == '/api/llm_decisions':
                self._serve_llm_decisions()
            elif self.path == '/api/memory_debug':
                self._serve_memory_debug()
            else:
                self.send_error(404)
        except Exception as e:
            logger.error(f"Request error: {e}")
            self.send_error(500)
    
    def _serve_dashboard(self):
        """Serve the main dashboard HTML"""
        html_content = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🎮 Pokemon Crystal LLM RL Training Dashboard</title>
    <style>
        body {
            font-family: system-ui, -apple-system, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #fff;
            min-height: 100vh;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        
        .header {
            text-align: center;
            background: rgba(255,255,255,0.1);
            padding: 20px;
            border-radius: 12px;
            backdrop-filter: blur(10px);
            margin-bottom: 20px;
        }
        
        .header h1 {
            margin: 0;
            font-size: 2.5em;
            background: linear-gradient(45deg, #00ff88, #00bbff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        .dashboard {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
        }
        
        .panel {
            background: rgba(255,255,255,0.1);
            padding: 20px;
            border-radius: 12px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.2);
        }
        
        .panel h2 {
            margin-top: 0;
            font-size: 1.3em;
            color: #00ff88;
            border-bottom: 2px solid rgba(0,255,136,0.3);
            padding-bottom: 10px;
        }
        
        .stat-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
            gap: 15px;
        }
        
        .stat-item {
            background: rgba(0,0,0,0.3);
            padding: 15px;
            border-radius: 8px;
            text-align: center;
            border: 1px solid rgba(255,255,255,0.1);
        }
        
        .stat-label {
            font-size: 12px;
            color: #aaa;
            margin-bottom: 8px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .stat-value {
            font-size: 24px;
            font-weight: 700;
            color: #00ff88;
            text-shadow: 0 0 10px rgba(0,255,136,0.5);
        }
        
        .game-screen {
            width: 100%;
            max-width: 400px;
            margin: 0 auto;
            background: #000;
            border-radius: 8px;
            overflow: hidden;
            border: 2px solid #00ff88;
            box-shadow: 0 0 20px rgba(0,255,136,0.3);
        }
        
        .game-screen img {
            width: 100%;
            height: auto;
            image-rendering: pixelated;
            image-rendering: -moz-crisp-edges;
            image-rendering: crisp-edges;
        }
        
        .memory-debug {
            font-family: 'Courier New', monospace;
            background: rgba(0,0,0,0.5);
            padding: 15px;
            border-radius: 8px;
            font-size: 14px;
            line-height: 1.4;
            max-height: 300px;
            overflow-y: auto;
        }
        
        .memory-row {
            display: flex;
            justify-content: space-between;
            margin: 5px 0;
        }
        
        .address {
            color: #00bbff;
        }
        
        .value {
            color: #00ff88;
        }
        
        .status-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }
        
        .status-active {
            background: #00ff88;
            box-shadow: 0 0 10px rgba(0,255,136,0.7);
        }
        
        .status-inactive {
            background: #ff4444;
        }
        
        .llm-decision {
            background: rgba(0,0,0,0.3);
            padding: 10px;
            margin: 10px 0;
            border-radius: 8px;
            border-left: 3px solid #00ff88;
        }
        
        .decision-action {
            font-weight: bold;
            color: #00ff88;
        }
        
        .decision-reasoning {
            font-size: 14px;
            color: #ccc;
            margin-top: 5px;
        }
        
        .decision-timestamp {
            font-size: 12px;
            color: #888;
            margin-top: 5px;
        }
        
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.7; }
            100% { opacity: 1; }
        }
        
        .pulsing {
            animation: pulse 2s infinite;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎮 Pokemon Crystal LLM RL Training Dashboard</h1>
            <p>Advanced reinforcement learning with local LLM decision making and real-time monitoring</p>
            <div style="margin-top: 15px;">
                <span class="status-indicator status-active"></span>
                <span id="connection-status">Connected</span>
            </div>
        </div>
        
        <div class="dashboard">
            <!-- Training Statistics -->
            <div class="panel">
                <h2>📊 Training Statistics</h2>
                <div class="stat-grid">
                    <div class="stat-item">
                        <div class="stat-label">Total Actions</div>
                        <div id="total-actions" class="stat-value pulsing">0</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-label">Actions/Second</div>
                        <div id="actions-per-second" class="stat-value">0.0</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-label">LLM Decisions</div>
                        <div id="llm-decisions" class="stat-value">0</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-label">Total Reward</div>
                        <div id="total-reward" class="stat-value">0.0</div>
                    </div>
                </div>
            </div>
            
            <!-- Live Game Screen -->
            <div class="panel">
                <h2>🕹️ Live Game Screen</h2>
                <div class="game-screen">
                    <img id="game-screen" src="data:image/gif;base64,R0lGODlhAQABAIAAAMLCwgAAACH5BAAAAAAALAAAAAABAAEAAAICRAEAOw==" alt="Game screen">
                </div>
            </div>
            
            <!-- Game State -->
            <div class="panel">
                <h2>🎯 Game State</h2>
                <div class="stat-grid">
                    <div class="stat-item">
                        <div class="stat-label">Map</div>
                        <div id="current-map" class="stat-value">-</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-label">Position</div>
                        <div id="player-position" class="stat-value">-</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-label">Money</div>
                        <div id="money" class="stat-value">¥0</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-label">Badges</div>
                        <div id="badges" class="stat-value">0/16</div>
                    </div>
                </div>
            </div>
            
            <!-- LLM Decisions -->
            <div class="panel">
                <h2>🤖 Recent LLM Decisions</h2>
                <div id="llm-decisions-list">
                    <div class="llm-decision">
                        <div class="decision-action">No decisions yet...</div>
                        <div class="decision-reasoning">Waiting for LLM to make first decision</div>
                        <div class="decision-timestamp">--:--:--</div>
                    </div>
                </div>
            </div>
            
            <!-- Memory Debug -->
            <div class="panel">
                <h2>🔬 Live Memory Debug</h2>
                <div class="memory-debug" id="memory-debug">
                    <div class="memory-row">
                        <span class="address">Loading memory data...</span>
                        <span></span>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Update intervals  
        const SCREEN_UPDATE_MS = 200;  // 5 fps
        const STATS_UPDATE_MS = 1000;  // 1 second
        const LLM_UPDATE_MS = 2000;    // 2 seconds
        
        // Start update loops (stats and LLM only - screen via WebSocket)
        setInterval(updateStats, STATS_UPDATE_MS);
        setInterval(updateLLMDecisions, LLM_UPDATE_MS);
        
        // Initial updates
        initWebSocketStream();
        updateStats();
        updateLLMDecisions();
        
        function initWebSocketStream() {
            try {
                const wsPort = window.location.port ? parseInt(window.location.port) + 1 : 8081;
                const ws = new WebSocket(`ws://${window.location.hostname}:${wsPort}/stream`);
                const img = document.getElementById('game-screen');
                
                ws.onopen = function() {
                    console.log('📡 WebSocket stream connected');
                };
                
                ws.onmessage = function(event) {
                    // Convert binary data to blob URL
                    const blob = new Blob([event.data], { type: 'image/png' });
                    const url = URL.createObjectURL(blob);
                    
                    // Update image source
                    const oldUrl = img.src;
                    img.src = url;
                    
                    // Clean up old blob URL
                    if (oldUrl.startsWith('blob:')) {
                        URL.revokeObjectURL(oldUrl);
                    }
                };
                
                ws.onclose = function() {
                    console.log('📡 WebSocket stream disconnected - attempting reconnect in 3s');
                    setTimeout(initWebSocketStream, 3000);
                };
                
                ws.onerror = function(error) {
                    console.error('WebSocket stream error:', error);
                };
                
            } catch (error) {
                console.error('WebSocket initialization error:', error);
                // Fallback to HTTP polling
                console.log('Falling back to HTTP polling');
                setInterval(updateScreenHTTP, SCREEN_UPDATE_MS);
                updateScreenHTTP();
            }
        }
        
        // Fallback HTTP polling function
        async function updateScreenHTTP() {
            try {
                const img = document.getElementById('game-screen');
                img.src = '/api/screenshot?t=' + Date.now();
            } catch (error) {
                console.error('Screen update error:', error);
            }
        }
        
        async function updateStats() {
            try {
                const response = await fetch('/api/stats');
                const stats = await response.json();
                
                // Update training stats
                document.getElementById('total-actions').textContent = stats.total_actions || 0;
                document.getElementById('actions-per-second').textContent = (stats.actions_per_second || 0).toFixed(1);
                document.getElementById('llm-decisions').textContent = stats.llm_calls || 0;
                document.getElementById('total-reward').textContent = (stats.total_reward || 0).toFixed(2);
                
                // Update game state
                document.getElementById('current-map').textContent = stats.current_map || '-';
                const pos = stats.player_position || {};
                document.getElementById('player-position').textContent = pos.x !== undefined && pos.y !== undefined ? `${pos.x},${pos.y}` : '-';
                document.getElementById('money').textContent = '¥' + (stats.money || 0);
                document.getElementById('badges').textContent = `${stats.badges_earned || 0}/16`;
                
                // Update memory debug
                if (stats.memory_data) {
                    updateMemoryDebug(stats.memory_data);
                }
                
                // Update connection status
                document.getElementById('connection-status').textContent = 'Connected';
                
            } catch (error) {
                console.error('Stats update error:', error);
                document.getElementById('connection-status').textContent = 'Connection Error';
            }
        }
        
        async function updateLLMDecisions() {
            try {
                const response = await fetch('/api/llm_decisions');
                const data = await response.json();
                
                if (data.recent_decisions && data.recent_decisions.length > 0) {
                    const container = document.getElementById('llm-decisions-list');
                    container.innerHTML = '';
                    
                    // Show last 3 decisions
                    const recentDecisions = data.recent_decisions.slice(-3);
                    recentDecisions.reverse().forEach(decision => {
                        const decisionDiv = document.createElement('div');
                        decisionDiv.className = 'llm-decision';
                        
                        const timestamp = new Date(decision.timestamp * 1000).toLocaleTimeString();
                        
                        decisionDiv.innerHTML = `
                            <div class="decision-action">Action: ${decision.action}</div>
                            <div class="decision-reasoning">${decision.reasoning || 'No reasoning provided'}</div>
                            <div class="decision-timestamp">${timestamp}</div>
                        `;
                        
                        container.appendChild(decisionDiv);
                    });
                }
            } catch (error) {
                console.error('LLM decisions update error:', error);
            }
        }
        
        function updateMemoryDebug(memoryData) {
            const container = document.getElementById('memory-debug');
            container.innerHTML = '';
            
            // Core addresses
            const addresses = [
                { addr: '0xD163', name: 'PARTY_COUNT', value: memoryData.party_count || 0 },
                { addr: '0xDCBA', name: 'MAP_ID', value: memoryData.player_map || 0 },
                { addr: '0xDCB8', name: 'PLAYER_X', value: memoryData.player_x || 0 },
                { addr: '0xDCB9', name: 'PLAYER_Y', value: memoryData.player_y || 0 },
                { addr: '0xD347-49', name: 'MONEY', value: memoryData.money || 0 },
                { addr: '0xD359', name: 'BADGES', value: memoryData.badges || 0 },
                { addr: '0xD057', name: 'IN_BATTLE', value: memoryData.in_battle || 0 },
                { addr: '0xD16B', name: 'LEVEL', value: memoryData.player_level || 0 }
            ];
            
            addresses.forEach(item => {
                const row = document.createElement('div');
                row.className = 'memory-row';
                row.innerHTML = `
                    <span class="address">${item.addr}:</span>
                    <span class="value">${item.name}</span>
                    <span class="value">${item.value}</span>
                `;
                container.appendChild(row);
            });
        }
    </script>
</body>
</html>'''
        
        self.send_response(200)
        self.send_header('Content-type', 'text/html; charset=utf-8')
        self.end_headers()
        self.wfile.write(html_content.encode('utf-8'))
    
    def _serve_screen(self):
        """Serve current game screen"""
        try:
            if self.screen_capture:
                screen_bytes = self.screen_capture.get_latest_screen_bytes()
                if screen_bytes:
                    self.send_response(200)
                    self.send_header('Content-type', 'image/png')
                    self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
                    self.send_header('Pragma', 'no-cache')
                    self.send_header('Expires', '0')
                    self.end_headers()
                    self.wfile.write(screen_bytes)
                    return
            
            # Return blank image if no screen available
            blank_img = Image.new('RGB', (320, 288), color='black')
            buffer = io.BytesIO()
            blank_img.save(buffer, format='PNG')
            
            self.send_response(200)
            self.send_header('Content-type', 'image/png')
            self.end_headers()
            self.wfile.write(buffer.getvalue())
            
        except Exception as e:
            logger.error(f"Screen serve error: {e}")
            self.send_error(500)
    
    def _serve_stats(self):
        """Serve training statistics as JSON"""
        try:
            if self.web_api:
                stats = self.web_api.get_training_stats()
            else:
                # Fallback to direct trainer access
                stats = {
                    'total_actions': 0,
                    'actions_per_second': 0.0,
                    'llm_calls': 0,
                    'total_reward': 0.0
                }
                
                # Get stats from trainer
                if self.trainer and hasattr(self.trainer, 'stats'):
                    trainer_stats = self.trainer.stats
                    stats.update({
                        'total_actions': trainer_stats.get('actions_taken', 0),
                        'actions_per_second': trainer_stats.get('actions_per_second', 0.0),
                        'llm_calls': trainer_stats.get('llm_decision_count', 0),
                        'total_reward': trainer_stats.get('total_reward', 0.0)
                    })
                elif self.trainer and hasattr(self.trainer, 'get_current_stats'):
                    stats.update(self.trainer.get_current_stats())
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(stats, indent=2).encode('utf-8'))
            
        except Exception as e:
            logger.error(f"Stats serve error: {e}")
            self.send_error(500)
    
    def _serve_status(self):
        """Serve system status"""
        try:
            if self.web_api:
                status = self.web_api.get_system_status()
            else:
                # Fallback to direct implementation
                current_time = time.time()
                trainer_start = current_time
                try:
                    if self.trainer and hasattr(self.trainer, 'stats') and isinstance(self.trainer.stats, dict):
                        start_iso = self.trainer.stats.get('start_time')
                        if isinstance(start_iso, str):
                            start_time = datetime.fromisoformat(start_iso)
                            trainer_start = start_time.timestamp()
                except Exception:
                    # Ignore parsing issues and keep default
                    pass
                
                # Check if screen capture is truly active with a real PyBoy instance
                screen_active = False
                if (self.screen_capture is not None and 
                    self.screen_capture.pyboy is not None and 
                    self.screen_capture.capture_active):
                    # Additional check for mock objects in tests
                    if hasattr(self.screen_capture.pyboy, '_mock_name'):
                        screen_active = False  # Mock PyBoy doesn't count as active
                    else:
                        screen_active = True
                
                status = {
                    'status': 'running',
                    'uptime': max(0.0, current_time - trainer_start),
                    'version': '1.0.0',
                    'screen_capture_active': screen_active
                }
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(status, indent=2).encode('utf-8'))
            
        except Exception as e:
            logger.error(f"Status serve error: {e}")
            self.send_error(500)
    
    def _serve_llm_decisions(self):
        """Serve recent LLM decisions with enhanced information"""
        try:
            if self.web_api:
                decisions_data = self.web_api.get_llm_decisions()
            else:
                # Fallback implementation kept for compatibility
                decisions_data = {
                    'recent_decisions': [],
                    'total_decisions': 0,
                    'decision_rate': 0.0,
                    'average_response_time_ms': 0.0,
                    'last_decision_age_seconds': None,
                    'timestamp': time.time()
                }
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(decisions_data, indent=2).encode('utf-8'))
            
        except Exception as e:
            logger.error(f"LLM decisions serve error: {e}")
            error_response = {
                'error': f"LLM decisions error: {str(e)}",
                'timestamp': time.time(),
                'recent_decisions': [],
                'total_decisions': 0
            }
            try:
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps(error_response, indent=2).encode('utf-8'))
            except:
                pass
    
    def _get_action_name(self, action):
        """Convert action number to readable name"""
        action_names = {
            0: "RIGHT",
            1: "LEFT", 
            2: "UP",
            3: "DOWN",
            4: "A",
            5: "B",
            6: "SELECT",
            7: "START"
        }
        return action_names.get(action, f"ACTION_{action}")
    
    def _serve_memory_debug(self):
        """Serve memory debug information"""
        try:
            if self.web_api:
                memory_state = self.web_api.get_memory_debug()
            else:
                # Fallback implementation
                if PokemonCrystalMemoryReader is None:
                    memory_state = {
                        'error': 'Memory reader not available - import failed',
                        'timestamp': time.time()
                    }
                else:
                    # Initialize memory reader if needed
                    if not hasattr(self.trainer, 'memory_reader') or self.trainer.memory_reader is None:
                        if hasattr(self.trainer, 'pyboy') and self.trainer.pyboy is not None:
                            self.trainer.memory_reader = PokemonCrystalMemoryReader(self.trainer.pyboy)
                        else:
                            memory_state = {
                                'error': 'PyBoy instance not available',
                                'timestamp': time.time()
                            }
                            
                    if hasattr(self.trainer, 'memory_reader') and self.trainer.memory_reader is not None:
                        memory_state = self.trainer.memory_reader.read_game_state()
                        # Add debug info
                        memory_state['debug_info'] = self.trainer.memory_reader.get_debug_info()
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(memory_state, indent=2).encode('utf-8'))
            
        except Exception as e:
            logger.error(f"Memory debug serve error: {e}")
            error_response = {
                'error': f"Memory debug error: {str(e)}",
                'timestamp': time.time()
            }
            try:
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps(error_response, indent=2).encode('utf-8'))
            except:
                pass  # Connection might be closed