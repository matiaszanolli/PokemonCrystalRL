"""
web_monitor.py - Real-time Web UI for Pokemon Crystal RL Training

Features:
- Live game screen streaming
- Real-time game stats monitoring
- Action history and button press visualization
- Training metrics and graphs
- Agent decision logs
- Visual analysis results
"""

import json
import time
import base64
import sqlite3
import threading
from datetime import datetime
from typing import Dict, List, Any, Optional
from queue import Queue
import io

import cv2
import numpy as np
from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
import sqlite3

# Import our modules
try:
    from ..agents.enhanced_llm_agent import EnhancedLLMPokemonAgent
except ImportError:
    try:
        from agents.enhanced_llm_agent import EnhancedLLMPokemonAgent
    except ImportError:
        # Create a stub if the agent is not available
        class EnhancedLLMPokemonAgent:
            def __init__(self, *args, **kwargs):
                pass
try:
    from ..core.vision_enhanced_training import VisionEnhancedTrainingSession
except ImportError:
    try:
        from vision_enhanced_training import VisionEnhancedTrainingSession
    except ImportError:
        # Create a stub if the training session is not available
        class VisionEnhancedTrainingSession:
            def __init__(self, *args, **kwargs):
                pass


class PokemonRLWebMonitor:
    """
    Real-time web monitoring dashboard for Pokemon Crystal RL
    """
    
    def __init__(self, training_session: Optional[VisionEnhancedTrainingSession] = None):
        self.training_session = training_session
        
        # Set absolute path for templates
        import os
        template_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'templates')
        
        self.app = Flask(__name__, 
                        template_folder=template_dir,
                        static_folder='static')
        self.app.config['SECRET_KEY'] = 'pokemon_rl_monitor_key'
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        
        # Data queues for real-time updates
        self.screen_queue = Queue(maxsize=10)
        self.stats_queue = Queue(maxsize=50)
        self.action_queue = Queue(maxsize=100)
        self.decision_queue = Queue(maxsize=50)
        
        # Monitoring state
        self.is_monitoring = False
        self.current_stats = {}
        self.action_history = []
        self.recent_decisions = []
        self.performance_metrics = []
        
        # Setup routes and socket events
        self._setup_routes()
        self._setup_socket_events()
        
        print("🌐 Pokemon RL Web Monitor initialized")
    
    def _setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/')
        def dashboard():
            """Main dashboard page"""
            return render_template('dashboard.html')
        
        @self.app.route('/api/status')
        def get_status():
            """Get current monitoring status"""
            return jsonify({
                'monitoring': self.is_monitoring,
                'training_active': self.training_session is not None,
                'current_stats': self.current_stats,
                'timestamp': datetime.now().isoformat()
            })
        
        @self.app.route('/api/history')
        def get_history():
            """Get historical training data"""
            try:
                if self.training_session:
                    db_path = self.training_session.agent.memory_db
                else:
                    db_path = "outputs/pokemon_agent_memory.db"
                
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                
                # Get recent game states
                cursor.execute("""
                    SELECT timestamp, player_x, player_y, player_map, money, badges, 
                           party_size, visual_summary, screen_type
                    FROM game_states 
                    ORDER BY timestamp DESC 
                    LIMIT 100
                """)
                
                states = []
                for row in cursor.fetchall():
                    states.append({
                        'timestamp': row[0],
                        'x': row[1],
                        'y': row[2],
                        'map': row[3],
                        'money': row[4],
                        'badges': row[5],
                        'party_size': row[6],
                        'visual_summary': row[7],
                        'screen_type': row[8]
                    })
                
                # Get recent decisions
                cursor.execute("""
                    SELECT timestamp, decision, reasoning, confidence_score, visual_context
                    FROM strategic_decisions 
                    ORDER BY timestamp DESC 
                    LIMIT 50
                """)
                
                decisions = []
                for row in cursor.fetchall():
                    visual_context = json.loads(row[4]) if row[4] else {}
                    decisions.append({
                        'timestamp': row[0],
                        'decision': row[1],
                        'reasoning': row[2][:200] + "..." if len(row[2]) > 200 else row[2],
                        'confidence': row[3],
                        'screen_type': visual_context.get('screen_type', 'unknown')
                    })
                
                conn.close()
                
                return jsonify({
                    'game_states': states,
                    'decisions': decisions,
                    'performance_metrics': self.performance_metrics
                })
                
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/training_reports')
        def get_training_reports():
            """Get available training reports"""
            import glob
            import os
            
            reports = []
            for report_file in glob.glob("outputs/training_report_*.json"):
                try:
                    with open(report_file, 'r') as f:
                        data = json.load(f)
                    
                    reports.append({
                        'filename': os.path.basename(report_file),
                        'timestamp': data['session_info'].get('timestamp', 'unknown'),
                        'episodes': data['session_info'].get('episodes', 0),
                        'total_steps': data['session_info'].get('total_steps', 0),
                        'duration': data['session_info'].get('duration', 0)
                    })
                except Exception as e:
                    continue
            
            # Sort by filename (which includes timestamp)
            reports.sort(key=lambda x: x['filename'], reverse=True)
            return jsonify(reports)
        
        @self.app.route('/api/system')
        def get_system_stats():
            """Get system resource statistics"""
            try:
                import psutil
                stats = {
                    'cpu_percent': psutil.cpu_percent(interval=0.1),
                    'memory_percent': psutil.virtual_memory().percent,
                    'memory_used': psutil.virtual_memory().used // (1024*1024),  # MB
                    'memory_total': psutil.virtual_memory().total // (1024*1024),  # MB
                    'disk_usage': psutil.disk_usage('/').percent,
                    'disk_free': psutil.disk_usage('/').free // (1024*1024*1024),  # GB
                    'gpu_available': False,  # Could be enhanced to detect GPU
                    'timestamp': datetime.now().isoformat()
                }
            except ImportError:
                stats = {
                    'cpu_percent': 0.0,
                    'memory_percent': 0.0,
                    'memory_used': 0,
                    'memory_total': 0,
                    'disk_usage': 0.0,
                    'disk_free': 0,
                    'gpu_available': False,
                    'error': 'psutil not available',
                    'timestamp': datetime.now().isoformat()
                }
            except Exception as e:
                stats = {
                    'cpu_percent': 0.0,
                    'memory_percent': 0.0,
                    'memory_used': 0,
                    'memory_total': 0,
                    'disk_usage': 0.0,
                    'disk_free': 0,
                    'gpu_available': False,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }
            return jsonify(stats)
        
        @self.app.route('/api/text')
        def get_text_data():
            """Get OCR text recognition data"""
            # Try to get text data from training session or bridge
            text_data = {
                'recent_text': [],
                'text_frequency': {},
                'total_texts': 0,
                'unique_texts': 0,
                'timestamp': datetime.now().isoformat()
            }
            
            if self.training_session and hasattr(self.training_session, 'recent_text'):
                text_data.update({
                    'recent_text': self.training_session.recent_text[-20:],
                    'text_frequency': dict(sorted(
                        getattr(self.training_session, 'text_frequency', {}).items(),
                        key=lambda x: x[1], reverse=True
                    )[:30]),
                    'total_texts': len(getattr(self.training_session, 'recent_text', [])),
                    'unique_texts': len(getattr(self.training_session, 'text_frequency', {}))
                })
            
            return jsonify(text_data)
        
        @self.app.route('/api/control', methods=['POST'])
        def training_control():
            """Control training session via API"""
            command = request.json.get('command')
            response = {'status': 'error', 'message': 'Unknown command'}
            
            if not self.training_session:
                return jsonify({'status': 'error', 'message': 'No active training session'}), 400
            
            if command == 'pause':
                # Implement pause functionality if available
                if hasattr(self.training_session, 'pause'):
                    self.training_session.pause()
                    response = {'status': 'success', 'message': 'Training paused'}
                else:
                    response = {'status': 'error', 'message': 'Pause not supported by this training session'}
            
            elif command == 'resume':
                # Implement resume functionality if available
                if hasattr(self.training_session, 'resume'):
                    self.training_session.resume()
                    response = {'status': 'success', 'message': 'Training resumed'}
                else:
                    response = {'status': 'error', 'message': 'Resume not supported by this training session'}
            
            elif command == 'stop':
                # Implement stop functionality if available
                if hasattr(self.training_session, 'stop'):
                    self.training_session.stop()
                    response = {'status': 'success', 'message': 'Training stopped'}
                else:
                    response = {'status': 'error', 'message': 'Stop not supported by this training session'}
            
            return jsonify(response)
    
    def _setup_socket_events(self):
        """Setup WebSocket events"""
        
        @self.socketio.on('connect')
        def handle_connect():
            print(f"🔌 Client connected: {request.sid}")
            emit('status', {
                'monitoring': self.is_monitoring,
                'message': 'Connected to Pokemon RL Monitor'
            })
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            print(f"🔌 Client disconnected: {request.sid}")
        
        @self.socketio.on('start_monitoring')
        def handle_start_monitoring():
            """Start monitoring training session"""
            self.start_monitoring()
            emit('status', {'monitoring': True, 'message': 'Monitoring started'})
        
        @self.socketio.on('stop_monitoring')
        def handle_stop_monitoring():
            """Stop monitoring training session"""
            self.stop_monitoring()
            emit('status', {'monitoring': False, 'message': 'Monitoring stopped'})
        
        @self.socketio.on('request_screenshot')
        def handle_screenshot_request():
            """Send current screenshot"""
            if not self.screen_queue.empty():
                screenshot_data = self.screen_queue.get()
                emit('screenshot', screenshot_data)
    
    def start_monitoring(self):
        """Start the monitoring process"""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        
        # Start monitoring thread
        monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        monitor_thread.start()
        
        print("🎯 Web monitoring started")
    
    def stop_monitoring(self):
        """Stop the monitoring process"""
        self.is_monitoring = False
        print("⏹️ Web monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                # Update statistics if training session is available
                if self.training_session:
                    self._update_training_stats()
                
                # Broadcast updates to all connected clients
                self.socketio.emit('stats_update', self.current_stats)
                
                # Send recent actions
                if self.action_history:
                    self.socketio.emit('action_update', {
                        'recent_actions': self.action_history[-20:],
                        'action_counts': self._get_action_counts()
                    })
                
                time.sleep(0.5)  # Update every 500ms
                
            except Exception as e:
                print(f"❌ Monitoring error: {e}")
                time.sleep(1)
    
    def _update_training_stats(self):
        """Update current training statistics"""
        if not self.training_session:
            return
        
        try:
            # Get current game state
            game_state = self.training_session.env.get_game_state()
            player = game_state.get('player', {})
            party = game_state.get('party', [])
            
            # Update current stats
            self.current_stats = {
                'timestamp': datetime.now().isoformat(),
                'player': {
                    'x': player.get('x', 0),
                    'y': player.get('y', 0),
                    'map': player.get('map', 0),
                    'money': player.get('money', 0),
                    'badges': player.get('badges', 0)
                },
                'party': [
                    {
                        'species': p.get('species', 0),
                        'level': p.get('level', 1),
                        'hp': p.get('hp', 0),
                        'max_hp': p.get('max_hp', 1),
                        'status': p.get('status', 0)
                    }
                    for p in party
                ],
                'training': {
                    'total_steps': self.training_session.training_stats['total_steps'],
                    'episodes': self.training_session.training_stats['episodes'],
                    'decisions_made': self.training_session.training_stats['decisions_made'],
                    'visual_analyses': self.training_session.training_stats['visual_analyses']
                }
            }
            
        except Exception as e:
            print(f"⚠️ Stats update error: {e}")
    
    def _get_action_counts(self):
        """Get action frequency counts"""
        action_counts = {}
        for action_data in self.action_history[-100:]:  # Last 100 actions
            # Extract action name from action data dict
            action_name = action_data.get('action', 'unknown') if isinstance(action_data, dict) else str(action_data)
            action_counts[action_name] = action_counts.get(action_name, 0) + 1
        return action_counts
    
    def update_screenshot(self, screenshot: np.ndarray):
        """Update screenshot for web streaming"""
        try:
            # Resize screenshot for web display
            height, width = screenshot.shape[:2]
            scale_factor = 3  # Make it bigger for web display
            new_width = width * scale_factor
            new_height = height * scale_factor
            
            resized = cv2.resize(screenshot, (new_width, new_height), 
                               interpolation=cv2.INTER_NEAREST)
            
            # Convert to base64 for web transmission
            _, buffer = cv2.imencode('.png', cv2.cvtColor(resized, cv2.COLOR_RGB2BGR))
            screenshot_b64 = base64.b64encode(buffer).decode('utf-8')
            
            screenshot_data = {
                'image': f"data:image/png;base64,{screenshot_b64}",
                'timestamp': datetime.now().isoformat(),
                'dimensions': {'width': new_width, 'height': new_height}
            }
            
            # Add to queue (remove old if full)
            if self.screen_queue.full():
                try:
                    self.screen_queue.get_nowait()
                except:
                    pass
            
            self.screen_queue.put(screenshot_data)
            
            # Emit to connected clients
            if self.is_monitoring:
                self.socketio.emit('screenshot', screenshot_data)
                
        except Exception as e:
            print(f"⚠️ Screenshot update error: {e}")
    
    def update_action(self, action: str, reasoning: str = ""):
        """Update action history"""
        action_data = {
            'action': action,
            'timestamp': datetime.now().isoformat(),
            'reasoning': reasoning[:100] + "..." if len(reasoning) > 100 else reasoning
        }
        
        self.action_history.append(action_data)
        
        # Keep only recent actions
        if len(self.action_history) > 200:
            self.action_history = self.action_history[-100:]
        
        # Emit to connected clients
        if self.is_monitoring:
            self.socketio.emit('new_action', action_data)
    
    def update_decision(self, decision_data: Dict[str, Any]):
        """Update agent decision data"""
        decision_entry = {
            'timestamp': datetime.now().isoformat(),
            'decision': decision_data.get('decision', ''),
            'reasoning': decision_data.get('reasoning', '')[:200] + "...",
            'confidence': decision_data.get('confidence', 0),
            'visual_context': decision_data.get('visual_context', {})
        }
        
        self.recent_decisions.append(decision_entry)
        
        # Keep only recent decisions
        if len(self.recent_decisions) > 100:
            self.recent_decisions = self.recent_decisions[-50:]
        
        # Emit to connected clients
        if self.is_monitoring:
            self.socketio.emit('new_decision', decision_entry)
    
    def add_performance_metric(self, metric_name: str, value: float):
        """Add performance metric for graphing"""
        metric_entry = {
            'name': metric_name,
            'value': value,
            'timestamp': datetime.now().isoformat()
        }
        
        self.performance_metrics.append(metric_entry)
        
        # Keep only recent metrics
        if len(self.performance_metrics) > 1000:
            self.performance_metrics = self.performance_metrics[-500:]
    
    def run(self, host='127.0.0.1', port=5000, debug=False):
        """Run the web monitoring server"""
        print(f"🚀 Starting Pokemon RL Web Monitor on http://{host}:{port}")
        print("📊 Dashboard features:")
        print("   - Real-time game screen streaming")
        print("   - Live game statistics")
        print("   - Action history and button visualization")
        print("   - Training metrics and performance graphs")
        print("   - Agent decision logs")
        
        self.socketio.run(self.app, host=host, port=port, debug=debug)


def create_dashboard_templates():
    """Create HTML templates for the web dashboard"""
    import os
    
    # Create templates directory
    os.makedirs('templates', exist_ok=True)
    
    # Main dashboard HTML
    dashboard_html = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Pokemon Crystal RL - Training Monitor</title>
    <script src="https://cdn.socket.io/4.0.0/socket.io.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #1a1a1a;
            color: #ffffff;
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
        
        .status.monitoring { background-color: #4CAF50; }
        .status.stopped { background-color: #f44336; }
        
        .nav-tabs {
            display: flex;
            justify-content: center;
            margin-bottom: 20px;
            border-radius: 10px;
            background-color: #2a2a2a;
            padding: 5px;
        }
        
        .nav-tab {
            padding: 10px 20px;
            background-color: transparent;
            color: #ccc;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.3s;
            margin: 0 5px;
        }
        
        .nav-tab.active {
            background-color: #4CAF50;
            color: white;
        }
        
        .tab-content {
            display: none;
        }
        
        .tab-content.active {
            display: block;
        }
        
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
        
        .pokemon-party {
            display: grid;
            gap: 10px;
        }
        
        .pokemon {
            padding: 10px;
            background-color: #444;
            border-radius: 5px;
            border-left: 4px solid #4CAF50;
        }
        
        .pokemon.low-hp {
            border-left-color: #f44336;
        }
        
        .hp-bar {
            width: 100%;
            height: 8px;
            background-color: #666;
            border-radius: 4px;
            margin-top: 5px;
        }
        
        .hp-fill {
            height: 100%;
            background-color: #4CAF50;
            border-radius: 4px;
            transition: width 0.3s ease;
        }
        
        .hp-fill.low { background-color: #f44336; }
        .hp-fill.medium { background-color: #FF9800; }
        
        .side-panel {
            display: grid;
            grid-template-rows: auto auto 1fr;
            gap: 20px;
        }
        
        .controls {
            background-color: #2a2a2a;
            border-radius: 10px;
            padding: 20px;
        }
        
        .control-buttons {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
        }
        
        button {
            padding: 12px;
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
            transition: background-color 0.3s;
        }
        
        button:hover {
            background-color: #45a049;
        }
        
        button.stop {
            background-color: #f44336;
        }
        
        button.stop:hover {
            background-color: #d32f2f;
        }
        
        .actions-panel {
            background-color: #2a2a2a;
            border-radius: 10px;
            padding: 20px;
        }
        
        .action-history {
            max-height: 300px;
            overflow-y: auto;
            background-color: #333;
            border-radius: 5px;
            padding: 15px;
        }
        
        .action-item {
            padding: 8px;
            margin: 5px 0;
            background-color: #444;
            border-radius: 3px;
            font-size: 0.9em;
            border-left: 3px solid #4CAF50;
        }
        
        .decisions-panel {
            background-color: #2a2a2a;
            border-radius: 10px;
            padding: 20px;
        }
        
        .decision-item {
            padding: 12px;
            margin: 10px 0;
            background-color: #333;
            border-radius: 5px;
            border-left: 3px solid #2196F3;
        }
        
        .decision-action {
            font-weight: bold;
            color: #4CAF50;
        }
        
        .decision-reasoning {
            font-size: 0.9em;
            color: #ccc;
            margin-top: 5px;
        }
        
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.7; }
            100% { opacity: 1; }
        }
        
        .updating {
            animation: pulse 1s infinite;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🎮 Pokemon Crystal RL Training Monitor</h1>
        <div id="status" class="status stopped">🔴 Not Monitoring</div>
    </div>

    <div class="dashboard">
        <div class="main-panel">
            <div class="game-screen">
                <h3>🎯 Live Game Screen</h3>
                <img id="screenshot" src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==" alt="Game Screen">
                <p id="screen-info">Waiting for game data...</p>
            </div>
            
            <div class="stats-panel">
                <h3>📊 Game Statistics</h3>
                
                <div class="stat-group">
                    <h3>👤 Player Info</h3>
                    <div class="stat-item">
                        <span>Location:</span>
                        <span id="player-location">Map ?, (?, ?)</span>
                    </div>
                    <div class="stat-item">
                        <span>💰 Money:</span>
                        <span id="player-money">$0</span>
                    </div>
                    <div class="stat-item">
                        <span>🏆 Badges:</span>
                        <span id="player-badges">0</span>
                    </div>
                </div>
                
                <div class="stat-group">
                    <h3>🐾 Pokemon Party</h3>
                    <div id="pokemon-party" class="pokemon-party">
                        <p>No Pokemon data available</p>
                    </div>
                </div>
                
                <div class="stat-group">
                    <h3>🎯 Training Stats</h3>
                    <div class="stat-item">
                        <span>Episodes:</span>
                        <span id="training-episodes">0</span>
                    </div>
                    <div class="stat-item">
                        <span>Total Steps:</span>
                        <span id="training-steps">0</span>
                    </div>
                    <div class="stat-item">
                        <span>Decisions Made:</span>
                        <span id="training-decisions">0</span>
                    </div>
                    <div class="stat-item">
                        <span>Visual Analyses:</span>
                        <span id="training-visual">0</span>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="side-panel">
            <div class="controls">
                <h3>🎛️ Controls</h3>
                <div class="control-buttons">
                    <button onclick="startMonitoring()">▶️ Start Monitor</button>
                    <button onclick="stopMonitoring()" class="stop">⏹️ Stop Monitor</button>
                </div>
            </div>
            
            <div class="actions-panel">
                <h3>🎮 Recent Actions</h3>
                <div id="action-history" class="action-history">
                    <p>No actions yet...</p>
                </div>
            </div>
            
            <div class="decisions-panel">
                <h3>🧠 Agent Decisions</h3>
                <div id="decision-history" style="max-height: 400px; overflow-y: auto;">
                    <p>No decisions yet...</p>
                </div>
            </div>
        </div>
    </div>

    <script>
        // WebSocket connection
        const socket = io();
        
        // Connection events
        socket.on('connect', function() {
            console.log('Connected to server');
        });
        
        socket.on('status', function(data) {
            updateStatus(data);
        });
        
        socket.on('screenshot', function(data) {
            updateScreenshot(data);
        });
        
        socket.on('stats_update', function(data) {
            updateStats(data);
        });
        
        socket.on('new_action', function(data) {
            addAction(data);
        });
        
        socket.on('new_decision', function(data) {
            addDecision(data);
        });
        
        // Control functions
        function startMonitoring() {
            socket.emit('start_monitoring');
        }
        
        function stopMonitoring() {
            socket.emit('stop_monitoring');
        }
        
        // Update functions
        function updateStatus(data) {
            const statusElement = document.getElementById('status');
            if (data.monitoring) {
                statusElement.textContent = '🟢 Monitoring Active';
                statusElement.className = 'status monitoring';
            } else {
                statusElement.textContent = '🔴 Not Monitoring';
                statusElement.className = 'status stopped';
            }
        }
        
        function updateScreenshot(data) {
            const imgElement = document.getElementById('screenshot');
            const infoElement = document.getElementById('screen-info');
            
            imgElement.src = data.image;
            infoElement.textContent = `Updated: ${new Date(data.timestamp).toLocaleTimeString()}`;
            imgElement.classList.add('updating');
            setTimeout(() => imgElement.classList.remove('updating'), 200);
        }
        
        function updateStats(data) {
            if (data.player) {
                document.getElementById('player-location').textContent = 
                    `Map ${data.player.map}, (${data.player.x}, ${data.player.y})`;
                document.getElementById('player-money').textContent = 
                    `$${data.player.money.toLocaleString()}`;
                document.getElementById('player-badges').textContent = 
                    data.player.badges;
            }
            
            if (data.party) {
                updatePokemonParty(data.party);
            }
            
            if (data.training) {
                document.getElementById('training-episodes').textContent = 
                    data.training.episodes;
                document.getElementById('training-steps').textContent = 
                    data.training.total_steps.toLocaleString();
                document.getElementById('training-decisions').textContent = 
                    data.training.decisions_made.toLocaleString();
                document.getElementById('training-visual').textContent = 
                    data.training.visual_analyses.toLocaleString();
            }
        }
        
        function updatePokemonParty(party) {
            const partyElement = document.getElementById('pokemon-party');
            
            if (party.length === 0) {
                partyElement.innerHTML = '<p>No Pokemon in party</p>';
                return;
            }
            
            partyElement.innerHTML = '';
            party.forEach((pokemon, index) => {
                const hpPercent = (pokemon.hp / pokemon.max_hp) * 100;
                const hpClass = hpPercent < 25 ? 'low' : hpPercent < 50 ? 'medium' : '';
                
                const pokemonDiv = document.createElement('div');
                pokemonDiv.className = `pokemon ${hpPercent < 25 ? 'low-hp' : ''}`;
                pokemonDiv.innerHTML = `
                    <div><strong>Pokemon #${pokemon.species}</strong> - Level ${pokemon.level}</div>
                    <div>HP: ${pokemon.hp}/${pokemon.max_hp}</div>
                    <div class="hp-bar">
                        <div class="hp-fill ${hpClass}" style="width: ${hpPercent}%"></div>
                    </div>
                `;
                partyElement.appendChild(pokemonDiv);
            });
        }
        
        function addAction(data) {
            const historyElement = document.getElementById('action-history');
            
            // Clear placeholder text
            if (historyElement.textContent.includes('No actions yet')) {
                historyElement.innerHTML = '';
            }
            
            const actionDiv = document.createElement('div');
            actionDiv.className = 'action-item';
            actionDiv.innerHTML = `
                <strong>${data.action}</strong>
                <div style="font-size: 0.8em; color: #aaa;">
                    ${new Date(data.timestamp).toLocaleTimeString()}
                </div>
                ${data.reasoning ? `<div style="font-size: 0.8em; margin-top: 5px;">${data.reasoning}</div>` : ''}
            `;
            
            historyElement.insertBefore(actionDiv, historyElement.firstChild);
            
            // Keep only recent actions (max 20)
            while (historyElement.children.length > 20) {
                historyElement.removeChild(historyElement.lastChild);
            }
        }
        
        function addDecision(data) {
            const historyElement = document.getElementById('decision-history');
            
            // Clear placeholder text
            if (historyElement.textContent.includes('No decisions yet')) {
                historyElement.innerHTML = '';
            }
            
            const decisionDiv = document.createElement('div');
            decisionDiv.className = 'decision-item';
            decisionDiv.innerHTML = `
                <div class="decision-action">${data.decision}</div>
                <div class="decision-reasoning">${data.reasoning}</div>
                <div style="font-size: 0.8em; color: #aaa; margin-top: 5px;">
                    ${new Date(data.timestamp).toLocaleTimeString()}
                    ${data.confidence ? ` | Confidence: ${(data.confidence * 100).toFixed(1)}%` : ''}
                </div>
            `;
            
            historyElement.insertBefore(decisionDiv, historyElement.firstChild);
            
            // Keep only recent decisions (max 15)
            while (historyElement.children.length > 15) {
                historyElement.removeChild(historyElement.lastChild);
            }
        }
        
        // Request screenshot on load
        socket.emit('request_screenshot');
        
        // Auto-refresh stats
        setInterval(() => {
            fetch('/api/status')
                .then(response => response.json())
                .then(data => {
                    if (data.current_stats && Object.keys(data.current_stats).length > 0) {
                        updateStats(data.current_stats);
                    }
                })
                .catch(console.error);
        }, 2000);
    </script>
</body>
</html>'''
    
    with open('templates/dashboard.html', 'w') as f:
        f.write(dashboard_html)
    
    print("✅ Dashboard template created at templates/dashboard.html")


def main():
    """Main function to run the web monitor standalone"""
    # Create templates
    create_dashboard_templates()
    
    # Create monitor instance
    monitor = PokemonRLWebMonitor()
    
    # Run the web server
    monitor.run(host='0.0.0.0', port=5000, debug=False)


if __name__ == "__main__":
    main()
