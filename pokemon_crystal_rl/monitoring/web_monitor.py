"""Web monitoring interface for Pokemon Crystal RL trainer."""

import http.server
import socketserver
import json
import time
import threading
import queue
import logging
import os
from typing import Optional, Dict, Any, List
from pathlib import Path
import numpy as np
from .config import MonitorConfig
from .database import DatabaseManager
from .error_handler import ErrorHandler, ErrorSeverity, RecoveryStrategy
from .training_state import TrainingState

class WebMonitor:
    """Web-based monitoring interface for Pokemon Crystal RL trainer."""
    
    def __init__(self, config: MonitorConfig):
        self.config = config
        self.db = DatabaseManager(Path(config.db_path))
        self.error_handler = ErrorHandler()
        
        # State
        self.training_state = TrainingState.INITIALIZING
        self.current_run_id = None
        self._subscribers = []
        self._update_thread = None
        self._running = False
        
        # Queues
        self.screenshot_queue = queue.Queue(maxsize=30)  # Bounded queue
        self.current_stats: Dict[str, Any] = {}
        self.last_action: Optional[str] = None
        self.last_action_reasoning: Optional[str] = None
        self.last_screenshot = None
        self.capture_active = False
        self.current_fps = 0
        self._start_time = time.time()
        
        # Performance metrics
        self.metrics = {
            'reward': [],
            'exploration': [],
            'game_progress': [],
            'fps': [],
        }
        
        # Decision history
        self.decisions: List[Dict] = []
        self.max_decisions = 100  # Keep last 100 decisions
        
        # Set up logging
        self.logger = logging.getLogger('web_monitor')
        self.logger.setLevel(logging.INFO)
    
    def run(self):
        """Start the web monitoring server."""
        host = '127.0.0.1'  # Default to localhost
        port = self.config.web_port
        
        class MonitorHandler(http.server.SimpleHTTPRequestHandler):
            def __init__(self, *args, monitor=self, **kwargs):
                self.monitor = monitor
                super().__init__(*args, **kwargs)
            
            def do_GET(self):
                """Handle GET requests."""
                if self.path == '/api/status':
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    
                    stats = self.monitor.current_stats.copy()
                    stats['uptime'] = time.time() - self.monitor._start_time
                    stats['last_action'] = self.monitor.last_action
                    stats['last_action_reasoning'] = self.monitor.last_action_reasoning
                    stats['fps'] = self.monitor.current_fps
                    
                    self.wfile.write(json.dumps(stats).encode())
                    
                elif self.path == '/api/screenshot':
                    try:
                        screenshot_data = self.monitor.screenshot_queue.get_nowait()
                        self.send_response(200)
                        self.send_header('Content-type', 'application/json')
                        self.end_headers()
                        self.wfile.write(json.dumps(screenshot_data).encode())
                        self.monitor.screenshot_queue.task_done()
                    except queue.Empty:
                        self.send_response(204)  # No content
                        self.end_headers()
                
                elif self.path == '/api/metrics':
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps(self.monitor.metrics).encode())
                
                elif self.path == '/api/decisions':
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps(self.monitor.decisions).encode())
                
                else:
                    # Serve static files from templates directory
                    try:
                        file_path = self.path
                        if file_path == '/':
                            file_path = '/index.html'
                        
                        template_dir = os.path.join(os.path.dirname(__file__), 'templates')
                        full_path = os.path.join(template_dir, file_path.lstrip('/'))
                        
                        with open(full_path, 'rb') as f:
                            self.send_response(200)
                            if file_path.endswith('.html'):
                                self.send_header('Content-type', 'text/html')
                            elif file_path.endswith('.js'):
                                self.send_header('Content-type', 'application/javascript')
                            elif file_path.endswith('.css'):
                                self.send_header('Content-type', 'text/css')
                            self.end_headers()
                            self.wfile.write(f.read())
                    except (FileNotFoundError, IsADirectoryError):
                        self.send_error(404)
        
        handler = lambda *args, **kwargs: MonitorHandler(*args, monitor=self, **kwargs)
        
        with socketserver.TCPServer((host, port), handler) as httpd:
            self.logger.info(f"Web monitor running at http://{host}:{port}")
            try:
                httpd.serve_forever()
            except KeyboardInterrupt:
                self.logger.info("Web monitor stopped")
    
    def update_screenshot(self, screenshot: np.ndarray):
        """Update the current screenshot."""
        if not self.capture_active:
            return
        
        if screenshot is None:
            return
        
        try:
            # Convert screenshot to base64
            import base64
            import cv2
            
            # Convert to 3-channel RGB if needed
            if len(screenshot.shape) == 2:
                screenshot = cv2.cvtColor(screenshot, cv2.COLOR_GRAY2RGB)
            elif screenshot.shape[2] == 4:
                screenshot = screenshot[:, :, :3]
            
            # Encode as JPEG for efficiency
            _, buffer = cv2.imencode('.jpg', screenshot)
            b64_str = base64.b64encode(buffer).decode('utf-8')
            
            # Add to queue, removing old if full
            screenshot_data = {
                'image_b64': b64_str,
                'timestamp': time.time(),
            }
            
            try:
                self.screenshot_queue.put_nowait(screenshot_data)
                self.last_screenshot = screenshot
            except queue.Full:
                # Remove oldest screenshot and try again
                try:
                    self.screenshot_queue.get_nowait()
                    self.screenshot_queue.put_nowait(screenshot_data)
                except (queue.Empty, queue.Full):
                    pass
                
        except Exception as e:
            self.logger.error(f"Screenshot update error: {e}")
    
    def update_action(self, action: str, reasoning: str = None):
        """Update the last action taken."""
        self.last_action = action
        self.last_action_reasoning = reasoning
    
    def update_stats(self, stats: Dict[str, Any]):
        """Update the current statistics."""
        self.current_stats.update(stats)
    
    def update_decision(self, decision: Dict[str, Any]):
        """Add a new decision to the history."""
        if len(self.decisions) >= self.max_decisions:
            self.decisions.pop(0)
        self.decisions.append(decision)
    
    def add_performance_metric(self, metric: str, value: float):
        """Add a performance metric data point."""
        if metric in self.metrics:
            self.metrics[metric].append(value)
            # Keep only last 1000 points
            if len(self.metrics[metric]) > 1000:
                self.metrics[metric] = self.metrics[metric][-1000:]


def create_dashboard_templates():
    """Create necessary HTML/JS/CSS files for the dashboard."""
    template_dir = os.path.join(os.path.dirname(__file__), 'templates')
    os.makedirs(template_dir, exist_ok=True)
    
    # Create index.html
    index_html = """<!DOCTYPE html>
<html>
<head>
    <title>Pokemon Crystal RL Monitor</title>
    <link rel="stylesheet" href="styles.css">
</head>
<body>
    <div id="dashboard">
        <div id="header">
            <h1>Pokemon Crystal RL Monitor</h1>
            <div id="controls">
                <button id="startBtn">Start Monitor</button>
                <button id="stopBtn">Stop Monitor</button>
            </div>
        </div>
        
        <div id="main-content">
            <div id="left-panel">
                <div id="screenshot-container">
                    <h2>Game Screen</h2>
                    <img id="screenshot" src="" alt="Game screen">
                </div>
                
                <div id="stats-container">
                    <h2>Statistics</h2>
                    <div id="stats"></div>
                </div>
            </div>
            
            <div id="right-panel">
                <div id="action-history">
                    <h2>Action History</h2>
                    <div id="actions"></div>
                </div>
                
                <div id="decision-log">
                    <h2>Agent Decisions</h2>
                    <div id="decisions"></div>
                </div>
                
                <div id="metrics">
                    <h2>Training Metrics</h2>
                    <canvas id="metrics-chart"></canvas>
                </div>
            </div>
        </div>
    </div>
    
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <script src="dashboard.js"></script>
</body>
</html>"""
    
    with open(os.path.join(template_dir, 'index.html'), 'w') as f:
        f.write(index_html)
    
    # Create styles.css
    styles_css = """body {
    font-family: Arial, sans-serif;
    margin: 0;
    padding: 20px;
    background-color: #f0f0f0;
}

#dashboard {
    max-width: 1200px;
    margin: 0 auto;
    background-color: white;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    padding: 20px;
}

#header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 20px;
}

#controls button {
    padding: 8px 16px;
    margin-left: 10px;
    border: none;
    border-radius: 4px;
    cursor: pointer;
}

#startBtn {
    background-color: #4CAF50;
    color: white;
}

#stopBtn {
    background-color: #f44336;
    color: white;
}

#main-content {
    display: flex;
    gap: 20px;
}

#left-panel, #right-panel {
    flex: 1;
}

#screenshot {
    max-width: 100%;
    border: 1px solid #ddd;
    border-radius: 4px;
}

.container {
    background-color: white;
    border-radius: 4px;
    padding: 15px;
    margin-bottom: 20px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}

h2 {
    margin-top: 0;
    color: #333;
}

#actions, #decisions {
    max-height: 300px;
    overflow-y: auto;
    padding: 10px;
    background-color: #f8f8f8;
    border-radius: 4px;
}

.action-item, .decision-item {
    padding: 8px;
    border-bottom: 1px solid #eee;
}

.action-item:last-child, .decision-item:last-child {
    border-bottom: none;
}

#metrics-chart {
    width: 100%;
    height: 300px;
}"""
    
    with open(os.path.join(template_dir, 'styles.css'), 'w') as f:
        f.write(styles_css)
    
    # Create dashboard.js
    dashboard_js = """let monitorActive = false;
let updateInterval;

document.getElementById('startBtn').addEventListener('click', startMonitoring);
document.getElementById('stopBtn').addEventListener('click', stopMonitoring);

function startMonitoring() {
    monitorActive = true;
    updateInterval = setInterval(updateDashboard, 100);
    document.getElementById('startBtn').disabled = true;
    document.getElementById('stopBtn').disabled = false;
}

function stopMonitoring() {
    monitorActive = false;
    clearInterval(updateInterval);
    document.getElementById('startBtn').disabled = false;
    document.getElementById('stopBtn').disabled = true;
}

async function updateDashboard() {
    if (!monitorActive) return;
    
    try {
        // Update screenshot
        const screenshotResponse = await fetch('/api/screenshot');
        if (screenshotResponse.ok && screenshotResponse.status !== 204) {
            const screenshotData = await screenshotResponse.json();
            document.getElementById('screenshot').src = `data:image/jpeg;base64,${screenshotData.image_b64}`;
        }
        
        // Update stats
        const statsResponse = await fetch('/api/status');
        if (statsResponse.ok) {
            const stats = await statsResponse.json();
            updateStats(stats);
        }
        
        // Update decisions
        const decisionsResponse = await fetch('/api/decisions');
        if (decisionsResponse.ok) {
            const decisions = await decisionsResponse.json();
            updateDecisions(decisions);
        }
        
        // Update metrics
        const metricsResponse = await fetch('/api/metrics');
        if (metricsResponse.ok) {
            const metrics = await metricsResponse.json();
            updateMetrics(metrics);
        }
    } catch (error) {
        console.error('Dashboard update error:', error);
    }
}

function updateStats(stats) {
    const statsContainer = document.getElementById('stats');
    let html = '<table>';
    
    for (const [key, value] of Object.entries(stats)) {
        html += `<tr><td>${key}:</td><td>${value}</td></tr>`;
    }
    
    html += '</table>';
    statsContainer.innerHTML = html;
}

function updateDecisions(decisions) {
    const decisionsContainer = document.getElementById('decisions');
    let html = '';
    
    decisions.slice(-10).reverse().forEach(decision => {
        html += `<div class="decision-item">
            <strong>${decision.decision}</strong><br>
            Reasoning: ${decision.reasoning}<br>
            Confidence: ${(decision.confidence * 100).toFixed(1)}%
        </div>`;
    });
    
    decisionsContainer.innerHTML = html;
}

function updateMetrics(metrics) {
    const data = [{
        x: Array.from({length: metrics.reward.length}, (_, i) => i),
        y: metrics.reward,
        name: 'Reward',
        type: 'scatter'
    }];
    
    const layout = {
        title: 'Training Progress',
        xaxis: {title: 'Steps'},
        yaxis: {title: 'Reward'}
    };
    
    Plotly.newPlot('metrics-chart', data, layout);
}

// Initial load
startMonitoring();"""
    
    with open(os.path.join(template_dir, 'dashboard.js'), 'w') as f:
        f.write(dashboard_js)
