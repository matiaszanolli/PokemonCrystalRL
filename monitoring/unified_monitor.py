"""Unified web monitoring server for Pokemon Crystal RL.

This implementation combines the best features from various monitoring implementations:
- Real-time WebSocket updates for game state, screenshots, and decisions
- Performance analytics and system monitoring
- Memory-efficient data management
- Comprehensive API endpoints
- Robust error handling
"""

import os
import sys
import time
import json
import base64
import logging
import threading
from datetime import datetime
from collections import defaultdict, deque
from typing import Dict, List, Any, Optional
from queue import Queue

try:
    from flask import Flask, render_template, request, jsonify, send_from_directory
    from flask_socketio import SocketIO, emit
except ImportError as e:
    print(f"❌ Required package missing: {e}")
    print("Install required packages with: pip install flask flask-socketio")
    sys.exit(1)

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

import cv2
import numpy as np

# Import TrainingState from web_monitor
from .web_monitor import TrainingState


class UnifiedMonitor:
    """Unified web monitoring server for Pokemon Crystal RL training."""
    
    def __init__(self, training_session=None, host='127.0.0.1', port=5000, config=None):
        self.training_session = training_session
        self.config = config
        self.host = host
        self.port = port
        self.logger = logging.getLogger(__name__)
        
        # Initialize training state and database
        self.training_state = TrainingState.INITIALIZING
        self.current_run_id = None
        self.db = None
        
        # Initialize database if config provided
        if self.config and hasattr(self.config, 'db_path'):
            try:
                from .database import DatabaseManager
                self.db = DatabaseManager(self.config.db_path)
            except ImportError as e:
                self.logger.warning(f"Database module not available: {e}")
            except Exception as e:
                self.logger.warning(f"Database initialization failed: {e}")
                self.db = None
        
        # Initialize error handler
        try:
            from .error_handler import ErrorHandler
            self.error_handler = ErrorHandler()
        except ImportError:
            self.error_handler = None
            self.logger.warning("Error handler not available")
        
        # Find template directory relative to this file
        template_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'static', 'templates')
        if not os.path.exists(template_dir):
            template_dir = 'templates'  # Fallback to local templates
        
        # Flask app setup
        self.app = Flask(__name__, 
                        template_folder=template_dir,
                        static_folder='static')
        self.app.config['SECRET_KEY'] = 'pokemon_rl_monitor'
        self.socketio = SocketIO(self.app, cors_allowed_origins="*", async_mode='threading')
        
        # Data storage with memory management
        self.current_stats = {}
        self.episode_data = deque(maxlen=100)  # Keep last 100 episodes
        self.performance_metrics = deque(maxlen=1000)  # Keep last 1000 metrics
        self.text_frequency = defaultdict(int)
        self.recent_text = deque(maxlen=100)
        self.recent_actions = deque(maxlen=100)
        self.recent_decisions = deque(maxlen=50)
        
        # Real-time data queues
        self.screen_queue = Queue(maxsize=10)
        self.stats_queue = Queue(maxsize=50)
        self.action_queue = Queue(maxsize=100)
        
        # Performance tracking
        self.session_start_time = time.time()
        self.last_update_time = time.time()
        self.update_count = 0
        self.is_monitoring = False
        
        # Memory management
        self.max_memory_usage = 500 * 1024 * 1024  # 500MB limit
        self.cleanup_interval = 300  # Cleanup every 5 minutes
        self.last_cleanup = time.time()
        
        # Setup routes and events
        self._setup_routes()
        self._setup_socket_events()
        
        self.logger.info("🚀 Unified web monitor initialized")
    
    def _setup_routes(self):
        """Setup Flask routes."""
        
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
                'uptime': time.time() - self.session_start_time,
                'timestamp': datetime.now().isoformat()
            })
        
        @self.app.route('/api/stats')
        def get_stats():
            """Get current training statistics"""
            try:
                if self.training_session:
                    stats = self.training_session.get_stats()
                else:
                    stats = self.current_stats
                return jsonify(stats)
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/screenshot')
        def get_screenshot():
            """Get latest screenshot"""
            try:
                if not self.screen_queue.empty():
                    screenshot_data = self.screen_queue.get()
                    return jsonify(screenshot_data)
                return jsonify({'error': 'No screenshot available'}), 404
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/text')
        def get_text():
            """Get text recognition data"""
            return jsonify({
                'recent_text': list(self.recent_text),
                'text_frequency': dict(self.text_frequency)
            })
        
        @self.app.route('/api/system')
        def get_system():
            """Get system resource statistics"""
            if not PSUTIL_AVAILABLE:
                return jsonify({'error': 'psutil not available'}), 503
            
            try:
                stats = {
                    'cpu_percent': psutil.cpu_percent(interval=0.1),
                    'memory_percent': psutil.virtual_memory().percent,
                    'memory_used': psutil.virtual_memory().used // (1024*1024),  # MB
                    'memory_total': psutil.virtual_memory().total // (1024*1024),  # MB
                    'disk_usage': psutil.disk_usage('/').percent,
                    'disk_free': psutil.disk_usage('/').free // (1024*1024*1024),  # GB
                    'timestamp': datetime.now().isoformat()
                }
                return jsonify(stats)
            except Exception as e:
                return jsonify({'error': str(e)}), 500
    
    def _setup_socket_events(self):
        """Setup WebSocket event handlers."""
        
        @self.socketio.on('connect')
        def handle_connect():
            self.logger.info(f"🔌 Client connected: {request.sid}")
            # Send initial data
            emit('status', {
                'monitoring': self.is_monitoring,
                'message': 'Connected to Pokemon RL Monitor'
            })
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            self.logger.info(f"🔌 Client disconnected: {request.sid}")
        
        @self.socketio.on('start_monitoring')
        def handle_start():
            self.start_monitoring()
            emit('status', {'monitoring': True, 'message': 'Monitoring started'})
        
        @self.socketio.on('stop_monitoring')
        def handle_stop():
            self.stop_monitoring()
            emit('status', {'monitoring': False, 'message': 'Monitoring stopped'})
        
        @self.socketio.on('request_screenshot')
        def handle_screenshot():
            if not self.screen_queue.empty():
                emit('screenshot', self.screen_queue.get())
    
    def start_monitoring(self):
        """Start the monitoring process."""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.training_state = TrainingState.RUNNING
        self.logger.info("🎯 Monitoring started")
        
        # Start monitoring thread
        threading.Thread(target=self._monitoring_loop, daemon=True).start()
    
    def stop_monitoring(self):
        """Stop the monitoring process."""
        self.is_monitoring = False
        self.training_state = TrainingState.STOPPED
        self.logger.info("⏹️ Monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                if self.training_session:
                    self._update_training_stats()
                
                # Emit current stats
                self.socketio.emit('stats_update', self.current_stats)
                
                # Emit system stats if available
                if PSUTIL_AVAILABLE:
                    self.socketio.emit('system_update', self._get_system_stats())
                
                time.sleep(0.5)  # Update every 500ms
                
            except Exception as e:
                self.logger.error(f"❌ Monitoring error: {e}")
                time.sleep(1)
    
    def _update_training_stats(self):
        """Update current training statistics."""
        try:
            if self.training_session:
                stats = self.training_session.get_stats()
                self.current_stats.update(stats)
        except Exception as e:
            self.logger.error(f"⚠️ Stats update error: {e}")
    
    def _get_system_stats(self):
        """Get system resource statistics."""
        if not PSUTIL_AVAILABLE:
            return {}
        
        try:
            return {
                'cpu_percent': psutil.cpu_percent(interval=0.1),
                'memory_percent': psutil.virtual_memory().percent,
                'memory_used': psutil.virtual_memory().used // (1024*1024),  # MB
                'memory_total': psutil.virtual_memory().total // (1024*1024),  # MB
                'disk_usage': psutil.disk_usage('/').percent,
                'disk_free': psutil.disk_usage('/').free // (1024*1024*1024),  # GB
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.error(f"⚠️ System stats error: {e}")
            return {}
    
    def update_screenshot(self, screenshot: np.ndarray):
        """Update screenshot for web streaming."""
        try:
            # Resize for web display
            height, width = screenshot.shape[:2]
            scale = 3  # Make it bigger for web display
            resized = cv2.resize(screenshot, (width * scale, height * scale),
                               interpolation=cv2.INTER_NEAREST)
            
            # Convert to base64
            _, buffer = cv2.imencode('.png', cv2.cvtColor(resized, cv2.COLOR_RGB2BGR))
            b64_string = base64.b64encode(buffer).decode('utf-8')
            
            screenshot_data = {
                'image': f"data:image/png;base64,{b64_string}",
                'timestamp': datetime.now().isoformat(),
                'dimensions': {'width': width * scale, 'height': height * scale}
            }
            
            # Update queue
            if self.screen_queue.full():
                try:
                    self.screen_queue.get_nowait()
                except:
                    pass
            
            self.screen_queue.put(screenshot_data)
            
            # Emit to clients if monitoring
            if self.is_monitoring:
                self.socketio.emit('screenshot', screenshot_data)
                
        except Exception as e:
            self.logger.error(f"⚠️ Screenshot update error: {e}")
    
    def update_text(self, text: str, location: str = 'unknown'):
        """Update text recognition data."""
        if not text or not text.strip():
            return
        
        text = text.strip()
        self.text_frequency[text] += 1
        
        text_data = {
            'text': text,
            'location': location,
            'timestamp': datetime.now().isoformat()
        }
        
        self.recent_text.append(text_data)
        
        if self.is_monitoring:
            self.socketio.emit('text_update', text_data)
    
    def update_action(self, action: str, details: str = ""):
        """Update action history."""
        action_data = {
            'action': action,
            'details': details,
            'timestamp': datetime.now().isoformat()
        }
        
        self.recent_actions.append(action_data)
        
        if self.is_monitoring:
            self.socketio.emit('action_update', action_data)
    
    def update_decision(self, decision: Dict[str, Any]):
        """Update agent decision history."""
        decision['timestamp'] = datetime.now().isoformat()
        self.recent_decisions.append(decision)
        
        if self.is_monitoring:
            self.socketio.emit('decision_update', decision)
    
    def cleanup(self):
        """Clean up old data to manage memory usage."""
        current_time = time.time()
        if current_time - self.last_cleanup < self.cleanup_interval:
            return
        
        self.last_cleanup = current_time
        
        # Keep only most frequent text entries
        if len(self.text_frequency) > 1000:
            sorted_items = sorted(self.text_frequency.items(), 
                               key=lambda x: x[1], reverse=True)[:1000]
            self.text_frequency = defaultdict(int, dict(sorted_items))
        
        self.logger.info("🧹 Memory cleanup completed")
    
    def run(self, debug=False):
        """Run the monitoring server."""
        self.logger.info(f"🌐 Starting web monitor on http://{self.host}:{self.port}")
        self.socketio.run(self.app, host=self.host, port=self.port, debug=debug)


def main():
    """Main entry point when run as a script."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Pokemon Crystal RL Web Monitor')
    parser.add_argument('--host', default='127.0.0.1', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)8s | %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Create and run monitor
    monitor = UnifiedMonitor(host=args.host, port=args.port)
    try:
        monitor.run(debug=args.debug)
    except KeyboardInterrupt:
        logging.info("\n🛑 Server stopped by user")
    except Exception as e:
        logging.error(f"❌ Server error: {e}")


if __name__ == '__main__':
    main()
