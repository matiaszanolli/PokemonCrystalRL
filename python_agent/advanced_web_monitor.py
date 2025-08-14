#!/usr/bin/env python3
"""
advanced_web_monitor.py - Advanced Web Monitoring Server for Pokemon Crystal RL

This server provides comprehensive real-time monitoring for Pokemon Crystal RL training:
- Real-time WebSocket updates for game state, screenshots, and decisions
- REST API for training control and data access
- Advanced statistics tracking and memory management
- Performance analytics and system monitoring
- Text analysis and frequency tracking
"""

import os
import sys
import time
import json
import base64
import sqlite3
import threading
from datetime import datetime, timedelta
from collections import defaultdict, deque
from typing import Dict, List, Any, Optional
import psutil

try:
    from flask import Flask, render_template, request, jsonify, send_from_directory
    from flask_socketio import SocketIO, emit
except ImportError:
    print("❌ Flask and Flask-SocketIO are required for the monitoring server")
    print("Install them with: pip install flask flask-socketio")
    sys.exit(1)


class AdvancedWebMonitor:
    """
    Advanced web monitoring server with comprehensive analytics
    """
    
    def __init__(self, host='127.0.0.1', port=5000):
        self.host = host
        self.port = port
        
        # Flask app setup
        self.app = Flask(__name__, template_folder='templates', static_folder='static')
        self.app.config['SECRET_KEY'] = 'pokemon_crystal_rl_monitor'
        self.socketio = SocketIO(self.app, cors_allowed_origins="*", async_mode='threading')
        
        # Data storage
        self.current_stats = {}
        self.episode_data = []
        self.performance_history = deque(maxlen=1000)
        self.text_frequency = defaultdict(int)
        self.recent_text = deque(maxlen=100)
        self.recent_actions = deque(maxlen=100)
        self.recent_decisions = deque(maxlen=50)
        self.screenshot_data = None
        self.system_stats = {}
        
        # Performance tracking
        self.session_start_time = time.time()
        self.last_update_time = time.time()
        self.update_count = 0
        
        # Memory management
        self.max_memory_usage = 500 * 1024 * 1024  # 500MB limit
        self.cleanup_interval = 300  # Cleanup every 5 minutes
        self.last_cleanup = time.time()
        
        # Setup routes and events
        self._setup_routes()
        self._setup_socketio_events()
        
        print(f"🚀 Advanced web monitor initialized on {host}:{port}")
    
    def _setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/')
        def dashboard():
            """Main dashboard page"""
            return render_template('dashboard.html')
        
        @self.app.route('/api/status')
        def status():
            """Server status endpoint"""
            return jsonify({
                'status': 'running',
                'uptime': time.time() - self.session_start_time,
                'version': '2.0.0',
                'features': ['realtime', 'analytics', 'vision', 'control']
            })
        
        @self.app.route('/api/stats')
        def get_stats():
            """Get current training statistics"""
            return jsonify(self.current_stats)
        
        @self.app.route('/api/performance', methods=['GET', 'POST'])
        def get_performance():
            """Get/Update performance history"""
            if request.method == 'POST':
                # Handle performance update
                data = request.get_json() or {}
                # Update internal performance data if needed
                return jsonify({'status': 'success'})
            else:
                # Return performance data
                return jsonify({
                    'episode_rewards': [ep.get('total_reward', 0) for ep in self.episode_data[-50:]],
                    'episode_steps': [ep.get('steps', 0) for ep in self.episode_data[-50:]],
                    'avg_reward': sum(ep.get('total_reward', 0) for ep in self.episode_data[-10:]) / max(len(self.episode_data[-10:]), 1),
                    'avg_steps': sum(ep.get('steps', 0) for ep in self.episode_data[-10:]) / max(len(self.episode_data[-10:]), 1),
                    'success_rate': sum(1 for ep in self.episode_data[-10:] if ep.get('success', False)) / max(len(self.episode_data[-10:]), 1) * 100,
                    'total_episodes': len(self.episode_data)
                })
        
        @self.app.route('/api/screenshot')
        def get_screenshot():
            """Get latest screenshot"""
            if self.screenshot_data:
                return jsonify({'image': self.screenshot_data})
            return jsonify({'error': 'No screenshot available'}), 404
        
        @self.app.route('/api/text')
        def get_text_data():
            """Get text analysis data"""
            return jsonify({
                'frequency': dict(self.text_frequency),
                'recent': list(self.recent_text)[-20:]
            })
        
        @self.app.route('/api/actions')
        def get_actions():
            """Get recent actions"""
            return jsonify(list(self.recent_actions)[-50:])
        
        @self.app.route('/api/decisions')
        def get_decisions():
            """Get recent LLM decisions"""
            return jsonify(list(self.recent_decisions)[-20:])
        
        # Control endpoints
        @self.app.route('/api/control/pause', methods=['POST'])
        def control_pause():
            """Pause/resume training"""
            # This would need to be implemented in the training script
            return jsonify({'status': 'acknowledged', 'action': 'pause'})
        
        @self.app.route('/api/control/save_state', methods=['POST'])
        def control_save_state():
            """Save current game state"""
            return jsonify({'status': 'acknowledged', 'action': 'save_state'})
        
        @self.app.route('/api/control/reset', methods=['POST'])
        def control_reset():
            """Reset training"""
            return jsonify({'status': 'acknowledged', 'action': 'reset'})
        
        # Data update endpoints for monitoring client
        @self.app.route('/api/episode', methods=['POST'])
        def update_episode():
            """Update episode data"""
            data = request.get_json() or {}
            self.update_episode(data)
            return jsonify({'status': 'success'})
        
        @self.app.route('/api/step', methods=['POST'])
        def update_step():
            """Update step data"""
            data = request.get_json() or {}
            self.update_stats(data)
            return jsonify({'status': 'success'})
        
        @self.app.route('/api/screenshot', methods=['POST'])
        def update_screenshot_endpoint():
            """Update screenshot data"""
            data = request.get_json() or {}
            if 'image' in data:
                self.update_screenshot(data['image'])
            return jsonify({'status': 'success'})
        
        @self.app.route('/api/decision', methods=['POST'])
        def update_decision():
            """Update LLM decision data"""
            data = request.get_json() or {}
            self.update_decision(data)
            return jsonify({'status': 'success'})
        
        @self.app.route('/api/text', methods=['POST'])
        def update_text_endpoint():
            """Update text data"""
            data = request.get_json() or {}
            if 'text' in data:
                self.update_text(data['text'], data.get('type', 'dialogue'))
            return jsonify({'status': 'success'})
        
        @self.app.route('/api/system', methods=['POST'])
        def update_system():
            """Update system stats"""
            data = request.get_json() or {}
            self.system_stats.update(data)
            self.socketio.emit('system_update', self.system_stats)
            return jsonify({'status': 'success'})
    
    def _setup_socketio_events(self):
        """Setup SocketIO event handlers"""
        
        @self.socketio.on('connect')
        def handle_connect():
            print(f"📱 Client connected: {request.sid}")
            # Send initial data
            emit('stats_update', self.current_stats)
            if self.screenshot_data:
                emit('screenshot_update', {'image': self.screenshot_data})
            emit('performance_update', self._get_performance_data())
            emit('system_update', self.system_stats)
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            print(f"📱 Client disconnected: {request.sid}")
        
        @self.socketio.on('request_update')
        def handle_request_update():
            """Client requesting data update"""
            self._broadcast_all_data()
    
    def _get_performance_data(self) -> Dict[str, Any]:
        """Get comprehensive performance data"""
        if not self.episode_data:
            return {
                'avg_reward': 0.0,
                'avg_steps': 0,
                'success_rate': 0.0,
                'actions_per_sec': 0.0,
                'episode_rewards': [],
                'episode_steps': []
            }
        
        recent_episodes = self.episode_data[-50:]
        rewards = [ep.get('total_reward', 0) for ep in recent_episodes]
        steps = [ep.get('steps', 0) for ep in recent_episodes]
        
        # Calculate actions per second
        current_time = time.time()
        time_diff = current_time - self.last_update_time
        actions_per_sec = 1.0 / max(time_diff, 0.1) if time_diff > 0 else 0.0
        
        return {
            'avg_reward': sum(rewards) / len(rewards),
            'avg_steps': sum(steps) / len(steps),
            'success_rate': sum(1 for ep in recent_episodes if ep.get('success', False)) / len(recent_episodes) * 100,
            'actions_per_sec': actions_per_sec,
            'episode_rewards': rewards,
            'episode_steps': steps
        }
    
    def _update_system_stats(self):
        """Update system performance statistics"""
        try:
            # CPU and memory usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            
            # Calculate uptime
            uptime_seconds = time.time() - self.session_start_time
            uptime_str = f"{int(uptime_seconds//3600):02d}:{int((uptime_seconds%3600)//60):02d}:{int(uptime_seconds%60):02d}"
            
            # Calculate FPS (approximate)
            fps = 1.0 / max(time.time() - self.last_update_time, 0.1)
            
            self.system_stats = {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_used_mb': memory.used / 1024 / 1024,
                'memory_total_mb': memory.total / 1024 / 1024,
                'uptime': uptime_str,
                'fps': min(fps, 999.9),  # Cap at reasonable value
                'update_count': self.update_count
            }
            
        except Exception as e:
            print(f"⚠️ System stats update failed: {e}")
            self.system_stats = {
                'cpu_percent': 0,
                'memory_percent': 0,
                'uptime': '00:00:00',
                'fps': 0,
                'update_count': self.update_count
            }
    
    def _cleanup_memory(self):
        """Cleanup old data to manage memory usage"""
        current_time = time.time()
        
        # Only cleanup every cleanup_interval seconds
        if current_time - self.last_cleanup < self.cleanup_interval:
            return
        
        self.last_cleanup = current_time
        
        # Trim episode data (keep last 100 episodes)
        if len(self.episode_data) > 100:
            self.episode_data = self.episode_data[-100:]
        
        # Trim text frequency (keep most frequent 1000 entries)
        if len(self.text_frequency) > 1000:
            sorted_items = sorted(self.text_frequency.items(), key=lambda x: x[1], reverse=True)
            self.text_frequency = defaultdict(int, dict(sorted_items[:1000]))
        
        print(f"🧹 Memory cleanup completed - Episodes: {len(self.episode_data)}, Text entries: {len(self.text_frequency)}")
    
    def _broadcast_all_data(self):
        """Broadcast all current data to connected clients"""
        self.socketio.emit('stats_update', self.current_stats)
        self.socketio.emit('performance_update', self._get_performance_data())
        self.socketio.emit('system_update', self.system_stats)
        
        if self.screenshot_data:
            self.socketio.emit('screenshot_update', {'image': self.screenshot_data})
        
        if self.text_frequency:
            self.socketio.emit('text_update', {
                'text': list(self.recent_text)[-1] if self.recent_text else '',
                'frequency': dict(self.text_frequency)
            })
        
        if self.recent_actions:
            self.socketio.emit('action_update', {
                'action': list(self.recent_actions)[-1] if self.recent_actions else ''
            })
        
        if self.recent_decisions:
            latest_decision = list(self.recent_decisions)[-1] if self.recent_decisions else {}
            self.socketio.emit('decision_update', latest_decision)
    
    # Public API methods for external integration
    
    def update_stats(self, stats: Dict[str, Any]):
        """Update current training statistics"""
        self.current_stats.update(stats)
        self._update_system_stats()
        self.update_count += 1
        self.last_update_time = time.time()
        
        # Broadcast to clients
        self.socketio.emit('stats_update', self.current_stats)
        self.socketio.emit('system_update', self.system_stats)
        
        # Cleanup memory if needed
        self._cleanup_memory()
    
    def update_episode(self, episode_data: Dict[str, Any]):
        """Update episode completion data"""
        episode_data['timestamp'] = datetime.now().isoformat()
        self.episode_data.append(episode_data)
        
        # Broadcast performance update
        self.socketio.emit('performance_update', self._get_performance_data())
    
    def update_screenshot(self, image_data: str):
        """Update game screenshot"""
        self.screenshot_data = image_data
        self.socketio.emit('screenshot_update', {'image': image_data})
    
    def update_action(self, action: str):
        """Update recent action"""
        action_data = {
            'action': action,
            'timestamp': datetime.now().isoformat()
        }
        self.recent_actions.append(action_data)
        self.socketio.emit('action_update', {'action': action})
    
    def update_decision(self, decision_data: Dict[str, Any]):
        """Update LLM decision"""
        decision_data['timestamp'] = datetime.now().isoformat()
        self.recent_decisions.append(decision_data)
        self.socketio.emit('decision_update', decision_data)
    
    def update_text(self, text: str, text_type: str = 'dialogue'):
        """Update detected text"""
        if text and len(text.strip()) > 0:
            clean_text = text.strip().upper()
            self.text_frequency[clean_text] += 1
            
            text_data = {
                'text': text,
                'type': text_type,
                'timestamp': datetime.now().isoformat()
            }
            self.recent_text.append(text_data)
            
            # Broadcast text update
            self.socketio.emit('text_update', {
                'text': text,
                'frequency': dict(list(self.text_frequency.items())[-20:])
            })
    
    def get_status(self) -> Dict[str, Any]:
        """Get monitoring server status"""
        return {
            'running': True,
            'uptime': time.time() - self.session_start_time,
            'connected_clients': len(self.socketio.server.manager.rooms.get('/', {})),
            'total_episodes': len(self.episode_data),
            'total_updates': self.update_count,
            'memory_usage_mb': psutil.Process().memory_info().rss / 1024 / 1024
        }
    
    def run(self, debug=False):
        """Run the monitoring server"""
        print(f"🌐 Starting advanced web monitor on {self.host}:{self.port}")
        print(f"📊 Dashboard will be available at: http://{self.host}:{self.port}")
        
        # Create templates directory if it doesn't exist
        os.makedirs('templates', exist_ok=True)
        
        # Start the server
        self.socketio.run(
            self.app,
            host=self.host,
            port=self.port,
            debug=debug,
            allow_unsafe_werkzeug=True
        )


def main():
    """Main entry point for standalone server"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Pokemon Crystal RL Advanced Web Monitor')
    parser.add_argument('--host', default='127.0.0.1', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    # Create monitor instance
    monitor = AdvancedWebMonitor(host=args.host, port=args.port)
    
    try:
        # Run the server
        monitor.run(debug=args.debug)
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Server error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
