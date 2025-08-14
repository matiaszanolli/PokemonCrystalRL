"""
enhanced_web_monitor.py - Advanced Pokemon Crystal RL Web Monitoring Dashboard

This module provides a comprehensive, modern web interface for monitoring
Pokemon Crystal RL training with enhanced statistics, memory management,
and clean UI design.
"""

import json
import time
import base64
import sqlite3
import threading
from datetime import datetime, timedelta
from collections import deque, defaultdict, Counter
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import psutil
import numpy as np

from flask import Flask, render_template, jsonify, request, send_from_directory
from flask_socketio import SocketIO, emit

# Import text logger for integration
try:
    from text_logger import PokemonTextLogger
except ImportError:
    PokemonTextLogger = None


@dataclass
class SessionStats:
    """Comprehensive session statistics"""
    session_id: str
    start_time: datetime
    duration: float
    total_steps: int
    total_episodes: int
    llm_calls: int
    visual_analyses: int
    text_detections: int
    unique_text: int
    avg_reward: float
    actions_per_second: float
    memory_usage_mb: float
    dialogue_count: int
    screen_types: Dict[str, int]
    text_locations: Dict[str, int]


@dataclass
class PerformanceMetric:
    """Real-time performance tracking"""
    timestamp: datetime
    metric_name: str
    value: float
    session_id: str


class EnhancedWebMonitor:
    """
    Advanced web monitoring dashboard with comprehensive statistics
    """
    
    def __init__(self, data_retention_hours: int = 24, max_memory_mb: int = 500):
        """Initialize enhanced web monitor"""
        
        # Flask and SocketIO setup
        self.app = Flask(__name__, template_folder='templates', static_folder='static')
        self.app.config['SECRET_KEY'] = 'pokemon_crystal_rl_monitor_2024'
        self.socketio = SocketIO(self.app, cors_allowed_origins="*", 
                                logger=False, engineio_logger=False)
        
        # Memory management
        self.data_retention_hours = data_retention_hours
        self.max_memory_mb = max_memory_mb
        self.cleanup_interval = 300  # 5 minutes
        
        # Enhanced data storage
        self.current_session = {
            'session_id': f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'start_time': datetime.now(),
            'is_active': False
        }
        
        # Real-time data (with size limits)
        self.screenshots = deque(maxlen=10)  # Last 10 screenshots
        self.actions = deque(maxlen=100)  # Last 100 actions
        self.decisions = deque(maxlen=50)  # Last 50 LLM decisions
        self.performance_metrics = deque(maxlen=1000)  # Last 1000 metric points
        
        # Session history and statistics
        self.session_history = deque(maxlen=20)  # Last 20 sessions
        self.session_stats = {}  # Current session comprehensive stats
        
        # Text and dialogue tracking
        self.dialogue_history = deque(maxlen=200)  # Recent dialogue
        self.text_frequency = Counter()  # All-time text frequency
        self.recent_text = deque(maxlen=50)  # Recent text detections
        
        # Performance tracking
        self.episode_rewards = deque(maxlen=100)  # Last 100 episode rewards
        self.training_progress = {
            'total_episodes': 0,
            'total_steps': 0,
            'total_training_time': 0,
            'sessions_count': 0
        }
        
        # Memory and system monitoring
        self.memory_usage_history = deque(maxlen=100)
        self.system_stats = {
            'cpu_percent': 0,
            'memory_percent': 0,
            'memory_usage_mb': 0
        }
        
        # Text logger integration
        self.text_logger: Optional[PokemonTextLogger] = None
        
        # Background tasks
        self.cleanup_thread = None
        self.monitoring_active = False
        
        # Register routes
        self._register_routes()
        self._register_socketio_events()
        
        print("🚀 Enhanced Web Monitor initialized")
    
    def set_text_logger(self, text_logger: PokemonTextLogger):
        """Integrate with text logging system"""
        self.text_logger = text_logger
        print("📝 Text logger integration enabled")
    
    def _register_routes(self):
        """Register Flask routes"""
        
        @self.app.route('/')
        def dashboard():
            """Main dashboard"""
            return render_template('enhanced_dashboard.html')
        
        @self.app.route('/api/session/stats')
        def get_session_stats():
            """Get comprehensive session statistics"""
            stats = self._get_comprehensive_stats()
            return jsonify(stats)
        
        @self.app.route('/api/performance/metrics')
        def get_performance_metrics():
            """Get performance metrics for charts"""
            metrics = self._get_performance_data()
            return jsonify(metrics)
        
        @self.app.route('/api/text/frequency')
        def get_text_frequency():
            """Get text frequency analysis"""
            return jsonify(self._get_text_analysis())
        
        @self.app.route('/api/text/dialogue')
        def get_dialogue_history():
            """Get recent dialogue history"""
            count = request.args.get('count', 20, type=int)
            dialogue = list(self.dialogue_history)[-count:]
            return jsonify(dialogue)
        
        @self.app.route('/api/session/history')
        def get_session_history():
            """Get session history"""
            return jsonify([asdict(session) for session in self.session_history])
        
        @self.app.route('/api/system/status')
        def get_system_status():
            """Get system performance status"""
            return jsonify(self._get_system_status())
        
        @self.app.route('/api/screenshot/current')
        def get_current_screenshot():
            """Get latest screenshot"""
            if self.screenshots:
                return jsonify({'screenshot': self.screenshots[-1]})
            return jsonify({'screenshot': None})
        
        @self.app.route('/static/<path:filename>')
        def static_files(filename):
            """Serve static files"""
            return send_from_directory('static', filename)
    
    def _register_socketio_events(self):
        """Register SocketIO events"""
        
        @self.socketio.on('connect')
        def handle_connect():
            """Handle client connection"""
            emit('status', {'connected': True, 'session_id': self.current_session['session_id']})
        
        @self.socketio.on('request_stats')
        def handle_stats_request():
            """Handle stats request"""
            stats = self._get_comprehensive_stats()
            emit('stats_update', stats)
        
        @self.socketio.on('request_text_search')
        def handle_text_search(data):
            """Handle text search request"""
            query = data.get('query', '')
            results = self._search_text(query)
            emit('text_search_results', results)
    
    def start_monitoring(self):
        """Start monitoring session"""
        self.monitoring_active = True
        self.current_session['is_active'] = True
        self.current_session['start_time'] = datetime.now()
        
        # Start cleanup thread
        if not self.cleanup_thread or not self.cleanup_thread.is_alive():
            self.cleanup_thread = threading.Thread(target=self._cleanup_worker, daemon=True)
            self.cleanup_thread.start()
        
        print("📊 Enhanced monitoring started")
    
    def stop_monitoring(self):
        """Stop monitoring session"""
        self.monitoring_active = False
        self.current_session['is_active'] = False
        
        # Archive current session
        self._archive_current_session()
        
        print("📊 Enhanced monitoring stopped")
    
    def update_screenshot(self, screenshot: np.ndarray):
        """Update current screenshot"""
        try:
            # Convert to base64
            from PIL import Image
            import io
            
            pil_image = Image.fromarray(screenshot)
            upscaled = pil_image.resize((320, 288), Image.NEAREST)
            
            buffer = io.BytesIO()
            upscaled.save(buffer, format='PNG')
            buffer.seek(0)
            encoded = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            self.screenshots.append(encoded)
            
            # Emit to connected clients
            self.socketio.emit('screenshot_update', {'screenshot': encoded})
            
        except Exception as e:
            print(f"⚠️ Screenshot update error: {e}")
    
    def update_action(self, action: str, reasoning: str = "", confidence: float = 1.0):
        """Update current action"""
        action_data = {
            'action': action,
            'reasoning': reasoning,
            'confidence': confidence,
            'timestamp': datetime.now().isoformat()
        }
        
        self.actions.append(action_data)
        self.socketio.emit('action_update', action_data)
    
    def update_decision(self, decision_data: Dict[str, Any]):
        """Update LLM decision"""
        decision_data['timestamp'] = datetime.now().isoformat()
        self.decisions.append(decision_data)
        self.socketio.emit('decision_update', decision_data)
    
    def add_performance_metric(self, metric_name: str, value: float):
        """Add performance metric"""
        metric = PerformanceMetric(
            timestamp=datetime.now(),
            metric_name=metric_name,
            value=value,
            session_id=self.current_session['session_id']
        )
        
        self.performance_metrics.append(metric)
        
        # Update system stats
        self._update_system_stats()
        
        # Emit real-time update
        self.socketio.emit('metric_update', {
            'metric': metric_name,
            'value': value,
            'timestamp': metric.timestamp.isoformat()
        })
    
    def add_episode_reward(self, reward: float, episode: int):
        """Add episode reward"""
        reward_data = {
            'reward': reward,
            'episode': episode,
            'timestamp': datetime.now().isoformat()
        }
        
        self.episode_rewards.append(reward_data)
        self.training_progress['total_episodes'] = episode
        
        self.socketio.emit('reward_update', reward_data)
    
    def update_text_detection(self, text: str, location: str, screen_type: str):
        """Update text detection"""
        text_data = {
            'text': text,
            'location': location,
            'screen_type': screen_type,
            'timestamp': datetime.now().isoformat()
        }
        
        self.recent_text.append(text_data)
        self.text_frequency[text] += 1
        
        # Track dialogue separately
        if location == 'dialogue':
            self.dialogue_history.append(text_data)
        
        self.socketio.emit('text_update', text_data)
    
    def _get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive session statistics"""
        now = datetime.now()
        session_duration = (now - self.current_session['start_time']).total_seconds()
        
        # Basic stats
        basic_stats = {
            'session_id': self.current_session['session_id'],
            'session_active': self.current_session['is_active'],
            'session_duration': session_duration,
            'total_episodes': self.training_progress['total_episodes'],
            'total_steps': self.training_progress['total_steps'],
            'total_sessions': len(self.session_history) + 1
        }
        
        # Performance stats
        performance_stats = {
            'llm_calls': len(self.decisions),
            'visual_analyses': len([m for m in self.performance_metrics if m.metric_name == 'visual_analysis']),
            'text_detections': len(self.recent_text),
            'unique_texts': len(self.text_frequency),
            'dialogue_count': len(self.dialogue_history)
        }
        
        # Recent performance
        recent_metrics = list(self.performance_metrics)[-10:] if self.performance_metrics else []
        avg_actions_per_second = np.mean([m.value for m in recent_metrics if m.metric_name == 'steps_per_second']) if recent_metrics else 0
        
        performance_stats.update({
            'avg_actions_per_second': float(avg_actions_per_second),
            'memory_usage_mb': self.system_stats['memory_usage_mb'],
            'cpu_percent': self.system_stats['cpu_percent']
        })
        
        # Episode statistics
        episode_stats = {
            'total_episodes': len(self.episode_rewards),
            'avg_reward': float(np.mean([r['reward'] for r in self.episode_rewards])) if self.episode_rewards else 0,
            'best_reward': float(max([r['reward'] for r in self.episode_rewards])) if self.episode_rewards else 0,
            'recent_trend': self._calculate_reward_trend()
        }
        
        # Text analysis
        text_stats = {
            'total_text_detections': len(self.recent_text),
            'unique_text_count': len(self.text_frequency),
            'dialogue_count': len(self.dialogue_history),
            'most_common_text': self.text_frequency.most_common(5),
            'text_by_location': self._get_text_by_location()
        }
        
        return {
            'basic': basic_stats,
            'performance': performance_stats,
            'episodes': episode_stats,
            'text': text_stats,
            'system': self.system_stats,
            'timestamp': now.isoformat()
        }
    
    def _get_performance_data(self) -> Dict[str, List]:
        """Get performance data for charts"""
        metrics_by_name = defaultdict(list)
        
        for metric in self.performance_metrics:
            metrics_by_name[metric.metric_name].append({
                'timestamp': metric.timestamp.isoformat(),
                'value': metric.value
            })
        
        # Add episode rewards
        metrics_by_name['episode_rewards'] = [
            {
                'timestamp': r['timestamp'],
                'value': r['reward'],
                'episode': r['episode']
            }
            for r in self.episode_rewards
        ]
        
        # Add memory usage
        metrics_by_name['memory_usage'] = [
            {
                'timestamp': m['timestamp'],
                'value': m['memory_mb']
            }
            for m in self.memory_usage_history
        ]
        
        return dict(metrics_by_name)
    
    def _get_text_analysis(self) -> Dict[str, Any]:
        """Get text frequency and analysis"""
        # Most common text
        common_text = self.text_frequency.most_common(20)
        
        # Text by location
        location_counts = defaultdict(int)
        for text_data in self.recent_text:
            location_counts[text_data['location']] += 1
        
        # Recent dialogue
        recent_dialogue = [
            {
                'text': d['text'],
                'timestamp': d['timestamp']
            }
            for d in list(self.dialogue_history)[-10:]
        ]
        
        return {
            'most_common': common_text,
            'by_location': dict(location_counts),
            'recent_dialogue': recent_dialogue,
            'total_unique': len(self.text_frequency),
            'total_detections': sum(self.text_frequency.values())
        }
    
    def _get_system_status(self) -> Dict[str, Any]:
        """Get system performance status"""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'memory_usage_mb': memory_info.rss / 1024 / 1024,
            'memory_available_gb': psutil.virtual_memory().available / 1024 / 1024 / 1024,
            'disk_usage_percent': psutil.disk_usage('/').percent,
            'timestamp': datetime.now().isoformat()
        }
    
    def _search_text(self, query: str) -> List[Dict[str, Any]]:
        """Search text detections"""
        if not query:
            return []
        
        results = []
        query_lower = query.lower()
        
        for text_data in self.recent_text:
            if query_lower in text_data['text'].lower():
                results.append(text_data)
        
        # Limit results
        return results[-50:]
    
    def _calculate_reward_trend(self) -> str:
        """Calculate recent reward trend"""
        if len(self.episode_rewards) < 5:
            return "insufficient_data"
        
        recent_rewards = [r['reward'] for r in list(self.episode_rewards)[-10:]]
        if len(recent_rewards) < 2:
            return "insufficient_data"
        
        # Simple trend calculation
        first_half = recent_rewards[:len(recent_rewards)//2]
        second_half = recent_rewards[len(recent_rewards)//2:]
        
        first_avg = np.mean(first_half)
        second_avg = np.mean(second_half)
        
        if second_avg > first_avg * 1.1:
            return "improving"
        elif second_avg < first_avg * 0.9:
            return "declining"
        else:
            return "stable"
    
    def _get_text_by_location(self) -> Dict[str, int]:
        """Get text count by location"""
        location_counts = defaultdict(int)
        for text_data in self.recent_text:
            location_counts[text_data['location']] += 1
        return dict(location_counts)
    
    def _update_system_stats(self):
        """Update system performance statistics"""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            
            self.system_stats.update({
                'cpu_percent': psutil.cpu_percent(),
                'memory_percent': psutil.virtual_memory().percent,
                'memory_usage_mb': memory_info.rss / 1024 / 1024
            })
            
            # Track memory usage history
            self.memory_usage_history.append({
                'timestamp': datetime.now().isoformat(),
                'memory_mb': self.system_stats['memory_usage_mb']
            })
            
        except Exception as e:
            print(f"⚠️ System stats update error: {e}")
    
    def _archive_current_session(self):
        """Archive current session to history"""
        session_stats = SessionStats(
            session_id=self.current_session['session_id'],
            start_time=self.current_session['start_time'],
            duration=(datetime.now() - self.current_session['start_time']).total_seconds(),
            total_steps=self.training_progress['total_steps'],
            total_episodes=self.training_progress['total_episodes'],
            llm_calls=len(self.decisions),
            visual_analyses=len([m for m in self.performance_metrics if m.metric_name == 'visual_analysis']),
            text_detections=len(self.recent_text),
            unique_text=len(self.text_frequency),
            avg_reward=float(np.mean([r['reward'] for r in self.episode_rewards])) if self.episode_rewards else 0,
            actions_per_second=float(np.mean([m.value for m in self.performance_metrics if m.metric_name == 'steps_per_second'])) if self.performance_metrics else 0,
            memory_usage_mb=self.system_stats['memory_usage_mb'],
            dialogue_count=len(self.dialogue_history),
            screen_types={},  # Could be expanded
            text_locations=self._get_text_by_location()
        )
        
        self.session_history.append(session_stats)
    
    def _cleanup_worker(self):
        """Background cleanup worker"""
        while self.monitoring_active:
            try:
                self._cleanup_old_data()
                time.sleep(self.cleanup_interval)
            except Exception as e:
                print(f"⚠️ Cleanup worker error: {e}")
    
    def _cleanup_old_data(self):
        """Clean up old data to manage memory"""
        cutoff_time = datetime.now() - timedelta(hours=self.data_retention_hours)
        
        # Clean performance metrics
        self.performance_metrics = deque(
            [m for m in self.performance_metrics if m.timestamp > cutoff_time],
            maxlen=1000
        )
        
        # Check memory usage
        if self.system_stats['memory_usage_mb'] > self.max_memory_mb:
            # Reduce data retention if memory is high
            self.screenshots = deque(list(self.screenshots)[-5:], maxlen=10)
            self.actions = deque(list(self.actions)[-50:], maxlen=100)
            print(f"🧹 Memory cleanup performed - reduced data retention")
    
    def run(self, host='127.0.0.1', port=5000, debug=False):
        """Run the web server"""
        print(f"🌐 Enhanced dashboard starting on {host}:{port}")
        self.socketio.run(self.app, host=host, port=port, debug=debug)


# Helper function to create enhanced dashboard templates
def create_enhanced_templates():
    """Create enhanced dashboard templates"""
    templates_dir = Path("templates")
    templates_dir.mkdir(exist_ok=True)
    
    static_dir = Path("static")
    static_dir.mkdir(exist_ok=True)
    
    print("📁 Enhanced template directories created")


if __name__ == "__main__":
    # Test the enhanced monitor
    monitor = EnhancedWebMonitor()
    monitor.start_monitoring()
    
    # Add some test data
    monitor.add_performance_metric("steps_per_second", 25.0)
    monitor.update_text_detection("Hello, world!", "dialogue", "dialogue")
    monitor.add_episode_reward(150.0, 1)
    
    print("🧪 Enhanced web monitor test completed")
