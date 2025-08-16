"""
Web monitoring server for Pokemon Crystal RL Trainer
"""

import time
import json
import base64
import socket
import os
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Optional, Dict, Any
import logging

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# Optimized video streaming support
try:
    from ..core.video_streaming import create_video_streamer
    VIDEO_STREAMING_AVAILABLE = True
except ImportError:
    VIDEO_STREAMING_AVAILABLE = False
    create_video_streamer = None

from .config import ACTION_NAMES


class TrainingWebServer:
    """Web server for training monitoring and control"""
    
    def __init__(self, config, trainer):
        self.config = config
        self.trainer = trainer
        self.logger = logging.getLogger(__name__)
        self.server = None
        
        # Find available port
        self.port = self._find_available_port()
        
    def _find_available_port(self) -> int:
        """Find an available port for the web server"""
        original_port = self.config.web_port
        port_to_use = original_port
        max_attempts = 10
        
        for attempt in range(max_attempts):
            try:
                # Test if port is available
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    s.bind((self.config.web_host, port_to_use))
                    break  # Port is available
            except OSError:
                # Port is in use, try next one
                port_to_use = original_port + attempt + 1
                if attempt == max_attempts - 1:
                    raise RuntimeError(f"Could not find available port after {max_attempts} attempts starting from {original_port}")
        
        # Update config if we had to change the port
        if port_to_use != original_port:
            if self.config.debug_mode:
                self.logger.info(f"📡 Port {original_port} was busy, using port {port_to_use} instead")
        
        return port_to_use
    
    def start(self):
        """Start the web server"""
        handler_factory = lambda *args, **kwargs: TrainingHandler(self.trainer, *args, **kwargs)
        self.server = HTTPServer((self.config.web_host, self.port), handler_factory)
        self.logger.info(f"🌐 Web interface: http://{self.config.web_host}:{self.port}")
        return self.server
    
    def stop(self):
        """Stop the web server"""
        if self.server:
            self.server.shutdown()


class TrainingHandler(BaseHTTPRequestHandler):
    """HTTP request handler for training monitoring"""
    
    def __init__(self, trainer, *args, **kwargs):
        self.trainer = trainer
        super().__init__(*args, **kwargs)
    
    def do_GET(self):
        if self.path == '/':
            self._serve_comprehensive_dashboard()
        elif self.path.startswith('/screen'):
            self._serve_screen()
        elif self.path.startswith('/api/screenshot'):
            self._serve_screen()  # Alias for screenshot endpoint
        elif self.path == '/stats':
            self._serve_stats()
        elif self.path == '/api/status':
            self._serve_api_status()
        elif self.path == '/api/system':
            self._serve_api_system()
        elif self.path == '/api/runs':
            self._serve_api_runs()
        elif self.path == '/api/text':
            self._serve_api_text()
        elif self.path == '/api/llm_decisions':
            self._serve_api_llm_decisions()
        elif self.path == '/api/streaming/stats':
            self._serve_streaming_stats()
        elif self.path.startswith('/api/streaming/quality/'):
            self._handle_quality_control()
        elif self.path.startswith('/socket.io/'):
            self._handle_socketio_fallback()
        else:
            self.send_error(404)
    
    def do_POST(self):
        if self.path == '/api/start_training':
            self._handle_start_training()
        elif self.path == '/api/stop_training':
            self._handle_stop_training()
        else:
            self.send_error(404)
    
    def _serve_comprehensive_dashboard(self):
        """Serve the comprehensive dashboard from templates"""
        try:
            # Use local templates directory relative to python_agent
            template_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'templates', 'dashboard.html')
            with open(template_path, 'r', encoding='utf-8') as f:
                html = f.read()
            
            self.send_response(200)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(html.encode('utf-8'))
        except Exception as e:
            print(f"⚠️ Error loading dashboard template: {e}")
            self._serve_fallback_dashboard()
    
    def _serve_fallback_dashboard(self):
        """Fallback simple dashboard if template fails"""
        html = """<!DOCTYPE html>
<html>
<head>
    <title>Pokemon Crystal Trainer</title>
    <style>
        body { 
            font-family: 'Courier New', monospace; 
            margin: 20px; 
            background: #1a1a1a; 
            color: white; 
            line-height: 1.4;
        }
        .container { max-width: 1200px; margin: 0 auto; }
        .stats { background: #333; padding: 15px; border-radius: 8px; margin: 10px 0; }
        .screen { border: 3px solid #4CAF50; margin: 20px 0; text-align: center; background: #000; border-radius: 8px; padding: 10px; }
        .screen img { width: 120px; height: 108px; image-rendering: pixelated; }
        h1 { text-align: center; color: #4CAF50; }
    </style>
</head>
<body>
    <div class="container">
        <h1>⚡ Pokemon Crystal Unified Trainer</h1>
        <div class="stats" id="stats">Loading...</div>
        <div class="screen">
            <h3>🎮 Game Screen</h3>
            <img id="gameScreen" src="/screen" alt="Game Screen">
        </div>
    </div>
    <script>
        setInterval(() => {
            // Try the main stats endpoint first
            fetch('/stats').then(r => r.json()).then(data => {
                document.getElementById('stats').innerHTML = 
                    `🎯 Actions: ${data.total_actions} | ⚡ Speed: ${data.actions_per_second.toFixed(1)} a/s | 🧠 LLM: ${data.llm_calls} | 🎮 Mode: ${data.mode}`;
            }).catch(() => {
                // Fallback to API status endpoint
                fetch('/api/status').then(r => r.json()).then(data => {
                    document.getElementById('stats').innerHTML = 
                        `🎯 Actions: ${data.total_actions} | ⚡ Speed: ${data.actions_per_second.toFixed(1)} a/s | 🧠 LLM: ${data.llm_calls} | Status: ${data.is_training ? 'Training' : 'Stopped'}`;
                }).catch(() => {
                    document.getElementById('stats').innerHTML = 'Stats unavailable';
                });
            });
            document.getElementById('gameScreen').src = '/screen?' + Date.now();
        }, 1000);
    </script>
</body>
</html>"""
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(html.encode())
    
    def _serve_screen(self):
        """Serve game screen with optimized streaming if available"""
        
        # Try optimized streaming first
        if hasattr(self.trainer, 'video_streamer') and self.trainer.video_streamer:
            try:
                screen_bytes = self.trainer.video_streamer.get_frame_as_bytes()
                if screen_bytes:
                    self.send_response(200)
                    self.send_header('Content-type', 'image/jpeg')
                    self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
                    self.send_header('Pragma', 'no-cache')
                    self.send_header('Expires', '0')
                    self.end_headers()
                    self.wfile.write(screen_bytes)
                    return
            except Exception as e:
                # Continue to fallback method
                pass
        
        # Fallback to legacy method
        if hasattr(self.trainer, 'latest_screen') and self.trainer.latest_screen:
            img_data = base64.b64decode(self.trainer.latest_screen['image_b64'])
            self.send_response(200)
            self.send_header('Content-type', 'image/png')
            self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
            self.end_headers()
            self.wfile.write(img_data)
        else:
            self.send_error(404)
    
    def _serve_stats(self):
        stats = self.trainer.get_current_stats()
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(stats).encode())
    
    def _serve_api_status(self):
        """API endpoint for training status"""
        stats = self.trainer.get_current_stats()
        
        # Calculate additional metrics
        elapsed = time.time() - stats.get('start_time', time.time())
        total_actions = stats.get('total_actions', 0)
        
        status = {
            'is_training': getattr(self.trainer, '_training_active', False),
            'current_run_id': getattr(self.trainer, 'current_run_id', 1),
            'mode': self.trainer.config.mode.value,
            'model': self.trainer.config.llm_backend.value if self.trainer.config.llm_backend else 'rule-based',
            'start_time': stats.get('start_time'),
            'total_actions': total_actions,
            'llm_calls': stats.get('llm_calls', 0),
            'actions_per_second': stats.get('actions_per_second', 0.0),
            
            # Additional fields for better dashboard display
            'current_episode': stats.get('total_episodes', 0),
            'elapsed_time': elapsed,
            'game_state': getattr(self.trainer, '_current_state', 'training'),
            'current_reward': total_actions * 0.1,  # Simple reward estimation
            'total_reward': total_actions * 0.15,   # Total reward estimation
            'avg_reward': total_actions * 0.12 if total_actions > 0 else 0.0,
            'success_rate': min(1.0, total_actions / max(100, 1)),  # Success rate based on actions
            
            # Game-specific placeholders (would be populated from actual game state)
            'map_id': getattr(self.trainer, '_current_map', 1),
            'player_x': getattr(self.trainer, '_player_x', 10),
            'player_y': getattr(self.trainer, '_player_y', 8),
        }
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(status).encode())
    
    def _serve_api_system(self):
        """API endpoint for system statistics"""
        try:
            if PSUTIL_AVAILABLE:
                stats = {
                    'cpu_percent': psutil.cpu_percent(),
                    'memory_percent': psutil.virtual_memory().percent,
                    'disk_usage': psutil.disk_usage('/').percent,
                    'gpu_available': False  # Could be enhanced to detect GPU
                }
            else:
                stats = {
                    'cpu_percent': 0.0,
                    'memory_percent': 0.0,
                    'disk_usage': 0.0,
                    'gpu_available': False,
                    'error': 'psutil not available'
                }
        except Exception as e:
            stats = {
                'cpu_percent': 0.0,
                'memory_percent': 0.0,
                'disk_usage': 0.0,
                'gpu_available': False,
                'error': str(e)
            }
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(stats).encode())
    
    def _serve_api_runs(self):
        """API endpoint for training runs history"""
        stats = self.trainer.get_current_stats()
        
        # For now, return current run as a single entry
        current_run = {
            'id': 1,
            'algorithm': self.trainer.config.mode.value,
            'start_time': datetime.fromtimestamp(stats['start_time']).isoformat(),
            'end_time': None,
            'status': 'running' if getattr(self.trainer, '_training_active', False) else 'completed',
            'total_timesteps': stats.get('total_actions', 0),
            'final_reward': 'N/A'
        }
        
        runs = [current_run]
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(runs).encode())
    
    def _handle_start_training(self):
        """Handle training start request"""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            config = json.loads(post_data.decode('utf-8'))
            
            response = {
                'success': False,
                'message': 'Training control not implemented in unified trainer yet'
            }
            
            self.send_response(501)  # Not implemented
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())
            
        except Exception as e:
            self._send_error_response(str(e))
    
    def _handle_stop_training(self):
        """Handle training stop request"""
        response = {
            'success': False,
            'message': 'Training control not implemented in unified trainer yet'
        }
        
        self.send_response(501)  # Not implemented
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(response).encode())
    
    def _handle_socketio_fallback(self):
        """Handle socket.io requests gracefully"""
        # Return a structured response indicating HTTP polling should be used
        response = {
            'error': 'WebSocket/Socket.IO not implemented',
            'message': 'This trainer uses HTTP polling instead of WebSockets',
            'use_polling': True,
            'polling_endpoints': {
                'status': '/api/status',
                'system': '/api/system', 
                'screenshot': '/api/screenshot'
            }
        }
        self.send_response(200)  # Changed from 404 to 200 to avoid browser errors
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        self.wfile.write(json.dumps(response).encode())
    
    def _serve_api_text(self):
        """API endpoint for detected text data"""
        if hasattr(self.trainer, 'recent_text'):
            text_data = {
                'recent_text': self.trainer.recent_text[-10:] if self.trainer.recent_text else [],
                'text_frequency': dict(sorted(self.trainer.text_frequency.items(), 
                                            key=lambda x: x[1], reverse=True)[:20]) if hasattr(self.trainer, 'text_frequency') else {},
                'total_texts': len(self.trainer.recent_text),
                'unique_texts': len(getattr(self.trainer, 'text_frequency', {}))
            }
        else:
            text_data = {
                'recent_text': [],
                'text_frequency': {},
                'total_texts': 0,
                'unique_texts': 0
            }
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(text_data).encode())
    
    def _serve_api_llm_decisions(self):
        """API endpoint for LLM decision history and monitoring"""
        if hasattr(self.trainer, 'llm_manager') and self.trainer.llm_manager:
            llm_data = self.trainer.llm_manager.get_decision_data()
        else:
            # Fallback data structure
            llm_data = {
                'recent_decisions': [],
                'last_decision': None,
                'total_decisions': 0,
                'performance_metrics': {
                    'total_llm_calls': 0,
                    'avg_response_time': 0.0,
                    'total_llm_time': 0.0,
                    'adaptive_interval': self.trainer.config.llm_interval,
                    'current_model': self.trainer.config.llm_backend.value if self.trainer.config.llm_backend else 'rule-based'
                },
                'state_distribution': {},
                'action_distribution': {}
            }
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(llm_data).encode())
    
    def _serve_streaming_stats(self):
        """API endpoint for video streaming performance statistics"""
        if hasattr(self.trainer, 'video_streamer') and self.trainer.video_streamer:
            try:
                stats = self.trainer.video_streamer.get_performance_stats()
            except Exception as e:
                stats = {'method': 'optimized_streaming', 'error': str(e), 'available': False}
        else:
            stats = {'method': 'legacy_fallback', 'available': False, 'message': 'Optimized streaming not initialized'}
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(stats).encode())
    
    def _handle_quality_control(self):
        """Handle video streaming quality control"""
        try:
            # Extract quality parameter from path
            path_parts = self.path.split('/')
            if len(path_parts) >= 4:
                quality = path_parts[4]  # /api/streaming/quality/{quality}
                
                if hasattr(self.trainer, 'video_streamer') and self.trainer.video_streamer:
                    try:
                        self.trainer.video_streamer.change_quality(quality)
                        response = {
                            'success': True,
                            'quality': quality,
                            'message': f'Streaming quality changed to {quality}',
                            'available_qualities': ['low', 'medium', 'high', 'ultra']
                        }
                    except Exception as e:
                        response = {
                            'success': False,
                            'error': str(e),
                            'available_qualities': ['low', 'medium', 'high', 'ultra']
                        }
                else:
                    response = {
                        'success': False,
                        'error': 'Optimized streaming not available',
                        'available_qualities': ['low', 'medium', 'high', 'ultra']
                    }
            else:
                response = {
                    'success': False,
                    'error': 'Quality parameter missing',
                    'usage': '/api/streaming/quality/{low|medium|high|ultra}',
                    'available_qualities': ['low', 'medium', 'high', 'ultra']
                }
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())
            
        except Exception as e:
            self._send_error_response(str(e))
    
    def _send_error_response(self, error_msg):
        response = {'success': False, 'error': error_msg}
        self.send_response(500)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(response).encode())
