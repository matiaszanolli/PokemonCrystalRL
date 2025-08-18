#!/usr/bin/env python3
"""
Web server implementation for Pokemon Crystal RL training visualization and control.
Provides endpoints for monitoring training progress, viewing game state, and controlling training.
"""

import json
import socket
import os
from http.server import HTTPServer, BaseHTTPRequestHandler
import base64
import logging
from typing import Dict, Any, Optional
import time

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


class TrainingWebServer:
    """Web server for Pokemon Crystal RL training visualization."""
    
    def __init__(self, config: Any, trainer: Any):
        """Initialize the training web server.
        
        Args:
            config: Configuration object with web_host, web_port and other settings
            trainer: Pokemon Crystal RL trainer instance
        """
        self.config = config
        self.trainer = trainer
        self.port = self._find_available_port()
        self.server = None
        self.logger = logging.getLogger(__name__)
        
        if self.config.debug_mode:
            self.logger.info(f"🌐 Web server initialized on port {self.port}")
    
    def _find_available_port(self) -> int:
        """Find an available port starting from the configured port.
        
        Returns:
            Available port number
        
        Raises:
            RuntimeError: If no port is available after max attempts
        """
        max_attempts = 10
        current_port = self.config.web_port
        
        for attempt in range(max_attempts):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    sock.bind((self.config.web_host, current_port))
                    return current_port
            except OSError:
                if self.config.debug_mode:
                    self.logger.warning(f"Port {current_port} is in use, trying {current_port + 1}")
                current_port += 1
        
        raise RuntimeError(f"Could not find available port after {max_attempts} attempts")
    
    def start(self) -> HTTPServer:
        """Start the web server.
        
        Returns:
            Running HTTPServer instance
        """
        def handler_factory(*args):
            return TrainingHandler(self.trainer, *args)
        
        self.server = HTTPServer((self.config.web_host, self.port), handler_factory)
        self.logger.info(f"🚀 Web server started on {self.config.web_host}:{self.port}")
        return self.server
    
    def stop(self) -> None:
        """Stop the web server."""
        if self.server:
            self.server.shutdown()
            self.logger.info("🛑 Web server stopped")


class TrainingHandler(BaseHTTPRequestHandler):
    """HTTP request handler for Pokemon Crystal RL training web interface."""
    
    def __init__(self, trainer: Any, *args):
        """Initialize the request handler.
        
        Args:
            trainer: Pokemon Crystal RL trainer instance
            *args: Additional arguments passed to parent class
        """
        self.trainer = trainer
        super().__init__(*args)
    
    def log_message(self, format: str, *args) -> None:
        """Override logging to use our logger."""
        if self.trainer.config.debug_mode:
            logging.getLogger(__name__).debug(format % args)
    
    def do_GET(self) -> None:
        """Handle GET requests."""
        try:
            if self.path == "/":
                self._serve_comprehensive_dashboard()
            elif self.path in ["/screen", "/api/screenshot"]:
                self._serve_screen()
            elif self.path == "/stats":
                self._serve_stats()
            elif self.path == "/api/status":
                self._serve_api_status()
            elif self.path == "/api/system":
                self._serve_api_system()
            elif self.path == "/api/runs":
                self._serve_api_runs()
            elif self.path == "/api/text":
                self._serve_api_text()
            elif self.path == "/api/llm_decisions":
                self._serve_api_llm_decisions()
            elif self.path == "/api/streaming/stats":
                self._serve_streaming_stats()
            elif "/api/streaming/quality/" in self.path:
                self._handle_quality_control()
            elif self.path.startswith("/socket.io/"):
                self._handle_socketio_fallback()
            else:
                self.send_error(404)
        except Exception as e:
            self._send_error_response(str(e))
    
    def do_POST(self) -> None:
        """Handle POST requests."""
        try:
            if self.path == "/api/start_training":
                self._handle_start_training()
            elif self.path == "/api/stop_training":
                self._handle_stop_training()
            else:
                self.send_error(404)
        except Exception as e:
            self._send_error_response(str(e))
    
    def _serve_comprehensive_dashboard(self) -> None:
        """Serve the main dashboard HTML page."""
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            template_path = os.path.join(os.path.dirname(current_dir), "templates", "dashboard.html")
            
            with open(template_path, 'rb') as f:
                content = f.read()
            
            self.send_response(200)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(content)
            
        except Exception:
            self._serve_fallback_dashboard()
    
    def _serve_fallback_dashboard(self) -> None:
        """Serve a simple fallback dashboard when template is unavailable."""
        content = f"""
        <html>
        <head><title>Pokemon Crystal Trainer</title></head>
        <body>
            <h1>Pokemon Crystal RL Training Monitor</h1>
            <div>
                <h2>Game Screen</h2>
                <img src="/screen" id="game-screen" width="320" height="288">
            </div>
            <div>
                <h2>Training Stats</h2>
                <pre id="stats"></pre>
            </div>
            <script>
                function updateScreen() {{
                    document.getElementById('game-screen').src = '/screen?' + new Date().getTime();
                }}
                function updateStats() {{
                    fetch('/stats')
                        .then(response => response.json())
                        .then(data => {{
                            document.getElementById('stats').textContent = JSON.stringify(data, null, 2);
                        }});
                }}
                setInterval(updateScreen, 1000);
                setInterval(updateStats, 2000);
            </script>
        </body>
        </html>
        """.encode()
        
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(content)
    
    def _serve_screen(self) -> None:
        """Serve the current game screen."""
        try:
            # Try optimized streaming first
            if hasattr(self.trainer, 'video_streamer'):
                screen_data = self.trainer.video_streamer.get_frame_as_bytes()
                if screen_data:
                    self.send_response(200)
                    self.send_header('Content-type', 'image/jpeg')
                    self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
                    self.send_header('Pragma', 'no-cache')
                    self.send_header('Expires', '0')
                    self.end_headers()
                    self.wfile.write(screen_data)
                    return
            
            # Fall back to legacy method
            if hasattr(self.trainer, 'latest_screen') and self.trainer.latest_screen:
                image_data = base64.b64decode(self.trainer.latest_screen['image_b64'])
                self.send_response(200)
                self.send_header('Content-type', 'image/png')
                self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
                self.send_header('Pragma', 'no-cache')
                self.send_header('Expires', '0')
                self.end_headers()
                self.wfile.write(image_data)
            else:
                self.send_error(404)
                
        except Exception as e:
            self._send_error_response(f"Error serving screen: {str(e)}")
    
    def _serve_stats(self) -> None:
        """Serve current training statistics."""
        try:
            stats = self.trainer.get_current_stats()
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(stats).encode())
        except Exception as e:
            self._send_error_response(f"Error serving stats: {str(e)}")
    
    def _serve_api_status(self) -> None:
        """Serve detailed API status information."""
        try:
            stats = self.trainer.get_current_stats()
            current_time = time.time()
            
            status = {
                'is_training': getattr(self.trainer, '_training_active', False),
                'current_run_id': getattr(self.trainer, 'current_run_id', None),
                'total_actions': stats.get('total_actions', 0),
                'elapsed_time': int(current_time - stats.get('start_time', current_time)),
                'mode': self.trainer.config.mode.value,
                'llm_backend': self.trainer.config.llm_backend.value,
                'llm_interval': self.trainer.config.llm_interval,
                'state': {
                    'current': getattr(self.trainer, '_current_state', 'unknown'),
                    'map_id': getattr(self.trainer, '_current_map', None),
                    'position': {
                        'x': getattr(self.trainer, '_player_x', None),
                        'y': getattr(self.trainer, '_player_y', None)
                    }
                }
            }
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(status).encode())
            
        except Exception as e:
            self._send_error_response(f"Error serving API status: {str(e)}")
    
    def _serve_api_system(self) -> None:
        """Serve system resource usage information."""
        try:
            data = {
                'cpu_percent': 0.0,
                'memory_percent': 0.0,
                'disk_usage': 0.0,
                'gpu_available': False
            }
            
            if PSUTIL_AVAILABLE:
                try:
                    data['cpu_percent'] = psutil.cpu_percent()
                    data['memory_percent'] = psutil.virtual_memory().percent
                    data['disk_usage'] = psutil.disk_usage('/').percent
                except Exception as e:
                    data['error'] = str(e)
            else:
                data['error'] = 'psutil not available'
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(data).encode())
            
        except Exception as e:
            self._send_error_response(f"Error serving system info: {str(e)}")
    
    def _serve_api_runs(self) -> None:
        """Serve information about training runs."""
        try:
            stats = self.trainer.get_current_stats()
            
            run_data = [{
                'id': getattr(self.trainer, 'current_run_id', 1),
                'start_time': stats.get('start_time', time.time()),
                'total_timesteps': stats.get('total_actions', 0),
                'status': 'active' if getattr(self.trainer, '_training_active', False) else 'completed'
            }]
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(run_data).encode())
            
        except Exception as e:
            self._send_error_response(f"Error serving runs: {str(e)}")
    
    def _serve_api_text(self) -> None:
        """Serve text recognition data."""
        try:
            recent_text = getattr(self.trainer, 'recent_text', [])
            text_frequency = getattr(self.trainer, 'text_frequency', {})
            
            data = {
                'recent_text': recent_text,
                'total_texts': len(recent_text),
                'unique_texts': len(text_frequency),
                'text_frequency': dict(sorted(
                    text_frequency.items(),
                    key=lambda x: x[1],
                    reverse=True
                ))
            }
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(data).encode())
            
        except Exception as e:
            self._send_error_response(f"Error serving text data: {str(e)}")
    
    def _serve_api_llm_decisions(self) -> None:
        """Serve LLM decision history."""
        try:
            if hasattr(self.trainer, 'llm_manager'):
                data = self.trainer.llm_manager.get_decision_data()
            else:
                data = {
                    'recent_decisions': [],
                    'total_decisions': 0,
                    'performance_metrics': {
                        'total_llm_calls': 0,
                        'current_model': self.trainer.config.llm_backend.value
                    }
                }
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(data).encode())
            
        except Exception as e:
            self._send_error_response(f"Error serving LLM decisions: {str(e)}")
    
    def _serve_streaming_stats(self) -> None:
        """Serve video streaming performance statistics."""
        try:
            data = {
                'available': False,
                'method': 'optimized_streaming'
            }
            
            if hasattr(self.trainer, 'video_streamer'):
                try:
                    data.update(self.trainer.video_streamer.get_performance_stats())
                    data['available'] = True
                except Exception as e:
                    data['error'] = str(e)
            else:
                data['method'] = 'legacy_fallback'
                data['message'] = 'Optimized streaming not initialized'
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(data).encode())
            
        except Exception as e:
            self._send_error_response(f"Error serving streaming stats: {str(e)}")
    
    def _handle_quality_control(self) -> None:
        """Handle streaming quality control requests."""
        try:
            path_parts = self.path.split('/')
            if len(path_parts) >= 4:
                quality = path_parts[4]
                
                data = {
                    'success': False,
                    'quality': quality,
                    'available_qualities': ['low', 'medium', 'high', 'ultra']
                }
                
                if hasattr(self.trainer, 'video_streamer'):
                    try:
                        self.trainer.video_streamer.change_quality(quality)
                        data['success'] = True
                        data['message'] = f'Quality changed to {quality}'
                    except Exception as e:
                        data['error'] = str(e)
                else:
                    data['error'] = 'Optimized streaming not available'
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps(data).encode())
                
        except Exception as e:
            self._send_error_response(str(e))
    
    def _handle_socketio_fallback(self) -> None:
        """Handle Socket.IO fallback requests."""
        data = {
            'error': 'WebSocket/Socket.IO not implemented',
            'use_polling': True,
            'polling_endpoints': {
                'status': '/api/status',
                'stats': '/stats',
                'screen': '/screen'
            }
        }
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())
    
    def _handle_start_training(self) -> None:
        """Handle start training requests (not implemented)."""
        data = {
            'success': False,
            'message': 'Start training endpoint not implemented'
        }
        
        self.send_response(501)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())
    
    def _handle_stop_training(self) -> None:
        """Handle stop training requests (not implemented)."""
        data = {
            'success': False,
            'message': 'Stop training endpoint not implemented'
        }
        
        self.send_response(501)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())
    
    def _send_error_response(self, error_message: str) -> None:
        """Send a JSON error response.
        
        Args:
            error_message: Error message to include in response
        """
        data = {
            'success': False,
            'error': error_message
        }
        
        self.send_response(500)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())
