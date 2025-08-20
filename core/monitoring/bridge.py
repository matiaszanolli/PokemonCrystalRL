"""
Bridge module to connect UnifiedPokemonTrainer with web monitoring.

Provides robust connection between the trainer and web monitoring interface
with enhanced error handling and stability features.
"""

import time
import threading
import numpy as np
import base64
import io
import logging
from typing import Optional, Dict, Any
from PIL import Image
from datetime import datetime, timedelta
from pathlib import Path

from flask import Flask, send_from_directory, jsonify
from flask_socketio import SocketIO, emit

class TrainerWebBridge:
    """
    Bridge between the UnifiedPokemonTrainer and web monitoring interface.
    Handles data transfer, error recovery, and automatic reconnection.
    """
    
    def __init__(self, trainer, host='127.0.0.1', port=5000, debug=False):
        """Initialize the bridge with enhanced stability features."""
        self.trainer = trainer
        self.host = host
        self.port = port
        self.debug = debug
        self.logger = logging.getLogger("pokemon_trainer.bridge")
        
        # Bridge state
        self.active = False
        self.bridge_thread = None
        self.cleanup_thread = None
        self.error_count = 0
        
        # Performance tracking
        self.last_screen_update = 0
        self.last_stats_update = 0
        self.screen_update_interval = 0.2  # 5 fps
        self.stats_update_interval = 1.0   # 1 hz
        
        # Statistics
        self.transfer_stats = {
            'screenshots_sent': 0,
            'stats_updates': 0,
            'errors': 0,
            'last_error': None,
            'start_time': None
        }
        
        # Flask and SocketIO setup
        self.app = Flask(__name__, 
                        template_folder=str(Path(__file__).parent / 'web/templates'),
                        static_folder=str(Path(__file__).parent / 'web/static'))
        self.socketio = SocketIO(self.app, cors_allowed_origins="*",
                               logger=False, engineio_logger=False)
        
        # Register routes and event handlers
        self._register_routes()
        self._register_socketio_events()
        
        self.logger.info("🌉 TrainerWebBridge initialized")
    
    def start(self):
        """Start the bridge with enhanced error monitoring."""
        if self.active:
            self.logger.warning("Bridge already active")
            return
        
        self.active = True
        self.transfer_stats['start_time'] = time.time()
        
        # Start bridge in background thread
        self.bridge_thread = threading.Thread(target=self._run_server, daemon=True)
        self.bridge_thread.start()
        
        # Start cleanup thread
        self.cleanup_thread = threading.Thread(target=self._cleanup_worker, daemon=True)
        self.cleanup_thread.start()
        
        self.logger.info(f"🌉 Bridge started - http://{self.host}:{self.port}")
    
    def stop(self):
        """Stop the bridge cleanly."""
        if not self.active:
            return
        
        self.logger.info("Stopping bridge...")
        self.active = False
        
        # Wait for threads to finish naturally
        if self.cleanup_thread and self.cleanup_thread.is_alive():
            self.cleanup_thread.join(timeout=2)
        
        # Stop server
        try:
            if hasattr(self, 'socketio'):
                # Use test client to trigger server shutdown
                with self.app.test_client() as client:
                    client.get('/shutdown')
                # Stop SocketIO
                self.socketio.stop()
        except Exception as e:
            self.logger.warning(f"Error during server shutdown: {e}")
        
        # Wait for bridge thread
        if self.bridge_thread and self.bridge_thread.is_alive():
            self.bridge_thread.join(timeout=2)
        
        self._log_final_stats()
    
    def _register_routes(self):
        """Register HTTP routes."""

        @self.app.route('/')
        def dashboard():
            """Main dashboard route."""
            return send_from_directory(self.app.template_folder, 'dashboard.html')

        @self.app.route('/shutdown')
        def shutdown():
            """Shutdown route for testing."""
            if self.active:
                func = self.app.request.environ.get('werkzeug.server.shutdown')
                if func is None:
                    self.logger.warning("Shutdown function not available")
                    return jsonify({'status': 'error'}), 500

                func()
                return jsonify({'status': 'success'})
            return jsonify({'status': 'not_active'})

        @self.app.route('/api/screenshot/current')
        def get_current_screenshot():
            """Get latest screenshot (HTTP fallback)."""
            screenshot = self._get_current_screenshot()
            if screenshot is not None:
                return jsonify({'screenshot': screenshot})
            return jsonify({'screenshot': None})

        @self.app.route('/api/session/stats')
        def get_session_stats():
            """Get current session stats (HTTP fallback)."""
            stats = self._get_session_stats()
            return jsonify(stats)

        @self.app.route('/api/bridge/status')
        def get_bridge_status():
            """Get bridge status information."""
            return jsonify(self._get_bridge_status())

    def _register_socketio_events(self):
        """Register SocketIO event handlers."""
        
        @self.socketio.on('connect')
        def handle_connect():
            """Handle client connection."""
            self.logger.debug("Client connected")
            emit('status', {'connected': True})
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            """Handle client disconnection."""
            self.logger.debug("Client disconnected")
        
        @self.socketio.on('request_screenshot')
        def handle_screenshot_request():
            """Handle screenshot request."""
            screenshot = self._get_current_screenshot()
            if screenshot is not None:
                emit('screenshot_update', {'screenshot': screenshot})
        
        @self.socketio.on('request_stats')
        def handle_stats_request():
            """Handle stats request."""
            stats = self._get_session_stats()
            emit('stats_update', stats)
    
    def _get_current_screenshot(self) -> Optional[str]:
        """Get current screenshot with validation."""
        try:
            if not hasattr(self.trainer, 'latest_screen') or self.trainer.latest_screen is None:
                return None
            
            screen = np.array(self.trainer.latest_screen)
            
            # Validate screen content
            valid = self._validate_screenshot(screen)

            try:
                if not valid:
                    raise ValueError("Invalid screenshot")
                
                # Convert to base64
                img = Image.fromarray(screen)
                buffer = io.BytesIO()
                img.save(buffer, format='PNG')
                img_str = base64.b64encode(buffer.getvalue()).decode()
                
                self.transfer_stats['screenshots_sent'] += 1
                self.last_screen_update = time.time()
                
                return img_str
            except Exception as e:
                self._handle_error(f"Screenshot conversion error: {e}")
                return None

        except Exception as e:
            self._handle_error(f"Screenshot error: {e}")
            return None

        # Fail-safe check to handle error case
        self._handle_error("Failed to process screenshot")
        return None
    
    def _validate_screenshot(self, screen: np.ndarray) -> bool:
        """Validate screenshot content."""
        try:
            if screen is None or screen.size == 0:
                return False
            
            # Shape validation
            if len(screen.shape) != 3 or screen.shape[2] not in (3, 4):
                return False
            
            # Content validation
            variance = np.var(screen)
            if variance < 1.0:  # Detect blank screens
                self.logger.warning(f"Low variance screenshot: {variance:.2f}")
                return False
            
            return True
            
        except Exception as e:
            self._handle_error(f"Screenshot validation error: {e}")
            return False
    
    def _get_session_stats(self) -> Dict[str, Any]:
        """Get current session statistics."""
        try:
            if not hasattr(self.trainer, 'stats'):
                return {}
            
            stats = self.trainer.stats
            current_time = time.time()
            
            # Basic stats
            basic_stats = {
                'total_steps': stats.get('total_actions', 0),
                'total_episodes': stats.get('total_episodes', 0),
                'session_duration': current_time - stats.get('start_time', current_time),
                'is_training': getattr(self.trainer, '_training_active', False)
            }
            
            # Performance stats
            performance_stats = {
                'avg_actions_per_second': stats.get('actions_per_second', 0),
                'llm_calls': stats.get('llm_calls', 0),
                'llm_avg_time': stats.get('llm_avg_time', 0)
            }
            
            # System stats
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            
            system_stats = {
                'cpu_percent': psutil.cpu_percent(),
                'memory_usage_mb': memory_info.rss / 1024 / 1024,
                'memory_percent': psutil.virtual_memory().percent
            }
            
            self.transfer_stats['stats_updates'] += 1
            self.last_stats_update = current_time
            
            return {
                'basic': basic_stats,
                'performance': performance_stats,
                'system': system_stats,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self._handle_error(f"Stats error: {e}")
            return {}
    
    def _get_bridge_status(self) -> Dict[str, Any]:
        """Get bridge status information."""
        current_time = time.time()
        start_time = self.transfer_stats['start_time'] or current_time
        
        # Calculate actual FPS based on recent updates
        actual_fps = 0
        if self.last_screen_update > 0:
            time_since_last = current_time - self.last_screen_update
            if time_since_last <= self.screen_update_interval:
                actual_fps = 1.0 / (time_since_last or 0.001)
        
        return {
            'active': self.active and self.bridge_thread and self.bridge_thread.is_alive(),
            'uptime': current_time - start_time,
            'screenshots_sent': self.transfer_stats['screenshots_sent'],
            'stats_updates': self.transfer_stats['stats_updates'],
            'error_count': self.error_count,
            'last_screenshot_update': current_time - self.last_screen_update,
            'last_stats_update': current_time - self.last_stats_update,
            'last_error': self.transfer_stats['last_error'],
            'screen_fps': actual_fps,
            'stats_hz': 1.0 / (self.stats_update_interval or 0.001),
            'server_port': self.port,
            'server_host': self.host
        }
    
    def _handle_error(self, error: str):
        """Handle errors with rate limiting."""
        self.error_count += 1
        self.transfer_stats['errors'] += 1
        self.transfer_stats['last_error'] = {
            'message': str(error),
            'time': datetime.now().isoformat()
        }
        
        # Rate limit error logging
        if self.error_count <= 5 or self.error_count % 10 == 0:
            self.logger.error(f"Bridge error #{self.error_count}: {error}")
    
    def _cleanup_worker(self):
        """Background cleanup worker."""
        while self.active:
            try:
                self._perform_cleanup()
                time.sleep(60)  # Check every minute
            except Exception as e:
                self._handle_error(f"Cleanup error: {e}")
    
    def _perform_cleanup(self):
        """Perform periodic cleanup tasks."""
        # Reset error count periodically
        if self.error_count > 0:
            if time.time() - self.last_screen_update > 300:  # 5 minutes
                self.error_count = 0
                self.logger.info("Reset error count")
    
    def _log_final_stats(self):
        """Log final statistics."""
        duration = time.time() - (self.transfer_stats['start_time'] or time.time())
        
        self.logger.info("\n=== Bridge Final Statistics ===")
        self.logger.info(f"Duration: {timedelta(seconds=int(duration))}")
        self.logger.info(f"Screenshots sent: {self.transfer_stats['screenshots_sent']}")
        self.logger.info(f"Stats updates: {self.transfer_stats['stats_updates']}")
        self.logger.info(f"Total errors: {self.transfer_stats['errors']}")
        
        if duration > 0:
            fps = self.transfer_stats['screenshots_sent'] / duration
            self.logger.info(f"Average FPS: {fps:.1f}")
    
    def _run_server(self):
        """Run the web server."""
        try:
            self.socketio.run(
                self.app,
                host=self.host,
                port=self.port,
                debug=self.debug,
                use_reloader=False,
                allow_unsafe_werkzeug=True  # Required for test environment
            )
        except Exception as e:
            self._handle_error(f"Server error: {e}")
            self.active = False
