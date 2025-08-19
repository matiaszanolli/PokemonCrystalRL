"""
web_interface.py - Web-based monitoring interface

Provides a web dashboard for monitoring training progress, viewing game state,
and analyzing system performance.
"""

import os
import json
import threading
import time
from typing import Dict, Any, Optional, List
from datetime import datetime
from queue import Queue, Empty
import numpy as np
from PIL import Image
import base64
import io
from flask import Flask, render_template, jsonify, send_from_directory

from core.monitoring.data_bus import DataType, get_data_bus
from core.error_handler import SafeOperation, ErrorHandler, error_boundary
from core.monitoring.stats_collector import StatsCollector
from core.monitoring.game_streamer import GameStreamer


class WebInterface:
    """Web-based monitoring dashboard"""
    
    def __init__(self, host: str = "localhost", port: int = 5000,
                 update_interval: float = 1.0):
        self.host = host
        self.port = port
        self.update_interval = update_interval
        
        # Components
        self.data_bus = get_data_bus()
        self.error_handler = ErrorHandler.get_instance()
        
        # Data storage
        self._latest_stats: Dict[str, Any] = {}
        self._latest_frame: Optional[np.ndarray] = None
        self._update_queue = Queue(maxlen=100)
        
        # Initialize Flask app
        self.app = Flask(__name__)
        self._setup_routes()
        
        # Subscribe to data bus events
        self.data_bus.subscribe(DataType.TRAINING_STATS, self._handle_stats)
        self.data_bus.subscribe(DataType.SCREEN_CAPTURE, self._handle_frame)
        self.data_bus.subscribe(DataType.ERROR_EVENT, self._handle_error)
        
        # Background tasks
        self._stop_event = threading.Event()
        self._update_thread = None
        
    def start(self) -> None:
        """Start the web interface"""
        if self._update_thread and self._update_thread.is_alive():
            return
            
        # Start update thread
        self._stop_event.clear()
        self._update_thread = threading.Thread(
            target=self._update_loop,
            name="WebInterface-Updater",
            daemon=True
        )
        self._update_thread.start()
        
        # Start Flask server
        self.app.run(host=self.host, port=self.port, debug=False, use_reloader=False)
        
    def stop(self) -> None:
        """Stop the web interface"""
        self._stop_event.set()
        if self._update_thread:
            self._update_thread.join(timeout=1.0)
            
    def _setup_routes(self) -> None:
        """Set up Flask routes"""
        
        @self.app.route('/')
        def index():
            """Render main dashboard"""
            return render_template('dashboard.html')
            
        @self.app.route('/api/stats')
        def get_stats():
            """Get latest statistics"""
            return jsonify(self._latest_stats)
            
        @self.app.route('/api/frame')
        def get_frame():
            """Get latest game frame"""
            if self._latest_frame is None:
                return jsonify({'error': 'No frame available'})
                
            # Convert frame to base64 image
            image = Image.fromarray(self._latest_frame)
            buffer = io.BytesIO()
            image.save(buffer, format='JPEG')
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            
            return jsonify({
                'frame': f'data:image/jpeg;base64,{image_base64}',
                'timestamp': time.time()
            })
            
        @self.app.route('/api/errors')
        def get_errors():
            """Get recent errors"""
            return jsonify({
                'errors': self.error_handler.get_error_statistics()
            })
            
    @error_boundary("WebInterface")
    def _handle_stats(self, stats: Dict[str, Any]) -> None:
        """Handle training statistics updates"""
        self._update_queue.put({
            'type': 'stats',
            'data': stats,
            'timestamp': time.time()
        })
        
    @error_boundary("WebInterface")
    def _handle_frame(self, frame_data: Dict[str, Any]) -> None:
        """Handle new game frames"""
        if 'frame' in frame_data:
            self._latest_frame = frame_data['frame']
            
    @error_boundary("WebInterface")
    def _handle_error(self, error_data: Dict[str, Any]) -> None:
        """Handle error events"""
        self._update_queue.put({
            'type': 'error',
            'data': error_data,
            'timestamp': time.time()
        })
        
    def _update_loop(self) -> None:
        """Background update loop"""
        while not self._stop_event.is_set():
            try:
                # Process any queued updates
                while not self._update_queue.empty():
                    update = self._update_queue.get_nowait()
                    
                    if update['type'] == 'stats':
                        self._latest_stats.update(update['data'])
                    elif update['type'] == 'error':
                        # Add error to stats
                        if 'errors' not in self._latest_stats:
                            self._latest_stats['errors'] = []
                        self._latest_stats['errors'].append(update['data'])
                        
                        # Trim error history
                        if len(self._latest_stats['errors']) > 100:
                            self._latest_stats['errors'] = \
                                self._latest_stats['errors'][-100:]
                            
                # Update system stats
                self._latest_stats.update({
                    'system': {
                        'timestamp': time.time(),
                        'memory': self.error_handler.memory_monitor.get_memory_info(),
                        'errors': self.error_handler.get_error_statistics()
                    }
                })
                
            except Empty:
                pass
            except Exception as e:
                print(f"Error in update loop: {e}")
                
            time.sleep(self.update_interval)
            
    def update_stats(self, stats: Dict[str, Any]) -> None:
        """Update dashboard statistics
        
        Args:
            stats: Dictionary of statistics to update
        """
        self._update_queue.put({
            'type': 'stats',
            'data': stats,
            'timestamp': time.time()
        })
