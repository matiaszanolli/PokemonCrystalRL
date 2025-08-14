#!/usr/bin/env python3
"""
monitor.py - Web-based training monitoring dashboard with real-time updates
"""

import os
import json
import time
import threading
from datetime import datetime, timedelta
from pathlib import Path
import sqlite3
import subprocess
import signal
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from flask import Flask, render_template, jsonify, request, send_from_directory
import plotly.graph_objs as go
import plotly.utils
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Logger
import psutil

app = Flask(__name__)

class TrainingMonitor:
    """Real-time training monitor with database logging"""
    
    def __init__(self, db_path: str = "training_logs.db"):
        self.db_path = db_path
        self.training_process = None
        self.is_training = False
        self.init_database()
        
    def init_database(self):
        """Initialize SQLite database for logging"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create training runs table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS training_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                start_time TIMESTAMP,
                end_time TIMESTAMP,
                algorithm TEXT,
                total_timesteps INTEGER,
                learning_rate REAL,
                status TEXT,
                final_reward REAL,
                config TEXT
            )
        ''')
        
        # Create training metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS training_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER,
                timestamp TIMESTAMP,
                timestep INTEGER,
                episode INTEGER,
                reward REAL,
                episode_length INTEGER,
                mean_reward REAL,
                std_reward REAL,
                fps REAL,
                level REAL,
                badges INTEGER,
                hp_ratio REAL,
                position_x REAL,
                position_y REAL,
                map_id INTEGER,
                money INTEGER,
                FOREIGN KEY (run_id) REFERENCES training_runs (id)
            )
        ''')
        
        conn.commit()
        conn.close()
        
    def start_training(self, config: Dict) -> int:
        """Start a new training run"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO training_runs 
            (start_time, algorithm, total_timesteps, learning_rate, status, config)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now(),
            config.get('algorithm', 'ppo'),
            config.get('total_timesteps', 1000000),
            config.get('learning_rate', 3e-4),
            'running',
            json.dumps(config)
        ))
        
        run_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        # Start training process
        cmd = [
            'python', 'train.py',
            '--algorithm', config.get('algorithm', 'ppo'),
            '--total-timesteps', str(config.get('total_timesteps', 1000000)),
            '--emulator-path', '/usr/local/bin/bizhawk',
            '--rom-path', '../pokecrystal.gbc',
            '--learning-rate', str(config.get('learning_rate', 3e-4)),
            '--model-save-path', f'models/run_{run_id}',
            '--log-path', f'logs/run_{run_id}',
            '--run-id', str(run_id)  # Pass run ID to training script
        ]
        
        os.chdir('python_agent')
        self.training_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        os.chdir('..')
        
        self.is_training = True
        self.current_run_id = run_id
        
        # Start monitoring thread
        monitor_thread = threading.Thread(target=self._monitor_training)
        monitor_thread.daemon = True
        monitor_thread.start()
        
        return run_id
    
    def _monitor_training(self):
        """Monitor training process and log metrics"""
        while self.is_training and self.training_process:
            if self.training_process.poll() is not None:
                # Process ended
                self.is_training = False
                self._update_run_status('completed')
                break
                
            # Check for new log data
            self._update_metrics_from_logs()
            time.sleep(10)  # Update every 10 seconds
    
    def _update_metrics_from_logs(self):
        """Parse TensorBoard logs and update database"""
        log_dir = f"python_agent/logs/run_{self.current_run_id}"
        if not os.path.exists(log_dir):
            return
            
        # This would parse TensorBoard event files
        # For now, we'll simulate with dummy data
        # In a real implementation, you'd use tensorboard's event file parsing
        pass
    
    def stop_training(self):
        """Stop current training run"""
        if self.training_process:
            self.training_process.terminate()
            self.training_process.wait()
            self.is_training = False
            self._update_run_status('stopped')
    
    def _update_run_status(self, status: str):
        """Update training run status in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE training_runs 
            SET end_time = ?, status = ?
            WHERE id = ?
        ''', (datetime.now(), status, self.current_run_id))
        
        conn.commit()
        conn.close()
    
    def get_training_runs(self) -> List[Dict]:
        """Get all training runs"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM training_runs 
            ORDER BY start_time DESC
        ''')
        
        columns = [desc[0] for desc in cursor.description]
        runs = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        conn.close()
        return runs
    
    def get_run_metrics(self, run_id: int) -> pd.DataFrame:
        """Get metrics for a specific training run"""
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT * FROM training_metrics 
            WHERE run_id = ? 
            ORDER BY timestamp
        '''
        
        df = pd.read_sql_query(query, conn, params=(run_id,))
        conn.close()
        
        return df
    
    def get_system_stats(self) -> Dict:
        """Get current system statistics"""
        return {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'gpu_available': self._check_gpu(),
            'disk_usage': psutil.disk_usage('.').percent,
            'processes': len(psutil.pids()),
            'uptime': time.time() - psutil.boot_time()
        }
    
    def _check_gpu(self) -> bool:
        """Check if GPU is available"""
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False

# Global monitor instance
monitor = TrainingMonitor()

@app.route('/')
def dashboard():
    """Main dashboard page"""
    return render_template('dashboard.html')

@app.route('/api/runs')
def api_runs():
    """Get all training runs"""
    runs = monitor.get_training_runs()
    return jsonify(runs)

@app.route('/api/run/<int:run_id>/metrics')
def api_run_metrics(run_id):
    """Get metrics for specific run"""
    df = monitor.get_run_metrics(run_id)
    return jsonify(df.to_dict('records'))

@app.route('/api/run/<int:run_id>/plots')
def api_run_plots(run_id):
    """Get plots for specific run"""
    df = monitor.get_run_metrics(run_id)
    
    if df.empty:
        return jsonify({})
    
    # Create reward plot
    reward_trace = go.Scatter(
        x=df['timestep'],
        y=df['reward'],
        mode='lines',
        name='Reward',
        line=dict(color='#1f77b4')
    )
    
    reward_plot = {
        'data': [reward_trace],
        'layout': {
            'title': 'Episode Reward Over Time',
            'xaxis': {'title': 'Timesteps'},
            'yaxis': {'title': 'Reward'}
        }
    }
    
    # Create progress plot
    progress_traces = []
    if 'level' in df.columns:
        progress_traces.append(go.Scatter(
            x=df['timestep'],
            y=df['level'],
            mode='lines',
            name='Level',
            yaxis='y'
        ))
    
    if 'badges' in df.columns:
        progress_traces.append(go.Scatter(
            x=df['timestep'],
            y=df['badges'],
            mode='lines',
            name='Badges',
            yaxis='y2'
        ))
    
    progress_plot = {
        'data': progress_traces,
        'layout': {
            'title': 'Game Progress',
            'xaxis': {'title': 'Timesteps'},
            'yaxis': {'title': 'Level', 'side': 'left'},
            'yaxis2': {'title': 'Badges', 'side': 'right', 'overlaying': 'y'}
        }
    }
    
    return jsonify({
        'reward_plot': json.loads(plotly.utils.PlotlyJSONEncoder().encode(reward_plot)),
        'progress_plot': json.loads(plotly.utils.PlotlyJSONEncoder().encode(progress_plot))
    })

@app.route('/api/system')
def api_system():
    """Get system statistics"""
    return jsonify(monitor.get_system_stats())

@app.route('/api/start_training', methods=['POST'])
def api_start_training():
    """Start a new training run"""
    config = request.get_json()
    
    if monitor.is_training:
        return jsonify({'error': 'Training already in progress'}), 400
    
    try:
        run_id = monitor.start_training(config)
        return jsonify({'run_id': run_id, 'status': 'started'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stop_training', methods=['POST'])
def api_stop_training():
    """Stop current training run"""
    try:
        monitor.stop_training()
        return jsonify({'status': 'stopped'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/status')
def api_status():
    """Get current training status"""
    return jsonify({
        'is_training': monitor.is_training,
        'current_run_id': getattr(monitor, 'current_run_id', None)
    })

def cleanup_handler(signum, frame):
    """Clean shutdown handler"""
    print("Shutting down monitor...")
    if monitor.is_training:
        monitor.stop_training()
    sys.exit(0)

if __name__ == '__main__':
    # Setup signal handlers for clean shutdown
    signal.signal(signal.SIGINT, cleanup_handler)
    signal.signal(signal.SIGTERM, cleanup_handler)
    
    # Create templates directory
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    
    print("🚀 Starting Pokémon Crystal RL Training Monitor")
    print("📊 Dashboard available at: http://localhost:5000")
    print("Press Ctrl+C to stop")
    
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
