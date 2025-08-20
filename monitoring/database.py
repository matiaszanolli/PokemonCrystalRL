"""
Database module for Pokemon Crystal RL monitoring.

Provides persistent storage for:
- Training runs and configurations
- Performance metrics and statistics
- Game events and progress
- System monitoring data
"""

import sqlite3
import json
import time
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import threading
from contextlib import contextmanager
import logging
import pandas as pd
import numpy as np


class DatabaseManager:
    """Manages SQLite database for monitoring data."""
    
    SCHEMA_VERSION = 1
    
    def __init__(self, db_path: Union[str, Path] = "monitoring.db"):
        self.db_path = Path(db_path)
        self.logger = logging.getLogger(__name__)
        self._connection_lock = threading.Lock()
        
        # Ensure database directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()
        self.logger.info("📀 Database initialized")
    
    @contextmanager
    def _get_connection(self):
        """Get a database connection with proper locking."""
        with self._connection_lock:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row  # Return rows as dictionaries
            try:
                yield conn
            finally:
                conn.close()
    
    def _init_database(self) -> None:
        """Initialize database schema."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Create schema version table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS schema_version (
                    version INTEGER PRIMARY KEY,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Check schema version
            cursor.execute('SELECT version FROM schema_version')
            result = cursor.fetchone()
            current_version = result['version'] if result else 0
            
            if current_version < self.SCHEMA_VERSION:
                self._create_schema(cursor)
                cursor.execute('INSERT OR REPLACE INTO schema_version (version) VALUES (?)',
                             (self.SCHEMA_VERSION,))
            
            conn.commit()
    
    def _create_schema(self, cursor: sqlite3.Cursor) -> None:
        """Create database schema."""
        # Training runs table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS training_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                start_time TIMESTAMP,
                end_time TIMESTAMP,
                status TEXT,
                total_episodes INTEGER DEFAULT 0,
                total_steps INTEGER DEFAULT 0,
                final_reward REAL,
                config TEXT,
                metadata TEXT
            )
        ''')
        
        # Episodes table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS episodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER,
                episode_number INTEGER,
                start_time TIMESTAMP,
                end_time TIMESTAMP,
                total_steps INTEGER,
                total_reward REAL,
                success BOOLEAN,
                metadata TEXT,
                FOREIGN KEY (run_id) REFERENCES training_runs (id)
            )
        ''')
        
        # Metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER,
                episode_id INTEGER,
                timestamp TIMESTAMP,
                metric_type TEXT,
                metric_name TEXT,
                value REAL,
                metadata TEXT,
                FOREIGN KEY (run_id) REFERENCES training_runs (id),
                FOREIGN KEY (episode_id) REFERENCES episodes (id)
            )
        ''')
        
        # Game states table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS game_states (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER,
                episode_id INTEGER,
                timestamp TIMESTAMP,
                state_type TEXT,
                map_id INTEGER,
                player_x INTEGER,
                player_y INTEGER,
                metadata TEXT,
                FOREIGN KEY (run_id) REFERENCES training_runs (id),
                FOREIGN KEY (episode_id) REFERENCES episodes (id)
            )
        ''')
        
        # Events table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER,
                episode_id INTEGER,
                timestamp TIMESTAMP,
                event_type TEXT,
                event_data TEXT,
                metadata TEXT,
                FOREIGN KEY (run_id) REFERENCES training_runs (id),
                FOREIGN KEY (episode_id) REFERENCES episodes (id)
            )
        ''')
        
        # System metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS system_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER,
                timestamp TIMESTAMP,
                cpu_percent REAL,
                memory_percent REAL,
                gpu_usage REAL,
                disk_usage REAL,
                metadata TEXT,
                FOREIGN KEY (run_id) REFERENCES training_runs (id)
            )
        ''')
        
        # Create indexes
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_run_time ON training_runs (start_time)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_episode_run ON episodes (run_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_metrics_run ON metrics (run_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_metrics_type ON metrics (metric_type)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_events_run ON events (run_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_events_type ON events (event_type)')
    
    def start_training_run(self, config: Dict[str, Any] = None,
                         metadata: Dict[str, Any] = None) -> int:
        """Start a new training run."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO training_runs
                (start_time, status, config, metadata)
                VALUES (?, ?, ?, ?)
            ''', (
                datetime.now().isoformat(),
                'running',
                json.dumps(config or {}),
                json.dumps(metadata or {})
            ))
            
            run_id = cursor.lastrowid
            conn.commit()
            
            self.logger.info(f"📝 Started training run {run_id}")
            return run_id
    
    def end_training_run(self, run_id: int, status: str = 'completed',
                        final_reward: Optional[float] = None) -> None:
        """End a training run."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE training_runs
                SET end_time = ?, status = ?, final_reward = ?
                WHERE id = ?
            ''', (
                datetime.now().isoformat(),
                status,
                final_reward,
                run_id
            ))
            
            conn.commit()
            self.logger.info(f"✅ Ended training run {run_id} with status: {status}")
    
    def record_episode(self, run_id: int, episode_number: int,
                      steps: int, reward: float, success: bool,
                      metadata: Dict[str, Any] = None) -> int:
        """Record an episode completion."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO episodes
                (run_id, episode_number, start_time, end_time,
                 total_steps, total_reward, success, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                run_id,
                episode_number,
                datetime.now().isoformat(),
                datetime.now().isoformat(),
                steps,
                reward,
                success,
                json.dumps(metadata or {})
            ))
            
            episode_id = cursor.lastrowid
            
            # Update run statistics
            cursor.execute('''
                UPDATE training_runs
                SET total_episodes = total_episodes + 1,
                    total_steps = total_steps + ?
                WHERE id = ?
            ''', (steps, run_id))
            
            conn.commit()
            return episode_id
    
    def record_metrics(self, run_id: int, metrics: Dict[str, Any],
                      episode_id: Optional[int] = None) -> None:
        """Record training metrics."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            timestamp = datetime.now().isoformat()
            
            for metric_type, type_metrics in metrics.items():
                if isinstance(type_metrics, dict):
                    for name, value in type_metrics.items():
                        if isinstance(value, (int, float)):
                            cursor.execute('''
                                INSERT INTO metrics
                                (run_id, episode_id, timestamp,
                                 metric_type, metric_name, value)
                                VALUES (?, ?, ?, ?, ?, ?)
                            ''', (
                                run_id,
                                episode_id,
                                timestamp,
                                metric_type,
                                name,
                                value
                            ))
            
            conn.commit()
    
    def record_game_state(self, run_id: int, state_type: str,
                         map_id: int, player_x: int, player_y: int,
                         metadata: Dict[str, Any] = None,
                         episode_id: Optional[int] = None) -> None:
        """Record game state."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO game_states
                (run_id, episode_id, timestamp, state_type,
                 map_id, player_x, player_y, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                run_id,
                episode_id,
                datetime.now().isoformat(),
                state_type,
                map_id,
                player_x,
                player_y,
                json.dumps(metadata or {})
            ))
            
            conn.commit()
    
    def record_event(self, run_id: int, event_type: str,
                    event_data: Dict[str, Any],
                    metadata: Dict[str, Any] = None,
                    episode_id: Optional[int] = None) -> None:
        """Record an event."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO events
                (run_id, episode_id, timestamp, event_type,
                 event_data, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                run_id,
                episode_id,
                datetime.now().isoformat(),
                event_type,
                json.dumps(event_data),
                json.dumps(metadata or {})
            ))
            
            conn.commit()
    
    def record_system_metrics(self, run_id: int,
                            cpu_percent: float,
                            memory_percent: float,
                            gpu_usage: Optional[float] = None,
                            disk_usage: Optional[float] = None,
                            metadata: Dict[str, Any] = None) -> None:
        """Record system performance metrics."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO system_metrics
                (run_id, timestamp, cpu_percent, memory_percent,
                 gpu_usage, disk_usage, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                run_id,
                datetime.now().isoformat(),
                cpu_percent,
                memory_percent,
                gpu_usage,
                disk_usage,
                json.dumps(metadata or {})
            ))
            
            conn.commit()
    
    def get_training_runs(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent training runs."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM training_runs
                ORDER BY start_time DESC
                LIMIT ?
            ''', (limit,))
            
            return [dict(row) for row in cursor.fetchall()]
    
    def get_run_metrics(self, run_id: int,
                       metric_type: Optional[str] = None) -> pd.DataFrame:
        """Get metrics for a specific run."""
        with self._get_connection() as conn:
            query = '''
                SELECT timestamp, metric_type, metric_name, value
                FROM metrics
                WHERE run_id = ?
            '''
            params = [run_id]
            
            if metric_type:
                query += ' AND metric_type = ?'
                params.append(metric_type)
            
            query += ' ORDER BY timestamp'
            
            return pd.read_sql_query(query, conn, params=params)
    
    def get_run_events(self, run_id: int,
                      event_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get events for a specific run."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            query = 'SELECT * FROM events WHERE run_id = ?'
            params = [run_id]
            
            if event_type:
                query += ' AND event_type = ?'
                params.append(event_type)
            
            query += ' ORDER BY timestamp'
            
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
    
    def get_run_system_metrics(self, run_id: int) -> pd.DataFrame:
        """Get system metrics for a specific run."""
        with self._get_connection() as conn:
            query = '''
                SELECT timestamp, cpu_percent, memory_percent,
                       gpu_usage, disk_usage
                FROM system_metrics
                WHERE run_id = ?
                ORDER BY timestamp
            '''
            
            return pd.read_sql_query(query, conn, params=[run_id])
    
    def get_run_summary(self, run_id: int) -> Dict[str, Any]:
        """Get a summary of a training run."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Get run info
            cursor.execute('SELECT * FROM training_runs WHERE id = ?', (run_id,))
            run = dict(cursor.fetchone())
            
            # Get episode stats
            cursor.execute('''
                SELECT COUNT(*) as episode_count,
                       AVG(total_reward) as avg_reward,
                       MAX(total_reward) as max_reward,
                       SUM(total_steps) as total_steps,
                       AVG(success) as success_rate
                FROM episodes
                WHERE run_id = ?
            ''', (run_id,))
            stats = dict(cursor.fetchone())
            
            # Merge stats
            run.update(stats)
            return run
    
    def export_run_data(self, run_id: int, format: str = 'json') -> Dict[str, Any]:
        """Export all data for a training run."""
        data = {
            'run': self.get_run_summary(run_id),
            'metrics': self.get_run_metrics(run_id).to_dict('records'),
            'events': self.get_run_events(run_id),
            'system_metrics': self.get_run_system_metrics(run_id).to_dict('records')
        }
        
        if format == 'json':
            return data
        elif format == 'csv':
            # Convert to CSV format
            csv_data = {}
            for key, value in data.items():
                if isinstance(value, list):
                    csv_data[key] = pd.DataFrame(value).to_csv(index=False)
                else:
                    csv_data[key] = pd.DataFrame([value]).to_csv(index=False)
            return csv_data
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def cleanup_old_data(self, days: int = 30) -> None:
        """Clean up data older than specified days."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            threshold = (datetime.now() - pd.Timedelta(days=days)).isoformat()
            
            for table in ['training_runs', 'episodes', 'metrics',
                         'game_states', 'events', 'system_metrics']:
                cursor.execute(f'''
                    DELETE FROM {table}
                    WHERE timestamp < ?
                ''', (threshold,))
            
            conn.commit()
            self.logger.info(f"🧹 Cleaned up data older than {days} days")
