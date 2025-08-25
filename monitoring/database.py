"""
Database manager for Pokemon Crystal RL monitoring system.
"""

import sqlite3
import json
import shutil
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import pandas as pd


class DatabaseManager:
    """Manages SQLite database for monitoring data."""
    
    def __init__(self, db_path: Path):
        """Initialize database manager."""
        self.db_path = db_path
        self.logger = logging.getLogger('database')
        self._init_database()
    
    def _init_database(self):
        """Initialize database schema."""
        with self._connect() as conn:
            cursor = conn.cursor()
            
            # Create tables
            cursor.executescript("""
                -- Training runs table
                CREATE TABLE IF NOT EXISTS training_runs (
                    run_id TEXT PRIMARY KEY,
                    start_time DATETIME DEFAULT CURRENT_TIMESTAMP,
                    end_time DATETIME,
                    status TEXT,
                    total_episodes INTEGER DEFAULT 0,
                    total_steps INTEGER DEFAULT 0,
                    final_reward REAL,
                    config TEXT,
                    notes TEXT
                );
                
                -- Episodes table
                CREATE TABLE IF NOT EXISTS episodes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    episode_number INTEGER NOT NULL,
                    start_time DATETIME DEFAULT CURRENT_TIMESTAMP,
                    end_time DATETIME,
                    total_steps INTEGER,
                    total_reward REAL,
                    success BOOLEAN,
                    metadata TEXT,
                    FOREIGN KEY (run_id) REFERENCES training_runs(run_id),
                    UNIQUE (run_id, episode_number)
                );
                
                -- Steps table (episode_id can be NULL)
                CREATE TABLE IF NOT EXISTS steps (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    episode_id INTEGER,
                    step_number INTEGER NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    action TEXT NOT NULL,
                    reward REAL,
                    inference_time REAL,
                    state_id INTEGER,
                    FOREIGN KEY (run_id) REFERENCES training_runs(run_id),
                    FOREIGN KEY (episode_id) REFERENCES episodes(id),
                    FOREIGN KEY (state_id) REFERENCES game_states(id)
                );
                
                -- Performance metrics table
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    FOREIGN KEY (run_id) REFERENCES training_runs(run_id)
                );
                
                -- System metrics table
                CREATE TABLE IF NOT EXISTS system_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    cpu_percent REAL,
                    memory_percent REAL,
                    disk_usage REAL,
                    FOREIGN KEY (run_id) REFERENCES training_runs(run_id)
                );
                
                -- Game states table
                CREATE TABLE IF NOT EXISTS game_states (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    frame_number INTEGER,
                    map_id INTEGER,
                    player_x INTEGER,
                    player_y INTEGER,
                    state_data TEXT,
                    FOREIGN KEY (run_id) REFERENCES training_runs(run_id)
                );
                
                -- Events table
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    event_type TEXT NOT NULL,
                    event_data TEXT,
                    FOREIGN KEY (run_id) REFERENCES training_runs(run_id)
                );
                
                -- Add indices
                CREATE INDEX IF NOT EXISTS idx_episodes_run_id ON episodes(run_id);
                CREATE INDEX IF NOT EXISTS idx_steps_run_id ON steps(run_id);
                CREATE INDEX IF NOT EXISTS idx_steps_episode_id ON steps(episode_id);
                CREATE INDEX IF NOT EXISTS idx_perf_metrics_run_id ON performance_metrics(run_id);
                CREATE INDEX IF NOT EXISTS idx_perf_metrics_name ON performance_metrics(metric_name);
                CREATE INDEX IF NOT EXISTS idx_sys_metrics_run_id ON system_metrics(run_id);
                CREATE INDEX IF NOT EXISTS idx_game_states_run_id ON game_states(run_id);
                CREATE INDEX IF NOT EXISTS idx_game_states_frame ON game_states(frame_number);
                CREATE INDEX IF NOT EXISTS idx_events_run_id ON events(run_id);
                CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);
            """)
    
    def _connect(self) -> sqlite3.Connection:
        """Create database connection."""
        return sqlite3.connect(
            self.db_path,
            detect_types=sqlite3.PARSE_DECLTYPES
        )
    
    def start_training_run(self, config: Dict[str, Any]) -> str:
        """Start a new training run."""
        run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO training_runs (
                    run_id,
                    status,
                    config
                ) VALUES (?, ?, ?)
            """, (
                run_id,
                "initializing",
                json.dumps(config)
            ))
        
        return run_id
    
    def end_training_run(
        self,
        run_id: str,
        final_reward: Optional[float] = None
    ):
        """End a training run."""
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE training_runs
                SET end_time = CURRENT_TIMESTAMP,
                    status = 'completed',
                    final_reward = ?
                WHERE run_id = ?
            """, (final_reward, run_id))
    
    def update_run_status(self, run_id: str, status: str):
        """Update training run status."""
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE training_runs
                SET status = ?
                WHERE run_id = ?
            """, (status, run_id))
    
    def record_metrics(
        self,
        run_id: str,
        metrics: Dict[str, float],
        timestamp: Optional[datetime] = None
    ):
        """Record performance metrics."""
        timestamp = timestamp or datetime.now()
        
        with self._connect() as conn:
            cursor = conn.cursor()
            for name, value in metrics.items():
                cursor.execute("""
                    INSERT INTO performance_metrics (
                        run_id,
                        timestamp,
                        metric_name,
                        metric_value
                    ) VALUES (?, ?, ?, ?)
                """, (run_id, timestamp, name, value))
    
    def record_metric(
        self,
        run_id: str,
        metric_name: str,
        metric_value: float,
        timestamp: Optional[datetime] = None
    ):
        """Record a single performance metric."""
        timestamp = timestamp or datetime.now()
        
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO performance_metrics (
                    run_id,
                    timestamp,
                    metric_name,
                    metric_value
                ) VALUES (?, ?, ?, ?)
            """, (run_id, timestamp, metric_name, metric_value))
    
    def record_system_metrics(
        self,
        run_id: str,
        cpu_percent: float,
        memory_percent: float,
        disk_usage: float
    ):
        """Record system resource metrics."""
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO system_metrics (
                    run_id,
                    cpu_percent,
                    memory_percent,
                    disk_usage
                ) VALUES (?, ?, ?, ?)
            """, (run_id, cpu_percent, memory_percent, disk_usage))
    
    def record_game_state(
        self,
        run_id: str,
        state: Dict[str, Any],
        frame_number: Optional[int] = None
    ):
        """Record game state."""
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO game_states (
                    run_id,
                    frame_number,
                    map_id,
                    player_x,
                    player_y,
                    state_data
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                run_id,
                frame_number,
                state.get('map_id'),
                state.get('player_x'),
                state.get('player_y'),
                json.dumps(state)
            ))
    
    def record_step(
        self,
        run_id: str,
        step: int,
        reward: float,
        action: str,
        inference_time: float,
        game_state: Dict[str, Any]
    ):
        """Record training step."""
        with self._connect() as conn:
            cursor = conn.cursor()
            
            # Record game state
            cursor.execute("""
                INSERT INTO game_states (
                    run_id,
                    map_id,
                    player_x,
                    player_y,
                    state_data
                ) VALUES (?, ?, ?, ?, ?)
            """, (
                run_id,
                game_state.get('map_id'),
                game_state.get('player_x'),
                game_state.get('player_y'),
                json.dumps(game_state)
            ))
            state_id = cursor.lastrowid
            
            # Get current episode (optional)
            cursor.execute("""
                SELECT id
                FROM episodes
                WHERE run_id = ?
                ORDER BY episode_number DESC
                LIMIT 1
            """, (run_id,))
            result = cursor.fetchone()
            episode_id = result[0] if result else None
            
            # Record step (episode_id can be NULL)
            cursor.execute("""
                INSERT INTO steps (
                    run_id,
                    episode_id,
                    step_number,
                    action,
                    reward,
                    inference_time,
                    state_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                run_id,
                episode_id,
                step,
                action,
                reward,
                inference_time,
                state_id
            ))
            
            # Update run totals
            cursor.execute("""
                UPDATE training_runs
                SET total_steps = total_steps + 1
                WHERE run_id = ?
            """, (run_id,))
    
    def record_episode(
        self,
        run_id: str,
        episode: int,
        total_reward: float,
        steps: int,
        success: bool,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Record episode completion."""
        with self._connect() as conn:
            cursor = conn.cursor()
            
            # Record episode
            cursor.execute("""
                INSERT INTO episodes (
                    run_id,
                    episode_number,
                    total_steps,
                    total_reward,
                    success,
                    metadata,
                    end_time
                ) VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (
                run_id,
                episode,
                steps,
                total_reward,
                success,
                json.dumps(metadata) if metadata else None
            ))
            
            # Update run totals
            cursor.execute("""
                UPDATE training_runs
                SET total_episodes = total_episodes + 1
                WHERE run_id = ?
            """, (run_id,))
    
    def record_event(
        self,
        run_id: str,
        event_type: str,
        event_data: Dict[str, Any],
        timestamp: Optional[datetime] = None
    ):
        """Record training event."""
        timestamp = timestamp or datetime.now()
        
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO events (
                    run_id,
                    timestamp,
                    event_type,
                    event_data
                ) VALUES (?, ?, ?, ?)
            """, (
                run_id,
                timestamp,
                event_type,
                json.dumps(event_data)
            ))
    
    def get_training_runs(
        self,
        limit: Optional[int] = None,
        status: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get list of training runs."""
        query = "SELECT * FROM training_runs"
        params = []
        
        if status:
            query += " WHERE status = ?"
            params.append(status)
        
        query += " ORDER BY start_time DESC"
        
        if limit:
            query += " LIMIT ?"
            params.append(limit)
        
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            columns = [d[0] for d in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]
    
    def get_run_metrics(
        self,
        run_id: str,
        metric_names: Optional[List[str]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> pd.DataFrame:
        """Get metrics for a training run."""
        query = "SELECT * FROM performance_metrics WHERE run_id = ?"
        params = [run_id]
        
        if metric_names:
            query += " AND metric_name IN ({})".format(
                ','.join('?' * len(metric_names))
            )
            params.extend(metric_names)
        
        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time)
        
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time)
        
        query += " ORDER BY timestamp"
        
        with self._connect() as conn:
            df = pd.read_sql_query(query, conn, params=params)
            
            # If we have metrics data, pivot to have metric names as columns
            if not df.empty and 'metric_name' in df.columns and 'metric_value' in df.columns:
                # Use pandas pivot_table to transform the data
                pivoted = df.pivot_table(
                    index=['run_id', 'timestamp'], 
                    columns='metric_name', 
                    values='metric_value',
                    aggfunc='first'
                )
                
                # Reset index to make run_id and timestamp regular columns
                pivoted = pivoted.reset_index()
                
                # Add id column (use the first id for each timestamp group)
                id_mapping = df.groupby('timestamp')['id'].first().to_dict()
                pivoted['id'] = pivoted['timestamp'].map(id_mapping)
                
                # Reorder columns to match expected format
                cols = ['id', 'run_id', 'timestamp'] + [col for col in pivoted.columns if col not in ['id', 'run_id', 'timestamp']]
                pivoted = pivoted[cols]
                
                # Reset column names index
                pivoted.columns.name = None
                
                return pivoted
            else:
                return df
    
    def get_run_system_metrics(
        self,
        run_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> pd.DataFrame:
        """Get system metrics for a training run."""
        query = "SELECT * FROM system_metrics WHERE run_id = ?"
        params = [run_id]
        
        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time)
        
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time)
        
        query += " ORDER BY timestamp"
        
        with self._connect() as conn:
            return pd.read_sql_query(query, conn, params=params)
    
    def get_run_events(
        self,
        run_id: str,
        event_type: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """Get events for a training run."""
        query = "SELECT * FROM events WHERE run_id = ?"
        params = [run_id]
        
        if event_type:
            query += " AND event_type = ?"
            params.append(event_type)
        
        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time)
        
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time)
        
        query += " ORDER BY timestamp"
        
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            columns = [d[0] for d in cursor.description]
            events = []
            for row in cursor.fetchall():
                event = dict(zip(columns, row))
                if event['event_data']:
                    event['event_data'] = json.loads(event['event_data'])
                events.append(event)
            return events
    
    def get_run_episodes(
        self,
        run_id: str,
        start_episode: Optional[int] = None,
        end_episode: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get episodes for a training run."""
        query = "SELECT * FROM episodes WHERE run_id = ?"
        params = [run_id]
        
        if start_episode is not None:
            query += " AND episode_number >= ?"
            params.append(start_episode)
        
        if end_episode is not None:
            query += " AND episode_number <= ?"
            params.append(end_episode)
        
        query += " ORDER BY episode_number"
        
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            columns = [d[0] for d in cursor.description]
            episodes = []
            for row in cursor.fetchall():
                episode = dict(zip(columns, row))
                if episode['metadata']:
                    episode['metadata'] = json.loads(episode['metadata'])
                episodes.append(episode)
            return episodes
    
    def get_run_states(
        self,
        run_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """Get game states for a training run."""
        query = "SELECT * FROM game_states WHERE run_id = ?"
        params = [run_id]
        
        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time)
        
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time)
        
        query += " ORDER BY timestamp"
        
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            columns = [d[0] for d in cursor.description]
            states = []
            for row in cursor.fetchall():
                state = dict(zip(columns, row))
                if state['state_data']:
                    state['game_state'] = json.loads(state['state_data'])
                states.append(state)
            return states
    
    def get_run_summary(self, run_id: str) -> Dict[str, Any]:
        """Get summary of a training run."""
        with self._connect() as conn:
            cursor = conn.cursor()
            
            # Get run info
            cursor.execute("""
                SELECT *
                FROM training_runs
                WHERE run_id = ?
            """, (run_id,))
            columns = [d[0] for d in cursor.description]
            row = cursor.fetchone()
            if not row:
                return None
            
            summary = dict(zip(columns, row))
            summary['config'] = json.loads(summary['config'])
            
            # Get episode stats
            cursor.execute("""
                SELECT
                    COUNT(*) as total_episodes,
                    SUM(total_steps) as total_steps,
                    SUM(total_reward) as total_reward,
                    AVG(total_reward) as mean_reward,
                    SUM(CASE WHEN success THEN 1 ELSE 0 END) as successes
                FROM episodes
                WHERE run_id = ?
            """, (run_id,))
            stats = dict(zip(
                ['total_episodes', 'total_steps', 'total_reward', 'mean_reward', 'successes'],
                cursor.fetchone()
            ))
            summary.update(stats)
            
            return summary
    
    def export_run_data(
        self,
        run_id: str,
        output_dir: Path,
        include_snapshots: bool = True
    ) -> Path:
        """Export training run data."""
        # Create temporary export directory
        export_dir = output_dir / f"export_{run_id}"
        export_dir.mkdir(parents=True, exist_ok=True)
        
        with self._connect() as conn:
            # Export tables to CSV
            tables = [
                'training_runs',
                'episodes',
                'steps',
                'performance_metrics',
                'system_metrics',
                'game_states',
                'events'
            ]
            
            for table in tables:
                df = pd.read_sql_query(
                    f"SELECT * FROM {table} WHERE run_id = ?",
                    conn,
                    params=[run_id]
                )
                df.to_csv(export_dir / f"{table}.csv", index=False)
        
        # Export snapshots if requested
        if include_snapshots and hasattr(self, 'data_dir'):
            snapshots_dir = export_dir / "snapshots"
            snapshots_dir.mkdir(exist_ok=True)
            
            snapshot_files = Path(self.data_dir).glob(f"snapshot_{run_id}_*.json")
            for snapshot in snapshot_files:
                shutil.copy2(snapshot, snapshots_dir)
        
        # Create zip archive
        output_path = output_dir / f"export_{run_id}.zip"
        shutil.make_archive(
            str(output_path.with_suffix("")),
            'zip',
            export_dir
        )
        
        # Clean up temporary directory
        shutil.rmtree(export_dir)
        
        return output_path
    
    def cleanup_old_data(
        self,
        older_than: datetime,
        dry_run: bool = False
    ) -> int:
        """Remove old monitoring data."""
        with self._connect() as conn:
            cursor = conn.cursor()
            
            # Get old run IDs
            cursor.execute("""
                SELECT run_id
                FROM training_runs
                WHERE end_time < ?
                    AND status = 'completed'
            """, (older_than,))
            run_ids = [r[0] for r in cursor.fetchall()]
            
            if not run_ids or dry_run:
                return len(run_ids)
            
            # Delete data from all tables
            tables = [
                'events',
                'game_states',
                'system_metrics',
                'performance_metrics',
                'steps',
                'episodes',
                'training_runs'
            ]
            
            for table in tables:
                cursor.execute(f"""
                    DELETE FROM {table}
                    WHERE run_id IN ({','.join('?' * len(run_ids))})
                """, run_ids)
            
            return len(run_ids)
    
    def optimize_database(self):
        """Optimize database and reclaim space."""
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("VACUUM")
