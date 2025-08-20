"""
Migration utilities for Pokemon Crystal RL monitoring system.

Handles:
- Database schema migrations
- Configuration format updates
- Data format conversions
- Backward compatibility layers
"""

import sqlite3
import json
import logging
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
from datetime import datetime

from .database import DatabaseManager
from .error_handler import ErrorHandler, ErrorCategory, ErrorSeverity


class MigrationError(Exception):
    """Exception raised for migration errors."""
    pass


class ConfigMigrator:
    """Handles configuration file migrations."""
    
    CURRENT_VERSION = 1
    
    @staticmethod
    def migrate_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate configuration to latest version."""
        version = config.get('version', 0)
        
        if version < ConfigMigrator.CURRENT_VERSION:
            # Add new required fields
            if 'db_path' not in config:
                config['db_path'] = 'monitoring.db'
            if 'data_dir' not in config:
                config['data_dir'] = 'monitor_data'
            if 'snapshot_interval' not in config:
                config['snapshot_interval'] = 300.0
            
            # Update version
            config['version'] = ConfigMigrator.CURRENT_VERSION
        
        return config
    
    @staticmethod
    def backup_config(config_path: Path) -> Path:
        """Create backup of existing config file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = config_path.with_suffix(f'.backup_{timestamp}')
        shutil.copy2(config_path, backup_path)
        return backup_path


class DatabaseMigrator:
    """Handles database schema migrations."""
    
    CURRENT_VERSION = 1
    
    def __init__(self, db_path: Path, error_handler: Optional[ErrorHandler] = None):
        self.db_path = db_path
        self.error_handler = error_handler or ErrorHandler()
        self.backup_path = None
    
    def get_db_version(self, conn: sqlite3.Connection) -> int:
        """Get current database version."""
        try:
            cursor = conn.cursor()
            cursor.execute("PRAGMA user_version")
            return cursor.fetchone()[0]
        except sqlite3.Error as e:
            raise MigrationError(f"Failed to get database version: {e}")
    
    def backup_database(self) -> Path:
        """Create backup of existing database."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.backup_path = self.db_path.with_suffix(f'.backup_{timestamp}')
        shutil.copy2(self.db_path, self.backup_path)
        return self.backup_path
    
    def migrate(self) -> bool:
        """Perform database migration if needed."""
        if not self.db_path.exists():
            return False
            
        try:
            conn = sqlite3.connect(self.db_path)
            current_version = self.get_db_version(conn)
            
            if current_version < self.CURRENT_VERSION:
                self.backup_database()
                self._migrate_schema(conn, current_version)
                return True
                
        except Exception as e:
            error_msg = f"Database migration failed: {e}"
            if self.backup_path:
                error_msg += f"\nBackup available at: {self.backup_path}"
            
            self.error_handler.handle_error(
                error=e,
                message=error_msg,
                category=ErrorCategory.DATABASE,
                severity=ErrorSeverity.ERROR
            )
            raise MigrationError(error_msg) from e
            
        finally:
            conn.close()
            
        return False
    
    def _migrate_schema(self, conn: sqlite3.Connection, from_version: int):
        """Migrate database schema to latest version."""
        cursor = conn.cursor()
        
        try:
            # Migration steps based on version
            if from_version < 1:
                # Add new tables
                cursor.executescript("""
                    -- New performance metrics table
                    CREATE TABLE IF NOT EXISTS performance_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        run_id TEXT NOT NULL,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        metric_name TEXT NOT NULL,
                        metric_value REAL NOT NULL,
                        FOREIGN KEY (run_id) REFERENCES training_runs(run_id)
                    );
                    
                    -- New system metrics table
                    CREATE TABLE IF NOT EXISTS system_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        run_id TEXT NOT NULL,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        cpu_percent REAL,
                        memory_percent REAL,
                        disk_usage REAL,
                        FOREIGN KEY (run_id) REFERENCES training_runs(run_id)
                    );
                    
                    -- Add indices
                    CREATE INDEX IF NOT EXISTS idx_perf_metrics_run_id 
                    ON performance_metrics(run_id);
                    
                    CREATE INDEX IF NOT EXISTS idx_sys_metrics_run_id 
                    ON system_metrics(run_id);
                """)
                
                # Migrate existing data if needed
                self._migrate_data_v0_to_v1(cursor)
            
            # Update version
            cursor.execute(f"PRAGMA user_version = {self.CURRENT_VERSION}")
            conn.commit()
            
        except sqlite3.Error as e:
            conn.rollback()
            raise MigrationError(f"Schema migration failed: {e}")
    
    def _migrate_data_v0_to_v1(self, cursor: sqlite3.Connection):
        """Migrate data from version 0 to 1."""
        try:
            # Check if old metrics table exists
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='metrics'
            """)
            
            if cursor.fetchone():
                # Migrate old metrics to new tables
                cursor.execute("""
                    INSERT INTO performance_metrics (
                        run_id, timestamp, metric_name, metric_value
                    )
                    SELECT 
                        run_id,
                        timestamp,
                        metric_name,
                        metric_value
                    FROM metrics
                    WHERE metric_type = 'performance'
                """)
                
                cursor.execute("""
                    INSERT INTO system_metrics (
                        run_id,
                        timestamp,
                        cpu_percent,
                        memory_percent,
                        disk_usage
                    )
                    SELECT 
                        run_id,
                        timestamp,
                        CAST(json_extract(metric_data, '$.cpu_percent') AS REAL),
                        CAST(json_extract(metric_data, '$.memory_percent') AS REAL),
                        CAST(json_extract(metric_data, '$.disk_usage') AS REAL)
                    FROM metrics
                    WHERE metric_type = 'system'
                """)
                
                # Drop old table
                cursor.execute("DROP TABLE metrics")
        
        except sqlite3.Error as e:
            raise MigrationError(f"Data migration failed: {e}")


class DataMigrator:
    """Handles data format migrations."""
    
    @staticmethod
    def migrate_snapshots(
        data_dir: Path,
        old_format: str,
        new_format: str
    ) -> List[Path]:
        """Migrate snapshot files to new format."""
        migrated_files = []
        
        # Find all snapshot files
        snapshot_files = list(data_dir.glob("snapshot_*.json"))
        
        for snapshot_file in snapshot_files:
            try:
                # Read old format
                with open(snapshot_file) as f:
                    data = json.load(f)
                
                # Convert format
                if old_format == "v0" and new_format == "v1":
                    data = DataMigrator._convert_v0_to_v1(data)
                
                # Save in new format
                backup_file = snapshot_file.with_suffix(f".{old_format}")
                shutil.move(snapshot_file, backup_file)
                
                with open(snapshot_file, 'w') as f:
                    json.dump(data, f, indent=2)
                
                migrated_files.append(snapshot_file)
                
            except Exception as e:
                logging.error(f"Failed to migrate {snapshot_file}: {e}")
                # Continue with other files
                continue
        
        return migrated_files
    
    @staticmethod
    def _convert_v0_to_v1(data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert snapshot data from v0 to v1 format."""
        # Add new required fields
        if 'version' not in data:
            data['version'] = 1
        if 'metadata' not in data:
            data['metadata'] = {
                'converted_from': 'v0',
                'conversion_time': datetime.now().isoformat()
            }
        
        # Update metric format
        if 'metrics' in data:
            metrics = data['metrics']
            if isinstance(metrics, dict):
                data['metrics'] = {
                    'performance': metrics.get('performance', {}),
                    'system': metrics.get('system', {}),
                    'custom': metrics.get('custom', {})
                }
        
        return data
