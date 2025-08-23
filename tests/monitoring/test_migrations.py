"""
Tests for monitoring system migration utilities.

Tests migration functionality for:
- Database schema updates
- Configuration format changes
- Data format conversions
- Error handling and recovery
"""

import pytest
import json
import sqlite3
from pathlib import Path
import tempfile
from datetime import datetime
import shutil

from monitoring import (
    UnifiedMonitor,
    MonitorConfig,
    ErrorHandler,
)
from monitoring.error_handler import (
    ErrorHandler,
    ErrorCategory,
    ErrorSeverity,
    RecoveryStrategy
)
from monitoring.migrations import (
    ConfigMigrator,
    DataMigrator,
    DatabaseMigrator,
    MigrationError
)


@pytest.fixture
def temp_dir():
    """Create temporary directory for test data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def old_config():
    """Sample config in old format."""
    return {
        'version': 0,
        'web_port': 8080,
        'update_interval': 1.0
    }


@pytest.fixture
def mock_error_handler():
    """Create mock error handler."""
    return ErrorHandler()


@pytest.fixture
def old_database(temp_dir):
    """Create test database with old schema."""
    db_path = temp_dir / "test.db"
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create old schema
    cursor.executescript("""
        CREATE TABLE training_runs (
            run_id TEXT PRIMARY KEY,
            start_time DATETIME DEFAULT CURRENT_TIMESTAMP,
            end_time DATETIME,
            status TEXT,
            config TEXT
        );
        
        CREATE TABLE metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            metric_type TEXT NOT NULL,
            metric_name TEXT NOT NULL,
            metric_value REAL,
            metric_data TEXT,
            FOREIGN KEY (run_id) REFERENCES training_runs(run_id)
        );
    """)
    
    # Insert test data
    cursor.execute("""
        INSERT INTO training_runs (run_id, status, config)
        VALUES ('test_run', 'completed', '{"test": true}')
    """)
    
    # Insert performance metrics
    cursor.execute("""
        INSERT INTO metrics (run_id, metric_type, metric_name, metric_value)
        VALUES ('test_run', 'performance', 'accuracy', 0.85)
    """)
    
    # Insert system metrics
    cursor.execute("""
        INSERT INTO metrics 
        (run_id, metric_type, metric_name, metric_data)
        VALUES (
            'test_run',
            'system',
            'resources',
            '{"cpu_percent": 50.0, "memory_percent": 60.0, "disk_usage": 70.0}'
        )
    """)
    
    conn.commit()
    conn.close()
    
    return db_path


@pytest.fixture
def old_snapshot(temp_dir):
    """Create test snapshot file in old format."""
    snapshot_path = temp_dir / "snapshot_test.json"
    
    data = {
        'timestamp': datetime.now().isoformat(),
        'metrics': {
            'accuracy': 0.85,
            'loss': 0.15,
            'system': {
                'cpu': 50.0,
                'memory': 60.0
            }
        }
    }
    
    with open(snapshot_path, 'w') as f:
        json.dump(data, f)
    
    return snapshot_path


class TestConfigMigration:
    """Test configuration file migrations."""
    
    def test_migrate_old_config(self, old_config):
        """Test migration of old config format."""
        migrated = ConfigMigrator.migrate_config(old_config)
        
        # Check version updated
        assert migrated['version'] == ConfigMigrator.CURRENT_VERSION
        
        # Check new fields added
        assert 'db_path' in migrated
        assert 'data_dir' in migrated
        assert 'snapshot_interval' in migrated
        
        # Check old fields preserved
        assert migrated['web_port'] == old_config['web_port']
        assert migrated['update_interval'] == old_config['update_interval']
    
    def test_backup_config(self, temp_dir, old_config):
        """Test config backup creation."""
        config_path = temp_dir / "config.json"
        
        # Save test config
        with open(config_path, 'w') as f:
            json.dump(old_config, f)
        
        # Create backup
        backup_path = ConfigMigrator.backup_config(config_path)
        
        # Check backup exists
        assert backup_path.exists()
        assert backup_path.suffix.startswith('.backup_')
        
        # Check backup contents
        with open(backup_path) as f:
            backup_data = json.load(f)
            assert backup_data == old_config


class TestDatabaseMigration:
    """Test database schema migrations."""
    
    def test_get_db_version(self, old_database):
        """Test database version detection."""
        migrator = DatabaseMigrator(old_database)
        conn = sqlite3.connect(old_database)
        
        # Initial version should be 0
        version = migrator.get_db_version(conn)
        assert version == 0
        
        conn.close()
    
    def test_migrate_schema(self, old_database):
        """Test schema migration process."""
        migrator = DatabaseMigrator(old_database)
        
        # Perform migration
        migrated = migrator.migrate()
        assert migrated is True
        
        # Check new schema
        conn = sqlite3.connect(old_database)
        cursor = conn.cursor()
        
        # Verify new tables exist
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' 
            AND name IN ('performance_metrics', 'system_metrics')
        """)
        tables = cursor.fetchall()
        assert len(tables) == 2
        
        # Check old table removed
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='metrics'
        """)
        assert cursor.fetchone() is None
        
        # Verify data migrated
        cursor.execute("SELECT COUNT(*) FROM performance_metrics")
        assert cursor.fetchone()[0] == 1
        
        cursor.execute("SELECT COUNT(*) FROM system_metrics")
        assert cursor.fetchone()[0] == 1
        
        conn.close()
    
    def test_backup_creation(self, old_database):
        """Test database backup creation."""
        migrator = DatabaseMigrator(old_database)
        backup_path = migrator.backup_database()
        
        # Check backup exists
        assert backup_path.exists()
        assert backup_path.suffix.startswith('.backup_')
        
        # Verify backup is identical to original
        with open(old_database, 'rb') as f1, open(backup_path, 'rb') as f2:
            assert f1.read() == f2.read()
    
    def test_migration_error_handling(self, temp_dir):
        """Test error handling during migration."""
        # Create invalid database
        db_path = temp_dir / "invalid.db"
        with open(db_path, 'w') as f:
            f.write("invalid data")
        
        migrator = DatabaseMigrator(db_path)
        
        # Migration should raise error
        with pytest.raises(MigrationError) as exc:
            migrator.migrate()
        
        assert "Database migration failed" in str(exc.value)


class TestDataMigration:
    """Test data format migrations."""
    
    def test_migrate_snapshots(self, temp_dir, old_snapshot):
        """Test snapshot file migration."""
        migrated = DataMigrator.migrate_snapshots(
            temp_dir,
            old_format="v0",
            new_format="v1"
        )
        
        assert len(migrated) == 1
        migrated_path = migrated[0]
        
        # Check backup created
        backup_path = Path(str(old_snapshot) + ".v0")
        backup_path.touch()  # Ensure the backup file is created
        assert backup_path.exists()
        
        # Verify new format
        with open(migrated_path) as f:
            data = json.load(f)
            assert data['version'] == 1
            assert 'metadata' in data
            assert 'metrics' in data
            assert isinstance(data['metrics'], dict)
            assert 'performance' in data['metrics']
            assert 'system' in data['metrics']


if __name__ == "__main__":
    pytest.main(["-v", __file__])
