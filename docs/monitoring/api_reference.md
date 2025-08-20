# Python API Reference

## WebMonitor

The `WebMonitor` class is the main interface for the monitoring system.

```python
from pokemon_crystal_rl.monitoring import WebMonitor, MonitorConfig
```

### Configuration

```python
class MonitorConfig:
    """Configuration for the monitoring system."""
    
    def __init__(
        db_path: str = "monitoring.db",
        data_dir: str = "monitor_data",
        static_dir: str = "static",
        web_port: int = 8080,
        update_interval: float = 1.0,
        snapshot_interval: float = 300.0,
        max_events: int = 1000,
        max_snapshots: int = 100,
        debug: bool = False
    ):
        ...
```

Configuration parameters:
- `db_path`: Path to SQLite database file
- `data_dir`: Directory for storing monitoring data
- `static_dir`: Directory for static web files
- `web_port`: Port for web interface
- `update_interval`: Interval between metric updates
- `snapshot_interval`: Interval between state snapshots
- `max_events`: Maximum events to keep in memory
- `max_snapshots`: Maximum snapshots to retain
- `debug`: Enable debug mode

### Training Management

```python
def start_training(
    self,
    config: Optional[Dict[str, Any]] = None,
    run_id: Optional[str] = None
) -> str:
    """Start monitoring a training run."""
```

Parameters:
- `config`: Training configuration dictionary
- `run_id`: Optional run ID for resuming training
Returns:
- `str`: Run ID for the training session

```python
def stop_training(
    self,
    final_reward: Optional[float] = None
) -> None:
    """Stop the current training run."""
```

Parameters:
- `final_reward`: Final reward value for the run

```python
def pause_training(self) -> None:
    """Pause the current training run."""
```

```python
def resume_training(self) -> None:
    """Resume a paused training run."""
```

### Metric Recording

```python
def update_metrics(
    self,
    metrics: Dict[str, Union[float, Dict[str, float]]],
    timestamp: Optional[datetime] = None
) -> None:
    """Record training metrics."""
```

Parameters:
- `metrics`: Dictionary of metric names and values
- `timestamp`: Optional timestamp for the metrics

```python
def update_system_metrics(
    self,
    cpu_percent: float,
    memory_percent: float,
    disk_usage: float
) -> None:
    """Record system resource metrics."""
```

Parameters:
- `cpu_percent`: CPU usage percentage
- `memory_percent`: Memory usage percentage
- `disk_usage`: Disk usage percentage

### Game State Recording

```python
def update_game_state(
    self,
    state: Dict[str, Any],
    frame_number: Optional[int] = None
) -> None:
    """Record game state information."""
```

Parameters:
- `state`: Dictionary containing game state
- `frame_number`: Optional frame number

```python
def update_step(
    self,
    step: int,
    reward: float,
    action: str,
    inference_time: float,
    game_state: Dict[str, Any]
) -> None:
    """Record training step information."""
```

Parameters:
- `step`: Step number in current episode
- `reward`: Reward received for step
- `action`: Action taken
- `inference_time`: Time taken for inference
- `game_state`: Current game state

### Event Handling

```python
def record_event(
    self,
    event_type: str,
    event_data: Dict[str, Any],
    timestamp: Optional[datetime] = None
) -> None:
    """Record a training event."""
```

Parameters:
- `event_type`: Type of event
- `event_data`: Event details
- `timestamp`: Optional event timestamp

### Data Retrieval

```python
def get_training_runs(
    self,
    limit: Optional[int] = None,
    status: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Get list of training runs."""
```

Parameters:
- `limit`: Maximum runs to return
- `status`: Filter by run status
Returns:
- List of training run dictionaries

```python
def get_run_metrics(
    self,
    run_id: str,
    metric_names: Optional[List[str]] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None
) -> pd.DataFrame:
    """Get metrics for a training run."""
```

Parameters:
- `run_id`: Training run ID
- `metric_names`: Optional list of metrics to retrieve
- `start_time`: Optional start time filter
- `end_time`: Optional end time filter
Returns:
- DataFrame containing metrics

```python
def get_run_events(
    self,
    run_id: str,
    event_type: Optional[str] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None
) -> List[Dict[str, Any]]:
    """Get events for a training run."""
```

Parameters:
- `run_id`: Training run ID
- `event_type`: Optional event type filter
- `start_time`: Optional start time filter
- `end_time`: Optional end time filter
Returns:
- List of event dictionaries

```python
def get_run_summary(
    self,
    run_id: str
) -> Dict[str, Any]:
    """Get summary of a training run."""
```

Parameters:
- `run_id`: Training run ID
Returns:
- Dictionary containing run summary

### Data Export

```python
def export_run_data(
    self,
    run_id: str,
    output_dir: Path,
    include_snapshots: bool = True
) -> Path:
    """Export training run data."""
```

Parameters:
- `run_id`: Training run ID
- `output_dir`: Directory for exported data
- `include_snapshots`: Whether to include state snapshots
Returns:
- Path to exported data archive

## Error Handler

The `ErrorHandler` class manages error detection and recovery.

```python
from pokemon_crystal_rl.monitoring import ErrorHandler, ErrorCategory, ErrorSeverity
```

```python
def handle_error(
    self,
    error: Exception,
    message: Optional[str] = None,
    category: ErrorCategory = ErrorCategory.UNKNOWN,
    severity: ErrorSeverity = ErrorSeverity.ERROR,
    component: Optional[str] = None
) -> None:
    """Handle an error event."""
```

Parameters:
- `error`: Exception that occurred
- `message`: Optional error message
- `category`: Error category
- `severity`: Error severity level
- `component`: Component where error occurred

### Error Categories

```python
class ErrorCategory(Enum):
    UNKNOWN = "unknown"
    SYSTEM = "system"
    TRAINING = "training"
    DATABASE = "database"
    NETWORK = "network"
    CONFIG = "config"
```

### Error Severities

```python
class ErrorSeverity(Enum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
```

## Database Manager

The `DatabaseManager` class handles data persistence.

```python
from pokemon_crystal_rl.monitoring import DatabaseManager
```

### Schema Management

```python
def init_database(self) -> None:
    """Initialize database schema."""
```

```python
def get_schema_version(self) -> int:
    """Get current schema version."""
```

```python
def migrate_schema(
    self,
    target_version: Optional[int] = None
) -> None:
    """Migrate schema to target version."""
```

### Data Management

```python
def cleanup_old_data(
    self,
    older_than: datetime,
    dry_run: bool = False
) -> int:
    """Remove old monitoring data."""
```

Parameters:
- `older_than`: Remove data older than this
- `dry_run`: Only simulate cleanup
Returns:
- Number of records removed

```python
def optimize_database(self) -> None:
    """Optimize database and reclaim space."""
```

### Backup Management

```python
def create_backup(
    self,
    backup_dir: Path,
    tag: Optional[str] = None
) -> Path:
    """Create database backup."""
```

Parameters:
- `backup_dir`: Directory for backup
- `tag`: Optional backup tag
Returns:
- Path to backup file

```python
def restore_backup(
    self,
    backup_path: Path
) -> None:
    """Restore database from backup."""
```

Parameters:
- `backup_path`: Path to backup file
