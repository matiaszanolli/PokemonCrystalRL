# Configuration Guide

The monitoring system can be configured through a configuration file or programmatically using the `MonitorConfig` class.

## Quick Start

```python
from pokemon_crystal_rl.monitoring import MonitorConfig

config = MonitorConfig(
    db_path="training.db",
    data_dir="monitor_data",
    web_port=8080,
    update_interval=1.0
)
```

## Configuration File

Create a configuration file `monitor_config.json`:

```json
{
    "version": 1,
    "db_path": "training.db",
    "data_dir": "monitor_data",
    "static_dir": "static",
    "web_port": 8080,
    "update_interval": 1.0,
    "snapshot_interval": 300.0,
    "max_events": 1000,
    "max_snapshots": 100,
    "debug": false,
    "logging": {
        "level": "INFO",
        "file": "monitor.log"
    }
}
```

Load configuration file:

```python
from pokemon_crystal_rl.monitoring import MonitorConfig

config = MonitorConfig.from_file("monitor_config.json")
```

## Configuration Options

### Core Settings

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `db_path` | str | "monitoring.db" | Path to SQLite database file |
| `data_dir` | str | "monitor_data" | Directory for monitoring data |
| `static_dir` | str | "static" | Directory for static web files |
| `web_port` | int | 8080 | Port for web interface |
| `debug` | bool | False | Enable debug mode |

### Update Intervals

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `update_interval` | float | 1.0 | Seconds between metric updates |
| `snapshot_interval` | float | 300.0 | Seconds between state snapshots |
| `cleanup_interval` | float | 3600.0 | Seconds between data cleanup |

### Capacity Limits

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `max_events` | int | 1000 | Maximum events in memory |
| `max_snapshots` | int | 100 | Maximum state snapshots |
| `max_metrics` | int | 10000 | Maximum metrics per run |

### Web Interface

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `web_host` | str | "localhost" | Web interface host |
| `web_workers` | int | 4 | Number of web workers |
| `web_timeout` | int | 30 | Request timeout seconds |
| `enable_ssl` | bool | False | Enable HTTPS |

### Logging

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `log_level` | str | "INFO" | Logging level |
| `log_file` | str | None | Log file path |
| `log_format` | str | None | Log message format |
| `log_rotation` | str | "1 MB" | Log rotation size |

### Data Storage

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `retention_days` | int | 30 | Days to keep old data |
| `backup_interval` | str | "1d" | Backup frequency |
| `compression` | bool | True | Compress old data |
| `auto_vacuum` | bool | True | Auto-optimize database |

### Error Handling

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `error_retries` | int | 3 | Max error retries |
| `retry_delay` | float | 1.0 | Seconds between retries |
| `error_hooks` | list | [] | Error callback URLs |
| `ignore_errors` | list | [] | Errors to ignore |

### Video Streaming

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `stream_fps` | int | 30 | Stream frame rate |
| `stream_quality` | int | 80 | JPEG quality (0-100) |
| `stream_size` | tuple | None | Resize dimensions |
| `enable_audio` | bool | False | Stream audio |

### Metrics

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `metric_aggregation` | str | "mean" | Aggregation method |
| `metric_window` | int | 100 | Rolling window size |
| `custom_metrics` | dict | {} | Custom metric config |
| `system_metrics` | bool | True | Track system metrics |

## Environment Variables

Configuration can be overridden with environment variables:

```bash
# Core settings
MONITOR_DB_PATH=training.db
MONITOR_DATA_DIR=monitor_data
MONITOR_WEB_PORT=8080

# Intervals
MONITOR_UPDATE_INTERVAL=1.0
MONITOR_SNAPSHOT_INTERVAL=300.0

# Limits
MONITOR_MAX_EVENTS=1000
MONITOR_MAX_SNAPSHOTS=100

# Logging
MONITOR_LOG_LEVEL=INFO
MONITOR_LOG_FILE=monitor.log
```

## Configuration Hierarchy

Settings are loaded in the following order (later overrides earlier):

1. Default values
2. Configuration file
3. Environment variables
4. Constructor arguments

## Configuration Examples

### Development Setup

```python
config = MonitorConfig(
    db_path=":memory:",  # In-memory database
    debug=True,
    update_interval=0.1,
    max_events=100,
    log_level="DEBUG"
)
```

### Production Setup

```python
config = MonitorConfig(
    db_path="/data/training.db",
    data_dir="/data/monitor",
    web_port=443,
    enable_ssl=True,
    web_workers=8,
    retention_days=90,
    backup_interval="6h",
    error_hooks=["http://alert.example.com/hook"]
)
```

### High Performance Setup

```python
config = MonitorConfig(
    update_interval=0.05,
    snapshot_interval=60.0,
    stream_fps=60,
    stream_quality=60,
    metric_window=1000,
    web_workers=16
)
```

### Minimal Resource Setup

```python
config = MonitorConfig(
    update_interval=5.0,
    snapshot_interval=900.0,
    max_events=100,
    max_snapshots=10,
    stream_fps=10,
    metric_window=10
)
```

## Configuration Migration

When upgrading, use the migration utility:

```python
from pokemon_crystal_rl.monitoring.migrations import ConfigMigrator

# Load old config
with open("old_config.json") as f:
    old_config = json.load(f)

# Migrate to new format
new_config = ConfigMigrator.migrate_config(old_config)

# Save updated config
with open("new_config.json", "w") as f:
    json.dump(new_config, f, indent=2)
```

## Configuration Validation

The `MonitorConfig` class validates settings:

```python
try:
    config = MonitorConfig(web_port=-80)
except ValueError as e:
    print(f"Invalid config: {e}")
```

## Best Practices

1. Use configuration files for permanent settings
2. Use environment variables for deployment-specific settings
3. Use constructor arguments for runtime settings
4. Enable debug mode during development
5. Use SSL in production
6. Configure appropriate retention periods
7. Set up error notifications
8. Monitor resource usage
9. Back up configuration files
10. Document custom settings
