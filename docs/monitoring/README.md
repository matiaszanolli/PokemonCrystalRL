# Pokemon Crystal RL Monitoring System

The monitoring system provides real-time monitoring, metrics collection, and visualization for the Pokemon Crystal RL training process. It combines database storage, web-based visualization, and comprehensive event tracking to help understand and optimize the training process.

## Features

- **Real-time Training Monitoring**
  - Game screen streaming
  - Live metric updates
  - Training state visualization
  - Action history tracking

- **Metric Collection**
  - Performance metrics (accuracy, reward, loss)
  - System metrics (CPU, memory, disk usage)
  - Custom metric support
  - Statistical aggregation

- **Data Storage**
  - SQLite-based persistent storage
  - Efficient schema design
  - Automatic data cleanup
  - Training run history

- **Error Handling**
  - Categorized error tracking
  - Severity-based handling
  - Automatic recovery
  - Error event logging

- **Web Interface**
  - Real-time dashboard
  - Interactive visualizations
  - Training control panel
  - Historical data views

## Architecture

The monitoring system consists of several key components:

```
+----------------+      +--------------+      +------------------+
|  Web Monitor   |<---->| Data Storage |<---->| Metric Collector |
+----------------+      +--------------+      +------------------+
        ^                      ^                      ^
        |                      |                      |
        v                      v                      v
+----------------+      +--------------+      +------------------+
|  Web Server    |      | Event Logger |      | Error Handler    |
+----------------+      +--------------+      +------------------+
```

- **Web Monitor**: Central coordination component
- **Data Storage**: SQLite-based persistent storage
- **Metric Collector**: Gathers performance and system metrics
- **Web Server**: Provides HTTP API and web interface
- **Event Logger**: Records training events and actions
- **Error Handler**: Manages error detection and recovery

## Installation

1. The monitoring system is included in the main package:

```bash
pip install -e .
```

2. Initialize the monitoring database:

```python
from pokemon_crystal_rl.monitoring import init_monitoring
init_monitoring()
```

## Configuration

The monitoring system can be configured through a configuration file or programmatically. See [Configuration Guide](configuration.md) for details.

Example configuration:

```python
config = MonitorConfig(
    db_path="training.db",
    data_dir="monitor_data",
    web_port=8080,
    update_interval=1.0,
    snapshot_interval=300,
    max_snapshots=100,
    debug=False
)
```

## Usage

1. Initialize the monitoring system:

```python
from pokemon_crystal_rl.monitoring import WebMonitor

monitor = WebMonitor(config)
```

2. Start training monitoring:

```python
monitor.start_training(config={
    "learning_rate": 0.001,
    "batch_size": 64
})
```

3. Record metrics during training:

```python
monitor.update_metrics({
    "loss": 0.5,
    "accuracy": 0.85,
    "reward": 1.0
})
```

4. Record game states:

```python
monitor.update_game_state({
    "map_id": 1,
    "player_x": 10,
    "player_y": 20,
    "inventory": ["POTION", "POKEBALL"]
})
```

5. Handle training events:

```python
monitor.record_event(
    event_type="achievement",
    event_data={"type": "badge", "value": 1}
)
```

6. End training:

```python
monitor.stop_training(final_reward=100.0)
```

## Web Interface

The monitoring system provides a web interface at `http://localhost:<port>` with:

- Real-time training visualization
- Metric charts and graphs
- Event timeline
- System resource monitoring
- Training run history

See [Web Interface Guide](web_interface.md) for more details.

## API Reference

The monitoring system provides a comprehensive Python API and HTTP API. See:

- [Python API Reference](api_reference.md)
- [HTTP API Reference](http_api.md)

## Database Schema

The monitoring data is stored in SQLite with the following schema:

- [Database Schema Reference](database_schema.md)

## Contributing

See [Contributing Guide](contributing.md) for development setup and guidelines.

## Testing

Run the test suite:

```bash
pytest tests/monitoring/
```

Run specific test categories:

```bash
pytest tests/monitoring/ -m "integration"  # Integration tests
pytest tests/monitoring/ -m "web"          # Web interface tests
pytest tests/monitoring/ -m "database"     # Database tests
```
