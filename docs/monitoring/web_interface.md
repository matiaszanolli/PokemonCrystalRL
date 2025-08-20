# Web Interface Guide

The monitoring system provides a web-based interface for real-time monitoring and control of the training process.

## Access

The web interface is available at `http://localhost:<port>` when training is running. The default port is 8080.

## Dashboard

The main dashboard provides an overview of the current training status and key metrics.

### Status Panel

![Status Panel](../images/status_panel.png)

Displays:
- Current training state
- Run ID and duration
- Episode count and progress
- Total steps completed
- Current reward
- Success rate

### Game View

![Game View](../images/game_view.png)

Shows:
- Live game screen
- Current action
- Player position
- Game state information
- Last reward received

Controls:
- Pause/Resume stream
- Change refresh rate
- Toggle annotations
- Save screenshot

### Metrics Panel

![Metrics Panel](../images/metrics_panel.png)

Displays real-time charts for:
- Training metrics (loss, accuracy, reward)
- System metrics (CPU, memory, disk)
- Custom metrics

Features:
- Configurable time window
- Multiple metric overlay
- Export data
- Auto-scaling

### Event Timeline

![Event Timeline](../images/timeline.png)

Shows:
- Training events
- Achievements
- Error events
- System events

Features:
- Event filtering
- Timeline zoom
- Event details
- Export events

## Training Control

### Control Panel

![Control Panel](../images/control_panel.png)

Actions:
- Start training
- Stop training
- Pause/Resume
- Save checkpoint
- Export data

### Configuration

![Configuration](../images/config_panel.png)

Settings:
- Learning parameters
- Training duration
- Save intervals
- Debug options

## Monitoring Views

### Performance View

![Performance View](../images/performance_view.png)

Shows detailed performance metrics:
- Learning curves
- Reward distribution
- Success rate trends
- Moving averages

Features:
- Custom time ranges
- Metric comparison
- Statistical analysis
- Export graphs

### System View

![System View](../images/system_view.png)

Displays system resource usage:
- CPU utilization
- Memory usage
- Disk I/O
- Network traffic

Features:
- Resource alerts
- Usage history
- Process details
- Export data

### Error View

![Error View](../images/error_view.png)

Shows error and warning events:
- Error messages
- Stack traces
- Warning alerts
- Recovery actions

Features:
- Error filtering
- Severity levels
- Error analysis
- Export logs

## History and Analysis

### Training History

![Training History](../images/history_view.png)

Lists all training runs:
- Start/end times
- Total episodes
- Final results
- Configuration

Features:
- Run comparison
- Filter runs
- Export data
- Delete runs

### Episode Analysis

![Episode Analysis](../images/episode_view.png)

Detailed episode information:
- Step sequences
- Reward breakdown
- State transitions
- Action history

Features:
- Episode replay
- Step analysis
- State inspection
- Export episode

### State Explorer

![State Explorer](../images/state_explorer.png)

Inspect game states:
- Map view
- Player position
- Inventory status
- Pokemon details

Features:
- State search
- Time navigation
- State comparison
- Export states

## Data Export

### Export Options

![Export Panel](../images/export_panel.png)

Available formats:
- JSON data
- CSV tables
- SQLite database
- Tensorboard logs

Features:
- Select data types
- Date range filter
- Compression options
- Auto-export

## Settings

### Interface Settings

![Settings Panel](../images/settings.png)

Configure:
- Update intervals
- Chart settings
- Data retention
- UI preferences

### Notification Settings

![Notifications](../images/notifications.png)

Configure alerts for:
- Error events
- Performance issues
- System warnings
- Achievements

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `S` | Start/Stop training |
| `P` | Pause/Resume |
| `R` | Reset view |
| `F` | Toggle fullscreen |
| `M` | Toggle metrics |
| `E` | Toggle events |
| `H` | Show help |

## Browser Support

The web interface is tested with:
- Chrome 80+
- Firefox 75+
- Safari 13+
- Edge 80+

## Troubleshooting

Common issues and solutions:

### Connection Issues

If the web interface is not accessible:
1. Check the server is running
2. Verify the port is correct
3. Check firewall settings
4. Clear browser cache

### Display Problems

If charts or game screen not updating:
1. Check WebSocket connection
2. Reduce update frequency
3. Clear browser cache
4. Reload page

### Performance Issues

If interface becomes slow:
1. Reduce time window
2. Close unused panels
3. Lower update rate
4. Clear old data

## Development

### Custom Panels

The interface can be extended with custom panels:

```javascript
class CustomPanel extends Panel {
    render() {
        // Panel implementation
    }
}

monitor.addPanel(new CustomPanel());
```

### Custom Metrics

Add custom metrics to the interface:

```javascript
monitor.addMetric({
    name: "custom_metric",
    label: "Custom Metric",
    type: "line",
    color: "#ff0000"
});
```

### Custom Events

Register custom event handlers:

```javascript
monitor.on("custom_event", (data) => {
    // Handle event
});
```
