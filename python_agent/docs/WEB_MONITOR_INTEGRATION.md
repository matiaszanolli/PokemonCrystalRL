# Pokemon RL Web Monitor Integration Guide

## Overview

The Pokemon Crystal RL training system now features a fully integrated web monitoring solution that provides real-time visualization of training progress, live game screenshots, and comprehensive analytics through a modern web interface.

## Architecture

### Components

1. **UnifiedPokemonTrainer**: Core training system with PyBoy emulation
2. **PokemonRLWebMonitor**: WebSocket-based monitoring dashboard  
3. **TrainerWebMonitorBridge**: Integration layer connecting trainer to monitor
4. **Web Interface**: Real-time HTML dashboard with live updates

### Data Flow

```
PyBoy Game → Trainer Screenshots → Bridge → Web Monitor → WebSocket → Browser
```

## Quick Start

### Basic Integration

```python
from monitoring.trainer_monitor_bridge import create_integrated_monitoring_system
from trainer.trainer import UnifiedPokemonTrainer
from trainer.config import TrainingConfig, TrainingMode

# Create trainer configuration
config = TrainingConfig(
    rom_path="path/to/pokemon_crystal.gbc",
    mode=TrainingMode.FAST_MONITORED,
    capture_screens=True,  # Required for web monitoring
    enable_web=False       # Disable trainer's built-in web server
)

# Create trainer
trainer = UnifiedPokemonTrainer(config)

# Create integrated monitoring system
web_monitor, bridge, monitor_thread = create_integrated_monitoring_system(
    trainer, 
    host='0.0.0.0', 
    port=5000
)

# Start the bridge
bridge.start_bridge()

# Start training with live monitoring
trainer.start_training()
```

### Advanced Configuration

```python
# Custom bridge configuration
bridge = TrainerWebMonitorBridge(trainer, web_monitor)
bridge.screenshot_update_interval = 0.3  # 3.3 FPS screenshots
bridge.stats_update_interval = 1.0       # Update stats every second
bridge.bridge_fps = 15                   # Higher bridge update rate

bridge.start_bridge()
```

## Web Interface Features

### Real-time Monitoring
- **Live Game Screen**: 3x scaled Pokemon game display with pixelated rendering
- **WebSocket Streaming**: Real-time updates at 2 FPS (configurable)
- **Automatic Reconnection**: Robust connection handling

### Training Analytics
- **Game Statistics**: Player position, money, badges, Pokemon party
- **Training Metrics**: Episodes, steps, decisions made, visual analyses
- **Action History**: Recent 20 actions with timestamps and reasoning
- **Agent Decisions**: LLM decision logs with confidence scores

### Interactive Controls
- **Start/Stop Monitoring**: Manual control of monitoring state
- **Real-time Status**: Connection and monitoring status indicators
- **Responsive Design**: Works on desktop and mobile devices

## API Reference

### TrainerWebMonitorBridge

#### Constructor
```python
TrainerWebMonitorBridge(trainer, web_monitor)
```

#### Methods
- `start_bridge()`: Start the bridge thread
- `stop_bridge()`: Stop the bridge thread
- `get_bridge_stats()`: Get performance statistics

#### Configuration Properties
- `screenshot_update_interval`: Time between screenshot updates (seconds)
- `stats_update_interval`: Time between stats updates (seconds)
- `bridge_fps`: Bridge thread update rate

### PokemonRLWebMonitor

#### Key Methods
- `update_screenshot(screenshot: np.ndarray)`: Send screenshot to clients
- `update_action(action: str, reasoning: str)`: Log training action
- `update_decision(decision_data: Dict)`: Log agent decision
- `start_monitoring()`: Begin monitoring loop
- `stop_monitoring()`: End monitoring loop

#### Web Endpoints
- `/`: Main dashboard
- `/api/status`: Monitoring status and current stats
- `/api/history`: Historical training data
- WebSocket events: `screenshot`, `stats_update`, `new_action`, `new_decision`

## Configuration Options

### Trainer Configuration
```python
config = TrainingConfig(
    # Required for web monitoring
    capture_screens=True,
    capture_fps=10,           # Screenshot capture rate
    
    # Optional: disable trainer's built-in web server
    enable_web=False,
    
    # Screen processing
    screen_resize=(480, 432), # 3x Game Boy resolution
)
```

### Bridge Configuration
```python
bridge.screenshot_update_interval = 0.5  # 2 FPS to web clients
bridge.stats_update_interval = 2.0       # Stats every 2 seconds  
bridge.bridge_fps = 10                   # Internal update rate
```

### Web Monitor Configuration
```python
web_monitor.run(
    host='0.0.0.0',    # Bind to all interfaces
    port=5000,         # Web server port
    debug=False        # Production mode
)
```

## Performance Tuning

### Screenshot Performance
- **Bridge FPS**: Higher = smoother but more CPU usage
- **Screenshot Interval**: Lower = higher web FPS but more bandwidth
- **Screenshot Quality**: Adjustable JPEG quality in bridge conversion

### Memory Management
- Bridge automatically manages screenshot queues
- Old screenshots are discarded when queues are full
- Statistics are truncated to prevent memory leaks

### Network Optimization
- WebSocket compression enabled
- Base64 image encoding with JPEG compression
- Configurable update intervals to manage bandwidth

## Troubleshooting

### Common Issues

#### Blank Screen in Web Interface
- **Cause**: Bridge not started or trainer not capturing screens
- **Solution**: Ensure `bridge.start_bridge()` is called and `capture_screens=True`

#### High CPU Usage
- **Cause**: Too high bridge FPS or screenshot update rate
- **Solution**: Reduce `bridge_fps` and increase `screenshot_update_interval`

#### WebSocket Connection Errors
- **Cause**: Firewall blocking WebSocket connections
- **Solution**: Check firewall settings and ensure port is accessible

#### Template Not Found Errors
- **Cause**: Missing HTML templates
- **Solution**: Run `create_dashboard_templates()` before starting web monitor

### Debug Mode
```python
# Enable verbose bridge logging
bridge._debug_mode = True

# Enable Flask debug mode
web_monitor.run(debug=True)
```

### Health Checks
```python
# Check bridge status
stats = bridge.get_bridge_stats()
print(f"Screenshots transferred: {stats['screenshots_transferred']}")
print(f"Errors: {stats['total_errors']}")

# Check web monitor status  
print(f"Monitoring active: {web_monitor.is_monitoring}")
print(f"Connected clients: {len(web_monitor.socketio.server.manager.rooms)}")
```

## Examples

### Basic Training with Web Monitor
```python
from monitoring.trainer_monitor_bridge import create_integrated_monitoring_system

# Create and start integrated system
web_monitor, bridge, thread = create_integrated_monitoring_system(trainer)
bridge.start_bridge()

print(f"Web interface: http://localhost:5000")
trainer.start_training()
```

### Custom Integration
```python
# Manual setup for more control
web_monitor = PokemonRLWebMonitor()
bridge = TrainerWebMonitorBridge(trainer, web_monitor)

# Start web monitor in background
import threading
monitor_thread = threading.Thread(
    target=lambda: web_monitor.run(host='0.0.0.0', port=8080),
    daemon=True
)
monitor_thread.start()

# Configure and start bridge
bridge.screenshot_update_interval = 0.25  # 4 FPS
bridge.start_bridge()

# Start training
trainer.start_training()
```

## Integration with Existing Systems

### With Existing Web Servers
```python
# Use different port to avoid conflicts
web_monitor, bridge, thread = create_integrated_monitoring_system(
    trainer, 
    port=5001  # Avoid port conflicts
)
```

### With Cloud Deployments
```python
# Bind to all interfaces for cloud access
web_monitor.run(
    host='0.0.0.0',  # Accept connections from any IP
    port=int(os.environ.get('PORT', 5000))  # Use cloud-assigned port
)
```

### With Docker
```dockerfile
# Dockerfile
EXPOSE 5000
CMD ["python", "-m", "monitoring.web_monitor"]
```

## Security Considerations

### Network Security
- Web monitor binds to all interfaces by default
- Consider using reverse proxy (nginx) for production
- Implement authentication if needed

### Data Privacy
- Screenshots may contain game state information
- Consider data retention policies for training logs
- Monitor bandwidth usage in production

## Future Enhancements

### Planned Features
- [ ] Historical screenshot playback
- [ ] Training session comparison
- [ ] Performance profiling integration
- [ ] Multi-trainer monitoring
- [ ] Export training data

### Extension Points
- Custom WebSocket events
- Additional web endpoints
- Custom dashboard components
- Integration with external monitoring systems

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review debug logs with `debug=True`
3. Check GitHub issues for known problems
4. Create new issue with reproduction steps

## Changelog

### v1.0 - Initial Release
- Real-time screenshot streaming
- WebSocket-based updates
- Training statistics integration
- Responsive web interface
- TrainerWebMonitorBridge integration layer
