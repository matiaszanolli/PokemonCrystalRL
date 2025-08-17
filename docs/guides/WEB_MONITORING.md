# 🌐 Web Monitoring System

The Pokemon Crystal RL web monitoring system provides real-time training visualization and control through an interactive web interface.

## 🌟 Features

- Real-time game screen streaming
- Training statistics and metrics
- Agent decision monitoring
- System resource tracking
- Performance analytics
- API endpoints for external tools

## 🚀 Quick Start

```bash
# Start trainer with web monitoring
python -m pokemon_crystal_rl.trainer \
    --rom pokemon_crystal.gbc \
    --web \
    --port 8080

# Open in browser
open http://localhost:8080
```

## 📊 Web Interface

### Main Dashboard
The main dashboard provides:
- Live game screen view
- Current game state
- Training statistics
- Agent decisions
- System resources

### Real-time Updates
- Game screen updates (10 FPS)
- Training metrics (every second)
- System stats (every 5 seconds)
- Action history (immediate)

### Interactive Controls
- Start/stop training
- Adjust parameters
- View detailed stats
- Export data

## 🔌 API Endpoints

### Status API
```bash
# Get current status
curl http://localhost:8080/api/status

# Response
{
    "is_training": true,
    "total_steps": 1000,
    "current_episode": 5,
    "reward": 100.5,
    "timestamp": "2025-08-17T23:15:30Z"
}
```

### Stats API
```bash
# Get detailed stats
curl http://localhost:8080/api/stats

# Response
{
    "training_stats": {
        "total_actions": 5000,
        "total_episodes": 10,
        "actions_per_second": 5.5,
        "average_reward": 50.2
    },
    "system_stats": {
        "cpu_percent": 35.2,
        "memory_used": 1024,
        "fps": 10.5
    }
}
```

### Screenshot API
```bash
# Get current screenshot
curl http://localhost:8080/api/screen

# Response: JPEG image data
```

### Text API
```bash
# Get detected text
curl http://localhost:8080/api/text

# Response
{
    "recent_text": [
        {
            "text": "PALLET TOWN",
            "location": "world",
            "confidence": 0.95
        }
    ],
    "text_frequency": {
        "PALLET TOWN": 5,
        "BATTLE": 3
    }
}
```

## ⚙️ Configuration

### Server Options
```python
monitor = UnifiedMonitor(
    host="0.0.0.0",     # Listen on all interfaces
    port=8080,          # Web server port
    debug_mode=False    # Enable debug logging
)
```

### Performance Tuning
```python
monitor = UnifiedMonitor(
    screen_fps=10,       # Screen update rate
    stats_interval=1.0,  # Stats update interval
    cache_size=1000      # Memory cache size
)
```

### Integration
```python
from pokemon_crystal_rl.monitoring import UnifiedMonitor

# Create monitor
monitor = UnifiedMonitor()

# Start server
monitor.start()

# Update data
monitor.update_stats(stats)
monitor.update_screenshot(screen)
monitor.update_text(text_data)

# Cleanup
monitor.stop()
```

## 🔧 Development

### Adding New Features
```python
class CustomMonitor(UnifiedMonitor):
    def __init__(self):
        super().__init__()
        self.custom_data = {}
    
    def add_custom_endpoint(self):
        @self.app.route('/api/custom')
        def custom_endpoint():
            return jsonify(self.custom_data)
```

### WebSocket Events
```python
# Server-side
@monitor.socketio.on('custom_event')
def handle_custom_event(data):
    process_data(data)
    monitor.socketio.emit('response', result)

# Client-side
socket.on('response', function(data) {
    updateUI(data);
});
```

### Error Handling
```python
try:
    monitor.start()
except PortInUseError:
    monitor.port += 1
    monitor.start()
except Exception as e:
    logger.error(f"Monitor error: {e}")
```

## 📊 Data Structure

### Training Stats
```python
stats = {
    'total_actions': 1000,
    'total_episodes': 5,
    'actions_per_second': 2.5,
    'current_reward': 100.5,
    'total_reward': 500.0,
    'success_rate': 0.75
}
```

### System Stats
```python
system_stats = {
    'cpu_percent': 45.2,
    'memory_percent': 65.8,
    'disk_usage': 82.1,
    'gpu_available': False,
    'memory_used': 1024,
    'memory_total': 8192
}
```

### Screenshot Data
```python
screenshot_data = {
    'image': 'base64_encoded_image',
    'timestamp': '2025-08-17T23:15:30Z',
    'dimensions': {
        'width': 160,
        'height': 144
    }
}
```

## 🔍 Troubleshooting

### Common Issues

1. **Connection Refused**
   - Check port availability
   - Verify firewall settings
   - Ensure correct host binding

2. **High Latency**
   - Reduce update frequency
   - Lower screenshot quality
   - Check network bandwidth

3. **Memory Usage**
   - Adjust cache sizes
   - Enable garbage collection
   - Monitor resource usage

4. **Browser Issues**
   - Clear cache/cookies
   - Try different browser
   - Check console errors

## 🚀 Best Practices

1. **Resource Management**
   ```python
   # Cleanup resources
   def __exit__(self):
       self.stop()
       self.cleanup()
   ```

2. **Error Recovery**
   ```python
   @monitor.error_handler
   def handle_error(error):
       logger.error(f"Monitor error: {error}")
       monitor.restart()
   ```

3. **Performance Optimization**
   ```python
   # Batch updates
   if time.time() - last_update > update_interval:
       monitor.batch_update(stats, screen, text)
   ```

4. **Security**
   ```python
   # Enable CORS protection
   monitor = UnifiedMonitor(
       cors_origins=['http://localhost:3000'],
       require_auth=True
   )
   ```

## 🔄 Future Improvements

1. **Enhanced Features**
   - Advanced visualization
   - Interactive training control
   - Custom metric tracking
   - Data export tools

2. **Performance**
   - WebSocket optimizations
   - Compressed data transfer
   - Lazy loading
   - Client-side caching

3. **Integration**
   - External tool APIs
   - Custom dashboards
   - Notification system
   - Mobile support

4. **Analytics**
   - Advanced metrics
   - Performance insights
   - Training recommendations
   - Automated reporting
