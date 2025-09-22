# Unified Web Dashboard for Pokemon Crystal RL

A complete, consolidated web dashboard system that replaces multiple fragmented implementations with a single, well-documented, and maintainable solution.

## 🎯 Overview

This unified dashboard consolidates functionality from multiple previous implementations:
- `core/web_monitor/` - Original web monitoring system
- `monitoring/web/` - Structured API system
- `static/templates/` - Various dashboard templates
- `monitoring/static/` - Modern dashboard UI

## ✨ Features

### 🔥 Core Capabilities
- **Real-time Training Monitoring** - Live statistics, rewards, and performance metrics
- **Game State Visualization** - Current map, position, money, badges, and Pokemon party
- **Memory Debugging** - Live memory address inspection with real-time values
- **LLM Decision Tracking** - Recent AI decisions with reasoning and confidence scores
- **System Status** - Training status, WebSocket connections, and health monitoring

### 🚀 Technical Features
- **Unified API** - Clean, documented REST endpoints with proper data models
- **WebSocket Streaming** - Real-time screen capture and live updates
- **Modern UI** - Responsive design with dark theme and smooth animations
- **Error Handling** - Comprehensive error recovery and user feedback
- **Performance Monitoring** - Built-in performance tracking and optimization

## 🏗️ Architecture

```
web_dashboard/
├── api/                      # API Layer
│   ├── __init__.py          # Package exports
│   ├── models.py            # Data models and schemas
│   └── endpoints.py         # API endpoint implementations
├── static/                   # Frontend Assets
│   ├── dashboard.html       # Main dashboard template
│   ├── styles.css          # Comprehensive CSS styles
│   └── app.js              # Frontend JavaScript application
├── server.py               # HTTP server implementation
├── websocket_handler.py    # WebSocket real-time updates
├── __init__.py            # Package initialization
└── README.md              # This documentation
```

## 🚦 Quick Start

### Basic Usage

```python
from web_dashboard import UnifiedWebServer

# Create and start web server
server = UnifiedWebServer(trainer=your_trainer)
server.start()

# Dashboard available at http://localhost:8080
# WebSocket streaming at ws://localhost:8081
```

### Factory Function

```python
from web_dashboard import create_web_server

# Use factory function for easier setup
server = create_web_server(
    trainer=your_trainer,
    host='localhost',
    http_port=8080,
    ws_port=8081
)
server.start()
```

### Integration with Unified Trainer

```python
from training.unified_pokemon_trainer import UnifiedPokemonTrainer
from web_dashboard import create_web_server

# Create trainer
trainer = UnifiedPokemonTrainer(...)

# Add web monitoring
web_server = create_web_server(trainer)
web_server.start()

# Start training with web monitoring
trainer.train(max_actions=1000)
```

## 📡 API Endpoints

All endpoints return JSON with the following structure:
```json
{
    "success": true,
    "data": { ... },
    "timestamp": 1234567890.123
}
```

### Core Endpoints

| Endpoint | Description | Response |
|----------|-------------|----------|
| `GET /` | Main dashboard HTML | Dashboard interface |
| `GET /api/dashboard` | Complete dashboard data | All data in one call |
| `GET /api/game_state` | Current game state | Map, position, money, badges |
| `GET /api/training_stats` | Training metrics | Actions, rewards, performance |
| `GET /api/memory_debug` | Memory debugging | Live memory addresses |
| `GET /api/llm_decisions` | Recent LLM decisions | AI reasoning and actions |
| `GET /api/system_status` | System health | Training status, connections |
| `GET /api/screen` | Current game screen | PNG image |
| `GET /health` | Health check | Server status |

### Data Models

#### GameStateModel
```python
{
    "current_map": 4,
    "player_position": {"x": 3, "y": 0},
    "money": 1500,
    "badges_earned": 2,
    "party_count": 3,
    "player_level": 25,
    "hp_current": 45,
    "hp_max": 50,
    "in_battle": false,
    "facing_direction": 1,
    "timestamp": 1234567890.123
}
```

#### TrainingStatsModel
```python
{
    "total_actions": 1500,
    "actions_per_second": 1.7,
    "llm_decisions": 150,
    "total_reward": -245.5,
    "session_duration": 882.3,
    "success_rate": 0.85,
    "exploration_rate": 0.65,
    "recent_rewards": [-0.2, -0.2, 10.0, -0.2],
    "timestamp": 1234567890.123
}
```

#### MemoryDebugModel
```python
{
    "memory_addresses": {
        "PARTY_COUNT": 2,
        "PLAYER_MAP": 4,
        "PLAYER_X": 3,
        "PLAYER_Y": 0,
        "MONEY": 1500,
        "BADGES": 2
    },
    "memory_read_success": true,
    "pyboy_available": true,
    "cache_info": {
        "cache_age_seconds": 0.001,
        "cache_duration": 0.1
    },
    "timestamp": 1234567890.123
}
```

## 🌐 WebSocket Events

Connect to `ws://localhost:8081` for real-time updates.

### Client → Server Messages
```javascript
// Request current screen
{"type": "request_screen"}

// Request current stats
{"type": "request_stats"}

// Ping for connection testing
{"type": "ping"}
```

### Server → Client Messages
```javascript
// Screen update
{
    "type": "screen_update",
    "data": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==",
    "timestamp": 1234567890.123
}

// Stats update
{
    "type": "stats_update",
    "data": { ... },
    "timestamp": 1234567890.123
}

// Connection established
{
    "type": "connection_established",
    "timestamp": 1234567890.123
}

// Pong response
{
    "type": "pong",
    "timestamp": 1234567890.123
}
```

## 🎨 Frontend Features

### Dashboard Sections

1. **Training Statistics** - Actions, rewards, LLM usage, performance metrics
2. **Live Game Screen** - Real-time screen capture with WebSocket streaming
3. **Game State** - Current map, position, money, badges from memory reading
4. **Recent LLM Decisions** - AI decision history with reasoning and confidence
5. **Live Memory Debug** - Real-time memory address inspection
6. **System Status** - Training status, connections, API health

### UI Features

- **Responsive Design** - Works on desktop, tablet, and mobile
- **Dark Theme** - Easy on the eyes for long monitoring sessions
- **Real-time Updates** - Live data without page refresh
- **Error Handling** - Clear error messages and recovery
- **Performance Monitor** - Built-in performance tracking
- **Connection Status** - Visual indicators for API and WebSocket status

### Browser Support

- Chrome 80+
- Firefox 75+
- Safari 13+
- Edge 80+

## 🔧 Configuration

### Server Configuration

```python
from web_dashboard import UnifiedWebServer

server = UnifiedWebServer(
    trainer=trainer,
    host='0.0.0.0',        # Bind to all interfaces
    http_port=8080,        # HTTP server port
    ws_port=8081           # WebSocket server port
)
```

### Frontend Configuration

The frontend automatically detects the server configuration. For custom setups, modify `app.js`:

```javascript
// Configuration in app.js
this.config = {
    apiBaseUrl: '',          // API base URL (same origin)
    wsUrl: 'ws://localhost:8081',  // WebSocket URL
    updateIntervals: {
        api: 2000,           // API polling interval (ms)
        screen: 100,         // Screen update interval (ms)
        stats: 1000          // Stats update interval (ms)
    },
    maxRetries: 5,           // Maximum retry attempts
    retryDelay: 1000         // Retry delay (ms)
};
```

## 🔍 Troubleshooting

### Common Issues

**Dashboard shows "Disconnected"**
- Check if the trainer is running with `--enable-web`
- Verify ports 8080 and 8081 are not blocked
- Check browser console for error messages

**Memory debug shows "Loading memory data..."**
- Ensure PyBoy instance is properly initialized
- Check that memory reader is available
- Verify save state is loaded correctly

**Game screen is blank or grey**
- Check WebSocket connection (port 8081)
- Ensure screen capture is enabled
- Verify PyBoy screen buffer is accessible

**LLM decisions not showing**
- Check that LLM interval is set (e.g., `--llm-interval 20`)
- Verify trainer has `llm_decisions` attribute
- Ensure LLM agent is properly initialized

### Debug Mode

Enable debug logging for detailed troubleshooting:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# This will show detailed API and WebSocket logs
server = UnifiedWebServer(trainer=trainer)
server.start()
```

### Browser Developer Tools

Use browser developer tools to diagnose frontend issues:

1. **Console Tab** - JavaScript errors and WebSocket messages
2. **Network Tab** - API request/response inspection
3. **WebSocket Tab** - Real-time WebSocket message monitoring

## 🧪 Testing

### Manual Testing

1. **Start Training Session**
   ```bash
   python main.py roms/pokemon_crystal.gbc --save-state roms/pokemon_crystal.gbc.state --max-actions 500 --enable-web --llm-interval 20
   ```

2. **Open Dashboard**
   - Navigate to http://localhost:8080
   - Verify all sections load with data
   - Check real-time updates

3. **Test API Endpoints**
   ```bash
   curl http://localhost:8080/api/dashboard | python -m json.tool
   curl http://localhost:8080/api/memory_debug | python -m json.tool
   curl http://localhost:8080/health
   ```

### Integration Testing

The dashboard integrates with the unified trainer system. Test with:

```python
from training.unified_pokemon_trainer import UnifiedPokemonTrainer
from web_dashboard import create_web_server

# Create trainer with web monitoring
trainer = UnifiedPokemonTrainer(...)
web_server = create_web_server(trainer)
web_server.start()

# Run training and monitor dashboard
trainer.train(max_actions=100)
```

## 📈 Performance

### Optimization Features

- **Efficient Polling** - Smart update intervals based on data type
- **Caching** - Memory reading cache to reduce PyBoy access
- **Compression** - Gzipped responses for large data
- **Connection Pooling** - Optimized WebSocket management
- **Lazy Loading** - Load data only when visible

### Performance Monitoring

The dashboard includes built-in performance monitoring:

- **Update Rate** - Real-time updates per second
- **Latency** - Average API response time
- **Memory Usage** - Frontend memory consumption
- **Connection Status** - WebSocket connection health

### Resource Usage

Typical resource usage during training:

- **CPU**: ~2-5% additional overhead
- **Memory**: ~50-100MB for web server
- **Network**: ~1-5KB/s for dashboard updates
- **Disk**: Minimal (logging only)

## 🔄 Migration Guide

### From Legacy Systems

If migrating from existing dashboard implementations:

1. **Stop Old Web Servers**
   ```python
   # Disable old web monitoring
   trainer.web_monitor = None
   ```

2. **Update Imports**
   ```python
   # Old
   from core.web_monitor import WebMonitor

   # New
   from web_dashboard import create_web_server
   ```

3. **Update Configuration**
   ```python
   # Old
   web_monitor = WebMonitor(trainer)
   web_monitor.start()

   # New
   web_server = create_web_server(trainer)
   web_server.start()
   ```

### Backward Compatibility

The unified dashboard maintains compatibility with existing trainer interfaces:

- Reads from `statistics_tracker` for training stats
- Uses `emulation_manager` for PyBoy access
- Supports both old and new memory reader patterns
- Works with existing LLM decision storage

## 🤝 Contributing

### Development Setup

1. **Install Dependencies**
   ```bash
   pip install websockets pillow
   ```

2. **Run in Development Mode**
   ```python
   server = UnifiedWebServer(trainer, host='localhost')
   server.start()
   ```

3. **Frontend Development**
   - Edit `static/dashboard.html`, `static/styles.css`, `static/app.js`
   - Changes take effect immediately (no build step required)
   - Use browser developer tools for debugging

### Code Style

- **Python**: Follow PEP 8, use type hints
- **JavaScript**: Use ES6+, modern async/await patterns
- **CSS**: Use CSS custom properties, mobile-first responsive design
- **HTML**: Semantic HTML5, accessibility best practices

### Testing

- **API**: Test all endpoints with various trainer states
- **WebSocket**: Test connection handling and real-time updates
- **Frontend**: Test on multiple browsers and devices
- **Integration**: Test with actual training sessions

## 📄 License

This unified web dashboard is part of the Pokemon Crystal RL project and follows the same licensing terms as the main project.

## 🙏 Acknowledgments

This unified dashboard consolidates and improves upon multiple previous implementations:

- Original `core/web_monitor` system
- Structured `monitoring/web` API design
- Modern UI elements from `monitoring/static`
- Best practices from various dashboard templates

Special thanks to all contributors who built the foundation systems that this unified implementation builds upon.