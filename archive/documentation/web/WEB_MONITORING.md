# Pokemon Crystal RL Web Monitoring System

The web monitoring system provides a real-time dashboard for visualizing and monitoring Pokemon Crystal RL training sessions. It features live game screen streaming, comprehensive statistics tracking, action history, and agent decision logs.

## Features

### 🎮 Live Game Screen
- Real-time streaming of the game screen at 3x scale for better visibility  
- Pixelated rendering to preserve authentic Game Boy aesthetics
- Updates multiple times per second during training

### 📊 Game Statistics
- **Player Info**: Current position (X, Y, Map), money, and badge count
- **Pokemon Party**: Live Pokemon status with HP bars, levels, and health indicators
- **Training Metrics**: Episode count, total steps, decisions made, and visual analyses

### 🎯 Action & Decision Tracking
- **Recent Actions**: Real-time log of button presses and controls
- **Agent Decisions**: Strategic decisions made by the LLM agent with reasoning and confidence scores
- **Action Frequency**: Visual breakdown of most common actions

### 🌐 Web Interface
- Modern, responsive dark theme optimized for long monitoring sessions
- Real-time WebSocket updates (no page refresh needed)  
- Start/stop monitoring controls
- Historical data access via REST API

## Quick Start

### 1. Test the Web Monitor (Recommended First Step)

```bash
# Run the test with mock data to verify everything works
python test_web_monitor.py
```

This will:
- Create the dashboard templates
- Start a web server on http://127.0.0.1:5000
- Generate mock Pokemon game data (screenshots, stats, actions)
- Allow you to test the web interface without needing a ROM

### 2. Full Training with Web Monitoring

```bash
# Run training with integrated web monitoring
python web_enhanced_training.py --rom pokecrystal.gbc --episodes 50
```

This will:
- Start the training session
- Launch the web dashboard on http://127.0.0.1:5000  
- Provide real-time monitoring during actual gameplay
- Hook into the training loop to capture all data

### 3. Web-Only Mode (Monitor Existing Data)

```bash
# Start web server to view historical data only
python web_enhanced_training.py --web-only --port 5000
```

## Web Dashboard Usage

1. **Open Browser**: Navigate to http://127.0.0.1:5000
2. **Start Monitoring**: Click the "Start Monitor" button to begin real-time updates
3. **View Live Data**: 
   - Game screen updates in real-time
   - Statistics refresh every 2 seconds
   - Actions and decisions appear immediately
4. **Stop Monitoring**: Click "Stop Monitor" to pause updates (data still collected in background)

## Dashboard Sections

### Main Panel

#### Game Screen
- Shows the current game screen at 3x magnification
- Uses pixelated rendering to maintain Game Boy aesthetic  
- Updates timestamp shows last refresh time
- Visual indicators when screen is updating

#### Game Statistics
- **Player Info**: Location coordinates, money, badge progress
- **Pokemon Party**: Each Pokemon shown with species ID, level, HP bar
- **Training Stats**: Real-time training progress metrics

### Side Panel

#### Controls
- **Start Monitor**: Begin real-time monitoring
- **Stop Monitor**: Pause monitoring updates

#### Recent Actions  
- Scrolling list of last 20 actions taken
- Shows button name, timestamp, and reasoning (if available)
- Color-coded by action type

#### Agent Decisions
- Strategic decisions made by the LLM agent
- Includes decision text, reasoning excerpt, and confidence score
- Shows timestamp and decision context

## API Endpoints

The web server provides REST API access to monitoring data:

### GET /api/status
Returns current monitoring status and latest statistics
```json
{
  "monitoring": true,
  "training_active": true, 
  "current_stats": { ... },
  "timestamp": "2024-01-01T12:00:00"
}
```

### GET /api/history  
Returns historical training data from the SQLite database
```json
{
  "game_states": [ ... ],
  "decisions": [ ... ],
  "performance_metrics": [ ... ]
}
```

### GET /api/training_reports
Returns available training report files
```json
[
  {
    "filename": "training_report_20240101_120000.json",
    "timestamp": "2024-01-01T12:00:00", 
    "episodes": 100,
    "total_steps": 15000,
    "duration": 1800
  }
]
```

## WebSocket Events

Real-time updates use WebSocket connections:

### Client → Server
- `start_monitoring`: Begin monitoring
- `stop_monitoring`: Stop monitoring  
- `request_screenshot`: Request current screenshot

### Server → Client
- `status`: Monitoring status updates
- `screenshot`: New screenshot data (base64 encoded)
- `stats_update`: Latest game statistics
- `new_action`: New action taken
- `new_decision`: New agent decision
- `action_update`: Action frequency data

## Configuration

### Command Line Options

```bash
python web_enhanced_training.py --help
```

- `--rom`: Path to Pokemon Crystal ROM file (default: pokecrystal.gbc)
- `--episodes`: Number of training episodes (default: 100)  
- `--host`: Web server host address (default: 127.0.0.1)
- `--port`: Web server port (default: 5000)
- `--web-only`: Start web server only, no training

### Custom Integration

To integrate the web monitor into custom training scripts:

```python
from web_monitor import PokemonRLWebMonitor

# Create monitor instance
monitor = PokemonRLWebMonitor(training_session=your_session)

# Start web server in background thread  
monitor.start_web_server()

# In your training loop:
monitor.update_screenshot(screenshot_array)
monitor.update_action(action_name, reasoning)
monitor.update_decision(decision_data)
monitor.add_performance_metric('reward', reward_value)
```

## Technical Details

### Architecture
- **Backend**: Flask web server with WebSocket support via Flask-SocketIO
- **Frontend**: Vanilla HTML/CSS/JavaScript with Socket.IO client
- **Data Flow**: WebSocket for real-time updates, REST API for historical data
- **Database**: SQLite for persistent storage of training history

### Performance  
- Screenshot encoding: PNG with base64 for web transmission
- Update frequency: Screenshots at 2-3 FPS, stats at 0.5 Hz
- Memory management: Automatic cleanup of old data to prevent memory leaks
- Multi-threading: Web server runs independently of training loop

### Browser Compatibility
- Modern browsers with WebSocket support (Chrome, Firefox, Safari, Edge)
- Mobile-responsive design works on tablets and phones
- No additional plugins required

## Troubleshooting

### Web Server Won't Start
- Check if port 5000 is already in use: `lsof -i :5000`
- Try different port: `--port 5001`
- Check firewall settings for local connections

### No Data Updates
- Verify "Start Monitor" button was clicked in web interface
- Check browser console for JavaScript errors  
- Ensure training session is actively running
- Check terminal output for connection errors

### Screenshots Not Updating
- Verify training environment provides valid screenshots
- Check screenshot array format (numpy RGB format expected)
- Monitor terminal for image encoding errors

### Database Errors
- Check SQLite database file permissions in `outputs/` directory
- Verify database schema matches expected format
- Try deleting database file to reset (will lose historical data)

## Development

### Adding New Metrics
```python
# Add custom performance metrics
monitor.add_performance_metric('custom_score', your_score)

# Add custom decision data  
decision_data = {
    'decision': 'Custom Action',
    'reasoning': 'Why this action was chosen',
    'confidence': 0.85,
    'visual_context': {...}
}
monitor.update_decision(decision_data)
```

### Extending the Dashboard
- Modify `templates/dashboard.html` for UI changes
- Add new API endpoints in `web_monitor.py` 
- Extend WebSocket events for additional real-time data
- Add custom CSS styling in the dashboard template

### Testing
```bash
# Run mock data test
python test_web_monitor.py

# Run with debug mode for development
python web_enhanced_training.py --web-only --host 0.0.0.0 --port 5000
```

The web monitoring system provides comprehensive visibility into Pokemon Crystal RL training, making it easier to debug issues, track progress, and understand agent behavior in real-time.
