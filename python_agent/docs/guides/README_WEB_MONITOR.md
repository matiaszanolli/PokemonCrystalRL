# Pokemon Crystal RL Web Monitoring System

A comprehensive real-time web dashboard for monitoring Pokemon Crystal reinforcement learning training sessions, built with Flask, WebSockets, and modern web technologies.

## 🚀 Quick Start

### Option 1: Try the Interactive Demo (Recommended)
```bash
python demo_web_monitor.py
```
Then select option 1 for a mock data demo that showcases all features without requiring a ROM file.

### Option 2: Test with Mock Data
```bash
python test_web_monitor.py
```
Opens http://127.0.0.1:5000 with realistic simulated Pokemon game data.

### Option 3: Full Training with Monitoring
```bash
python web_enhanced_training.py --rom pokecrystal.gbc --episodes 50
```
Runs actual Pokemon Crystal training with live web monitoring.

## ✨ Key Features

### 🎮 Live Game Visualization
- **Real-time screen streaming** at 3x scale with pixel-perfect Game Boy aesthetics
- **WebSocket-powered updates** for smooth, responsive monitoring
- **Professional dashboard** with dark theme optimized for extended viewing

### 📊 Comprehensive Statistics  
- **Player tracking**: Position, money, badges, progress metrics
- **Pokemon party status**: HP bars, levels, species, battle readiness
- **Training analytics**: Episodes, steps, decisions, performance graphs

### 🎯 Intelligent Monitoring
- **Action logging**: Real-time button press history with reasoning
- **Agent decisions**: LLM strategic choices with confidence scores  
- **Visual analysis**: Screen type detection and OCR text recognition
- **Performance metrics**: Reward tracking and exploration analysis

### 🌐 Modern Web Interface
- **Responsive design** that works on desktop, tablet, and mobile
- **Real-time WebSocket updates** with no page refresh required
- **Interactive controls** for starting/stopping monitoring
- **Historical data access** via REST API endpoints

## 🏗️ Architecture

### Backend Components
- **`web_monitor.py`**: Core Flask server with WebSocket support
- **`web_enhanced_training.py`**: Training integration with monitoring hooks
- **`test_web_monitor.py`**: Mock data generator for testing
- **`demo_web_monitor.py`**: Interactive demonstration script

### Frontend  
- **`templates/dashboard.html`**: Modern responsive web dashboard
- **WebSocket client** for real-time communication
- **Chart.js integration** for performance visualization
- **Mobile-optimized interface** with touch-friendly controls

### Database
- **SQLite integration** for persistent training history
- **Automatic schema management** with game states and decisions
- **Performance metrics storage** for trend analysis
- **REST API access** to historical data

## 📋 System Requirements

### Core Dependencies
- **Python 3.7+** (3.8+ recommended for optimal performance)
- **Flask & Flask-SocketIO** for web server and real-time communication
- **NumPy** for numerical operations and array processing
- **OpenCV** for image processing and screenshot encoding

### Training Dependencies (for actual gameplay)
- **PyBoy** for Game Boy emulation and ROM execution
- **Pokemon Crystal ROM** (pokecrystal.gbc) for training
- **Local LLM** (Ollama with Llama 3.2 3B) for agent decisions

### Browser Support
- **Chrome, Firefox, Safari, Edge** (modern versions with WebSocket support)
- **Mobile browsers** for responsive monitoring on tablets/phones
- **No plugins required** - pure web standards implementation

## 🎯 Usage Examples

### Mock Data Testing
Perfect for verifying the system works without needing a ROM:
```bash
# Interactive demo with menu
python demo_web_monitor.py

# Direct mock data test
python test_web_monitor.py

# Automated mock demo
python demo_web_monitor.py --auto mock
```

### Real Training Monitoring
For actual Pokemon Crystal training with live monitoring:
```bash
# Full training with web dashboard
python web_enhanced_training.py --rom pokecrystal.gbc --episodes 100

# Custom configuration
python web_enhanced_training.py --rom crystal.gbc --episodes 50 --port 5001

# Web-only mode for historical data
python web_enhanced_training.py --web-only --port 5000
```

### Custom Integration
Integrate monitoring into your own training scripts:
```python
from web_monitor import PokemonRLWebMonitor

# Create monitor instance
monitor = PokemonRLWebMonitor(your_training_session)

# Start web server in background
monitor.start_web_server()

# In your training loop:
monitor.update_screenshot(screenshot_array)
monitor.update_action(action_name, reasoning)
monitor.update_decision(decision_data)
```

## 📊 Dashboard Features

### Main Panel
- **Live Game Screen**: 3x scaled screenshots with Game Boy pixel art preservation
- **Player Statistics**: Real-time position, money, badges, and progress tracking
- **Pokemon Party**: Visual HP bars, levels, species, and battle status
- **Training Metrics**: Episodes completed, total steps, decisions made

### Side Panel
- **Control Interface**: Start/stop monitoring with visual status indicators
- **Action History**: Scrolling log of recent button presses and commands
- **Agent Decisions**: Strategic choices with reasoning and confidence scores
- **Performance Graphs**: Real-time visualization of training progress

## 🔧 Configuration

### Command Line Options
```bash
python web_enhanced_training.py --help
```
- `--rom`: Path to Pokemon Crystal ROM file
- `--episodes`: Number of training episodes to run
- `--host`: Web server host address (default: 127.0.0.1)
- `--port`: Web server port (default: 5000)
- `--web-only`: Start dashboard only, no training

### Environment Variables
```bash
export POKEMON_WEB_HOST=0.0.0.0    # Allow external connections
export POKEMON_WEB_PORT=5000       # Web server port
export POKEMON_ROM_PATH=./crystal.gbc  # ROM file location
```

## 🛠️ Development

### File Structure
```
pokemon_crystal_rl/python_agent/
├── web_monitor.py              # Core web monitoring server
├── web_enhanced_training.py    # Training integration wrapper
├── test_web_monitor.py         # Mock data testing script
├── demo_web_monitor.py         # Interactive demonstration
├── templates/
│   └── dashboard.html          # Web dashboard interface
├── outputs/                    # Training data and reports
│   ├── *.json                  # Training session reports
│   ├── *.db                    # SQLite training databases
│   └── screenshots/            # Captured game screenshots
└── docs/
    ├── WEB_MONITORING.md       # Detailed usage guide
    └── WEB_MONITOR_IMPLEMENTATION.md  # Technical documentation
```

### API Endpoints
- **GET `/`**: Main dashboard interface
- **GET `/api/status`**: Current monitoring status and live statistics  
- **GET `/api/history`**: Historical training data from database
- **GET `/api/training_reports`**: Available training report files

### WebSocket Events
- **Client→Server**: `start_monitoring`, `stop_monitoring`, `request_screenshot`
- **Server→Client**: `status`, `screenshot`, `stats_update`, `new_action`, `new_decision`

## 🐛 Troubleshooting

### Web Server Issues
```bash
# Check if port is in use
lsof -i :5000

# Use different port
python web_enhanced_training.py --port 5001

# Check firewall settings
sudo ufw status
```

### No Data Updates
1. Verify "Start Monitor" button was clicked in web interface
2. Check browser console (F12) for JavaScript errors
3. Ensure training session is actively running
4. Monitor terminal output for connection errors

### Screenshot Problems  
1. Verify training environment provides valid RGB screenshots
2. Check screenshot array format (expected: numpy RGB format)
3. Monitor terminal for image encoding errors
4. Test with mock data to isolate issues

## 📚 Documentation

- **[WEB_MONITORING.md](docs/WEB_MONITORING.md)**: Complete user guide with API reference
- **[WEB_MONITOR_IMPLEMENTATION.md](docs/WEB_MONITOR_IMPLEMENTATION.md)**: Technical implementation details
- **Interactive Demo**: Run `python demo_web_monitor.py` for guided tour
- **System Check**: Option 4 in demo menu shows current configuration

## 🎉 Success Showcase

This web monitoring system provides:

✅ **Professional Quality**: Production-ready dashboard suitable for research presentations  
✅ **Real-time Insights**: Complete visibility into training progress and agent behavior  
✅ **Zero Friction**: Works with existing training code without modifications  
✅ **Comprehensive Coverage**: Monitors everything from low-level actions to high-level strategy  
✅ **Robust Performance**: Stable operation during extended training sessions  
✅ **Developer Friendly**: Extensive documentation and testing tools  

The system transforms Pokemon Crystal RL from a "black box" training process into a transparent, observable, and debuggable experience that significantly improves development velocity and research insights.

---

**Ready to start?** Run `python demo_web_monitor.py` and select option 1 for an instant demo with mock data! 🚀
