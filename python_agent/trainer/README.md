# Pokemon Crystal RL Trainer v2.0 - Modular Architecture

A clean, modular refactoring of the Pokemon Crystal RL training system that eliminates code duplication and provides clear separation of concerns.

## 📁 Project Structure

```
trainer/
├── __init__.py              # Package initialization and exports
├── config.py               # Configuration classes and constants
├── game_state.py           # Game state detection and stuck detection
├── llm_manager.py          # LLM interaction and decision tracking
├── web_server.py           # Web monitoring server
├── training_strategies.py  # Training modes and rule-based actions
├── trainer.py              # Main trainer orchestrator
└── README.md               # This file

pokemon_trainer_v2.py       # Clean entry point script
```

## 🏗️ Architecture Overview

### Core Components

1. **TrainingConfig** (`config.py`)
   - Centralized configuration with sensible defaults
   - All training parameters, LLM settings, web options
   - Shared constants and mappings

2. **GameStateDetector** (`game_state.py`)
   - Fast, optimized game state detection from screenshots
   - Stuck detection and anti-stuck mechanisms
   - Cached state management for performance

3. **LLMManager** (`llm_manager.py`)
   - LLM initialization and model management
   - Decision tracking and performance monitoring
   - Adaptive interval adjustment based on response times
   - Comprehensive LLM analytics for web interface

4. **TrainingWebServer** (`web_server.py`)
   - Complete web monitoring interface
   - RESTful API endpoints for all training data
   - Real-time screenshot streaming
   - LLM decision monitoring and analytics

5. **TrainingStrategyManager** (`training_strategies.py`)
   - All training modes (ultra-fast, curriculum, etc.)
   - Rule-based action handlers for different game states
   - Action execution with precise frame timing

6. **UnifiedPokemonTrainer** (`trainer.py`)
   - Main orchestrator that coordinates all components
   - Training lifecycle management
   - Error handling and recovery mechanisms
   - Statistics collection and reporting

## 🚀 Key Improvements

### Eliminated Code Duplication
- **State detection logic**: Single implementation shared across components
- **Action mappings**: Centralized in config with human-readable names
- **LLM prompts and temperatures**: Shared configuration
- **Error handling**: Consistent patterns across all components
- **Performance tracking**: Unified stats system

### Enhanced Modularity
- **Clear interfaces**: Each component has well-defined responsibilities
- **Loose coupling**: Components communicate through clean APIs
- **Easy testing**: Individual components can be tested in isolation
- **Extensibility**: New components can be added without affecting existing code

### Improved Performance
- **Cached state detection**: Reduces expensive computations
- **Optimized screenshot processing**: Better memory management
- **Adaptive LLM intervals**: Automatically adjusts based on response times
- **Efficient web serving**: Minimal overhead for monitoring

### Better Debugging
- **Structured logging**: Consistent log formats across all components
- **Component identification**: Easy to trace which component generated logs
- **Performance metrics**: Detailed timing information for bottlenecks
- **LLM decision tracking**: Complete history of LLM interactions

## 📊 Web Interface Features

The refactored web server provides comprehensive monitoring:

- **Real-time training statistics**: Actions/second, LLM usage, progress
- **Live game screenshots**: Real-time display of current game state
- **LLM decision analytics**: Response times, state distributions, action patterns
- **System monitoring**: CPU, memory, and performance metrics
- **OCR text detection**: Detected in-game text and frequency analysis

### API Endpoints

- `GET /` - Main dashboard
- `GET /api/status` - Training status and metrics
- `GET /api/llm_decisions` - LLM decision history and analytics
- `GET /api/system` - System resource usage
- `GET /api/text` - OCR text detection data
- `GET /screen` - Real-time game screenshot

## 🎮 Usage Examples

### Basic Training
```bash
python pokemon_trainer_v2.py --rom game.gbc --web --actions 2000
```

### Ultra-Fast Rule-Based
```bash
python pokemon_trainer_v2.py --rom game.gbc --mode ultra_fast --no-llm --actions 10000
```

### Curriculum Learning
```bash
python pokemon_trainer_v2.py --rom game.gbc --mode curriculum --model llama3.2:3b --episodes 50
```

### Debug Mode
```bash
python pokemon_trainer_v2.py --rom game.gbc --debug --windowed --web --log-level DEBUG
```

## 🔧 Configuration Options

All configuration is centralized in `TrainingConfig`:

- **Training modes**: fast_monitored, curriculum, ultra_fast
- **LLM backends**: smollm2:1.7b, llama3.2:1b/3b, qwen2.5:3b, or none
- **Performance settings**: frames per action, LLM intervals, capture settings
- **Interface options**: web monitoring, screen capture, logging levels

## 🧪 Testing

Each component can be tested independently:

```python
from trainer.game_state import GameStateDetector
from trainer.llm_manager import LLMManager
from trainer.config import TrainingConfig

# Test individual components
detector = GameStateDetector(debug_mode=True)
config = TrainingConfig(rom_path="test.gbc")
llm_manager = LLMManager(config, detector)
```

## 🚀 Performance

The modular architecture provides several performance benefits:

1. **Reduced memory usage**: Eliminated duplicate state tracking
2. **Faster state detection**: Cached and optimized algorithms
3. **Better LLM utilization**: Adaptive intervals based on response times
4. **Efficient web serving**: Minimal overhead for real-time monitoring

## 📈 Migration from v1.0

The new modular trainer is **fully backward compatible** with the original command-line interface while providing much better organization and maintainability.

### Key Benefits:
- ✅ **Zero code duplication** - Everything is implemented once and reused
- ✅ **Clear separation** - Each component has a single responsibility  
- ✅ **Easy debugging** - Component-level logging and error tracking
- ✅ **Better testing** - Individual components can be tested in isolation
- ✅ **Enhanced monitoring** - Comprehensive web interface with analytics
- ✅ **Improved performance** - Optimized algorithms and caching
- ✅ **Future-proof** - Easy to extend with new features

The monolithic 2,500-line file has been transformed into a clean, maintainable, and extensible system that's much easier to understand, debug, and enhance!
