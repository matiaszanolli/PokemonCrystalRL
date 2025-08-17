# 🎮 Pokemon Crystal RL - Deep Learning Training Platform

**An advanced reinforcement learning platform for Pokemon Crystal with LLM integration, vision processing, and real-time monitoring.**

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Optional: Install Ollama for LLM support
curl -fsSL https://ollama.com/install.sh | sh
ollama pull smollm2:1.7b

# Run the trainer
python -m pokemon_crystal_rl.trainer --rom path/to/pokemon_crystal.gbc --mode fast --web
```

Monitor training at [http://localhost:8080](http://localhost:8080)

## 🌟 Features

### Core Features
- 🎮 **Multiple Training Modes**: Fast, LLM-powered, and curriculum learning
- 🧠 **LLM Integration**: Local models via Ollama with zero API costs
- 👁️ **Vision Processing**: ROM-based font recognition and scene analysis
- 📊 **Real-time Monitoring**: Web dashboard with performance metrics

### Technical Stack
- 🎮 **Emulation**: PyBoy Game Boy emulator integration
- 🌐 **Web Interface**: Real-time monitoring and control
- 📸 **Vision Pipeline**: Text detection and UI analysis
- 📈 **Analytics**: Comprehensive training metrics

## 📦 Project Structure

```
pokemon_crystal_rl/
├── pokemon_crystal_rl/         # Main package
│   ├── agents/                # Agent implementations
│   ├── core/                 # Core game integration
│   ├── monitoring/           # Web monitoring interface
│   ├── trainer/              # Training orchestration
│   ├── utils/               # Shared utilities
│   └── vision/              # Computer vision
├── docs/                    # Documentation
├── tests/                   # Test suite
└── tools/                   # Helper scripts
```

## 📚 Documentation

- [Training Guide](docs/TRAINING_OVERVIEW.md): Complete training documentation
- [Web Monitoring](docs/WEB_MONITORING.md): Using the web interface
- [Vision System](docs/VISION_SYSTEM.md): Vision processing details
- [Development](docs/DEVELOPMENT.md): Contributing and development

## 🚀 Usage Examples

### Standard Training
```bash
python -m pokemon_crystal_rl.trainer \
    --rom pokemon_crystal.gbc \
    --mode fast \
    --actions 10000 \
    --web
```

### LLM-Powered Training
```bash
python -m pokemon_crystal_rl.trainer \
    --rom pokemon_crystal.gbc \
    --mode llm \
    --llm-backend ollama \
    --model smollm2:1.7b \
    --web
```

### Curriculum Learning
```bash
python -m pokemon_crystal_rl.trainer \
    --rom pokemon_crystal.gbc \
    --mode curriculum \
    --stages 5 \
    --web
```

## 🔧 Configuration

### LLM Models
- **Fast**: `smollm2:1.7b` - 1GB VRAM, ~50ms inference
- **Balanced**: `llama2:3b` - 2GB VRAM, ~100ms inference
- **Powerful**: `qwen:7b` - 4GB VRAM, ~200ms inference

### Performance Tuning
```bash
# Speed optimization
--target-fps 60 --no-capture --disable-vision

# Quality optimization
--capture-screens --quality high --enable-ocr

# Memory optimization
--batch-size 32 --memory-limit 2048
```

## 📊 Monitoring

### Web Dashboard
- Main interface: [http://localhost:8080](http://localhost:8080)
- Real-time stats: [http://localhost:8080/api/stats](http://localhost:8080/api/stats)
- Screenshots: [http://localhost:8080/api/screen](http://localhost:8080/api/screen)

### Metrics
The platform tracks:
- Training progress and rewards
- System resource usage
- Game state and progress
- Agent decisions and reasoning

## 🎮 Game State Analysis

The agent tracks:
- **Location**: Map position and context
- **Party**: Pokemon team status
- **Progress**: Badges, items, money
- **Goals**: Current objectives and strategies

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

See [CONTRIBUTING.md](docs/CONTRIBUTING.md) for guidelines.

## 📄 License

MIT License - See [LICENSE](LICENSE) for details.

---

**Built with 💚 by the Pokemon Crystal RL team**
