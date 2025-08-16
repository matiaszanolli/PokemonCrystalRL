# 🎮 Pokemon Crystal AI Agent - Local LLM Edition

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyBoy](https://img.shields.io/badge/PyBoy-2.6.0-green.svg)](https://github.com/Baekalfen/PyBoy)
[![Ollama](https://img.shields.io/badge/Ollama-Llama3.2-blue.svg)](https://ollama.com/)

**A cost-free, intelligent Pokemon Crystal AI agent powered by local Llama models via Ollama.**

> **🎯 Latest Script: `pokemon_trainer.py` - Unified training system with intelligent LLM decision-making and real-time monitoring**

<p align="center">
  <img src="docs/assets/pokemon-crystal-rl-banner.png" alt="Pokemon Crystal RL" width="600">
</p>

---

## 🧠 **Features**

### **Enhanced LLM Intelligence**
- ✅ **Cost-Free**: No OpenAI/API fees - runs entirely locally
- ✅ **Fast**: ~2.3 actions/second with intelligent decision making
- ✅ **State-Aware**: Detects game states (menus, dialogue, overworld, battles)
- ✅ **Anti-Stuck Logic**: Automatically breaks out of loops and stuck situations
- ✅ **Context-Driven**: Uses state-specific guidance for optimal gameplay
- ✅ **Multi-Model**: Supports SmolLM2, Llama3.2 1B/3B, and rule-based fallback

### **Advanced Technical Stack**
- 🎮 **PyBoy Emulator**: Direct Game Boy emulation with frame-perfect timing
- 🤖 **Enhanced LLM Pipeline**: State detection + context-aware prompting
- 📸 **Real-time Capture**: 5 FPS synchronized screenshot analysis
- 👁️ **Vision Processing**: OCR text detection and game UI analysis
- 🌐 **Real-time Web Monitor**: Live Pokemon game streaming with WebSocket updates
- 🧠 **Smart Decision Making**: Temperature-adjusted responses based on game context
- 🔄 **Integrated Monitoring**: Seamless bridge connecting trainer to web dashboard

### **Performance Comparison**

| Method | Cost | Speed | Intelligence | Privacy |
|--------|------|-------|--------------|---------|
| **Local LLM** | $0 | ~0.1s | ⭐⭐⭐⭐ | 🔒 100% |
| Traditional RL | $0 | ~0.01s | ⭐⭐ | 🔒 100% |
| OpenAI GPT-4 | ~$0.03/1K | ~2-5s | ⭐⭐⭐⭐⭐ | ❌ Cloud |

---

## 📚 **Table of Contents**

- [🚀 Quick Start](#-quick-start)
- [📋 Installation](#-installation)  
- [🎯 Training Modes](#-training-modes)
- [🤖 LLM Models](#-llm-models)
- [📊 Performance](#-performance)
- [🌐 Web Interface](#-web-interface)
- [📖 Documentation](#-documentation)
- [🛠️ Development](#️-development)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)

---

## 🚀 **Quick Start**

### **1. Install Ollama & Dependencies**
```bash
# 1. Install Ollama (if not already installed)
curl -fsSL https://ollama.com/install.sh | sh

# 2. Pull the local LLM model
ollama pull llama3.2:3b

# 3. Install Python dependencies
pip install pyboy pillow numpy ollama
```

### **2. Run Intelligent Gameplay (Enhanced Unified Trainer)**
```bash
# 🎯 Main script - Watch AI play intelligently with enhanced LLM decision-making
cd python_agent
python pokemon_trainer.py --rom ../roms/pokemon_crystal.gbc --mode fast_monitored --debug --web

# Fast training with SmolLM2 (recommended)
python pokemon_trainer.py --rom ../roms/pokemon_crystal.gbc --actions 1000 --llm-interval 10

# Ultra-fast training with different models
python pokemon_trainer.py --rom ../roms/pokemon_crystal.gbc --model llama3.2:1b --actions 5000
```

### **3. Alternative Training Modes**
```bash
# Legacy LLM play script (still available)
python llm_play.py --no-headless --max-steps 5000

# Ultra-fast rule-based training
python pokemon_trainer.py --rom ../roms/pokemon_crystal.gbc --mode ultra_fast --no-llm

# Curriculum-based progressive learning
python pokemon_trainer.py --rom ../roms/pokemon_crystal.gbc --mode curriculum --episodes 50
```

### **4. Monitor Progress**
- **Terminal Output**: Real-time gameplay decisions and progress
- **Agent Memory**: SQLite database with decision history
- **Web Interface**: Visit `http://localhost:8080` for unified trainer monitoring

---

## 🎯 **LLM Play - Interactive Gameplay**

The `llm_play.py` script is the latest and most intelligent way to run the Pokemon Crystal AI agent using local LLM models.

### **📁 Script Features**
- ✅ **Cost-Free**: Uses local Ollama models (no API fees)
- ✅ **Strategic Decision Making**: Analyzes game state contextually
- ✅ **Memory Formation**: Records 295+ decisions per session
- ✅ **Real-Time Monitoring**: Watch AI decisions in terminal
- ✅ **Speed**: 2-10 steps/second intelligent gameplay

### **🎮 Usage Examples**

```bash
# Watch the AI play (recommended first run)
python llm_play.py --no-headless --max-steps 5000

# Fast training session (headless)
python llm_play.py --fast --max-steps 50000

# Try different LLM models
ollama pull phi3.5:3.8b-mini-instruct
python llm_play.py --model phi3.5:3.8b-mini-instruct

# Custom ROM and save state
python llm_play.py --rom-path ../custom.gbc --save-state-path ../custom.state

# Long training session with custom delay
python llm_play.py --max-steps 100000 --step-delay 0.05
```

### **🧠 Game State Analysis**

The agent tracks comprehensive game state:
- **Player Position**: Map coordinates and location
- **Party Status**: Pokemon levels, HP, species
- **Game Progress**: Badges, money, items
- **Strategic Context**: Current goals and threats

### **💭 Sample Agent Decision**
```
Pokemon Crystal - Current Situation:

LOCATION: Map 1, Position (5, 10)
MONEY: $3000
BADGES: 0

TEAM: 1 Pokemon
  1. Cyndaquil (Fire) - Level 8 (HP: 25/30)

GAME PHASE: early_game
GOALS: Catch more Pokemon, Train team, Head to first gym

What should I do next? → Action: A (interact)
```

### **📊 Monitoring & Analytics**

```bash
# View agent's memory database
sqlite3 pokemon_agent_memory.db
.tables  # See stored decisions and game states

# Session summary shows:
# - Duration and steps completed
# - Actions per second
# - Memory formation statistics
# - Game progress indicators
```

---

## 📋 **Installation**

### **Prerequisites**
- **Python 3.11+** (recommended)
- **Pokemon Crystal ROM** (not included)
- **Ollama** for LLM inference
- **4GB+ RAM** (8GB+ recommended for larger models)

### **Quick Install**
```bash
git clone https://github.com/your-repo/pokemon-crystal-rl
cd pokemon-crystal-rl/python_agent
pip install -r requirements.txt
```

### **Detailed Setup**
See [📖 Installation Guide](docs/guides/installation.md) for complete setup instructions, including:
- System requirements
- ROM setup
- Model configuration
- Troubleshooting

---

## 🎯 **Training Modes**

### **🏃 Fast Local** *(Recommended)*
- **Performance**: ~40 actions/sec with LLM
- **Best for**: Content creation, balanced training
- **Features**: Real-time capture, web monitoring
```bash
python pokemon_trainer.py --rom game.gbc --mode fast_local --web
```

### **⚡ Ultra Fast**
- **Performance**: 600+ actions/sec (rule-based)
- **Best for**: Speed testing, benchmarking
- **Features**: Maximum performance, no LLM overhead
```bash
python pokemon_trainer.py --rom game.gbc --mode ultra_fast --no-llm
```

### **📚 Curriculum**
- **Performance**: Variable (20-30 actions/sec)
- **Best for**: Progressive learning, research
- **Features**: 5-stage mastery system
```bash
python pokemon_trainer.py --rom game.gbc --mode curriculum --episodes 100
```

### **🔬 Monitored**
- **Performance**: 5-15 actions/sec (comprehensive)
- **Best for**: Research, detailed analysis
- **Features**: Full logging, enhanced agent
```bash
python pokemon_trainer.py --rom game.gbc --mode monitored --debug
```

---

## 🤖 **LLM Models**

| Model | Speed | Intelligence | Memory | Recommendation |
|-------|--------|-------------|---------|----------------|
| **SmolLM2-1.7B** ⭐ | ⚡⚡⚡ | ⭐⭐⭐ | 2GB | **Best overall** |
| Llama3.2-1B | ⚡⚡ | ⭐⭐ | 1GB | Ultra-fast |
| Llama3.2-3B | ⚡ | ⭐⭐⭐⭐ | 3GB | Highest quality |
| Rule-based | ⚡⚡⚡⚡ | ⭐ | 0MB | Speed testing |

### **Model Performance**
- **SmolLM2-1.7B**: ~25ms inference, optimal for Pokemon RL
- **Llama3.2-1B**: ~30ms inference, good fallback option  
- **Llama3.2-3B**: ~60ms inference, best decision quality
- **Rule-based**: <1ms, pattern-based exploration

---

## 📊 **Performance**

### **Benchmark Results**
Based on testing with Pokemon Crystal ROM:

| Mode | Actions/sec | LLM Calls/sec | Memory Usage |
|------|------------|---------------|--------------|
| `ultra_fast --no-llm` | **630+** | 0 | ~1GB |
| `fast_local` | **40** | 4 | ~3GB |
| `curriculum` | **25** | 1.7 | ~3GB |
| `monitored` | **10** | 1 | ~4GB |

### **System Requirements**
- **Minimum**: 4GB RAM, 2 CPU cores, 10GB storage
- **Recommended**: 8GB RAM, 4+ CPU cores, 20GB storage
- **Optimal**: 16GB RAM, 8+ CPU cores, GPU (optional)

---

## 🌐 **Web Interface - NEW! Real-time Game Monitoring**

Watch your AI play Pokemon in real-time with our completely redesigned web monitoring system!

### **🎥 Live Game Streaming**
- **Real-time Screenshots**: Watch the actual Pokemon game screen at 2 FPS
- **WebSocket Streaming**: Instant updates with no page refreshes
- **Pixelated Rendering**: Authentic Game Boy visual experience
- **Responsive Design**: Works on desktop and mobile

### **📊 Training Analytics Dashboard**
- **Live Statistics**: Player position, money, badges, Pokemon party
- **Training Metrics**: Episodes completed, total steps, decisions made
- **Action History**: Recent 20 actions with timestamps and reasoning
- **Agent Decisions**: LLM decision logs with confidence scores
- **Performance Tracking**: Actions/sec, LLM response times, error rates

### **🚀 Quick Setup (Integrated Monitoring)**

```python
# New! Integrated web monitoring with trainer
from monitoring.trainer_monitor_bridge import create_integrated_monitoring_system

# Create trainer (your existing setup)
trainer = UnifiedPokemonTrainer(config)

# Add web monitoring in 3 lines!
web_monitor, bridge, thread = create_integrated_monitoring_system(trainer)
bridge.start_bridge()
trainer.start_training()

# Visit http://localhost:5000 to see live gameplay!
```

### **⚡ Legacy Trainer Web Interface**
```bash
# Built-in trainer web interface (alternative)
python pokemon_trainer.py --rom game.gbc --mode fast_monitored --web --port 8080
# Visit http://localhost:8080 for trainer dashboard
```

### **🎯 Web Interface Features**

| Feature | Integrated Monitor | Legacy Interface |
|---------|-------------------|------------------|
| **Live Game Screen** | ✅ 2 FPS WebSocket | ✅ 1 FPS HTTP |
| **Real-time Updates** | ✅ WebSocket | ✅ Polling |
| **Action History** | ✅ With reasoning | ✅ Basic |
| **Training Stats** | ✅ Comprehensive | ✅ Limited |
| **Agent Decisions** | ✅ LLM insights | ❌ Not available |
| **Modern UI** | ✅ Responsive | ✅ Functional |

### **📱 Monitoring URLs**
- **Integrated Monitor**: `http://localhost:5000` (default)
- **Trainer Dashboard**: `http://localhost:8080` (with --web flag)
- **Custom Port**: Configure via `create_integrated_monitoring_system(port=YOUR_PORT)`

### **🔧 Advanced Configuration**
```python
# Custom monitoring setup
bridge.screenshot_update_interval = 0.3  # 3.3 FPS
bridge.stats_update_interval = 1.0       # Update stats every second
bridge.bridge_fps = 15                   # Higher bridge update rate
```

### **📖 Web Monitoring Documentation**
See [📖 Web Monitor Integration Guide](docs/WEB_MONITOR_INTEGRATION.md) for:
- Complete setup instructions
- API reference
- Performance tuning
- Troubleshooting
- Advanced integration examples

---

## 📖 **Documentation**

### **🚀 Quick Access**
- **[📚 Complete Documentation Hub](docs/README.md)** - **Start here!** Navigation to all guides
- **[🚀 Getting Started](docs/guides/getting-started.md)** - Setup in 10 minutes
- **[🎬 Content Creation](docs/examples/content-creation.md)** - Record videos and streams
- **[🔧 API Reference](docs/api/unified-trainer-api.md)** - Developer documentation

### **📚 Documentation by Use Case**

| **👤 User Type** | **📖 Start Here** | **⏱️ Time** |
|------------------|-------------------|----------|
| **🎮 Gamers** | [Getting Started](docs/guides/getting-started.md) | 10 min |
| **🎬 Content Creators** | [Content Creation Guide](docs/examples/content-creation.md) | 15 min |
| **🔬 Researchers** | [Curriculum Training](docs/research/CURRICULUM_TRAINING_GUIDE.md) | 30 min |
| **💻 Developers** | [API Reference](docs/api/unified-trainer-api.md) | 45 min |

### **🎯 Popular Guides**
- [🎯 Pokemon Trainer Guide](docs/guides/POKEMON_TRAINER_GUIDE.md) - Complete unified trainer usage
- [📊 Speed Optimization](docs/guides/SPEED_OPTIMIZATION_REPORT.md) - Performance tuning
- [🌐 Web Monitor Guide](docs/guides/README_WEB_MONITOR.md) - Real-time monitoring setup
- [📚 Curriculum Learning](docs/research/CURRICULUM_TRAINING_GUIDE.md) - Progressive training

> **💡 TIP**: Visit our [📚 Documentation Hub](docs/README.md) for organized navigation by use case!

---

## 🛠️ **Development**

### **Project Structure**
```
pokemon-crystal-rl/
├── python_agent/
│   ├── pokemon_trainer.py      # Main unified trainer
│   ├── pyboy_env.py           # Game environment
│   ├── enhanced_llm_agent.py  # LLM integration
│   ├── docs/                  # Documentation
│   ├── tests/                 # Test suite
│   └── archive/               # Legacy scripts
├── roms/                      # ROM files (not included)
├── models/                    # Trained models
└── docs/                      # Main documentation
```

### **Key Components**
- **`UnifiedPokemonTrainer`**: Main training orchestrator
- **`TrainingMode`**: Mode-specific implementations
- **`LLMBackend`**: Model abstraction layer
- **Web Server**: Real-time monitoring interface

### **🧪 Testing**

Comprehensive pytest-based test suite ensuring code quality and reliability:

```bash
# Run all tests
pytest

# Run with verbose output and coverage
pytest -v --cov=. --cov-report=html

# Run specific test modules
pytest tests/test_dialogue_state_machine.py
pytest tests/test_integration_and_performance.py
```

#### **Test Coverage**
- 🎯 **Dialogue Systems**: Choice recognition, semantic processing
- 🎮 **Game Integration**: PyBoy environment, state management
- 🔧 **Performance**: Speed benchmarks, memory usage
- 🌐 **Web Interface**: Monitoring endpoints, real-time updates

#### **Test Organization**
- **`tests/conftest.py`**: Shared fixtures and configuration
- **`tests/test_*.py`**: Organized test modules by functionality
- **`pytest.ini`**: Pytest configuration and settings

> 📖 **Test Documentation**: See [archive/tests/README.md](archive/tests/README.md) for migration details

### **Contributing**
See [🤝 Contributing Guide](CONTRIBUTING.md) for:
- Code standards and style
- Pull request process
- Issue reporting
- Development setup

---

## 📈 **Roadmap**

### **🚀 Current (v1.0)**
- ✅ Unified training system
- ✅ SmolLM2 integration
- ✅ Real-time monitoring
- ✅ Curriculum learning
- ✅ Performance optimization

### **🔮 Planned (v2.0)**
- 🔲 Multi-game support (Red/Blue, Gold/Silver)
- 🔲 Advanced curriculum designer
- 🔲 Model fine-tuning capabilities
- 🔲 Distributed training support
- 🔲 Advanced analytics dashboard

### **💡 Future Ideas**
- Vision-language model integration
- Reinforcement learning hybrid approach
- Tournament and battle optimization
- Community challenges and leaderboards

---

## 🏆 **Achievements**

### **Performance Records**
- **Fastest Training**: 630+ actions/second
- **Most Intelligent**: SmolLM2-1.7B integration
- **Most Comprehensive**: 4 complete training modes
- **Best Monitored**: Real-time web interface

### **Technical Milestones**
- **18 scripts unified** into 1 comprehensive system
- **25ms LLM inference** with SmolLM2-1.7B
- **10 FPS screen capture** with minimal overhead
- **5-stage curriculum** with mastery validation

---

## 🤝 **Contributing**

We welcome contributions! Here's how to help:

### **Ways to Contribute**
- 🐛 **Bug Reports**: File issues with detailed reproduction steps
- 💡 **Feature Requests**: Suggest improvements and new features
- 📖 **Documentation**: Improve guides and add examples
- 🔧 **Code**: Submit PRs for bug fixes and enhancements
- 🧪 **Testing**: Help test on different systems and configurations

### **Getting Started**
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests and documentation
5. Submit a pull request

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

---

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### **Third-Party Licenses**
- **PyBoy**: MIT License
- **Ollama**: MIT License  
- **SmolLM2**: Apache 2.0 License
- **Pokemon Crystal**: Game content owned by Nintendo/Game Freak

---

## 🙏 **Acknowledgments**

- **PyBoy Team**: Excellent Game Boy emulation
- **Hugging Face**: SmolLM2 model and ecosystem
- **Ollama**: Local LLM inference platform
- **Pokemon Community**: Inspiration and ROM hacking knowledge
- **Contributors**: Everyone who helped build and improve this project

---

## 📞 **Support**

### **Need Help?**
- 📖 **Documentation**: Check our comprehensive guides
- 💬 **Discussions**: Join GitHub Discussions for Q&A
- 🐛 **Issues**: Report bugs via GitHub Issues
- 📧 **Contact**: Reach out to maintainers

### **Community**
- 🌟 **Star** this repo if you find it useful
- 🍴 **Fork** to create your own variations
- 📢 **Share** with others interested in RL/Pokemon
- 🤝 **Contribute** to make it even better

---

<div align="center">

**🎮 Ready to train the ultimate Pokemon Crystal AI? 🚀**

[Get Started](docs/guides/getting-started.md) | [View Examples](docs/examples/) | [Join Community](https://github.com/your-repo/pokemon-crystal-rl/discussions)

</div>
