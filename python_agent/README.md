# Pokemon Crystal RL - Unified Training System

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyBoy](https://img.shields.io/badge/PyBoy-2.6.0-green.svg)](https://github.com/Baekalfen/PyBoy)
[![SmolLM2](https://img.shields.io/badge/SmolLM2-1.7B-orange.svg)](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B)

> **🎮 The most advanced Pokemon Crystal RL training system ever built**
> 
> Train AI agents to play Pokemon Crystal using state-of-the-art LLMs, real-time monitoring, and progressive curriculum learning. Achieve speeds of 600+ actions/second with intelligent decision-making.

<p align="center">
  <img src="docs/assets/pokemon-crystal-rl-banner.png" alt="Pokemon Crystal RL" width="600">
</p>

---

## 🌟 **Highlights**

- **🚀 Ultra-Fast Performance**: Up to 630+ actions/second
- **🧠 LLM-Powered**: SmolLM2-1.7B integration for intelligent decisions  
- **📚 Curriculum Learning**: Progressive 5-stage skill development
- **📺 Real-Time Monitoring**: Web interface with live gameplay
- **🎯 Multiple Training Modes**: From speed testing to research
- **📊 Comprehensive Analytics**: Detailed performance tracking
- **🔧 Easy to Use**: One unified script for all functionality

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

### **1. Install Dependencies**
```bash
# Install core dependencies
pip install pyboy pillow numpy ollama

# Pull the recommended LLM model
ollama pull smollm2:1.7b
```

### **2. Run Training**
```bash
# Fast training with web interface (recommended)
python pokemon_trainer.py --rom path/to/pokemon_crystal.gbc --mode fast_local --web

# Ultra-fast speed testing  
python pokemon_trainer.py --rom path/to/pokemon_crystal.gbc --mode ultra_fast --no-llm

# Progressive curriculum learning
python pokemon_trainer.py --rom path/to/pokemon_crystal.gbc --mode curriculum --episodes 50
```

### **3. Monitor Progress**
- **Web Interface**: Visit `http://localhost:8080` for real-time monitoring
- **Statistics**: Check `training_stats.json` for detailed metrics
- **Terminal**: Watch live progress updates during training

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

## 🌐 **Web Interface**

Real-time monitoring dashboard available at `http://localhost:8080`:

### **Features**
- 📸 **Live Gameplay**: 10 FPS screen capture
- 📊 **Performance Metrics**: Actions/sec, LLM calls, episodes
- 🎯 **Training Progress**: Stage advancement, success rates
- 📈 **Real-time Charts**: Performance graphs and trends
- 🔧 **Configuration**: Runtime parameter adjustment

### **Usage**
```bash
# Enable web interface for any mode
python pokemon_trainer.py --rom game.gbc --mode fast_local --web --port 8080
```

### **Screenshots**
See [📖 Web Interface Guide](docs/guides/web-interface.md) for detailed screenshots and usage instructions.

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
