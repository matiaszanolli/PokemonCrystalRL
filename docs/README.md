# 🎮 Pokemon Crystal RL - Unified Training System

**An advanced Pokemon Crystal AI training platform with multiple training modes, optimized streaming, and intelligent LLM integration.**

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. For LLM modes - Install Ollama and pull model
curl -fsSL https://ollama.com/install.sh | sh
ollama pull smollm2:1.7b

# 3. Run the unified trainer
cd python_agent
python pokemon_trainer.py --rom path/to/pokecrystal.gbc --mode ultra_fast --actions 1000

# 4. Monitor training with web interface
open http://localhost:8080
```

## 🧠 Features

### **🎯 Multiple Training Modes**
- ✅ **Ultra Fast**: 100+ FPS rule-based training for speed
- ✅ **Synchronized**: Frame-perfect LLM decision making
- ✅ **Curriculum**: Progressive learning with adaptive difficulty
- ✅ **Legacy Fast**: Traditional high-speed training

### **🚀 Performance Optimizations**
- ✅ **Optimized Video Streaming**: 10x latency reduction, 4x smaller frames
- ✅ **Smart Screen Capture**: Adaptive quality and compression
- ✅ **Real-time Monitoring**: Web dashboard with live metrics
- ✅ **Error Recovery**: Automatic crash detection and recovery

### **🤖 LLM Integration**
- ✅ **Multiple Backends**: Ollama, OpenAI, Anthropic support
- ✅ **Vision Processing**: OCR and game state analysis
- ✅ **Strategic Planning**: Long-term goal understanding
- ✅ **Cost-Efficient**: Local models with zero API costs

### **🔧 Technical Stack**
- 🎮 **PyBoy Emulator**: Pure Python Game Boy emulator
- 🌐 **Web Interface**: Real-time monitoring and control
- 📊 **Advanced Analytics**: Performance metrics and statistics
- 🖼️ **Vision Pipeline**: OCR and visual context processing

### **Performance Comparison**

| Method | Cost | Speed | Intelligence | Privacy |
|--------|------|-------|--------------|---------|
| **Local LLM** | $0 | ~0.1s | ⭐⭐⭐⭐ | 🔒 100% |
| Traditional RL | $0 | ~0.01s | ⭐⭐ | 🔒 100% |
| OpenAI GPT-4 | ~$0.03/1K | ~2-5s | ⭐⭐⭐⭐⭐ | ❌ Cloud |

## 📁 Project Structure

```
pokemon_crystal_rl/
├── python_agent/
│   ├── pokemon_trainer.py      # 🎯 Main unified trainer entry point
│   ├── trainer/                # 🏗️ Core training system
│   │   ├── trainer.py         # Main orchestrator
│   │   ├── config.py          # Training configuration
│   │   ├── training_strategies.py # Mode-specific strategies
│   │   └── web_server.py      # Real-time web interface
│   ├── core/                  # 🔧 Core modules
│   │   ├── video_streaming.py # Optimized streaming
│   │   └── monitoring.py      # Performance monitoring
│   ├── vision/                # 👁️ Computer vision
│   │   └── vision_processor.py # OCR and analysis
│   ├── llm/                   # 🧠 LLM integration
│   │   └── llm_manager.py     # Multi-backend LLM support
│   ├── demos/                 # 📖 Example scripts
│   └── docs/                  # 📚 Documentation
├── requirements.txt
└── README.md
```

## 🎯 Usage Examples

### **🚀 Ultra-Fast Training (Recommended)**
```bash
cd python_agent
python pokemon_trainer.py --rom pokecrystal.gbc --mode ultra_fast --actions 10000 --no-llm
```

### **🧠 LLM-Powered Strategic Training**
```bash
python pokemon_trainer.py --rom pokecrystal.gbc --mode synchronized --llm-backend ollama --model smollm2:1.7b --actions 1000
```

### **🌐 Web Monitoring**
```bash
# Start with web interface
python pokemon_trainer.py --rom pokecrystal.gbc --mode curriculum --enable-web --port 8080
# Open http://localhost:8080 to monitor
```

### **🎬 Content Creation Mode**
```bash
# High-quality streaming for recording
python pokemon_trainer.py --rom pokecrystal.gbc --mode synchronized --capture-screens --quality high
```

### **📊 Performance Testing**
```bash
# Run demo with metrics
python demos/optimized_streaming_demo.py
```

## 🔧 Advanced Configuration

### **📊 Training Modes**
```bash
# Ultra Fast: Rule-based, 100+ FPS
--mode ultra_fast --no-llm

# Synchronized: Frame-perfect LLM decisions
--mode synchronized --llm-backend ollama --model smollm2:1.7b

# Curriculum: Progressive difficulty
--mode curriculum --curriculum-stages 5

# Legacy Fast: Traditional fast training
--mode legacy_fast --target-fps 30
```

### **🤖 LLM Model Options**
```bash
# Ultra-fast for real-time (recommended)
ollama pull smollm2:1.7b        # 1GB VRAM, ~50ms inference

# Balanced performance
ollama pull llama3.2:3b         # 2GB VRAM, ~100ms inference

# High capability
ollama pull qwen2.5:7b          # 4GB VRAM, ~200ms inference
```

### **⚡ Performance Tuning**
```bash
# Optimize for speed
--target-fps 60 --no-capture --disable-vision

# Optimize for quality
--capture-screens --quality high --enable-ocr

# Optimize for streaming
--enable-web --stream-quality medium --compression-level 6
```

## 📊 Monitoring & Analytics

### **🌐 Web Dashboard**
```bash
# Real-time monitoring interface
open http://localhost:8080

# API endpoints
curl http://localhost:8080/api/status
curl http://localhost:8080/api/screenshot
curl http://localhost:8080/api/stats
```

### **📈 Performance Metrics**
```bash
# View trainer logs
tail -f pokemon_trainer.log

# Check performance stats
python -c "from trainer.trainer import UnifiedPokemonTrainer; trainer = UnifiedPokemonTrainer(); print(trainer.get_stats())"
```

### **🔍 Debug Information**
```bash
# Enable debug mode
python pokemon_trainer.py --debug --log-level DEBUG

# View captured text (OCR)
curl http://localhost:8080/api/ocr_text
```

## 🎮 Game State Analysis

The agent tracks comprehensive game state:
- **Player Position**: Map coordinates and location
- **Party Status**: Pokemon levels, HP, species
- **Game Progress**: Badges, money, items
- **Strategic Context**: Current goals and threats

### **Sample Agent Decision**
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

## 🚀 Next Steps & Improvements

### **1. Vision Integration**
```python
# Add screenshot analysis for better context
screenshot = env.render(mode="rgb_array")
# Send to vision-language model
```

### **2. Hierarchical Planning**
```python
# Multi-level goal decomposition
# Long-term: Beat Elite Four
# Medium-term: Get to next gym
# Short-term: Heal Pokemon at Pokemon Center
```

### **3. Faster Models**
```bash
# Try cutting-edge efficient models
ollama pull qwen2:1.5b          # Alibaba's compact model
ollama pull tinyllama:1.1b      # Ultra-fast 1B parameters
```

### **4. Multi-Agent Coordination**
```python
# Multiple agents with different roles
# Explorer, Trainer, Battle Specialist, Item Manager
```

## 🏆 Success Metrics

Current agent achievements:
- ✅ **Strategic Decision Making**: Makes contextual game decisions
- ✅ **Memory Formation**: Records 295+ decisions per session  
- ✅ **Game Interaction**: Successfully gains money and progresses
- ✅ **Cost Efficiency**: $0 operational cost vs $$$$ cloud AI
- ✅ **Speed**: Real-time gameplay at 2-10 steps/second

## 🤝 Contributing

1. **Add New Models**: Test different Ollama models for better performance
2. **Improve Prompts**: Enhance strategic decision-making prompts
3. **Add Vision**: Integrate screenshot analysis capabilities
4. **Memory Enhancement**: Improve episodic memory and learning

## 📜 License

MIT License - See [LICENSE](LICENSE) for details.

---

**🎮 Ready to watch an AI master Pokemon Crystal without spending a penny on API calls!**
