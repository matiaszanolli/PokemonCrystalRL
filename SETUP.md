# 🚀 Quick Setup Guide

Get started with Pokemon Crystal LLM RL training in under 5 minutes!

## 📋 Prerequisites

- **Python 3.8+** installed
- **Legal Pokemon Crystal ROM file**
- **4GB+ RAM** (for LLM features)

## 🔧 Step-by-Step Setup

### 1. **Clone & Install Dependencies**
```bash
# Clone the repository
git clone <repository-url>
cd pokemon_crystal_rl

# Install Python dependencies
pip install -r requirements.txt
```

### 2. **Install Ollama (for LLM features)**
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama service
ollama serve

# In another terminal, pull a model
ollama pull smollm2:1.7b
```

### 3. **Add Your ROM File**
```bash
# Create roms directory
mkdir roms

# Copy your legal Pokemon Crystal ROM
cp /path/to/your/pokemon_crystal.gbc roms/
```

### 4. **Create Save State (Optional but Recommended)**
Skip the intro by creating a save state after getting your first Pokemon:

```bash
python3 -c "
from pyboy import PyBoy
import time

print('🎮 Play through the intro, then press Ctrl+C to save')
pyboy = PyBoy('roms/pokemon_crystal.gbc', window='SDL2')

try:
    while True:
        pyboy.tick()
        time.sleep(0.016)
except KeyboardInterrupt:
    print('💾 Saving state...')
    with open('roms/pokemon_crystal.gbc.state', 'wb') as f:
        pyboy.save_state(f)
    print('✅ Save state created!')
finally:
    pyboy.stop()
"
```

### 5. **Start Training!**
```bash
# Basic LLM training
python3 llm_trainer.py --rom roms/pokemon_crystal.gbc --actions 2000

# View web dashboard
# Open: http://localhost:8080
```

## ✅ Verify Installation

### **Test Basic Components**
```bash
# Test PyBoy
python3 -c "from pyboy import PyBoy; print('✅ PyBoy works!')"

# Test LLM connection (if Ollama running)
python3 -c "import requests; print('✅ LLM:', requests.get('http://localhost:11434/api/tags').status_code == 200)"

# Test memory mapping
python3 -c "from core.memory_map import MEMORY_ADDRESSES; print(f'✅ Memory map: {len(MEMORY_ADDRESSES)} addresses')"
```

### **Quick Test Run**
```bash
# 30-second test run
python3 llm_trainer.py --rom roms/pokemon_crystal.gbc --actions 50
```

## 🎛️ Configuration Options

### **Training Intensity**
```bash
# Fast training (more rule-based)
python3 llm_trainer.py --rom roms/pokemon_crystal.gbc --actions 1000 --llm-interval 30

# Balanced training  
python3 llm_trainer.py --rom roms/pokemon_crystal.gbc --actions 2000 --llm-interval 15

# High-intelligence training
python3 llm_trainer.py --rom roms/pokemon_crystal.gbc --actions 5000 --llm-interval 8 --llm-model llama3.2:3b
```

### **Different Models**
```bash
# Install additional models
ollama pull llama3.2:1b      # Lightweight (1GB)
ollama pull llama3.2:3b      # Balanced (2GB) 
ollama pull deepseek-coder   # Code-focused (4GB)

# Use different model
python3 llm_trainer.py --rom roms/pokemon_crystal.gbc --llm-model llama3.2:3b
```

## 🌐 Web Dashboard

Once training starts, visit **http://localhost:8080** to see:

- **🖼️ Live Game Screen**: Real-time gameplay
- **🧠 LLM Decisions**: AI reasoning for each action
- **📊 Performance Stats**: Speed, rewards, progress
- **💰 Reward Breakdown**: Detailed reward analysis

## 📁 Output Files

Training generates these files:
- **`llm_training_stats_TIMESTAMP.json`**: Training statistics
- **`llm_decisions_TIMESTAMP.json`**: LLM decision history
- **`roms/pokemon_crystal.gbc.state`**: Save state file

## 🔧 Troubleshooting

### **Common Issues**

**"ROM not found"**:
```bash
ls -la roms/pokemon_crystal.gbc  # Should exist
```

**"LLM not available"**:
```bash
ollama list                      # Check installed models
ollama serve                     # Start Ollama service
```

**"Port 8080 in use"**:
```bash
python3 llm_trainer.py --web-port 8081  # Use different port
```

**Poor performance**:
```bash
# Use smaller model
python3 llm_trainer.py --llm-model smollm2:1.7b --llm-interval 25
```

### **System Requirements**

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **Python** | 3.8+ | 3.10+ |
| **RAM** | 2GB | 4GB+ |
| **Storage** | 1GB | 2GB+ |
| **CPU** | 2 cores | 4+ cores |

### **Model Requirements**

| Model | Size | RAM | Speed | Quality |
|-------|------|-----|-------|---------|
| `smollm2:1.7b` | 1GB | 2GB | Fast | Good |
| `llama3.2:1b` | 1GB | 2GB | Fast | Good |
| `llama3.2:3b` | 2GB | 4GB | Medium | Better |
| `deepseek-coder` | 4GB | 6GB | Slower | Best |

## 🎯 What's Next?

1. **📖 Read the [README.md](README.md)** for comprehensive documentation
2. **🔍 Check [CHANGELOG.md](CHANGELOG.md)** for recent improvements
3. **⭐ Star the repository** if you find it useful!
4. **🤝 Contribute** improvements and bug fixes

## 💡 Pro Tips

- **Create save states** at different game positions for varied training
- **Monitor the web dashboard** to understand AI decision patterns
- **Experiment with different models** to find the best balance of speed/quality
- **Adjust `--llm-interval`** based on your hardware capabilities
- **Use longer training sessions** (5000+ actions) for meaningful progress

---

**Happy Training!** 🎮✨
