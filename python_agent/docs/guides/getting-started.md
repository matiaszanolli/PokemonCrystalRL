# 🚀 Getting Started with Pokemon Crystal RL

Welcome to the Pokemon Crystal RL training system! This guide will get you up and running in under 10 minutes.

## 📋 Prerequisites

### System Requirements
- **Operating System**: Linux, macOS, or Windows with WSL2
- **Python**: 3.11 or higher (3.12 recommended)
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 10GB free space minimum
- **CPU**: 2+ cores (4+ recommended for better performance)

### Required Files
- **Pokemon Crystal ROM** (`pokemon_crystal.gbc`) - You must legally obtain this
- **Python Environment** - We recommend using `pyenv` or `conda`

---

## 🛠️ Installation

### Step 1: Clone Repository
```bash
git clone https://github.com/your-repo/pokemon-crystal-rl
cd pokemon-crystal-rl/python_agent
```

### Step 2: Set Up Python Environment
```bash
# Using pyenv (recommended)
pyenv install 3.12.0
pyenv virtualenv 3.12.0 pokemon-rl
pyenv activate pokemon-rl

# Using conda (alternative)
conda create -n pokemon-rl python=3.12
conda activate pokemon-rl

# Using venv (basic)
python -m venv pokemon-rl
source pokemon-rl/bin/activate  # Linux/macOS
# pokemon-rl\Scripts\activate  # Windows
```

### Step 3: Install Dependencies
```bash
# Core dependencies
pip install pyboy pillow numpy ollama python-ollama

# Optional dependencies for enhanced features
pip install gymnasium torch torchvision  # For advanced RL features
pip install matplotlib seaborn  # For plotting and analysis
```

### Step 4: Install Ollama
Ollama is required for LLM inference:

```bash
# macOS/Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama service
ollama serve
```

**Windows**: Download from [ollama.ai](https://ollama.ai/download) and follow installer

### Step 5: Download LLM Models
```bash
# Recommended model (fast and intelligent)
ollama pull smollm2:1.7b

# Alternative models
ollama pull llama3.2:1b    # Faster, less intelligent
ollama pull llama3.2:3b    # Slower, more intelligent
```

### Step 6: Set Up ROM
```bash
# Create ROM directory
mkdir -p ../roms

# Copy your legally obtained Pokemon Crystal ROM
cp path/to/your/pokemon_crystal.gbc ../roms/

# Verify ROM is accessible
ls -la ../roms/pokemon_crystal.gbc
```

---

## 🎮 First Run

### Quick Test
Let's verify everything works with a quick 50-action test:

```bash
python pokemon_trainer.py --rom ../roms/pokemon_crystal.gbc --mode ultra_fast --actions 50 --no-llm
```

**Expected Output:**
```
🚀 Initializing Ultra_Fast Training Mode
✅ PyBoy initialized (headless)
✅ Trainer initialized successfully!

⚡ STARTING ULTRA_FAST TRAINING
🎯 Target: 50 actions / 10 episodes
🚀 Ultra-fast rule-based training (no LLM overhead)

📊 TRAINING SUMMARY
⏱️ Duration: 0.1 seconds
🎯 Total actions: 50
🚀 Speed: 500.0 actions/sec
```

### LLM Test
Test with the SmolLM2 model:

```bash
python pokemon_trainer.py --rom ../roms/pokemon_crystal.gbc --mode fast_local --actions 30 --model smollm2:1.7b
```

**Expected Output:**
```
🚀 Initializing Fast_Local Training Mode
✅ PyBoy initialized (headless)
✅ Using LLM model: smollm2:1.7b
✅ Trainer initialized successfully!

⚡ STARTING FAST_LOCAL TRAINING
📊 Progress: 30/30 (38.5 a/s)
```

### Web Interface Test
Test the web monitoring interface:

```bash
python pokemon_trainer.py --rom ../roms/pokemon_crystal.gbc --mode fast_local --actions 100 --web
```

Then visit `http://localhost:8080` to see the live interface.

---

## 🎯 Training Modes Overview

### 1. Fast Local (Recommended for beginners)
**Best for**: Learning the system, content creation

```bash
python pokemon_trainer.py --rom ../roms/pokemon_crystal.gbc --mode fast_local --actions 500 --web
```

**Features:**
- 40+ actions/second with LLM intelligence
- Real-time web monitoring
- Balanced performance and quality
- Easy to understand and modify

### 2. Ultra Fast (Speed testing)
**Best for**: Performance benchmarking

```bash
python pokemon_trainer.py --rom ../roms/pokemon_crystal.gbc --mode ultra_fast --actions 2000 --no-llm
```

**Features:**
- 600+ actions/second
- Rule-based (no LLM overhead)
- Maximum performance
- Great for system validation

### 3. Curriculum (Progressive learning)
**Best for**: Research and systematic training

```bash
python pokemon_trainer.py --rom ../roms/pokemon_crystal.gbc --mode curriculum --episodes 20
```

**Features:**
- 5-stage progressive learning
- Mastery validation at each stage
- Comprehensive skill development
- Research-oriented approach

---

## 📊 Understanding Output

### Terminal Output
```
📊 Progress: 100/500 (45.2 a/s)
```
- `100/500`: Actions completed / Total target actions
- `45.2 a/s`: Actions per second (performance metric)

### Generated Files
- **`training_stats.json`**: Detailed performance statistics
- **`*.db` files**: Database files for advanced modes
- **Log files**: Debugging and analysis information

### Web Interface
Visit `http://localhost:8080` when using `--web` flag:
- **Live screen**: Game footage at 10 FPS
- **Real-time stats**: Performance metrics
- **Progress tracking**: Training advancement

---

## 🔧 Configuration Tips

### Performance Optimization
```bash
# Maximum speed (no LLM, no capture)
python pokemon_trainer.py --rom ../roms/pokemon_crystal.gbc --mode ultra_fast --no-llm --no-capture --actions 5000

# Balanced (recommended for most users)
python pokemon_trainer.py --rom ../roms/pokemon_crystal.gbc --mode fast_local --model smollm2:1.7b --web

# Quality focus (slower but more intelligent)
python pokemon_trainer.py --rom ../roms/pokemon_crystal.gbc --mode fast_local --model llama3.2:3b --llm-interval 5
```

### Resource Management
```bash
# Low memory systems (reduce LLM calls)
python pokemon_trainer.py --rom ../roms/pokemon_crystal.gbc --mode fast_local --llm-interval 20

# High-performance systems (more frequent LLM)
python pokemon_trainer.py --rom ../roms/pokemon_crystal.gbc --mode fast_local --llm-interval 5
```

---

## 🚨 Troubleshooting

### Common Issues and Solutions

#### 1. "PyBoy not available"
```bash
pip install pyboy
# If still issues, try:
pip install pyboy --force-reinstall
```

#### 2. "ROM file not found"
```bash
# Check ROM path
ls -la ../roms/pokemon_crystal.gbc

# Use absolute path if needed
python pokemon_trainer.py --rom /full/path/to/pokemon_crystal.gbc --mode fast_local
```

#### 3. "Model not found"
```bash
# Verify Ollama is running
ollama serve

# Pull model manually
ollama pull smollm2:1.7b

# List available models
ollama list
```

#### 4. Slow performance
```bash
# Use fastest model
python pokemon_trainer.py --rom ../roms/pokemon_crystal.gbc --mode fast_local --model smollm2:1.7b --llm-interval 15

# Or disable LLM entirely for maximum speed
python pokemon_trainer.py --rom ../roms/pokemon_crystal.gbc --mode ultra_fast --no-llm
```

#### 5. Web interface not loading
```bash
# Try different port
python pokemon_trainer.py --rom ../roms/pokemon_crystal.gbc --mode fast_local --web --port 8081

# Check firewall settings
# Ensure localhost connections are allowed
```

### Performance Diagnostics
```bash
# Test system performance
python -c "
import time
import ollama

# Test LLM speed
start = time.time()
ollama.generate(model='smollm2:1.7b', prompt='Test', options={'num_predict': 1})
print(f'LLM inference: {time.time() - start:.3f}s')

# Test PyBoy
from pyboy import PyBoy
start = time.time()
pyboy = PyBoy('../roms/pokemon_crystal.gbc', window='null')
for _ in range(100):
    pyboy.tick()
pyboy.stop()
print(f'PyBoy 100 ticks: {time.time() - start:.3f}s')
"
```

---

## ✅ Verification Checklist

Before proceeding to advanced usage, verify:

- [ ] Python 3.11+ installed and active
- [ ] All dependencies installed without errors
- [ ] Ollama service running (`ollama serve`)
- [ ] SmolLM2 model downloaded (`ollama list`)
- [ ] Pokemon Crystal ROM accessible
- [ ] Ultra-fast mode test successful (>300 actions/sec)
- [ ] Fast local mode test successful (~40 actions/sec)
- [ ] Web interface loading (if using `--web`)

---

## 🎯 Next Steps

Now that you have the basics working:

1. **📖 Learn the training modes**: [Training Modes Guide](training-modes.md)
2. **🤖 Explore LLM options**: [LLM Integration Guide](llm-integration.md)  
3. **🌐 Master the web interface**: [Web Interface Guide](web-interface.md)
4. **📚 Try curriculum learning**: [Curriculum Design](../examples/curriculum-design.md)
5. **🎬 Create content**: [Content Creation Guide](../examples/content-creation.md)

### Example Workflows

**Content Creator:**
```bash
python pokemon_trainer.py --rom ../roms/pokemon_crystal.gbc --mode fast_local --actions 2000 --web --windowed
```

**Researcher:**
```bash
python pokemon_trainer.py --rom ../roms/pokemon_crystal.gbc --mode curriculum --episodes 100 --debug
```

**Performance Tester:**
```bash
python pokemon_trainer.py --rom ../roms/pokemon_crystal.gbc --mode ultra_fast --actions 10000 --no-llm --no-capture
```

---

**🎉 Congratulations! You're ready to train Pokemon Crystal AI agents!**

For more advanced features and detailed configuration options, check out our [comprehensive documentation](../README.md).
