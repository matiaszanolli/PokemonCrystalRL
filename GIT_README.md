# 🎮 Pokémon Crystal Reinforcement Learning

A complete reinforcement learning system for training AI agents to play Pokémon Crystal using BizHawk emulator and Stable Baselines3.

## ⚠️ **IMPORTANT: ROM File Notice**

**This repository does NOT include ROM files.** You must provide your own legally obtained Pokémon Crystal ROM file:

1. **ROM File Required**: `pokecrystal.gbc` (2MB, Game Boy Color ROM)
2. **Legal Requirement**: You must own the original game cartridge
3. **Placement**: Put the ROM file in the project root directory
4. **Git Protection**: ROM files are automatically ignored by `.gitignore`

**We do not provide, distribute, or assist with obtaining ROM files.**

## 🚀 Quick Start

### Prerequisites
- **Linux System** (Ubuntu/Debian recommended)
- **Python 3.8+** with pip
- **BizHawk Emulator** (installed via Mono)
- **Legal Pokémon Crystal ROM** (not included)

### Installation
```bash
# 1. Clone repository
git clone <your-repo-url>
cd pokemon_crystal_rl

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Install BizHawk emulator (Ubuntu/Debian)
sudo apt update
sudo apt install mono-complete wget unzip
# Follow README.md for complete BizHawk setup

# 4. Place your ROM file
cp /path/to/your/pokemon_crystal.gbc ./pokecrystal.gbc

# 5. Verify system
python3 verify_setup.py

# 6. Start monitoring system
python3 start_monitoring.py
```

### Quick Launch
```bash
# Start monitoring and training system
./quick_start.sh

# Access dashboards:
# Web Dashboard: http://localhost:5000
# TensorBoard: http://localhost:6006
```

## 🎯 Features

### 🤖 **Machine Learning**
- **Algorithms**: PPO, DQN, A2C (Stable Baselines3)
- **GPU Support**: CUDA acceleration 
- **Environment**: Custom Gymnasium wrapper
- **Monitoring**: Real-time training metrics

### 📊 **Monitoring Dashboard**
- **Web Interface**: Professional training dashboard
- **System Metrics**: CPU, Memory, GPU, Disk usage
- **Training Control**: Start/stop runs via web UI
- **TensorBoard**: Advanced metrics visualization

### 🎮 **Emulator Integration**
- **BizHawk**: Game Boy Color emulation
- **Lua Bridge**: Python ↔ Game state communication
- **Save States**: Training checkpoint management
- **Memory Reading**: Direct game memory access

## 🏗️ Project Structure

```
pokemon_crystal_rl/
├── lua_bridge/              # Emulator-Python communication
│   ├── crystal_bridge.lua   # Main Lua script for BizHawk
│   └── json.lua            # JSON utilities for Lua
├── python_agent/           # RL training environment
│   ├── env.py              # Gymnasium environment wrapper
│   ├── train.py            # Main training script
│   ├── memory_map.py       # Game memory addresses
│   └── utils.py            # Helper functions
├── templates/              # Web dashboard templates
│   └── dashboard.html      # Main monitoring interface
├── monitor.py              # Training monitoring system
├── start_monitoring.py     # Monitoring system launcher
├── verify_setup.py         # System verification
└── README.md              # Complete documentation
```

## 🔧 Configuration

### Training Parameters
```python
# Example training configuration
{
    "algorithm": "ppo",           # PPO, DQN, or A2C
    "total_timesteps": 1000000,   # Training duration
    "learning_rate": 0.0003,      # Learning rate
    "n_envs": 1                   # Parallel environments
}
```

### Web API Endpoints
```bash
# Check training status
GET /api/status

# Start training run
POST /api/start_training

# View system metrics
GET /api/system

# List training runs
GET /api/runs
```

## 🧪 Development

### Testing
```bash
# Verify complete system
python3 verify_setup.py

# Test training script
cd python_agent
python3 train.py --run-id 1 --algorithm ppo --total-timesteps 1000

# Test emulator connection
bizhawk pokecrystal.gbc --lua=lua_bridge/crystal_bridge.lua
```

### Debugging
```bash
# Check running services
ps aux | grep -E "(flask|tensorboard)"

# View logs
tail -f monitoring.log

# Check API status
curl http://localhost:5000/api/status
```

## 📋 System Requirements

### Minimum Requirements
- **OS**: Ubuntu 20.04+ (or compatible Linux)
- **RAM**: 8GB (16GB recommended)
- **Storage**: 5GB free space
- **Python**: 3.8+
- **Internet**: For package installation

### Recommended Setup
- **CPU**: 4+ cores
- **GPU**: NVIDIA GPU with CUDA support
- **RAM**: 16GB+
- **Storage**: SSD with 10GB+ free

## ⚡ Performance Tips

1. **Use GPU**: CUDA acceleration for faster training
2. **Parallel Environments**: Multiple game instances
3. **Save States**: Quick episode resets
4. **Reward Shaping**: Optimize reward functions
5. **Hyperparameter Tuning**: Experiment with learning rates

## 📚 Documentation

- **README.md**: Complete setup and usage guide
- **CURRENT_STATUS.md**: Development status and next steps
- **API Documentation**: Built-in web dashboard help
- **Code Comments**: Extensive inline documentation

## ⚖️ Legal & Ethical

- **ROM Files**: Must be legally owned
- **Fair Use**: Educational/research purposes
- **No Distribution**: ROMs not included or provided
- **Respect Copyright**: Follow Nintendo's terms

## 🤝 Contributing

1. **Issues**: Report bugs and feature requests
2. **Pull Requests**: Code contributions welcome
3. **Documentation**: Help improve guides
4. **Testing**: Try on different systems

## 📧 Support

- **System Issues**: Check `verify_setup.py` output
- **Training Problems**: Monitor dashboard logs
- **Emulator Issues**: Test BizHawk installation
- **Performance**: Review system requirements

## 🎯 Roadmap

- [x] Basic RL training pipeline
- [x] Web monitoring dashboard  
- [x] BizHawk emulator integration
- [ ] Lua bridge communication debugging
- [ ] Advanced reward engineering
- [ ] Multi-agent training support
- [ ] Custom neural network architectures

---

**⚠️ Remember**: This is a research/educational project. Ensure you have legal rights to any ROM files you use.
