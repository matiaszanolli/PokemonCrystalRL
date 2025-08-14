# Pokémon Crystal Reinforcement Learning Project

🎮 **STATUS: READY FOR TRAINING** 🚀

A complete reinforcement learning system that trains AI agents to play Pokémon Crystal using BizHawk emulator with Lua scripting, real-time monitoring dashboard, and state-of-the-art RL algorithms.

## 🎯 Current Project Status

✅ **COMPLETED COMPONENTS:**
- BizHawk emulator installed and configured
- Complete Lua bridge system for game state extraction
- OpenAI Gym environment wrapper
- Support for PPO, DQN, and A2C algorithms
- Real-time web monitoring dashboard
- TensorBoard integration
- System monitoring and metrics
- Training history and progress tracking
- Comprehensive memory mapping for Pokémon Crystal
- Reward shaping and state preprocessing
- ROM file detected and ready

🔧 **READY TO USE:**
- Web dashboard at http://localhost:5000
- TensorBoard at http://localhost:6006
- One-command startup system
- Real-time training control

## Project Structure

```
pokemon_crystal_rl/
├── lua_bridge/
│   ├── crystal_bridge.lua       # Lua script to expose state + receive actions
│   └── json.lua                 # Lightweight Lua JSON encoder/decoder
├── python_agent/
│   ├── env.py                   # Gym-style environment wrapper for Python RL
│   ├── train.py                 # RL training script using SB3
│   ├── memory_map.py            # Memory addresses for Pokémon Crystal
│   └── utils.py                 # Helper functions (reward shaping, preprocessing)
├── README.md
└── requirements.txt             # Python dependencies
```

## Prerequisites

### Software Requirements
- **Emulator**: BizHawk (recommended) or similar emulator with Lua scripting support
- **ROM**: Pokémon Crystal ROM file (not included - obtain legally)
- **Python**: Python 3.8+ with pip
- **Operating System**: Windows, macOS, or Linux

### Hardware Requirements
- **CPU**: Multi-core processor recommended for training
- **RAM**: At least 8GB recommended
- **GPU**: CUDA-compatible GPU recommended for faster training (optional)

## Installation

1. **Clone or download this project structure**

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download and install BizHawk**:
   
   **For Ubuntu/Debian Linux:**
   ```bash
   # Install Mono runtime (required for BizHawk on Linux)
   sudo apt update
   sudo apt install -y mono-complete mono-devel libmono-system-windows-forms4.0-cil
   
   # Download and install BizHawk
   cd /tmp
   wget https://github.com/TASEmulators/BizHawk/releases/download/2.10/BizHawk-2.10-linux-x64.tar.gz
   tar -xzf BizHawk-2.10-linux-x64.tar.gz
   sudo mkdir -p /opt/bizhawk
   sudo mv EmuHawk* DiscoHawk* defctrl.json dll Gameboy gamedb Lua NES overlay Shaders /opt/bizhawk/
   sudo chown -R $USER:$USER /opt/bizhawk
   sudo ln -sf /opt/bizhawk/EmuHawkMono.sh /usr/local/bin/bizhawk
   chmod +x /opt/bizhawk/EmuHawkMono.sh
   ```
   
   **For Windows:**
   - Download from: https://github.com/TASEmulators/BizHawk/releases
   - Extract to a directory of your choice
   - Note the path to the executable (usually `EmuHawk.exe`)

4. **Obtain Pokémon Crystal ROM**:
   - You must legally obtain a Pokémon Crystal ROM file
   - Place it in the project directory or note its path

## Configuration

### Memory Addresses
The memory addresses in `memory_map.py` are configured for a standard Pokémon Crystal ROM. If you're using a different version or ROM hack, you may need to update these addresses.

### Emulator Setup
1. Open BizHawk
2. Load your Pokémon Crystal ROM
3. Go to Tools → Lua Console
4. Load the `lua_bridge/crystal_bridge.lua` script

## 🚀 Quick Start Guide

**Get started in 3 simple steps:**

### 1. Launch the Monitoring System
```bash
# Start the complete monitoring dashboard
python3 start_monitoring.py

# Or use the quick start script
./quick_start.sh
```

### 2. Open Your Browser
The system will automatically open:
- **Dashboard**: http://localhost:5000 (training control & monitoring)
- **TensorBoard**: http://localhost:6006 (detailed metrics)

### 3. Start Training
- Configure your training parameters in the dashboard
- Click "Start Training" 
- Monitor progress in real-time!

---

## 🗺️ Project Roadmap

### 🔴 Phase 1: IMMEDIATE ACTIONS (Next 1-2 hours)
**Goal: Get first successful training run**

✅ **COMPLETED:**
- [x] System setup and dependencies
- [x] BizHawk installation
- [x] ROM verification
- [x] Monitoring dashboard
- [x] Basic training infrastructure

🟡 **NEXT STEPS:**
- [ ] **Test the complete pipeline** (30 minutes)
  - [ ] Start monitoring system
  - [ ] Verify BizHawk launches with ROM
  - [ ] Test Lua script execution
  - [ ] Confirm Python-Lua communication
  - [ ] Run short training test (1000 steps)

- [ ] **Debug and fix initial issues** (60 minutes)
  - [ ] Memory address verification
  - [ ] Communication timeout tuning
  - [ ] Reward function validation
  - [ ] State preprocessing check

### 🟡 Phase 2: OPTIMIZATION (Next 2-4 hours)
**Goal: Stable, efficient training**

- [ ] **Performance Optimization**
  - [ ] Tune hyperparameters for faster convergence
  - [ ] Optimize memory usage and processing speed
  - [ ] Test parallel environments (if needed)
  - [ ] GPU utilization optimization

- [ ] **Training Stability**
  - [ ] Implement robust error handling
  - [ ] Add automatic restart on crashes
  - [ ] Improve reward shaping
  - [ ] Add curriculum learning (optional)

- [ ] **Monitoring Enhancements**
  - [ ] Real-time game state visualization
  - [ ] Live video feed from emulator
  - [ ] Advanced metrics and KPIs
  - [ ] Training alerts and notifications

### 🟠 Phase 3: ADVANCED FEATURES (Next 1-2 days)
**Goal: Production-ready system**

- [ ] **Enhanced AI Capabilities**
  - [ ] Multi-objective training (speed vs completion)
  - [ ] Hierarchical RL for complex strategies
  - [ ] Memory/attention mechanisms
  - [ ] Imitation learning from human play

- [ ] **Advanced Game Integration**
  - [ ] Save state management
  - [ ] Multiple ROM support
  - [ ] Battle strategy optimization
  - [ ] Item and team management

- [ ] **Research Features**
  - [ ] Experiment tracking and comparison
  - [ ] A/B testing framework
  - [ ] Model interpretability tools
  - [ ] Performance benchmarking

### 🟢 Phase 4: RESULTS & ANALYSIS (Ongoing)
**Goal: Achieve superhuman performance**

- [ ] **Training Campaigns**
  - [ ] Complete Johto region (8 badges)
  - [ ] Elite Four completion
  - [ ] Kanto region (16 badges total)
  - [ ] Champion battle victory

- [ ] **Analysis and Documentation**
  - [ ] Performance analysis and comparison
  - [ ] Strategy documentation
  - [ ] Video generation of best runs
  - [ ] Research paper preparation

---

## 🏁 Success Metrics

### **Short-term (24 hours)**
- ✅ System runs without crashes
- ✅ Agent learns basic movement
- ✅ Positive reward progression
- ✅ First wild Pokémon encounter

### **Medium-term (1 week)**
- ✅ First gym badge obtained
- ✅ Stable training for 1M+ timesteps
- ✅ Consistent improvement in performance
- ✅ Multiple algorithm comparison

### **Long-term (1 month)**
- ✅ Complete Johto region
- ✅ Superhuman performance metrics
- ✅ Documented winning strategies
- ✅ Publication-ready results

---

## Usage

### Basic Training

Run the training script with default parameters:

```bash
cd python_agent
python train.py
```

### Advanced Training Options

Train with custom parameters:

```bash
python train.py \
    --algorithm ppo \
    --total-timesteps 1000000 \
    --emulator-path "/usr/local/bin/bizhawk" \
    --rom-path "/path/to/pokemon_crystal.gbc" \
    --learning-rate 3e-4 \
    --n-envs 1 \
    --model-save-path "./models" \
    --log-path "./logs"
```

### Available Algorithms
- **PPO** (Proximal Policy Optimization) - Default, good balance of performance and stability
- **DQN** (Deep Q-Network) - Value-based method, good for discrete actions
- **A2C** (Advantage Actor-Critic) - Policy gradient method, faster but less stable

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--algorithm` | RL algorithm (ppo, dqn, a2c) | ppo |
| `--total-timesteps` | Total training steps | 1000000 |
| `--emulator-path` | Path to emulator executable | bizhawk |
| `--rom-path` | Path to Pokémon Crystal ROM | pokemon_crystal.gbc |
| `--learning-rate` | Learning rate | 3e-4 |
| `--n-envs` | Number of parallel environments | 1 |
| `--batch-size` | Training batch size | 64 |
| `--eval-freq` | Evaluation frequency | 10000 |
| `--save-freq` | Model saving frequency | 50000 |
| `--resume-from` | Resume from saved model | None |

### Monitoring Training

Training progress can be monitored using TensorBoard:

```bash
tensorboard --logdir logs
```

Open your browser to `http://localhost:6006` to view training metrics.

## How It Works

### Architecture Overview

1. **Lua Bridge**: 
   - Runs within the emulator
   - Reads game memory addresses to extract state
   - Receives actions from Python and executes them
   - Communicates via JSON files

2. **Python Environment**:
   - Wraps the game as an OpenAI Gym environment
   - Handles state preprocessing and reward calculation
   - Manages communication with the Lua bridge

3. **RL Agent**:
   - Uses Stable Baselines3 implementations
   - Trains on the preprocessed game state
   - Learns to maximize cumulative reward

### State Representation

The agent observes a 20-dimensional feature vector including:
- Player position (x, y coordinates)
- Current map/location
- Health points ratio
- Experience points and level
- Money and badge count
- Party information
- Battle state
- Menu/UI state

### Action Space

The agent can take 9 discrete actions:
- 0: No action
- 1: Up
- 2: Down  
- 3: Left
- 4: Right
- 5: A button
- 6: B button
- 7: Start
- 8: Select

### Reward Function

The reward function encourages:
- **Survival**: Small positive reward for staying alive
- **Level progression**: Large reward for gaining levels
- **Badge collection**: Very large reward for earning badges
- **Exploration**: Moderate reward for visiting new areas
- **Battle engagement**: Reward for entering and winning battles

Penalties for:
- **Taking damage**: Proportional to HP lost
- **Death**: Large negative reward
- **Time passage**: Small penalty to encourage efficiency

## Customization

### Modifying Rewards
Edit the `calculate_reward()` function in `utils.py` to change how the agent is rewarded for different behaviors.

### Adding State Features
1. Update memory addresses in `memory_map.py`
2. Modify `preprocess_state()` in `utils.py` to include new features
3. Update observation space dimensions in `env.py`

### Changing Memory Addresses
If using a different ROM version, update the addresses in `memory_map.py`. You can find addresses using:
- Cheat Engine
- BizHawk's memory viewer
- Community-maintained address lists

## Troubleshooting

### Common Issues

1. **Emulator not starting**:
   - Check emulator path is correct
   - Ensure ROM file exists and is readable
   - Try running emulator manually first

2. **Lua script errors**:
   - Check that BizHawk supports the Game Boy core
   - Verify JSON library is working
   - Look at BizHawk's Lua console for error messages

3. **Memory address issues**:
   - ROM version mismatch - try different addresses
   - Use memory viewer to verify correct values
   - Check endianness (Game Boy is little-endian)

4. **Training not progressing**:
   - Reduce learning rate
   - Check reward function is providing meaningful signals
   - Verify state preprocessing is working correctly

5. **Communication timeout**:
   - Increase timeout in `_wait_for_state()`
   - Check file permissions
   - Ensure Lua script is running continuously

### Getting Memory Addresses

If you need to find memory addresses for a different ROM:

1. Use BizHawk's hex editor and memory viewer
2. Search for known values (like player level, money)
3. Use save states to compare memory before/after changes
4. Consult community resources like Datacrystal.romhacking.net

## Performance Tips

### Training Performance
- Use GPU if available (set CUDA environment)
- Increase batch size if you have enough memory
- Use multiple parallel environments (`--n-envs`)
- Consider using A2C for faster training with less stability

### Stability Improvements
- Use PPO for most stable training
- Reduce learning rate if training is unstable
- Add reward clipping if rewards have high variance
- Use more frequent evaluation to catch issues early

## Contributing

Contributions are welcome! Areas for improvement:

- **Better reward functions**: More sophisticated reward shaping
- **Additional state features**: Include more game state information
- **Visual observations**: Support for pixel-based observations
- **Multi-objective learning**: Balance multiple goals (speed vs. completion)
- **Memory address verification**: Confirm addresses for different ROM versions

## Legal Notice

This project is for educational and research purposes. You must legally own a copy of Pokémon Crystal to use this software. The ROM file is not included and must be obtained separately.

## References

- [Stable Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [OpenAI Gym](https://gym.openai.com/)
- [BizHawk Emulator](https://github.com/TASvideos/BizHawk)
- [Pokémon Crystal Memory Map](https://datacrystal.romhacking.net/wiki/Pok%C3%A9mon_Crystal)

## License

This project is released under the MIT License. See individual dependencies for their respective licenses.
