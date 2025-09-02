# 🎮 Pokemon Crystal RL Training Platform

**An advanced reinforcement learning environment for Pokemon Crystal featuring hybrid LLM-RL training, intelligent decision making, comprehensive memory mapping, and real-time web monitoring.**

## 🌟 Features

### 🤖 **Hybrid LLM-RL Training System**
- **Intelligent Architecture**: Combines LLM strategic guidance with RL optimization
- **Curriculum Learning**: Progressive transition from LLM-heavy to RL-optimized decisions
- **Adaptive Strategies**: Dynamic switching between decision-making approaches based on performance
- **Decision Pattern Learning**: Learns from decision history with SQLite-backed persistence
- **Multi-Modal Observations**: Screen capture, state variables, and strategic context integration
- **Action Masking**: Prevents invalid moves based on current game state

### 🎮 **Advanced Game Integration**
- **PyBoy Emulation**: Full Pokemon Crystal emulation with memory access
- **Smart Screen Analysis**: Multi-metric state detection (overworld, battle, dialogue, menu)
- **Memory Mapping**: Detailed game state extraction (HP, level, badges, party, etc.)
- **Save State Support**: Resume training from specific game positions

### 💰 **Sophisticated Reward System**
- **Multi-Factor Rewards**: Health, leveling, badges, money, exploration, battles
- **Early Game Focus**: Special rewards for getting first Pokemon (+100 points!)
- **Progressive Scaling**: Bigger rewards for major milestones (badges = +500)
- **Smart Health Logic**: Only applies health rewards when player has Pokemon

### 🌐 **Real-Time Monitoring**
- **Live Web Dashboard**: Beautiful interface at http://localhost:8080
- **Game Screen Capture**: Real-time visual monitoring
- **LLM Decision Display**: See AI reasoning for each decision
- **Comprehensive Stats**: Performance metrics, rewards, game progress

## 🚀 Quick Start

### 1. **Setup Environment**
```bash
pip install -r requirements.txt
```

### 2. **Install Ollama (for LLM features)**
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull a model
ollama pull smollm2:1.7b
```

### 3. **Place ROM File**
```bash
mkdir roms
# Place pokemon_crystal.gbc in the roms/ directory
```

### 4. **Run Hybrid LLM-RL Training**
```bash
# Basic hybrid training
python3 examples/run_hybrid_training.py

# Advanced hybrid training with custom configuration
python3 -c "
from trainer.hybrid_llm_rl_trainer import create_trainer_from_config
trainer = create_trainer_from_config('hybrid_training_config.json')
trainer.train(total_episodes=1000, max_steps_per_episode=10000)
"

# Legacy LLM-only training (still supported)
python3 llm_trainer.py --rom roms/pokemon_crystal.gbc --actions 2000
```

### 5. **Monitor Training**
- Visit **http://localhost:8080** for the web dashboard
- Watch real-time LLM decisions and game progress
- View detailed reward breakdowns and statistics

## 🏗️ Architecture

### **Core Components**

#### 🤖 **Hybrid Training System** (`trainer/hybrid_llm_rl_trainer.py`)
- **HybridLLMRLTrainer**: Main training orchestrator with curriculum learning
- **AdaptiveStrategySystem**: Performance-based strategy switching (630 lines)
- **DecisionHistoryAnalyzer**: Pattern learning with SQLite persistence (677 lines)
- **HybridAgent**: Combines LLM and RL agents with decision arbitration (651 lines)
- **EnhancedPyBoyPokemonCrystalEnv**: Multi-modal Gymnasium environment (599 lines)

#### 🧠 **Legacy LLM Integration** (`llm_trainer.py`)
- **LLMAgent**: Handles communication with Ollama
- **Context Building**: Creates rich prompts with game state
- **Decision Parsing**: Extracts valid actions from LLM responses
- **Fallback Logic**: Smart rule-based decisions when LLM unavailable

#### 💰 **Advanced Rewards** (`PokemonRewardCalculator`)
- **Health Rewards**: Only when player has Pokemon (fixed bug!)
- **Progression Rewards**: +100 for first Pokemon, +25 for additional
- **Battle Rewards**: +20 for victories, +2 for engagement
- **Exploration Rewards**: +10 for new maps, +0.1 for movement
- **Badge Rewards**: +500 per badge (major milestones)
- **Level Rewards**: +50 per level gained

#### 🌐 **Web Monitoring** (`WebMonitor`)
- **Real-time Dashboard**: Live stats and game screen
- **LLM Decision Tracking**: Recent decisions with reasoning
- **Reward Visualization**: Color-coded reward categories
- **Performance Metrics**: Actions/sec, total rewards, progress

#### 🗺️ **Memory Mapping** (`core/memory_map.py`)
- **Comprehensive State**: 25+ memory addresses mapped
- **Derived Values**: Calculated stats and battle state
- **Badge System**: Full Johto + Kanto badge tracking
- **Pokemon Data**: Party, levels, HP, species

### **Available Models**

Supported LLM models via Ollama:
- **`smollm2:1.7b`** (recommended): Fast, good reasoning
- **`llama3.2:1b`**: Lightweight alternative
- **`llama3.2:3b`**: More sophisticated reasoning
- **`deepseek-coder:latest`**: Code-focused model

## 📊 Training Configuration

### **Hybrid LLM-RL Training Options**

```bash
# Create configuration file
cat > hybrid_training_config.json << 'EOF'
{
  "rom_path": "roms/pokemon_crystal.gbc",
  "headless": true,
  "observation_type": "multi_modal",
  "llm_model": "gpt-4",
  "max_context_length": 8000,
  "initial_strategy": "llm_heavy",
  "decision_db_path": "pokemon_decisions.db",
  "save_dir": "training_checkpoints",
  "log_level": "INFO"
}
EOF

# Run hybrid training
python3 examples/run_hybrid_training.py
```

### **Legacy LLM-Enhanced Training Options**

```bash
# Standard configuration
python3 llm_trainer.py \
    --rom roms/pokemon_crystal.gbc \
    --actions 3000 \
    --llm-interval 15 \
    --web-port 8080

# High-intelligence training
python3 llm_trainer.py \
    --rom roms/pokemon_crystal.gbc \
    --actions 5000 \
    --llm-model llama3.2:3b \
    --llm-interval 10 \
    --web-port 8080

# Fast training (more rule-based)
python3 llm_trainer.py \
    --rom roms/pokemon_crystal.gbc \
    --actions 2000 \
    --llm-interval 25 \
    --web-port 8080
```

### **Command Line Options**

| Option | Description | Default |
|--------|-------------|---------|
| `--rom` | ROM file path | `roms/pokemon_crystal.gbc` |
| `--actions` | Number of actions to execute | `2000` |
| `--llm-model` | LLM model to use | `smollm2:1.7b` |
| `--llm-interval` | Actions between LLM decisions | `20` |
| `--web-port` | Web monitoring port | `8080` |
| `--no-web` | Disable web monitoring | `False` |

## 🎯 Key Improvements

### **✅ Fixed Reward System**
- **Bug Fix**: Eliminated incorrect health penalties in early game
- **Smart Logic**: Health rewards only apply when player has Pokemon
- **Progression Focus**: Strong incentives for getting first Pokemon

### **🤖 LLM Intelligence**
- **Contextual Decisions**: AI understands game state and recent actions
- **Screen Awareness**: Responds appropriately to dialogue, battle, overworld
- **Strategic Thinking**: Considers long-term goals and immediate needs

### **📈 Advanced Analytics**
- **Decision Tracking**: Complete LLM reasoning history
- **Reward Breakdown**: Detailed category-wise reward analysis
- **Performance Monitoring**: Real-time stats and progress tracking

## 📁 Project Structure

```
pokemon_crystal_rl/
├── examples/
│   └── run_hybrid_training.py  # 🤖 Hybrid training example (MAIN)
├── trainer/
│   ├── hybrid_llm_rl_trainer.py # 🧠 Hybrid LLM-RL trainer (455 lines)
│   └── llm_manager.py          # 💬 LLM communication manager
├── core/
│   ├── hybrid_agent.py         # 🤝 LLM+RL decision arbitration (651 lines)
│   ├── adaptive_strategy_system.py # 📊 Performance-based strategies (630 lines)
│   ├── decision_history_analyzer.py # 🧠 Pattern learning (677 lines)
│   ├── enhanced_pyboy_env.py   # 🎮 Multi-modal environment (599 lines)
│   ├── goal_oriented_planner.py # 🎯 Strategic goal planning
│   ├── state_variable_dictionary.py # 📊 Comprehensive state mapping
│   └── memory_map.py           # 🗺️ Memory address definitions
├── tests/
│   ├── trainer/                # 🧪 Trainer tests (13 test methods)
│   ├── integration/            # 🔗 Integration tests
│   └── core/                   # ⚙️ Core component tests
├── llm_trainer.py              # 🤖 Legacy LLM trainer
├── roms/                       # 💾 ROM files (not included)
├── requirements.txt            # 📦 Dependencies
└── README.md                   # 📖 This documentation
```

## 🛠️ Development

### **Adding New Memory Addresses**
```python
# Edit core/memory_map.py
MEMORY_ADDRESSES = {
    'new_stat': 0xD123,      # Add new memory location
    # ... existing addresses
}

# Add derived calculation
DERIVED_VALUES = {
    'custom_metric': lambda state: state['new_stat'] * 2,
}
```

### **Customizing LLM Prompts**
```python
# Edit _build_prompt() in llm_trainer.py
def _build_prompt(self, game_state, screen_analysis, recent_actions):
    # Customize prompt content
    prompt = f"""Custom instructions for Pokemon Crystal AI...
    Current status: {game_state}
    Your goal: [custom objective]
    """
    return prompt
```

### **Modifying Reward Functions**
```python
# Edit PokemonRewardCalculator in llm_trainer.py
def _calculate_custom_reward(self, current, previous):
    # Add custom reward logic
    if current['custom_condition']:
        return 50.0  # Custom reward
    return 0.0
```

## 📋 Requirements

### **System Requirements**
- Python 3.8+
- 4GB+ RAM (for LLM models)
- Legal Pokemon Crystal ROM file

### **Python Dependencies**
```bash
pip install pyboy numpy pillow requests
```

### **LLM Requirements (Optional)**
- Ollama installed locally
- At least one model downloaded (smollm2:1.7b recommended)

## 🎮 Usage Examples

### **Creating Save States**
To skip the intro and start training from a better position:

```bash
python3 -c "
from pyboy import PyBoy
import time

print('🎮 Launching PyBoy...')
print('Play through intro, then press Ctrl+C to save state')

pyboy = PyBoy('roms/pokemon_crystal.gbc', window='SDL2')

try:
    while True:
        pyboy.tick()
        time.sleep(0.016)  # ~60 FPS
except KeyboardInterrupt:
    print('💾 Saving state...')
    with open('roms/pokemon_crystal.gbc.state', 'wb') as f:
        pyboy.save_state(f)
    print('✅ Save state created!')
finally:
    pyboy.stop()
"
```

### **Monitoring Training Progress**
1. Start training: `python3 llm_trainer.py --rom roms/pokemon_crystal.gbc --actions 3000`
2. Open browser: http://localhost:8080
3. Watch live game screen and AI decisions
4. Monitor reward progression and statistics

### **Analyzing Training Data**
```python
import json

# Load training statistics
with open('llm_training_stats_20240831_123456.json', 'r') as f:
    stats = json.load(f)

print(f"Total reward: {stats['total_reward']}")
print(f"Actions taken: {stats['actions_taken']}")
print(f"LLM decisions: {stats['llm_decision_count']}")

# Load LLM decision history
with open('llm_decisions_20240831_123456.json', 'r') as f:
    decisions = json.load(f)

# Analyze decision patterns
for decision in decisions[-5:]:  # Last 5 decisions
    print(f"Action: {decision['action']}")
    print(f"Reasoning: {decision['reasoning'][:100]}...")
    print("---")
```

## 🏆 Training Results

Expected progression with LLM trainer:

1. **Early Game** (-0.01/action): Time penalties only, no false health penalties
2. **First Pokemon** (+100): Massive reward for getting starter
3. **Level Progression** (+50/level): Steady rewards for Pokemon growth
4. **Badge Collection** (+500/badge): Major milestone rewards
5. **Battle Mastery** (+20/victory): Combat skill development

## 📊 Performance Metrics

### **Typical Performance**
- **Speed**: ~24 actions/second
- **LLM Response**: ~100-500ms per decision
- **Memory Usage**: ~2GB with smollm2:1.7b
- **Screen Analysis**: Multi-metric detection (variance, brightness, colors)

### **Web Dashboard Metrics**
- Real-time game screen (320x288 scaled)
- Actions per second
- Total reward progression
- LLM decision count and reasoning
- Game state (level, badges, party, money)
- Reward breakdown by category

## 🔧 Troubleshooting

### **Common Issues**

**LLM not available**: 
```bash
# Check Ollama is running
ollama list
ollama pull smollm2:1.7b
```

**ROM not found**:
```bash
# Ensure ROM is in correct location
ls roms/pokemon_crystal.gbc
```

**Web interface not loading**:
- Check port 8080 is available
- Try different port: `--web-port 8081`

**Poor performance**:
- Reduce LLM frequency: `--llm-interval 30`
- Use smaller model: `--llm-model llama3.2:1b`

## ⚖️ Legal Notice

This project is for **educational and research purposes**. You must own a legal copy of Pokemon Crystal to use this software. ROM files are not included and must be obtained legally.

## 🤝 Contributing

Contributions welcome! Areas of interest:
- New LLM models and prompt engineering
- Advanced reward function design
- Memory mapping improvements
- Web dashboard enhancements

Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request with detailed description

## 📜 License

MIT License - see LICENSE file for details.

---

**Built with 💚 for Pokemon Crystal RL training**
