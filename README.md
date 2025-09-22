# 🎮 Pokemon Crystal RL Training Platform

**An advanced reinforcement learning environment for Pokemon Crystal featuring hybrid LLM-RL training, intelligent decision making, comprehensive memory mapping, and real-time web monitoring. Now with unified single entry point architecture and enhanced LLM intelligence.**

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
- **Integrated Architecture**: Web monitoring built directly into training system

### 🧠 **Enhanced LLM Intelligence** ⭐ **NEW**
- **Advanced Response Parsing**: Natural language synonyms (north/up, attack/a, flee/b)
- **Stuck Pattern Detection**: Automatically detects and breaks repetitive loops
- **Strategic Context Display**: Shows game phase, threats, and opportunities
- **Phase-Aware Decisions**: Different strategies for early game vs exploration
- **Comprehensive Logging**: Detailed decision logs with strategic context

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

### 4. **Run Training - Single Entry Point** ⭐ **SIMPLIFIED**
```bash
# UNIFIED ENTRY POINT - All features in one command!
python3 llm_trainer.py roms/pokemon_crystal.gbc --max-actions 2000

# Enhanced training with all new features
python3 llm_trainer.py roms/pokemon_crystal.gbc \
    --max-actions 3000 \
    --llm-model smollm2:1.7b \
    --llm-interval 15 \
    --web-port 8080

# Advanced hybrid training (coming soon)
python3 examples/run_hybrid_training.py

# Quick test run
python3 llm_trainer.py roms/pokemon_crystal.gbc --max-actions 500
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

#### 🧠 **Enhanced LLM Integration** (`llm_trainer.py`) ⭐ **CONSOLIDATED**
- **Single Entry Point**: Unified training system with all features
- **Enhanced LLMAgent**: Advanced communication with Ollama + smart parsing
- **Strategic Context Building**: Rich prompts with game state and strategic analysis  
- **Intelligent Fallback Logic**: Stuck detection, phase-aware decisions
- **Integrated Web Monitoring**: Built-in real-time dashboard

#### 💰 **Advanced Rewards** (`PokemonRewardCalculator`)
- **Health Rewards**: Only when player has Pokemon (fixed bug!)
- **Progression Rewards**: +100 for first Pokemon, +25 for additional
- **Battle Rewards**: +20 for victories, +2 for engagement
- **Exploration Rewards**: +10 for new maps, +0.1 for movement
- **Badge Rewards**: +500 per badge (major milestones)
- **Level Rewards**: +50 per level gained

#### 🌐 **Consolidated Web Monitoring** (`core/web_monitor.py`) ⭐ **ENHANCED**
- **Integrated Dashboard**: Live stats and game screen built into trainer
- **Advanced LLM Decision Tracking**: Recent decisions with strategic reasoning
- **Real-time Screen Capture**: Threaded screen streaming with error recovery
- **Enhanced Performance Metrics**: Actions/sec, threats, opportunities, phase info
- **Single Port Architecture**: Everything accessible at http://localhost:8080

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

### **Enhanced LLM Training Options** ⭐ **UPDATED**

```bash
# Standard configuration with all enhanced features
python3 llm_trainer.py roms/pokemon_crystal.gbc \
    --max-actions 3000 \
    --llm-interval 15 \
    --web-port 8080

# High-intelligence training with advanced parsing
python3 llm_trainer.py roms/pokemon_crystal.gbc \
    --max-actions 5000 \
    --llm-model llama3.2:3b \
    --llm-interval 10 \
    --web-port 8080

# Fast training with smart fallbacks
python3 llm_trainer.py roms/pokemon_crystal.gbc \
    --max-actions 2000 \
    --llm-interval 25 \
    --web-port 8080

# Debug mode with enhanced logging
python3 llm_trainer.py roms/pokemon_crystal.gbc \
    --max-actions 1000 \
    --debug \
    --web-port 8080
```

### **Command Line Options** ⭐ **ENHANCED**

| Option | Description | Default | New Features |
|--------|-------------|---------|-------------|
| `rom_path` | ROM file path (positional) | Required | ✅ Simplified syntax |
| `--max-actions` | Number of actions to execute | `10000` | ✅ Updated default |
| `--llm-model` | LLM model to use | `smollm2:1.7b` | ✅ Enhanced parsing |
| `--llm-interval` | Actions between LLM decisions | `20` | ✅ Smart fallbacks |
| `--web-port` | Web monitoring port | `8080` | ✅ Integrated dashboard |
| `--headless` | Run without visual output | `False` | ✅ Performance mode |
| `--debug` | Enable debug mode | `False` | ✅ Enhanced logging |
| `--no-dqn` | Disable DQN (LLM-only mode) | `False` | ✅ Hybrid training |

## 🎯 Key Improvements

### **🏗️ Consolidated Architecture** ⭐ **NEW**
- **Single Entry Point**: `llm_trainer.py` now handles ALL functionality
- **Unified Web Monitoring**: Built-in dashboard eliminates separate server setup
- **Enhanced LLM Intelligence**: Advanced parsing, stuck detection, strategic context
- **Simplified Usage**: One command for complete training with all features
- **Reduced Complexity**: Eliminated redundant training implementations

### **✅ Fixed Reward System**
- **Bug Fix**: Eliminated incorrect health penalties in early game
- **Smart Logic**: Health rewards only apply when player has Pokemon
- **Progression Focus**: Strong incentives for getting first Pokemon

### **🧠 Enhanced LLM Intelligence** ⭐ **NEW**
- **Natural Language Parsing**: Understands synonyms (north/up, attack/a, flee/b)
- **Stuck Pattern Breaking**: Automatically detects and resolves repetitive loops
- **Strategic Context**: Shows game phase, immediate threats, and opportunities
- **Phase-Aware Decisions**: Different strategies for early game vs exploration phases
- **Comprehensive Logging**: Detailed decision logs with strategic analysis

### **📈 Advanced Analytics**
- **Enhanced Decision Tracking**: Complete LLM reasoning with strategic context
- **Performance Metrics**: Comprehensive session analysis and export
- **Real-time Strategic Info**: Live threat/opportunity detection
- **Detailed Summaries**: JSON exports for training analysis

## 📁 Project Structure

```
pokemon_crystal_rl/
├── llm_trainer.py              # 🤖 **MAIN ENTRY POINT** - Enhanced unified trainer
├── core/
│   ├── web_monitor.py          # 🌐 **NEW** - Consolidated web monitoring system
│   ├── hybrid_agent.py         # 🤝 LLM+RL decision arbitration (651 lines)
│   ├── adaptive_strategy_system.py # 📊 Performance-based strategies (630 lines)
│   ├── decision_history_analyzer.py # 🧠 Pattern learning (677 lines)
│   ├── enhanced_pyboy_env.py   # 🎮 Multi-modal environment (599 lines)
│   ├── goal_oriented_planner.py # 🎯 Strategic goal planning
│   ├── state_variable_dictionary.py # 📊 Comprehensive state mapping
│   └── memory_map.py           # 🗺️ Memory address definitions
├── examples/
│   └── run_hybrid_training.py  # 🤖 Hybrid training example
├── trainer/
│   ├── hybrid_llm_rl_trainer.py # 🧠 Hybrid LLM-RL trainer (455 lines)
│   └── llm_manager.py          # 💬 LLM communication manager
├── tests/
│   ├── trainer/                # 🧪 Trainer tests (13 test methods)
│   ├── integration/            # 🔗 Integration tests
│   └── core/                   # ⚙️ Core component tests
├── roms/                       # 💾 ROM files (not included)
├── requirements.txt            # 📦 Dependencies
├── CONSOLIDATED_WEB_UI.md      # 📋 Web UI consolidation summary
├── SMART_TRAINER_MERGE.md      # 📋 Smart trainer merge summary
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

## 📋 Changelog

### **v2.0.0 - Consolidated Architecture** ⭐ **LATEST**
- **🏗️ Single Entry Point**: `llm_trainer.py` now handles all functionality
- **🌐 Integrated Web Monitoring**: Built-in dashboard eliminates separate server setup
- **🧠 Enhanced LLM Intelligence**: Advanced parsing with natural language synonyms
- **🎯 Smart Stuck Detection**: Automatic pattern breaking and recovery
- **📊 Strategic Context Display**: Live threats, opportunities, and phase information
- **📈 Comprehensive Logging**: Enhanced decision logs with strategic analysis
- **🗑️ Code Cleanup**: Removed redundant training implementations
- **✅ Unified Testing**: Single command for complete training experience

### **v1.x - Previous Features**
- Hybrid LLM-RL training system
- Advanced reward calculation
- Memory mapping integration
- Basic web monitoring
- PyBoy emulation support

## 🤝 Contributing

Contributions welcome! Areas of interest:
- Enhanced LLM intelligence and reasoning
- Advanced strategic decision making
- Memory mapping improvements  
- Web dashboard feature additions
- Performance optimizations

Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request with detailed description

## 📜 License

MIT License - see LICENSE file for details.

---

**Built with 💚 for Pokemon Crystal RL training**
