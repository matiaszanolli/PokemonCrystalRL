# 📝 Changelog

All notable changes to the Pokemon Crystal RL Training Platform.

## [v2.0.0] - 2025-08-31 - LLM Enhancement Release

### 🚀 **Major Features Added**

#### 🤖 **LLM Integration**
- **Added `llm_trainer.py`**: Complete LLM-enhanced training system
- **Ollama Integration**: Local LLM support with multiple models
- **Context-Aware Prompts**: Rich game state information in prompts
- **Decision Tracking**: Full LLM reasoning history and analysis
- **Hybrid Decision Making**: LLM + rule-based fallback system

#### 🌐 **Enhanced Web Monitoring**
- **Real-time LLM Decisions**: See AI reasoning for each action
- **Improved Dashboard**: Better styling and more comprehensive stats
- **Live Game Screen**: Real-time screenshot capture and display
- **Detailed Metrics**: Performance, rewards, and progress tracking

### 🐛 **Critical Bug Fixes**

#### ✅ **Reward System Overhaul**
- **Fixed Health Penalty Bug**: Eliminated incorrect -0.50 health penalties in early game
- **Smart Health Logic**: Health rewards only apply when player has Pokemon (`party_count > 0`)
- **Early Game Progression**: Added +100 reward for getting first Pokemon
- **Balanced Penalties**: Only -0.01 time penalty per action in early game

#### 🎯 **Screen State Detection**
- **Improved Classification**: Better overworld vs dialogue detection
- **Multi-Metric Analysis**: Uses variance, brightness, and color count
- **Fixed False Positives**: No more incorrect dialogue detection in overworld

### 🔧 **Technical Improvements**

#### 📊 **Advanced Analytics**
- **Comprehensive Logging**: Detailed training statistics and decision history
- **JSON Export**: Complete training data saved for analysis
- **Performance Metrics**: Actions/second, LLM response times, memory usage
- **Reward Breakdown**: Category-wise reward visualization

#### 🎮 **Game Integration**
- **Save State Support**: Automatic save/load state management
- **Memory Optimization**: More efficient game state extraction
- **Error Handling**: Robust error handling and graceful degradation

### 📈 **Performance Improvements**

#### ⚡ **Speed Optimizations**
- **24+ Actions/Second**: Optimized execution speed
- **Efficient LLM Calls**: Smart interval-based LLM decision making
- **Reduced Overhead**: Streamlined memory access and state calculation

#### 🧠 **LLM Performance**
- **Fast Inference**: 100-500ms response times with `smollm2:1.7b`
- **Smart Caching**: Efficient prompt building and response parsing
- **Fallback System**: Seamless fallback when LLM unavailable

### 🎯 **Reward System Details**

#### 💰 **New Reward Categories**
| Category | Trigger | Reward | Description |
|----------|---------|--------|-------------|
| **Progression** | First Pokemon | +100 | Huge early game milestone |
| **Progression** | Additional Pokemon | +25 | Party expansion |
| **Badges** | Gym Badge | +500 | Major game milestone |
| **Level** | Pokemon Level Up | +50/level | Character progression |
| **Battle** | Battle Victory | +20 | Combat success |
| **Battle** | Battle Engagement | +2 | Combat participation |
| **Exploration** | New Map | +10 | Area discovery |
| **Exploration** | Movement | +0.1/tile | Exploration encouragement |
| **Money** | Earning Money | +0.01/¥ | Economic progress |
| **Health** | Healing | +5.0/% | Health management (Pokemon only) |
| **Health** | Damage | -10.0/% | Health penalty (Pokemon only) |
| **Time** | Each Action | -0.01 | Efficiency encouragement |

#### 🎯 **Reward Logic**
- **Early Game**: Only time penalties (-0.01/action) until first Pokemon
- **Mid Game**: Health and battle rewards activate after getting Pokemon  
- **Late Game**: Badge and exploration rewards for advanced progress

### 📁 **New Files Added**

- **`llm_trainer.py`**: Main LLM-enhanced training script
- **`llm_training_stats_*.json`**: Training statistics files
- **`llm_decisions_*.json`**: LLM decision history files
- **`CHANGELOG.md`**: This changelog file
- **Updated `README.md`**: Comprehensive documentation

### 🔄 **Breaking Changes**

- **New Main Script**: Use `llm_trainer.py` instead of `enhanced_trainer.py` for best results
- **Different Reward Scale**: Rewards now properly balanced (no false health penalties)
- **New Command Line Options**: LLM-specific options added (`--llm-model`, `--llm-interval`)

### 🛠️ **Migration Guide**

#### From Enhanced Trainer to LLM Trainer
```bash
# Old way
python3 enhanced_trainer.py --rom roms/pokemon_crystal.gbc --actions 1000

# New way  
python3 llm_trainer.py --rom roms/pokemon_crystal.gbc --actions 1000
```

#### Updated Dependencies
```bash
# Install new dependencies
pip install requests  # For LLM communication

# Install Ollama (optional)
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull smollm2:1.7b
```

---

## [v1.0.0] - 2025-08-30 - Initial Release

### ✅ **Initial Features**
- **PyBoy Integration**: Pokemon Crystal emulation
- **Memory Mapping**: Basic game state extraction
- **Enhanced Trainer**: Rule-based training system
- **Web Monitoring**: Basic web dashboard
- **Reward System**: Initial reward calculation

### 🐛 **Known Issues (Fixed in v2.0.0)**
- Health penalties applied incorrectly in early game
- Screen state detection issues
- Limited decision intelligence

---

**Legend:**
- 🚀 Major Features
- 🐛 Bug Fixes  
- 🔧 Technical Improvements
- 📈 Performance Improvements
- 🔄 Breaking Changes
