# 🎮 Pokemon Crystal Vision-Enhanced LLM Training System

## Overview

This is a complete reinforcement learning training system for Pokemon Crystal that combines:

- **🎯 PyBoy Emulator**: Clean, headless Game Boy Color emulation
- **🤖 Local LLM Agent**: Strategic decision making using Ollama + Llama 3.2
- **👁️ Computer Vision**: Screenshot analysis with OCR and UI detection  
- **💾 Episodic Memory**: SQLite-based learning from past decisions
- **📊 Training Pipeline**: Complete training loop with analytics

## Key Features

### ✅ No API Costs
- Runs entirely locally using Ollama
- No OpenAI API calls required
- Uses lightweight Llama 3.2 3B model

### ✅ Visual Understanding
- Real-time screenshot analysis
- Text detection using EasyOCR
- UI element recognition (health bars, menus, dialogue boxes)
- Game phase detection (battle, overworld, dialogue, menu)

### ✅ Strategic Intelligence
- Context-aware decision making
- Memory of past actions and outcomes
- Visual cues integrated with game state data
- Pokemon type effectiveness knowledge

### ✅ Complete Training Pipeline
- Episode-based training sessions
- Progress tracking (badges, Pokemon caught, money)
- Detailed analytics and reporting
- Screenshot analysis saving

## System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   PyBoy Env     │    │ Vision Processor │    │ Enhanced Agent  │
│                 │    │                 │    │                 │
│ • ROM loading   │◄──►│ • Screenshot    │◄──►│ • Local LLM     │
│ • Save states   │    │ • OCR detection │    │ • Memory system │
│ • Game state    │    │ • UI analysis   │    │ • Strategy      │
│ • Actions       │    │ • Context       │    │ • Decision      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                        │                        │
         └────────────────────────┼────────────────────────┘
                                  │
                    ┌─────────────────┐
                    │ Training System │
                    │                 │
                    │ • Episode loop  │
                    │ • Progress      │
                    │ • Analytics     │
                    │ • Reporting     │
                    └─────────────────┘
```

## Files Overview

### Core Components

- **`pyboy_env.py`** - Pokemon Crystal PyBoy environment with Gym interface
- **`vision_processor.py`** - Computer vision module for screenshot analysis  
- **`enhanced_llm_agent.py`** - Local LLM agent with visual context integration
- **`vision_enhanced_training.py`** - Complete training pipeline

### Legacy Files

- **`local_llm_agent.py`** - Original text-only LLM agent
- **`pokemon_crystal_train.py`** - Original training script

## Installation

### Prerequisites

```bash
# Install Python dependencies
pip install numpy opencv-python easyocr pillow ollama pyboy stable-baselines3 matplotlib

# Install and start Ollama
curl -fsSL https://ollama.ai/install.sh | sh
ollama serve

# Pull the Llama model
ollama pull llama3.2:3b
```

### Game Files

You need to provide your own Pokemon Crystal ROM file:

```bash
mkdir -p /path/to/project/roms/
# Place your pokemon_crystal.gbc file in the roms/ directory
```

### Optional: Save States

Create save states to skip the intro:

```bash
mkdir -p /path/to/project/save_states/
# Use PyBoy or other tools to create save states
```

## Usage

### Quick Test

Test individual components:

```python
# Test vision processor
python vision_processor.py

# Test enhanced LLM agent  
python enhanced_llm_agent.py
```

### Training Session

Run a complete training session:

```python
python vision_enhanced_training.py
```

### Custom Training

```python
from vision_enhanced_training import VisionEnhancedTrainingSession

# Initialize training
session = VisionEnhancedTrainingSession(
    rom_path="roms/pokemon_crystal.gbc",
    save_state_path="save_states/intro_done.state",  # Optional
    model_name="llama3.2:3b",
    max_steps_per_episode=3000,
    screenshot_interval=10  # Visual analysis every 10 steps
)

# Run training
session.run_training_session(num_episodes=10)

# Save screenshot analysis
session.save_screenshot_analysis()
```

## Training Output

### Console Logs

```
🎮 Initializing PyBoy environment...
🤖 Initializing Enhanced LLM Agent...
👁️ Computer vision enabled
✅ Training session initialized

🚀 Starting Episode 1
👁️ Visual: Screen: overworld | UI: | Colors: green, bright
🎯 Decided action with vision: 1 (UP)
  📊 Step 100: {'step': 100, 'location': 'Map 1 (5, 8)', 'money': 3000, 'badges': 0, 'party_size': 1, 'last_action': 'A'}
🏆 PROGRESS: Caught new Pokemon! Party size: 2
```

### Generated Files

- **`training_report_YYYYMMDD_HHMMSS.json`** - Detailed training analytics
- **`screenshot_analysis/`** - Screenshots with visual analysis JSON files  
- **`pokemon_agent_memory.db`** - SQLite database with decision history

## Configuration Options

### Training Parameters

```python
VisionEnhancedTrainingSession(
    rom_path="roms/pokemon_crystal.gbc",         # Required
    save_state_path="save_states/game.state",   # Optional
    model_name="llama3.2:3b",                   # Ollama model
    max_steps_per_episode=5000,                 # Episode length
    log_interval=100,                           # Progress logging
    screenshot_interval=10                      # Visual analysis frequency
)
```

### Vision Processing

```python
EnhancedLLMPokemonAgent(
    model_name="llama3.2:3b",      # LLM model
    memory_db="memory.db",         # SQLite database  
    use_vision=True                # Enable/disable vision
)
```

## Example Training Report

```json
{
  "session_info": {
    "duration": 1234.5,
    "episodes": 5,
    "total_steps": 12500,
    "model_used": "llama3.2:3b"
  },
  "training_stats": {
    "decisions_made": 12500,
    "visual_analyses": 1250,
    "last_progress": {
      "badges": 1,
      "money": 15000,
      "party_size": 3,
      "location": [10, 15, 5]
    }
  },
  "memory_summary": {
    "decisions_stored": 12500,
    "visual_analyses": 1250,
    "screen_type_breakdown": {
      "overworld": 800,
      "battle": 200,
      "dialogue": 150,
      "menu": 100
    }
  }
}
```

## Computer Vision Analysis

### Detected Screen Types

- **`overworld`** - Normal gameplay exploration
- **`battle`** - Pokemon battles with health bars
- **`dialogue`** - Conversations with NPCs  
- **`menu`** - Game menus and interfaces
- **`intro`** - Title screens and intro sequences

### Visual Context Features

- **Text Detection**: OCR extraction of on-screen text
- **UI Elements**: Health bars, dialogue boxes, menus
- **Color Analysis**: Dominant colors for environment detection
- **Location Classification**: Indoor/outdoor, grass/water areas

## Memory System

### Database Schema

```sql
-- Strategic decisions with visual context
CREATE TABLE strategic_decisions (
    id INTEGER PRIMARY KEY,
    timestamp TEXT,
    situation TEXT,      -- Game state analysis
    decision TEXT,       -- Action taken
    reasoning TEXT,      -- LLM reasoning
    visual_context TEXT, -- Screenshot analysis
    confidence_score REAL
);

-- Game state snapshots
CREATE TABLE game_states (
    id INTEGER PRIMARY KEY,
    timestamp TEXT,
    player_x INTEGER,
    player_y INTEGER,
    player_map INTEGER,
    party_size INTEGER,
    money INTEGER,
    badges INTEGER,
    visual_summary TEXT,
    screen_type TEXT
);

-- Visual analysis results
CREATE TABLE visual_analysis (
    id INTEGER PRIMARY KEY,
    timestamp TEXT,
    screen_type TEXT,
    game_phase TEXT,
    detected_text TEXT,
    ui_elements TEXT,
    dominant_colors TEXT,
    visual_summary TEXT
);
```

## Performance Considerations

### System Requirements

- **RAM**: 4GB+ recommended (EasyOCR model loading)
- **CPU**: Multi-core recommended for vision processing
- **GPU**: Optional, can accelerate EasyOCR if available
- **Storage**: ~1GB for models and game files

### Optimization Settings

```python
# Faster training with less vision processing
screenshot_interval=20  # Analyze every 20 steps instead of 10

# Shorter episodes for rapid iteration  
max_steps_per_episode=2000

# Smaller LLM for faster responses
model_name="llama3.2:1b"  # Lighter model
```

## Troubleshooting

### Common Issues

**Vision processor fails to initialize**
```bash
# Install required packages
pip install easyocr opencv-python pillow
```

**Ollama connection errors**
```bash
# Start Ollama service
ollama serve

# Verify model is available
ollama list
```

**ROM file not found**
```
Place your Pokemon Crystal ROM file in the roms/ directory
```

**Memory database errors**  
The system automatically handles database schema updates for backward compatibility.

## Contributing

1. Fork the repository
2. Create feature branches for new functionality
3. Add tests for new components
4. Submit pull requests with detailed descriptions

## License

This project is for educational and research purposes. Ensure you own legal copies of any ROM files used.

---

**Happy Training! 🎉**

For questions or issues, please check the troubleshooting section or create an issue in the repository.
