# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Pokemon Crystal reinforcement learning platform that combines LLM-based decision making with traditional RL training. The system uses PyBoy emulation to train AI agents to play Pokemon Crystal, featuring hybrid LLM-RL training, memory mapping, and real-time web monitoring.

## Core Architecture

### Primary Entry Points
- **`main.py`** - Main entry point for training (replaces deprecated `llm_trainer.py`)
- **`examples/run_hybrid_training.py`** - Hybrid LLM-RL training example
- **`quick_start.sh`** - Quick start script for monitoring system

⚠️ **IMPORTANT**: `llm_trainer.py` is deprecated and shows a deprecation warning. Always use `main.py` instead.

### Key Components
- **`core/`** - Core systems (memory mapping, web monitoring, game intelligence, reward calculation)
- **`trainer/`** - Training systems and LLM integration
- **`agents/`** - LLM and DQN agent implementations
- **`environments/`** - Game state detection and PyBoy environment wrappers
- **`training/`** - Unified training orchestration and configuration
- **`utils/`** - Memory reading, screen analysis, action parsing utilities
- **`config/`** - Memory addresses, constants, and configuration
- **`rewards/`** - Reward calculation system

### Training Modes
1. **LLM-only training** - Uses Ollama models for decision making
2. **Hybrid LLM-RL training** - Combines LLM guidance with DQN optimization
3. **Traditional RL training** - Pure reinforcement learning approach

## Common Commands

### Running Training
```bash
# Main entry point (unified trainer) - ALWAYS use save state for accurate stats
python3 main.py roms/pokemon_crystal.gbc --save-state roms/pokemon_crystal.gbc.state --max-actions 2000

# With LLM integration (recommended)
python3 main.py roms/pokemon_crystal.gbc --save-state roms/pokemon_crystal.gbc.state --max-actions 500 --llm-model smollm2:1.7b --llm-interval 10 --enable-web

# Hybrid training example
python3 examples/run_hybrid_training.py

# Quick start monitoring
./quick_start.sh
```

### ⚠️ **CRITICAL: Always Use Save States**
**Without save state**: Memory addresses read garbage data, causing inflated rewards (1000s of points), false badge detection, and incorrect game state analysis.

**With save state**: Proper game state loaded, realistic rewards (~20-100 points), accurate badge counting, and correct memory reading.

**Save state initialization**: The save state may load the game in a dialogue/menu state. The system automatically sends B button inputs after loading to advance past these states and ensure the game is playable.

**Reward system stability**: The reward calculator includes a capped penalty for location revisits to prevent runaway negative rewards. Extended training sessions should show stable ~-0.5 reward per action when the agent is stuck, not escalating penalties.

### Testing
```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test categories
python -m pytest tests/core/ tests/trainer/ -v
python -m pytest tests/integration/ -v

# Run tests with markers
python -m pytest -m "unit" -v
python -m pytest -m "integration" -v
python -m pytest -m "web_monitoring" -v
```

### Development Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Install for development (uses setup.py)
pip install -e .

# Code formatting and linting
black .
flake8

# Run single test file
python -m pytest tests/core/test_specific_file.py -v

# Run specific test method
python -m pytest tests/core/test_file.py::TestClass::test_method -v

# Run tests with coverage
python -m pytest tests/ --cov=. --cov-report=html
```

### LLM Setup (Required for LLM features)
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull recommended model
ollama pull smollm2:1.7b
```

## Important Architecture Notes

### Memory System
- Memory addresses defined in `config/memory_addresses.py`
- Memory reading utilities in `utils/memory_reader.py`
- Game state extracted includes HP, level, badges, party data, money, etc.
- Memory mapping system in `core/memory_map.py` provides derived calculations

### Web Monitoring
- Integrated web dashboard at http://localhost:8080 (or custom port with --web-port)
- **Real-time live game screen streaming** at 12fps capture, 30fps frontend display
- LLM decision tracking with reasoning and confidence scores
- Performance metrics and reward breakdown
- Located in `web_dashboard/` with unified server architecture
- **Fixed Issues**: Action execution pipeline, screen streaming, training status detection

### State Detection
- Multi-metric screen analysis (overworld, battle, dialogue, menu)
- Screen state analyzer in `utils/screen_analyzer.py`
- Game state detection in `environments/game_state_detection.py`

### LLM Integration
- Ollama-based LLM communication in `trainer/llm_manager.py`
- Enhanced parsing with natural language synonyms
- Stuck pattern detection and recovery
- Strategic context building for prompts

### Reward System
- Multi-factor rewards: health, leveling, badges, money, exploration, battles
- Progressive scaling with bigger rewards for major milestones
- Smart health logic that only applies when player has Pokemon
- Early game focus with special rewards for getting first Pokemon

### Training Configuration
- Hybrid training config in `hybrid_training_config.json`
- Training parameters in `config/constants.py`
- Supports curriculum learning and adaptive strategy switching

## Development Patterns

### Adding New Memory Addresses
Edit `config/memory_addresses.py` to add new memory locations and update `core/memory_map.py` for derived calculations.

### Customizing Rewards
Modify `rewards/calculator.py` or the PokemonRewardCalculator class to add custom reward logic.

### Extending LLM Prompts
Update prompt building methods in LLM trainer classes to customize AI decision-making context.

### Adding Test Categories
Use pytest markers defined in `pytest.ini` for organizing tests by functionality. Available markers include:

- `unit`: Unit tests
- `integration`: Integration tests
- `web_monitoring`: Web monitoring related tests
- `llm`: LLM functionality tests
- `performance`: Performance tests
- `state_detection`: Game state detection tests
- `memory_mapping`: Memory mapping and address tests
- `trainer_validation`: Trainer memory validation tests
- `anti_stuck`: Anti-stuck logic and recovery tests
- `unified_trainer`: Unified trainer system tests

Use markers to run specific test categories:
```bash
python -m pytest -m "unit and not integration" -v
python -m pytest -m "llm or web_monitoring" -v
```

## Project Status

This project is in active development with recent major refactoring:
- Consolidated architecture with unified entry points
- Enhanced LLM intelligence with advanced parsing
- Integrated web monitoring system
- Comprehensive test coverage across core components
- Deprecation of legacy `llm_trainer.py` in favor of `main.py`

### Current Development Branch
Currently on `project-wide-refactoring` branch with ongoing improvements. Main branch is `main`.

### Critical Requirements
- Legal Pokemon Crystal ROM file placed in the `roms/` directory
- Save state files (`.gbc.state`) strongly recommended for proper memory reading
- Ollama installation required for LLM features

### Architecture Notes for Developers
- The project uses `setup.py` for package installation
- Test organization uses extensive pytest markers for granular test selection
- Memory corruption protection and validation systems are in place
- Web monitoring is integrated directly into training systems
- Hybrid training combines multiple AI approaches

## Web Dashboard Troubleshooting

### Recent Critical Fixes (September 2024)

**✅ LLM Action Execution Bug (CRITICAL)**
- **Issue**: LLM decisions not executing - game character stuck at (0,0) despite LLM making decisions
- **Root Cause**: Action string to integer conversion bug in `training/components/llm_decision_engine.py:192-206`
- **Fix**: Added proper action mapping dictionary for "up"→1, "down"→2, etc.
- **Files Modified**: `/training/components/llm_decision_engine.py`

**✅ Live Game Screen Streaming**
- **Issue**: Game screen not displaying in web dashboard
- **Root Cause**: Screen capture only triggered by WebSocket connections, HTTP API serving stale/empty data
- **Fix**: Added `update_screen_for_http()` method called on every `/api/screen` request
- **Files Modified**: `/web_dashboard/websocket_handler.py`, `/web_dashboard/server.py`
- **Performance**: 12fps capture, 30fps frontend polling for smooth streaming

**✅ Training Status Detection**
- **Issue**: "Training Active" showing "No" during active training
- **Root Cause**: Inadequate training status detection logic
- **Fix**: Enhanced detection to infer from statistics tracker activity
- **Files Modified**: `/web_dashboard/api/endpoints.py:333-341`

**✅ Memory Debug Panel**
- **Issue**: "Memory reading failed" when PyBoy instance accessible
- **Root Cause**: Missing fallback to statistics tracker data
- **Fix**: Added comprehensive fallback chain for memory debug data
- **Files Modified**: `/web_dashboard/api/endpoints.py:241-300`

**✅ Additional Improvements**
- Favicon.ico handler to prevent broken pipe errors
- Enhanced CORS support and error handling
- Framerate optimization from 10fps to 12fps
- Updated web dashboard file structure documentation

### Common Issues and Solutions

**Issue**: Game screen shows placeholder image
- **Check**: PyBoy emulator initialization in logs
- **Check**: Screen capture manager status
- **Solution**: Restart training with `--enable-web` flag

**Issue**: LLM decisions not affecting game
- **Check**: Action execution logs in training output
- **Check**: LLM decision engine initialization
- **Solution**: Verify Ollama service running and model available

**Issue**: "Training Active" shows incorrect status
- **Check**: Statistics tracker activity in `/api/dashboard` response
- **Solution**: Status is inferred from action count activity

**Issue**: Memory debug shows empty data
- **Check**: PyBoy instance accessibility and save state loading
- **Solution**: Ensure save state file exists and loads properly