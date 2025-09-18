# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Pokemon Crystal reinforcement learning platform that combines LLM-based decision making with traditional RL training. The system uses PyBoy emulation to train AI agents to play Pokemon Crystal, featuring hybrid LLM-RL training, memory mapping, and real-time web monitoring.

## Core Architecture

### Primary Entry Points
- **`main.py`** - Main entry point for training (replaces deprecated `llm_trainer.py`)
- **`examples/run_hybrid_training.py`** - Hybrid LLM-RL training example
- **`quick_start.sh`** - Quick start script for monitoring system

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
# Main entry point (unified trainer)
python3 main.py roms/pokemon_crystal.gbc --max-actions 2000

# Hybrid training example
python3 examples/run_hybrid_training.py

# Quick start monitoring
./quick_start.sh
```

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

# Install for development
pip install -e .

# Code formatting and linting
black .
flake8
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
- Memory addresses defined in `config/memory_addresses.py` (or `core/memory/addresses.py`)
- Memory reading utilities in `utils/memory_reader.py`
- Game state extracted includes HP, level, badges, party data, money, etc.

### Web Monitoring
- Integrated web dashboard at http://localhost:8080
- Real-time screen capture and LLM decision tracking
- Performance metrics and reward breakdown
- Located in `core/web_monitor/`

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
Edit `config/memory_addresses.py` or `core/memory/addresses.py` to add new memory locations and derived calculations.

### Customizing Rewards
Modify `rewards/calculator.py` or the PokemonRewardCalculator class to add custom reward logic.

### Extending LLM Prompts
Update prompt building methods in LLM trainer classes to customize AI decision-making context.

### Adding Test Categories
Use pytest markers defined in `pytest.ini` for organizing tests by functionality (unit, integration, web_monitoring, llm, etc).

## Project Status

This project is in active development with recent major refactoring:
- Consolidated architecture with unified entry points
- Enhanced LLM intelligence with advanced parsing
- Integrated web monitoring system
- Comprehensive test coverage across core components
- Deprecation of legacy `llm_trainer.py` in favor of `main.py`

The system requires a legal Pokemon Crystal ROM file placed in the `roms/` directory to function.