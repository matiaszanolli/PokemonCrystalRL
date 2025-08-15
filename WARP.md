# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

This is Pokemon Crystal RL - a sophisticated AI training system that combines reinforcement learning with local Large Language Models to play Pokemon Crystal intelligently. The project features multiple training modes, from ultra-fast rule-based gameplay to strategic LLM-powered decision making.

**Key Technologies**: PyBoy Game Boy emulator, Ollama (local LLM inference), SQLite for episodic memory, PyTorch for RL, gymnasium for environment interfaces.

## Quick Start Commands

### Prerequisites Setup
```bash
# Install Ollama (required for LLM modes)
curl -fsSL https://ollama.com/install.sh | sh

# Pull recommended LLM model
ollama pull smollm2:1.7b

# Install core dependencies
pip install pyboy pillow numpy ollama python-ollama
```

### Main Scripts & Usage

**Primary entry points** (located in `python_agent/`):

1. **LLM-powered gameplay** (current main script):
   ```bash
   cd python_agent
   python llm_play.py --no-headless --max-steps 1000
   ```

2. **Unified trainer** (comprehensive system):
   ```bash
   python pokemon_trainer.py --rom ../roms/pokemon_crystal.gbc --mode fast_local --web
   ```

3. **Quick performance test**:
   ```bash
   python pokemon_trainer.py --rom ../roms/pokemon_crystal.gbc --mode ultra_fast --actions 100 --no-llm
   ```

### Testing Commands

```bash
# Run all tests
pytest

# Run with verbose output and coverage
pytest -v --cov=. --cov-report=html

# Run specific test categories
pytest -m "unit"           # Unit tests only
pytest -m "integration"    # Integration tests only
pytest -m "not slow"       # Skip slow tests
```

### Development Commands

```bash
# Quick installation verification
python test_installation.py

# Final setup verification
python verify_final_setup.py

# Start monitoring system
python start_monitoring.py

# Launch quick start script
./quick_start.sh
```

## Architecture Overview

### Core System Design

The project uses a **layered architecture** with multiple training modes:

1. **Environment Layer** (`pyboy_env.py`): Wraps PyBoy emulator in OpenAI Gym interface
2. **Agent Layer** (`local_llm_agent.py`, `enhanced_llm_agent.py`): LLM-powered decision making
3. **Training Layer** (`pokemon_trainer.py`): Unified training orchestration
4. **Monitoring Layer** (`web_monitor.py`, `monitoring_client.py`): Real-time performance tracking

### Key Components

**UnifiedPokemonTrainer** (`pokemon_trainer.py`):
- Central orchestrator supporting 5 training modes
- Handles LLM backend switching (SmolLM2, Llama3.2, etc.)
- Web interface management and real-time monitoring

**LocalLLMPokemonAgent** (`local_llm_agent.py`):
- Strategic decision making using local Ollama models
- Episodic memory storage in SQLite database
- Game state analysis and context-aware planning

**PyBoyPokemonCrystalEnv** (`pyboy_env.py`):
- Clean gymnasium interface to Game Boy emulator
- Memory address mapping for Pokemon Crystal game state
- Reward calculation and episode management

### Training Modes

1. **fast_local**: ~40 actions/sec with LLM intelligence (recommended)
2. **ultra_fast**: 600+ actions/sec rule-based (benchmarking)
3. **curriculum**: Progressive 5-stage learning system
4. **monitored**: Full logging and analysis (research)
5. **custom**: User-defined configuration

### Memory System

The agent maintains **episodic memory** in SQLite databases:
- `pokemon_agent_memory.db`: Strategic decisions and game states
- `semantic_context.db`: Enhanced contextual understanding
- Memory tables: `game_states`, `strategic_decisions`, `pokemon_encounters`

## Important File Locations

### Configuration Files
- `requirements.txt`: Root project dependencies (stable-baselines3, torch, etc.)
- `python_agent/requirements.txt`: Agent-specific dependencies (pyboy, ollama)
- `python_agent/pytest.ini`: Test configuration with markers

### ROM and Save States
- Place Pokemon Crystal ROM in `roms/` or use `../pokecrystal.gbc`
- Save states: `pokemon_crystal_intro.state`, `pokecrystal.ss1`
- The project requires a **legally obtained** Pokemon Crystal ROM

### Documentation
- `python_agent/docs/`: Comprehensive documentation hub
- `python_agent/docs/guides/getting-started.md`: 10-minute setup guide
- `python_agent/README.md`: Detailed project overview

## Development Guidelines

### Code Structure
- **Main logic**: All in `python_agent/` directory
- **Archive**: Legacy/deprecated code in `python_agent/archive/`
- **Tests**: Comprehensive pytest suite in `python_agent/tests/`
- **Utils**: Shared utilities in `utils.py`, `memory_map.py`

### Running Modes Efficiently
- Use `--no-headless` only for debugging/watching gameplay
- Web interface (`--web`) provides better monitoring than terminal
- Start with `ultra_fast --no-llm` for quick system validation
- Use `fast_local` for most development work

### Performance Considerations
- **Ultra Fast Mode**: 600+ actions/sec, rule-based only
- **Fast Local Mode**: ~40 actions/sec with LLM intelligence
- **SmolLM2-1.7B**: Recommended model (~25ms inference)
- Web monitoring adds minimal overhead

### Testing Strategy
The project has extensive pytest-based testing:
- **Markers**: `unit`, `integration`, `performance`, `slow`, `database`
- **Coverage**: Dialogue systems, game integration, performance benchmarks
- **Fixtures**: Shared test setup in `conftest.py`

### Memory Management
- SQLite databases track agent decisions and learning
- Memory addresses for Pokemon Crystal are pre-mapped
- Game state analysis provides strategic context for decision making

### Monitoring & Debugging
- Web interface at `http://localhost:8080` when using `--web`
- Real-time screen capture and performance metrics
- Debug mode available with `--debug` flag
- Monitoring client connects to Flask-based server

## Common Development Tasks

### Adding New LLM Models
1. Add model to `LLMBackend` enum in `pokemon_trainer.py`
2. Pull model: `ollama pull model_name`
3. Test with: `python llm_play.py --model model_name`

### Modifying Training Behavior
- Edit prompts in `local_llm_agent.py` for strategic decisions
- Adjust `TrainingConfig` in `pokemon_trainer.py` for performance settings
- Memory behavior controlled in `_init_memory_db()` methods

### Performance Tuning
- Adjust `llm_interval` to control frequency of LLM calls
- Modify `capture_fps` for web interface performance
- Use `step_delay` in `llm_play.py` to control action timing

This codebase combines traditional RL techniques with modern LLM capabilities, offering both high-speed rule-based training and intelligent strategic gameplay through local language models.
