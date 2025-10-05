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
- **`core/`** - Core systems (event system, plugin system, memory mapping, game intelligence, tournament system)
- **`agents/`** - Multi-agent framework with specialist agents (battle, explorer, progression)
- **`plugins/`** - Modular plugin system for battle strategies, exploration patterns, rewards
- **`training/`** - **Unified training orchestration** with component-based architecture
- **`environments/`** - Game state detection and PyBoy environment wrappers
- **`utils/`** - Memory reading, screen analysis, action parsing utilities
- **`config/`** - Memory addresses, constants, and configuration
- **`rewards/`** - Reward calculation system
- **`web_dashboard/`** - Real-time web monitoring with REST API and live streaming

### ⚠️ **IMPORTANT ARCHITECTURAL CHANGES** (September 2024)
- **`trainer/` directory REMOVED** - Legacy compatibility layer eliminated for cleaner architecture
  - **Note**: `tests/trainer/` still exists for testing training components
- **Descriptive file naming** - Eliminated confusing duplicates:
  - `core/reward_calculator.py` → `core/game_state_extractor.py` (game state extraction utility)
  - `training/components/reward_calculator.py` → `training/components/training_reward_tracker.py` (training pipeline tracker)
  - `rewards/calculator.py` remains as the main reward calculation logic
- **All imports updated** - Use descriptive names for better maintainability

### Training Modes
1. **LLM-only training** - Uses Ollama models for decision making
2. **Advanced Hybrid LLM-RL training** - Combines strategic LLM reasoning with tactical RL optimization using temporal memory
3. **Basic Hybrid training** - Simple DQN + LLM combination with basic coordination
4. **Multi-agent training** - Specialist agents coordinated by event system
5. **Plugin-based training** - Modular components for different strategies
6. **Curriculum learning** - Progressive difficulty training with save state library integration
7. **Tournament mode** - Competitive AI configuration battles with automated scheduling

## Common Commands

### Running Training
```bash
# Main entry point (unified trainer) - ALWAYS use save state for accurate stats
python3 main.py roms/pokemon_crystal.gbc --save-state roms/pokemon_crystal.gbc.state --max-actions 2000

# With LLM integration (recommended)
python3 main.py roms/pokemon_crystal.gbc --save-state roms/pokemon_crystal.gbc.state --max-actions 500 --llm-model smollm2:1.7b --llm-interval 10 --enable-web

# Advanced Hybrid LLM-RL training (combines strategic LLM with tactical RL)
python3 main.py roms/pokemon_crystal.gbc --save-state roms/pokemon_crystal.gbc.state --enable-hybrid-llm-rl --max-episodes 50 --llm-weight 0.7 --rl-weight 0.3 --enable-web

# Hybrid training examples
python3 examples/run_hybrid_training.py
python3 examples/run_hybrid_llm_rl_training.py

# Curriculum learning training
python3 main.py roms/pokemon_crystal.gbc --enable-curriculum --max-actions 500
python3 examples/run_curriculum_training.py roms/pokemon_crystal.gbc --episodes 10

# A/B testing examples
python3 examples/ab_testing_demo.py
python3 examples/ab_testing_automation_demo.py
python3 examples/ab_testing_api_test.py
python3 examples/ab_testing_realtime_demo.py
python3 examples/ab_testing_web_demo.py

# Tournament mode examples
python3 examples/tournament_demo.py

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

# Run with coverage
python -m pytest tests/ --cov=. --cov-report=html

# Run specific test categories
python -m pytest tests/core/ tests/trainer/ tests/monitoring/ -v
python -m pytest tests/integration/ -v

# Run tests with markers
python -m pytest -m "unit" -v
python -m pytest -m "integration" -v
python -m pytest -m "web_monitoring" -v

# Run specific test file or method
python -m pytest tests/core/test_adaptive_strategy_system.py -v
python -m pytest tests/core/test_adaptive_strategy_system.py::TestAdaptiveStrategySystem::test_strategy_switching -v
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
```

### LLM Setup (Required for LLM features)
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull recommended model
ollama pull smollm2:1.7b
```

## Important Architecture Notes

### Multi-Agent Framework (`agents/`)
- **MultiAgentCoordinator** - Orchestrates specialist agents with intelligent coordination
- **BattleAgent** - Combat optimization specialist leveraging enhanced BattleStrategy
- **ExplorerAgent** - Map discovery specialist with multiple exploration patterns
- **ProgressionAgent** - Story completion specialist with phase-aware progression
- **Event-driven coordination** - Agents react to game events and publish performance updates
- **Adaptive performance tracking** - Agent weights adjust based on success rates

### Event System - Reactive Architecture (`core/event_system.py`)
- **EventBus** - Central publish-subscribe system with filtering and priority handling
- **20+ Event Types** - Battle, level-up, badge, location, agent decisions, performance updates
- **EventSubscriber interface** - Components subscribe to relevant events for reactive behavior
- **GameStateEventDetector** - Automatically detects and publishes game state changes
- **EventDrivenAnalytics** - Real-time metrics tracking with performance insights
- **Event correlation** - Track related events with correlation IDs for pattern analysis

### Plugin System - Modular Architecture (`core/plugin_system.py`, `plugins/`)
- **PluginRegistry** - Centralized plugin discovery, loading, and lifecycle management
- **Hot-swappable plugins** - Runtime plugin updates without stopping training
- **Plugin types**: Battle strategies, exploration patterns, reward calculators, screen analyzers
- **PluginManager** - High-level interface for plugin coordination and recommendation aggregation
- **Performance tracking** - Built-in monitoring for plugin call counts and execution time
- **Configuration validation** - Ensure plugin configs are valid before activation

### Official Plugin Implementations
- **Battle Strategies**: Aggressive, Defensive, Balanced with move recommendation and switch assessment
- **Exploration Patterns**: Systematic sweep, spiral search, wall following, random walk
- **Reward Calculators**: Progression-focused, battle-focused, exploration-focused, balanced
- **Plugin coordination** - Multiple plugins work together with priority-based selection

### Tournament Mode - Competitive AI Battles (`core/tournament/`)
- **TournamentManager** - Complete tournament lifecycle management with A/B testing integration
- **BracketGenerator** - Support for single elimination, double elimination, round robin, Swiss system
- **Tournament Analytics** - Performance insights, strategy effectiveness analysis, competitive balance metrics
- **REST API Integration** - Full tournament management through web dashboard API endpoints
- **Automated Scheduling** - Leverages A/B testing automation framework for match execution
- **Pre-built Profiles** - Quick battle, championship, research, endurance, speedrun configurations
- **Real-time Monitoring** - Live tournament progress tracking and bracket visualization

### Advanced AI Systems

#### **Hybrid LLM-RL Training System** (`training/hybrid_llm_rl_trainer.py`, `agents/hybrid_llm_rl_agent.py`)
- **Advanced Decision Engine**: Intelligently combines LLM strategic reasoning with RL tactical optimization
- **Temporal Memory Integration**: Bridges decisions across time scales using `TemporalMemoryBuffer`
- **Curriculum Learning Integration**: Progressive skill development with save state library
- **Adaptive Weight Adjustment**: Dynamic balancing of LLM vs RL influence based on performance
- **Multiple Decision Modes**: Strategic LLM, Tactical RL, Hybrid Balanced, Exploration, Curriculum-Guided
- **Novelty Detection**: Uses state similarity to trigger appropriate decision modes
- **Performance Tracking**: Comprehensive metrics for LLM success rates, RL optimization, mode switches

**Usage**:
```bash
# Basic hybrid training
python3 main.py roms/pokemon_crystal.gbc --enable-hybrid-llm-rl --max-episodes 100

# Advanced configuration with curriculum learning
python3 main.py roms/pokemon_crystal.gbc --enable-hybrid-llm-rl --enable-curriculum \
  --llm-weight 0.8 --rl-weight 0.2 --exploration-rate 0.15 --max-episodes 200 \
  --temporal-buffer-size 150000 --enable-web --web-port 8080

# Standalone advanced trainer
python3 examples/run_hybrid_llm_rl_training.py roms/pokemon_crystal.gbc \
  --max-episodes 50 --enable-web --curriculum-config custom_curriculum.json
```

#### **Game Intelligence Module** (`core/game_intelligence.py`)
- **BattleIntelligence**: Advanced combat analysis with type effectiveness, move evaluation
- **LocationIntelligence**: Map understanding, navigation optimization, area classification
- **ProgressIntelligence**: Story progression tracking, objective detection, phase analysis
- **GameStateAnalyzer**: Comprehensive game state interpretation with contextual analysis

#### **Decision Analysis System** (`core/decision_analysis/`)
- **Decision Database**: Persistent storage of AI decisions with outcome tracking
- **Pattern Detector**: Behavioral pattern recognition and learning optimization
- **Performance Analyzer**: Decision effectiveness analysis and strategy recommendations
- **Correlation Analysis**: Multi-factor decision outcome analysis

#### **Strategic Context Builder** (`core/strategic_context_builder.py`)
- **Context Assembly**: Multi-source game state aggregation for LLM prompts
- **Situation Assessment**: Threat/opportunity detection with confidence scoring
- **Action Consequence Prediction**: Predictive modeling for decision outcomes
- **Adaptive Prompting**: Dynamic LLM prompt optimization based on game phase

### Memory System
- Memory addresses defined in `config/memory_addresses.py`
- Memory reading utilities in `utils/memory_reader.py`
- Game state extracted includes HP, level, badges, party data, money, etc.
- Memory mapping system in `core/memory_map.py` provides derived calculations

### Web Monitoring & REST API
- **Integrated web dashboards** at http://localhost:8080 (or custom port with --web-port):
  - `/dashboard` - Original unified training dashboard
  - `/hybrid` - **NEW: Modern hybrid LLM-RL monitoring dashboard**
  - `/advanced` - Advanced analytics & debugging dashboard
- **Real-time live game screen streaming** at 12fps capture, 30fps frontend display
- **Hybrid Dashboard Features**:
  - **Information-dense design** - Maximum screen space utilization
  - **Real-time LLM vs RL decision tracking** with live decision stream
  - **Advanced metrics**: Success rates, mode distribution, temporal memory stats
  - **Live performance charts** - Reward trends, confidence tracking, distribution analysis
  - **Curriculum learning progress** with stage indicators
  - **System diagnostics** - Response times, memory usage, GPU utilization
  - **WebSocket-powered live updates** with 60fps dashboard refresh
- Located in `web_dashboard/` with unified server architecture
- **REST API**: Complete programmatic interface at `/api/v1/` - see [API.md](API.md) for full documentation
  - Training session management (`/training/sessions`)
  - Multi-agent control (`/agents`)
  - Plugin system management (`/plugins`)
  - Real-time monitoring and metrics
- **Fixed Issues**: Action execution pipeline, screen streaming, training status detection

### State Detection
- Multi-metric screen analysis (overworld, battle, dialogue, menu)
- Screen state analyzer in `utils/screen_analyzer.py`
- Game state detection in `environments/game_state_detection.py`

### LLM Integration
- Ollama-based LLM communication in `training/components/llm_manager.py`
- Enhanced parsing with natural language synonyms
- Stuck pattern detection and recovery
- Strategic context building for prompts

### Reward System
- **Component-based architecture** - Modular reward components in `rewards/components/`
  - `progress.py` - Health, level, and badge rewards
  - `movement.py` - Exploration, movement, and blocked movement penalties
  - `interaction.py` - Battle, dialogue, money, and progression rewards
- Main calculator (`rewards/calculator.py` - 124 lines) orchestrates components
- Multi-factor rewards: health, leveling, badges, money, exploration, battles
- Progressive scaling with bigger rewards for major milestones
- Smart health logic that only applies when player has Pokemon
- Early game focus with special rewards for getting first Pokemon
- **Plugin-based rewards** - Modular reward calculators with different focuses
- **Recent refactoring (2025-10-04)**: Removed 556 lines of duplicate code (81% reduction)

### Save State Library & Curriculum Learning

#### **Save State Library** (`core/save_state_library.py`, `scripts/manage_save_states.py`)
- **Metadata-driven save state management** with categorization by phase, scenario, difficulty, badges
- **CLI management interface** with comprehensive add/list/info/recommend/verify commands
- **Integrity verification** with checksums, usage tracking, and automatic validation
- **Recommendation engine** for optimal training scenario selection based on curriculum level
- **Advanced filtering** by tags, difficulty combinations, badge ranges, and scenarios

```bash
# Manage save state library
python3 scripts/manage_save_states.py add tutorial.state "Tutorial Start" "Beginning of game" \
  --phase tutorial --scenario first_pokemon --difficulty easy --badges 0

python3 scripts/manage_save_states.py list --scenario gym_battle --difficulty medium
python3 scripts/manage_save_states.py info
python3 scripts/manage_save_states.py recommend gym_battle --min-badges 1 --max-badges 3
```

#### **Curriculum Learning System** (`training/curriculum_learning.py`)
- **5-stage progressive training**: Tutorial → Basic → Intermediate → Advanced → Expert
- **Adaptive advancement** based on success rates (60%-80%) and episode counts (5-50 episodes)
- **Automatic save state selection** integrated with library recommendation engine
- **Configurable JSON curriculum** with custom advancement criteria and scenarios
- **Progress persistence** with comprehensive status reporting and advancement history

```bash
# Curriculum learning modes
python3 main.py roms/pokemon_crystal.gbc --enable-curriculum --curriculum-episodes 10
python3 examples/run_curriculum_training.py roms/pokemon_crystal.gbc --curriculum-config custom.json

# Generate default curriculum configuration
python3 -m training.curriculum_learning
```

### Training Configuration
- Hybrid training config in `hybrid_training_config.json`
- Training parameters in `config/constants.py`
- Curriculum config in `curriculum_config.json` with 5-stage progression
- Supports curriculum learning and adaptive strategy switching

## Development Patterns

### Adding New Memory Addresses
Edit `config/memory_addresses.py` to add new memory locations and update `core/memory_map.py` for derived calculations.

### Customizing Rewards
The reward system uses a component-based architecture. To customize rewards:
1. Modify existing components in `rewards/components/` (progress, movement, interaction)
2. Create new reward components by extending `RewardComponent` base class
3. Register new components in `PokemonRewardCalculator.__init__()` in `rewards/calculator.py`

The main calculator file (`rewards/calculator.py`) should only contain orchestration logic - keep component implementations in separate files.

### Extending LLM Prompts
Update prompt building methods in LLM trainer classes to customize AI decision-making context.

### Adding Test Categories
Use pytest markers defined in `pytest.ini` for organizing tests by functionality. Available markers include:

- `unit`: Unit tests
- `integration`: Integration tests
- `performance`: Performance tests
- `web_monitoring`: Web monitoring related tests
- `llm`: LLM functionality tests
- `multi_model`: Tests involving multiple model configurations
- `state_detection`: Game state detection tests
- `enhanced_rewards`: Enhanced reward system tests
- `monitoring`: Data monitoring system tests
- `memory_mapping`: Memory mapping and address tests
- `memory_corruption`: Memory corruption protection tests
- `trainer_validation`: Trainer memory validation tests
- `pyboy_integration`: PyBoy integration tests
- `streaming`: Game streaming functionality tests
- `cleanup`: Resource cleanup and shutdown tests
- `web`: Web server and HTTP interface tests
- `anti_stuck`: Anti-stuck logic and recovery tests
- `unified_trainer`: Unified trainer system tests
- `system`: System and component integration tests

Use markers to run specific test categories:
```bash
python -m pytest -m "unit and not integration" -v
python -m pytest -m "llm or web_monitoring" -v
```

## Project Status

This project is in active development with major systems completed (2024-2025):
- **Core Platform**: Multi-agent framework, event system, plugin architecture ✅
- **Advanced AI**: Game intelligence, strategic context building, experience memory ✅
- **Web Platform**: Real-time dashboard, REST API, live streaming ✅
- **A/B Testing**: Complete automation framework with 6 workflow templates ✅
- **Save State Library**: Metadata-driven scenario management with CLI interface ✅
- **Curriculum Learning**: 5-stage progressive difficulty training system ✅
- **Tournament Mode**: Complete competitive AI battle system with automated scheduling ✅
- **Architecture Unification**: Streamlined codebase with descriptive naming and legacy removal ✅
- **Test Coverage**: 175+ test methods across critical AI modules (85%+ coverage) ✅
- **Current Focus**: Advanced analytics dashboard, distributed training, production deployment

### Current Development Branch
Currently on `learn_to_play` branch with ongoing improvements. Main branch is `main`.
Last updated: October 2025

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

### Recent Critical Fixes (2024)

**✅ LLM Action Execution Bug (CRITICAL)**
- **Issue**: LLM decisions not executing - game character stuck at (0,0) despite LLM making decisions
- **Root Cause**: Action string to integer conversion bug in `training/components/llm_decision_engine.py:192-206`
- **Fix**: Added proper action mapping dictionary for "up"→1, "down"→2, etc.
- **Files Modified**: `/training/components/llm_decision_engine.py`
- **Note**: LLM manager moved from deprecated `trainer/llm_manager.py` to `training/components/llm_manager.py`

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