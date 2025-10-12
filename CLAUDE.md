# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Pokemon Crystal reinforcement learning platform that combines LLM-based decision making with traditional RL training. The system uses PyBoy emulation to train AI agents to play Pokemon Crystal, featuring hybrid LLM-RL training, memory mapping, and real-time web monitoring.

## Quick Reference - Most Used Commands

```bash
# Testing (ALWAYS use full path to pytest)
~/.pyenv/versions/pokemon-3.11.11/bin/pytest tests/ -v
~/.pyenv/versions/pokemon-3.11.11/bin/pytest tests/core/test_ab_testing_framework.py -v
~/.pyenv/versions/pokemon-3.11.11/bin/pytest tests/ -v --tb=no  # Cleaner output

# Training (ALWAYS use save state!)
python3 main.py roms/pokemon_crystal.gbc --save-state roms/pokemon_crystal.gbc.state --max-actions 2000 --enable-web

# LLM Training with monitoring
python3 main.py roms/pokemon_crystal.gbc --save-state roms/pokemon_crystal.gbc.state --max-actions 500 --llm-model smollm2:1.7b --llm-interval 10 --enable-web

# Clean Python cache (if imports break)
find . -type d -name __pycache__ -exec rm -r {} + && find . -type f -name "*.pyc" -delete && pip install -e .

# Check Ollama models
ollama list
```

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

**Current Status:** 13/27 tests passing (48%) - See [TESTING_ROADMAP.md](TESTING_ROADMAP.md) for detailed status and refactoring plan

**Test Organization**:
- **Unit Tests**: Fast, isolated component tests (~500+ planned)
- **Integration Tests**: Component interaction tests (69 tests, 4435 lines)
- **E2E Tests**: Full workflow validation - See [E2E_TESTING_PROPOSAL.md](E2E_TESTING_PROPOSAL.md) for implementation plan

```bash
# Run all tests (IMPORTANT: Use full pyenv path)
~/.pyenv/versions/pokemon-3.11.11/bin/pytest tests/ -v

# Run with coverage
~/.pyenv/versions/pokemon-3.11.11/bin/pytest tests/ --cov=. --cov-report=html

# Run specific test categories
~/.pyenv/versions/pokemon-3.11.11/bin/pytest tests/core/ tests/trainer/ tests/monitoring/ -v
~/.pyenv/versions/pokemon-3.11.11/bin/pytest tests/integration/ -v

# Run tests with markers
~/.pyenv/versions/pokemon-3.11.11/bin/pytest -m "unit" -v
~/.pyenv/versions/pokemon-3.11.11/bin/pytest -m "integration" -v
~/.pyenv/versions/pokemon-3.11.11/bin/pytest -m "web_monitoring" -v

# Run E2E tests (when implemented)
~/.pyenv/versions/pokemon-3.11.11/bin/pytest tests/e2e -m "e2e_smoke" -v  # Quick smoke tests
~/.pyenv/versions/pokemon-3.11.11/bin/pytest tests/e2e -m "e2e_medium" -v  # Medium tests
~/.pyenv/versions/pokemon-3.11.11/bin/pytest tests/e2e -m "e2e_slow" -v  # Long-running tests

# Run specific test file or method
~/.pyenv/versions/pokemon-3.11.11/bin/pytest tests/core/test_adaptive_strategy_system.py -v
~/.pyenv/versions/pokemon-3.11.11/bin/pytest tests/core/test_adaptive_strategy_system.py::TestAdaptiveStrategySystem::test_strategy_switching -v
```

**Known Issues:**
- 8 integration tests in `test_complex_behavioral_workflows.py` need event system refactoring
- No true end-to-end tests currently exist (proposal in [E2E_TESTING_PROPOSAL.md](E2E_TESTING_PROPOSAL.md))
- See [TESTING_ROADMAP.md](TESTING_ROADMAP.md) for prioritized fix plan

### Development Setup

**IMPORTANT**: Always use the `pokemon-3.11.11` pyenv virtualenv for all Python commands.

```bash
# If virtualenv is broken (missing pip), recreate it
pyenv virtualenv-delete pokemon-3.11.11
pyenv virtualenv 3.11.11 pokemon-3.11.11
pyenv activate pokemon-3.11.11

# Activate virtualenv (if not already active)
pyenv activate pokemon-3.11.11

# Install dependencies
pip install -r requirements.txt

# Install pytest and coverage tools (already in requirements.txt)
pip install pytest pytest-cov

# Install for development (uses setup.py)
pip install -e .

# Code formatting and linting
black .
flake8

# Run tests (IMPORTANT: Use ~/.pyenv/versions/pokemon-3.11.11/bin/pytest for consistency)
~/.pyenv/versions/pokemon-3.11.11/bin/pytest tests/ -v

# Run tests with coverage
~/.pyenv/versions/pokemon-3.11.11/bin/pytest tests/ --cov=. --cov-report=html

# Run specific test file
~/.pyenv/versions/pokemon-3.11.11/bin/pytest tests/core/test_adaptive_strategy_system.py -v

# Run tests with verbose output and no traceback (for cleaner CI output)
~/.pyenv/versions/pokemon-3.11.11/bin/pytest tests/ -v --tb=no
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

#### Battle Strategies (`plugins/battle_strategies.py`)
- **Aggressive, Defensive, Balanced** - Move recommendation and switch assessment strategies

#### Exploration Patterns (`plugins/exploration/`)
Modular exploration pattern implementations with hot-swappable patterns:
- **SystematicSweepPattern** (`systematic_sweep.py`) - Horizontal sweeping for thorough map coverage
- **SpiralSearchPattern** (`spiral_search.py`) - Expanding outward search from center point
- **WallFollowingPattern** (`wall_following.py`) - Boundary exploration following walls (right/left configurable)
- **RandomWalkPattern** (`random_walk.py`) - Biased random walk toward unexplored areas

All patterns available via backward-compatible imports:
```python
# New style (recommended)
from plugins.exploration import SystematicSweepPattern, SpiralSearchPattern

# Old style (still supported)
from plugins.exploration_patterns import SystematicSweepPattern, SpiralSearchPattern
```

#### Reward Calculators (`rewards/components/`)
Modular component-based reward system:
- **Progression-focused, battle-focused, exploration-focused, balanced** reward calculators
- Component architecture in `rewards/components/` with specialized reward components

#### Plugin Coordination
- Multiple plugins work together with priority-based selection
- Hot-swappable plugins for runtime configuration changes

### A/B Testing Framework (`core/ab_testing/`)
**Production-ready experimental framework for comparing AI configurations**

- **ExperimentManager** - Complete experiment lifecycle with concurrent execution control
- **StatisticalAnalyzer** - Rigorous statistical analysis (t-tests, Cohen's d, confidence intervals)
- **AutomationFramework** - 6 pre-built workflow templates for common testing scenarios
- **Event Integration** - Publishes experiment events for real-time monitoring
- **REST API** - Full programmatic control via `/api/v1/experiments` endpoints

**Pre-built Automation Templates**:
1. **Hyperparameter Sweep** - Systematic parameter grid search
2. **Weekend Stress Test** - Long-duration stability testing
3. **Regression Testing Suite** - Validate changes don't break existing functionality
4. **Custom Workflows** - Build your own experiment sequences

**Usage**:
```bash
# Run A/B testing demos
python3 examples/ab_testing_demo.py
python3 examples/ab_testing_automation_demo.py
python3 examples/ab_testing_web_demo.py

# API-based experiment control
python3 examples/ab_testing_api_test.py
```

**Key Features**:
- Concurrent experiment limit (default: 3)
- Automatic statistical significance testing
- Minimum sample size validation (10 per variant)
- Thread-safe experiment tracking
- Winner determination based on configurable metrics

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

#### **Game Intelligence Module** (`core/intelligence/`)
- **Modular architecture** with focused, maintainable modules:
  - `location.py` - Location analysis, context, and strategic recommendations (136 lines)
  - `progression.py` - Game phase tracking and objective determination (85 lines)
  - `battle.py` - Battle strategy, type effectiveness, move recommendations (205 lines)
  - `inventory.py` - Item management and usage strategies (208 lines)
  - `orchestrator.py` - GameIntelligence coordinator (164 lines)
- **Main Classes**:
  - `LocationAnalyzer`: Map understanding, navigation optimization, area classification
  - `ProgressTracker`: Story progression tracking, objective detection, phase analysis
  - `BattleStrategy`: Advanced combat analysis with type effectiveness, move evaluation
  - `InventoryManager`: Intelligent item usage and inventory management
  - `GameIntelligence`: Main coordinator for comprehensive game state analysis
- **Backward compatible**: `core/game_intelligence.py` re-exports all classes
- **Recent refactoring (2025-10-04)**: Split 763-line monolith into 5 focused modules

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

### Error Handling System (`monitoring/error_handling/`)
Modular error handling system with centralized error management, circuit breaking, and recovery strategies.

**Package Structure**:
- **`types.py`** - Error enums (ErrorSeverity, ErrorCategory, RecoveryStrategy) and data structures (ErrorContext, ErrorEvent)
- **`decorators.py`** - `@error_boundary` decorator and `SafeOperation` context manager for safe code execution
- **`circuit_breaker.py`** - Circuit breaker for preventing cascading failures (configurable thresholds and timeouts)
- **`memory_monitor.py`** - Memory usage tracking, garbage collection, and threshold-based callbacks
- **`handler.py`** - Main `ErrorHandler` singleton for centralized error management

**Key Features**:
- **Centralized error handling** with singleton ErrorHandler pattern
- **Error categorization** by severity (CRITICAL, HIGH, ERROR, MEDIUM, WARNING, INFO) and category (SYSTEM, NETWORK, DATABASE, GAME, TRAINING, MEMORY, PERFORMANCE)
- **Circuit breaker protection** - Automatically disables components experiencing high error rates
- **Recovery strategies** - Pluggable recovery mechanisms (RETRY, RESTART, RESET, GRACEFUL_SHUTDOWN, FALLBACK)
- **Component health tracking** - Monitor registered components and trigger recovery callbacks
- **Error deduplication** - Prevents duplicate error logging within time windows
- **Notification system** - Batch error notifications via data bus
- **Memory monitoring** - Track memory usage and trigger garbage collection
- **Database integration** - Optional error recording to database

**Usage Examples**:
```python
# Import (both old and new styles work)
from monitoring.error_handler import ErrorHandler, ErrorSeverity  # Old style
from monitoring.error_handling import ErrorHandler, ErrorSeverity  # New style (recommended)

# Using error boundary decorator
from monitoring.error_handling import error_boundary

@error_boundary(max_retries=3, category=ErrorCategory.GAME)
def risky_operation():
    # Code that might fail
    pass

# Using SafeOperation context manager
from monitoring.error_handling import SafeOperation

with SafeOperation("my_component", "data_processing"):
    # Protected code
    process_data()

# Manual error handling
handler = ErrorHandler.get_instance()
try:
    risky_code()
except Exception as e:
    handler.handle_error(
        e,
        severity=ErrorSeverity.HIGH,
        category=ErrorCategory.TRAINING,
        component="trainer"
    )
```

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

## Important Development Notes

### Pyenv Virtual Environment
This project **requires** the `pokemon-3.11.11` pyenv virtual environment. Always prefix Python commands with the full path:
- `~/.pyenv/versions/pokemon-3.11.11/bin/pytest` (not just `pytest`)
- `~/.pyenv/versions/pokemon-3.11.11/bin/python` (not just `python3`)

This ensures consistency and avoids issues with system Python or other virtual environments.

### Event System Singleton Pattern
**CRITICAL**: The event system uses a singleton pattern. Always use `get_event_bus()` instead of creating a new `EventBus()` instance:
```python
# Correct
from core.event_system import get_event_bus
event_bus = get_event_bus()

# Wrong - creates separate instance
from core.event_system import EventBus
event_bus = EventBus()  # Don't do this!
```

### Backward Compatible Imports
The codebase maintains backward compatibility for imports during the modular refactoring:
```python
# Both work (new style preferred):
from core.intelligence import GameIntelligence
from core.game_intelligence import GameIntelligence  # Still works

# Both work:
from plugins.exploration import SystematicSweepPattern
from plugins.exploration_patterns import SystematicSweepPattern  # Still works

# Both work:
from monitoring.error_handling import ErrorHandler
from monitoring.error_handler import ErrorHandler  # Still works
```

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
Last updated: October 2025 (2025-10-11)

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

#### Modular Architecture (2025 Refactoring)
The codebase has been refactored into modular packages for better maintainability:

**Modular Packages** (backward compatible re-exports):
- **`core/intelligence/`** - Game intelligence modules (location, progression, battle, inventory, orchestrator)
  - Old import: `from core.game_intelligence import GameIntelligence`
  - New import: `from core.intelligence import GameIntelligence`
- **`plugins/exploration/`** - Exploration pattern plugins (systematic sweep, spiral search, wall following, random walk)
  - Old import: `from plugins.exploration_patterns import SystematicSweepPattern`
  - New import: `from plugins.exploration import SystematicSweepPattern`
- **`monitoring/error_handling/`** - Error handling system (types, decorators, circuit breaker, memory monitor, handler)
  - Old import: `from monitoring.error_handler import ErrorHandler`
  - New import: `from monitoring.error_handling import ErrorHandler`
- **`rewards/components/`** - Reward calculation components (modular reward system)

**Benefits**: Each module has single responsibility, independent testability, and clearer separation of concerns. All old imports still work for backward compatibility.

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

## Common Development Pitfalls

### Testing Issues

**Flaky Event System Tests**
- **Problem**: Events not being received in tests despite being published
- **Cause**: Multiple event bus instances instead of singleton
- **Fix**: Always use `get_event_bus()` and add cleanup fixture:
```python
@pytest.fixture(autouse=True)
def reset_event_bus():
    event_bus = get_event_bus()
    event_bus.subscribers.clear()
    yield
    event_bus.subscribers.clear()
```

**Statistical Test Failures**
- **Problem**: A/B testing framework tests fail with significance errors
- **Cause**: Sample sizes too small (minimum is 10 per variant)
- **Fix**: Use `sample_size_per_variant: 10` or higher in test configs

**Time-based Test Issues**
- **Problem**: Tests depending on time.time() failing
- **Cause**: Immutable mock values
- **Fix**: Use mutable state in time mocks:
```python
time_state = {"current": 0.0}
mock_time.side_effect = lambda: time_state["current"]
# Then update: time_state["current"] += 1.0
```

### Integration Issues

**Import Errors After Refactoring**
- **Problem**: `ModuleNotFoundError` for recently moved files
- **Cause**: Stale `__pycache__` or `.pyc` files
- **Fix**: Clean Python cache and reinstall:
```bash
find . -type d -name __pycache__ -exec rm -r {} +
find . -type f -name "*.pyc" -delete
pip install -e .
```

**LLM Decisions Not Executing**
- **Problem**: LLM makes decisions but game character doesn't move
- **Cause**: Action string to integer conversion bug (fixed in Oct 2024)
- **Location**: `training/components/llm_decision_engine.py` has proper action mapping
- **Verify**: Check logs for "Action mapping: {action_string} -> {action_int}"

**Reward System Giving Huge Negative/Positive Values**
- **Problem**: Rewards in thousands instead of -0.5 to +500 range
- **Cause**: Training without save state - memory reads garbage data
- **Fix**: Always use `--save-state roms/pokemon_crystal.gbc.state` flag

### File Structure Confusion

**Reward Calculator Files**
The project has THREE different reward-related files with distinct purposes:
- `rewards/calculator.py` - Main reward calculation logic (use this for game rewards)
- `core/game_state_extractor.py` - Extracts game state from memory (not a reward calculator)
- `training/components/training_reward_tracker.py` - Tracks rewards during training pipeline

**Trainer Directory Removed**
- `trainer/` directory was removed in September 2024
- Use `training/` directory instead
- Exception: `tests/trainer/` still exists for testing training components