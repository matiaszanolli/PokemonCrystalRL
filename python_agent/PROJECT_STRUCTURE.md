# Pokemon Crystal RL - Python Agent

## Project Structure

```
python_agent/
├── 📁 Core Components
│   ├── pyboy_env.py              # PyBoy Pokemon Crystal environment
│   ├── enhanced_llm_agent.py     # Main LLM agent with vision capabilities
│   ├── vision_processor.py       # Computer vision processing
│   ├── vision_enhanced_training.py  # Main training loop
│   ├── memory_map.py             # Game memory mappings
│   ├── env.py                    # Environment wrapper
│   └── utils.py                  # Utility functions
│
├── 📁 tools/                     # Development and monitoring tools
│   ├── terminal_training_monitor.py
│   └── visual_training_monitor.py
│
├── 📁 models/                    # Model storage (empty, for future use)
│   └── .gitkeep
│
├── 📁 outputs/                   # Generated outputs
│   ├── training_report_*.json    # Training session reports
│   ├── screenshot_analysis/      # Visual analysis outputs
│   ├── step_*_info.json         # Step-by-step debug info
│   └── step_*_screen.txt        # Screen state captures
│
├── 📁 docs/                      # Documentation
│   ├── FIX_SUMMARY.md
│   ├── PYBOY_FIX_COMPLETE.md
│   ├── README.md
│   └── TRAINING_OVERVIEW.md
│
├── 📁 archive/                   # Deprecated/old files
│   ├── train.py                  # Old training script
│   ├── train_pyboy.py           # Old PyBoy training
│   ├── local_llm_agent.py       # Previous agent version
│   ├── llm_play.py              # Demo script
│   ├── emulator_demo.py         # PyBoy demo
│   ├── test_env_fix.py          # Environment test
│   ├── test_pyboy_screen.py     # Screen capture test
│   └── system_status.py         # System monitoring
│
└── PROJECT_STRUCTURE.md         # This file
```

## Main Entry Points

- **`vision_enhanced_training.py`** - Main training script with full vision capabilities
- **`enhanced_llm_agent.py`** - Core agent that combines LLM reasoning with vision
- **`pyboy_env.py`** - Game environment interface

## Quick Start

```bash
# Run main training session
python vision_enhanced_training.py

# Monitor training (in separate terminal)
python tools/terminal_training_monitor.py
```

## Recent Improvements

- ✅ Fixed Gym API compatibility issues
- ✅ Enhanced money progress detection logic
- ✅ Improved project organization
- ✅ Consolidated training pipeline

## Status

The project is fully functional with vision-enhanced LLM training capabilities.
All major compatibility issues have been resolved.
