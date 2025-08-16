#!/usr/bin/env python3
"""
Pokemon Crystal RL Trainer v2.0 - Production Edition

Unified entry point for the comprehensive Pokemon Crystal RL training system.
This wrapper provides access to the full production-grade trainer with:

- Multiple training modes (ultra-fast, synchronized, curriculum)
- Local LLM integration with zero API costs
- Real-time web monitoring with 10x optimized streaming
- Professional logging and crash-free operation
- Production-grade error handling and recovery

Usage Examples:
  # Quick production demo
  python run_pokemon_trainer.py --rom crystal.gbc --quick
  
  # Full production training
  python run_pokemon_trainer.py --rom crystal.gbc --production
  
  # Content creation mode
  python run_pokemon_trainer.py --rom crystal.gbc --windowed --web --hq
"""

import sys
import os

# Add the parent directory to Python path so we can import as a package
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the enhanced production trainer
from scripts.pokemon_trainer import main

if __name__ == "__main__":
    exit(main())
