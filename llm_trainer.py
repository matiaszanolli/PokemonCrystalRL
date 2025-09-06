#!/usr/bin/env python3

### DEPRECATION NOTICE ###
# This module has been split into:
# - main.py: Entry point and orchestration
# - trainer/llm_pokemon_trainer.py: Core trainer implementation
# - config/memory_addresses.py: Memory address mappings
# - utils/memory_reader.py: Memory reading utilities
# - agents/llm_agent.py: LLM agent implementation
# - rewards/calculator.py: Reward calculation system
#
# Please use main.py as the entry point instead of this file.
### END NOTICE ###

import warnings
warnings.warn(
    'llm_trainer.py is deprecated. Use main.py as entry point instead.',
    DeprecationWarning,
    stacklevel=2
)

# Re-export key components for backwards compatibility
from config.memory_addresses import MEMORY_ADDRESSES
from utils.memory_reader import build_observation
from trainer.llm_pokemon_trainer import LLMPokemonTrainer
from rewards.calculator import PokemonRewardCalculator
from agents.llm_agent import LLMAgent

# Alias main entry point
from main import main

if __name__ == '__main__':
    main()