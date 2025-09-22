"""
Agents Package

This package contains all AI agents for Pokemon Crystal decision-making:
- LLM-based agents for intelligent decision making
- RL-based agents (DQN) for learned behavior  
- Hybrid agents combining multiple approaches
- Base agent abstractions
"""

from .base_agent import BaseAgent
from .llm_agent import LLMAgent
from .dqn_agent import DQNAgent
from .hybrid_agent import HybridAgent
# BasicHybridTrainer removed - was unused dead code

__all__ = [
    'BaseAgent',
    'LLMAgent', 
    'DQNAgent',
    'HybridAgent',
# 'BasicHybridTrainer'  # Removed - unused
]