"""
Agents Package

This package contains all AI agents for Pokemon Crystal decision-making:
- LLM-based agents for intelligent decision making
- RL-based agents (DQN) for learned behavior
- Hybrid agents combining multiple approaches
- Specialist agents for specific game aspects
- Multi-agent coordination system
- Base agent abstractions
"""

from .base_agent import BaseAgent
from .llm_agent import LLMAgent
from .dqn_agent import DQNAgent
from .hybrid_agent import HybridAgent

# Specialist agents for Multi-Agent Framework
from .battle_agent import BattleAgent
from .explorer_agent import ExplorerAgent
from .progression_agent import ProgressionAgent
from .multi_agent_coordinator import MultiAgentCoordinator

__all__ = [
    'BaseAgent',
    'LLMAgent',
    'DQNAgent',
    'HybridAgent',
    # Multi-Agent Framework
    'BattleAgent',
    'ExplorerAgent',
    'ProgressionAgent',
    'MultiAgentCoordinator',
]