"""
Agents module for Pokemon Crystal RL

This module contains AI agents for Pokemon Crystal decision-making,
including LLM-based agents and rule-based fallback systems.
"""

from .llm_agent import LLMAgent

__all__ = ['LLMAgent']