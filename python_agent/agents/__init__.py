"""
Agents module for Pokemon Crystal RL

Contains AI agents and LLM interfaces for intelligent gameplay.
"""

__all__ = []

# Try to import agents with external dependencies
try:
    from .local_llm_agent import LocalLLMPokemonAgent
    __all__.append('LocalLLMPokemonAgent')
except ImportError:
    pass

try:
    from .enhanced_llm_agent import EnhancedLLMPokemonAgent
    __all__.append('EnhancedLLMPokemonAgent')
except ImportError:
    pass
