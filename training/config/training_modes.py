"""
Authoritative Training Modes Configuration

This module provides the single source of truth for all training modes
used across the Pokemon Crystal RL system.
"""

from enum import Enum


class TrainingMode(Enum):
    """Comprehensive training modes for Pokemon Crystal RL system.
    
    This enum consolidates all training modes from across the codebase
    into a single authoritative definition.
    """
    
    # Fast training modes
    FAST_MONITORED = "fast_monitored"      # Fast training with monitoring
    ULTRA_FAST = "ultra_fast"              # Ultra-fast training, minimal overhead
    FAST = "fast"                          # Legacy alias for FAST_MONITORED
    
    # LLM-based training modes  
    LLM_ONLY = "llm_only"                  # Pure LLM decision making
    LLM_HYBRID = "llm_hybrid"              # LLM + RL hybrid approach
    LLM = "llm"                            # Legacy alias for LLM_ONLY
    
    # RL-based training modes
    RL_ONLY = "rl_only"                    # Pure reinforcement learning
    RULE_BASED = "rule_based"              # Rule-based fallback system
    
    # Progressive training modes
    CURRICULUM = "curriculum"              # Progressive difficulty training
    
    # Special modes
    HYBRID = "hybrid"                      # General hybrid approach
    CUSTOM = "custom"                      # Custom user-defined strategy
    EVALUATION = "evaluation"              # Evaluation/testing mode
    DEBUG = "debug"                        # Debug and development mode


class LLMBackend(Enum):
    """Available LLM backends for training."""
    
    NONE = None                            # No LLM backend
    SMOLLM2 = "smollm2:1.7b"              # SmolLM2 1.7B model
    LLAMA32_1B = "llama3.2:1b"            # Llama 3.2 1B model
    LLAMA32_3B = "llama3.2:3b"            # Llama 3.2 3B model  
    QWEN25_3B = "qwen2.5:3b"              # Qwen 2.5 3B model


__all__ = ['TrainingMode', 'LLMBackend']