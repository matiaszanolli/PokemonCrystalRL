"""
Decision Analysis Data Models

Contains all data classes and enums used throughout the decision analysis system.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from ..state.analyzer import GamePhase, SituationCriticality


class OutcomeType(Enum):
    """Types of decision outcomes"""
    SUCCESS = "success"           # Led to positive reward/progress
    FAILURE = "failure"           # Led to negative reward/setback  
    NEUTRAL = "neutral"           # No significant impact
    STUCK = "stuck"               # Led to being stuck/repetitive
    BREAKTHROUGH = "breakthrough"  # Broke out of stuck state
    EMERGENCY = "emergency"       # Emergency situation resolution


class PatternType(Enum):
    """Types of patterns detected in decision history"""
    SUCCESSFUL_SEQUENCE = "successful_sequence"     # Actions that consistently work well
    FAILURE_SEQUENCE = "failure_sequence"          # Actions that consistently fail
    STUCK_LOOP = "stuck_loop"                      # Repetitive actions causing no progress
    CONTEXT_SUCCESS = "context_success"            # Actions that work in specific contexts
    PHASE_STRATEGY = "phase_strategy"              # Strategies for specific game phases
    EMERGENCY_RESOLUTION = "emergency_resolution"   # How to handle critical situations


@dataclass
class DecisionRecord:
    """Individual decision record for analysis"""
    timestamp: datetime
    game_state: Dict[str, Any]  # Simplified state snapshot
    action_taken: int
    llm_reasoning: Optional[str]
    reward_received: float
    outcome_type: OutcomeType
    
    # Context
    game_phase: GamePhase
    criticality: SituationCriticality  
    health_percentage: float
    progression_score: float
    
    # Results
    led_to_progress: bool
    was_effective: bool
    time_to_next_decision: float = 0.0


@dataclass  
class DecisionPattern:
    """A recognized pattern in decision history"""
    pattern_id: str
    pattern_type: PatternType
    action_sequence: List[int]
    context_conditions: Dict[str, Any]
    
    # Statistics
    times_observed: int = 0
    success_rate: float = 0.0
    average_reward: float = 0.0
    effectiveness_score: float = 0.0
    
    # Learning
    confidence: float = 0.0
    last_seen: datetime = field(default_factory=datetime.now)
    examples: List[str] = field(default_factory=list)  # Decision record IDs
    
    # Recommendations
    recommended_contexts: List[str] = field(default_factory=list)
    avoid_contexts: List[str] = field(default_factory=list)