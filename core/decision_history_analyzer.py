#!/usr/bin/env python3
"""
Decision History Analysis System for Pokemon Crystal RL

This module implements pattern recognition and learning from decision history
as specified in ROADMAP_ENHANCED Phase 2.2
"""

from typing import Dict, List, Tuple, Optional, Any, NamedTuple
from dataclasses import dataclass, field
from collections import deque, defaultdict
from enum import Enum
import json
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
import statistics
import logging

from .game_state_analyzer import GameStateAnalysis, GamePhase, SituationCriticality

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

class DecisionHistoryAnalyzer:
    """Analyzes decision history to extract patterns and learning insights"""
    
    def __init__(self, db_path: Optional[str] = None, max_history: int = 1000):
        self.logger = logging.getLogger("pokemon_trainer.history_analyzer")
        
        # Database setup
        self.db_path = Path(db_path) if db_path else Path("decision_history.db")
        self._init_database()
        
        # In-memory analysis structures
        self.max_history = max_history
        self.decision_history = deque(maxlen=max_history)
        self.patterns: Dict[str, DecisionPattern] = {}
        
        # Analysis configuration
        self.min_pattern_observations = 3
        self.success_threshold = 0.7
        self.stuck_detection_window = 10
        
        # Performance tracking
        self.analysis_stats = {
            "patterns_discovered": 0,
            "decisions_analyzed": 0,
            "successful_predictions": 0,
            "failed_predictions": 0
        }
        
        # Load existing patterns
        self._load_patterns_from_db()
    
    def _init_database(self):
        """Initialize SQLite database for persistent storage"""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Decision history table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS decision_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    game_state TEXT,
                    action_taken INTEGER,
                    llm_reasoning TEXT,
                    reward_received REAL,
                    outcome_type TEXT,
                    game_phase TEXT,
                    criticality TEXT,
                    health_percentage REAL,
                    progression_score REAL,
                    led_to_progress BOOLEAN,
                    was_effective BOOLEAN,
                    time_to_next_decision REAL
                )
            ''')
            
            # Pattern discovery table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS decision_patterns (
                    pattern_id TEXT PRIMARY KEY,
                    pattern_type TEXT,
                    action_sequence TEXT,
                    context_conditions TEXT,
                    times_observed INTEGER,
                    success_rate REAL,
                    average_reward REAL,
                    effectiveness_score REAL,
                    confidence REAL,
                    last_seen DATETIME,
                    examples TEXT,
                    recommended_contexts TEXT,
                    avoid_contexts TEXT
                )
            ''')
            
            # Performance metrics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS analysis_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    metric_name TEXT,
                    metric_value REAL,
                    context TEXT
                )
            ''')
            
            conn.commit()
    
    def record_decision(self, game_state: GameStateAnalysis, action_taken: int,
                       llm_reasoning: Optional[str], reward_received: float,
                       led_to_progress: bool = False, was_effective: bool = True) -> str:
        """
        Record a decision for analysis
        
        Args:
            game_state: Current game state analysis
            action_taken: Action that was taken (1-8)
            llm_reasoning: LLM's reasoning for the decision
            reward_received: Reward received after the action
            led_to_progress: Whether the action led to meaningful progress
            was_effective: Whether the action was effective
            
        Returns:
            Decision record ID for reference
        """
        
        # Determine outcome type
        outcome_type = self._classify_outcome(reward_received, led_to_progress, was_effective)
        
        # Create decision record
        record = DecisionRecord(
            timestamp=datetime.now(),
            game_state=self._simplify_state(game_state),
            action_taken=action_taken,
            llm_reasoning=llm_reasoning,
            reward_received=reward_received,
            outcome_type=outcome_type,
            game_phase=game_state.phase,
            criticality=game_state.criticality,
            health_percentage=game_state.health_percentage,
            progression_score=game_state.progression_score,
            led_to_progress=led_to_progress,
            was_effective=was_effective
        )
        
        # Store in memory
        self.decision_history.append(record)
        
        # Store in database
        record_id = self._store_decision_in_db(record)
        
        # Analyze patterns with new data
        self._analyze_recent_patterns()
        
        # Update statistics
        self.analysis_stats["decisions_analyzed"] += 1
        
        return record_id
    
    def _classify_outcome(self, reward: float, led_to_progress: bool, was_effective: bool) -> OutcomeType:
        """Classify the outcome of a decision"""
        if reward > 2.0 or led_to_progress:
            return OutcomeType.SUCCESS
        elif reward < -1.0 or not was_effective:
            return OutcomeType.FAILURE
        elif abs(reward) < 0.1:
            return OutcomeType.NEUTRAL
        else:
            return OutcomeType.SUCCESS if reward > 0 else OutcomeType.FAILURE
    
    def _simplify_state(self, analysis: GameStateAnalysis) -> Dict[str, Any]:
        """Create simplified state representation for pattern matching"""
        return {
            "phase": analysis.phase.value,
            "criticality": analysis.criticality.value,
            "health_percentage": round(analysis.health_percentage, 1),
            "progression_score": round(analysis.progression_score, 1),
            "has_threats": len(analysis.immediate_threats) > 0,
            "has_opportunities": len(analysis.opportunities) > 0,
            # Add key state variables
            "key_variables": {
                var_name: var.current_value 
                for var_name, var in analysis.state_variables.items() 
                if var_name in ["in_battle", "party_count", "badges", "player_map"]
            }
        }
    
    def _analyze_recent_patterns(self):
        """Analyze recent decision history for patterns"""
        if len(self.decision_history) < self.min_pattern_observations:
            return
        
        # Analyze different types of patterns
        self._detect_successful_sequences()
        self._detect_failure_sequences()
        self._detect_stuck_loops()
        self._detect_context_patterns()
        self._detect_phase_strategies()
        self._detect_emergency_resolutions()
    
    def _detect_successful_sequences(self):
        """Detect sequences of actions that consistently lead to success"""
        window_size = 5
        min_success_rate = 0.8
        
        for window_start in range(len(self.decision_history) - window_size + 1):
            window = list(self.decision_history)[window_start:window_start + window_size]
            
            # Check if this is a successful sequence
            success_count = sum(1 for record in window 
                              if record.outcome_type in [OutcomeType.SUCCESS, OutcomeType.BREAKTHROUGH])
            success_rate = success_count / len(window)
            
            if success_rate >= min_success_rate:
                actions = [record.action_taken for record in window]
                context = window[0].game_state  # Use first state as context
                
                pattern_id = f"success_seq_{hash(tuple(actions + [str(context)]))}"
                
                if pattern_id not in self.patterns:
                    self.patterns[pattern_id] = DecisionPattern(
                        pattern_id=pattern_id,
                        pattern_type=PatternType.SUCCESSFUL_SEQUENCE,
                        action_sequence=actions,
                        context_conditions=context,
                        success_rate=success_rate,
                        average_reward=statistics.mean(r.reward_received for r in window),
                        confidence=min(success_rate, len(window) / 10.0)
                    )
                    
                    self.analysis_stats["patterns_discovered"] += 1
                    self.logger.info(f"Discovered successful sequence: {actions} with {success_rate:.2f} success rate")
    
    def _detect_failure_sequences(self):
        """Detect sequences that consistently lead to failure"""
        window_size = 3
        failure_threshold = 0.7
        
        for window_start in range(len(self.decision_history) - window_size + 1):
            window = list(self.decision_history)[window_start:window_start + window_size]
            
            failure_count = sum(1 for record in window 
                              if record.outcome_type == OutcomeType.FAILURE)
            failure_rate = failure_count / len(window)
            
            if failure_rate >= failure_threshold:
                actions = [record.action_taken for record in window]
                context = window[0].game_state
                
                pattern_id = f"failure_seq_{hash(tuple(actions + [str(context)]))}"
                
                if pattern_id not in self.patterns:
                    self.patterns[pattern_id] = DecisionPattern(
                        pattern_id=pattern_id,
                        pattern_type=PatternType.FAILURE_SEQUENCE,
                        action_sequence=actions,
                        context_conditions=context,
                        success_rate=1.0 - failure_rate,
                        average_reward=statistics.mean(r.reward_received for r in window),
                        confidence=failure_rate,
                        avoid_contexts=[f"{context['phase']}_{context['criticality']}"]
                    )
    
    def _detect_stuck_loops(self):
        """Detect when the agent gets stuck in repetitive patterns"""
        if len(self.decision_history) < self.stuck_detection_window:
            return
        
        recent_actions = [r.action_taken for r in list(self.decision_history)[-self.stuck_detection_window:]]
        
        # Check for repetitive patterns
        if len(set(recent_actions)) <= 2:  # Only 1-2 unique actions
            recent_rewards = [r.reward_received for r in list(self.decision_history)[-self.stuck_detection_window:]]
            avg_reward = statistics.mean(recent_rewards)
            
            if avg_reward <= 0.1:  # Low average reward indicates stuck
                pattern_id = f"stuck_loop_{hash(tuple(recent_actions))}"
                
                if pattern_id not in self.patterns:
                    context = list(self.decision_history)[-1].game_state
                    
                    self.patterns[pattern_id] = DecisionPattern(
                        pattern_id=pattern_id,
                        pattern_type=PatternType.STUCK_LOOP,
                        action_sequence=list(set(recent_actions)),  # Unique actions in loop
                        context_conditions=context,
                        success_rate=0.0,
                        average_reward=avg_reward,
                        confidence=0.9,
                        avoid_contexts=[f"stuck_situation_{context['phase']}"]
                    )
                    
                    self.logger.warning(f"Detected stuck loop: {recent_actions}")
    
    def _detect_context_patterns(self):
        """Detect actions that work well in specific contexts"""
        context_successes = defaultdict(lambda: defaultdict(list))
        
        # Group decisions by context and action
        for record in self.decision_history:
            context_key = f"{record.game_phase.value}_{record.criticality.value}"
            action = record.action_taken
            context_successes[context_key][action].append(record)
        
        # Find context-action pairs with high success rates
        for context, action_records in context_successes.items():
            for action, records in action_records.items():
                if len(records) >= self.min_pattern_observations:
                    success_count = sum(1 for r in records if r.outcome_type == OutcomeType.SUCCESS)
                    success_rate = success_count / len(records)
                    
                    if success_rate >= self.success_threshold:
                        pattern_id = f"context_success_{context}_{action}"
                        
                        if pattern_id not in self.patterns:
                            avg_reward = statistics.mean(r.reward_received for r in records)
                            
                            self.patterns[pattern_id] = DecisionPattern(
                                pattern_id=pattern_id,
                                pattern_type=PatternType.CONTEXT_SUCCESS,
                                action_sequence=[action],
                                context_conditions={"context": context},
                                success_rate=success_rate,
                                average_reward=avg_reward,
                                times_observed=len(records),
                                confidence=min(success_rate, len(records) / 10.0),
                                recommended_contexts=[context]
                            )
    
    def _detect_phase_strategies(self):
        """Detect successful strategies for different game phases"""
        phase_data = defaultdict(list)
        
        # Group by game phase
        for record in self.decision_history:
            phase_data[record.game_phase].append(record)
        
        # Analyze each phase
        for phase, records in phase_data.items():
            if len(records) >= 10:  # Need substantial data
                # Find most successful actions in this phase
                action_success = defaultdict(list)
                for record in records:
                    action_success[record.action_taken].append(record)
                
                # Identify top performing actions
                best_actions = []
                for action, action_records in action_success.items():
                    if len(action_records) >= 3:
                        avg_reward = statistics.mean(r.reward_received for r in action_records)
                        success_rate = sum(1 for r in action_records if r.outcome_type == OutcomeType.SUCCESS) / len(action_records)
                        
                        if success_rate >= 0.6 and avg_reward > 0.5:
                            best_actions.append((action, success_rate, avg_reward))
                
                if best_actions:
                    # Create phase strategy pattern
                    best_actions.sort(key=lambda x: x[2], reverse=True)  # Sort by avg reward
                    top_actions = [action for action, _, _ in best_actions[:3]]
                    
                    pattern_id = f"phase_strategy_{phase.value}"
                    
                    if pattern_id not in self.patterns:
                        self.patterns[pattern_id] = DecisionPattern(
                            pattern_id=pattern_id,
                            pattern_type=PatternType.PHASE_STRATEGY,
                            action_sequence=top_actions,
                            context_conditions={"phase": phase.value},
                            success_rate=statistics.mean(sr for _, sr, _ in best_actions[:3]),
                            average_reward=statistics.mean(ar for _, _, ar in best_actions[:3]),
                            times_observed=len(records),
                            confidence=min(len(records) / 20.0, 1.0),
                            recommended_contexts=[phase.value]
                        )
    
    def _detect_emergency_resolutions(self):
        """Detect patterns for resolving emergency situations"""
        emergency_records = [r for r in self.decision_history 
                           if r.criticality == SituationCriticality.EMERGENCY]
        
        if len(emergency_records) >= 3:
            resolution_patterns = defaultdict(list)
            
            for record in emergency_records:
                # Look for what action was taken and if it resolved the emergency
                resolution_patterns[record.action_taken].append(record)
            
            for action, records in resolution_patterns.items():
                if len(records) >= 2:
                    # Check how often this action resolves emergencies
                    resolution_rate = sum(1 for r in records 
                                        if r.outcome_type in [OutcomeType.SUCCESS, OutcomeType.BREAKTHROUGH]) / len(records)
                    
                    if resolution_rate >= 0.5:  # 50% or better resolution rate
                        pattern_id = f"emergency_resolution_{action}"
                        
                        if pattern_id not in self.patterns:
                            self.patterns[pattern_id] = DecisionPattern(
                                pattern_id=pattern_id,
                                pattern_type=PatternType.EMERGENCY_RESOLUTION,
                                action_sequence=[action],
                                context_conditions={"criticality": "emergency"},
                                success_rate=resolution_rate,
                                average_reward=statistics.mean(r.reward_received for r in records),
                                times_observed=len(records),
                                confidence=resolution_rate,
                                recommended_contexts=["emergency_situations"]
                            )
    
    def get_action_recommendations(self, current_state: GameStateAnalysis, 
                                 recent_actions: List[int] = None) -> List[Tuple[int, str, float]]:
        """
        Get action recommendations based on learned patterns
        
        Args:
            current_state: Current game state
            recent_actions: Recent actions taken (for stuck detection)
            
        Returns:
            List of (action, reason, confidence) recommendations
        """
        recommendations = []
        current_context = self._simplify_state(current_state)
        
        # Check for stuck patterns to avoid
        if recent_actions and len(recent_actions) >= 3:
            for pattern in self.patterns.values():
                if pattern.pattern_type == PatternType.STUCK_LOOP:
                    if set(recent_actions[-3:]).issubset(set(pattern.action_sequence)):
                        # Recommend breaking out of this pattern
                        alternative_actions = [1, 2, 3, 4, 5]  # Basic actions
                        for alt_action in alternative_actions:
                            if alt_action not in recent_actions[-3:]:
                                recommendations.append((
                                    alt_action,
                                    f"Break stuck pattern: {pattern.action_sequence}",
                                    0.8
                                ))
                                break
        
        # Look for successful patterns that match current context
        for pattern in self.patterns.values():
            if pattern.pattern_type == PatternType.CONTEXT_SUCCESS:
                if self._context_matches(current_context, pattern.context_conditions):
                    action = pattern.action_sequence[0]
                    recommendations.append((
                        action,
                        f"Successful in {pattern.context_conditions.get('context', 'similar context')} ({pattern.success_rate:.1%} success rate)",
                        pattern.confidence
                    ))
            
            elif pattern.pattern_type == PatternType.PHASE_STRATEGY:
                if current_state.phase.value == pattern.context_conditions.get("phase"):
                    for action in pattern.action_sequence[:2]:  # Top 2 actions
                        recommendations.append((
                            action,
                            f"Phase strategy for {current_state.phase.value}",
                            pattern.confidence * 0.8
                        ))
            
            elif pattern.pattern_type == PatternType.EMERGENCY_RESOLUTION:
                if current_state.criticality == SituationCriticality.EMERGENCY:
                    action = pattern.action_sequence[0]
                    recommendations.append((
                        action,
                        f"Emergency resolution ({pattern.success_rate:.1%} effective)",
                        pattern.confidence
                    ))
        
        # Sort by confidence and return top recommendations
        recommendations.sort(key=lambda x: x[2], reverse=True)
        return recommendations[:5]
    
    def _context_matches(self, current_context: Dict[str, Any], 
                        pattern_context: Dict[str, Any]) -> bool:
        """Check if current context matches pattern context"""
        if "context" in pattern_context:
            expected_context = pattern_context["context"]
            current_context_key = f"{current_context['phase']}_{current_context['criticality']}"
            return expected_context == current_context_key
        
        # Check other context conditions
        for key, value in pattern_context.items():
            if key in current_context:
                if current_context[key] != value:
                    return False
        
        return True
    
    def get_patterns_summary(self) -> Dict[str, Any]:
        """Get summary of discovered patterns"""
        pattern_counts = defaultdict(int)
        for pattern in self.patterns.values():
            pattern_counts[pattern.pattern_type.value] += 1
        
        return {
            "total_patterns": len(self.patterns),
            "pattern_types": dict(pattern_counts),
            "decisions_analyzed": self.analysis_stats["decisions_analyzed"],
            "patterns_discovered": self.analysis_stats["patterns_discovered"],
            "top_patterns": [
                {
                    "id": p.pattern_id,
                    "type": p.pattern_type.value,
                    "success_rate": p.success_rate,
                    "confidence": p.confidence,
                    "observations": p.times_observed
                }
                for p in sorted(self.patterns.values(), 
                              key=lambda x: x.confidence * x.success_rate, reverse=True)[:5]
            ]
        }
    
    def _store_decision_in_db(self, record: DecisionRecord) -> str:
        """Store decision record in database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO decision_history 
                (timestamp, game_state, action_taken, llm_reasoning, reward_received,
                 outcome_type, game_phase, criticality, health_percentage, progression_score,
                 led_to_progress, was_effective, time_to_next_decision)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                record.timestamp,
                json.dumps(record.game_state),
                record.action_taken,
                record.llm_reasoning,
                record.reward_received,
                record.outcome_type.value,
                record.game_phase.value,
                record.criticality.value,
                record.health_percentage,
                record.progression_score,
                record.led_to_progress,
                record.was_effective,
                record.time_to_next_decision
            ))
            
            return str(cursor.lastrowid)
    
    def _load_patterns_from_db(self):
        """Load existing patterns from database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM decision_patterns')
                
                for row in cursor.fetchall():
                    (pattern_id, pattern_type, action_sequence, context_conditions,
                     times_observed, success_rate, average_reward, effectiveness_score,
                     confidence, last_seen, examples, recommended_contexts, avoid_contexts) = row
                    
                    pattern = DecisionPattern(
                        pattern_id=pattern_id,
                        pattern_type=PatternType(pattern_type),
                        action_sequence=json.loads(action_sequence),
                        context_conditions=json.loads(context_conditions),
                        times_observed=times_observed,
                        success_rate=success_rate,
                        average_reward=average_reward,
                        effectiveness_score=effectiveness_score,
                        confidence=confidence,
                        last_seen=datetime.fromisoformat(last_seen),
                        examples=json.loads(examples) if examples else [],
                        recommended_contexts=json.loads(recommended_contexts) if recommended_contexts else [],
                        avoid_contexts=json.loads(avoid_contexts) if avoid_contexts else []
                    )
                    
                    self.patterns[pattern_id] = pattern
                    
        except (sqlite3.Error, json.JSONDecodeError, ValueError) as e:
            self.logger.warning(f"Could not load patterns from database: {e}")
    
    def save_patterns_to_db(self):
        """Save current patterns to database"""
        try:
            # Ensure parent directory exists
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                for pattern in self.patterns.values():
                    cursor.execute('''
                        INSERT OR REPLACE INTO decision_patterns VALUES 
                        (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                    pattern.pattern_id,
                    pattern.pattern_type.value,
                    json.dumps(pattern.action_sequence),
                    json.dumps(pattern.context_conditions),
                    pattern.times_observed,
                    pattern.success_rate,
                    pattern.average_reward,
                    pattern.effectiveness_score,
                    pattern.confidence,
                    pattern.last_seen.isoformat(),
                    json.dumps(pattern.examples),
                    json.dumps(pattern.recommended_contexts),
                    json.dumps(pattern.avoid_contexts)
                ))
            
                conn.commit()
        except (sqlite3.Error, OSError) as e:
            self.logger.warning(f"Could not save patterns to database: {e}")
    
    def add_decision(self, decision_data: Dict[str, Any]):
        """Add decision (compatibility method for tests)"""
        # Convert test format to proper format
        from .game_state_analyzer import GameStateAnalysis, GamePhase, SituationCriticality
        
        # Mock a game state analysis for compatibility
        mock_state = GameStateAnalysis(
            phase=GamePhase.EXPLORATION,
            criticality=SituationCriticality.MODERATE,
            health_percentage=80.0,
            progression_score=50.0,
            exploration_score=30.0,
            immediate_threats=[],
            opportunities=[],
            recommended_priorities=["exploration", "survival"],
            situation_summary="Test situation for compatibility",
            strategic_context="Player exploring with moderate health",
            risk_assessment="Low risk situation",
            state_variables={}
        )
        
        action = decision_data.get('action', 0)
        llm_reasoning = decision_data.get('context', {}).get('reasoning', 'Test decision')
        reward = decision_data.get('total_episode_reward', 0.0)
        was_effective = decision_data.get('outcome') == 'success'
        led_to_progress = reward > 5.0
        
        self.record_decision(mock_state, action, llm_reasoning, reward, led_to_progress, was_effective)
    
    def get_recent_decisions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent decisions (compatibility method for tests)"""
        recent = list(self.decision_history)[-limit:]
        return [{
            'action': r.action_taken,
            'reward': r.reward_received,
            'outcome': r.outcome_type.value,
            'timestamp': r.timestamp,
            'reasoning': r.llm_reasoning
        } for r in recent]
    
    def close(self):
        """Close the analyzer (compatibility method for tests)"""
        # Save patterns before closing
        self.save_patterns_to_db()
        # No other cleanup needed for SQLite


if __name__ == "__main__":
    # Example usage
    analyzer = DecisionHistoryAnalyzer()
    print("Decision History Analyzer initialized")
    print(f"Patterns loaded: {len(analyzer.patterns)}")
    
    # Example pattern summary
    summary = analyzer.get_patterns_summary()
    print(f"Analysis summary: {summary}")