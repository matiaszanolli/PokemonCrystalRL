"""
Decision Pattern Detection

Contains all pattern recognition algorithms for analyzing decision history.
"""

import logging
import statistics
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Any

from .models import DecisionRecord, DecisionPattern, PatternType, OutcomeType
from ..state.analyzer import GameStateAnalysis, SituationCriticality

logger = logging.getLogger(__name__)


class PatternDetector:
    """Detects patterns in decision history"""
    
    def __init__(self, min_pattern_observations: int = 3, success_threshold: float = 0.6,
                 stuck_detection_window: int = 10):
        self.min_pattern_observations = min_pattern_observations
        self.success_threshold = success_threshold
        self.stuck_detection_window = stuck_detection_window
        self.logger = logging.getLogger("pokemon_trainer.pattern_detector")
        self.analysis_stats = {"patterns_discovered": 0}
    
    def analyze_patterns(self, decision_history: deque, existing_patterns: Dict[str, DecisionPattern]) -> Dict[str, DecisionPattern]:
        """Analyze decision history for patterns"""
        if len(decision_history) < self.min_pattern_observations:
            return existing_patterns
        
        patterns = existing_patterns.copy()
        
        # Analyze different types of patterns
        patterns.update(self._detect_successful_sequences(decision_history))
        patterns.update(self._detect_failure_sequences(decision_history))
        patterns.update(self._detect_stuck_loops(decision_history))
        patterns.update(self._detect_context_patterns(decision_history))
        patterns.update(self._detect_phase_strategies(decision_history))
        patterns.update(self._detect_emergency_resolutions(decision_history))
        
        return patterns
    
    def _detect_successful_sequences(self, decision_history: deque) -> Dict[str, DecisionPattern]:
        """Detect sequences of actions that consistently lead to success"""
        patterns = {}
        window_size = 5
        min_success_rate = 0.8
        
        for window_start in range(len(decision_history) - window_size + 1):
            window = list(decision_history)[window_start:window_start + window_size]
            
            # Check if this is a successful sequence
            success_count = sum(1 for record in window 
                              if record.outcome_type in [OutcomeType.SUCCESS, OutcomeType.BREAKTHROUGH])
            success_rate = success_count / len(window)
            
            if success_rate >= min_success_rate:
                actions = [record.action_taken for record in window]
                context = window[0].game_state  # Use first state as context
                
                pattern_id = f"success_seq_{hash(tuple(actions + [str(context)]))}"
                
                if pattern_id not in patterns:
                    patterns[pattern_id] = DecisionPattern(
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
        
        return patterns
    
    def _detect_failure_sequences(self, decision_history: deque) -> Dict[str, DecisionPattern]:
        """Detect sequences that consistently lead to failure"""
        patterns = {}
        window_size = 3
        failure_threshold = 0.7
        
        for window_start in range(len(decision_history) - window_size + 1):
            window = list(decision_history)[window_start:window_start + window_size]
            
            failure_count = sum(1 for record in window 
                              if record.outcome_type == OutcomeType.FAILURE)
            failure_rate = failure_count / len(window)
            
            if failure_rate >= failure_threshold:
                actions = [record.action_taken for record in window]
                context = window[0].game_state
                
                pattern_id = f"failure_seq_{hash(tuple(actions + [str(context)]))}"
                
                if pattern_id not in patterns:
                    patterns[pattern_id] = DecisionPattern(
                        pattern_id=pattern_id,
                        pattern_type=PatternType.FAILURE_SEQUENCE,
                        action_sequence=actions,
                        context_conditions=context,
                        success_rate=1.0 - failure_rate,
                        average_reward=statistics.mean(r.reward_received for r in window),
                        confidence=failure_rate,
                        avoid_contexts=[f"{context['phase']}_{context['criticality']}"]
                    )
        
        return patterns
    
    def _detect_stuck_loops(self, decision_history: deque) -> Dict[str, DecisionPattern]:
        """Detect when the agent gets stuck in repetitive patterns"""
        patterns = {}
        
        if len(decision_history) < self.stuck_detection_window:
            return patterns
        
        recent_actions = [r.action_taken for r in list(decision_history)[-self.stuck_detection_window:]]
        
        # Check for repetitive patterns
        if len(set(recent_actions)) <= 2:  # Only 1-2 unique actions
            recent_rewards = [r.reward_received for r in list(decision_history)[-self.stuck_detection_window:]]
            avg_reward = statistics.mean(recent_rewards)
            
            if avg_reward <= 0.1:  # Low average reward indicates stuck
                pattern_id = f"stuck_loop_{hash(tuple(recent_actions))}"
                
                if pattern_id not in patterns:
                    context = list(decision_history)[-1].game_state
                    
                    patterns[pattern_id] = DecisionPattern(
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
        
        return patterns
    
    def _detect_context_patterns(self, decision_history: deque) -> Dict[str, DecisionPattern]:
        """Detect actions that work well in specific contexts"""
        patterns = {}
        context_successes = defaultdict(lambda: defaultdict(list))
        
        # Group decisions by context and action
        for record in decision_history:
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
                        
                        if pattern_id not in patterns:
                            avg_reward = statistics.mean(r.reward_received for r in records)
                            
                            patterns[pattern_id] = DecisionPattern(
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
        
        return patterns
    
    def _detect_phase_strategies(self, decision_history: deque) -> Dict[str, DecisionPattern]:
        """Detect successful strategies for different game phases"""
        patterns = {}
        phase_data = defaultdict(list)
        
        # Group by game phase
        for record in decision_history:
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
                    
                    if pattern_id not in patterns:
                        patterns[pattern_id] = DecisionPattern(
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
        
        return patterns
    
    def _detect_emergency_resolutions(self, decision_history: deque) -> Dict[str, DecisionPattern]:
        """Detect patterns for resolving emergency situations"""
        patterns = {}
        emergency_records = [r for r in decision_history 
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
                        
                        if pattern_id not in patterns:
                            patterns[pattern_id] = DecisionPattern(
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
        
        return patterns
    
    def get_action_recommendations(self, current_state: GameStateAnalysis, 
                                 patterns: Dict[str, DecisionPattern],
                                 recent_actions: List[int] = None) -> List[Tuple[int, str, float]]:
        """
        Get action recommendations based on learned patterns
        
        Returns:
            List of (action, reasoning, confidence) tuples
        """
        recommendations = []
        current_context = {
            "phase": current_state.phase.value,
            "criticality": current_state.criticality.value,
            "health": current_state.health_percentage,
            "progression": current_state.progression_score
        }
        
        # Check for emergency patterns first
        if current_state.criticality == SituationCriticality.EMERGENCY:
            for pattern in patterns.values():
                if pattern.pattern_type == PatternType.EMERGENCY_RESOLUTION:
                    if self._context_matches(current_context, pattern.context_conditions):
                        for action in pattern.action_sequence:
                            recommendations.append((
                                action,
                                f"Emergency resolution pattern (success rate: {pattern.success_rate:.2f})",
                                pattern.confidence
                            ))
        
        # Check for phase strategies
        for pattern in patterns.values():
            if pattern.pattern_type == PatternType.PHASE_STRATEGY:
                if pattern.context_conditions.get("phase") == current_context["phase"]:
                    for action in pattern.action_sequence[:2]:  # Top 2 actions
                        recommendations.append((
                            action,
                            f"Phase strategy for {current_context['phase']} (avg reward: {pattern.average_reward:.2f})",
                            pattern.confidence
                        ))
        
        # Check for context-specific successes
        for pattern in patterns.values():
            if pattern.pattern_type == PatternType.CONTEXT_SUCCESS:
                if self._context_matches(current_context, pattern.context_conditions):
                    for action in pattern.action_sequence:
                        recommendations.append((
                            action,
                            f"Context-specific success (success rate: {pattern.success_rate:.2f})",
                            pattern.confidence
                        ))
        
        # Avoid actions from failure patterns and stuck loops
        avoid_actions = set()
        for pattern in patterns.values():
            if pattern.pattern_type in [PatternType.FAILURE_SEQUENCE, PatternType.STUCK_LOOP]:
                if self._context_matches(current_context, pattern.context_conditions):
                    avoid_actions.update(pattern.action_sequence)
        
        # Filter out actions to avoid
        recommendations = [(a, r, c) for a, r, c in recommendations if a not in avoid_actions]
        
        # Sort by confidence
        recommendations.sort(key=lambda x: x[2], reverse=True)
        
        return recommendations[:5]  # Top 5 recommendations
    
    def _context_matches(self, current_context: Dict[str, Any], 
                        pattern_context: Dict[str, Any], 
                        tolerance: float = 0.1) -> bool:
        """Check if current context matches pattern context with tolerance"""
        if not pattern_context:
            return True
        
        # Special handling for pattern contexts
        if "context" in pattern_context:
            context_str = pattern_context["context"]
            phase, criticality = context_str.split("_", 1)
            return (current_context["phase"] == phase and 
                    current_context["criticality"] == criticality)
        
        # Direct field matching with tolerance for numeric values
        for key, value in pattern_context.items():
            if key not in current_context:
                continue
                
            if isinstance(value, (int, float)) and isinstance(current_context[key], (int, float)):
                if abs(current_context[key] - value) > tolerance:
                    return False
            elif current_context[key] != value:
                return False
        
        return True