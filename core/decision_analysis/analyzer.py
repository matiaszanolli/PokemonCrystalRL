"""
Main Decision History Analyzer

Orchestrates all components of the decision analysis system.
"""

import logging
from collections import deque
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from pathlib import Path

from .models import DecisionRecord, DecisionPattern, OutcomeType
from .database import DecisionDatabase
from .pattern_detector import PatternDetector
from environments.state.analyzer import GameStateAnalysis


class DecisionHistoryAnalyzer:
    """Analyzes decision history to extract patterns and learning insights"""
    
    def __init__(self, db_path: Optional[str] = None, max_history: int = 1000):
        self.logger = logging.getLogger("pokemon_trainer.history_analyzer")
        
        # Initialize components
        self.database = DecisionDatabase(db_path)
        self.pattern_detector = PatternDetector()
        
        # In-memory analysis structures
        self.max_history = max_history
        self.decision_history = deque(maxlen=max_history)
        self.patterns: Dict[str, DecisionPattern] = {}
        
        # Analysis configuration
        self.min_pattern_observations = 3
        self.success_threshold = 0.6
        self.stuck_detection_window = 10
        
        # Statistics
        self.analysis_stats = {
            "decisions_analyzed": 0,
            "patterns_discovered": 0,
            "recommendations_made": 0
        }
        
        # Load existing patterns from database
        self.patterns = self.database.load_patterns()
        
        self.logger.info(f"DecisionHistoryAnalyzer initialized with {len(self.patterns)} existing patterns")
    
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
        record_id = self.database.store_decision(record)
        
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
        
        # Use pattern detector to find new patterns
        self.patterns = self.pattern_detector.analyze_patterns(
            self.decision_history, 
            self.patterns
        )
        
        # Update stats
        self.analysis_stats["patterns_discovered"] = len(self.patterns)
    
    def analyze_patterns(self, min_frequency: int = 2) -> List[Dict[str, Any]]:
        """
        Analyze patterns and return summary statistics
        
        Returns:
            List of pattern summaries sorted by effectiveness
        """
        self._analyze_recent_patterns()
        
        pattern_summaries = []
        for pattern in self.patterns.values():
            if pattern.times_observed >= min_frequency:
                pattern_summaries.append({
                    'pattern_id': pattern.pattern_id,
                    'pattern_type': pattern.pattern_type.value,
                    'action_sequence': pattern.action_sequence,
                    'success_rate': pattern.success_rate,
                    'average_reward': pattern.average_reward,
                    'times_observed': pattern.times_observed,
                    'confidence': pattern.confidence,
                    'effectiveness_score': pattern.effectiveness_score,
                    'recommended_contexts': pattern.recommended_contexts,
                    'avoid_contexts': pattern.avoid_contexts
                })
        
        # Sort by effectiveness (combination of success rate and frequency)
        pattern_summaries.sort(
            key=lambda x: x['success_rate'] * x['times_observed'], 
            reverse=True
        )
        
        return pattern_summaries
    
    def get_action_recommendations(self, current_state: GameStateAnalysis, 
                                 recent_actions: List[int] = None) -> List[Tuple[int, str, float]]:
        """
        Get action recommendations based on learned patterns
        
        Args:
            current_state: Current game state
            recent_actions: Recent actions taken (for stuck detection)
            
        Returns:
            List of (action, reasoning, confidence) tuples
        """
        self.analysis_stats["recommendations_made"] += 1
        
        return self.pattern_detector.get_action_recommendations(
            current_state, 
            self.patterns, 
            recent_actions
        )
    
    def get_patterns_summary(self) -> Dict[str, Any]:
        """Get a summary of discovered patterns"""
        pattern_types = {}
        total_patterns = len(self.patterns)
        
        for pattern in self.patterns.values():
            pattern_type = pattern.pattern_type.value
            if pattern_type not in pattern_types:
                pattern_types[pattern_type] = 0
            pattern_types[pattern_type] += 1
        
        return {
            'total_patterns': total_patterns,
            'pattern_types': pattern_types,
            'analysis_stats': self.analysis_stats.copy(),
            'decision_history_size': len(self.decision_history),
            'most_confident_patterns': [
                {
                    'pattern_id': p.pattern_id,
                    'pattern_type': p.pattern_type.value,
                    'confidence': p.confidence,
                    'success_rate': p.success_rate
                }
                for p in sorted(self.patterns.values(), key=lambda x: x.confidence, reverse=True)[:5]
            ]
        }
    
    def save_patterns_to_db(self):
        """Save current patterns to database"""
        self.database.save_patterns(self.patterns)
    
    def add_decision(self, decision_data: Dict[str, Any]):
        """Add decision (compatibility method for tests)"""
        # Convert test format to proper format
        from environments.state.analyzer import GameStateAnalysis, GamePhase, SituationCriticality
        
        # Mock a game state analysis for compatibility
        mock_state = type('MockState', (), {
            'phase': GamePhase(decision_data.get('phase', 'early_game')),
            'criticality': SituationCriticality(decision_data.get('criticality', 'normal')),
            'health_percentage': decision_data.get('health_percentage', 100.0),
            'progression_score': decision_data.get('progression_score', 0.0),
            'immediate_threats': decision_data.get('threats', []),
            'opportunities': decision_data.get('opportunities', []),
            'state_variables': {}
        })()
        
        return self.record_decision(
            game_state=mock_state,
            action_taken=decision_data['action'],
            llm_reasoning=decision_data.get('reasoning'),
            reward_received=decision_data['reward'],
            led_to_progress=decision_data.get('led_to_progress', False),
            was_effective=decision_data.get('was_effective', True)
        )
    
    def get_recent_decisions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent decisions"""
        return self.database.get_recent_decisions(limit)
    
    def close(self):
        """Clean up resources"""
        self.save_patterns_to_db()
        self.database.close()