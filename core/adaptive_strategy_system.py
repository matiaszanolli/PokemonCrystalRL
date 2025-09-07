#!/usr/bin/env python3
"""
Adaptive Strategy System for Pokemon Crystal RL

This module implements adaptive strategy selection based on success/failure patterns,
adjusting LLM decision frequency and using rule-based fallbacks as specified in
ROADMAP_ENHANCED Phase 2.2
"""

from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
import time
import json
import logging
from datetime import datetime, timedelta
import statistics

from environments.state.analyzer import GameStateAnalysis, GamePhase, SituationCriticality
from .decision_history_analyzer import DecisionHistoryAnalyzer, OutcomeType, PatternType
from .goal_oriented_planner import GoalOrientedPlanner

class StrategyType(Enum):
    """Types of strategies available"""
    LLM_HEAVY = "llm_heavy"             # Frequent LLM decisions
    LLM_MODERATE = "llm_moderate"       # Balanced LLM/rule-based  
    LLM_MINIMAL = "llm_minimal"         # Minimal LLM, mostly rules
    RULE_BASED = "rule_based"           # Pure rule-based decisions
    EMERGENCY = "emergency"             # Emergency protocols
    LEARNING = "learning"               # Active learning mode
    BALANCED = "balanced"               # Alias for moderate (for test compatibility)

class DecisionSource(Enum):
    """Source of decision"""
    LLM = "llm"
    RULE_BASED = "rule_based"
    PATTERN_BASED = "pattern_based"
    GOAL_DIRECTED = "goal_directed"
    EMERGENCY_OVERRIDE = "emergency_override"

@dataclass
class StrategyConfig:
    """Configuration for a strategy"""
    strategy_type: StrategyType
    llm_frequency: float  # 0.0 to 1.0, how often to use LLM
    confidence_threshold: float  # Minimum confidence for LLM decisions
    rule_based_actions: Dict[str, List[int]]  # Context -> preferred actions
    emergency_actions: Dict[str, int]  # Emergency -> immediate action
    learning_rate: float = 0.1  # How fast to adapt
    success_threshold: float = 0.7  # Success rate to maintain strategy
    failure_threshold: float = 0.3  # Failure rate to change strategy

@dataclass
class PerformanceMetrics:
    """Performance tracking for strategies"""
    decisions_made: int = 0
    successes: int = 0
    failures: int = 0
    average_reward: float = 0.0
    time_to_progress: float = 0.0
    stuck_incidents: int = 0
    emergency_resolutions: int = 0
    last_success_time: Optional[datetime] = None

class AdaptiveStrategySystem:
    """Manages adaptive strategy selection based on performance"""
    
    def __init__(self, history_analyzer: Optional[DecisionHistoryAnalyzer] = None,
                 goal_planner: Optional[GoalOrientedPlanner] = None):
        self.logger = logging.getLogger("pokemon_trainer.adaptive_strategy")
        
        # Core components
        self.history_analyzer = history_analyzer or DecisionHistoryAnalyzer()
        self.goal_planner = goal_planner or GoalOrientedPlanner()
        
        # Current strategy state
        self.current_strategy = StrategyType.LLM_MODERATE
        self.strategy_start_time = datetime.now()
        self.decisions_since_strategy_change = 0
        
        # Strategy configurations
        self._strategy_configs = self._initialize_strategy_configs()
        
        # Performance tracking
        self.performance_history: Dict[StrategyType, PerformanceMetrics] = {
            strategy: PerformanceMetrics() for strategy in StrategyType
        }
        
        # Decision tracking
        self.recent_decisions = deque(maxlen=50)
        self.recent_rewards = deque(maxlen=50)
        self.recent_sources = deque(maxlen=50)
        
        # Adaptation parameters
        self.adaptation_interval = 20  # Decisions between strategy evaluations
        self.min_strategy_duration = 10  # Minimum decisions before changing
        self.performance_window = 30  # Window for performance calculation
        
        # Rule-based decision making
        self.rule_based_policies = self._initialize_rule_policies()
        
        # Emergency detection
        self.stuck_detection_threshold = 8
        self.emergency_override_active = False
        
    def _initialize_strategy_configs(self) -> Dict[StrategyType, StrategyConfig]:
        """Initialize strategy configurations"""
        return {
            StrategyType.LLM_HEAVY: StrategyConfig(
                strategy_type=StrategyType.LLM_HEAVY,
                llm_frequency=0.9,
                confidence_threshold=0.3,
                rule_based_actions={},
                emergency_actions={"critical_health": 7, "stuck": 5},
                success_threshold=0.8,
                failure_threshold=0.4
            ),
            
            StrategyType.LLM_MODERATE: StrategyConfig(
                strategy_type=StrategyType.LLM_MODERATE,
                llm_frequency=0.5,
                confidence_threshold=0.5,
                rule_based_actions={
                    "battle": [5, 6],  # A or B in battle
                    "dialogue": [5],   # A to advance dialogue
                    "menu": [1, 2, 5, 6]  # Navigate menus
                },
                emergency_actions={"critical_health": 7, "stuck": 5, "fainted": 7},
                success_threshold=0.6,
                failure_threshold=0.3
            ),
            
            StrategyType.LLM_MINIMAL: StrategyConfig(
                strategy_type=StrategyType.LLM_MINIMAL,
                llm_frequency=0.2,
                confidence_threshold=0.7,
                rule_based_actions={
                    "overworld": [1, 2, 3, 4, 5],  # Movement and interaction
                    "battle": [5, 6],
                    "dialogue": [5],
                    "menu": [1, 2, 5, 6],
                    "stuck": [1, 2, 3, 4]  # Movement when stuck
                },
                emergency_actions={"critical_health": 7, "stuck": 5, "fainted": 7},
                success_threshold=0.5,
                failure_threshold=0.2
            ),
            
            StrategyType.RULE_BASED: StrategyConfig(
                strategy_type=StrategyType.RULE_BASED,
                llm_frequency=0.0,
                confidence_threshold=1.0,
                rule_based_actions={
                    "overworld": [1, 2, 3, 4, 5],
                    "battle": [5],  # Always attack
                    "dialogue": [5],
                    "menu": [5, 6],
                    "critical_health": [7],  # Open menu for items
                    "stuck": [1, 2, 3, 4, 5]
                },
                emergency_actions={"critical_health": 7, "stuck": 5, "fainted": 7},
                success_threshold=0.4,
                failure_threshold=0.1
            ),
            
            StrategyType.EMERGENCY: StrategyConfig(
                strategy_type=StrategyType.EMERGENCY,
                llm_frequency=0.1,  # Minimal LLM in emergency
                confidence_threshold=0.9,
                rule_based_actions={
                    "critical_health": [7, 6],  # Menu or flee
                    "fainted": [7],  # Menu for revive
                    "stuck": [1, 2, 3, 4]  # Try movement
                },
                emergency_actions={"critical_health": 7, "stuck": 5, "fainted": 7},
                success_threshold=0.3,
                failure_threshold=0.0
            ),
            
            StrategyType.LEARNING: StrategyConfig(
                strategy_type=StrategyType.LEARNING,
                llm_frequency=0.7,  # High LLM for exploration
                confidence_threshold=0.2,  # Accept low confidence for learning
                rule_based_actions={},
                emergency_actions={"critical_health": 7, "stuck": 5, "fainted": 7},
                learning_rate=0.2,
                success_threshold=0.7,
                failure_threshold=0.4
            ),
            
            StrategyType.BALANCED: StrategyConfig(
                strategy_type=StrategyType.BALANCED,
                llm_frequency=0.5,  # Same as moderate
                confidence_threshold=0.5,
                rule_based_actions={
                    "battle": [5, 6],  # A or B in battle
                    "dialogue": [5],   # A to advance dialogue
                    "menu": [1, 2, 5, 6]  # Navigate menus
                },
                emergency_actions={"critical_health": 7, "stuck": 5, "fainted": 7},
                success_threshold=0.6,
                failure_threshold=0.3
            )
        }
    
    def _initialize_rule_policies(self) -> Dict[str, Callable]:
        """Initialize rule-based decision policies"""
        return {
            "critical_health": self._critical_health_policy,
            "battle": self._battle_policy,
            "dialogue": self._dialogue_policy,
            "menu": self._menu_policy,
            "stuck": self._stuck_policy,
            "overworld": self._overworld_policy,
            "emergency": self._emergency_policy
        }
    
    def get_next_action(self, game_state: GameStateAnalysis, 
                       llm_manager=None, recent_actions: List[int] = None) -> Tuple[int, DecisionSource, str]:
        """
        Get the next action using adaptive strategy
        
        Args:
            game_state: Current game state analysis
            llm_manager: LLM manager for LLM-based decisions
            recent_actions: Recent actions taken
            
        Returns:
            Tuple of (action, decision_source, reasoning)
        """
        
        # Check for emergency situations first
        emergency_action = self._check_emergency_conditions(game_state, recent_actions)
        if emergency_action:
            action, reasoning = emergency_action
            self._record_decision(action, DecisionSource.EMERGENCY_OVERRIDE, reasoning)
            return action, DecisionSource.EMERGENCY_OVERRIDE, reasoning
        
        # Get current strategy configuration
        config = self.strategy_configs[self.current_strategy]
        
        # Determine decision source based on strategy
        should_use_llm = self._should_use_llm(game_state, config)
        
        if should_use_llm and llm_manager:
            # Try LLM decision
            llm_action = llm_manager.get_action(game_state=self._get_game_state_string(game_state))
            if llm_action is not None:
                reasoning = f"LLM decision using {self.current_strategy.value} strategy"
                self._record_decision(llm_action, DecisionSource.LLM, reasoning)
                return llm_action, DecisionSource.LLM, reasoning
        
        # Check for pattern-based decisions
        pattern_action = self._get_pattern_based_action(game_state, recent_actions)
        if pattern_action:
            action, reasoning = pattern_action
            self._record_decision(action, DecisionSource.PATTERN_BASED, reasoning)
            return action, DecisionSource.PATTERN_BASED, reasoning
        
        # Check for goal-directed decisions
        goal_action = self._get_goal_directed_action(game_state)
        if goal_action:
            action, reasoning = goal_action
            self._record_decision(action, DecisionSource.GOAL_DIRECTED, reasoning)
            return action, DecisionSource.GOAL_DIRECTED, reasoning
        
        # Fall back to rule-based decision
        action, reasoning = self._get_rule_based_action(game_state, recent_actions)
        self._record_decision(action, DecisionSource.RULE_BASED, reasoning)
        return action, DecisionSource.RULE_BASED, reasoning
    
    def _should_use_llm(self, game_state: GameStateAnalysis, config: StrategyConfig) -> bool:
        """Determine if LLM should be used for this decision"""
        import random
        
        # Emergency situations may override LLM frequency
        if game_state.criticality == SituationCriticality.EMERGENCY:
            return random.random() < config.llm_frequency * 0.3  # Reduce LLM in emergency
        
        # Novel situations favor LLM
        if game_state.criticality == SituationCriticality.URGENT:
            return random.random() < config.llm_frequency * 1.5  # Increase LLM for urgent
        
        # Standard frequency check
        return random.random() < config.llm_frequency
    
    def _check_emergency_conditions(self, game_state: GameStateAnalysis, 
                                   recent_actions: List[int]) -> Optional[Tuple[int, str]]:
        """Check for emergency conditions requiring immediate action"""
        
        # Critical health emergency
        if game_state.health_percentage < 5:
            return 7, "Emergency: Critical health - opening menu for healing"
        
        # Fainted Pokemon (0% health)
        if game_state.health_percentage == 0:
            return 7, "Emergency: Pokemon fainted - need revival items"
        
        # Stuck detection
        if recent_actions and len(recent_actions) >= self.stuck_detection_threshold:
            recent_unique = set(recent_actions[-self.stuck_detection_threshold:])
            if len(recent_unique) <= 2:  # Very repetitive
                # Try an action not recently used
                all_actions = [1, 2, 3, 4, 5, 6, 7, 8]
                for action in all_actions:
                    if action not in recent_actions[-4:]:
                        return action, f"Emergency: Breaking stuck pattern {list(recent_unique)}"
        
        # Critical situation with immediate threats
        if (game_state.criticality == SituationCriticality.EMERGENCY and 
            game_state.immediate_threats):
            config = self.strategy_configs[StrategyType.EMERGENCY]
            if "critical_health" in config.emergency_actions:
                action = config.emergency_actions["critical_health"]
                return action, f"Emergency protocol for: {game_state.immediate_threats}"
        
        return None
    
    def _get_pattern_based_action(self, game_state: GameStateAnalysis, 
                                 recent_actions: List[int]) -> Optional[Tuple[int, str]]:
        """Get action based on learned patterns"""
        if not self.history_analyzer:
            return None
        
        recommendations = self.history_analyzer.get_action_recommendations(
            game_state, recent_actions or []
        )
        
        if recommendations:
            action, reason, confidence = recommendations[0]  # Top recommendation
            if confidence >= 0.5:  # Reasonable confidence threshold
                return action, f"Pattern-based: {reason} (confidence: {confidence:.2f})"
        
        return None
    
    def _get_goal_directed_action(self, game_state: GameStateAnalysis) -> Optional[Tuple[int, str]]:
        """Get action based on current goals"""
        if not self.goal_planner:
            return None
        
        recommendations = self.goal_planner.get_recommended_actions(game_state)
        
        if recommendations:
            action_str, reason, weight = recommendations[0]  # Top recommendation
            try:
                # Convert string action to int (assuming action_str is like "a", "up", etc.)
                action_map = {"up": 1, "down": 2, "left": 3, "right": 4, 
                             "a": 5, "b": 6, "start": 7, "select": 8}
                action = action_map.get(action_str.lower(), 5)  # Default to A
                return action, f"Goal-directed: {reason}"
            except (ValueError, AttributeError):
                return None
        
        return None
    
    def _get_rule_based_action(self, game_state: GameStateAnalysis, 
                              recent_actions: List[int]) -> Tuple[int, str]:
        """Get action using rule-based policies"""
        
        # Determine context
        context = self._determine_context(game_state, recent_actions)
        
        # Apply appropriate policy
        if context in self.rule_based_policies:
            action = self.rule_based_policies[context](game_state, recent_actions)
            return action, f"Rule-based policy for {context}"
        else:
            # Default policy
            action = self._overworld_policy(game_state, recent_actions)
            return action, "Default rule-based policy"
    
    def _determine_context(self, game_state: GameStateAnalysis, recent_actions: List[int]) -> str:
        """Determine the current context for rule-based decisions"""
        
        # Check for critical health first
        if game_state.health_percentage < 25:
            return "critical_health"
        
        # Check for stuck situation
        if recent_actions and len(recent_actions) >= 5:
            if len(set(recent_actions[-5:])) <= 2:
                return "stuck"
        
        # Check for emergency
        if game_state.criticality == SituationCriticality.EMERGENCY:
            return "emergency"
        
        # Check game state variables for specific contexts
        state_vars = game_state.state_variables
        
        # Battle context
        if state_vars.get('in_battle') and state_vars['in_battle'].current_value:
            return "battle"
        
        # Menu context (simplified detection)
        if state_vars.get('menu_state') and state_vars['menu_state'].current_value > 0:
            return "menu"
        
        # Default to overworld
        return "overworld"
    
    def _critical_health_policy(self, game_state: GameStateAnalysis, recent_actions: List[int]) -> int:
        """Policy for critical health situations"""
        return 7  # Open menu for healing items
    
    def _battle_policy(self, game_state: GameStateAnalysis, recent_actions: List[int]) -> int:
        """Policy for battle situations"""
        if game_state.health_percentage < 30:
            return 6  # Try to flee
        else:
            return 5  # Attack
    
    def _dialogue_policy(self, game_state: GameStateAnalysis, recent_actions: List[int]) -> int:
        """Policy for dialogue situations"""
        return 5  # A button to advance dialogue
    
    def _menu_policy(self, game_state: GameStateAnalysis, recent_actions: List[int]) -> int:
        """Policy for menu navigation"""
        import random
        return random.choice([1, 2, 5, 6])  # Navigate or select/cancel
    
    def _stuck_policy(self, game_state: GameStateAnalysis, recent_actions: List[int]) -> int:
        """Policy for stuck situations"""
        if not recent_actions:
            return 1  # Default to UP
        
        # Try an action not recently used
        all_actions = [1, 2, 3, 4, 5]
        for action in all_actions:
            if action not in recent_actions[-3:]:
                return action
        
        # If all recent actions used, try interaction
        return 5
    
    def _overworld_policy(self, game_state: GameStateAnalysis, recent_actions: List[int]) -> int:
        """Policy for overworld exploration"""
        import random
        
        # Favor interaction and movement
        actions = [1, 2, 3, 4, 5]  # Movement + interaction
        weights = [1, 1, 1, 1, 2]   # Slightly favor interaction
        
        return random.choices(actions, weights=weights)[0]
    
    def _emergency_policy(self, game_state: GameStateAnalysis, recent_actions: List[int]) -> int:
        """Policy for emergency situations"""
        if game_state.health_percentage <= 10:
            return 7  # Menu for healing
        elif game_state.immediate_threats:
            return 6  # Try to escape/cancel
        else:
            return 5  # Try interaction
    
    def _get_game_state_string(self, game_state: GameStateAnalysis) -> str:
        """Convert game state to string for LLM manager"""
        if game_state.state_variables.get('in_battle'):
            return "battle"
        elif game_state.criticality == SituationCriticality.EMERGENCY:
            return "emergency"
        else:
            return "overworld"
    
    def _record_decision(self, action: int, source: DecisionSource, reasoning: str):
        """Record decision for tracking and adaptation"""
        self.recent_decisions.append({
            "action": action,
            "source": source,
            "reasoning": reasoning,
            "timestamp": datetime.now()
        })
        
        self.recent_sources.append(source)
        self.decisions_since_strategy_change += 1
        
        # Update performance metrics
        current_metrics = self.performance_history[self.current_strategy]
        current_metrics.decisions_made += 1
        
        # Check if it's time to evaluate strategy
        if self.decisions_since_strategy_change >= self.adaptation_interval:
            self._evaluate_and_adapt_strategy()
    
    def record_outcome(self, reward: float, led_to_progress: bool, was_effective: bool):
        """Record the outcome of the last decision"""
        self.recent_rewards.append(reward)
        
        # Update performance metrics
        current_metrics = self.performance_history[self.current_strategy]
        
        if reward > 0 and was_effective:
            current_metrics.successes += 1
            current_metrics.last_success_time = datetime.now()
        elif reward < 0 or not was_effective:
            current_metrics.failures += 1
        
        # Update average reward
        if current_metrics.decisions_made > 0:
            current_metrics.average_reward = (
                (current_metrics.average_reward * (current_metrics.decisions_made - 1) + reward) / 
                current_metrics.decisions_made
            )
    
    def _evaluate_and_adapt_strategy(self):
        """Evaluate current strategy performance and adapt if necessary"""
        current_metrics = self.performance_history[self.current_strategy]
        config = self.strategy_configs[self.current_strategy]
        
        if current_metrics.decisions_made < self.min_strategy_duration:
            return  # Too early to evaluate
        
        # Calculate performance metrics
        success_rate = (current_metrics.successes / 
                       max(current_metrics.decisions_made, 1))
        
        recent_rewards = list(self.recent_rewards)[-self.performance_window:]
        avg_recent_reward = statistics.mean(recent_rewards) if recent_rewards else 0
        
        # Check if strategy change is needed
        should_change = False
        reason = ""
        
        if success_rate < config.failure_threshold:
            should_change = True
            reason = f"Low success rate: {success_rate:.2f}"
        elif avg_recent_reward < -0.5:
            should_change = True
            reason = f"Poor recent performance: {avg_recent_reward:.2f}"
        elif self._detect_prolonged_ineffectiveness():
            should_change = True
            reason = "Prolonged ineffectiveness detected"
        
        if should_change:
            self._adapt_strategy(reason)
        
        # Reset counter
        self.decisions_since_strategy_change = 0
    
    def _detect_prolonged_ineffectiveness(self) -> bool:
        """Detect if the current strategy has been ineffective for too long"""
        current_metrics = self.performance_history[self.current_strategy]
        
        # Check if no recent successes
        if current_metrics.last_success_time:
            time_since_success = datetime.now() - current_metrics.last_success_time
            if time_since_success > timedelta(minutes=5):  # No success in 5 minutes
                return True
        
        # Check recent decision sources - if too many rule-based, LLM might be failing
        if len(self.recent_sources) >= 10:
            recent_sources = list(self.recent_sources)[-10:]
            rule_based_count = sum(1 for source in recent_sources 
                                 if source == DecisionSource.RULE_BASED)
            if rule_based_count > 8:  # >80% rule-based decisions
                return True
        
        return False
    
    def _adapt_strategy(self, reason: str):
        """Adapt strategy based on current performance"""
        old_strategy = self.current_strategy
        
        # Strategy adaptation logic
        if self.current_strategy == StrategyType.LLM_HEAVY:
            # If LLM-heavy is failing, reduce LLM dependency
            self.current_strategy = StrategyType.LLM_MODERATE
        elif self.current_strategy == StrategyType.LLM_MODERATE:
            # Try minimal LLM or check if we should go back to heavy
            current_metrics = self.performance_history[self.current_strategy]
            if current_metrics.average_reward < 0:
                self.current_strategy = StrategyType.LLM_MINIMAL
            else:
                self.current_strategy = StrategyType.LLM_HEAVY  # Try more LLM
        elif self.current_strategy == StrategyType.LLM_MINIMAL:
            # Go to rule-based if minimal LLM fails
            self.current_strategy = StrategyType.RULE_BASED
        elif self.current_strategy == StrategyType.RULE_BASED:
            # Try learning mode to discover new patterns
            self.current_strategy = StrategyType.LEARNING
        elif self.current_strategy == StrategyType.LEARNING:
            # Return to moderate after learning
            self.current_strategy = StrategyType.LLM_MODERATE
        else:
            # Default fallback
            self.current_strategy = StrategyType.LLM_MODERATE
        
        self.strategy_start_time = datetime.now()
        self.decisions_since_strategy_change = 0
        
        self.logger.info(f"Strategy adapted: {old_strategy.value} -> {self.current_strategy.value} ({reason})")
    
    def force_strategy(self, strategy: StrategyType):
        """Force a specific strategy (for testing/debugging)"""
        old_strategy = self.current_strategy
        self.current_strategy = strategy
        self.strategy_start_time = datetime.now()
        self.decisions_since_strategy_change = 0
        self.logger.info(f"Strategy forced: {old_strategy.value} -> {strategy.value}")
    
    def select_strategy(self, context: Dict[str, Any]) -> StrategyType:
        """Select strategy based on context (for compatibility with tests)"""
        # Handle None context
        if context is None:
            context = {}
        
        # Use current strategy or adapt based on context
        if context.get('recent_performance', {}).get('success_rate', 0.5) < 0.3:
            return StrategyType.LLM_MINIMAL
        elif context.get('recent_performance', {}).get('success_rate', 0.5) > 0.8:
            return StrategyType.LLM_HEAVY
        return self.current_strategy
    
    def evaluate_performance(self, metrics: Dict[str, float]):
        """Evaluate performance and adapt strategy (for compatibility with tests)"""
        episode_reward = metrics.get('episode_reward', 0)
        llm_usage_rate = metrics.get('llm_usage_rate', 0.5)
        
        # Record outcome based on metrics
        led_to_progress = episode_reward > 10.0
        was_effective = episode_reward > 0
        self.record_outcome(episode_reward, led_to_progress, was_effective)
    
    def get_strategy_stats(self) -> Dict[str, Any]:
        """Get statistics about strategy performance"""
        # Count strategy switches
        total_switches = sum(1 for decision in self.recent_decisions 
                           if len(self.recent_decisions) > 1 and decision != self.recent_decisions[0])
        
        stats = {
            "current_strategy": self.current_strategy.value,
            "time_in_current_strategy": (datetime.now() - self.strategy_start_time).total_seconds(),
            "decisions_in_current_strategy": self.decisions_since_strategy_change,
            "total_switches": total_switches,
            "performance_by_strategy": {}
        }
        
        # Performance by strategy
        for strategy, metrics in self.performance_history.items():
            if metrics.decisions_made > 0:
                success_rate = metrics.successes / metrics.decisions_made
                stats["performance_by_strategy"][strategy.value] = {
                    "decisions_made": metrics.decisions_made,
                    "success_rate": success_rate,
                    "average_reward": metrics.average_reward,
                    "last_success": metrics.last_success_time.isoformat() if metrics.last_success_time else None
                }
        
        # Recent decision sources
        if self.recent_sources:
            source_counts = defaultdict(int)
            for source in list(self.recent_sources)[-20:]:  # Last 20 decisions
                source_counts[source.value] += 1
            stats["recent_decision_sources"] = dict(source_counts)
        
        # Performance history for tests
        performance_history = []
        for strategy, metrics in self.performance_history.items():
            if metrics.decisions_made > 0:
                performance_history.append({
                    "strategy": strategy.value,
                    "decisions": metrics.decisions_made,
                    "successes": metrics.successes,
                    "average_reward": metrics.average_reward
                })
        stats["performance_history"] = performance_history
        
        return stats
    
    def get_current_config(self) -> StrategyConfig:
        """Get current strategy configuration"""
        return self.strategy_configs[self.current_strategy]
    
    @property
    def strategy_configs(self) -> Dict[StrategyType, StrategyConfig]:
        """Get strategy configurations"""
        return self._strategy_configs
    
    @strategy_configs.setter 
    def strategy_configs(self, configs: Dict[StrategyType, StrategyConfig]):
        """Set strategy configurations"""
        self._strategy_configs = configs


if __name__ == "__main__":
    # Example usage
    adaptive_system = AdaptiveStrategySystem()
    print(f"Adaptive Strategy System initialized")
    print(f"Current strategy: {adaptive_system.current_strategy.value}")
    print(f"Available strategies: {[s.value for s in StrategyType]}")
    
    # Show current configuration
    config = adaptive_system.get_current_config()
    print(f"LLM frequency: {config.llm_frequency}")
    print(f"Confidence threshold: {config.confidence_threshold}")