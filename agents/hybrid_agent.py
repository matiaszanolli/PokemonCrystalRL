#!/usr/bin/env python3
"""
Hybrid LLM+RL Agent for Pokemon Crystal RL

Implements the hybrid architecture combining LLM strategic guidance with RL optimization
as specified in ROADMAP_ENHANCED Phase 3.1
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
import logging
from abc import ABC, abstractmethod
import json
import time
from collections import deque

from environments.state.analyzer import GameStateAnalysis, GamePhase, SituationCriticality
from core.adaptive_strategy_system import AdaptiveStrategySystem, StrategyType, DecisionSource
from core.decision_validator import DecisionValidator, ValidationResult
from trainer.llm_manager import LLMManager


class AgentMode(Enum):
    """Operating modes for the hybrid agent"""
    LLM_GUIDED = "llm_guided"           # LLM makes primary decisions
    RL_OPTIMIZED = "rl_optimized"       # RL makes primary decisions
    COLLABORATIVE = "collaborative"     # Both agents collaborate
    ADAPTIVE = "adaptive"               # Dynamically switch based on situation
    LEARNING = "learning"               # Active learning mode

class DecisionConfidence(Enum):
    """Confidence levels for decisions"""
    VERY_LOW = "very_low"     # 0.0 - 0.2
    LOW = "low"               # 0.2 - 0.4  
    MEDIUM = "medium"         # 0.4 - 0.6
    HIGH = "high"             # 0.6 - 0.8
    VERY_HIGH = "very_high"   # 0.8 - 1.0

@dataclass
class AgentDecision:
    """Represents a decision made by an agent"""
    agent_type: str
    action: int
    confidence: float
    reasoning: str
    expected_reward: float = 0.0
    risk_assessment: str = "unknown"
    alternative_actions: List[Tuple[int, float]] = None  # (action, confidence) pairs
    
    def __post_init__(self):
        if self.alternative_actions is None:
            self.alternative_actions = []

class BaseAgent(ABC):
    """Abstract base class for agents in the hybrid system"""
    
    @abstractmethod
    def get_action(self, observation: Dict[str, Any], info: Dict[str, Any]) -> AgentDecision:
        """Get action decision from the agent"""
        pass
    
    @abstractmethod
    def update(self, observation: Dict[str, Any], action: int, reward: float, 
              next_observation: Dict[str, Any], done: bool):
        """Update agent with experience"""
        pass
    
    @abstractmethod
    def get_confidence(self) -> float:
        """Get current confidence level of the agent"""
        pass

class LLMAgent(BaseAgent):
    """LLM-based agent for strategic decision making"""
    
    def __init__(self, llm_manager: LLMManager, adaptive_strategy: AdaptiveStrategySystem):
        self.llm_manager = llm_manager
        self.adaptive_strategy = adaptive_strategy
        self.decision_validator = DecisionValidator()
        self.logger = logging.getLogger("pokemon_trainer.llm_agent")
        
        # Performance tracking
        self.recent_rewards = deque(maxlen=20)
        self.recent_confidences = deque(maxlen=20)
        self.decision_quality_score = 0.5  # Start neutral
        
    def get_action(self, observation: Dict[str, Any], info: Dict[str, Any]) -> AgentDecision:
        """Get LLM decision with validation"""
        
        # Convert observation to game state analysis
        game_analysis = self._obs_to_game_analysis(observation, info)
        
        # Get action from adaptive strategy system
        recent_actions = observation.get('action_history', [])
        recent_actions_list = [int(a) for a in recent_actions if a > 0]  # Filter out padding
        
        action, source, reasoning = self.adaptive_strategy.get_next_action(
            game_analysis, self.llm_manager, recent_actions_list
        )
        
        # Validate the decision
        validation = self.decision_validator.validate_action(action, game_analysis, recent_actions_list)
        
        # Adjust action if validation failed
        final_action = validation.approved_action if validation.approved_action is not None else action
        
        # Calculate confidence based on validation and recent performance
        confidence = self._calculate_confidence(validation, source)
        
        # Build reasoning
        full_reasoning = f"{reasoning}"
        if validation.result != ValidationResult.APPROVED:
            full_reasoning += f" (Validated: {validation.reasoning})"
        
        # Assess risk
        risk_assessment = validation.risk_level.value if validation else "unknown"
        
        # Get alternative actions from validation
        alternatives = [(alt[0], alt[2]) for alt in validation.suggested_alternatives]
        
        return AgentDecision(
            agent_type="llm",
            action=final_action,
            confidence=confidence,
            reasoning=full_reasoning,
            expected_reward=self._estimate_reward(final_action, game_analysis),
            risk_assessment=risk_assessment,
            alternative_actions=alternatives
        )
    
    def update(self, observation: Dict[str, Any], action: int, reward: float,
              next_observation: Dict[str, Any], done: bool):
        """Update LLM agent with experience"""
        self.recent_rewards.append(reward)
        
        # Update decision quality score
        if reward > 0:
            self.decision_quality_score = min(1.0, self.decision_quality_score + 0.05)
        elif reward < 0:
            self.decision_quality_score = max(0.0, self.decision_quality_score - 0.02)
        
        # Update adaptive strategy with outcome
        led_to_progress = reward > 1.0
        was_effective = reward >= 0
        self.adaptive_strategy.record_outcome(reward, led_to_progress, was_effective)
    
    def get_confidence(self) -> float:
        """Get current confidence of LLM agent"""
        return self.decision_quality_score
    
    def _obs_to_game_analysis(self, observation: Dict[str, Any], info: Dict[str, Any]) -> GameStateAnalysis:
        """Convert observation to GameStateAnalysis"""
        # This is a simplified conversion - in practice, you'd reconstruct the full analysis
        from environments.state.analyzer import GameStateAnalysis, GamePhase, SituationCriticality
        
        # Get phase and criticality from observation
        phase_idx = int(observation.get('game_phase', 0))
        criticality_idx = int(observation.get('criticality', 0))
        
        phase = list(GamePhase)[phase_idx] if phase_idx < len(GamePhase) else GamePhase.EARLY_GAME
        criticality = list(SituationCriticality)[criticality_idx] if criticality_idx < len(SituationCriticality) else SituationCriticality.MODERATE
        
        # Extract other info
        health_percentage = info.get('health_percentage', 50.0)
        progression_score = info.get('progression_score', 10.0)
        
        return GameStateAnalysis(
            phase=phase,
            criticality=criticality,
            health_percentage=health_percentage,
            progression_score=progression_score,
            exploration_score=info.get('exploration_score', 15.0),
            immediate_threats=info.get('immediate_threats', []),
            opportunities=info.get('opportunities', []),
            recommended_priorities=info.get('recommended_priorities', []),
            situation_summary=info.get('situation_summary', "Current game state"),
            strategic_context=info.get('strategic_context', "Strategic context"),
            risk_assessment=info.get('risk_assessment', "Standard risk"),
            state_variables={}
        )
    
    def _calculate_confidence(self, validation, decision_source) -> float:
        """Calculate confidence in the decision"""
        base_confidence = 0.5
        
        # Boost confidence based on validation result
        if validation.result == ValidationResult.APPROVED:
            base_confidence += 0.3
        elif validation.result == ValidationResult.APPROVED_WITH_WARNING:
            base_confidence += 0.1
        elif validation.result in [ValidationResult.REJECTED_HARMFUL, ValidationResult.REJECTED_INEFFECTIVE]:
            base_confidence -= 0.2
        
        # Adjust based on decision source
        if decision_source == DecisionSource.LLM:
            base_confidence += 0.1
        elif decision_source == DecisionSource.PATTERN_BASED:
            base_confidence += 0.2
        elif decision_source == DecisionSource.GOAL_DIRECTED:
            base_confidence += 0.15
        
        # Adjust based on recent performance
        if self.recent_rewards:
            recent_avg = sum(self.recent_rewards) / len(self.recent_rewards)
            if recent_avg > 0.5:
                base_confidence += 0.1
            elif recent_avg < -0.5:
                base_confidence -= 0.1
        
        return np.clip(base_confidence, 0.0, 1.0)
    
    def _estimate_reward(self, action: int, analysis: GameStateAnalysis) -> float:
        """Estimate expected reward for an action"""
        # Simple heuristic-based reward estimation
        base_reward = 0.0
        
        # Emergency actions in emergency situations
        if analysis.criticality == SituationCriticality.EMERGENCY:
            if action in [6, 7]:  # B or START
                base_reward = 2.0
        
        # Interaction actions generally good for progress
        if action == 5:  # A button
            base_reward = 1.0
        
        # Movement actions for exploration
        if action in [1, 2, 3, 4]:  # Movement
            base_reward = 0.5
        
        return base_reward

class RLAgent(BaseAgent):
    """RL-based agent for action optimization"""
    
    def __init__(self, action_space_size: int = 9):
        self.action_space_size = action_space_size
        self.logger = logging.getLogger("pokemon_trainer.rl_agent")
        
        # Simple Q-table for now (could be replaced with neural network)
        self.q_table = {}
        self.learning_rate = 0.1
        self.epsilon = 0.1  # Exploration rate
        self.gamma = 0.99   # Discount factor
        
        # Performance tracking
        self.recent_rewards = deque(maxlen=50)
        self.confidence_score = 0.3  # Start lower than LLM
        
    def get_action(self, observation: Dict[str, Any], info: Dict[str, Any]) -> AgentDecision:
        """Get RL decision using Q-learning"""
        
        # Convert observation to state key
        state_key = self._obs_to_state_key(observation)
        
        # Get Q-values for this state
        q_values = self.q_table.get(state_key, np.zeros(self.action_space_size))
        
        # Epsilon-greedy action selection
        if np.random.random() < self.epsilon:
            action = np.random.randint(0, self.action_space_size)
            confidence = 0.2  # Low confidence for random actions
        else:
            action = np.argmax(q_values)
            confidence = self._calculate_confidence(q_values)
        
        # Apply action masking if available
        if 'action_mask' in observation:
            valid_actions = observation['action_mask']
            if not valid_actions[action]:  # Invalid action
                valid_indices = np.where(valid_actions)[0]
                if len(valid_indices) > 0:
                    # Choose best valid action
                    masked_q_values = q_values * valid_actions
                    action = np.argmax(masked_q_values)
                    confidence = self._calculate_confidence(masked_q_values)
        
        # Get alternative actions
        alternatives = []
        sorted_indices = np.argsort(q_values)[::-1]  # Descending order
        for i, alt_action in enumerate(sorted_indices[1:4]):  # Top 3 alternatives
            alt_confidence = q_values[alt_action] / (np.max(q_values) + 1e-8)
            alternatives.append((int(alt_action), float(alt_confidence)))
        
        return AgentDecision(
            agent_type="rl",
            action=action,
            confidence=confidence,
            reasoning=f"Q-learning decision (Q-value: {q_values[action]:.3f})",
            expected_reward=float(q_values[action]),
            risk_assessment="low" if confidence > 0.6 else "medium",
            alternative_actions=alternatives
        )
    
    def update(self, observation: Dict[str, Any], action: int, reward: float,
              next_observation: Dict[str, Any], done: bool):
        """Update Q-table with experience"""
        
        state_key = self._obs_to_state_key(observation)
        next_state_key = self._obs_to_state_key(next_observation)
        
        # Initialize Q-values if not seen before
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_space_size)
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(self.action_space_size)
        
        # Q-learning update
        current_q = self.q_table[state_key][action]
        if done:
            target_q = reward
        else:
            max_next_q = np.max(self.q_table[next_state_key])
            target_q = reward + self.gamma * max_next_q
        
        self.q_table[state_key][action] += self.learning_rate * (target_q - current_q)
        
        # Update performance tracking
        self.recent_rewards.append(reward)
        
        # Update confidence based on recent performance
        if len(self.recent_rewards) >= 10:
            avg_reward = sum(list(self.recent_rewards)[-10:]) / 10
            if avg_reward > 0.5:
                self.confidence_score = min(1.0, self.confidence_score + 0.02)
            elif avg_reward < -0.5:
                self.confidence_score = max(0.1, self.confidence_score - 0.01)
        
        # Decay exploration
        self.epsilon = max(0.01, self.epsilon * 0.9995)
    
    def get_confidence(self) -> float:
        """Get current confidence of RL agent"""
        return self.confidence_score
    
    def _obs_to_state_key(self, observation: Dict[str, Any]) -> str:
        """Convert observation to state key for Q-table"""
        # Use key features for state representation
        key_features = []
        
        # Game phase and criticality
        key_features.append(str(observation.get('game_phase', 0)))
        key_features.append(str(observation.get('criticality', 0)))
        
        # Health and progress (quantized)
        state_vars = observation.get('state_variables', np.array([]))
        if len(state_vars) > 0:
            # Use first few state variables (health, progress indicators)
            key_features.extend([str(int(x * 10)) for x in state_vars[:5]])
        
        # Recent actions pattern
        action_history = observation.get('action_history', [])
        if len(action_history) > 0:
            last_actions = action_history[-3:]  # Last 3 actions
            key_features.append('_'.join([str(int(a)) for a in last_actions]))
        
        return '|'.join(key_features)
    
    def _calculate_confidence(self, q_values: np.ndarray) -> float:
        """Calculate confidence based on Q-value distribution"""
        if len(q_values) == 0:
            return 0.3
        
        # Higher confidence when there's a clear best action
        max_q = np.max(q_values)
        second_max_q = np.partition(q_values, -2)[-2]
        
        if max_q == second_max_q:
            return 0.3  # No clear preference
        
        # Confidence based on margin between best and second-best
        margin = max_q - second_max_q
        confidence = min(0.9, 0.3 + margin * 0.5)
        
        return confidence

class HybridAgent:
    """
    Hybrid LLM+RL Agent that combines strategic LLM guidance with RL optimization
    
    Implements the architecture from ROADMAP_ENHANCED Phase 3.1:
    - LLM provides strategic oversight and handles novel situations
    - RL optimizes well-understood patterns and actions
    - Dynamic switching based on confidence and situation
    """
    
    def __init__(self, llm_manager: LLMManager, adaptive_strategy: AdaptiveStrategySystem,
                 action_space_size: int = 9, curriculum_config: Optional[Dict] = None):
        self.logger = logging.getLogger("pokemon_trainer.hybrid_agent")
        
        # Component agents
        self.llm_agent = LLMAgent(llm_manager, adaptive_strategy)
        self.rl_agent = RLAgent(action_space_size)
        
        # Hybrid configuration
        self.mode = AgentMode.ADAPTIVE
        self.curriculum_config = curriculum_config or self._default_curriculum_config()
        
        # Decision arbitration
        self.confidence_threshold = 0.6
        self.experience_counter = 0
        self.llm_preference_bonus = 0.1  # Initial bias toward LLM
        
        # Performance tracking
        self.agent_usage_stats = {"llm": 0, "rl": 0, "hybrid": 0}
        self.performance_by_agent = {"llm": deque(maxlen=100), "rl": deque(maxlen=100)}
        self.recent_rewards = deque(maxlen=100)  # Initialize recent_rewards
        
        # Curriculum learning parameters
        self.llm_confidence_threshold = 0.7
        
        # Curriculum learning state
        self.curriculum_stage = 0
        self.stage_progress = 0
        
    def _default_curriculum_config(self) -> Dict:
        """Default curriculum learning configuration"""
        return {
            "stages": [
                {"name": "llm_heavy", "llm_weight": 0.9, "duration": 1000},
                {"name": "balanced", "llm_weight": 0.5, "duration": 2000}, 
                {"name": "rl_heavy", "llm_weight": 0.2, "duration": float('inf')}
            ]
        }
    
    def get_action(self, observation: Dict[str, Any], info: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
        """Get action from hybrid agent"""
        
        # Get decisions from both agents
        llm_decision = self.llm_agent.get_action(observation, info)
        rl_decision = self.rl_agent.get_action(observation, info)
        
        # Decide which agent to use
        chosen_agent, final_decision = self._arbitrate_decision(llm_decision, rl_decision, observation, info)
        
        # Update usage stats
        self.agent_usage_stats[chosen_agent] += 1
        
        # Create decision info for trainer
        decision_info = {
            'source': chosen_agent,
            'confidence': final_decision.confidence,
            'reasoning': final_decision.reasoning,
            'llm_confidence': llm_decision.confidence,
            'rl_confidence': rl_decision.confidence
        }
        
        # Log decision for analysis
        self.logger.debug(f"Hybrid decision: {chosen_agent} chose action {final_decision.action} "
                         f"(LLM: {llm_decision.confidence:.2f}, RL: {rl_decision.confidence:.2f})")
        
        return final_decision.action, decision_info
    
    def get_state_dict(self) -> Dict[str, Any]:
        """Get state dictionary for saving/loading"""
        return {
            'rl_model_state': self.rl_agent.q_table,
            'agent_usage_stats': self.agent_usage_stats,
            'experience_buffer': getattr(self, 'experience_buffer', []),
            'performance_history': getattr(self, 'performance_history', [])
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load state from dictionary"""
        if 'rl_model_state' in state_dict:
            self.rl_agent.q_table = state_dict['rl_model_state']
        if 'agent_usage_stats' in state_dict:
            self.agent_usage_stats = state_dict['agent_usage_stats']
    
    
    def _arbitrate_decision(self, llm_decision: AgentDecision, rl_decision: AgentDecision,
                           observation: Dict[str, Any], info: Dict[str, Any]) -> Tuple[str, AgentDecision]:
        """Arbitrate between LLM and RL decisions"""
        
        # Curriculum learning - adjust weights based on stage
        stages = self.curriculum_config["stages"]
        current_stage = stages[min(self.curriculum_stage, len(stages) - 1)]
        llm_weight = current_stage["llm_weight"]
        
        # Situation-based arbitration
        criticality = observation.get('criticality', 0)
        game_phase = observation.get('game_phase', 0)
        
        # Emergency situations - prefer LLM for strategic thinking
        if criticality >= 3:  # Emergency/urgent
            llm_weight += 0.3
        
        # Early game phases - prefer LLM for exploration
        if game_phase <= 2:  # Early phases
            llm_weight += 0.2
        
        # Confidence-based arbitration
        llm_confidence_adjusted = llm_decision.confidence + self.llm_preference_bonus
        rl_confidence_adjusted = rl_decision.confidence
        
        # Weight by curriculum and situation
        llm_score = llm_confidence_adjusted * llm_weight
        rl_score = rl_confidence_adjusted * (1 - llm_weight)
        
        # Decision logic
        if self.mode == AgentMode.LLM_GUIDED:
            return "llm", llm_decision
        elif self.mode == AgentMode.RL_OPTIMIZED:
            return "rl", rl_decision
        elif self.mode == AgentMode.ADAPTIVE:
            if llm_score > rl_score:
                return "llm", llm_decision
            else:
                return "rl", rl_decision
        elif self.mode == AgentMode.COLLABORATIVE:
            # Blend decisions if both are confident
            if llm_decision.confidence > 0.7 and rl_decision.confidence > 0.7:
                # For now, choose the higher confidence one
                # TODO: Could implement action blending here
                if llm_decision.confidence > rl_decision.confidence:
                    return "hybrid", llm_decision
                else:
                    return "hybrid", rl_decision
            elif llm_decision.confidence > 0.5:
                return "llm", llm_decision
            else:
                return "rl", rl_decision
        else:  # LEARNING mode
            # Favor exploration and LLM for learning
            return "llm", llm_decision
    
    def update(self, observation: Dict[str, Any], action: int, reward: float,
              next_observation: Dict[str, Any], done: bool):
        """Update both agents with experience"""
        
        # Update both agents
        self.llm_agent.update(observation, action, reward, next_observation, done)
        self.rl_agent.update(observation, action, reward, next_observation, done)
        
        # Track performance by agent type
        # Note: This is simplified - in practice you'd track which agent made the decision
        self.performance_by_agent["llm"].append(reward)
        self.performance_by_agent["rl"].append(reward)
        
        # Update curriculum learning
        self._update_curriculum(reward)
        
        # Update LLM preference based on relative performance
        self._update_agent_preferences()
        
        self.experience_counter += 1
    
    def _update_curriculum(self, reward: float):
        """Update curriculum learning stage"""
        self.stage_progress += 1
        
        current_stage = self.curriculum_config["stages"][min(self.curriculum_stage, len(self.curriculum_config["stages"]) - 1)]
        
        if self.stage_progress >= current_stage["duration"]:
            if self.curriculum_stage < len(self.curriculum_config["stages"]) - 1:
                self.curriculum_stage += 1
                self.stage_progress = 0
                self.logger.info(f"Advanced to curriculum stage {self.curriculum_stage}: "
                               f"{self.curriculum_config['stages'][self.curriculum_stage]['name']}")
    
    def _update_agent_preferences(self):
        """Update agent preferences based on recent performance"""
        if len(self.performance_by_agent["llm"]) >= 20 and len(self.performance_by_agent["rl"]) >= 20:
            llm_avg = sum(list(self.performance_by_agent["llm"])[-20:]) / 20
            rl_avg = sum(list(self.performance_by_agent["rl"])[-20:]) / 20
            
            # Adjust LLM preference based on relative performance
            performance_diff = llm_avg - rl_avg
            if performance_diff > 0.1:
                self.llm_preference_bonus = min(0.3, self.llm_preference_bonus + 0.01)
            elif performance_diff < -0.1:
                self.llm_preference_bonus = max(-0.1, self.llm_preference_bonus - 0.01)
    
    def set_mode(self, mode: AgentMode):
        """Set the operating mode of the hybrid agent"""
        old_mode = self.mode
        self.mode = mode
        self.logger.info(f"Hybrid agent mode changed: {old_mode.value} -> {mode.value}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get hybrid agent statistics"""
        total_decisions = sum(self.agent_usage_stats.values())
        
        stats = {
            "mode": self.mode.value,
            "experience_counter": self.experience_counter,
            "curriculum_stage": self.curriculum_stage,
            "stage_progress": self.stage_progress,
            "llm_preference_bonus": self.llm_preference_bonus,
            "agent_usage": {
                agent: count / total_decisions if total_decisions > 0 else 0
                for agent, count in self.agent_usage_stats.items()
            },
            "agent_confidence": {
                "llm": self.llm_agent.get_confidence(),
                "rl": self.rl_agent.get_confidence()
            }
        }
        
        # Recent performance
        if self.performance_by_agent["llm"]:
            stats["recent_performance"] = {
                "llm": sum(list(self.performance_by_agent["llm"])[-10:]) / min(10, len(self.performance_by_agent["llm"])),
                "rl": sum(list(self.performance_by_agent["rl"])[-10:]) / min(10, len(self.performance_by_agent["rl"]))
            }
        
        return stats
    
    def save_state(self, filepath: str):
        """Save hybrid agent state"""
        state = {
            "curriculum_stage": self.curriculum_stage,
            "stage_progress": self.stage_progress,
            "llm_preference_bonus": self.llm_preference_bonus,
            "agent_usage_stats": self.agent_usage_stats,
            "rl_q_table": {k: v.tolist() for k, v in self.rl_agent.q_table.items()}
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
    
    def load_state(self, filepath: str):
        """Load hybrid agent state"""
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            self.curriculum_stage = state.get("curriculum_stage", 0)
            self.stage_progress = state.get("stage_progress", 0) 
            self.llm_preference_bonus = state.get("llm_preference_bonus", 0.1)
            self.agent_usage_stats = state.get("agent_usage_stats", {"llm": 0, "rl": 0, "hybrid": 0})
            
            # Load RL Q-table
            if "rl_q_table" in state:
                self.rl_agent.q_table = {
                    k: np.array(v) for k, v in state["rl_q_table"].items()
                }
            
            self.logger.info(f"Loaded hybrid agent state from {filepath}")
        except Exception as e:
            self.logger.warning(f"Could not load hybrid agent state: {e}")


if __name__ == "__main__":
    # Example usage
    from trainer.llm_manager import LLMManager
    from core.adaptive_strategy_system import AdaptiveStrategySystem
    from unittest.mock import Mock
    
    # Mock components for testing
    llm_manager = Mock(spec=LLMManager)
    adaptive_strategy = Mock(spec=AdaptiveStrategySystem)
    
    # Create hybrid agent
    hybrid_agent = HybridAgent(llm_manager, adaptive_strategy)
    
    print("Hybrid LLM+RL Agent initialized")
    print(f"Current mode: {hybrid_agent.mode.value}")
    print(f"Available modes: {[mode.value for mode in AgentMode]}")
    
    # Example observation
    mock_obs = {
        'state_variables': np.random.random(20),
        'game_phase': 1,
        'criticality': 2,
        'action_history': [1, 2, 5, 0, 0],
        'action_mask': np.ones(9)
    }
    
    mock_info = {
        'health_percentage': 80.0,
        'progression_score': 25.0
    }
    
    # Test decision making
    action = hybrid_agent.get_action(mock_obs, mock_info)
    print(f"Hybrid agent chose action: {action}")
    
    # Show stats
    stats = hybrid_agent.get_stats()
    print(f"Agent stats: {stats}")