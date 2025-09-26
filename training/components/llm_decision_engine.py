"""
LLM Decision Engine - AI-powered decision making for Pokemon Crystal RL

Extracted from LLMTrainer to handle LLM interactions, decision making,
and action generation with fallback mechanisms.
"""

import time
import logging
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from collections import deque

try:
    from agents.llm_agent import LLMAgent
except ImportError:
    print("⚠️  LLMAgent not available")
    LLMAgent = None

try:
    from .strategic_context_builder import StrategicContextBuilder
except ImportError:
    print("⚠️  StrategicContextBuilder not available")
    StrategicContextBuilder = None

try:
    from .context_aware_action_filter import ContextAwareActionFilter
except ImportError:
    print("⚠️  ContextAwareActionFilter not available")
    ContextAwareActionFilter = None

try:
    from .enhanced_stuck_detection import EnhancedStuckDetector
except ImportError:
    print("⚠️  EnhancedStuckDetector not available")
    EnhancedStuckDetector = None

try:
    from .experience_memory_system import ExperienceMemorySystem
except ImportError:
    print("⚠️  ExperienceMemorySystem not available")
    ExperienceMemorySystem = None


@dataclass
class LLMConfig:
    """Configuration for LLM decision making."""
    model: str = "smollm2:1.7b"
    base_url: str = "http://localhost:11434"
    temperature: float = 0.7
    interval: int = 20  # Use LLM every N actions
    timeout: float = 5.0
    max_retries: int = 3
    fallback_enabled: bool = True


class LLMDecisionEngine:
    """Handles LLM-based decision making with fallback strategies."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.logger = logging.getLogger("LLMDecisionEngine")
        
        # LLM agent setup
        self.llm_agent = None
        if LLMAgent is not None:
            try:
                self.llm_agent = LLMAgent(config.model, config.base_url)
                self.logger.info(f"LLM agent initialized: {config.model}")
            except Exception as e:
                self.logger.error(f"Failed to initialize LLM agent: {e}")
        
        # Decision tracking
        self.decision_count = 0
        self.success_count = 0
        self.failure_count = 0
        self.response_times = deque(maxlen=50)
        
        # Fallback action sequences
        self.fallback_actions = [5, 1, 2, 3, 4]  # A, UP, DOWN, LEFT, RIGHT
        self.stuck_actions = [2, 3, 4, 1]  # DOWN, LEFT, RIGHT, UP
        
        # State tracking for context
        self.last_decision_time = 0
        self.consecutive_failures = 0
        self.is_available = self.llm_agent is not None

        # Strategic context builder for enhanced LLM context
        self.context_builder = None
        if StrategicContextBuilder is not None:
            try:
                self.context_builder = StrategicContextBuilder()
                self.logger.info("Strategic context builder initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize strategic context builder: {e}")

        # Action filter for context-aware decisions
        self.action_filter = None
        if ContextAwareActionFilter is not None:
            try:
                self.action_filter = ContextAwareActionFilter()
                self.logger.info("Context-aware action filter initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize action filter: {e}")

        # Enhanced stuck detector
        self.stuck_detector = None
        if EnhancedStuckDetector is not None:
            try:
                self.stuck_detector = EnhancedStuckDetector()
                self.logger.info("Enhanced stuck detector initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize stuck detector: {e}")

        # Experience memory system
        self.experience_memory = None
        if ExperienceMemorySystem is not None:
            try:
                self.experience_memory = ExperienceMemorySystem()
                self.logger.info("Experience memory system initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize experience memory: {e}")
    
    def should_use_llm(self, action_count: int) -> bool:
        """Determine if LLM should be used for this decision.
        
        Args:
            action_count: Current action count
            
        Returns:
            bool: True if LLM should be used
        """
        if not self.is_available:
            return False
        
        # Use LLM at specified intervals
        return action_count % self.config.interval == 0
    
    def get_decision(self, 
                    game_state: Dict[str, Any],
                    screen_data: Optional[np.ndarray] = None,
                    context: Optional[Dict[str, Any]] = None) -> Tuple[int, Dict[str, Any]]:
        """Get decision from LLM or fallback.
        
        Args:
            game_state: Current game state information
            screen_data: Screen image data (optional)
            context: Additional context information
            
        Returns:
            Tuple[int, Dict]: Action (1-8) and decision metadata
        """
        decision_meta = {
            'source': 'fallback',
            'confidence': 0.5,
            'response_time': 0.0,
            'success': True
        }
        
        if not self.is_available:
            action = self._get_fallback_action(game_state, context)
            decision_meta['source'] = 'fallback_no_llm'
            return action, decision_meta
        
        # Update stuck detection with current state
        stuck_pattern = None
        recovery_actions = None
        action_count = context.get('action_count', 0) if context else 0

        if self.stuck_detector:
            try:
                # Update stuck detector history
                reward = context.get('last_reward', 0.0) if context else 0.0
                last_action = context.get('last_action', 1) if context else 1
                self.stuck_detector.update_history(game_state, last_action, reward, screen_data)

                # Check for stuck patterns
                stuck_pattern = self.stuck_detector.detect_stuck_patterns(action_count)

                # Get recovery actions if needed
                if stuck_pattern and self.stuck_detector.should_trigger_recovery(stuck_pattern, action_count):
                    recovery_strategy = self.stuck_detector.get_recovery_strategy(stuck_pattern)
                    if recovery_strategy:
                        recovery_actions = self.stuck_detector.execute_recovery(recovery_strategy, action_count)
                        self.logger.info(f"Stuck detected ({stuck_pattern.stuck_type.value}), executing recovery: {recovery_strategy.name}")

            except Exception as e:
                self.logger.warning(f"Stuck detection failed: {e}")

        # If we have recovery actions, return them immediately
        if recovery_actions:
            action = recovery_actions[0]  # Take first action from recovery sequence
            decision_meta.update({
                'source': 'stuck_recovery',
                'confidence': 0.9,
                'response_time': 0.0,
                'success': True,
                'stuck_pattern': stuck_pattern.stuck_type.value if stuck_pattern else None,
                'recovery_action': True
            })
            return action, decision_meta

        # Get action filtering recommendations
        action_filter_result = None
        if self.action_filter:
            try:
                # Prepare context for action filtering
                llm_context = self._prepare_llm_context(game_state, context)
                action_filter_result = self.action_filter.filter_actions(
                    game_state, llm_context
                )
            except Exception as e:
                self.logger.warning(f"Action filtering failed: {e}")

        # Get experience-based recommendations
        experience_recommendation = None
        if self.experience_memory:
            try:
                # Prepare context for experience memory
                llm_context = self._prepare_llm_context(game_state, context)
                experience_recommendation = self.experience_memory.get_recommendation(
                    game_state, llm_context
                )
            except Exception as e:
                self.logger.warning(f"Experience memory recommendation failed: {e}")

        # Try LLM decision
        start_time = time.time()
        try:
            action = self._get_llm_decision(game_state, screen_data, context, action_filter_result, experience_recommendation)
            response_time = time.time() - start_time

            # Validate action with filter if available
            if action_filter_result and not self._is_action_appropriate(action, action_filter_result):
                self.logger.info(f"LLM chose inappropriate action {action}, using filtered recommendation")
                action = action_filter_result.get('recommended_action', action)

            # Consider experience recommendation as a fallback
            if experience_recommendation and experience_recommendation.get('confidence', 0) > 0.7:
                experience_actions = experience_recommendation.get('recommended_actions', [])
                if experience_actions and action not in experience_actions[:2]:
                    self.logger.info(f"Experience memory suggests {experience_actions[0]} over LLM choice {action}")
                    # Don't override but note the suggestion

            if self._is_valid_action(action):
                self.decision_count += 1
                self.success_count += 1
                self.response_times.append(response_time)
                self.consecutive_failures = 0

                decision_meta.update({
                    'source': 'llm',
                    'confidence': 0.8,
                    'response_time': response_time,
                    'success': True,
                    'action_filter': action_filter_result.get('reasoning') if action_filter_result else None,
                    'experience_recommendation': experience_recommendation.get('reasoning') if experience_recommendation else None
                })
                return action, decision_meta
            else:
                self.logger.warning(f"Invalid LLM action: {action}")
                raise ValueError(f"Invalid action: {action}")

        except Exception as e:
            self.logger.debug(f"LLM decision failed: {e}")
            self.failure_count += 1
            self.consecutive_failures += 1

            # Disable LLM temporarily if too many failures
            if self.consecutive_failures >= 5:
                self.logger.warning("Too many LLM failures, using fallback for next decisions")
                self.is_available = False
        
        # Fallback decision
        action = self._get_fallback_action(game_state, context)
        decision_meta.update({
            'source': 'fallback',
            'confidence': 0.6,
            'response_time': time.time() - start_time
        })
        
        return action, decision_meta
    
    def _get_llm_decision(self,
                         game_state: Dict[str, Any],
                         screen_data: Optional[np.ndarray],
                         context: Optional[Dict[str, Any]],
                         action_filter_result: Optional[Dict[str, Any]] = None,
                         experience_recommendation: Optional[Dict[str, Any]] = None) -> int:
        """Get decision from LLM agent.
        
        Args:
            game_state: Game state information
            screen_data: Screen image data
            context: Additional context
            
        Returns:
            int: Action code (1-8)
        """
        if not self.llm_agent:
            raise RuntimeError("LLM agent not available")
        
        # Prepare context for LLM
        llm_context = self._prepare_llm_context(game_state, context)
        
        # Get action from LLM
        # Convert parameters to match LLMAgent.get_decision() signature
        screen_analysis = {'type': llm_context.get('current_state', 'unknown')}
        recent_actions = []  # TODO: Pass actual recent actions from training loop

        action_str, reasoning = self.llm_agent.get_decision(
            game_state,
            screen_analysis,
            recent_actions
        )

        # Convert action string to PyBoy action code (1-8)
        action_map = {
            'up': 1,
            'down': 2,
            'left': 3,
            'right': 4,
            'a': 5,
            'b': 6,
            'start': 7,
            'select': 8
        }

        try:
            # Try direct mapping first
            action = action_map.get(action_str.lower(), None)
            if action is None:
                # If it's a numeric string, convert it
                if action_str.isdigit():
                    action = int(action_str)
                    # Validate range
                    if not (1 <= action <= 8):
                        action = 1
                else:
                    action = 1  # Default to UP if unknown
        except (ValueError, AttributeError):
            action = 1  # Default action

        return action
    
    def _prepare_llm_context(self,
                           game_state: Dict[str, Any],
                           context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Prepare context information for LLM.

        Args:
            game_state: Game state data
            context: Additional context

        Returns:
            Dict: Prepared context for LLM
        """
        # Use strategic context builder if available
        if self.context_builder:
            try:
                action_count = context.get('action_count', 0) if context else 0
                recent_rewards = context.get('recent_rewards', []) if context else []

                # Build enhanced strategic context
                enhanced_context = self.context_builder.build_enhanced_context(
                    game_state=game_state,
                    action_count=action_count,
                    recent_rewards=recent_rewards
                )

                # Add any additional context from the training loop
                if context:
                    enhanced_context.update({
                        'detected_state': context.get('detected_state', 'overworld'),
                        'stuck_counter': context.get('stuck_counter', 0)
                    })

                return enhanced_context

            except Exception as e:
                self.logger.warning(f"Strategic context builder failed, using fallback: {e}")

        # Fallback to basic context if strategic builder not available or failed
        llm_context = {
            'current_state': 'overworld',
            'step': 0,
            'stuck_counter': 0
        }

        # Extract relevant information
        if game_state:
            llm_context.update({
                'player_map': game_state.get('player_map', 0),
                'player_x': game_state.get('player_x', 0),
                'player_y': game_state.get('player_y', 0),
                'badges': game_state.get('badges', 0),
                'party_count': game_state.get('party_count', 0),
                'in_battle': game_state.get('in_battle', 0)
            })

        if context:
            llm_context.update({
                'step': context.get('action_count', 0),
                'stuck_counter': context.get('stuck_counter', 0),
                'current_state': context.get('detected_state', 'overworld')
            })

        return llm_context
    
    def _get_fallback_action(self, 
                           game_state: Dict[str, Any],
                           context: Optional[Dict[str, Any]]) -> int:
        """Get fallback action using rule-based logic.
        
        Args:
            game_state: Game state information
            context: Additional context
            
        Returns:
            int: Action code (1-8)
        """
        # Check if stuck (use special stuck actions)
        if context and context.get('stuck_counter', 0) > 5:
            action_idx = context['action_count'] % len(self.stuck_actions)
            return self.stuck_actions[action_idx]
        
        # Check for special game states
        if game_state:
            # In battle - use A button
            if game_state.get('in_battle', 0):
                return 5  # A button
            
            # At title screen - use START or A
            if context and context.get('detected_state') == 'title_screen':
                return 7 if (context.get('action_count', 0) % 3) == 0 else 5
        
        # Default exploration pattern
        action_count = context.get('action_count', 0) if context else 0
        return self.fallback_actions[action_count % len(self.fallback_actions)]
    
    def _is_valid_action(self, action: Any) -> bool:
        """Check if action is valid.
        
        Args:
            action: Action to validate
            
        Returns:
            bool: True if action is valid
        """
        return isinstance(action, int) and 1 <= action <= 8

    def _is_action_appropriate(self, action: int, filter_result: Dict[str, Any]) -> bool:
        """Check if action is appropriate based on filter result.

        Args:
            action: Action to check
            filter_result: Result from action filter

        Returns:
            bool: True if action is appropriate
        """
        # Allow if action is in primary or secondary
        if action in filter_result.get('primary_actions', []):
            return True
        if action in filter_result.get('secondary_actions', []):
            return True

        # Disallow if action is forbidden
        if action in filter_result.get('forbidden_actions', []):
            return False

        # Be more lenient with discouraged actions (allow but log)
        if action in filter_result.get('discouraged_actions', []):
            self.logger.debug(f"Action {action} is discouraged but allowed")
            return True

        # Default to allowing if not categorized
        return True
    
    def record_experience(self,
                         game_state: Dict[str, Any],
                         context: Dict[str, Any],
                         action_taken: int,
                         outcome_reward: float,
                         consequences: Dict[str, Any] = None) -> None:
        """Record an experience in the memory system.

        Args:
            game_state: Game state when decision was made
            context: Context information
            action_taken: Action that was taken
            outcome_reward: Reward received
            consequences: What happened as a result
        """
        if self.experience_memory:
            try:
                self.experience_memory.record_experience(
                    game_state=game_state,
                    context=context,
                    action_taken=action_taken,
                    outcome_reward=outcome_reward,
                    consequences=consequences
                )
            except Exception as e:
                self.logger.warning(f"Failed to record experience: {e}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get decision engine statistics.

        Returns:
            Dict: Statistics about decision making
        """
        total_decisions = self.decision_count + self.failure_count
        success_rate = (self.success_count / total_decisions) if total_decisions > 0 else 0
        avg_response_time = sum(self.response_times) / len(self.response_times) if self.response_times else 0

        stats = {
            'total_decisions': total_decisions,
            'llm_decisions': self.decision_count,
            'success_count': self.success_count,
            'failure_count': self.failure_count,
            'success_rate': success_rate,
            'avg_response_time': avg_response_time,
            'consecutive_failures': self.consecutive_failures,
            'is_available': self.is_available
        }

        # Add experience memory statistics
        if self.experience_memory:
            try:
                experience_stats = self.experience_memory.get_statistics()
                stats['experience_memory'] = experience_stats
            except Exception as e:
                self.logger.warning(f"Failed to get experience memory stats: {e}")

        return stats
    
    def reset_failure_count(self) -> None:
        """Reset failure tracking to re-enable LLM."""
        self.consecutive_failures = 0
        if self.llm_agent is not None:
            self.is_available = True
            self.logger.info("LLM decision engine re-enabled")
    
    def shutdown(self) -> None:
        """Shutdown decision engine and cleanup resources."""
        if self.llm_agent and hasattr(self.llm_agent, 'shutdown'):
            try:
                self.llm_agent.shutdown()
            except Exception as e:
                self.logger.error(f"Error shutting down LLM agent: {e}")

        if self.experience_memory and hasattr(self.experience_memory, 'shutdown'):
            try:
                self.experience_memory.shutdown()
            except Exception as e:
                self.logger.error(f"Error shutting down experience memory: {e}")

        self.logger.info("LLM decision engine shutdown complete")