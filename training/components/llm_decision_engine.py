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
    from trainer.llm import LLMAgent
except ImportError:
    print("⚠️  LLMAgent not available")
    LLMAgent = None


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
        
        # Try LLM decision
        start_time = time.time()
        try:
            action = self._get_llm_decision(game_state, screen_data, context)
            response_time = time.time() - start_time
            
            if self._is_valid_action(action):
                self.decision_count += 1
                self.success_count += 1
                self.response_times.append(response_time)
                self.consecutive_failures = 0
                
                decision_meta.update({
                    'source': 'llm',
                    'confidence': 0.8,
                    'response_time': response_time,
                    'success': True
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
                         context: Optional[Dict[str, Any]]) -> int:
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

        # Convert action string to integer (1-8)
        try:
            action = int(action_str) if action_str.isdigit() else 1
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
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get decision engine statistics.
        
        Returns:
            Dict: Statistics about decision making
        """
        total_decisions = self.decision_count + self.failure_count
        success_rate = (self.success_count / total_decisions) if total_decisions > 0 else 0
        avg_response_time = sum(self.response_times) / len(self.response_times) if self.response_times else 0
        
        return {
            'total_decisions': total_decisions,
            'llm_decisions': self.decision_count,
            'success_count': self.success_count,
            'failure_count': self.failure_count,
            'success_rate': success_rate,
            'avg_response_time': avg_response_time,
            'consecutive_failures': self.consecutive_failures,
            'is_available': self.is_available
        }
    
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
        
        self.logger.info("LLM decision engine shutdown complete")