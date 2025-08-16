"""
LLM management and decision tracking for Pokemon Crystal RL Trainer
"""

import time
import ollama
import numpy as np
from datetime import datetime
from typing import Optional, List, Dict, Any
import logging

from .config import STATE_TEMPERATURES, STATE_GUIDANCE, ACTION_NAMES


class LLMManager:
    """Manages LLM interactions, decision tracking, and performance monitoring"""
    
    def __init__(self, config, game_state_detector):
        self.config = config
        self.game_state_detector = game_state_detector
        self.logger = logging.getLogger(__name__)
        
        # Performance tracking
        self.llm_response_times = []
        self.adaptive_llm_interval = config.llm_interval
        
        # Decision tracking for web monitoring
        self.llm_decisions = []
        self.last_llm_decision = None
        
        # Stats integration
        self.stats = {
            'llm_calls': 0,
            'llm_total_time': 0.0,
            'llm_avg_time': 0.0
        }
        
        # Initialize LLM if backend is specified
        if self.config.llm_backend and self.config.llm_backend.value:
            self._initialize_llm_backend()
    
    def _initialize_llm_backend(self):
        """Initialize LLM backend"""
        model_name = self.config.llm_backend.value
        
        try:
            # Check if model is available
            ollama.show(model_name)
            self.logger.info(f"✅ Using LLM model: {model_name}")
        except:
            self.logger.info(f"📥 Pulling LLM model: {model_name}")
            ollama.pull(model_name)
    
    def get_llm_action(self, screenshot: Optional[np.ndarray], stage: str = "BASIC_CONTROLS") -> int:
        """Get action from LLM with state-aware configuration and performance monitoring"""
        if not self.config.llm_backend or not self.config.llm_backend.value:
            return 5  # Default A button
        
        # Start timing
        llm_start_time = time.time()
        
        try:
            # Time state detection  
            state_start = time.time()
            current_state = self.game_state_detector.detect_game_state(screenshot)
            state_time = time.time() - state_start
            
            # State-specific temperature settings
            temperature = STATE_TEMPERATURES.get(current_state, 0.6)
            
            # Build state-aware prompt
            state_guidance = STATE_GUIDANCE.get(current_state, "Use 5=A to interact or movement keys 1/2/3/4")
            prompt = f"""Pokemon Crystal Game Bot

State: {current_state}
Stage: {stage}
Guidance: {state_guidance}

Controls:
1=UP, 2=DOWN, 3=LEFT, 4=RIGHT, 5=A, 6=B, 7=START, 8=SELECT

Choose action number (1-8):"""
            
            # Time LLM generation
            generation_start = time.time()
            response = ollama.generate(
                model=self.config.llm_backend.value,
                prompt=prompt,
                options={
                    'num_predict': 3,
                    'temperature': temperature,
                    'top_k': 8,
                    'timeout': 8  # Add timeout to prevent hanging
                }
            )
            generation_time = time.time() - generation_start
            
            # Total timing
            total_time = time.time() - llm_start_time
            
            # Track response times for adaptive performance
            self._track_llm_performance(total_time)
            
            # Log performance periodically
            if self.config.debug_mode or self.stats['llm_calls'] % 10 == 0:
                self.logger.info(f"⚡ LLM Performance: Total={total_time:.2f}s (State={state_time:.3f}s, "
                               f"Generation={generation_time:.2f}s) State={current_state}")
            
            # Track slow calls
            if total_time > 5.0:
                self.logger.warning(f"🐌 SLOW LLM CALL: {total_time:.2f}s - investigating...")
                if generation_time > 4.0:
                    self.logger.warning(f"  ⚠️ Generation bottleneck: {generation_time:.2f}s")
                if state_time > 0.5:
                    self.logger.warning(f"  ⚠️ State detection bottleneck: {state_time:.2f}s")
            
            # Parse action
            text = response['response'].strip()
            action = 5  # Default
            for char in text:
                if char.isdigit() and '1' <= char <= '8':
                    action = int(char)
                    break
            
            # Save LLM decision for monitoring
            self._save_llm_decision(action, current_state, prompt, text, total_time)
            
            return action
        
        except Exception as e:
            total_time = time.time() - llm_start_time
            self.logger.warning(f"⚠️ LLM error after {total_time:.2f}s: {str(e)[:100]}")
            return 5
    
    def get_llm_action_with_vision(self, screenshot: Optional[np.ndarray], total_actions: int) -> int:
        """Get LLM action using visual input with enhanced context and performance monitoring"""
        if not self.config.llm_backend or not self.config.llm_backend.value:
            return self._get_fallback_action(total_actions)
        
        # Start timing
        llm_start_time = time.time()
        
        try:
            # Time state detection
            state_start = time.time()
            current_state = self.game_state_detector.detect_game_state(screenshot) if screenshot is not None else "unknown"
            state_time = time.time() - state_start
            
            # Check if we're stuck and need anti-stuck behavior
            if self.game_state_detector.is_stuck():
                if self.config.debug_mode:
                    stuck_info = self.game_state_detector.get_stuck_info()
                    self.logger.info(f"🤖 LLM: Anti-stuck mode activated (stuck for {stuck_info['consecutive_same_screens']} frames)")
                return self._get_fallback_action(total_actions)
            
            # Create state-specific prompts for better decision making
            state_specific_guidance = STATE_GUIDANCE.get(current_state, "Try 5=A to interact, or movement keys 1/2/3/4, use 6=B to exit menus")
            
            prompt = f"""Pokemon Crystal Game Bot

State: {current_state}
Goal: {state_specific_guidance}
Step: {total_actions}

Actions: 1=UP 2=DOWN 3=LEFT 4=RIGHT 5=A 6=B 7=START 8=SELECT

Respond with only one digit (1-8):
"""
            
            # State-specific temperature settings
            temperature = STATE_TEMPERATURES.get(current_state, 0.6)
            
            # Time LLM generation
            generation_start = time.time()
            response = ollama.generate(
                model=self.config.llm_backend.value,
                prompt=prompt,
                options={
                    'num_predict': 3,
                    'temperature': temperature,
                    'top_k': 8,
                    'timeout': 8  # Increased timeout consistent with other method
                }
            )
            generation_time = time.time() - generation_start
            
            # Total timing
            total_time = time.time() - llm_start_time
            
            # Track response times for adaptive performance
            self._track_llm_performance(total_time)
            
            # Log performance periodically
            if self.config.debug_mode or self.stats['llm_calls'] % 10 == 0:
                self.logger.info(f"⚡ LLM Vision Performance: Total={total_time:.2f}s "
                               f"(State={state_time:.3f}s, Generation={generation_time:.2f}s) State={current_state}")
            
            # Track slow calls
            if total_time > 5.0:
                self.logger.warning(f"🐌 SLOW LLM VISION CALL: {total_time:.2f}s - investigating...")
                if generation_time > 4.0:
                    self.logger.warning(f"  ⚠️ Generation bottleneck: {generation_time:.2f}s")
                if state_time > 0.5:
                    self.logger.warning(f"  ⚠️ State detection bottleneck: {state_time:.2f}s")
            
            # Parse action from response
            text = response['response'].strip().lower()
            
            # Look for numbers in response
            action = None
            for char in text:
                if char.isdigit() and '1' <= char <= '8':
                    action = int(char)
                    if self.config.debug_mode and total_actions % 20 == 0:
                        self.logger.info(f"🤖 LLM chose action {action} for state '{current_state}'")
                    break
            
            if action is not None:
                # Save LLM decision for monitoring
                self._save_llm_decision(action, current_state, prompt, text, total_time)
                return action
            
            # Fallback if parsing fails
            if self.config.debug_mode:
                self.logger.warning(f"⚠️ LLM response couldn't be parsed: '{text[:20]}...', using fallback")
            return self._get_fallback_action(total_actions)
            
        except Exception as e:
            total_time = time.time() - llm_start_time
            self.logger.warning(f"⚠️ LLM vision call error after {total_time:.2f}s: {str(e)[:100]}, using fallback")
            return self._get_fallback_action(total_actions)
    
    def _get_fallback_action(self, step: int) -> int:
        """Get fallback action when LLM is not available or fails"""
        # Import here to avoid circular import
        from .game_state import get_unstuck_action
        from .training_strategies import get_rule_based_action
        
        if self.game_state_detector.is_stuck():
            stuck_info = self.game_state_detector.get_stuck_info()
            return get_unstuck_action(step, stuck_info['stuck_counter'])
        else:
            return get_rule_based_action(self.game_state_detector._cached_state, step)
    
    def _save_llm_decision(self, action: int, state: str, prompt: str, response_text: str, response_time: float):
        """Save LLM decision for web monitoring and debugging"""
        decision_data = {
            'timestamp': time.time(),
            'timestamp_iso': datetime.now().isoformat(),
            'step': getattr(self, '_current_step', 0),
            'action': action,
            'state': state,
            'prompt': prompt[:200] + '...' if len(prompt) > 200 else prompt,  # Truncate long prompts
            'response': response_text[:100] + '...' if len(response_text) > 100 else response_text,
            'response_time': response_time,
            'model': self.config.llm_backend.value if self.config.llm_backend else 'unknown',
        }
        
        # Update last decision for quick access
        self.last_llm_decision = decision_data
        
        # Add to history (keep last 50 decisions)
        self.llm_decisions.append(decision_data)
        if len(self.llm_decisions) > 50:
            self.llm_decisions = self.llm_decisions[-25:]  # Keep last 25
    
    def _track_llm_performance(self, response_time: float):
        """Track LLM performance and adapt interval if needed"""
        # Add to response times (keep last 20 calls)
        self.llm_response_times.append(response_time)
        if len(self.llm_response_times) > 20:
            self.llm_response_times.pop(0)
        
        # Update stats
        self.stats['llm_calls'] += 1
        self.stats['llm_total_time'] += response_time
        self.stats['llm_avg_time'] = self.stats['llm_total_time'] / self.stats['llm_calls']
        
        # Adaptive interval adjustment every 10 calls
        if len(self.llm_response_times) >= 10 and len(self.llm_response_times) % 10 == 0:
            avg_time = sum(self.llm_response_times[-10:]) / 10
            
            # If LLM is consistently slow (>3s), increase interval
            if avg_time > 3.0 and self.adaptive_llm_interval < 50:
                old_interval = self.adaptive_llm_interval
                self.adaptive_llm_interval = min(50, int(self.adaptive_llm_interval * 1.5))
                if self.config.debug_mode:
                    self.logger.info(f"📈 LLM slow ({avg_time:.1f}s avg), increasing interval: {old_interval} → {self.adaptive_llm_interval}")
            
            # If LLM is consistently fast (<1s), decrease interval
            elif avg_time < 1.0 and self.adaptive_llm_interval > self.config.llm_interval:
                old_interval = self.adaptive_llm_interval
                self.adaptive_llm_interval = max(self.config.llm_interval, int(self.adaptive_llm_interval * 0.8))
                if self.config.debug_mode:
                    self.logger.info(f"📉 LLM fast ({avg_time:.1f}s avg), decreasing interval: {old_interval} → {self.adaptive_llm_interval}")
    
    def should_use_llm(self, step: int) -> bool:
        """Determine if LLM should be used for this step"""
        if not self.config.llm_backend or not self.config.llm_backend.value:
            return False
        return step % self.adaptive_llm_interval == 0
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get LLM performance metrics"""
        return {
            'total_llm_calls': self.stats['llm_calls'],
            'avg_response_time': self.stats['llm_avg_time'],
            'total_llm_time': self.stats['llm_total_time'],
            'adaptive_interval': self.adaptive_llm_interval,
            'current_model': self.config.llm_backend.value if self.config.llm_backend else 'rule-based'
        }
    
    def get_decision_data(self) -> Dict[str, Any]:
        """Get LLM decision data for web monitoring"""
        return {
            'recent_decisions': self.llm_decisions[-20:] if self.llm_decisions else [],
            'last_decision': self.last_llm_decision,
            'total_decisions': len(self.llm_decisions),
            'performance_metrics': self.get_performance_metrics(),
            'state_distribution': self._calculate_state_distribution(),
            'action_distribution': self._calculate_action_distribution()
        }
    
    def _calculate_state_distribution(self) -> Dict[str, int]:
        """Calculate distribution of game states from recent LLM decisions"""
        if not self.llm_decisions:
            return {}
        
        state_counts = {}
        for decision in self.llm_decisions[-30:]:  # Last 30 decisions
            state = decision.get('state', 'unknown')
            state_counts[state] = state_counts.get(state, 0) + 1
        
        return state_counts
    
    def _calculate_action_distribution(self) -> Dict[str, int]:
        """Calculate distribution of actions from recent LLM decisions"""
        if not self.llm_decisions:
            return {}
        
        action_counts = {}
        
        for decision in self.llm_decisions[-30:]:  # Last 30 decisions
            action = decision.get('action', 0)
            action_name = ACTION_NAMES.get(action, f'UNKNOWN({action})')
            action_counts[action_name] = action_counts.get(action_name, 0) + 1
        
        return action_counts
    
    def update_step(self, step: int):
        """Update current step for decision tracking"""
        self._current_step = step
