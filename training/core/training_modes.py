"""
Training mode implementations for Pokemon Crystal RL.
"""

import logging
from typing import Any, Dict, Optional
from ..config import TrainingMode


class TrainingModeManager:
    """Manages different training mode implementations."""
    
    def __init__(self, trainer: Any, logger: Optional[logging.Logger] = None):
        self.trainer = trainer
        self.logger = logger or logging.getLogger(__name__)
    
    def run_training_mode(self, mode: TrainingMode):
        """Run training in specified mode."""
        if mode == TrainingMode.FAST_MONITORED:
            self._run_legacy_fast_training()
        elif mode == TrainingMode.ULTRA_FAST:
            self._run_ultra_fast_training()
        elif mode == TrainingMode.CURRICULUM:
            self._run_synchronized_training()
        elif mode == TrainingMode.CUSTOM:
            self._run_synchronized_training()  # Default to synchronized for custom
        else:
            raise ValueError(f"Unknown training mode: {mode}")
    
    def simulate_integration_decisions(self, episodes: int, steps_per_episode: int):
        """Simulate decision recording for integration testing."""
        # Check if decision analyzer was attached for integration testing
        decision_analyzer = getattr(self.trainer, 'decision_analyzer', None)
        
        # Record some sample decisions if analyzer available
        if decision_analyzer and hasattr(decision_analyzer, 'add_decision'):
            for episode in range(episodes):
                for step in range(min(steps_per_episode, 5)):  # Limit for test efficiency
                    decision_data = {
                        'state_hash': hash(f'episode_{episode}_step_{step}'),
                        'action': step % 8,  # Mock action
                        'context': {'source': 'integration_test', 'confidence': 0.8},
                        'outcome': 'success' if step % 2 == 0 else 'neutral',
                        'step_in_episode': step,
                        'total_episode_reward': 10.0 + episode * 2
                    }
                    try:
                        decision_analyzer.add_decision(decision_data)
                    except Exception:
                        pass  # Ignore errors in test context
    
    def _run_legacy_fast_training(self):
        """Run training in legacy fast mode with basic monitoring."""
        self.trainer.stats['total_actions'] = 0
        while (self.trainer.stats['total_actions'] < self.trainer.config.max_actions and 
               self.trainer._training_active):
            try:
                action = self.trainer._get_rule_based_action(self.trainer.stats['total_actions'])
                self.trainer._execute_action(action)
                self.trainer.stats['total_actions'] += 1

                if self.trainer.stats['total_actions'] % 20 == 0:
                    self.trainer._update_stats()
            except Exception:
                # Don't count towards max actions if failed
                self.trainer.stats['total_actions'] -= 1
                continue
                
    def _run_ultra_fast_training(self):
        """Run training in ultra-fast mode."""
        self.trainer.stats['total_actions'] = 0
        self.trainer._training_active = True
        pyboy = self.trainer.pyboy_manager.get_pyboy() if self.trainer.pyboy_manager else None
        
        while (self.trainer.stats['total_actions'] < self.trainer.config.max_actions and 
               self.trainer._training_active and pyboy):
            try:
                # Execute multiple frames per action without checks
                action = self.trainer._get_rule_based_action(self.trainer.stats['total_actions'])
                for _ in range(self.trainer.config.frames_per_action):
                    pyboy.send_input(action)
                    pyboy.tick()
                self.trainer.stats['total_actions'] += 1

                if self.trainer.stats['total_actions'] % 100 == 0:
                    self.trainer._update_stats()
            except Exception:
                # Don't count towards max actions if failed
                self.trainer.stats['total_actions'] -= 1
                continue

    def _run_synchronized_training(self):
        """Run training in synchronized mode with monitoring."""
        self.trainer._training_active = True
        while (self.trainer.stats['total_actions'] < self.trainer.config.max_actions and 
               self.trainer._training_active):
            
            # Get action (LLM or rule-based)
            if (self.trainer.llm_manager and 
                self.trainer.stats['total_actions'] % self.trainer.config.llm_interval == 0):
                action = self.trainer._get_llm_action()
            else:
                action = self.trainer._get_rule_based_action(self.trainer.stats['total_actions'])

            if action:
                try:
                    self.trainer._execute_synchronized_action(action)
                    self.trainer.stats['total_actions'] += 1

                    # Update stats and capture periodically
                    if self.trainer.stats['total_actions'] % 10 == 0:
                        self.trainer._update_stats()
                        
                    if self.trainer.stats['total_actions'] % 50 == 0:
                        self.trainer._capture_screenshot()
                        
                except Exception as e:
                    self.logger.error(f"Error in synchronized training: {e}")
                    continue
            else:
                break