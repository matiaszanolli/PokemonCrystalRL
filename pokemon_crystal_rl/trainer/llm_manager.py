"""
LLM Manager module for Pokemon Crystal RL
"""

import os
import time
import logging
from typing import Optional, List, Dict, Any

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False


class LLMManager:
    """Manages interactions with local LLM models."""

    def __init__(self, model: str = "smollm2:1.7b", interval: int = 10):
        """Initialize LLM manager.
        
        Args:
            model: Name of the LLM model to use
            interval: How often to call LLM (in steps)
        """
        self.model = model
        self.interval = interval
        self.logger = logging.getLogger("pokemon_trainer.llm")

        # Stats tracking
        self.stats = {
            'calls': 0,
            'successes': 0,
            'failures': 0,
            'total_time': 0.0,
            'avg_time': 0.0
        }

        # Response tracking for adaptive interval
        self.llm_response_times: List[float] = []
        self.adaptive_llm_interval = interval

        if not OLLAMA_AVAILABLE:
            raise RuntimeError("Ollama not available - install with: pip install ollama")

        # Validate model exists
        try:
            ollama.show(model)
        except Exception as e:
            raise RuntimeError(f"Failed to load model {model}: {e}")

    def get_action(self) -> Optional[int]:
        """Get next action from LLM model."""
        try:
            start_time = time.time()
            # Call LLM to get action
            response = ollama.generate(
                model=self.model,
                prompt="You are playing Pokemon Crystal. What action should you take next? Choose from: 1=UP, 2=DOWN, 3=LEFT, 4=RIGHT, 5=A, 6=B, 7=START, 8=SELECT",
                system="You are an AI player playing Pokemon Crystal. Choose one action in each response."
            )
            action = int(response['response'].strip())

            # Track performance
            self._track_llm_performance(time.time() - start_time)
            self.stats['calls'] += 1
            self.stats['successes'] += 1

            return action if 1 <= action <= 8 else None

        except Exception as e:
            self.stats['failures'] += 1
            self.logger.warning(f"LLM action failed: {e}")
            return None

    def _track_llm_performance(self, response_time: float):
        """Track LLM performance for adaptive intervals."""
        self.stats['total_time'] += response_time
        if self.stats['calls'] > 0:
            self.stats['avg_time'] = self.stats['total_time'] / self.stats['calls']

        # Keep last 20 response times
        self.llm_response_times.append(response_time)
        if len(self.llm_response_times) > 20:
            self.llm_response_times = self.llm_response_times[-20:]

        # Adjust interval every 10 calls
        if len(self.llm_response_times) >= 10:
            avg_time = sum(self.llm_response_times[-10:]) / 10

            # If consistently slow (>3s), increase interval
            if avg_time > 3.0:
                self.adaptive_llm_interval = min(50, int(self.adaptive_llm_interval * 1.5))

            # If consistently fast (<1.5s), decrease interval
            elif avg_time < 1.5:
                self.adaptive_llm_interval = max(
                    self.interval,
                    int(self.adaptive_llm_interval * 0.8)
                )
