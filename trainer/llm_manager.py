"""
LLM Manager module for Pokemon Crystal RL
"""

import os
import time
import logging
import re
import numpy as np
from typing import Optional, List, Dict, Any, Tuple

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
        
        # Temperature configurations for different states
        self.state_temperatures = {
            "dialogue": 0.8,      # High temperature for variety in responses
            "menu": 0.6,          # Medium temperature for navigation
            "battle": 0.8,        # High temperature for combat variety
            "overworld": 0.7,     # Medium-high for exploration
            "title_screen": 0.5   # Lower temperature for consistent startup
        }
        
        # Prompt templates for different states
        self.state_prompts = {
            "dialogue": "You are in a dialogue. Use A (5) to advance text, or make a choice with UP (1) or DOWN (2) if options are present.",
            "menu": "You are in a menu. Navigate with UP (1), DOWN (2), SELECT items with A (5), or exit with B (6).",
            "battle": "You are in a battle! Choose moves with A (5), switch Pokemon with UP/DOWN (1/2), or use items with RIGHT (4).",
            "overworld": "You are exploring the world. Move with UP (1), DOWN (2), LEFT (3), RIGHT (4). Use A (5) to interact.",
            "title_screen": "You are at the title screen. Press START (7) to begin, or navigate options with UP (1) and DOWN (2)."
        }
        
        # Prompt effectiveness tracking
        self.prompt_effectiveness = {}
        self.last_prompt = None
        self.last_action = None

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

    def get_action(self, screenshot: Optional[np.ndarray] = None, game_state: str = "overworld", step: int = 0, stuck_counter: int = 0) -> Optional[int]:
        """Get next action from LLM model."""
        try:
            start_time = time.time()
            # Get state-specific prompt and temperature
            prompt = self._get_state_prompt(game_state, stuck_counter)
            temperature = self.state_temperatures.get(game_state, 0.7)
            
            # Call LLM to get action
            response = ollama.generate(
                model=self.model,
                prompt=prompt,
                system="You are an AI player playing Pokemon Crystal. Choose one action in each response.",
                options={
                    'temperature': temperature
                }
            )
            
            # Parse action from response
            action = self._parse_action(response['response'])
            
            # Track performance
            self._track_llm_performance(time.time() - start_time)
            self.stats['calls'] += 1
            
            if action is not None:
                self.stats['successes'] += 1
                self._track_prompt_effectiveness(prompt, True)
                return action
            else:
                self.stats['failures'] += 1
                self._track_prompt_effectiveness(prompt, False)
                return self._get_fallback_action(game_state, stuck_counter)

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
                
    def _parse_action(self, response: str) -> Optional[int]:
        """Parse action from LLM response.
        
        Args:
            response: LLM response text
            
        Returns:
            Parsed action number or None if invalid
        """
        # First try to find a single digit
        matches = re.findall(r'\b[1-8]\b', response)
        if matches:
            action = int(matches[0])
            if 1 <= action <= 8:
                return action
        
        # Try more complex patterns
        patterns = [
            r'press\s+([1-8])',
            r'action[:\s]+([1-8])',
            r'([1-8])\s*[-=]',
            r'key\s+([1-8])',
            r'button\s+([1-8])'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response.lower())
            if match:
                action = int(match.group(1))
                if 1 <= action <= 8:
                    return action
        
        return None
        
    def _get_state_prompt(self, game_state: str, stuck_counter: int) -> str:
        """Get appropriate prompt for current game state.
        
        Args:
            game_state: Current game state
            stuck_counter: Number of times agent has been stuck
            
        Returns:
            Prompt string for LLM
        """
        base_prompt = "You are playing Pokemon Crystal. Choose the next action from: "
        controls = "1=UP, 2=DOWN, 3=LEFT, 4=RIGHT, 5=A, 6=B, 7=START, 8=SELECT"
        
        # Get state-specific guidance
        state_guidance = self.state_prompts.get(
            game_state,
            "Choose an appropriate action for the current situation."
        )
        
        # Add stuck guidance if needed
        if stuck_counter > 0:
            state_guidance += "\nYou seem to be stuck. Try a different action than before."
        
        # Combine components
        prompt = f"{base_prompt}{controls}\n\n{state_guidance}\n\nRespond with a single action number."
        
        return prompt
        
    def _get_fallback_action(self, game_state: str, stuck_counter: int) -> int:
        """Get fallback action when LLM fails.
        
        Args:
            game_state: Current game state
            stuck_counter: Number of times agent has been stuck
            
        Returns:
            Fallback action number
        """
        # State-specific fallbacks
        fallbacks = {
            "dialogue": 5,     # A button
            "menu": 2,        # DOWN
            "battle": 5,      # A button
            "title_screen": 7 # START
        }
        
        if stuck_counter > 0:
            # When stuck, cycle through movement actions
            return (stuck_counter % 4) + 1  # 1-4 for UP, DOWN, LEFT, RIGHT
        
        return fallbacks.get(game_state, 5)  # Default to A button
        
    def _track_prompt_effectiveness(self, prompt: str, success: bool) -> None:
        """Track effectiveness of prompts for optimization.
        
        Args:
            prompt: The prompt that was used
            success: Whether it produced a valid action
        """
        if prompt not in self.prompt_effectiveness:
            self.prompt_effectiveness[prompt] = {
                'uses': 0,
                'successes': 0,
                'failures': 0
            }
            
        self.prompt_effectiveness[prompt]['uses'] += 1
        if success:
            self.prompt_effectiveness[prompt]['successes'] += 1
        else:
            self.prompt_effectiveness[prompt]['failures'] += 1
