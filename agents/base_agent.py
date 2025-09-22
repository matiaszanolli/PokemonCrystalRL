"""Base agent implementation."""
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod
from interfaces.trainers import AgentInterface


class BaseAgent(AgentInterface):
    """Base class for Pokemon Crystal agents."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the agent.
        
        Args:
            config: Optional configuration parameters
        """
        self.config = config or {}
        self.total_steps = 0
        self.total_reward = 0.0
        self.current_episode = 0
        self.episode_reward = 0.0
        self.episode_steps = 0
        self._is_training = True
        
    def update(self, reward: float) -> None:
        """Update agent with reward from environment.
        
        Args:
            reward: Reward value from last action
        """
        self.total_reward += reward
        self.episode_reward += reward
        self.total_steps += 1
        self.episode_steps += 1
        
    def reset(self) -> None:
        """Reset agent state for new episode."""
        self.episode_reward = 0.0
        self.episode_steps = 0
        self.current_episode += 1
        
    def train(self) -> None:
        """Put agent in training mode."""
        self._is_training = True
        
    def eval(self) -> None:
        """Put agent in evaluation mode."""
        self._is_training = False
        
    @property
    def is_training(self) -> bool:
        """Whether agent is in training mode."""
        return self._is_training

    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics.
        
        Returns:
            Dictionary of agent statistics
        """
        return {
            'total_steps': self.total_steps,
            'total_reward': self.total_reward,
            'current_episode': self.current_episode,
            'episode_reward': self.episode_reward,
            'episode_steps': self.episode_steps,
            'is_training': self.is_training
        }
