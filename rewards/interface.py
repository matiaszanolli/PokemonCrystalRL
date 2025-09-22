"""
Reward Calculator Interface Definition

This module defines the core interface that all reward calculators must implement
for the Pokemon Crystal RL training system. It establishes a consistent contract
for how reward calculation should be performed.
"""

from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional

class RewardCalculatorInterface(ABC):
    """Abstract base class defining the reward calculator interface.
    
    Any reward calculator implementation must provide these core methods to be
    compatible with the training system.
    """
    
    @abstractmethod
    def calculate_reward(self, current_state: Dict, previous_state: Dict) -> Tuple[float, Dict[str, float]]:
        """Calculate the reward based on current and previous game states.
        
        Args:
            current_state: Dictionary containing current game state data
            previous_state: Dictionary containing previous game state data
            
        Returns:
            tuple: (total_reward, reward_breakdown)
                - total_reward (float): The total calculated reward
                - reward_breakdown (Dict[str,float]): Itemized breakdown of rewards
        """
        pass
        
    @abstractmethod
    def get_reward_summary(self, rewards: Dict[str, float]) -> str:
        """Get a human-readable summary of the reward breakdown.
        
        Args:
            rewards: Dictionary mapping reward categories to values
            
        Returns:
            str: Human-readable summary of the rewards
        """
        pass
        
    @property
    @abstractmethod
    def last_screen_state(self) -> str:
        """Get the last known screen state.
        
        Returns:
            str: Current screen state identifier
        """
        pass
    
    @last_screen_state.setter
    @abstractmethod
    def last_screen_state(self, state: str):
        """Set the current screen state.
        
        Args:
            state: Current screen state identifier
        """
        pass
        
    @property
    @abstractmethod  
    def prev_screen_state(self) -> str:
        """Get the previous screen state.
        
        Returns:
            str: Previous screen state identifier
        """
        pass
    
    @prev_screen_state.setter
    @abstractmethod
    def prev_screen_state(self, state: str):
        """Set the previous screen state.
        
        Args:
            state: Previous screen state identifier 
        """
        pass
        
    @property
    @abstractmethod
    def last_action(self) -> Optional[str]:
        """Get the last action performed.
        
        Returns:
            Optional[str]: The last action taken, if any
        """
        pass
    
    @last_action.setter 
    @abstractmethod
    def last_action(self, action: Optional[str]):
        """Set the last action performed.
        
        Args:
            action: The last action taken
        """
        pass
