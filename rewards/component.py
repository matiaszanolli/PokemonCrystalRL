"""
Reward Component Base Class

This module defines the base component class used to construct modular
reward calculators. Each type of reward (health, level, badges, etc.)
should be implemented as a separate component inheriting from this base.
"""

from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional, Set
from dataclasses import dataclass, field

@dataclass
class StateValidation:
    """Data class for state validation rules."""
    required_fields: Set[str] = field(default_factory=set)
    value_ranges: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    require_screen_state: bool = False
    allowed_screen_states: Set[str] = field(default_factory=set)

class RewardComponent(ABC):
    """Base class for all reward components.
    
    Each component handles one specific aspect of reward calculation
    (e.g., health rewards, level rewards, etc.) and can maintain its
    own state and validation rules.
    """
    
    def __init__(self, name: str):
        """Initialize reward component.
        
        Args:
            name: Unique identifier for this reward component
        """
        self.name = name
        self._last_screen_state: Optional[str] = None
        self._prev_screen_state: Optional[str] = None
        self._last_action: Optional[str] = None
        
    @property
    def last_screen_state(self) -> Optional[str]:
        """Get current screen state."""
        return self._last_screen_state
        
    @last_screen_state.setter
    def last_screen_state(self, state: str):
        """Set current screen state."""
        self._last_screen_state = state
        
    @property
    def prev_screen_state(self) -> Optional[str]:
        """Get previous screen state."""
        return self._prev_screen_state
        
    @prev_screen_state.setter 
    def prev_screen_state(self, state: str):
        """Set previous screen state."""
        self._prev_screen_state = state
        
    @property
    def last_action(self) -> Optional[str]:
        """Get last action."""
        return self._last_action
        
    @last_action.setter
    def last_action(self, action: Optional[str]):
        """Set last action."""
        self._last_action = action
    
    @abstractmethod
    def get_validation_rules(self) -> StateValidation:
        """Get the validation rules for this component.
        
        Returns:
            StateValidation: The validation rules to apply
        """
        pass
    
    def validate_state(self, state: Dict) -> bool:
        """Validate state data against component rules.
        
        Args:
            state: Game state dictionary to validate
            
        Returns:
            bool: True if state is valid, False otherwise
        """
        rules = self.get_validation_rules()
        
        # Check required fields
        if not all(field in state for field in rules.required_fields):
            return False
            
        # Check value ranges
        for field, (min_val, max_val) in rules.value_ranges.items():
            if field in state:
                value = state[field]
                if not (min_val <= value <= max_val):
                    return False
                    
        # Check screen state if required
        if rules.require_screen_state:
            if not self.last_screen_state or self.last_screen_state not in rules.allowed_screen_states:
                return False
            if not self.prev_screen_state or self.prev_screen_state not in rules.allowed_screen_states:
                return False
                
        return True
        
    @abstractmethod
    def calculate(self, current_state: Dict, previous_state: Dict) -> Tuple[float, Dict[str, float]]:
        """Calculate reward value for this component.
        
        Args:
            current_state: Current game state
            previous_state: Previous game state
            
        Returns:
            tuple: (reward_value, reward_details)
                - reward_value (float): The calculated reward
                - reward_details (dict): Additional reward details/breakdowns
        """
        pass
