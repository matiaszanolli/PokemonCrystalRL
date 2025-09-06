"""Base interfaces for trainer components."""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import numpy as np


@dataclass
class TrainerConfig:
    """Configuration for trainer components."""
    rom_path: str = ""
    save_path: Optional[str] = None
    headless: bool = True
    observation_type: str = "minimal"
    max_steps: int = 1000000
    action_repeat: int = 1
    checkpoint_interval: int = 1000
    

@dataclass 
class TrainingState:
    """Current state of training session."""
    episode: int = 0
    total_steps: int = 0
    total_reward: float = 0.0
    current_step: int = 0
    is_running: bool = False
    is_paused: bool = False
    current_phase: str = "initializing"


@dataclass
class GameState:
    """Current state of game environment."""
    in_battle: bool = False
    in_menu: bool = False
    in_overworld: bool = True
    map_id: int = 0
    player_x: int = 0
    player_y: int = 0
    party_count: int = 0
    money: int = 0
    badges: int = 0


class PyBoyInterface(ABC):
    """Interface for PyBoy emulator functionality."""
    
    @abstractmethod
    def get_memory_value(self, address: int) -> int:
        """Get value at memory address."""
        pass
        
    @abstractmethod
    def set_memory_value(self, address: int, value: int) -> None:
        """Set value at memory address."""
        pass
        
    @abstractmethod
    def get_screen(self) -> np.ndarray:
        """Get current screen state."""
        pass
        
    @abstractmethod
    def press(self, button: str) -> None:
        """Press a button."""
        pass
        
    @abstractmethod
    def release(self, button: str) -> None:
        """Release a button."""
        pass
        
    @abstractmethod
    def tick(self) -> None:
        """Advance emulator by one frame."""
        pass


class AgentInterface(ABC):
    """Interface for AI agent implementations."""
    
    @abstractmethod
    def get_action(self, observation: Dict[str, Any], info: Dict[str, Any] = None) -> str:
        """Get next action from agent.
        
        Args:
            observation: Current observation from environment
            info: Additional information about environment state
            
        Returns:
            Action to take
        """
        pass
        
    @abstractmethod
    def update(self, reward: float) -> None:
        """Update agent with reward.
        
        Args:
            reward: Reward from last action
        """
        pass
        
    @abstractmethod
    def reset(self) -> None:
        """Reset agent state."""
        pass


class RewardCalculatorInterface(ABC):
    """Interface for reward calculation."""
    
    @abstractmethod
    def calculate(self, prev_state: GameState, current_state: GameState) -> float:
        """Calculate reward between states.
        
        Args:
            prev_state: Previous game state
            current_state: Current game state
            
        Returns:
            Calculated reward value
        """
        pass
        
    @abstractmethod
    def get_reward_breakdown(self) -> Dict[str, float]:
        """Get breakdown of reward components.
        
        Returns:
            Dictionary mapping reward sources to values
        """
        pass


class TrainerInterface(ABC):
    """Interface for Pokemon trainer implementations."""
    
    @abstractmethod
    def initialize(self, config: TrainerConfig) -> bool:
        """Initialize the trainer.
        
        Args:
            config: Trainer configuration
            
        Returns:
            True if initialization successful
        """
        pass
    
    @abstractmethod
    def start_training(self) -> None:
        """Start or resume training."""
        pass
        
    @abstractmethod
    def stop_training(self) -> None:
        """Stop training."""
        pass
        
    @abstractmethod
    def pause_training(self) -> None:
        """Pause training."""
        pass
        
    @abstractmethod
    def get_state(self) -> TrainingState:
        """Get current training state.
        
        Returns:
            Current training state
        """
        pass
        
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get training statistics.
        
        Returns:
            Dictionary of training statistics
        """
        pass
        
    @abstractmethod
    def save_checkpoint(self) -> str:
        """Save training checkpoint.
        
        Returns:
            Path to saved checkpoint
        """
        pass
        
    @abstractmethod
    def load_checkpoint(self, checkpoint_path: str) -> bool:
        """Load training checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint to load
            
        Returns:
            True if checkpoint loaded successfully
        """
        pass
