"""
Core Pokemon trainer implementation focusing on training logic only.
Infrastructure concerns delegated to manager classes.
"""

import os
import time
import queue
import logging
import threading
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List

from monitoring.data_bus import get_data_bus, DataType
from environments.game_state_detection import GameStateDetector

from ..config import TrainingConfig, TrainingMode
from ..infrastructure import PyBoyManager, WebIntegrationManager
from .training_modes import TrainingModeManager


class PokemonTrainer:
    """Core Pokemon trainer focusing on training logic."""
    
    # Class-level cache for fallback actions
    _FALLBACK_ACTIONS = [5, 7, 5, 1, 2, 3, 4]  # A, START, A, UP, DOWN, LEFT, RIGHT
    
    def __init__(self, config: TrainingConfig):
        """Initialize the trainer with configuration."""
        self.config = config
        self.data_bus = None  # Initialize data bus to None
        
        # Initialize synchronization primitives FIRST
        self._shared_lock = threading.RLock()  # Re-entrant lock for thread safety
        self._sync_lock = self._shared_lock  # Alias for backward compatibility
        self._shutdown_event = threading.Event()
        
        # Initialize managers
        self.logger = logging.getLogger("pokemon_trainer")
        self.pyboy_manager = PyBoyManager(config, self.logger)
        self.web_manager = WebIntegrationManager(config, self, self.logger)
        self.training_modes = TrainingModeManager(self, self.logger)
        
        # Skip heavy initialization in test mode
        if getattr(self.config, 'test_mode', False):
            self.web_monitor = None  # Disable web monitoring in tests
            self.llm_manager = None  # Disable LLM in tests
            self.stats = self.init_stats()
            self.init_error_tracking()
            self._setup_queues()
        else:
            self.setup_logging()
            self.init_error_tracking()
            self._setup_queues()
            
            # Use managers for infrastructure setup
            if config.rom_path:
                self.pyboy_manager.setup_pyboy()
            
            self.web_manager.setup_web_server()
            self.setup_llm_manager()
            self.stats = self.init_stats()
            
            # Connect PyBoy to web monitor if both available
            if self.pyboy_manager.is_initialized() and self.web_manager.is_web_enabled():
                self.web_manager.update_pyboy_reference(self.pyboy_manager.get_pyboy())
        
        # Setup core training components
        self.game_state_detector = GameStateDetector()
        self._training_active = False
        
        # Backward compatibility properties
        self.pyboy = self.pyboy_manager.get_pyboy()
        self.web_monitor = self.web_manager.get_web_monitor()
        self.web_server = self.web_manager.web_server
        self.web_thread = self.web_manager.web_thread
    
    def _setup_queues(self):
        """Setup communication queues."""
        try:
            self.screenshot_queue = queue.Queue(maxsize=self.config.screenshot_buffer_size)
            self.action_queue = queue.Queue(maxsize=100)
            self.stats_queue = queue.Queue(maxsize=50)
        except Exception as e:
            self.logger.error(f"Failed to setup queues: {e}")
            # Create basic queues as fallback
            self.screenshot_queue = queue.Queue()
            self.action_queue = queue.Queue()
            self.stats_queue = queue.Queue()
    
    def setup_logging(self):
        """Setup logging configuration."""
        log_level = getattr(logging, self.config.log_level.upper(), logging.INFO)
        self.logger = logging.getLogger("pokemon_trainer")
        
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        self.logger.setLevel(log_level)
    
    def init_error_tracking(self):
        """Initialize error tracking system."""
        self.error_counts = {'total_errors': 0}
        self.last_errors = []
        
        # Data bus integration
        try:
            self.data_bus = get_data_bus()
        except Exception as e:
            self.logger.warning(f"Failed to initialize data bus: {e}")
            self.data_bus = None
    
    def setup_llm_manager(self):
        """Setup LLM manager if enabled."""
        # For now, keep minimal LLM setup
        self.llm_manager = None
        if self.config.llm_backend and self.config.llm_backend.value:
            # Future: Initialize LLM manager here
            pass
    
    def init_stats(self) -> Dict[str, Any]:
        """Initialize training statistics."""
        return {
            'total_actions': 0,
            'total_episodes': 0,
            'current_episode': 0,
            'start_time': time.time(),
            'last_update': time.time(),
            'actions_per_second': 0.0,
            'episodes_completed': 0,
            'episode_rewards': [],
            'session_duration': 0.0
        }
    
    def train(self, total_episodes=None, max_steps_per_episode=None, save_interval=None, eval_interval=None, **kwargs):
        """Main training entry point."""
        if not self.pyboy_manager.is_initialized():
            self.logger.warning("PyBoy not initialized - running simulation mode")
            
            # For integration tests, simulate some training decisions
            episodes = total_episodes or self.config.max_episodes
            steps = max_steps_per_episode or 100
            self.training_modes.simulate_integration_decisions(episodes, steps)
            
            self.stats['total_episodes'] = episodes
            self.stats['total_actions'] = episodes * steps
            return
        
        self.logger.info(f"Starting training in {self.config.mode} mode")
        
        try:
            self._training_active = True
            self.training_modes.run_training_mode(self.config.mode)
        except Exception as e:
            self.logger.error(f"Training error: {e}")
            raise
        finally:
            self._finalize_training()
    
    def start_training(self):
        """Start the training process."""
        if not self._is_pyboy_alive():
            self.logger.warning("PyBoy not available for training")
            return
        
        self.logger.info("Starting training session...")
        self._training_active = True
        self.training_modes.run_training_mode(self.config.mode)
    
    def get_current_stats(self) -> Dict[str, Any]:
        """Get current training statistics."""
        current_time = time.time()
        duration = current_time - self.stats['start_time']
        self.stats['session_duration'] = duration
        
        # Calculate actions per second
        if duration > 0:
            self.stats['actions_per_second'] = self.stats['total_actions'] / duration
        
        return self.stats.copy()
    
    def _is_pyboy_alive(self) -> bool:
        """Check if PyBoy instance is alive and responsive."""
        pyboy = self.pyboy_manager.get_pyboy()
        if pyboy is None:
            return False
        
        try:
            # Try to access PyBoy state
            _ = pyboy.tick
            return True
        except Exception:
            return False
    
    def _get_rule_based_action(self, step: int, skip_state_detection: bool = False) -> int:
        """Get rule-based action for the current game state."""
        try:
            # Use fallback actions for basic navigation
            return self._FALLBACK_ACTIONS[step % len(self._FALLBACK_ACTIONS)]
        except Exception as e:
            self.logger.warning(f"Error in rule-based action selection: {e}")
            return 5  # Default to A button
    
    def _execute_action(self, action: int):
        """Execute an action in the game."""
        pyboy = self.pyboy_manager.get_pyboy()
        if pyboy is None:
            return
            
        try:
            pyboy.send_input(action)
            for _ in range(self.config.frames_per_action):
                pyboy.tick()
        except Exception as e:
            self.logger.warning(f"Error executing action {action}: {e}")
    
    def _execute_synchronized_action(self, action: int):
        """Execute action in synchronized mode with monitoring."""
        self._execute_action(action)
    
    def _get_llm_action(self) -> Optional[int]:
        """Get action from LLM if available."""
        if self.llm_manager is None:
            return None
        # Future: Implement LLM action selection
        return self._get_rule_based_action(0)
    
    def _update_stats(self):
        """Update training statistics."""
        current_time = time.time()
        self.stats['last_update'] = current_time
        
        # Publish stats to data bus if available
        if self.data_bus:
            try:
                self.data_bus.publish(DataType.STATS, self.get_current_stats())
            except Exception as e:
                self.logger.debug(f"Failed to publish stats: {e}")
    
    def _capture_screenshot(self):
        """Capture screenshot for monitoring."""
        pyboy = self.pyboy_manager.get_pyboy()
        if pyboy is None:
            return
            
        try:
            screen = pyboy.screen.ndarray
            if not self.screenshot_queue.full():
                self.screenshot_queue.put(screen.copy(), block=False)
        except Exception as e:
            self.logger.debug(f"Failed to capture screenshot: {e}")
    
    def _finalize_training(self):
        """Clean up after training session."""
        self._training_active = False
        self.logger.info("Training session completed")
        
        # Cleanup managers
        self.pyboy_manager.cleanup()
        self.web_manager.cleanup()
    
    def set_mock_pyboy(self, mock_instance):
        """Set mock PyBoy instance for testing."""
        self.pyboy_manager.set_mock_instance(mock_instance)
        self.pyboy = mock_instance