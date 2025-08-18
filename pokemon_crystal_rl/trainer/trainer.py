"""
Core trainer module for Pokemon Crystal RL
"""

import os
import time
import logging
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

try:
    from pyboy import PyBoy
    PYBOY_AVAILABLE = True
except ImportError:
    PYBOY_AVAILABLE = False
    PyBoy = None


class TrainingMode(Enum):
    FAST_MONITORED = "fast_monitored"
    ULTRA_FAST = "ultra_fast"
    CURRICULUM = "curriculum"
    CUSTOM = "custom"


class LLMBackend(Enum):
    NONE = None
    SMOLLM2 = "smollm2:1.7b"
    LLAMA32_1B = "llama3.2:1b"
    LLAMA32_3B = "llama3.2:3b"
    QWEN25_3B = "qwen2.5:3b"


@dataclass
class TrainingConfig:
    """Configuration for the Pokemon Crystal trainer."""
    rom_path: str
    mode: TrainingMode = TrainingMode.FAST_MONITORED
    llm_backend: Optional[LLMBackend] = LLMBackend.SMOLLM2
    max_actions: int = 1000
    max_episodes: int = 10
    llm_interval: int = 10
    frames_per_action: int = 24
    headless: bool = True
    debug_mode: bool = False
    save_state_path: Optional[str] = None
    enable_web: bool = False
    web_port: int = 8080
    web_host: str = "localhost"
    capture_screens: bool = True
    capture_fps: int = 10
    screen_resize: tuple = (320, 288)
    save_stats: bool = True
    stats_file: str = "training_stats.json"
    log_level: str = "INFO"
    curriculum_stages: int = 5


class UnifiedPokemonTrainer:
    """Unified Pokemon Crystal Trainer class."""

    def __init__(self, config: TrainingConfig):
        """Initialize the trainer with configuration."""
        if not PYBOY_AVAILABLE:
            raise RuntimeError("PyBoy not available - install with: pip install pyboy")

        self.config = config
        self.setup_logging()
        self.init_error_tracking()
        self.setup_pyboy()
        self.setup_web_server()
        self.setup_llm_manager()
        self.stats = self.init_stats()

    def setup_logging(self):
        """Setup logging system."""
        self.logger = logging.getLogger("pokemon_trainer")
        self.logger.setLevel(getattr(logging, self.config.log_level))

        # Add console handler
        if not self.logger.handlers:
            console = logging.StreamHandler()
            console.setFormatter(logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s'
            ))
            self.logger.addHandler(console)

    def init_error_tracking(self):
        """Initialize error tracking system."""
        self.error_count = {
            'pyboy_crashes': 0,
            'llm_failures': 0,
            'capture_errors': 0,
            'total_errors': 0
        }
        self.last_error_time = None
        self.recovery_attempts = 0

    def setup_pyboy(self):
        """Setup PyBoy emulator."""
        try:
            self.pyboy = PyBoy(
                self.config.rom_path,
                window_type="headless" if self.config.headless else "SDL2",
                debug=self.config.debug_mode
            )
            
            # Load save state if provided
            if self.config.save_state_path and os.path.exists(self.config.save_state_path):
                with open(self.config.save_state_path, 'rb') as f:
                    self.pyboy.load_state(f)

        except Exception as e:
            self.logger.error(f"Failed to initialize PyBoy: {e}")
            raise

    def setup_web_server(self):
        """Setup web monitoring server if enabled."""
        if not self.config.enable_web:
            self.web_server = None
            self.web_thread = None
            return

        # Import here to avoid overhead when not using web features
        from pokemon_crystal_rl.monitoring.web_interface import TrainingWebServer
        from threading import Thread

        try:
            self.web_server = TrainingWebServer(self)
            self.web_thread = Thread(target=self.web_server.serve_forever)
            self.web_thread.daemon = True
            self.web_thread.start()
        except Exception as e:
            self.logger.error(f"Failed to start web server: {e}")
            self.web_server = None
            self.web_thread = None

    def setup_llm_manager(self):
        """Setup LLM manager if LLM backend is enabled."""
        if not self.config.llm_backend:
            self.llm_manager = None
            return

        # Import here to avoid overhead when not using LLM features
        from pokemon_crystal_rl.local_llm_agent import LLMManager

        try:
            self.llm_manager = LLMManager(
                model=self.config.llm_backend.value,
                interval=self.config.llm_interval
            )
        except Exception as e:
            self.logger.error(f"Failed to initialize LLM manager: {e}")
            self.llm_manager = None

    def init_stats(self) -> Dict[str, Any]:
        """Initialize statistics tracking."""
        return {
            'mode': self.config.mode.value,
            'model': self.config.llm_backend.value if self.config.llm_backend else None,
            'start_time': time.time(),
            'total_actions': 0,
            'llm_calls': 0,
            'llm_total_time': 0,
            'llm_avg_time': 0,
            'actions_per_second': 0,
        }

    def start_training(self):
        """Start the training process."""
        if not self._is_pyboy_alive():
            raise RuntimeError("PyBoy not initialized or crashed")

        self._training_active = True
        
        try:
            if self.config.mode == TrainingMode.ULTRA_FAST:
                self._run_ultra_fast_training()
            else:
                self._run_synchronized_training()
        finally:
            self._training_active = False
            self._finalize_training()

    def get_current_stats(self) -> Dict[str, Any]:
        """Get current training statistics."""
        if self._training_active:
            self._update_stats()
        return self.stats

    def _run_ultra_fast_training(self):
        """Run training in ultra-fast mode."""
        while (self.stats['total_actions'] < self.config.max_actions and 
               self._training_active):
            action = self._get_rule_based_action(self.stats['total_actions'])
            self._execute_action(action)
            self.stats['total_actions'] += 1

            if self.stats['total_actions'] % 100 == 0:
                self._update_stats()

    def _run_synchronized_training(self):
        """Run training in synchronized mode with monitoring."""
        while (self.stats['total_actions'] < self.config.max_actions and 
               self._training_active):
            
            # Get action (LLM or rule-based)
            if (self.llm_manager and 
                self.stats['total_actions'] % self.config.llm_interval == 0):
                action = self._get_llm_action()
            else:
                action = self._get_rule_based_action(self.stats['total_actions'])

            if action:
                self._execute_synchronized_action(action)
                self.stats['total_actions'] += 1

            # Update stats and capture periodically
            if self.stats['total_actions'] % 20 == 0:
                self._update_stats()

            if (self.config.capture_screens and 
                self.stats['total_actions'] % 5 == 0):
                self._capture_and_queue_screen()

    def _get_llm_action(self) -> Optional[int]:
        """Get action from LLM manager."""
        try:
            start_time = time.time()
            action = self.llm_manager.get_action()
            
            if action is not None:
                self._track_llm_performance(time.time() - start_time)
                self.stats['llm_calls'] += 1
            
            return action

        except Exception as e:
            self.logger.warning(f"LLM action failed: {e}")
            self.error_count['llm_failures'] += 1
            return None

    def _get_rule_based_action(self, step: int) -> int:
        """Get action using rule-based system."""
        return 5  # Default to A button

    def _execute_action(self, action: int):
        """Execute action on PyBoy instance."""
        self.pyboy.send_input(action)
        self.pyboy.tick()

    def _update_stats(self):
        """Update training statistics."""
        current_time = time.time()
        elapsed = current_time - self.stats['start_time']
        
        if elapsed > 0:
            self.stats['actions_per_second'] = self.stats['total_actions'] / elapsed

    def _finalize_training(self):
        """Clean up resources after training."""
        if self.pyboy:
            try:
                self.pyboy.stop()
            except Exception as e:
                self.logger.error(f"Error stopping PyBoy: {e}")

        if self.web_server:
            try:
                self.web_server.shutdown()
            except Exception as e:
                self.logger.error(f"Error stopping web server: {e}")

    def _is_pyboy_alive(self) -> bool:
        """Check if PyBoy instance is alive and functioning."""
        if not self.pyboy:
            return False
        
        try:
            _ = self.pyboy.frame_count
            return True
        except Exception:
            return False
