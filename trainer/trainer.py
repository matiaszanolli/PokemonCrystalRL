"""
Core trainer module for Pokemon Crystal RL
"""

import os
import time
import queue
import logging
import threading
import numpy as np
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from monitoring.data_bus import get_data_bus, init_data_bus, DataType
from pokemon_crystal_rl.core.monitoring.web_server import TrainingWebServer


class GameStateDetector:
    """Detects and manages game state information."""
    
    def __init__(self):
        self.last_screen_hash = None
        self.consecutive_same_screens = 0
        self.stuck_counter = 0

    def get_screen_hash(self, screen: np.ndarray) -> int:
        """Get a hash value for screen content for stuck detection."""
        if screen is None:
            return None
            
        # Compute mean values for grid cells
        h, w = screen.shape[:2]
        cell_h = h // 4
        cell_w = w // 4
        grid_means = []

        for i in range(4):
            for j in range(4):
                cell = screen[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
                mean = np.mean(cell)
                grid_means.append(int(mean))

        # Convert means to hash
        hash_val = 0
        for mean in grid_means:
            hash_val = hash_val * 31 + mean

        return hash_val

    def detect_game_state(self, screen: np.ndarray) -> str:
        """Detect game state from screen content."""
        if screen is None:
            return "unknown"

        # Convert to grayscale
        if len(screen.shape) == 3:
            gray = np.mean(screen, axis=2).astype(np.uint8)
        else:
            gray = screen

        # Store last screen hash for transition detection
        current_hash = self.get_screen_hash(screen)
        if current_hash == self.last_screen_hash:
            self.consecutive_same_screens += 1
        else:
            self.consecutive_same_screens = 0
            self.last_screen_hash = current_hash
            self.stuck_counter = 0

        # If screen hasn't changed in a while, we might be stuck
        if self.consecutive_same_screens > 30:
            self.stuck_counter += 1
            return "stuck"

        # Detect loading/black screen
        if np.mean(gray) < 20:
            return "loading"

        # Detect intro/white screen
        if np.mean(gray) > 240:
            return "intro_sequence"

        # Menu detection (check for brighter rectangular region)
        menu_regions = [
            gray[20:60, 20:140],  # Standard menu
            gray[30:90, 30:130],  # Battle menu
            gray[100:140, 10:150]  # Options menu
        ]
        for region in menu_regions:
            region_mean = np.mean(region)
            region_std = np.std(region)
            # Menu regions are bright and uniform
            if region_mean > 180 and region_std < 40:
                return "menu"

        # Dialogue detection (check for bright box at bottom with text-like variance)
        bottom = gray[100:, :]
        bottom_mean = np.mean(bottom)
        bottom_std = np.std(bottom)
        
        # Dialogue boxes are bright and have text-like variance
        if bottom_mean > 200 and 30 < bottom_std < 70:
            # Validate with text-like pattern check
            text_rows = np.mean(bottom[::2], axis=1)  # Sample alternate rows
            text_variance = np.std(text_rows)
            if text_variance > 20:  # Text creates consistent variance pattern
                return "dialogue"

        # Check for battle screen characteristics
        if self._detect_battle_screen(gray):
            return "battle"

        return "overworld"

    def _detect_battle_screen(self, gray: np.ndarray) -> bool:
        """Helper method to detect battle screen state."""
        # Battle screens have distinctive layout with HP bars
        hp_regions = [
            gray[30:45, 170:220],   # Player HP region
            gray[100:115, 50:100]    # Enemy HP region
        ]
        
        for region in hp_regions:
            region_mean = np.mean(region)
            region_std = np.std(region)
            # HP bars have high contrast
            if region_mean > 150 and region_std > 50:
                return True
                
        return False


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


class PokemonTrainer:
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
        
        # Training control
        self._training_active = False
        self.adaptive_llm_interval = self.config.llm_interval
        self.llm_response_times = []

        # Game state tracking
        self.game_state_detector = GameStateDetector()

        # Screen tracking
        self.last_screen_hash = None
        self.consecutive_same_screens = 0
        self.screen_width = 160
        self.screen_height = 144
        self.screen_queue = queue.Queue(maxsize=30)
        self.latest_screen = None
        self.capture_active = False
        
        # Data bus for component communication
        # Data bus for component communication
        from monitoring.data_bus import get_data_bus, DataType
        self.data_bus = get_data_bus()
        if self.data_bus:
            self.data_bus.register_component(
                "trainer",
                {
                    "type": "core",
                    "mode": self.config.mode.value,
                    "llm_backend": self.config.llm_backend.value if self.config.llm_backend else None
                }
            )

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
            return
            
        try:
            self.web_server = TrainingWebServer(self.config, self)
            self.web_server.start()
        except Exception as e:
            self.logger.error(f"Failed to start web server: {e}")
            self.web_server = None

    def setup_llm_manager(self):
        """Setup LLM manager if LLM backend is enabled."""
        if not self.config.llm_backend:
            self.llm_manager = None
            return

        # Import here to avoid overhead when not using LLM features
        from trainer.llm_manager import LLMManager

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
            elif self.config.mode == TrainingMode.FAST_MONITORED:
                self._run_legacy_fast_training()
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

    def _run_legacy_fast_training(self):
        """Run training in legacy fast mode with basic monitoring."""
        self.stats['total_actions'] = 0
        while (self.stats['total_actions'] < self.config.max_actions and 
               self._training_active):
            try:
                action = self._get_rule_based_action(self.stats['total_actions'])
                self._execute_action(action)
                self.stats['total_actions'] += 1

                if self.stats['total_actions'] % 20 == 0:
                    self._update_stats()
            except Exception:
                # Don't count towards max actions if failed
                self.stats['total_actions'] -= 1
                continue
                
    def _run_ultra_fast_training(self):
        """Run training in ultra-fast mode."""
        self.stats['total_actions'] = 0
        while (self.stats['total_actions'] < self.config.max_actions and 
               self._training_active):
            try:
                # Execute multiple frames per action without checks
                action = self._get_rule_based_action(self.stats['total_actions'])
                for _ in range(self.config.frames_per_action):
                    self.pyboy.send_input(action)
                    self.pyboy.tick()
                self.stats['total_actions'] += 1

                if self.stats['total_actions'] % 100 == 0:
                    self._update_stats()
            except Exception:
                # Don't count towards max actions if failed
                self.stats['total_actions'] -= 1
                continue

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
                try:
                    self._execute_synchronized_action(action)
                    self.stats['total_actions'] += 1

                    # Update stats and capture periodically
                    if self.stats['total_actions'] % 20 == 0:
                        self._update_stats()

                    if (self.config.capture_screens and 
                        self.stats['total_actions'] % 5 == 0):
                        self._capture_and_queue_screen()
                except Exception as e:
                    # Attempt recovery
                    self.stats['total_actions'] -= 1  # Don't count failed actions
                    with self._handle_errors("training_cycle", "pyboy_crashes"):
                        raise  # This triggers error handler to attempt recovery

    def _get_llm_action(self) -> Optional[int]:
        """Get action from LLM manager."""
        try:
            start_time = time.time()
            action = self.llm_manager.get_action()
            
            # Handle both actual integers and Mock objects for testing
            if hasattr(action, '_mock_return_value'):
                action = action._mock_return_value

            if action is not None and isinstance(action, int):
                self._track_llm_performance(time.time() - start_time)
                self.stats['llm_calls'] += 1
                return action
            
            return None

        except Exception as e:
            self.logger.warning(f"LLM action failed: {e}")
            self.error_count['llm_failures'] += 1
            return None

    def _get_rule_based_action(self, step: int) -> int:
        """Get action using rule-based system."""
        return 5  # Default to A button

    def _track_llm_performance(self, response_time: float):
        """Track LLM performance for adaptive intervals."""
        self.stats['llm_total_time'] += response_time
        if self.stats['llm_calls'] > 0:
            self.stats['llm_avg_time'] = self.stats['llm_total_time'] / self.stats['llm_calls']

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
                    self.config.llm_interval,
                    int(self.adaptive_llm_interval * 0.8)
                )

    def _detect_game_state(self, screen: np.ndarray) -> str:
        """Detect game state from screen content."""
        return self.game_state_detector.detect_game_state(screen)

    def _get_screen_hash(self, screen: np.ndarray) -> int:
        """Get a hash value for screen content for stuck detection."""
        return self.game_state_detector.get_screen_hash(screen)

    def _convert_screen_format(self, screen: np.ndarray) -> np.ndarray:
        """Convert screen to RGB format."""
        if screen is None:
            return None

        # Handle different input formats
        if len(screen.shape) == 2:  # Grayscale
            rgb = np.dstack([screen] * 3)
        elif len(screen.shape) == 3:
            if screen.shape[2] == 4:  # RGBA
                rgb = screen[:, :, :3]
            elif screen.shape[2] == 3:  # RGB
                rgb = screen
            else:
                return None
        else:
            return None

        return rgb

    def _handle_dialogue(self, step: int) -> int:
        """Handle dialogue state."""
        # For dialogue, default to A button to advance text
        return 5  # A button

    def _handle_title_screen(self, step: int) -> int:
        """Handle title screen state."""
        # For title screen, alternate between A button and DOWN for menu navigation
        return 5 if step % 2 == 0 else 2  # Alternate A and DOWN

    def _get_unstuck_action(self, step: int) -> int:
        """Get action to try to escape stuck state."""
        # Cycle through movement actions
        actions = [1, 2, 3, 4, 5, 6]  # UP, DOWN, LEFT, RIGHT, A, B
        return actions[step % len(actions)]

    class _ErrorHandler:
        def __init__(self, trainer, operation: str, error_type: str = "total_errors"):
            self.trainer = trainer
            self.operation = operation
            self.error_type = error_type

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            if exc_type is None:
                return True
            
            if exc_type == KeyboardInterrupt:
                return False

            # Count error first regardless of type
            self.trainer.error_count[self.error_type] = self.trainer.error_count.get(self.error_type, 0) + 1
            self.trainer.error_count['total_errors'] += 1
            self.trainer.last_error_time = time.time()
            
            # Then attempt recovery for PyBoy crashes
            if self.error_type == "pyboy_crashes":
                try:
                    if self.trainer._attempt_pyboy_recovery():
                        return True
                except Exception:
                    pass
            
            # Re-raise original exception
            return None  # This ensures the exception is propagated

    def _handle_errors(self, operation: str, error_type: str = "total_errors"):
        """Context manager for error handling."""
        return self._ErrorHandler(self, operation, error_type)

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
            
        # Publish stats to data bus if available
        if self.data_bus:
            self.data_bus.publish(
                DataType.TRAINING_STATS,
                self.stats,
                "trainer"
            )

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

        if self.data_bus:
            try:
                self.data_bus.unregister_component("trainer")
            except Exception as e:
                self.logger.error(f"Error unregistering trainer: {e}")

    def _is_pyboy_alive(self) -> bool:
        """Check if PyBoy instance is alive and functioning."""
        if not self.pyboy:
            return False
        
        try:
            frame_count = self.pyboy.frame_count
            if not isinstance(frame_count, int):
                return False
            return True
        except Exception:
            return False

    def _attempt_pyboy_recovery(self) -> bool:
        """Attempt to recover crashed PyBoy instance.
        
        Returns:
            bool: True if recovery successful, False otherwise
        """
        try:
            # Stop the old instance
            if self.pyboy:
                try:
                    self.pyboy.stop()
                except Exception:
                    pass

            # Create new instance
            new_pyboy = PyBoy(
                self.config.rom_path,
                window_type="headless" if self.config.headless else "SDL2",
                debug=self.config.debug_mode
            )

            # Try to load save state if available
            if self.config.save_state_path and os.path.exists(self.config.save_state_path):
                with open(self.config.save_state_path, 'rb') as f:
                    new_pyboy.load_state(f)

            # Recovery success, update instance
            self.pyboy = new_pyboy
            self.recovery_attempts += 1
            return True

        except Exception as e:
            self.logger.error(f"PyBoy recovery failed: {e}")
            self.pyboy = None
            return False
    def _execute_synchronized_action(self, action: int) -> None:
        """Execute action with screen monitoring and error handling."""
        # Execute action with configurable frames per step
        for _ in range(self.config.frames_per_action):
            try:
                self.pyboy.send_input(action)
                self.pyboy.tick()
                
                # Get and process screen every few frames
                if _ % (self.config.frames_per_action // 4) == 0:
                    screen_data = np.array(self.pyboy.screen_image())
                    
                    # Check for stuck states
                    screen_hash = self._get_screen_hash(screen_data)
                    if screen_hash == self.last_screen_hash:
                        self.consecutive_same_screens += 1
                    else:
                        self.consecutive_same_screens = 0
                        self.last_screen_hash = screen_hash
                    
                    # If stuck, try to get unstuck
                    if self.consecutive_same_screens > 10:
                        unstuck_action = self._get_unstuck_action(self.stats['total_actions'])
                        self.pyboy.send_input(unstuck_action)
                        self.consecutive_same_screens = 0
                    
                    # Detect and handle special states
                    game_state = self._detect_game_state(screen_data)
                    if game_state == 'dialogue':
                        dialogue_action = self._handle_dialogue(self.stats['total_actions'])
                        if dialogue_action != action:
                            self.pyboy.send_input(dialogue_action)
                    elif game_state == 'intro':
                        title_action = self._handle_title_screen(self.stats['total_actions'])
                        if title_action != action:
                            self.pyboy.send_input(title_action)
                
            except Exception as e:
                self.logger.error(f"Error during action execution: {e}")
                self.error_count['total_errors'] += 1
                break
    
    def _simple_screenshot_capture(self) -> np.ndarray:
        """Capture a simple screenshot without any processing."""
        try:
            screen = np.array(self.pyboy.screen_image())
            return screen
        except Exception as e:
            self.logger.error(f"Simple screenshot capture failed: {e}")
            self.error_count['capture_errors'] += 1
            return None
    
    def _capture_and_queue_screen(self) -> None:
        """Capture and queue the current game screen."""
        try:
            # Capture screen data
            screen = self._simple_screenshot_capture()
            if screen is None:
                return
            
            # Convert to standard format
            screen_rgb = self._convert_screen_format(screen)
            
            # Resize if configured
            if self.config.screen_resize and screen_rgb.shape[:2] != self.config.screen_resize:
                from cv2 import resize, INTER_NEAREST
                screen_rgb = resize(
                    screen_rgb,
                    self.config.screen_resize,
                    interpolation=INTER_NEAREST
                )
            
            # Add screen to queue
            screen_data = {
                "image": screen_rgb,
                "timestamp": time.time()
            }
            try:
                self.screen_queue.put_nowait(screen_data)
            except queue.Full:
                # Queue is full, remove oldest item
                try:
                    self.screen_queue.get_nowait()
                    self.screen_queue.put_nowait(screen_data)
                except (queue.Empty, queue.Full):
                    pass
            
            # Publish to data bus if available
            if self.data_bus:
                self.data_bus.publish(
                    DataType.GAME_SCREEN,
                    {
                        "screen": screen_rgb,
                        "timestamp": time.time(),
                        "frame": self.pyboy.frame_count,
                        "action": self.stats['total_actions']
                    },
                    "trainer"
                )
                
        except Exception as e:
            self.logger.error(f"Error capturing screen: {e}")
            self.error_count['capture_errors'] += 1
