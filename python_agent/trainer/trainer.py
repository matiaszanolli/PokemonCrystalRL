"""
Main trainer orchestrator for Pokemon Crystal RL Trainer
"""

import time
import threading
import queue
import io
import base64
import json
import numpy as np
from datetime import datetime
from typing import Optional, Dict, Any
from PIL import Image
import logging
from contextlib import contextmanager

# PyBoy imports
try:
    from pyboy import PyBoy
    PYBOY_AVAILABLE = True
except ImportError:
    PYBOY_AVAILABLE = False

# Vision processing
try:
    from ..vision.vision_processor import PokemonVisionProcessor, VisualContext
    VISION_AVAILABLE = True
except ImportError:
    try:
        from vision.vision_processor import PokemonVisionProcessor, VisualContext
        VISION_AVAILABLE = True
    except ImportError:
        VISION_AVAILABLE = False

from .config import TrainingConfig, TrainingMode, LLMBackend
from .game_state import GameStateDetector
from .llm_manager import LLMManager
from .web_server import TrainingWebServer
from .training_strategies import TrainingStrategyManager


class UnifiedPokemonTrainer:
    """Unified Pokemon Crystal training system - main orchestrator"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        
        # Initialize logging system
        self._setup_logging()
        
        # Core components
        self.pyboy = None
        self.game_state_detector = GameStateDetector(config.debug_mode)
        self.llm_manager = None
        self.strategy_manager = None
        self.web_server = None
        
        # Performance tracking
        self.stats = {
            'start_time': time.time(),
            'total_actions': 0,
            'total_episodes': 0,
            'actions_per_second': 0.0,
            'mode': config.mode.value,
            'model': config.llm_backend.value if config.llm_backend else "rule-based"
        }
        
        # Screen capture and monitoring
        self.screen_queue = queue.Queue(maxsize=30)
        self.latest_screen = None
        self.capture_thread = None
        self.capture_active = False
        
        # Initialize vision processor for OCR
        if VISION_AVAILABLE:
            self.vision_processor = PokemonVisionProcessor()
            self.logger.info("👁️ Vision processor initialized for OCR")
        else:
            self.vision_processor = None
            
        # Text detection history for web display
        self.recent_text = []
        self.text_frequency = {}
        
        # Training state
        self._training_active = False
        self.web_thread = None
        
        # Error tracking and recovery
        self.error_count = {
            'pyboy_crashes': 0, 
            'llm_failures': 0, 
            'capture_errors': 0, 
            'general': 0, 
            'total_errors': 0
        }
        self.last_error_time = None
        self.recovery_attempts = 0
        
        self._initialize_components()
    
    def _setup_logging(self):
        """Setup comprehensive logging system"""
        # Create logger
        self.logger = logging.getLogger('pokemon_trainer')
        self.logger.setLevel(getattr(logging, self.config.log_level.upper()))
        
        # Clear any existing handlers
        self.logger.handlers.clear()
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)8s | %(message)s',
            datefmt='%H:%M:%S'
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler for detailed logs
        try:
            file_handler = logging.FileHandler('pokemon_trainer.log')
            file_handler.setLevel(getattr(logging, self.config.log_level.upper()))
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        except Exception as e:
            print(f"⚠️ Could not setup file logging: {e}")
        
        self.logger.info("Logging system initialized")
    
    @contextmanager
    def _handle_errors(self, operation_name: str, error_type: str = 'general'):
        """Context manager for consistent error handling and recovery"""
        try:
            yield
        except KeyboardInterrupt:
            self.logger.info(f"Operation {operation_name} interrupted by user")
            raise
        except Exception as e:
            self.error_count[error_type] += 1
            self.error_count['total_errors'] += 1
            self.last_error_time = time.time()
            
            error_msg = f"Error in {operation_name}: {str(e)}"
            
            if self.config.debug_mode:
                self.logger.exception(error_msg)
            else:
                self.logger.error(error_msg)
            
            # Attempt recovery based on error type
            if error_type == 'pyboy_crashes' and self.error_count[error_type] < 5:
                self.logger.info(f"Attempting PyBoy recovery (attempt {self.recovery_attempts + 1})")
                self.recovery_attempts += 1
                if self._attempt_pyboy_recovery():
                    self.logger.info("Recovery successful")
                else:
                    self.logger.error("Recovery failed")
            
            # Re-raise for caller to handle
            raise
    
    def _initialize_components(self):
        """Initialize training components based on mode"""
        self.logger.info(f"🚀 Initializing {self.config.mode.value.title()} Training Mode")
        
        # Initialize PyBoy
        self._init_pyboy()
        
        # Initialize LLM manager
        if self.config.llm_backend and self.config.llm_backend != LLMBackend.NONE:
            self.llm_manager = LLMManager(self.config, self.game_state_detector)
        
        # Initialize strategy manager
        self.strategy_manager = TrainingStrategyManager(
            self.config, self.pyboy, self.llm_manager, self.game_state_detector
        )
        
        # Initialize web server
        if self.config.enable_web:
            self.web_server = TrainingWebServer(self.config, self)
        
        self.logger.info("✅ Trainer initialized successfully!")
    
    def _init_pyboy(self):
        """Initialize PyBoy for game emulation"""
        if not PYBOY_AVAILABLE:
            raise RuntimeError("PyBoy not available for training")
        
        self.pyboy = PyBoy(
            self.config.rom_path,
            window="null" if self.config.headless else "SDL2",
            debug=self.config.debug_mode
        )
        
        self.logger.info(f"✅ PyBoy initialized ({'headless' if self.config.headless else 'windowed'})")
    
    def start_training(self):
        """Start the training process"""
        self.logger.info(f"⚡ STARTING {self.config.mode.value.upper()} TRAINING")
        self.logger.info("=" * 60)
        self._print_config_summary()
        
        # Set training active flag
        self._training_active = True
        
        # Start screen capture if enabled
        if self.config.capture_screens:
            self._start_screen_capture()
        
        # Start web server if enabled
        if self.config.enable_web and self.web_server:
            server = self.web_server.start()
            self.web_thread = threading.Thread(target=server.serve_forever, daemon=True)
            self.web_thread.start()
        
        # Route to appropriate training method
        try:
            if self.config.mode == TrainingMode.FAST_MONITORED:
                self._run_fast_monitored_training()
            elif self.config.mode == TrainingMode.CURRICULUM:
                self._run_curriculum_training()
            elif self.config.mode == TrainingMode.ULTRA_FAST:
                self._run_ultra_fast_training()
            else:
                raise ValueError(f"Unknown training mode: {self.config.mode}")
        finally:
            self._finalize_training()
    
    def _run_fast_monitored_training(self):
        """Run unified fast training with comprehensive monitoring"""
        # Check if synchronized training is requested
        if self.config.capture_screens:
            return self._run_synchronized_training()
        else:
            return self._run_legacy_fast_training()
    
    def _run_synchronized_training(self):
        """Run frame-synchronized training with screenshot-decision-action cycles"""
        actions_taken = 0
        
        # Load save state if available
        self._load_save_state()
        
        frame_duration_ms = 1000.0 / 60.0  # Game Boy runs at 60 FPS
        action_duration_ms = self.config.frames_per_action * frame_duration_ms
        
        self.logger.info(f"🔄 Synchronized training: {self.config.frames_per_action} frames per action ({action_duration_ms:.1f}ms)")
        self.logger.info(f"⏱️ Expected speed: {1000.0 / action_duration_ms:.1f} actions/second")
        
        try:
            while actions_taken < self.config.max_actions:
                cycle_start = time.time()
                
                # Update step in LLM manager
                if self.llm_manager:
                    self.llm_manager.update_step(actions_taken)
                
                # 1. SCREENSHOT PHASE - Capture current game state
                screenshot = self._capture_synchronized_screenshot()
                
                # 2. DECISION PHASE - Make intelligent decision
                if self.llm_manager and self.llm_manager.should_use_llm(actions_taken):
                    try:
                        action = self.llm_manager.get_llm_action_with_vision(screenshot, actions_taken)
                    except Exception as e:
                        if self.config.debug_mode:
                            self.logger.warning(f"LLM call failed: {e}")
                        action = self._get_rule_based_action(actions_taken, screenshot)
                else:
                    # Use rule-based logic
                    action = self._get_rule_based_action(actions_taken, screenshot)
                
                # 3. OCR PHASE - Process text recognition
                if self.vision_processor and screenshot is not None and actions_taken % 5 == 0:
                    self._process_text_recognition(screenshot)
                
                # 4. ACTION EXECUTION PHASE - Execute for exact frame duration
                self.strategy_manager.execute_synchronized_action(action, self.config.frames_per_action)
                
                actions_taken += 1
                self.stats['total_actions'] = actions_taken
                
                # Progress monitoring
                if actions_taken % 50 == 0:
                    self._update_stats()
                    elapsed = time.time() - self.stats['start_time']
                    aps = actions_taken / elapsed
                    cycle_time = (time.time() - cycle_start) * 1000
                    self.logger.info(f"🔄 Step {actions_taken}: Action {action}, Cycle: {cycle_time:.0f}ms, Speed: {aps:.1f} a/s")
                    
                    # Detailed monitoring every 200 actions
                    if actions_taken % 200 == 0:
                        if self.llm_manager:
                            llm_calls = self.llm_manager.stats['llm_calls']
                            llm_ratio = llm_calls / actions_taken * 100 if actions_taken > 0 else 0
                            self.logger.info(f"📊 Progress: {actions_taken}/{self.config.max_actions} | 🤖 LLM: {llm_ratio:.1f}%")
                        
                        if self.config.enable_web and self.web_server:
                            self.logger.info(f"🌐 Monitor: http://{self.config.web_host}:{self.web_server.port}")
        
        except KeyboardInterrupt:
            self.logger.info("⏸️ Synchronized training interrupted")
    
    def _run_legacy_fast_training(self):
        """Legacy fast training method (no synchronization)"""
        actions_taken = 0
        last_action = 5  # Default to A button
        
        # Load save state if available
        self._load_save_state()
        
        self.logger.info("⚡ Legacy fast training: maximum speed without synchronization")
        
        try:
            while actions_taken < self.config.max_actions:
                # Advance game with PyBoy speed constraints
                if self.pyboy:
                    self.pyboy.tick()
                
                # Update step in LLM manager
                if self.llm_manager:
                    self.llm_manager.update_step(actions_taken)
                
                # Get action - LLM decisions at intervals for intelligence
                if self.llm_manager and self.llm_manager.should_use_llm(actions_taken):
                    try:
                        screenshot = self._simple_screenshot_capture()
                        action = self.llm_manager.get_llm_action(screenshot)
                        last_action = action
                    except Exception as e:
                        if self.config.debug_mode:
                            self.logger.warning(f"LLM call failed: {e}")
                        action = last_action  # Use previous action if LLM fails
                else:
                    # Reuse last action for speed
                    action = last_action
                
                # Execute action
                self.strategy_manager.execute_action(action)
                actions_taken += 1
                self.stats['total_actions'] = actions_taken
                
                # Progress monitoring
                if actions_taken % 100 == 0:
                    self._update_stats()
                    elapsed = time.time() - self.stats['start_time']
                    aps = actions_taken / elapsed
                    self.logger.info(f"📊 Progress: {actions_taken}/{self.config.max_actions} ({aps:.1f} a/s)")
                    
                    # Additional monitoring info every 500 actions
                    if actions_taken % 500 == 0:
                        if self.llm_manager:
                            llm_calls = self.llm_manager.stats['llm_calls']
                            llm_ratio = llm_calls / actions_taken * 100 if actions_taken > 0 else 0
                            self.logger.info(f"🤖 LLM decisions: {llm_calls} ({llm_ratio:.1f}% of actions)")
                        
                        if self.config.enable_web and self.web_server:
                            self.logger.info(f"🌐 Web monitor: http://{self.config.web_host}:{self.web_server.port}")
                
                # Optimal delay for PyBoy stability
                time.sleep(0.008)  # ~125 actions/second max
        
        except KeyboardInterrupt:
            self.logger.info("⏸️ Training interrupted")
    
    def _run_curriculum_training(self):
        """Run progressive curriculum training"""
        result = self.strategy_manager.run_curriculum_training(self.config.max_episodes)
        self.stats.update(result)
        self.logger.info(f"📚 Curriculum training completed: {result}")
    
    def _run_ultra_fast_training(self):
        """Run rule-based ultra-fast training"""
        result = self.strategy_manager.run_ultra_fast_training(self.config.max_actions)
        self.stats.update(result)
        self.logger.info(f"🚀 Ultra-fast training completed: {result}")
    
    def _get_rule_based_action(self, step: int, screenshot: Optional[np.ndarray] = None) -> int:
        """Get rule-based action with state detection"""
        current_state = self.game_state_detector.get_current_state(screenshot, step)
        
        # Import here to avoid circular import
        from .training_strategies import get_rule_based_action
        from .game_state import get_unstuck_action
        
        # Check if stuck
        if self.game_state_detector.is_stuck():
            stuck_info = self.game_state_detector.get_stuck_info()
            return get_unstuck_action(step, stuck_info['stuck_counter'])
        else:
            return get_rule_based_action(current_state, step)
    
    def _load_save_state(self):
        """Load save state if available"""
        if self.config.save_state_path and self.pyboy:
            try:
                with open(self.config.save_state_path, 'rb') as f:
                    self.pyboy.load_state(f)
                self.logger.info(f"💾 Loaded save state: {self.config.save_state_path}")
            except Exception as e:
                self.logger.warning(f"⚠️ Could not load save state: {e}")
    
    def _capture_synchronized_screenshot(self) -> Optional[np.ndarray]:
        """Capture screenshot synchronously for decision making"""
        try:
            if not self.pyboy:
                return None
            
            # Lightweight PyBoy health check
            try:
                _ = self.pyboy.frame_count
            except Exception:
                self.logger.warning("🩹 PyBoy health check failed, attempting recovery...")
                self.error_count['total_errors'] += 1
                if self._attempt_pyboy_recovery():
                    return self._simple_screenshot_capture()
                return None
            
            # Get current screen data
            screen_array = self.pyboy.screen.ndarray
            if screen_array is None:
                return None
            
            # Convert to standard format (RGB)
            return self._convert_screen_format(screen_array)
            
        except Exception as e:
            self.error_count['total_errors'] += 1
            if self.config.debug_mode:
                self.logger.debug(f"Screenshot capture failed: {e}")
            
            if self.error_count['total_errors'] > 5:
                self._attempt_pyboy_recovery()
            
            return None
    
    def _simple_screenshot_capture(self) -> Optional[np.ndarray]:
        """Simple screenshot capture without crash detection"""
        try:
            if not self.pyboy:
                return None
            
            screen_array = self.pyboy.screen.ndarray
            if screen_array is None:
                return None
            
            return self._convert_screen_format(screen_array)
            
        except Exception:
            return None
    
    def _convert_screen_format(self, screen_array: np.ndarray) -> Optional[np.ndarray]:
        """Convert screen array to standard RGB format"""
        try:
            if len(screen_array.shape) == 3 and screen_array.shape[-1] == 4:
                # RGBA to RGB
                return screen_array[:, :, :3].astype(np.uint8).copy()
            elif len(screen_array.shape) == 3 and screen_array.shape[-1] == 3:
                # Already RGB
                return screen_array.astype(np.uint8).copy()
            elif len(screen_array.shape) == 2:
                # Grayscale, convert to RGB
                return np.stack([screen_array, screen_array, screen_array], axis=2).astype(np.uint8)
            else:
                return None
        except Exception:
            return None
    
    def _attempt_pyboy_recovery(self) -> bool:
        """Attempt to recover from PyBoy crash"""
        if self.config.debug_mode:
            self.logger.info("🩹 Attempting PyBoy recovery...")
        
        try:
            # Clean up current instance
            if self.pyboy:
                try:
                    self.pyboy.stop()
                except:
                    pass
                
                time.sleep(0.2)
                self.pyboy = None
            
            # Reinitialize PyBoy
            self.pyboy = PyBoy(
                self.config.rom_path,
                window="null" if self.config.headless else "SDL2",
                debug=False  # Always disable debug on recovery
            )
            
            # Update strategy manager
            if self.strategy_manager:
                self.strategy_manager.pyboy = self.pyboy
            
            if self.config.debug_mode:
                self.logger.info("✅ PyBoy recovery successful")
            
            # Attempt to reload save state
            self._load_save_state()
            
            return True
                
        except Exception as e:
            if self.config.debug_mode:
                self.logger.error(f"💀 PyBoy recovery failed: {e}")
            self.pyboy = None
            return False
    
    def _start_screen_capture(self):
        """Start screen capture thread"""
        self.capture_active = True
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        self.logger.info("📸 Screen capture started")
    
    def _capture_loop(self):
        """Screen capture loop for web monitoring"""
        consecutive_errors = 0
        max_consecutive_errors = 10
        
        self.logger.info("🔍 Starting screen capture loop...")
        
        while self.capture_active:
            try:
                screenshot = self._simple_screenshot_capture()
                if screenshot is not None:
                    self._process_and_queue_screenshot(screenshot)
                    consecutive_errors = 0
                else:
                    consecutive_errors += 1
                
                # Frame rate limiting with error backoff
                if consecutive_errors > 0:
                    time.sleep(0.5)  # Slower capture when having issues
                else:
                    base_delay = 1.0 / self.config.capture_fps
                    time.sleep(base_delay)
            
            except Exception as e:
                consecutive_errors += 1
                if consecutive_errors <= 3:
                    self.logger.warning(f"⚠️ Screen capture loop error: {e}")
                
                if consecutive_errors >= max_consecutive_errors:
                    self.logger.error(f"❌ Too many screen capture errors ({consecutive_errors}), stopping capture")
                    self.capture_active = False
                    break
                
                time.sleep(0.5)
        
        self.logger.info("🔍 Screen capture loop ended")
    
    def _process_and_queue_screenshot(self, screenshot: np.ndarray):
        """Process screenshot for web display and queue it"""
        try:
            # Resize for web display
            h, w = screenshot.shape[:2]
            target_h, target_w = self.config.screen_resize[1], self.config.screen_resize[0]
            
            # Simple resize
            screen_pil = Image.fromarray(screenshot.astype(np.uint8))
            screen_resized = screen_pil.resize((target_w, target_h), Image.NEAREST)
            
            # Convert to base64
            buffer = io.BytesIO()
            screen_resized.save(buffer, format='PNG', optimize=True, compress_level=6)
            screen_b64 = base64.b64encode(buffer.getvalue()).decode()
            
            # Update latest screen
            screen_data = {
                'image_b64': screen_b64,
                'timestamp': time.time(),
                'size': screen_resized.size,
                'frame_id': int(time.time() * 1000),
                'data_length': len(screen_b64)
            }
            
            self.latest_screen = screen_data
            
            # Add to queue
            try:
                if self.screen_queue.full():
                    try:
                        self.screen_queue.get_nowait()  # Remove oldest
                    except queue.Empty:
                        pass
                self.screen_queue.put_nowait(screen_data)
            except (queue.Full, queue.Empty):
                pass  # Non-critical
                
        except Exception as e:
            if self.config.debug_mode:
                self.logger.debug(f"Screen processing failed: {e}")
    
    def _process_text_recognition(self, screenshot: np.ndarray):
        """Process OCR text recognition and update web display data"""
        if not self.vision_processor:
            return
        
        try:
            visual_context = self.vision_processor.process_screenshot(screenshot)
            
            for detected_text in visual_context.detected_text:
                if detected_text.text and len(detected_text.text.strip()) > 0:
                    text_entry = {
                        'text': detected_text.text.strip(),
                        'location': detected_text.location,
                        'confidence': detected_text.confidence,
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    self.recent_text.append(text_entry)
                    
                    # Update frequency tracking
                    text_clean = detected_text.text.strip().lower()
                    if text_clean:
                        self.text_frequency[text_clean] = self.text_frequency.get(text_clean, 0) + 1
            
            # Keep only recent text (last 50 entries)
            if len(self.recent_text) > 50:
                self.recent_text = self.recent_text[-25:]  # Keep last 25
            
            # Log interesting findings
            if visual_context.detected_text and self.config.debug_mode:
                text_summary = ', '.join([t.text[:20] for t in visual_context.detected_text[:3]])
                self.logger.debug(f"OCR: {visual_context.screen_type} - {text_summary}")
                
        except Exception as e:
            if self.config.debug_mode:
                self.logger.warning(f"Text recognition failed: {e}")
    
    def _update_stats(self):
        """Update performance statistics"""
        elapsed = time.time() - self.stats['start_time']
        self.stats['actions_per_second'] = self.stats['total_actions'] / max(elapsed, 0.001)
        
        # Merge LLM stats if available
        if self.llm_manager:
            self.stats.update(self.llm_manager.stats)
    
    def get_current_stats(self) -> Dict[str, Any]:
        """Get current statistics for web API"""
        self._update_stats()
        return self.stats.copy()
    
    def _finalize_training(self):
        """Cleanup and final statistics"""
        # Clear training active flag
        self._training_active = False
        
        # Stop capture
        if self.capture_active:
            self.capture_active = False
            if self.capture_thread:
                self.capture_thread.join(timeout=1)
        
        # Update final stats
        self._update_stats()
        
        # Print summary
        elapsed = time.time() - self.stats['start_time']
        self.logger.info("📊 TRAINING SUMMARY")
        self.logger.info("=" * 40)
        self.logger.info(f"⏱️ Duration: {elapsed:.1f} seconds")
        self.logger.info(f"🎯 Total actions: {self.stats['total_actions']}")
        self.logger.info(f"📈 Episodes: {self.stats['total_episodes']}")
        self.logger.info(f"🚀 Speed: {self.stats['actions_per_second']:.1f} actions/sec")
        if self.llm_manager:
            self.logger.info(f"🧠 LLM calls: {self.llm_manager.stats['llm_calls']}")
        
        # Save stats if enabled
        if self.config.save_stats:
            self.stats['end_time'] = time.time()
            with open(self.config.stats_file, 'w') as f:
                json.dump(self.stats, f, indent=2)
            self.logger.info(f"💾 Stats saved to {self.config.stats_file}")
        
        # Cleanup
        if self.pyboy:
            self.pyboy.stop()
        
        if self.web_server:
            self.web_server.stop()
        
        self.logger.info("🛑 Training completed and cleaned up")
    
    def _print_config_summary(self):
        """Print training configuration summary"""
        self.logger.info(f"🎮 ROM: {self.config.rom_path}")
        self.logger.info(f"🤖 LLM: {self.config.llm_backend.value if self.config.llm_backend else 'None (rule-based)'}")
        self.logger.info(f"🎯 Target: {self.config.max_actions} actions / {self.config.max_episodes} episodes")
        self.logger.info(f"📸 Capture: {'ON' if self.config.capture_screens else 'OFF'}")
        self.logger.info(f"🌐 Web UI: {'ON' if self.config.enable_web else 'OFF'}")
        self.logger.info("")
