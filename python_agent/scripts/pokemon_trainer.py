#!/usr/bin/env python3
"""
pokemon_trainer.py - Unified Pokemon Crystal RL Training System

One script to rule them all! Consolidates all training modes:
- Fast local training with real-time capture
- Curriculum-based progressive learning  
- Rule-based ultra-fast training
- Web monitoring and visualization
- Multiple LLM model support
- Flexible configuration options
"""

import time
import numpy as np
import threading
import queue
import io
import base64
import json
import sqlite3
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from enum import Enum
from dataclasses import dataclass
from PIL import Image
import ollama

# System monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# Core imports
try:
    from pyboy import PyBoy
    from pyboy.utils import WindowEvent
    PYBOY_AVAILABLE = True
except ImportError:
    PYBOY_AVAILABLE = False
    print("⚠️ PyBoy not available")

# Optional imports for different modes
try:
    from ..core.pyboy_env import PyBoyPokemonCrystalEnv
    PYBOY_ENV_AVAILABLE = True
except ImportError:
    PYBOY_ENV_AVAILABLE = False

try:
    from ..agents.enhanced_llm_agent import EnhancedLLMPokemonAgent
    ENHANCED_AGENT_AVAILABLE = True
except ImportError:
    ENHANCED_AGENT_AVAILABLE = False


class TrainingMode(Enum):
    """Available training modes"""
    FAST_MONITORED = "fast_monitored"   # Fast training with comprehensive monitoring
    CURRICULUM = "curriculum"           # Progressive skill-based training (legacy)
    ULTRA_FAST = "ultra_fast"          # Rule-based maximum speed (legacy)
    CUSTOM = "custom"                  # User-defined configuration


class LLMBackend(Enum):
    """Available LLM backends"""
    SMOLLM2 = "smollm2:1.7b"          # Ultra-fast, optimized
    LLAMA32_1B = "llama3.2:1b"        # Fastest Llama
    LLAMA32_3B = "llama3.2:3b"        # Balanced speed/quality
    QWEN25_3B = "qwen2.5:3b"          # Alternative fast option
    NONE = None                        # Rule-based only


@dataclass
class TrainingConfig:
    """Unified training configuration"""
    # Core settings
    rom_path: str
    mode: TrainingMode = TrainingMode.FAST_MONITORED
    llm_backend: LLMBackend = LLMBackend.SMOLLM2
    
    # Training parameters
    max_actions: int = 1000
    max_episodes: int = 10
    llm_interval: int = 10             # Actions between LLM calls
    
    # Performance settings
    headless: bool = True
    debug_mode: bool = False
    save_state_path: Optional[str] = None
    
    # Frame timing (Game Boy runs at 60 FPS)
    frames_per_action: int = 24         # Standard RL timing: 24 frames = 400ms = 2.5 actions/sec
    
    # Web interface
    enable_web: bool = False
    web_port: int = 8080
    web_host: str = "localhost"
    
    # Screen capture
    capture_screens: bool = True
    capture_fps: int = 5               # Reduced FPS for stability
    screen_resize: tuple = (240, 216)  # Smaller size for column width
    
    # Curriculum settings (for curriculum mode)
    curriculum_stages: int = 5
    stage_mastery_threshold: float = 0.7
    min_stage_episodes: int = 5
    max_stage_episodes: int = 20
    
    # Output settings
    save_stats: bool = True
    stats_file: str = "training_stats.json"
    log_level: str = "INFO"            # DEBUG, INFO, WARNING, ERROR


class UnifiedPokemonTrainer:
    """Unified Pokemon Crystal training system"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        
        # Initialize components based on mode
        self.pyboy = None
        self.env = None
        self.llm_agent = None
        
        # Performance tracking
        self.stats = {
            'start_time': time.time(),
            'total_actions': 0,
            'total_episodes': 0,
            'llm_calls': 0,
            'actions_per_second': 0.0,
            'mode': config.mode.value,
            'model': config.llm_backend.value if config.llm_backend else "rule-based"
        }
        
        # Screen capture
        self.screen_queue = queue.Queue(maxsize=30)
        self.latest_screen = None
        self.capture_thread = None
        self.capture_active = False
        
        # Web server
        self.web_server = None
        self.web_thread = None
        
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize training components based on mode"""
        print(f"🚀 Initializing {self.config.mode.value.title()} Training Mode")
        
        # Fast monitored mode uses direct PyBoy for speed with full monitoring
        if self.config.mode in [TrainingMode.FAST_MONITORED, TrainingMode.ULTRA_FAST]:
            self._init_direct_pyboy()
        elif self.config.mode == TrainingMode.CURRICULUM:
            self._init_environment_wrapper()
        
        if self.config.llm_backend and self.config.llm_backend != LLMBackend.NONE:
            self._init_llm_backend()
        
        if self.config.enable_web:
            self._init_web_server()
        
        print("✅ Trainer initialized successfully!")
    
    def _init_direct_pyboy(self):
        """Initialize direct PyBoy for maximum performance"""
        if not PYBOY_AVAILABLE:
            raise RuntimeError("PyBoy not available for fast local training")
        
        self.pyboy = PyBoy(
            self.config.rom_path,
            window="null" if self.config.headless else "SDL2",
            debug=self.config.debug_mode
        )
        
        # Action mappings
        self.actions = {
            1: WindowEvent.PRESS_ARROW_UP,
            2: WindowEvent.PRESS_ARROW_DOWN, 
            3: WindowEvent.PRESS_ARROW_LEFT,
            4: WindowEvent.PRESS_ARROW_RIGHT,
            5: WindowEvent.PRESS_BUTTON_A,
            6: WindowEvent.PRESS_BUTTON_B,
            7: WindowEvent.PRESS_BUTTON_START,
            8: WindowEvent.PRESS_BUTTON_SELECT,
            0: None
        }
        
        print(f"✅ PyBoy initialized ({'headless' if self.config.headless else 'windowed'})")
    
    def _init_environment_wrapper(self):
        """Initialize environment wrapper for advanced training"""
        if not PYBOY_ENV_AVAILABLE:
            print("⚠️ PyBoy environment not available, falling back to direct PyBoy")
            self._init_direct_pyboy()
            return
        
        self.env = PyBoyPokemonCrystalEnv(
            rom_path=self.config.rom_path,
            save_state_path=self.config.save_state_path,
            headless=self.config.headless,
            debug_mode=self.config.debug_mode
        )
        
        print("✅ Environment wrapper initialized")
    
    def _init_llm_backend(self):
        """Initialize LLM backend"""
        model_name = self.config.llm_backend.value
        
        try:
            # Check if model is available
            ollama.show(model_name)
            print(f"✅ Using LLM model: {model_name}")
        except:
            print(f"📥 Pulling LLM model: {model_name}")
            ollama.pull(model_name)
        
        # Initialize enhanced agent if available
        if ENHANCED_AGENT_AVAILABLE and self.config.mode == TrainingMode.MONITORED:
            self.llm_agent = EnhancedLLMPokemonAgent(
                model_name=model_name,
                use_vision=True
            )
        else:
            # Use simple LLM interface
            self.llm_agent = SimpleLLMAgent(model_name)
    
    def _init_web_server(self):
        """Initialize web monitoring server"""
        self.web_server = self._create_web_server()
        self.web_thread = threading.Thread(
            target=self.web_server.serve_forever, 
            daemon=True
        )
        self.web_thread.start()
        
        print(f"🌐 Web interface: http://{self.config.web_host}:{self.config.web_port}")
    
    def start_training(self):
        """Start the training process"""
        print(f"\n⚡ STARTING {self.config.mode.value.upper()} TRAINING")
        print("=" * 60)
        self._print_config_summary()
        
        # Set training active flag (for API status)
        self._training_active = True
        
        # Start screen capture if enabled
        if self.config.capture_screens:
            self._start_screen_capture()
        
        # Route to appropriate training method
        if self.config.mode == TrainingMode.FAST_MONITORED:
            self._run_fast_monitored_training()
        elif self.config.mode == TrainingMode.CURRICULUM:
            self._run_curriculum_training()
        elif self.config.mode == TrainingMode.ULTRA_FAST:
            self._run_ultra_fast_training()
        else:
            raise ValueError(f"Unknown training mode: {self.config.mode}")
    
    def _run_fast_monitored_training(self):
        """Run unified fast training with comprehensive monitoring"""
        # Check if synchronized training is requested
        if self.config.capture_screens:
            return self._run_synchronized_training()
        
        # Fallback to legacy fast training
        return self._run_legacy_fast_training()
    
    def _run_synchronized_training(self):
        """Run frame-synchronized training with screenshot-decision-action cycles"""
        actions_taken = 0
        
        # Load save state if available
        if self.config.save_state_path and self.pyboy:
            try:
                with open(self.config.save_state_path, 'rb') as f:
                    self.pyboy.load_state(f)
                print(f"💾 Loaded save state: {self.config.save_state_path}")
            except Exception as e:
                print(f"⚠️ Could not load save state: {e}")
        
        frame_duration_ms = 1000.0 / 60.0  # Game Boy runs at 60 FPS
        action_duration_ms = self.config.frames_per_action * frame_duration_ms
        
        print(f"🔄 Synchronized training: {self.config.frames_per_action} frames per action ({action_duration_ms:.1f}ms)")
        print(f"⏱️ Expected speed: {1000.0 / action_duration_ms:.1f} actions/second")
        
        try:
            while actions_taken < self.config.max_actions:
                cycle_start = time.time()
                
                # 1. SCREENSHOT PHASE - Capture current game state
                screenshot = self._capture_synchronized_screenshot()
                
                # 2. DECISION PHASE - Make intelligent decision
                if self.config.llm_backend and actions_taken % self.config.llm_interval == 0:
                    action = self._get_llm_action_with_vision(screenshot)
                    self.stats['llm_calls'] += 1
                else:
                    # Use rule-based logic or previous decision
                    action = self._get_rule_based_action(actions_taken)
                
                # 3. ACTION EXECUTION PHASE - Execute for exact frame duration
                self._execute_synchronized_action(action, self.config.frames_per_action)
                
                actions_taken += 1
                self.stats['total_actions'] = actions_taken
                
                # Progress monitoring
                if actions_taken % 50 == 0:
                    self._update_stats()
                    elapsed = time.time() - self.stats['start_time']
                    aps = actions_taken / elapsed
                    cycle_time = (time.time() - cycle_start) * 1000
                    print(f"🔄 Step {actions_taken}: Action {action}, Cycle: {cycle_time:.0f}ms, Speed: {aps:.1f} a/s")
                    
                    # Detailed monitoring every 200 actions
                    if actions_taken % 200 == 0:
                        llm_ratio = self.stats['llm_calls'] / actions_taken * 100
                        print(f"📊 Progress: {actions_taken}/{self.config.max_actions} | 🤖 LLM: {llm_ratio:.1f}%")
                        if self.config.enable_web:
                            print(f"🌐 Monitor: http://{self.config.web_host}:{self.config.web_port}")
        
        except KeyboardInterrupt:
            print("\n⏸️ Synchronized training interrupted")
        
        finally:
            self._finalize_training()
    
    def _run_legacy_fast_training(self):
        """Legacy fast training method (no synchronization)"""
        actions_taken = 0
        last_llm_action = 5  # Default to A button
        
        # Load save state if available
        if self.config.save_state_path and self.pyboy:
            try:
                with open(self.config.save_state_path, 'rb') as f:
                    self.pyboy.load_state(f)
                print(f"💾 Loaded save state: {self.config.save_state_path}")
            except Exception as e:
                print(f"⚠️ Could not load save state: {e}")
        
        print("⚡ Legacy fast training: maximum speed without synchronization")
        
        try:
            while actions_taken < self.config.max_actions:
                # Advance game with PyBoy speed constraints (max ~5x speed)
                if self.pyboy:
                    self.pyboy.tick()
                
                # Get action - LLM decisions at intervals for intelligence
                if self.config.llm_backend and actions_taken % self.config.llm_interval == 0:
                    action = self._get_llm_action()
                    last_llm_action = action
                else:
                    # Reuse last LLM action for speed
                    action = last_llm_action
                
                # Execute action
                self._execute_action(action)
                actions_taken += 1
                self.stats['total_actions'] = actions_taken
                
                # Comprehensive progress monitoring
                if actions_taken % 100 == 0:
                    self._update_stats()
                    elapsed = time.time() - self.stats['start_time']
                    aps = actions_taken / elapsed
                    print(f"📊 Progress: {actions_taken}/{self.config.max_actions} ({aps:.1f} a/s)")
                    
                    # Additional monitoring info every 500 actions
                    if actions_taken % 500 == 0:
                        llm_ratio = self.stats['llm_calls'] / actions_taken * 100
                        print(f"🤖 LLM decisions: {self.stats['llm_calls']} ({llm_ratio:.1f}% of actions)")
                        if self.config.enable_web:
                            print(f"🌐 Web monitor: http://{self.config.web_host}:{self.config.web_port}")
                
                # Optimal delay for PyBoy stability (respects <5x speed limit)
                time.sleep(0.008)  # ~125 actions/second max
        
        except KeyboardInterrupt:
            print("\n⏸️ Training interrupted")
        
        finally:
            self._finalize_training()
    
    def _run_fast_local_training(self):
        """Run optimized local training"""
        actions_taken = 0
        last_llm_action = 5  # Default to A button
        
        try:
            while actions_taken < self.config.max_actions:
                # Advance game
                if self.pyboy:
                    self.pyboy.tick()
                
                # Get action
                if self.config.llm_backend and actions_taken % self.config.llm_interval == 0:
                    action = self._get_llm_action()
                    last_llm_action = action
                else:
                    action = last_llm_action
                
                # Execute action
                self._execute_action(action)
                actions_taken += 1
                self.stats['total_actions'] = actions_taken
                
                # Progress updates
                if actions_taken % 100 == 0:
                    self._update_stats()
                    elapsed = time.time() - self.stats['start_time']
                    aps = actions_taken / elapsed
                    print(f"📊 Progress: {actions_taken}/{self.config.max_actions} ({aps:.1f} a/s)")
                
                # Small delay for stability
                time.sleep(0.005)
        
        except KeyboardInterrupt:
            print("\n⏸️ Training interrupted")
        
        finally:
            self._finalize_training()
    
    def _run_curriculum_training(self):
        """Run progressive curriculum training"""
        current_stage = 1
        stage_episodes = 0
        stage_successes = 0
        
        print(f"📚 Starting {self.config.curriculum_stages}-stage curriculum")
        
        try:
            while (current_stage <= self.config.curriculum_stages and 
                   self.stats['total_episodes'] < self.config.max_episodes):
                
                # Run single episode
                success = self._run_curriculum_episode(current_stage)
                
                stage_episodes += 1
                self.stats['total_episodes'] += 1
                
                if success:
                    stage_successes += 1
                
                # Check stage mastery
                success_rate = stage_successes / stage_episodes
                
                print(f"📖 Stage {current_stage}, Episode {stage_episodes}: "
                      f"{'✅' if success else '❌'} ({success_rate:.1%} success)")
                
                # Advance stage if mastered
                if (stage_episodes >= self.config.min_stage_episodes and 
                    success_rate >= self.config.stage_mastery_threshold):
                    
                    print(f"🎓 Stage {current_stage} mastered! Advancing...")
                    current_stage += 1
                    stage_episodes = 0
                    stage_successes = 0
                
                # Timeout check
                elif stage_episodes >= self.config.max_stage_episodes:
                    print(f"⏰ Stage {current_stage} timeout, advancing anyway")
                    current_stage += 1
                    stage_episodes = 0
                    stage_successes = 0
        
        except KeyboardInterrupt:
            print("\n⏸️ Curriculum training interrupted")
        
        finally:
            self._finalize_training()
    
    def _run_curriculum_episode(self, stage: int) -> bool:
        """Run single curriculum episode"""
        if self.env:
            state = self.env.reset()
        else:
            self.pyboy.load_state(self.config.save_state_path) if self.config.save_state_path else None
        
        actions_taken = 0
        max_actions = 500  # Episode length
        success_indicators = 0
        
        while actions_taken < max_actions:
            # Get stage-appropriate action
            action = self._get_stage_action(stage)
            
            # Execute action
            if self.env:
                next_state, reward, done, info = self.env.step(action)
                if reward > 0:
                    success_indicators += 1
                state = next_state
                if done:
                    break
            else:
                self._execute_action(action)
                # Simple progress detection for PyBoy
                success_indicators += 1 if actions_taken % 50 == 0 else 0
            
            actions_taken += 1
            self.stats['total_actions'] += 1
        
        # Success criteria: multiple indicators of progress
        return success_indicators >= 5
    
    def _run_ultra_fast_training(self):
        """Run rule-based ultra-fast training"""
        actions_taken = 0
        action_pattern = [5, 5, 1, 1, 4, 4, 2, 2, 3, 3]  # Exploration pattern
        pattern_index = 0
        
        print("🚀 Ultra-fast rule-based training (no LLM overhead)")
        
        try:
            while actions_taken < self.config.max_actions:
                # Get action from pattern
                action = action_pattern[pattern_index % len(action_pattern)]
                pattern_index += 1
                
                # Execute action
                self._execute_action(action)
                actions_taken += 1
                self.stats['total_actions'] = actions_taken
                
                # Progress updates
                if actions_taken % 200 == 0:
                    elapsed = time.time() - self.stats['start_time']
                    aps = actions_taken / elapsed
                    print(f"🚀 Ultra-fast: {actions_taken}/{self.config.max_actions} ({aps:.0f} a/s)")
                
                # Minimal delay
                time.sleep(0.001)
        
        except KeyboardInterrupt:
            print("\n⏸️ Ultra-fast training interrupted")
        
        finally:
            self._finalize_training()
    
    def _run_monitored_training(self):
        """Run full monitoring training"""
        if not self.env:
            print("⚠️ Environment required for monitored training")
            return
        
        episode = 0
        
        try:
            while episode < self.config.max_episodes:
                print(f"\n📊 Episode {episode + 1}/{self.config.max_episodes}")
                
                state = self.env.reset()
                episode_reward = 0
                actions_taken = 0
                
                while actions_taken < 1000:  # Max actions per episode
                    # Get intelligent action
                    if self.llm_agent:
                        action = self.llm_agent.decide_action(state)
                    else:
                        action = self._get_llm_action()
                    
                    # Execute action
                    next_state, reward, done, info = self.env.step(action)
                    
                    episode_reward += reward
                    actions_taken += 1
                    self.stats['total_actions'] += 1
                    
                    state = next_state
                    
                    if done:
                        break
                
                episode += 1
                self.stats['total_episodes'] = episode
                
                print(f"✅ Episode reward: {episode_reward}, Actions: {actions_taken}")
        
        except KeyboardInterrupt:
            print("\n⏸️ Monitored training interrupted")
        
        finally:
            self._finalize_training()
    
    def _get_llm_action(self, stage: str = "BASIC_CONTROLS") -> int:
        """Get action from LLM"""
        if not self.config.llm_backend:
            return 5  # Default A button
        
        prompt = f"""Pokemon Crystal - Stage: {stage}
Choose action number:
1=UP, 2=DOWN, 3=LEFT, 4=RIGHT, 5=A, 6=B, 7=START

Action:"""
        
        try:
            response = ollama.generate(
                model=self.config.llm_backend.value,
                prompt=prompt,
                options={
                    'num_predict': 2,
                    'temperature': 0.2,
                    'top_k': 8
                }
            )
            
            self.stats['llm_calls'] += 1
            
            # Parse action
            text = response['response'].strip()
            for char in text:
                if char.isdigit() and '1' <= char <= '7':
                    return int(char)
            
            return 5  # Default
        
        except Exception as e:
            print(f"⚠️ LLM error: {e}")
            return 5
    
    def _get_stage_action(self, stage: int) -> int:
        """Get stage-appropriate action for curriculum training"""
        stage_prompts = {
            1: "BASIC_CONTROLS - Focus on navigation",
            2: "DIALOGUE - Focus on text interaction", 
            3: "POKEMON_SELECTION - Focus on menu choices",
            4: "BATTLE_FUNDAMENTALS - Focus on combat",
            5: "EXPLORATION - Focus on world navigation"
        }
        
        stage_name = stage_prompts.get(stage, "GENERAL")
        return self._get_llm_action(stage_name)
    
    def _execute_action(self, action: int):
        """Execute action in the game"""
        if self.pyboy and action in self.actions and self.actions[action]:
            self.pyboy.send_input(self.actions[action])
            self.pyboy.tick()
    
    def _capture_synchronized_screenshot(self) -> Optional[np.ndarray]:
        """Capture screenshot synchronously for decision making"""
        try:
            if not self.pyboy:
                return None
            
            # Get current screen data
            screen_array = self.pyboy.screen.ndarray
            if screen_array is None:
                return None
            
            # Convert to standard format (RGB)
            if len(screen_array.shape) == 3 and screen_array.shape[-1] == 4:
                # RGBA to RGB
                screen_rgb = screen_array[:, :, :3].astype(np.uint8)
            elif len(screen_array.shape) == 3 and screen_array.shape[-1] == 3:
                # Already RGB
                screen_rgb = screen_array.astype(np.uint8)
            elif len(screen_array.shape) == 2:
                # Grayscale, convert to RGB
                screen_rgb = np.stack([screen_array, screen_array, screen_array], axis=2).astype(np.uint8)
            else:
                return None
            
            return screen_rgb.copy()  # Return a copy for safety
            
        except Exception as e:
            print(f"⚠️ Synchronized screenshot failed: {e}")
            return None
    
    def _get_llm_action_with_vision(self, screenshot: Optional[np.ndarray]) -> int:
        """Get LLM action using visual input"""
        if not self.config.llm_backend:
            return self._get_rule_based_action(self.stats['total_actions'])
        
        # For now, use text-based LLM (vision integration can be added later)
        # Include visual context in the prompt
        visual_context = "Screen captured" if screenshot is not None else "No screen data"
        
        prompt = f"""Pokemon Crystal RL Training

Visual Context: {visual_context}
Current Step: {self.stats['total_actions']}

Choose action number:
1=UP, 2=DOWN, 3=LEFT, 4=RIGHT, 5=A, 6=B, 7=START, 8=SELECT

Action:"""
        
        try:
            response = ollama.generate(
                model=self.config.llm_backend.value,
                prompt=prompt,
                options={
                    'num_predict': 2,
                    'temperature': 0.3,
                    'top_k': 8
                }
            )
            
            # Parse action from response
            text = response['response'].strip()
            for char in text:
                if char.isdigit() and '1' <= char <= '8':
                    return int(char)
            
            # Fallback to rule-based if parsing fails
            return self._get_rule_based_action(self.stats['total_actions'])
            
        except Exception as e:
            print(f"⚠️ LLM vision action failed: {e}")
            return self._get_rule_based_action(self.stats['total_actions'])
    
    def _get_rule_based_action(self, step: int) -> int:
        """Get rule-based action for synchronized training"""
        # Smart rule-based logic for Pokemon Crystal
        action_patterns = {
            # Early game: Navigate menus and start game
            'menu_navigation': [7, 5, 5, 2, 5],  # START, A, A, DOWN, A
            'exploration': [1, 1, 4, 4, 2, 2, 3, 3, 5, 5],  # Basic movement + interactions
            'dialogue': [5, 5, 5, 2, 5],  # A spam for dialogue
            'battle': [5, 1, 5, 2, 5],  # Simple battle actions
        }
        
        # Determine current phase based on step count
        if step < 50:
            pattern = action_patterns['menu_navigation']
        elif step < 200:
            pattern = action_patterns['exploration']
        elif step < 400:
            pattern = action_patterns['dialogue']
        else:
            pattern = action_patterns['exploration']
        
        # Select action from pattern
        pattern_index = step % len(pattern)
        return pattern[pattern_index]
    
    def _execute_synchronized_action(self, action: int, frames: int):
        """Execute action for exact frame duration"""
        if not self.pyboy:
            return
        
        # Press the button (if action is not 0/no-op)
        if action in self.actions and self.actions[action]:
            self.pyboy.send_input(self.actions[action])
        
        # Run the exact number of frames
        for frame in range(frames):
            self.pyboy.tick()
            
            # Optional: Release button halfway through for more realistic timing
            if frame == frames // 2 and action in self.actions and self.actions[action]:
                # Release the button (this helps with some games that need button releases)
                pass  # PyBoy handles button releases automatically
    
    def _start_screen_capture(self):
        """Start screen capture thread"""
        self.capture_active = True
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        print("📸 Screen capture started")
    
    def _capture_loop(self):
        """Fixed screen capture loop with proper RGBA handling"""
        consecutive_errors = 0
        max_consecutive_errors = 10
        
        print("🔍 Starting screen capture loop...")
        
        while self.capture_active:
            try:
                if self.pyboy:
                    # Get screen data using the confirmed working method
                    screen_array = None
                    try:
                        # Method 1: Use screen.ndarray (confirmed working)
                        screen_array = self.pyboy.screen.ndarray
                        if screen_array is not None:
                            screen_array = screen_array.copy()  # Make a copy to be safe
                    except Exception as e:
                        print(f"⚠️ Screen ndarray failed: {e}")
                        # Fallback: Try screen.image
                        try:
                            if hasattr(self.pyboy.screen, 'image'):
                                pil_image = self.pyboy.screen.image
                                if pil_image is not None:
                                    screen_array = np.array(pil_image)
                        except Exception as e2:
                            print(f"⚠️ Screen image fallback failed: {e2}")
                            continue
                    
                    # Validate screen array
                    if screen_array is None or screen_array.size == 0:
                        consecutive_errors += 1
                        if consecutive_errors <= 3:
                            print(f"⚠️ Invalid screen data: None or empty")
                        continue
                    
                    # Process screen with detailed error handling
                    try:
                        # Debug info on first successful capture
                        if consecutive_errors == 0 and self.stats['total_actions'] < 10:
                            print(f"📊 Screen data - Shape: {screen_array.shape}, Type: {screen_array.dtype}, Range: {screen_array.min()}-{screen_array.max()}")
                        
                        # Handle different data formats
                        if len(screen_array.shape) == 3:
                            # Multi-channel image data
                            if screen_array.shape[-1] == 4:  # RGBA
                                # Convert RGBA to RGB by dropping alpha channel
                                screen_rgb = screen_array[:, :, :3].astype(np.uint8)
                                screen_pil = Image.fromarray(screen_rgb, mode='RGB')
                            elif screen_array.shape[-1] == 3:  # RGB
                                screen_pil = Image.fromarray(screen_array.astype(np.uint8), mode='RGB')
                            else:
                                print(f"⚠️ Unexpected channel count: {screen_array.shape[-1]}")
                                continue
                        elif len(screen_array.shape) == 2:
                            # Grayscale or palette data
                            screen_pil = Image.fromarray(screen_array.astype(np.uint8), mode='L')
                            # Convert to RGB for consistency
                            screen_pil = screen_pil.convert('RGB')
                        else:
                            print(f"⚠️ Unexpected array shape: {screen_array.shape}")
                            continue
                        
                        # Resize the image
                        screen_resized = screen_pil.resize(self.config.screen_resize, Image.NEAREST)
                        
                        # Convert to base64
                        buffer = io.BytesIO()
                        screen_resized.save(buffer, format='PNG', optimize=True, compress_level=6)
                        screen_b64 = base64.b64encode(buffer.getvalue()).decode()
                        
                        # Validate the encoded data
                        if len(screen_b64) > 100:  # Reasonable minimum size for a PNG
                            # Update latest screen atomically
                            self.latest_screen = {
                                'image_b64': screen_b64,
                                'timestamp': time.time(),
                                'size': screen_resized.size,
                                'frame_id': int(time.time() * 1000),  # Unique frame ID
                                'data_length': len(screen_b64)
                            }
                            
                            # Add to queue (drop old frames if full)
                            try:
                                if self.screen_queue.full():
                                    try:
                                        self.screen_queue.get_nowait()  # Remove oldest
                                    except:
                                        pass
                                self.screen_queue.put_nowait(self.latest_screen)
                            except:
                                pass  # Queue operations are non-critical
                            
                            # Debug info on first successful capture
                            if consecutive_errors > 0 or (self.stats['total_actions'] < 10 and self.stats['total_actions'] % 5 == 0):
                                print(f"✅ Screen captured - Size: {screen_resized.size}, B64 length: {len(screen_b64)}")
                        else:
                            print(f"⚠️ Base64 data too small: {len(screen_b64)} chars")
                            continue
                        
                        # Reset error counter on success
                        consecutive_errors = 0
                        
                    except Exception as e:
                        consecutive_errors += 1
                        if consecutive_errors <= 3:  # Only log first few errors
                            print(f"⚠️ Screen processing error: {e}")
                            import traceback
                            print(f"📊 Error details: {traceback.format_exc()[:200]}...")
                
                # Frame rate limiting
                base_delay = 1.0 / self.config.capture_fps
                time.sleep(base_delay)
            
            except Exception as e:
                consecutive_errors += 1
                if consecutive_errors <= 3:
                    print(f"⚠️ Screen capture loop error: {e}")
                
                # Break loop if too many consecutive errors
                if consecutive_errors >= max_consecutive_errors:
                    print(f"❌ Too many screen capture errors ({consecutive_errors}), stopping capture")
                    self.capture_active = False
                    break
                
                time.sleep(0.5)  # Longer delay on critical errors
        
        print("🔍 Screen capture loop ended")
    
    def _update_stats(self):
        """Update performance statistics"""
        elapsed = time.time() - self.stats['start_time']
        self.stats['actions_per_second'] = self.stats['total_actions'] / max(elapsed, 0.001)
    
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
        print(f"\n📊 TRAINING SUMMARY")
        print("=" * 40)
        print(f"⏱️ Duration: {elapsed:.1f} seconds")
        print(f"🎯 Total actions: {self.stats['total_actions']}")
        print(f"📈 Episodes: {self.stats['total_episodes']}")
        print(f"🚀 Speed: {self.stats['actions_per_second']:.1f} actions/sec")
        print(f"🧠 LLM calls: {self.stats['llm_calls']}")
        
        # Save stats if enabled
        if self.config.save_stats:
            self.stats['end_time'] = time.time()
            with open(self.config.stats_file, 'w') as f:
                json.dump(self.stats, f, indent=2)
            print(f"💾 Stats saved to {self.config.stats_file}")
        
        # Cleanup
        if self.pyboy:
            self.pyboy.stop()
        
        if self.web_server:
            self.web_server.shutdown()
        
        print("🛑 Training completed and cleaned up")
    
    def _print_config_summary(self):
        """Print training configuration summary"""
        print(f"🎮 ROM: {self.config.rom_path}")
        print(f"🤖 LLM: {self.config.llm_backend.value if self.config.llm_backend else 'None (rule-based)'}")
        print(f"🎯 Target: {self.config.max_actions} actions / {self.config.max_episodes} episodes")
        print(f"📸 Capture: {'ON' if self.config.capture_screens else 'OFF'}")
        print(f"🌐 Web UI: {'ON' if self.config.enable_web else 'OFF'}")
        print()
    
    def _create_web_server(self):
        """Create comprehensive web monitoring server"""
        from http.server import HTTPServer, BaseHTTPRequestHandler
        import os
        import psutil
        
        class TrainingHandler(BaseHTTPRequestHandler):
            def __init__(self, trainer, *args, **kwargs):
                self.trainer = trainer
                super().__init__(*args, **kwargs)
            
            def do_GET(self):
                if self.path == '/':
                    self._serve_comprehensive_dashboard()
                elif self.path.startswith('/screen'):
                    self._serve_screen()
                elif self.path.startswith('/api/screenshot'):
                    self._serve_screen()  # Alias for screenshot endpoint
                elif self.path == '/stats':
                    self._serve_stats()
                elif self.path == '/api/status':
                    self._serve_api_status()
                elif self.path == '/api/system':
                    self._serve_api_system()
                elif self.path == '/api/runs':
                    self._serve_api_runs()
                elif self.path.startswith('/socket.io/'):
                    self._handle_socketio_fallback()
                else:
                    self.send_error(404)
            
            def do_POST(self):
                if self.path == '/api/start_training':
                    self._handle_start_training()
                elif self.path == '/api/stop_training':
                    self._handle_stop_training()
                else:
                    self.send_error(404)
            
            def _serve_comprehensive_dashboard(self):
                """Serve the comprehensive dashboard from templates"""
                try:
                    # Use local templates directory relative to python_agent
                    template_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'templates', 'dashboard.html')
                    with open(template_path, 'r', encoding='utf-8') as f:
                        html = f.read()
                    
                    self.send_response(200)
                    self.send_header('Content-type', 'text/html; charset=utf-8')
                    self.end_headers()
                    self.wfile.write(html.encode('utf-8'))
                except Exception as e:
                    print(f"⚠️ Error loading dashboard template: {e}")
                    self._serve_fallback_dashboard()
            
            def _serve_fallback_dashboard(self):
                """Fallback simple dashboard if template fails"""
                html = """<!DOCTYPE html>
<html>
<head>
    <title>Pokemon Crystal Trainer</title>
    <style>
        body { 
            font-family: 'Courier New', monospace; 
            margin: 20px; 
            background: #1a1a1a; 
            color: white; 
            line-height: 1.4;
        }
        .container { max-width: 1200px; margin: 0 auto; }
        .stats { background: #333; padding: 15px; border-radius: 8px; margin: 10px 0; }
        .screen { border: 3px solid #4CAF50; margin: 20px 0; text-align: center; background: #000; border-radius: 8px; padding: 10px; }
        .screen img { max-width: 100%; image-rendering: pixelated; }
        h1 { text-align: center; color: #4CAF50; }
    </style>
</head>
<body>
    <div class="container">
        <h1>⚡ Pokemon Crystal Unified Trainer</h1>
        <div class="stats" id="stats">Loading...</div>
        <div class="screen">
            <h3>🎮 Game Screen</h3>
            <img id="gameScreen" src="/screen" alt="Game Screen">
        </div>
    </div>
    <script>
        setInterval(() => {
            fetch('/stats').then(r => r.json()).then(data => {
                document.getElementById('stats').innerHTML = 
                    `🎯 Actions: ${data.total_actions} | ⚡ Speed: ${data.actions_per_second.toFixed(1)} a/s | 🧠 LLM: ${data.llm_calls}`;
            });
            document.getElementById('gameScreen').src = '/screen?' + Date.now();
        }, 1000);
    </script>
</body>
</html>"""
                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                self.wfile.write(html.encode())
            
            def _serve_screen(self):
                if self.trainer.latest_screen:
                    img_data = base64.b64decode(self.trainer.latest_screen['image_b64'])
                    self.send_response(200)
                    self.send_header('Content-type', 'image/png')
                    self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
                    self.end_headers()
                    self.wfile.write(img_data)
                else:
                    self.send_error(404)
            
            def _serve_stats(self):
                self.trainer._update_stats()
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps(self.trainer.stats).encode())
            
            def _serve_api_status(self):
                """API endpoint for training status"""
                status = {
                    'is_training': hasattr(self.trainer, '_training_active') and self.trainer._training_active,
                    'current_run_id': getattr(self.trainer, 'current_run_id', None),
                    'mode': self.trainer.config.mode.value,
                    'model': self.trainer.config.llm_backend.value if self.trainer.config.llm_backend else 'rule-based',
                    'start_time': self.trainer.stats.get('start_time'),
                    'total_actions': self.trainer.stats.get('total_actions', 0),
                    'llm_calls': self.trainer.stats.get('llm_calls', 0),
                    'actions_per_second': self.trainer.stats.get('actions_per_second', 0.0)
                }
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps(status).encode())
            
            def _serve_api_system(self):
                """API endpoint for system statistics"""
                try:
                    stats = {
                        'cpu_percent': psutil.cpu_percent(),
                        'memory_percent': psutil.virtual_memory().percent,
                        'disk_usage': psutil.disk_usage('/').percent,
                        'gpu_available': False  # Could be enhanced to detect GPU
                    }
                except Exception as e:
                    stats = {
                        'cpu_percent': 0.0,
                        'memory_percent': 0.0,
                        'disk_usage': 0.0,
                        'gpu_available': False,
                        'error': str(e)
                    }
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps(stats).encode())
            
            def _serve_api_runs(self):
                """API endpoint for training runs history"""
                # For now, return current run as a single entry
                current_run = {
                    'id': 1,
                    'algorithm': self.trainer.config.mode.value,
                    'start_time': datetime.fromtimestamp(self.trainer.stats['start_time']).isoformat(),
                    'end_time': None,
                    'status': 'running' if hasattr(self.trainer, '_training_active') and self.trainer._training_active else 'completed',
                    'total_timesteps': self.trainer.stats.get('total_actions', 0),
                    'final_reward': 'N/A'
                }
                
                runs = [current_run]
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps(runs).encode())
            
            def _handle_start_training(self):
                """Handle training start request"""
                try:
                    content_length = int(self.headers['Content-Length'])
                    post_data = self.rfile.read(content_length)
                    config = json.loads(post_data.decode('utf-8'))
                    
                    response = {
                        'success': False,
                        'message': 'Training control not implemented in unified trainer yet'
                    }
                    
                    self.send_response(501)  # Not implemented
                    self.send_header('Content-type', 'application/json')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    self.wfile.write(json.dumps(response).encode())
                    
                except Exception as e:
                    self._send_error_response(str(e))
            
            def _handle_stop_training(self):
                """Handle training stop request"""
                response = {
                    'success': False,
                    'message': 'Training control not implemented in unified trainer yet'
                }
                
                self.send_response(501)  # Not implemented
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps(response).encode())
            
            def _handle_socketio_fallback(self):
                """Handle socket.io requests gracefully"""
                # Return a polite 404 for socket.io requests since we don't implement WebSockets
                response = {
                    'error': 'WebSocket/Socket.IO not implemented',
                    'message': 'This trainer uses HTTP polling instead of WebSockets'
                }
                self.send_response(404)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps(response).encode())
            
            def _send_error_response(self, error_msg):
                response = {'success': False, 'error': error_msg}
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps(response).encode())
        
        # Create handler class with trainer reference
        def handler_factory(trainer):
            return lambda *args, **kwargs: TrainingHandler(trainer, *args, **kwargs)
        
        return HTTPServer((self.config.web_host, self.config.web_port), handler_factory(self))


class SimpleLLMAgent:
    """Simple LLM agent for basic training modes"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name


def main():
    """Main CLI interface"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Unified Pokemon Crystal RL Trainer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Training Modes:
  fast_local    - Optimized local training with real-time capture
  curriculum    - Progressive skill-based learning (5 stages)
  ultra_fast    - Rule-based maximum speed training
  monitored     - Full analysis and monitoring
  custom        - User-defined configuration

Examples:
  python pokemon_trainer.py --rom game.gbc --mode fast_local --actions 1000 --web
  python pokemon_trainer.py --rom game.gbc --mode curriculum --episodes 50
  python pokemon_trainer.py --rom game.gbc --mode ultra_fast --actions 5000
        """
    )
    
    # Required arguments
    parser.add_argument('--rom', required=True, help='Path to Pokemon Crystal ROM')
    
    # Training mode
    parser.add_argument('--mode', choices=[m.value for m in TrainingMode], 
                       default='fast_monitored', help='Training mode')
    
    # LLM settings
    parser.add_argument('--model', choices=[m.value for m in LLMBackend if m.value], 
                       default='smollm2:1.7b', help='LLM model to use')
    parser.add_argument('--no-llm', action='store_true', help='Use rule-based training only')
    
    # Training parameters
    parser.add_argument('--actions', type=int, default=1000, help='Maximum actions')
    parser.add_argument('--episodes', type=int, default=10, help='Maximum episodes')
    parser.add_argument('--llm-interval', type=int, default=10, help='Actions between LLM calls')
    parser.add_argument('--frames-per-action', type=int, default=24, help='Frames per action (24=standard, 16=faster, 8=legacy)')
    
    # Interface options
    parser.add_argument('--web', action='store_true', help='Enable web interface')
    parser.add_argument('--port', type=int, default=8080, help='Web interface port')
    parser.add_argument('--no-capture', action='store_true', help='Disable screen capture')
    
    # Other options
    parser.add_argument('--save-state', help='Save state file to load from')
    parser.add_argument('--windowed', action='store_true', help='Show game window')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    # Create configuration
    config = TrainingConfig(
        rom_path=args.rom,
        mode=TrainingMode(args.mode),
        llm_backend=None if args.no_llm else LLMBackend(args.model),
        max_actions=args.actions,
        max_episodes=args.episodes,
        llm_interval=args.llm_interval,
        frames_per_action=args.frames_per_action,
        headless=not args.windowed,
        debug_mode=args.debug,
        save_state_path=args.save_state,
        enable_web=args.web,
        web_port=args.port,
        capture_screens=not args.no_capture
    )
    
    # Create and start trainer
    trainer = UnifiedPokemonTrainer(config)
    
    try:
        trainer.start_training()
    except KeyboardInterrupt:
        print("\n⏸️ Interrupted by user")
    except Exception as e:
        print(f"❌ Training error: {e}")
        raise


if __name__ == "__main__":
    main()
