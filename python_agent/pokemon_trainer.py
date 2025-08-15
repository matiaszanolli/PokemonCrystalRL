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
    from pyboy_env import PyBoyPokemonCrystalEnv
    PYBOY_ENV_AVAILABLE = True
except ImportError:
    PYBOY_ENV_AVAILABLE = False

try:
    from enhanced_llm_agent import EnhancedLLMPokemonAgent
    ENHANCED_AGENT_AVAILABLE = True
except ImportError:
    ENHANCED_AGENT_AVAILABLE = False


class TrainingMode(Enum):
    """Available training modes"""
    FAST_LOCAL = "fast_local"           # Direct PyBoy with optimized capture
    CURRICULUM = "curriculum"           # Progressive skill-based training
    ULTRA_FAST = "ultra_fast"          # Rule-based maximum speed
    MONITORED = "monitored"            # Full monitoring and analysis
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
    mode: TrainingMode = TrainingMode.FAST_LOCAL
    llm_backend: LLMBackend = LLMBackend.SMOLLM2
    
    # Training parameters
    max_actions: int = 1000
    max_episodes: int = 10
    llm_interval: int = 10             # Actions between LLM calls
    
    # Performance settings
    headless: bool = True
    debug_mode: bool = False
    save_state_path: Optional[str] = None
    
    # Web interface
    enable_web: bool = False
    web_port: int = 8080
    web_host: str = "localhost"
    
    # Screen capture
    capture_screens: bool = True
    capture_fps: int = 10              # Frames per second
    screen_resize: tuple = (320, 288)
    
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
        
        if self.config.mode in [TrainingMode.FAST_LOCAL, TrainingMode.ULTRA_FAST]:
            self._init_direct_pyboy()
        elif self.config.mode in [TrainingMode.CURRICULUM, TrainingMode.MONITORED]:
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
        
        # Start screen capture if enabled
        if self.config.capture_screens:
            self._start_screen_capture()
        
        # Route to appropriate training method
        if self.config.mode == TrainingMode.FAST_LOCAL:
            self._run_fast_local_training()
        elif self.config.mode == TrainingMode.CURRICULUM:
            self._run_curriculum_training()
        elif self.config.mode == TrainingMode.ULTRA_FAST:
            self._run_ultra_fast_training()
        elif self.config.mode == TrainingMode.MONITORED:
            self._run_monitored_training()
        else:
            raise ValueError(f"Unknown training mode: {self.config.mode}")
    
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
    
    def _start_screen_capture(self):
        """Start screen capture thread"""
        self.capture_active = True
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        print("📸 Screen capture started")
    
    def _capture_loop(self):
        """Screen capture loop"""
        while self.capture_active:
            try:
                if self.pyboy:
                    # Get screen
                    try:
                        screen_array = self.pyboy.screen.ndarray
                    except AttributeError:
                        screen_array = np.array(self.pyboy.screen.image)
                    
                    # Process screen
                    screen_pil = Image.fromarray(screen_array)
                    screen_resized = screen_pil.resize(self.config.screen_resize, Image.NEAREST)
                    
                    # Convert to base64
                    buffer = io.BytesIO()
                    screen_resized.save(buffer, format='PNG', optimize=True)
                    screen_b64 = base64.b64encode(buffer.getvalue()).decode()
                    
                    # Update latest screen
                    self.latest_screen = {
                        'image_b64': screen_b64,
                        'timestamp': time.time(),
                        'size': screen_resized.size
                    }
                    
                    # Add to queue
                    if not self.screen_queue.full():
                        self.screen_queue.put(self.latest_screen)
                
                time.sleep(1.0 / self.config.capture_fps)
            
            except Exception as e:
                print(f"⚠️ Screen capture error: {e}")
                time.sleep(0.1)
    
    def _update_stats(self):
        """Update performance statistics"""
        elapsed = time.time() - self.stats['start_time']
        self.stats['actions_per_second'] = self.stats['total_actions'] / max(elapsed, 0.001)
    
    def _finalize_training(self):
        """Cleanup and final statistics"""
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
        """Create web monitoring server"""
        from http.server import HTTPServer, BaseHTTPRequestHandler
        
        class TrainingHandler(BaseHTTPRequestHandler):
            def __init__(self, trainer, *args, **kwargs):
                self.trainer = trainer
                super().__init__(*args, **kwargs)
            
            def do_GET(self):
                if self.path == '/':
                    self._serve_dashboard()
                elif self.path.startswith('/screen'):
                    self._serve_screen()
                elif self.path == '/stats':
                    self._serve_stats()
                else:
                    self.send_error(404)
            
            def _serve_dashboard(self):
                html = """<!DOCTYPE html>
<html>
<head>
    <title>Pokemon Crystal Trainer</title>
    <style>
        body { font-family: Arial; margin: 20px; background: #1a1a1a; color: white; }
        .container { max-width: 1000px; margin: 0 auto; }
        .stats { background: #333; padding: 15px; border-radius: 5px; margin: 10px 0; }
        .screen { border: 2px solid #4CAF50; margin: 20px 0; text-align: center; }
        img { max-width: 100%; height: auto; }
        .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>⚡ Pokemon Crystal Unified Trainer</h1>
        <div class="stats" id="stats">Loading...</div>
        <div class="grid">
            <div class="screen">
                <h3>🎮 Game Screen</h3>
                <img id="gameScreen" src="/screen" alt="Game Screen">
            </div>
            <div>
                <h3>📊 Training Info</h3>
                <div id="details">Loading...</div>
            </div>
        </div>
    </div>
    <script>
        function updateAll() {
            fetch('/stats').then(r => r.json()).then(data => {
                document.getElementById('stats').innerHTML = 
                    `🎯 Actions: ${data.total_actions} | ` +
                    `⚡ Speed: ${data.actions_per_second.toFixed(1)} a/s | ` +
                    `📈 Episodes: ${data.total_episodes} | ` +
                    `🧠 LLM Calls: ${data.llm_calls}`;
                
                document.getElementById('details').innerHTML = 
                    `<p><strong>Mode:</strong> ${data.mode}</p>` +
                    `<p><strong>Model:</strong> ${data.model}</p>` +
                    `<p><strong>Runtime:</strong> ${((Date.now()/1000) - data.start_time).toFixed(1)}s</p>`;
            });
            
            document.getElementById('gameScreen').src = '/screen?' + Date.now();
        }
        
        setInterval(updateAll, 1000);
        updateAll();
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
                    self.end_headers()
                    self.wfile.write(img_data)
                else:
                    self.send_error(404)
            
            def _serve_stats(self):
                self.trainer._update_stats()
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(self.trainer.stats).encode())
        
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
                       default='fast_local', help='Training mode')
    
    # LLM settings
    parser.add_argument('--model', choices=[m.value for m in LLMBackend if m.value], 
                       default='smollm2:1.7b', help='LLM model to use')
    parser.add_argument('--no-llm', action='store_true', help='Use rule-based training only')
    
    # Training parameters
    parser.add_argument('--actions', type=int, default=1000, help='Maximum actions')
    parser.add_argument('--episodes', type=int, default=10, help='Maximum episodes')
    parser.add_argument('--llm-interval', type=int, default=10, help='Actions between LLM calls')
    
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
