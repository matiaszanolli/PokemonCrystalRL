#!/usr/bin/env python3
"""
fast_local_training.py - Optimized Local Pokemon Crystal Training

Fast training with optimized screen capture for real-time UI display.
Removes network overhead and focuses on local performance.
"""

import time
import numpy as np
from typing import Dict, Any
import threading
import queue
import io
import base64
from PIL import Image
import ollama

# Try importing PyBoy directly for faster access
try:
    from pyboy import PyBoy
    from pyboy.utils import WindowEvent
    PYBOY_AVAILABLE = True
except ImportError:
    PYBOY_AVAILABLE = False
    print("⚠️ PyBoy not available, falling back to environment wrapper")

class FastPokemonTrainer:
    """Ultra-fast local Pokemon trainer with optimized screen capture"""
    
    def __init__(self, rom_path: str, model_name: str = "smollm2:1.7b"):
        self.rom_path = rom_path
        self.model_name = model_name
        
        # Initialize PyBoy directly for maximum speed
        print("🚀 Initializing Fast Local Trainer...")
        
        if PYBOY_AVAILABLE:
            self.pyboy = PyBoy(
                rom_path,
                window="headless",  # No window for speed
                debug=False
            )
            # Try to get Pokemon game wrapper if available
            try:
                self.pokemon = self.pyboy.game_wrapper
            except:
                self.pokemon = None
            print("✅ PyBoy initialized directly")
        else:
            print("❌ PyBoy not available")
            return
        
        # Action mappings
        self.actions = {
            1: WindowEvent.PRESS_ARROW_UP,
            2: WindowEvent.PRESS_ARROW_DOWN, 
            3: WindowEvent.PRESS_ARROW_LEFT,
            4: WindowEvent.PRESS_ARROW_RIGHT,
            5: WindowEvent.PRESS_BUTTON_A,
            6: WindowEvent.PRESS_BUTTON_B,
            7: WindowEvent.PRESS_BUTTON_START,
            0: None  # No action
        }
        
        # Screen capture optimization
        self.screen_queue = queue.Queue(maxsize=10)
        self.latest_screen = None
        self.capture_thread = None
        self.running = False
        
        # Performance tracking
        self.total_actions = 0
        self.start_time = time.time()
        
        print("✅ Fast trainer ready!")
    
    def start_screen_capture(self):
        """Start optimized screen capture thread"""
        self.running = True
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        print("📸 Screen capture started")
    
    def _capture_loop(self):
        """Optimized screen capture loop"""
        while self.running:
            try:
                # Get screen as numpy array (fastest method)
                try:
                    screen_array = self.pyboy.screen.ndarray
                except AttributeError:
                    # Fallback method for different PyBoy versions
                    screen_array = np.array(self.pyboy.screen.image)
                
                # Convert to PIL Image for easy handling
                screen_pil = Image.fromarray(screen_array)
                
                # Resize for UI efficiency (optional)
                screen_resized = screen_pil.resize((320, 288), Image.NEAREST)
                
                # Convert to base64 for web display
                buffer = io.BytesIO()
                screen_resized.save(buffer, format='PNG', optimize=True)
                screen_b64 = base64.b64encode(buffer.getvalue()).decode()
                
                # Update latest screen
                self.latest_screen = {
                    'image_b64': screen_b64,
                    'timestamp': time.time(),
                    'size': screen_resized.size
                }
                
                # Add to queue (non-blocking)
                if not self.screen_queue.full():
                    self.screen_queue.put(self.latest_screen)
                
                # Limit capture rate (60 FPS max)
                time.sleep(0.016)
                
            except Exception as e:
                print(f"⚠️ Screen capture error: {e}")
                time.sleep(0.1)
    
    def get_latest_screen(self):
        """Get latest screen capture for UI"""
        return self.latest_screen
    
    def make_llm_decision(self, stage: str = "BASIC_CONTROLS") -> int:
        """Fast LLM decision making"""
        
        prompt = f"""Pokemon Crystal - Stage: {stage}
Choose action number:
1=UP, 2=DOWN, 3=LEFT, 4=RIGHT, 5=A, 6=B, 7=START

Action:"""
        
        try:
            start_time = time.time()
            
            response = ollama.generate(
                model=self.model_name,
                prompt=prompt,
                options={
                    'num_predict': 2,
                    'temperature': 0.2,
                    'top_k': 8
                }
            )
            
            inference_time = time.time() - start_time
            
            # Parse action quickly
            action_text = response['response'].strip()
            action = self._parse_action(action_text)
            
            print(f"🧠 LLM: {action_text} -> {action} ({inference_time:.3f}s)")
            return action
            
        except Exception as e:
            print(f"⚠️ LLM error: {e}")
            return 5  # Default to A button
    
    def _parse_action(self, response: str) -> int:
        """Parse LLM response to action number"""
        for char in response:
            if char.isdigit() and '1' <= char <= '7':
                return int(char)
        return 5  # Default to A
    
    def execute_action(self, action: int):
        """Execute action in the game"""
        if action in self.actions and self.actions[action]:
            self.pyboy.send_input(self.actions[action])
            
            # Release button after a frame
            self.pyboy.tick()
            
            # Track performance
            self.total_actions += 1
    
    def run_fast_training(self, max_actions: int = 1000):
        """Run fast training session with real-time capture"""
        
        print(f"\n⚡ STARTING FAST LOCAL TRAINING")
        print("=" * 50)
        print(f"🎯 Target actions: {max_actions}")
        print(f"🤖 Model: {self.model_name}")
        print(f"📸 Real-time screen capture: ON")
        print()
        
        # Start screen capture
        self.start_screen_capture()
        
        # Training loop
        actions_taken = 0
        last_llm_decision = 0
        llm_interval = 10  # LLM decision every N actions
        
        try:
            while actions_taken < max_actions:
                
                # Advance game by one frame
                self.pyboy.tick()
                
                # Make LLM decision periodically
                if actions_taken % llm_interval == 0:
                    action = self.make_llm_decision()
                    last_llm_decision = action
                else:
                    # Use last LLM decision or simple exploration
                    action = last_llm_decision if last_llm_decision else 5
                
                # Execute action
                if action > 0:
                    self.execute_action(action)
                
                actions_taken += 1
                
                # Progress update
                if actions_taken % 50 == 0:
                    elapsed = time.time() - self.start_time
                    aps = actions_taken / elapsed
                    
                    print(f"📊 Progress: {actions_taken}/{max_actions} "
                          f"({aps:.1f} actions/sec)")
                
                # Small delay to prevent overwhelming the system
                time.sleep(0.01)
        
        except KeyboardInterrupt:
            print("\n⏸️ Training interrupted by user")
        
        finally:
            self.stop_training()
        
        # Final stats
        total_time = time.time() - self.start_time
        final_aps = actions_taken / total_time
        
        print(f"\n📊 TRAINING COMPLETED!")
        print(f"⏱️ Duration: {total_time:.1f} seconds")
        print(f"🎯 Actions taken: {actions_taken}")
        print(f"🚀 Average speed: {final_aps:.1f} actions/sec")
        print(f"📸 Screen captures available in real-time!")
    
    def stop_training(self):
        """Stop training and cleanup"""
        self.running = False
        if self.capture_thread:
            self.capture_thread.join(timeout=1)
        if hasattr(self, 'pyboy'):
            self.pyboy.stop()
        print("🛑 Training stopped and cleaned up")
    
    def get_performance_stats(self):
        """Get current performance statistics"""
        elapsed = time.time() - self.start_time
        aps = self.total_actions / max(elapsed, 0.001)
        
        return {
            'total_actions': self.total_actions,
            'elapsed_time': elapsed,
            'actions_per_second': aps,
            'screen_captures': self.screen_queue.qsize(),
            'latest_screen_available': self.latest_screen is not None
        }


def create_simple_web_server(trainer: FastPokemonTrainer, port: int = 8080):
    """Create simple web server to display training progress"""
    from http.server import HTTPServer, BaseHTTPRequestHandler
    import json
    
    class TrainingHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == '/':
                # Serve simple HTML page
                html = f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Pokemon Crystal Fast Training</title>
                    <style>
                        body {{ font-family: Arial; margin: 20px; background: #1a1a1a; color: white; }}
                        .container {{ max-width: 800px; margin: 0 auto; }}
                        .screen {{ border: 2px solid #4CAF50; margin: 20px 0; }}
                        .stats {{ background: #333; padding: 15px; border-radius: 5px; }}
                        img {{ max-width: 100%; height: auto; }}
                    </style>
                </head>
                <body>
                    <div class="container">
                        <h1>⚡ Pokemon Crystal Fast Training</h1>
                        <div class="stats" id="stats">Loading...</div>
                        <div class="screen">
                            <img id="gameScreen" src="/screen" alt="Game Screen">
                        </div>
                    </div>
                    <script>
                        function updateStats() {{
                            fetch('/stats').then(r => r.json()).then(data => {{
                                document.getElementById('stats').innerHTML = 
                                    `🎯 Actions: ${{data.total_actions}} | ` +
                                    `⚡ Speed: ${{data.actions_per_second.toFixed(1)}} a/s | ` +
                                    `⏱️ Time: ${{data.elapsed_time.toFixed(1)}}s`;
                            }});
                        }}
                        
                        function updateScreen() {{
                            document.getElementById('gameScreen').src = '/screen?' + Date.now();
                        }}
                        
                        setInterval(updateStats, 1000);
                        setInterval(updateScreen, 100);  // 10 FPS refresh
                        updateStats();
                    </script>
                </body>
                </html>
                """
                
                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                self.wfile.write(html.encode())
                
            elif self.path.startswith('/screen'):
                # Serve latest screen capture
                screen = trainer.get_latest_screen()
                if screen:
                    img_data = base64.b64decode(screen['image_b64'])
                    self.send_response(200)
                    self.send_header('Content-type', 'image/png')
                    self.end_headers()
                    self.wfile.write(img_data)
                else:
                    self.send_response(404)
                    self.end_headers()
                    
            elif self.path == '/stats':
                # Serve performance stats
                stats = trainer.get_performance_stats()
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(stats).encode())
                
            else:
                self.send_response(404)
                self.end_headers()
    
    server = HTTPServer(('localhost', port), TrainingHandler)
    print(f"🌐 Web interface available at http://localhost:{port}")
    return server


def main():
    """Main function for fast local training"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Fast Local Pokemon Crystal Training')
    parser.add_argument('--rom', required=True, help='Path to Pokemon Crystal ROM')
    parser.add_argument('--actions', type=int, default=500, help='Number of actions to perform')
    parser.add_argument('--model', default='smollm2:1.7b', help='LLM model to use')
    parser.add_argument('--web', action='store_true', help='Start web interface')
    parser.add_argument('--port', type=int, default=8080, help='Web interface port')
    
    args = parser.parse_args()
    
    print("⚡ FAST LOCAL POKEMON CRYSTAL TRAINING")
    print("=" * 50)
    print(f"🎮 ROM: {args.rom}")
    print(f"🤖 Model: {args.model}")
    print(f"🎯 Actions: {args.actions}")
    print(f"🌐 Web UI: {'ON' if args.web else 'OFF'}")
    print()
    
    # Create trainer
    trainer = FastPokemonTrainer(args.rom, args.model)
    
    # Start web server if requested
    web_server = None
    if args.web:
        web_server = create_simple_web_server(trainer, args.port)
        web_thread = threading.Thread(target=web_server.serve_forever, daemon=True)
        web_thread.start()
    
    try:
        # Run training
        trainer.run_fast_training(max_actions=args.actions)
        
    except KeyboardInterrupt:
        print("\\n⏸️ Interrupted by user")
        
    finally:
        trainer.stop_training()
        if web_server:
            web_server.shutdown()


if __name__ == "__main__":
    main()
