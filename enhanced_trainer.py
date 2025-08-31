#!/usr/bin/env python3
"""
Enhanced Pokemon Crystal RL Training Script with Web Monitoring

An enhanced training script with web monitoring, screenshot capture,
and save state support for Pokemon Crystal training.
"""

import time
import numpy as np
from pyboy import PyBoy
import json
import signal
import sys
import os
import threading
import base64
import io
from datetime import datetime
from PIL import Image
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse

class WebMonitor(BaseHTTPRequestHandler):
    """Simple web server for monitoring training"""
    
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            
            html = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Pokemon Crystal RL Training Monitor</title>
                <meta http-equiv="refresh" content="2">
                <style>
                    body { font-family: Arial; margin: 20px; background: #1a1a1a; color: #fff; }
                    .header { background: #2a2a2a; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
                    .stats { display: flex; gap: 20px; flex-wrap: wrap; margin-bottom: 20px; }
                    .stat-box { background: #333; padding: 15px; border-radius: 8px; min-width: 150px; }
                    .stat-value { font-size: 24px; font-weight: bold; color: #4CAF50; }
                    .stat-label { color: #aaa; font-size: 12px; }
                    .screen { text-align: center; background: #2a2a2a; padding: 20px; border-radius: 8px; }
                    .game-screen { border: 2px solid #4CAF50; border-radius: 4px; }
                    .actions { background: #2a2a2a; padding: 15px; border-radius: 8px; margin-top: 20px; }
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>🎮 Pokemon Crystal RL Training Monitor</h1>
                    <p>Real-time monitoring of Pokemon Crystal reinforcement learning training session</p>
                </div>
                
                <div class="stats">
                    <div class="stat-box">
                        <div class="stat-value" id="actions">-</div>
                        <div class="stat-label">ACTIONS</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-value" id="aps">-</div>
                        <div class="stat-label">ACTIONS/SEC</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-value" id="time">-</div>
                        <div class="stat-label">TRAINING TIME</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-value" id="state">-</div>
                        <div class="stat-label">GAME STATE</div>
                    </div>
                </div>
                
                <div class="screen">
                    <h3>🖼️ Live Game Screen</h3>
                    <img id="gameScreen" class="game-screen" src="/screenshot" width="320" height="288" alt="Game Screen">
                </div>
                
                <div class="actions">
                    <h3>⚡ Recent Actions</h3>
                    <div id="recent-actions">Loading...</div>
                </div>
                
                <script>
                    async function updateStats() {
                        try {
                            const response = await fetch('/stats');
                            const stats = await response.json();
                            
                            document.getElementById('actions').textContent = stats.actions_taken || '-';
                            document.getElementById('aps').textContent = (stats.actions_per_second || 0).toFixed(1);
                            document.getElementById('time').textContent = (stats.training_time || 0).toFixed(1) + 's';
                            document.getElementById('state').textContent = stats.current_state || '-';
                            
                            const actionsDiv = document.getElementById('recent-actions');
                            if (stats.recent_actions) {
                                actionsDiv.innerHTML = stats.recent_actions.join(' → ');
                            }
                        } catch (e) {
                            console.error('Failed to update stats:', e);
                        }
                    }
                    
                    setInterval(updateStats, 1000);
                    updateStats();
                </script>
            </body>
            </html>
            """
            self.wfile.write(html.encode())
            
        elif self.path == '/stats':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            stats = getattr(self.server, 'trainer_stats', {})
            self.wfile.write(json.dumps(stats).encode())
            
        elif self.path == '/screenshot':
            self.send_response(200)
            self.send_header('Content-type', 'image/png')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            screenshot_data = getattr(self.server, 'screenshot_data', None)
            if screenshot_data:
                self.wfile.write(screenshot_data)
            else:
                # Send empty 1x1 PNG if no screenshot
                empty_img = Image.new('RGB', (1, 1), (0, 0, 0))
                buf = io.BytesIO()
                empty_img.save(buf, format='PNG')
                self.wfile.write(buf.getvalue())
        
        else:
            self.send_response(404)
            self.end_headers()
    
    def log_message(self, format, *args):
        pass  # Suppress HTTP server logs

class EnhancedPokemonTrainer:
    def __init__(self, rom_path, max_actions=5000, enable_web=True, web_port=8080):
        self.rom_path = rom_path
        self.max_actions = max_actions
        self.enable_web = enable_web
        self.web_port = web_port
        self.pyboy = None
        self.actions_taken = 0
        self.start_time = time.time()
        
        self.stats = {
            'actions_taken': 0,
            'frames_processed': 0,
            'training_time': 0,
            'actions_per_second': 0,
            'current_state': 'Initializing',
            'recent_actions': [],
            'start_time': datetime.now().isoformat()
        }
        
        self.running = True
        self.recent_actions = []
        
        # Web server setup
        self.web_server = None
        self.web_thread = None
        
        # Setup graceful shutdown
        signal.signal(signal.SIGINT, self.shutdown)
        signal.signal(signal.SIGTERM, self.shutdown)
        
    def shutdown(self, signum, frame):
        """Graceful shutdown handler"""
        print(f"\n⏸️ Shutting down training...")
        self.running = False
        
        if self.web_server:
            print("Stopping web server...")
            self.web_server.shutdown()
            
        if self.pyboy:
            self.pyboy.stop()
            
        self.save_stats()
        print("✅ Training stopped cleanly")
        sys.exit(0)
    
    def setup_web_server(self):
        """Setup web monitoring server"""
        if not self.enable_web:
            return
            
        try:
            self.web_server = HTTPServer(('localhost', self.web_port), WebMonitor)
            self.web_server.trainer_stats = self.stats
            self.web_server.screenshot_data = None
            
            self.web_thread = threading.Thread(target=self.web_server.serve_forever, daemon=True)
            self.web_thread.start()
            
            print(f"🌐 Web monitor started: http://localhost:{self.web_port}")
            
        except Exception as e:
            print(f"⚠️ Failed to start web server: {e}")
            self.enable_web = False
    
    def initialize_pyboy(self):
        """Initialize PyBoy emulator"""
        print(f"🎮 Initializing PyBoy with {self.rom_path}")
        self.pyboy = PyBoy(self.rom_path, window="null", debug=False)
        
        # Load save state if available
        save_state_path = self.rom_path + '.state'
        if os.path.exists(save_state_path):
            print(f"💾 Loading save state: {save_state_path}")
            with open(save_state_path, 'rb') as f:
                self.pyboy.load_state(f)
            print("✅ Save state loaded - starting from saved position")
        else:
            print("⚠️ No save state found - starting from beginning")
            
        print("✅ PyBoy initialized successfully")
        
    def skip_intro_if_needed(self):
        """Skip intro if we're at the beginning"""
        self.stats['current_state'] = 'Checking game state'
        
        # Check if we need to skip intro
        screen = self.pyboy.screen.ndarray
        variance = np.var(screen.astype(np.float32))
        
        if variance < 1000:  # Likely in intro/menu
            print("⚡ Skipping intro sequence...")
            self.stats['current_state'] = 'Skipping intro'
            
            for i in range(200):
                if not self.running:
                    break
                    
                # Skip intro with START and A buttons
                if i % 15 == 0:
                    self.pyboy.button_press('start')
                    for _ in range(3):
                        self.pyboy.tick()
                    self.pyboy.button_release('start')
                elif i % 15 == 7:
                    self.pyboy.button_press('a')
                    for _ in range(3):
                        self.pyboy.tick()
                    self.pyboy.button_release('a')
                else:
                    self.pyboy.tick()
                
                # Check if we've reached gameplay
                if i % 40 == 0:
                    screen = self.pyboy.screen.ndarray
                    variance = np.var(screen.astype(np.float32))
                    if variance > 2000:
                        print(f"✅ Reached gameplay (variance: {variance:.1f})")
                        return True
        
        print("✅ Game ready for training")
        return True
    
    def get_enhanced_action(self):
        """Get next action using enhanced rule-based system"""
        # More sophisticated action pattern
        action_patterns = {
            'exploration': ['up', 'up', 'a', 'right', 'right', 'a', 'down', 'down', 'a', 'left', 'left', 'a'],
            'interaction': ['a', 'a', 'b', 'start', 'a', 'b'],
            'menu_navigation': ['up', 'down', 'a', 'b', 'start', 'select']
        }
        
        # Choose pattern based on action count
        cycle_length = 30
        pattern_type = ['exploration', 'interaction', 'menu_navigation'][
            (self.actions_taken // cycle_length) % 3
        ]
        
        pattern = action_patterns[pattern_type]
        return pattern[self.actions_taken % len(pattern)]
    
    def execute_action(self, action):
        """Execute an action and update monitoring"""
        if not self.running:
            return
            
        # Press button
        self.pyboy.button_press(action)
        
        # Hold for frames
        for _ in range(8):
            if not self.running:
                break
            self.pyboy.tick()
            self.stats['frames_processed'] += 1
        
        # Release button
        self.pyboy.button_release(action)
        
        # Wait frames
        for _ in range(4):
            if not self.running:
                break
            self.pyboy.tick()
            self.stats['frames_processed'] += 1
        
        self.actions_taken += 1
        self.stats['actions_taken'] = self.actions_taken
        
        # Update recent actions
        self.recent_actions.append(action.upper())
        if len(self.recent_actions) > 10:
            self.recent_actions.pop(0)
        self.stats['recent_actions'] = self.recent_actions.copy()
    
    def analyze_screen(self):
        """Analyze current screen"""
        screen = self.pyboy.screen.ndarray
        variance = np.var(screen.astype(np.float32))
        
        if variance < 100:
            state = "loading"
        elif variance < 2000:
            state = "menu"
        elif variance < 8000:
            state = "dialogue"
        elif variance < 15000:
            state = "overworld"
        else:
            state = "battle"
            
        return {
            'variance': variance,
            'state': state,
            'colors': len(np.unique(screen.reshape(-1, screen.shape[-1]), axis=0))
        }
    
    def update_web_data(self):
        """Update data for web monitoring"""
        if not self.enable_web or not self.web_server:
            return
            
        # Update stats
        elapsed = time.time() - self.start_time
        self.stats['training_time'] = elapsed
        self.stats['actions_per_second'] = self.actions_taken / elapsed if elapsed > 0 else 0
        
        screen_analysis = self.analyze_screen()
        self.stats['current_state'] = screen_analysis['state']
        
        # Update screenshot
        try:
            screen = self.pyboy.screen.ndarray
            # Convert RGBA to RGB
            if screen.shape[2] == 4:
                screen = screen[:, :, :3]
            
            # Resize for better web display
            img = Image.fromarray(screen)
            img = img.resize((320, 288), Image.NEAREST)
            
            # Convert to PNG bytes
            buf = io.BytesIO()
            img.save(buf, format='PNG', optimize=True)
            self.web_server.screenshot_data = buf.getvalue()
            
        except Exception as e:
            print(f"Screenshot update failed: {e}")
        
        # Update server stats
        self.web_server.trainer_stats = self.stats.copy()
    
    def print_progress(self):
        """Print training progress"""
        elapsed = time.time() - self.start_time
        aps = self.actions_taken / elapsed if elapsed > 0 else 0
        
        screen_analysis = self.analyze_screen()
        recent = ' → '.join(self.recent_actions[-5:]) if self.recent_actions else 'None'
        
        print(f"⚡ Action {self.actions_taken}/{self.max_actions} "
              f"| {aps:.1f} a/s "
              f"| State: {screen_analysis['state']} "
              f"| Recent: {recent}")
    
    def save_stats(self):
        """Save training statistics"""
        stats_file = f"enhanced_training_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        final_stats = self.stats.copy()
        
        # Convert numpy types to native Python types for JSON serialization
        screen_analysis = self.analyze_screen()
        final_stats['final_screen_analysis'] = {
            'variance': float(screen_analysis['variance']),
            'state': screen_analysis['state'],
            'colors': int(screen_analysis['colors'])
        }
        
        with open(stats_file, 'w') as f:
            json.dump(final_stats, f, indent=2)
        print(f"📊 Stats saved to {stats_file}")
    
    def start_training(self):
        """Start the enhanced training process"""
        print("🚀 Starting Enhanced Pokemon Crystal RL Training")
        print("=" * 70)
        
        try:
            # Setup web server
            self.setup_web_server()
            
            # Initialize emulator
            self.initialize_pyboy()
            
            # Skip intro if needed
            self.skip_intro_if_needed()
            
            print(f"\n🎯 Starting enhanced training loop ({self.max_actions} actions)")
            if self.enable_web:
                print(f"🌐 Monitor at: http://localhost:{self.web_port}")
            print("🔄 Press Ctrl+C to stop training gracefully")
            print()
            
            # Main training loop
            while self.running and self.actions_taken < self.max_actions:
                # Get next action
                action = self.get_enhanced_action()
                
                # Execute action
                self.execute_action(action)
                
                # Update web monitoring
                if self.actions_taken % 5 == 0:  # Update web every 5 actions
                    self.update_web_data()
                
                # Print progress every 100 actions
                if self.actions_taken % 100 == 0:
                    self.print_progress()
            
            # Training completed
            if self.actions_taken >= self.max_actions:
                print(f"\n✅ Enhanced training completed! {self.actions_taken} actions executed")
                
            return True
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            import traceback
            traceback.print_exc()
            return False
            
        finally:
            # Cleanup
            if self.web_server:
                try:
                    self.web_server.shutdown()
                except:
                    pass
                    
            if self.pyboy:
                self.pyboy.stop()
            self.save_stats()

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced Pokemon Crystal RL Training with Web Monitor")
    parser.add_argument("--rom", default="roms/pokemon_crystal.gbc", help="ROM file path")
    parser.add_argument("--actions", type=int, default=5000, help="Number of actions to execute")
    parser.add_argument("--web-port", type=int, default=8080, help="Web monitoring port")
    parser.add_argument("--no-web", action="store_true", help="Disable web monitoring")
    
    args = parser.parse_args()
    
    # Validate ROM file
    if not os.path.exists(args.rom):
        print(f"❌ ROM file not found: {args.rom}")
        return 1
    
    # Create and start trainer
    trainer = EnhancedPokemonTrainer(
        args.rom, 
        args.actions, 
        enable_web=not args.no_web,
        web_port=args.web_port
    )
    success = trainer.start_training()
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
