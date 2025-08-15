#!/usr/bin/env python3
"""
launch_training.py - Comprehensive Pokemon Crystal RL Training Launcher

This script provides a unified way to start Pokemon Crystal RL training with:
- Automatic monitoring server startup
- Environment validation
- Configuration management
- Graceful coordination between components
- Easy command-line interface
"""

import os
import sys
import time
import signal
import subprocess
import threading
import argparse
import requests
from typing import Optional, List


class PokemonRLLauncher:
    """
    Unified launcher for Pokemon Crystal RL training with monitoring
    """
    
    def __init__(self):
        self.monitoring_process = None
        self.training_process = None
        self.monitor_port = 5000
        self.monitor_host = "127.0.0.1"
        self.monitor_url = f"http://{self.monitor_host}:{self.monitor_port}"
        
        # Signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        print(f"\n🛑 Received signal {signum}, shutting down...")
        self.stop_all()
        sys.exit(0)
    
    def validate_environment(self, rom_path: str, save_state_path: Optional[str] = None) -> bool:
        """Validate the training environment setup"""
        print("🔍 Validating environment...")
        
        # Check ROM file
        if not os.path.exists(rom_path):
            print(f"❌ ROM file not found: {rom_path}")
            return False
        print(f"✓ ROM file found: {rom_path}")
        
        # Check save state if provided
        if save_state_path:
            if not os.path.exists(save_state_path):
                print(f"⚠️ Save state file not found: {save_state_path}")
                print("  Training will start from the beginning")
            else:
                print(f"✓ Save state found: {save_state_path}")
        
        # Check Python dependencies
        required_packages = [
            ('pyboy', 'pyboy'),
            ('ollama', 'ollama'), 
            ('numpy', 'numpy'),
            ('opencv-python', 'cv2'),
            ('pillow', 'PIL'),
            ('flask', 'flask'),
            ('flask-socketio', 'flask_socketio'),
            ('requests', 'requests')
        ]
        
        missing_packages = []
        for package_name, import_name in required_packages:
            try:
                __import__(import_name)
                print(f"✓ {package_name}")
            except ImportError:
                missing_packages.append(package_name)
                print(f"❌ {package_name}")
        
        if missing_packages:
            print(f"\n❌ Missing required packages: {', '.join(missing_packages)}")
            print("Install them with: pip install " + " ".join(missing_packages))
            return False
        
        # Check Ollama service
        try:
            import ollama
            # Try to list models to check if Ollama is running
            models = ollama.list()
            print("✓ Ollama service is running")
            
            # Check for required model
            try:
                model_names = [model.get('name', '') for model in models.get('models', [])]
                if any('llama3.2:3b' in name for name in model_names):
                    print("✓ Required model (llama3.2:3b) is available")
                else:
                    print("⚠️ Required model (llama3.2:3b) not found")
                    print("  Available models:", model_names[:3])
                    print("  Download it with: ollama pull llama3.2:3b")
                    # Don't fail validation for missing model - just warn
                    print("⚠️ Continuing without LLM model validation")
            except Exception as model_check_error:
                print(f"⚠️ Could not check models: {model_check_error}")
                print("⚠️ Continuing without LLM model validation")
                
        except Exception as e:
            print(f"❌ Ollama service not available: {e}")
            print("  Please install and start Ollama first")
            return False
        
        print("✅ Environment validation passed!")
        return True
    
    def start_monitoring_server(self) -> bool:
        """Start the advanced monitoring server"""
        print(f"🚀 Starting monitoring server on {self.monitor_url}...")
        
        # Check if server is already running
        if self._check_server_running():
            print("✓ Monitoring server is already running")
            return True
        
        try:
            # Start the monitoring server
            cmd = [sys.executable, "advanced_web_monitor.py"]
            self.monitoring_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=dict(os.environ, FLASK_ENV="production")
            )
            
            # Wait for server to start
            for attempt in range(10):
                time.sleep(1)
                if self._check_server_running():
                    print(f"✅ Monitoring server started on {self.monitor_url}")
                    return True
                print(f"⏳ Waiting for server to start... ({attempt + 1}/10)")
            
            print("❌ Failed to start monitoring server")
            if self.monitoring_process:
                self.monitoring_process.terminate()
                self.monitoring_process = None
            return False
            
        except Exception as e:
            print(f"❌ Failed to start monitoring server: {e}")
            return False
    
    def _check_server_running(self) -> bool:
        """Check if the monitoring server is running"""
        try:
            response = requests.get(f"{self.monitor_url}/api/status", timeout=2)
            return response.status_code == 200
        except:
            return False
    
    def start_training(self, **kwargs) -> bool:
        """Start the training process"""
        print("🎮 Starting Pokemon Crystal RL training...")
        
        # Build training command
        cmd = [sys.executable, "enhanced_monitored_training.py"]
        
        # Add arguments
        if kwargs.get('rom'):
            cmd.extend(['--rom', kwargs['rom']])
        if kwargs.get('save_state'):
            cmd.extend(['--save-state', kwargs['save_state']])
        if kwargs.get('episodes'):
            cmd.extend(['--episodes', str(kwargs['episodes'])])
        if kwargs.get('max_steps'):
            cmd.extend(['--max-steps', str(kwargs['max_steps'])])
        if kwargs.get('llm_interval'):
            cmd.extend(['--llm-interval', str(kwargs['llm_interval'])])
        if kwargs.get('visual_interval'):
            cmd.extend(['--visual-interval', str(kwargs['visual_interval'])])
        if kwargs.get('screenshot_interval'):
            cmd.extend(['--screenshot-interval', str(kwargs['screenshot_interval'])])
        if kwargs.get('no_vision'):
            cmd.append('--no-vision')
        if kwargs.get('no_monitoring'):
            cmd.append('--no-monitoring')
        
        cmd.extend(['--monitor-url', self.monitor_url])
        
        print(f"📋 Training command: {' '.join(cmd)}")
        
        try:
            # Start training
            self.training_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            # Monitor training output
            def monitor_training():
                for line in iter(self.training_process.stdout.readline, ''):
                    print(line.rstrip())
                self.training_process.stdout.close()
                return_code = self.training_process.wait()
                print(f"\n🏁 Training completed with exit code: {return_code}")
            
            # Start monitoring thread
            monitor_thread = threading.Thread(target=monitor_training, daemon=True)
            monitor_thread.start()
            
            print("✅ Training started successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Failed to start training: {e}")
            return False
    
    def wait_for_completion(self):
        """Wait for training to complete"""
        if self.training_process:
            try:
                print("⏳ Training in progress...")
                print(f"🌐 Monitor dashboard: {self.monitor_url}")
                print("🛑 Press Ctrl+C to stop training\n")
                
                # Wait for training to complete
                self.training_process.wait()
                
            except KeyboardInterrupt:
                print("\n🛑 Training interrupted by user")
            finally:
                self.stop_all()
    
    def stop_all(self):
        """Stop all processes"""
        print("🔄 Stopping all processes...")
        
        # Stop training
        if self.training_process:
            print("  Stopping training process...")
            self.training_process.terminate()
            try:
                self.training_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.training_process.kill()
            self.training_process = None
        
        # Stop monitoring server
        if self.monitoring_process:
            print("  Stopping monitoring server...")
            self.monitoring_process.terminate()
            try:
                self.monitoring_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.monitoring_process.kill()
            self.monitoring_process = None
        
        print("✅ All processes stopped")
    
    def launch_full_training(self, **kwargs) -> bool:
        """Launch complete training session with monitoring"""
        print("🚀 Launching Pokemon Crystal RL Training Session")
        print("=" * 50)
        
        # Validate environment
        if not self.validate_environment(kwargs.get('rom', 'pokecrystal.gbc'), 
                                       kwargs.get('save_state')):
            return False
        
        print()
        
        # Start monitoring server
        if not kwargs.get('no_monitoring', False):
            if not self.start_monitoring_server():
                print("⚠️ Continuing without monitoring server")
                kwargs['no_monitoring'] = True
        
        print()
        
        # Start training
        if not self.start_training(**kwargs):
            self.stop_all()
            return False
        
        # Wait for completion
        self.wait_for_completion()
        
        return True


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Pokemon Crystal RL Training Launcher',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --rom pokecrystal.gbc --episodes 50
  %(prog)s --rom pokecrystal.gbc --save-state save.ss1 --episodes 100
  %(prog)s --rom pokecrystal.gbc --no-vision --llm-interval 5
        """
    )
    
    # Required arguments
    parser.add_argument('--rom', type=str, default='pokecrystal.gbc',
                       help='Path to Pokemon Crystal ROM file (default: pokecrystal.gbc)')
    
    # Optional arguments
    parser.add_argument('--save-state', type=str, default=None,
                       help='Path to save state file (optional)')
    parser.add_argument('--episodes', type=int, default=50,
                       help='Number of episodes to run (default: 50)')
    parser.add_argument('--max-steps', type=int, default=2000,
                       help='Maximum steps per episode (default: 2000)')
    parser.add_argument('--llm-interval', type=int, default=10,
                       help='LLM decision interval in steps (default: 10)')
    parser.add_argument('--visual-interval', type=int, default=20,
                       help='Visual analysis interval in steps (default: 20)')
    parser.add_argument('--screenshot-interval', type=int, default=5,
                       help='Screenshot update interval in steps (default: 5)')
    
    # Feature toggles
    parser.add_argument('--no-vision', action='store_true',
                       help='Disable computer vision processing')
    parser.add_argument('--no-monitoring', action='store_true',
                       help='Disable web monitoring dashboard')
    
    # Commands
    parser.add_argument('--validate-only', action='store_true',
                       help='Only validate environment, don\'t start training')
    parser.add_argument('--monitor-only', action='store_true',
                       help='Only start monitoring server, don\'t start training')
    
    args = parser.parse_args()
    
    # Create launcher
    launcher = PokemonRLLauncher()
    
    try:
        if args.validate_only:
            # Only validate environment
            success = launcher.validate_environment(args.rom, args.save_state)
            return 0 if success else 1
        
        elif args.monitor_only:
            # Only start monitoring server
            if launcher.start_monitoring_server():
                print(f"🌐 Monitoring dashboard: {launcher.monitor_url}")
                print("🛑 Press Ctrl+C to stop")
                try:
                    while True:
                        time.sleep(1)
                except KeyboardInterrupt:
                    pass
                launcher.stop_all()
                return 0
            else:
                return 1
        
        else:
            # Full training launch
            success = launcher.launch_full_training(
                rom=args.rom,
                save_state=args.save_state,
                episodes=args.episodes,
                max_steps=args.max_steps,
                llm_interval=args.llm_interval,
                visual_interval=args.visual_interval,
                screenshot_interval=args.screenshot_interval,
                no_vision=args.no_vision,
                no_monitoring=args.no_monitoring
            )
            return 0 if success else 1
    
    except KeyboardInterrupt:
        print("\n🛑 Launcher interrupted by user")
        launcher.stop_all()
        return 0
    except Exception as e:
        print(f"\n❌ Launcher failed: {e}")
        import traceback
        traceback.print_exc()
        launcher.stop_all()
        return 1


if __name__ == "__main__":
    exit(main())
