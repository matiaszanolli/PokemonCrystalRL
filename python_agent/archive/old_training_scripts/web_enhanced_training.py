"""
web_enhanced_training.py - Pokemon Crystal RL Training with Web Monitoring

This combines the vision-enhanced training with real-time web monitoring
for an optimal training and debugging experience.
"""

import time
import threading
import signal
import sys
from typing import Optional
import numpy as np

from vision_enhanced_training import VisionEnhancedTrainingSession
from web_monitor import PokemonRLWebMonitor, create_dashboard_templates


class WebEnhancedTrainingSession:
    """
    Enhanced training session with integrated web monitoring
    """
    
    def __init__(self, rom_path: str = "pokecrystal.gbc", episodes: int = 100):
        self.rom_path = rom_path
        self.episodes = episodes
        
        # Create training session
        print("🎮 Initializing training session...")
        self.training_session = VisionEnhancedTrainingSession(
            rom_path=rom_path,
            save_state_path=None,  # No save state for now
            max_steps_per_episode=1000  # Reasonable default
        )
        
        # Create web monitor
        print("🌐 Initializing web monitor...")
        self.web_monitor = PokemonRLWebMonitor(self.training_session)
        
        # Monitoring state
        self.training_active = False
        self.web_server_thread = None
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        print("✅ Web-enhanced training session ready!")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        print(f"\n🛑 Received signal {signum}, shutting down gracefully...")
        self.stop()
        sys.exit(0)
    
    def start_web_server(self, host='127.0.0.1', port=5000):
        """Start the web monitoring server in a separate thread"""
        print(f"🚀 Starting web server on {host}:{port}")
        
        # Create templates if they don't exist
        create_dashboard_templates()
        
        # Start web server in separate thread
        def run_web_server():
            self.web_monitor.run(host=host, port=port, debug=False)
        
        self.web_server_thread = threading.Thread(target=run_web_server, daemon=True)
        self.web_server_thread.start()
        
        # Give the server a moment to start
        time.sleep(2)
        
        print(f"📊 Dashboard available at: http://{host}:{port}")
        print("🎯 Click 'Start Monitor' in the web interface to begin monitoring")
    
    def train_with_monitoring(self):
        """Run training with integrated web monitoring"""
        print("🎯 Starting training with web monitoring...")
        
        self.training_active = True
        
        try:
            # Hook into the training session to send updates to web monitor
            self._setup_training_hooks()
            
            # Start the actual training
            print("▶️ Beginning training episodes...")
            self.training_session.run_training_session(num_episodes=self.episodes)
            
        except KeyboardInterrupt:
            print("\n⏸️ Training interrupted by user")
        except Exception as e:
            print(f"❌ Training error: {e}")
            raise
        finally:
            self.training_active = False
            print("🏁 Training session completed")
    
    def _setup_training_hooks(self):
        """Setup hooks to send data to web monitor"""
        
        # Hook into the environment step method to capture data
        original_env_step = self.training_session.env.step
        
        def monitored_env_step(action):
            # Call original step
            result = original_env_step(action)
            
            try:
                # Get current screenshot
                screenshot = self.training_session.env.get_screenshot()
                if screenshot is not None and screenshot.size > 0:
                    self.web_monitor.update_screenshot(screenshot)
                
                # Update action history with action name
                action_names = {0: "NO-OP", 1: "UP", 2: "DOWN", 3: "LEFT", 4: "RIGHT", 
                              5: "A", 6: "B", 7: "START", 8: "SELECT"}
                action_name = action_names.get(action, f"Action_{action}")
                self.web_monitor.update_action(action_name)
                
                # Add performance metrics
                if len(result) >= 2:  # Has reward
                    reward = result[1] if len(result) == 4 else result[1]  # Handle both API versions
                    self.web_monitor.add_performance_metric('reward', reward)
                
            except Exception as e:
                print(f"⚠️ Monitoring hook error: {e}")
            
            return result
        
        # Replace the environment step method
        self.training_session.env.step = monitored_env_step
        
        # Hook into agent decisions
        if hasattr(self.training_session, 'agent'):
            original_decide = self.training_session.agent.decide_next_action
            
            def monitored_decide(state, screenshot=None, recent_history=None):
                # Call original decision method
                action = original_decide(state, screenshot, recent_history)
                
                try:
                    # Create decision data for web monitor
                    decision_data = {
                        'decision': f"Action: {action}",
                        'reasoning': f"LLM chose action {action} based on current state",
                        'confidence': 0.8,  # Default confidence
                        'visual_context': {
                            'screen_type': 'gameplay',
                            'detected_text': []
                        }
                    }
                    self.web_monitor.update_decision(decision_data)
                    
                except Exception as e:
                    print(f"⚠️ Decision monitoring error: {e}")
                
                return action
            
            self.training_session.agent.decide_next_action = monitored_decide
    
    def stop(self):
        """Stop training and monitoring"""
        print("🛑 Stopping training and monitoring...")
        
        self.training_active = False
        
        if hasattr(self.training_session, 'stop'):
            self.training_session.stop()
        
        if self.web_monitor:
            self.web_monitor.stop_monitoring()
        
        print("✅ Training and monitoring stopped")
    
    def run_interactive(self):
        """Run in interactive mode with web monitoring"""
        print("🎮 Pokemon Crystal RL - Web Enhanced Training")
        print("=" * 50)
        print()
        
        # Start web server
        self.start_web_server()
        
        print("🌐 Web monitoring dashboard started!")
        print()
        print("📋 Instructions:")
        print("1. Open http://127.0.0.1:5000 in your browser")
        print("2. Click 'Start Monitor' to begin real-time monitoring")
        print("3. Training will start automatically with live updates")
        print()
        print("Press Ctrl+C at any time to stop training gracefully")
        print("=" * 50)
        print()
        
        # Wait a moment for user to open browser
        print("⏳ Waiting 10 seconds for you to open the web dashboard...")
        time.sleep(10)
        
        # Start training
        self.train_with_monitoring()


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Pokemon Crystal RL with Web Monitoring")
    parser.add_argument("--rom", default="pokecrystal.gbc", help="Path to Pokemon Crystal ROM")
    parser.add_argument("--episodes", type=int, default=100, help="Number of training episodes")
    parser.add_argument("--host", default="127.0.0.1", help="Web server host")
    parser.add_argument("--port", type=int, default=5000, help="Web server port")
    parser.add_argument("--web-only", action="store_true", help="Start web server only (no training)")
    
    args = parser.parse_args()
    
    if args.web_only:
        # Start web server only for monitoring existing data
        print("🌐 Starting web monitor only (no training)")
        create_dashboard_templates()
        monitor = PokemonRLWebMonitor()
        monitor.run(host=args.host, port=args.port, debug=False)
    else:
        # Start full training with web monitoring
        session = WebEnhancedTrainingSession(args.rom, args.episodes)
        session.run_interactive()


if __name__ == "__main__":
    main()
