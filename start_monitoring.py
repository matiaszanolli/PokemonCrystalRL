#!/usr/bin/env python3
"""
start_monitoring.py - Launch the complete Pokémon Crystal RL training monitoring system
"""

import os
import sys
import subprocess
import time
import signal
import threading
import webbrowser
from pathlib import Path

def check_dependencies():
    """Check if all required dependencies are installed"""
    print("🔍 Checking dependencies...")
    
    try:
        import flask
        import plotly
        import psutil
        import stable_baselines3
        import gymnasium
        import torch
        import numpy
        print("✅ All Python dependencies are installed")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Run: pip install -r requirements.txt")
        return False
    
    # Check if BizHawk is installed
    bizhawk_path = '/usr/local/bin/bizhawk'
    if not os.path.exists(bizhawk_path):
        print(f"❌ BizHawk not found at {bizhawk_path}")
        print("Please install BizHawk using the instructions in README.md")
        return False
    
    print("✅ BizHawk is installed")
    
    # Check if ROM exists
    rom_path = 'pokecrystal.gbc'
    if not os.path.exists(rom_path):
        print(f"❌ ROM file not found: {rom_path}")
        print("Please place your Pokémon Crystal ROM file in the project directory")
        return False
    
    print("✅ ROM file found")
    return True

def create_directories():
    """Create necessary directories"""
    directories = [
        'models',
        'logs',
        'python_agent/models',
        'python_agent/logs',
        'templates',
        'static'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("✅ Created necessary directories")

def start_tensorboard(port=6006):
    """Start TensorBoard server"""
    print(f"🚀 Starting TensorBoard on port {port}...")
    
    cmd = ['tensorboard', '--logdir', 'python_agent/logs', '--port', str(port), '--host', '0.0.0.0']
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Give TensorBoard a moment to start
        time.sleep(3)
        
        if process.poll() is None:
            print(f"✅ TensorBoard started successfully")
            print(f"📊 TensorBoard available at: http://localhost:{port}")
            return process
        else:
            print("❌ Failed to start TensorBoard")
            return None
    except FileNotFoundError:
        print("❌ TensorBoard not found. Install with: pip install tensorboard")
        return None

def start_monitoring_dashboard(port=5000):
    """Start the monitoring dashboard"""
    print(f"🚀 Starting monitoring dashboard on port {port}...")
    
    # Import and run the monitor
    try:
        from monitor import app, monitor
        
        # Start the Flask app in a separate thread
        def run_app():
            app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
        
        app_thread = threading.Thread(target=run_app, daemon=True)
        app_thread.start()
        
        # Give Flask a moment to start
        time.sleep(2)
        
        print(f"✅ Monitoring dashboard started successfully")
        print(f"🎮 Dashboard available at: http://localhost:{port}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to start monitoring dashboard: {e}")
        return False

def open_browser_tabs():
    """Open browser tabs for dashboard and TensorBoard"""
    print("🌐 Opening browser tabs...")
    
    try:
        # Open dashboard
        webbrowser.open('http://localhost:5000')
        time.sleep(1)
        
        # Open TensorBoard
        webbrowser.open('http://localhost:6006')
        
        print("✅ Browser tabs opened")
    except Exception as e:
        print(f"⚠️  Could not open browser automatically: {e}")
        print("You can manually open:")
        print("  Dashboard: http://localhost:5000")
        print("  TensorBoard: http://localhost:6006")

def print_banner():
    """Print startup banner"""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║    🎮 Pokémon Crystal RL Training System                     ║
║                                                              ║
║    🤖 AI Agent Training for Pokémon Crystal                  ║
║    📊 Real-time Monitoring Dashboard                         ║
║    🔥 Powered by Stable Baselines3 & BizHawk                ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def print_usage_instructions():
    """Print usage instructions"""
    instructions = """
🚀 System is ready! Here's how to use it:

📊 MONITORING DASHBOARD (http://localhost:5000)
   • Start/stop training with different algorithms
   • Monitor real-time system stats
   • View training progress charts
   • Manage training history

📈 TENSORBOARD (http://localhost:6006)
   • Detailed training metrics
   • Loss curves and performance graphs
   • Model architecture visualization

🎯 GETTING STARTED:
   1. Open the dashboard in your browser
   2. Configure training parameters (algorithm, timesteps, etc.)
   3. Click "Start Training" to begin
   4. Monitor progress in real-time
   5. Use TensorBoard for detailed analysis

⚙️  SUPPORTED ALGORITHMS:
   • PPO (Proximal Policy Optimization) - Recommended
   • DQN (Deep Q-Network)
   • A2C (Advantage Actor-Critic)

🎮 GAME INTEGRATION:
   • BizHawk emulator with Lua scripting
   • Real-time game state extraction
   • Automatic action execution
   • Reward-based learning

💡 TIPS:
   • Start with PPO for stable training
   • Use 1M timesteps for initial experiments
   • Monitor CPU/GPU usage in the dashboard
   • Check logs if training doesn't start

Press Ctrl+C to stop the monitoring system.
    """
    print(instructions)

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    print("\n🛑 Shutting down monitoring system...")
    sys.exit(0)

def main():
    """Main function"""
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print_banner()
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Start TensorBoard
    tensorboard_process = start_tensorboard()
    
    # Start monitoring dashboard
    if not start_monitoring_dashboard():
        if tensorboard_process:
            tensorboard_process.terminate()
        sys.exit(1)
    
    # Give everything a moment to start
    time.sleep(3)
    
    # Open browser tabs
    open_browser_tabs()
    
    # Print instructions
    print_usage_instructions()
    
    try:
        # Keep the main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
        
        if tensorboard_process:
            print("Stopping TensorBoard...")
            tensorboard_process.terminate()
            tensorboard_process.wait()
        
        print("✅ Shutdown complete")

if __name__ == "__main__":
    main()
