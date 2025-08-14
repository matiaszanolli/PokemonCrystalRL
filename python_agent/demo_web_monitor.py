#!/usr/bin/env python3
"""
demo_web_monitor.py - Complete demonstration of Pokemon Crystal RL Web Monitoring

This script demonstrates the full capabilities of the web monitoring system
with both real and simulated data to showcase all features.
"""

import os
import sys
import time
import threading
import argparse
from typing import Optional

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from test_web_monitor import run_mock_data_generator
from web_monitor import PokemonRLWebMonitor, create_dashboard_templates
from web_enhanced_training import WebEnhancedTrainingSession


def print_banner():
    """Print an attractive banner for the demo"""
    banner = """
╔═══════════════════════════════════════════════════════════════════╗
║                                                                   ║
║    🎮 POKEMON CRYSTAL RL - WEB MONITORING SYSTEM DEMO 🎮         ║
║                                                                   ║
║    Real-time training visualization and monitoring dashboard      ║
║    Built with Flask, WebSockets, and modern web technologies     ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝

Welcome to the Pokemon Crystal RL Web Monitoring System!

This demo showcases a comprehensive real-time dashboard for monitoring
Pokemon Crystal reinforcement learning training sessions.

🌟 KEY FEATURES:
   ✅ Live game screen streaming (3x scaled with pixel art preservation)
   ✅ Real-time statistics tracking (player, party, training metrics)  
   ✅ Action history with button press logging and reasoning
   ✅ Agent decision tracking with confidence scores and visual context
   ✅ Performance metrics collection and visualization
   ✅ Professional web dashboard with responsive design
   ✅ WebSocket-based real-time updates (no page refresh needed)
   ✅ Historical data access via REST API
   ✅ Multi-threaded architecture for optimal performance

📊 TECHNICAL HIGHLIGHTS:
   • Flask web server with SocketIO for real-time communication
   • Modern responsive UI with dark theme optimized for monitoring
   • SQLite database integration for persistent training history  
   • Screenshot encoding with base64 transmission for web compatibility
   • Memory-efficient with automatic cleanup and queue management
   • Cross-platform compatibility (Windows, macOS, Linux)

"""
    print(banner)


def demo_menu():
    """Display the demo menu options"""
    menu = """
📋 DEMO OPTIONS:

1. 🧪 Mock Data Demo (Recommended)
   → Test all features with simulated Pokemon game data
   → No ROM file required, perfect for trying out the interface
   → Generates realistic game screenshots, stats, and actions

2. 🎮 Full Training Demo (Requires ROM)
   → Real Pokemon Crystal training with live monitoring
   → Requires 'pokecrystal.gbc' ROM file in current directory
   → Complete integration with actual gameplay

3. 🌐 Web-Only Mode
   → Start dashboard to view existing training data  
   → Monitor historical data from previous training sessions
   → Explore saved training reports and statistics

4. ℹ️  System Information
   → Display technical details and requirements
   → Show current configuration and file structure
   → Verify all dependencies are properly installed

5. 🚪 Exit Demo

"""
    print(menu)


def demo_mock_data():
    """Run the mock data demonstration"""
    print("\n" + "="*60)
    print("🧪 STARTING MOCK DATA DEMONSTRATION")
    print("="*60)
    print()
    
    print("📄 Creating dashboard templates...")
    create_dashboard_templates()
    
    print("🌐 Initializing web monitor...")
    monitor = PokemonRLWebMonitor()
    
    print("🎲 Starting mock data generator...")
    mock_thread = threading.Thread(
        target=run_mock_data_generator, 
        args=(monitor,), 
        daemon=True
    )
    mock_thread.start()
    
    print("\n✅ MOCK DEMO READY!")
    print()
    print("📋 Instructions:")
    print("1. Open your web browser")
    print("2. Navigate to: http://127.0.0.1:5000")
    print("3. Click 'Start Monitor' button in the web interface")
    print("4. Watch the live mock data in action!")
    print()
    print("🎯 What you'll see:")
    print("   • Mock Game Boy screenshots updating every second")
    print("   • Simulated player stats (position, money, badges)")
    print("   • Mock Pokemon party with HP bars and levels")
    print("   • Random action sequences (UP, DOWN, A, B, etc.)")
    print("   • Agent decisions with reasoning and confidence")
    print("   • Training metrics and performance graphs")
    print()
    print("⏳ Starting web server (this may take a few seconds)...")
    print("🛑 Press Ctrl+C when done to return to menu")
    print()
    
    try:
        monitor.run(host='127.0.0.1', port=5000, debug=False)
    except KeyboardInterrupt:
        print("\n✅ Mock demo completed!")


def demo_full_training(rom_path: str):
    """Run the full training demonstration"""
    print("\n" + "="*60)
    print("🎮 STARTING FULL TRAINING DEMONSTRATION")  
    print("="*60)
    print()
    
    if not os.path.exists(rom_path):
        print(f"❌ ROM file not found: {rom_path}")
        print()
        print("To run this demo, you need a Pokemon Crystal ROM file.")
        print("Please ensure 'pokecrystal.gbc' is in the current directory.")
        print()
        return
    
    print(f"✅ ROM file found: {rom_path}")
    print("🎯 Initializing training with web monitoring...")
    
    try:
        session = WebEnhancedTrainingSession(rom_path, episodes=10)
        print("\n📋 Instructions:")
        print("1. The training session will start automatically")
        print("2. Web dashboard will be available at http://127.0.0.1:5000")
        print("3. Open your browser and click 'Start Monitor'")
        print("4. Watch real Pokemon Crystal gameplay with live monitoring!")
        print()
        print("🛑 Press Ctrl+C to stop training gracefully")
        print()
        
        session.run_interactive()
        
    except Exception as e:
        print(f"❌ Training error: {e}")
        print("\nThis could be due to:")
        print("- Missing dependencies")
        print("- ROM compatibility issues")
        print("- PyBoy configuration problems")
        print()
        print("Try the Mock Data Demo (#1) to test the web interface.")


def demo_web_only():
    """Run web-only mode for viewing historical data"""
    print("\n" + "="*60)
    print("🌐 STARTING WEB-ONLY DEMONSTRATION")
    print("="*60)
    print()
    
    print("📄 Creating dashboard templates...")
    create_dashboard_templates()
    
    print("🌐 Starting web server for historical data viewing...")
    monitor = PokemonRLWebMonitor()
    
    print("\n✅ WEB-ONLY MODE READY!")
    print()
    print("📋 Instructions:")
    print("1. Open your browser to: http://127.0.0.1:5000") 
    print("2. Use the dashboard to explore:")
    print("   • Historical training data")
    print("   • Previous training reports")
    print("   • Saved game states and decisions")
    print("   • Performance metrics from past sessions")
    print()
    print("ℹ️  Note: Live data won't update in this mode")
    print("🛑 Press Ctrl+C when done to return to menu")
    print()
    
    try:
        monitor.run(host='127.0.0.1', port=5000, debug=False)
    except KeyboardInterrupt:
        print("\n✅ Web-only demo completed!")


def show_system_info():
    """Display system information and requirements"""
    print("\n" + "="*60)
    print("ℹ️  SYSTEM INFORMATION")
    print("="*60)
    print()
    
    print("📋 SYSTEM REQUIREMENTS:")
    print("   • Python 3.7+ (3.8+ recommended)")
    print("   • Flask and Flask-SocketIO for web server")
    print("   • NumPy for numerical operations")
    print("   • OpenCV for image processing")
    print("   • PyBoy for Game Boy emulation (training mode)")
    print("   • Modern web browser with WebSocket support")
    print()
    
    # Check current setup
    print("🔍 CURRENT CONFIGURATION:")
    try:
        import flask
        print(f"   ✅ Flask {flask.__version__}")
    except ImportError:
        print("   ❌ Flask not installed")
    
    try:
        import flask_socketio
        print(f"   ✅ Flask-SocketIO installed")
    except ImportError:
        print("   ❌ Flask-SocketIO not installed")
    
    try:
        import numpy as np
        print(f"   ✅ NumPy {np.__version__}")
    except ImportError:
        print("   ❌ NumPy not installed")
        
    try:
        import cv2
        print(f"   ✅ OpenCV {cv2.__version__}")
    except ImportError:
        print("   ❌ OpenCV not installed")
    
    print()
    print("📁 FILE STRUCTURE:")
    files_to_check = [
        "web_monitor.py",
        "web_enhanced_training.py", 
        "test_web_monitor.py",
        "templates/dashboard.html",
        "docs/WEB_MONITORING.md"
    ]
    
    for file_path in files_to_check:
        if os.path.exists(file_path):
            print(f"   ✅ {file_path}")
        else:
            print(f"   ❌ {file_path}")
    
    print()
    print("🌐 NETWORK CONFIGURATION:")
    print("   • Web server runs on 127.0.0.1:5000 by default")
    print("   • Uses WebSocket for real-time communication")
    print("   • REST API available for historical data access")
    print("   • No external internet connection required")
    print()
    
    print("💡 TROUBLESHOOTING TIPS:")
    print("   • If port 5000 is busy, try: --port 5001")
    print("   • For network issues, check firewall settings")
    print("   • Browser console shows WebSocket connection status")
    print("   • See docs/WEB_MONITORING.md for detailed help")
    print()


def main():
    """Main demo function"""
    parser = argparse.ArgumentParser(description="Pokemon Crystal RL Web Monitoring Demo")
    parser.add_argument("--rom", default="pokecrystal.gbc", 
                       help="Path to Pokemon Crystal ROM file")
    parser.add_argument("--auto", choices=["mock", "training", "web"], 
                       help="Run demo mode automatically without menu")
    
    args = parser.parse_args()
    
    # Auto mode for non-interactive execution
    if args.auto:
        if args.auto == "mock":
            demo_mock_data()
        elif args.auto == "training":
            demo_full_training(args.rom)
        elif args.auto == "web":
            demo_web_only()
        return
    
    # Interactive mode
    print_banner()
    
    while True:
        demo_menu()
        
        try:
            choice = input("Enter your choice (1-5): ").strip()
            
            if choice == "1":
                demo_mock_data()
            elif choice == "2":
                demo_full_training(args.rom)
            elif choice == "3":
                demo_web_only()
            elif choice == "4":
                show_system_info()
            elif choice == "5":
                print("\n👋 Thank you for trying the Pokemon Crystal RL Web Monitor!")
                print("Visit the docs/ directory for complete documentation.")
                break
            else:
                print(f"\n❌ Invalid choice: {choice}")
                print("Please enter a number between 1 and 5.")
            
            if choice in ["1", "2", "3"]:
                print("\n" + "="*60)
                print("🔄 RETURNING TO MAIN MENU")
                print("="*60)
                time.sleep(2)
        
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Returning to main menu...")
            time.sleep(2)


if __name__ == "__main__":
    main()
