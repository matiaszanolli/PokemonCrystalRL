#!/usr/bin/env python3
"""
Test script to verify monitoring integration
"""

import time
import numpy as np
import subprocess
import sys
import signal
from monitoring.monitoring_client import MonitoringClient

def test_monitoring_integration():
    """Test the monitoring client integration"""
    print("🧪 Testing Monitoring Integration")
    print("=" * 40)
    
    # Start monitoring server
    print("🚀 Starting monitoring server...")
    server_process = subprocess.Popen([
        sys.executable, "advanced_web_monitor.py"
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Give server time to start
    time.sleep(3)
    
    try:
        # Create monitoring client
        print("📊 Creating monitoring client...")
        client = MonitoringClient(auto_start=False)
        
        # Test server availability
        if not client.is_server_available():
            print("❌ Server not available")
            return False
        
        print("✓ Server is available")
        
        # Test step update
        print("📈 Testing step update...")
        client.update_step(
            step=1,
            reward=0.5,
            action="RIGHT",
            screen_type="overworld",
            map_id=1,
            player_x=10,
            player_y=15
        )
        
        # Test screenshot update
        print("📸 Testing screenshot update...")
        test_image = np.random.randint(0, 255, (144, 160, 3), dtype=np.uint8)
        # Add some patterns to make it visible
        test_image[50:90, 50:110] = [255, 0, 0]  # Red square
        test_image[10:30, 10:50] = [0, 255, 0]   # Green square
        client.update_screenshot(test_image)
        
        # Test LLM decision
        print("🤖 Testing LLM decision update...")
        client.update_llm_decision("RIGHT", "Moving towards the goal", {
            "confidence": 0.8,
            "alternatives": ["UP", "DOWN"]
        })
        
        # Test text update
        print("📝 Testing text update...")
        client.update_text("PROFESSOR OAK: Hello there!")
        client.update_text("Welcome to the world of Pokemon!")
        
        # Test episode update
        print("🎮 Testing episode update...")
        client.update_episode(
            episode=1,
            total_reward=10.5,
            steps=100,
            success=True
        )
        
        print("✅ All tests passed!")
        print(f"🌐 View the dashboard at: http://localhost:5000")
        print("🛑 Press Ctrl+C to stop the test server")
        
        # Keep running to allow manual inspection
        try:
            while True:
                # Send periodic updates to keep dashboard alive
                time.sleep(2)
                client.update_step(
                    step=client.current_step + 1,
                    reward=np.random.uniform(-0.1, 0.5),
                    action=np.random.choice(["UP", "DOWN", "LEFT", "RIGHT", "A", "B"]),
                    screen_type="overworld",
                    map_id=1,
                    player_x=np.random.randint(0, 20),
                    player_y=np.random.randint(0, 20)
                )
                
                # Update screenshot occasionally
                if client.current_step % 5 == 0:
                    test_image = np.random.randint(0, 255, (144, 160, 3), dtype=np.uint8)
                    client.update_screenshot(test_image)
        
        except KeyboardInterrupt:
            print("\n🛑 Test interrupted by user")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Stop server
        print("🔄 Stopping monitoring server...")
        server_process.terminate()
        try:
            server_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server_process.kill()
        print("✅ Server stopped")

if __name__ == "__main__":
    test_monitoring_integration()
