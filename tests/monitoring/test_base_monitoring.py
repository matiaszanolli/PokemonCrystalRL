#!/usr/bin/env python3
"""
Test script to verify monitoring integration
"""

import time
import numpy as np
import subprocess
import sys
import signal
import pytest
from monitoring.monitoring_client import MonitoringClient

@pytest.mark.skip(reason="Integration test requires external monitoring server")
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
        server_available = client.is_server_available()
        print(f"✓ Server is available: {server_available}")
        assert server_available, "Server should be available"
        
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
        
        # Test completed successfully - no return value needed for pytest
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        # Re-raise the exception so pytest can catch it
        raise
    
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
