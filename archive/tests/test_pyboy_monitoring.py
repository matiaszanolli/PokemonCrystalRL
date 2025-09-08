#!/usr/bin/env python3
"""
Test PyBoy environment with monitoring integration
"""

import time
import subprocess
import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from environments.pyboy_env import PyBoyPokemonCrystalEnv

def test_pyboy_monitoring():
    """Test PyBoy environment with monitoring"""
    print("🧪 Testing PyBoy Environment with Monitoring")
    print("=" * 50)
    
    # Start monitoring server first
    print("🚀 Starting monitoring server...")
    server_process = subprocess.Popen([
        sys.executable, "advanced_web_monitor.py"
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Give server time to start
    time.sleep(3)
    
    try:
        # Create environment with monitoring enabled
        print("🎮 Creating PyBoy environment...")
        env = PyBoyPokemonCrystalEnv(
            rom_path="../roms/pokemon_crystal.gbc",
            save_state_path=None,
            max_steps=100,
            headless=True,
            debug_mode=True,  # Enable debug mode
            enable_monitoring=True
        )
        
        # Check if monitoring is available
        if not env.is_monitoring_available():
            print("❌ Monitoring not available")
            return False
        
        print("✓ Monitoring is available")
        
        # Reset environment
        print("🔄 Resetting environment...")
        obs, info = env.reset()
        print(f"✓ Environment reset - observation shape: {obs.shape}")
        
        # Run a few steps with monitoring
        print("👣 Running test steps...")
        for step in range(10):
            action = (step % 5) + 1  # Cycle through movement actions
            obs, reward, terminated, truncated, info = env.step(action)
            
            print(f"  Step {step + 1}: Action={env.action_map[action]}, Reward={reward:.3f}")
            
            # Add small delay to see updates
            time.sleep(0.5)
            
            if terminated or truncated:
                break
        
        # Send episode end
        env.send_episode_end(success=False)
        
        print("✅ PyBoy monitoring test completed!")
        print(f"🌐 View the dashboard at: http://localhost:5000")
        
        # Clean up
        env.close()
        
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
    test_pyboy_monitoring()
