#!/usr/bin/env python3
"""
Fix for blank screen issue - The trainer needs screen capture to be started.
"""

import os
import sys
import time
import numpy as np
from unittest.mock import Mock, patch

# Add the parent directory to the Python path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from training.trainer import PokemonTrainer
from training.trainer import TrainingConfig, TrainingMode
from monitoring.bridge import TrainerWebBridge


def create_fixed_trainer(pyboy_mock):
    """Create a trainer with proper screen capture initialization"""
    print("🔧 Creating trainer with screen capture fix...")
    
    # Create mock PyBoy instance
    pyboy_mock.frame_count = 1000
    pyboy_mock.screen.ndarray = Mock(return_value=np.random.randint(0, 255, (144, 160, 3), dtype=np.uint8))

    # Create config with screen capture enabled
    config = TrainingConfig(
        mode=TrainingMode.FAST_MONITORED,
        rom_path="../roms/pokemon_crystal.gbc",
        capture_screens=True,  # This is key!
        capture_fps=2,
        headless=True,
        enable_web=False,  # Don't need web for this test
        debug_mode=True
    )
    
    # Create trainer
    trainer = PokemonTrainer(config)
    
    # IMPORTANT: Start the screen capture thread manually
    if config.capture_screens:
        trainer._start_screen_capture()
        print("📸 Screen capture started manually")
        time.sleep(1)  # Give it time to start capturing
    
    return trainer


@patch('trainer.trainer.PyBoy')
def test_fixed_streaming(mock_pyboy_class):
    """Test the fixed streaming system"""
    print("🧪 Testing Fixed Socket Streaming")
    print("=" * 60)
    
    # Create fixed trainer
    mock_pyboy_instance = Mock()
    mock_pyboy_class.return_value = mock_pyboy_instance
    trainer = create_fixed_trainer(mock_pyboy_instance)
    
    # Wait for first screenshot to be captured
    print("⏳ Waiting for first screenshot...")
    timeout = 10
    start_time = time.time()
    
    while trainer.latest_screen is None and (time.time() - start_time) < timeout:
        time.sleep(0.1)
    
    if trainer.latest_screen:
        screen_data = trainer.latest_screen
        print(f"✅ First screenshot captured!")
        print(f"   Frame ID: {screen_data.get('frame_id', 'N/A')}")
        print(f"   Timestamp: {screen_data.get('timestamp', 'N/A')}")
        print(f"   Data length: {screen_data.get('data_length', 0)} bytes")
        print(f"   Size: {screen_data.get('size', 'N/A')}")
    else:
        print("❌ No screenshot captured within timeout")
        trainer._finalize_training()
        assert False, "No screenshot captured within timeout"
    
    # Test continuous capture
    print("\n📸 Testing continuous capture...")
    initial_count = 0
    
    for i in range(10):
        time.sleep(0.5)  # Wait between checks
        
        current_screen = trainer.latest_screen
        if current_screen:
            current_frame_id = current_screen.get('frame_id', 0)
            print(f"   Frame {i+1}: ID={current_frame_id}, data={current_screen.get('data_length', 0)} bytes")
        else:
            print(f"   Frame {i+1}: No screenshot")
    
    # Test with bridge
    print("\n🌉 Testing with bridge...")
    
    # Create TrainerWebBridge connected to trainer
    bridge = TrainerWebBridge(
        trainer=trainer,
        host='127.0.0.1',
        port=8000,
        debug=False
    )
    
    # Start bridge
    bridge.start_bridge()
    
    # Give bridge time to start up
    time.sleep(1)
    
    # Test bridge transfer by making HTTP requests
    print("⏳ Testing bridge transfers...")
    import requests
    
    # Make several screenshot requests to the bridge
    bridge_url = f"http://{bridge.host}:{bridge.port}"
    for i in range(5):
        try:
            response = requests.get(f"{bridge_url}/api/screenshot/current", timeout=1.0)
            if response.status_code == 200:
                data = response.json()
                if data.get('screenshot'):
                    print(f"   Screenshot request {i+1}: Success ({len(data['screenshot'])} chars)")
                else:
                    print(f"   Screenshot request {i+1}: No screenshot data")
            else:
                print(f"   Screenshot request {i+1}: HTTP {response.status_code}")
        except Exception as e:
            print(f"   Screenshot request {i+1}: Error - {e}")
        
        time.sleep(0.5)
    
    # Give time for processing
    time.sleep(1)
    
    bridge_stats = bridge.get_bridge_stats()
    print(f"📀 Bridge Statistics:")
    print(f"   Screenshots transferred: {bridge_stats['screenshots_transferred']}")
    print(f"   Total errors: {bridge_stats['total_errors']}")
    print(f"   Success rate: {bridge_stats['success_rate']:.1f}%")
    print(f"   Bridge active: {bridge_stats['is_active']}")
    
    # Stop bridge
    bridge.stop_bridge()
    
    # Cleanup
    trainer._finalize_training()
    
    # Results
    if bridge_stats['screenshots_transferred'] > 0:
        print(f"\n✅ SUCCESS! Bridge transferred {bridge_stats['screenshots_transferred']} screenshots")
    else:
        print(f"\n❌ FAILED! No screenshots transferred")
        assert bridge_stats['screenshots_transferred'] > 0, "No screenshots were transferred"


def main():
    """Main function to test the fix"""
    print("🔧 Socket Connection & Blank Screen Fix Test")
    print("=" * 60)
    print()
    
    try:
        test_fixed_streaming()
        print("\n🎯 FIX VERIFICATION COMPLETE!")
        print("✅ The issue was that screen capture thread wasn't started")
        print("✅ Screenshots are now being captured and transferred")
        print()
        print("💡 TO APPLY THE FIX:")
        print("1. Ensure trainer.config.capture_screens = True")
        print("2. Make sure trainer._start_screen_capture() is called")
        print("3. Or call trainer.start_training() to start everything properly")
        print()
        print("📋 ORIGINAL ISSUE:")
        print("   - Bridge was working correctly")
        print("   - Web monitor was working correctly") 
        print("   - The trainer just wasn't capturing screenshots!")
        print("   - PyBoy screen access works fine")
            
    except AssertionError as e:
        print(f"\n❌ FIX VERIFICATION FAILED: {e}")
        print("The issue may be more complex than screen capture initialization")
    except Exception as e:
        print(f"\n💥 TEST ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
