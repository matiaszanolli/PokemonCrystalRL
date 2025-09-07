#!/usr/bin/env python3
"""
Quick test to verify WebSocket streaming setup
"""

import sys
import os
import time
import asyncio
from unittest.mock import Mock

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_websocket_setup():
    """Test WebSocket setup with mock PyBoy"""
    print("🧪 Testing WebSocket streaming setup...")
    
    try:
        from training.trainer import PokemonTrainer, TrainingConfig, TrainingMode
        
        # Create config with web enabled
        config = TrainingConfig(
            rom_path="test.gbc",  # Will be mocked
            mode=TrainingMode.ULTRA_FAST,
            enable_web=True,
            web_port=8080,
            headless=True,
            debug_mode=True
        )
        
        # Create mock PyBoy
        mock_pyboy = Mock()
        mock_pyboy.screen = Mock()
        mock_pyboy.screen.ndarray = Mock()
        # Create a realistic screen array
        import numpy as np
        mock_screen = np.random.randint(0, 255, (144, 160, 3), dtype=np.uint8)
        mock_pyboy.screen.ndarray = mock_screen
        mock_pyboy.frame_count = 0
        
        print("✅ Mock PyBoy created")
        
        # Create trainer with mock
        trainer = PokemonTrainer(config)
        trainer.pyboy = mock_pyboy  # Override with mock
        
        print("✅ Trainer created")
        
        # Check web monitor
        if trainer.web_monitor:
            print(f"✅ Web monitor initialized: {trainer.web_monitor.get_url()}")
            
            # Check screen capture
            if trainer.web_monitor.screen_capture:
                print("✅ Screen capture initialized")
                
                # Test screen capture
                trainer.web_monitor.screen_capture.pyboy = mock_pyboy
                trainer.web_monitor.screen_capture.start_capture()
                
                print("✅ Screen capture started")
                
                # Wait a bit for captures
                time.sleep(2)
                
                # Check if we got frames
                if trainer.web_monitor.screen_capture.latest_screen:
                    print(f"✅ Screen captured: {trainer.web_monitor.screen_capture.stats}")
                else:
                    print("❌ No screen captured")
                
                # Check WebSocket clients
                print(f"📡 WebSocket clients: {len(trainer.web_monitor.ws_clients)}")
                
                trainer.web_monitor.screen_capture.stop_capture()
            else:
                print("❌ Screen capture not initialized")
        else:
            print("❌ Web monitor not initialized")
            
        # Test WebSocket connection
        print("\n🔌 Testing WebSocket connection...")
        try:
            async def test_ws_client():
                try:
                    import websockets
                    uri = f"ws://localhost:{trainer.web_monitor.ws_port}/stream"
                    print(f"📡 Connecting to {uri}")
                    
                    async with websockets.connect(uri) as websocket:
                        print("✅ WebSocket connected!")
                        
                        # Wait for a frame
                        frame = await asyncio.wait_for(websocket.recv(), timeout=3.0)
                        print(f"✅ Received frame: {len(frame)} bytes")
                        
                        return True
                        
                except asyncio.TimeoutError:
                    print("⚠️ WebSocket connection timeout")
                    return False
                except Exception as e:
                    print(f"❌ WebSocket error: {e}")
                    return False
            
            # Run WebSocket test
            loop = asyncio.get_event_loop()
            ws_result = loop.run_until_complete(test_ws_client())
            
            if ws_result:
                print("✅ WebSocket streaming working!")
            else:
                print("❌ WebSocket streaming not working")
                
        except ImportError:
            print("⚠️ websockets module not available for testing")
        except Exception as e:
            print(f"❌ WebSocket test error: {e}")
            
        # Cleanup
        if trainer.web_monitor:
            trainer.web_monitor.stop()
            
        print("\n🎉 Test completed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_websocket_setup()
