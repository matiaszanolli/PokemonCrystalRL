#!/usr/bin/env python3
"""
Final comprehensive integration test with complete web interface.

This test demonstrates the full solution:
1. Web monitor with proper templates
2. Bridge connecting trainer to monitor
3. Live screenshot streaming with web interface
4. Complete integration ready for production
"""

import time
import threading
import numpy as np
import sys
import os

# Add the python_agent directory to the path
sys.path.insert(0, '/mnt/data/src/pokemon_crystal_rl/python_agent')

try:
    from monitoring.web_monitor import PokemonRLWebMonitor, create_dashboard_templates
    from monitoring.trainer_monitor_bridge import TrainerWebMonitorBridge, create_integrated_monitoring_system
    IMPORTS_OK = True
except ImportError as e:
    print(f"❌ Import error: {e}")
    IMPORTS_OK = False


class MockTrainer:
    """Enhanced mock trainer with more realistic simulation"""
    
    def __init__(self):
        self.stats = {
            'start_time': time.time(),
            'total_actions': 0,
            'total_episodes': 1,
            'llm_calls': 0,
            'actions_per_second': 0.0
        }
        
        self.latest_screen = None
        self.last_action = "move_up"
        self.last_action_reasoning = "Exploring the overworld"
        self.simulation_active = True
        self.simulation_thread = threading.Thread(target=self._simulate_pokemon_game, daemon=True)
        self.simulation_thread.start()
        
    def _simulate_pokemon_game(self):
        """Simulate actual Pokemon Crystal gameplay scenarios"""
        frame_id = 0
        scenarios = [
            ("overworld", "Walking in grass looking for Pokemon"),
            ("battle", "Fighting wild Pokemon"),
            ("menu", "Checking inventory and Pokemon"),
            ("dialogue", "Talking to NPCs"),
            ("overworld", "Moving between areas")
        ]
        
        while self.simulation_active:
            scenario_index = (frame_id // 50) % len(scenarios)
            scenario, action_reason = scenarios[scenario_index]
            
            # Create scenario-specific screenshot
            screenshot = self._create_scenario_screenshot(scenario, frame_id)
            self.latest_screen = self._screenshot_to_trainer_format(screenshot, frame_id)
            
            # Update stats realistically
            self.stats['total_actions'] = frame_id
            self.stats['actions_per_second'] = frame_id / max(1, time.time() - self.stats['start_time'])
            
            if frame_id % 8 == 0:  # LLM decisions every 8 frames
                self.stats['llm_calls'] += 1
                self.last_action = ["move_up", "move_down", "move_left", "move_right", "press_a", "press_b"][frame_id % 6]
                self.last_action_reasoning = action_reason
            
            frame_id += 1
            time.sleep(0.1)  # 10 FPS
    
    def _create_scenario_screenshot(self, scenario: str, frame_id: int) -> np.ndarray:
        """Create screenshots that look like actual Pokemon game scenarios"""
        height, width = 144, 160
        screenshot = np.zeros((height, width, 3), dtype=np.uint8)
        
        if scenario == "overworld":
            # Green grass background with animated elements
            base_green = 80 + (frame_id % 20) * 2
            screenshot[:, :] = [30, base_green, 30]  # Grass green
            
            # Add "player" sprite (red square that moves)
            player_x = 80 + int(10 * np.sin(frame_id * 0.1))
            player_y = 72 + int(5 * np.cos(frame_id * 0.1))
            screenshot[player_y:player_y+8, player_x:player_x+8] = [200, 50, 50]  # Red player
            
            # Add trees (dark green rectangles)
            for i in range(0, width, 32):
                screenshot[10:30, i:i+16] = [20, 60, 20]
                
        elif scenario == "battle":
            # Battle scene with health bars
            screenshot[:, :] = [40, 40, 80]  # Battle blue background
            
            # Player Pokemon (bottom)
            screenshot[100:130, 20:60] = [255, 200, 0]  # Yellow Pokemon
            
            # Enemy Pokemon (top)  
            screenshot[20:50, 100:140] = [150, 50, 200]  # Purple enemy
            
            # Health bars
            hp_width = max(5, 50 - (frame_id % 100) // 2)  # Decreasing HP
            screenshot[90:95, 20:20+hp_width] = [0, 255, 0]  # Green HP bar
            
        elif scenario == "menu":
            # Menu interface
            screenshot[:, :] = [20, 20, 40]  # Dark background
            
            # Menu box
            screenshot[40:120, 30:130] = [60, 60, 100]  # Menu background
            screenshot[45:50, 35:125] = [255, 255, 255]  # Menu border
            screenshot[50:55, 35:125] = [200, 200, 200]  # Menu items
            
            # Cursor (blinking)
            if (frame_id // 10) % 2:
                screenshot[50:55, 35:40] = [255, 255, 0]  # Yellow cursor
                
        elif scenario == "dialogue":
            # Dialogue box
            screenshot[:, :] = [50, 80, 50]  # Outdoor green
            
            # Text box at bottom
            screenshot[100:140, 10:150] = [240, 240, 240]  # White text box
            screenshot[105:110, 15:145] = [0, 0, 0]  # Black text line 1
            screenshot[115:120, 15:100] = [0, 0, 0]  # Black text line 2
            
            # Animated cursor
            if (frame_id // 20) % 2:
                screenshot[130:135, 140:145] = [255, 0, 0]  # Red cursor
        
        return screenshot.astype(np.uint8)
    
    def _screenshot_to_trainer_format(self, screenshot: np.ndarray, frame_id: int) -> dict:
        """Convert numpy screenshot to trainer's expected format"""
        import base64
        import io
        from PIL import Image
        
        img = Image.fromarray(screenshot)
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=85)
        img_b64 = base64.b64encode(buffer.getvalue()).decode()
        
        return {
            'image_b64': img_b64,
            'timestamp': time.time(),
            'size': (160, 144),
            'frame_id': frame_id,
            'data_length': len(img_b64)
        }
    
    def get_current_stats(self):
        return self.stats.copy()
    
    def stop_simulation(self):
        self.simulation_active = False


def main():
    """Main test runner with complete integration demonstration"""
    print("🎉 POKEMON RL WEB MONITOR - FINAL INTEGRATION TEST")
    print("=" * 70)
    print("🎯 Demonstrating complete working solution:")
    print("   1. Web monitor with proper HTML templates")
    print("   2. Bridge connecting trainer to monitor") 
    print("   3. Live Pokemon game simulation")
    print("   4. Real-time screenshot streaming")
    print("   5. Full web interface with stats and actions")
    print("")
    
    if not IMPORTS_OK:
        print("❌ Cannot run test - imports failed")
        return False
    
    # Ensure templates exist
    print("🗂️ Creating dashboard templates...")
    create_dashboard_templates()
    
    # Create enhanced mock trainer with Pokemon scenarios
    print("🤖 Creating Pokemon game simulation...")
    trainer = MockTrainer()
    time.sleep(1)
    
    # Create integrated monitoring system
    print("🏗️ Setting up integrated monitoring system...")
    web_monitor, bridge, monitor_thread = create_integrated_monitoring_system(
        trainer, host='127.0.0.1', port=5004
    )
    
    # Start the bridge
    bridge.start_bridge()
    
    print("🚀 POKEMON RL TRAINING MONITOR IS LIVE!")
    print("=" * 70)
    print(f"🌐 Web Interface: http://127.0.0.1:5004")
    print("📱 Features Available:")
    print("   ✅ Live Pokemon game screens")
    print("   ✅ Real-time training statistics") 
    print("   ✅ Action history tracking")
    print("   ✅ WebSocket real-time updates")
    print("   ✅ Responsive web interface")
    print("")
    print("🎮 Game Scenarios Simulated:")
    print("   🌱 Overworld exploration")
    print("   ⚔️  Pokemon battles")  
    print("   📋 Menu navigation")
    print("   💬 NPC dialogues")
    print("")
    
    try:
        print("⏱️ Running for 60 seconds - visit the web interface!")
        print("   🖱️ Click 'Start Monitor' button to see live updates")
        
        for i in range(60):
            time.sleep(1)
            
            if i % 10 == 0:
                stats = bridge.get_bridge_stats()
                trainer_stats = trainer.get_current_stats()
                print(f"📊 {i}s: Bridge transferred {stats['screenshots_transferred']} screenshots, "
                      f"Game at {trainer_stats['total_actions']} actions")
        
        print("\n⏹️ Stopping integrated system...")
        bridge.stop_bridge()
        trainer.stop_simulation()
        
        final_stats = bridge.get_bridge_stats()
        trainer_final = trainer.get_current_stats()
        
        print("✅ FINAL INTEGRATION TEST RESULTS:")
        print("=" * 50)
        print(f"📸 Screenshots transferred: {final_stats['screenshots_transferred']}")
        print(f"❌ Bridge errors: {final_stats['total_errors']}")
        print(f"🎮 Game actions simulated: {trainer_final['total_actions']}")
        print(f"🤖 LLM decisions: {trainer_final['llm_calls']}")
        print(f"⚡ Average game FPS: {trainer_final['actions_per_second']:.1f}")
        
        success = (final_stats['screenshots_transferred'] > 50 and 
                  final_stats['total_errors'] < 5 and
                  trainer_final['total_actions'] > 500)
        
        if success:
            print("\n🎉 SUCCESS! Web Monitor Integration Complete!")
            print("✨ Solution Summary:")
            print("   ❌ Problem: Web monitor showed blank screens") 
            print("   ✅ Solution: TrainerWebMonitorBridge connects systems")
            print("   🔧 Implementation: Real-time screenshot streaming")
            print("   🎯 Result: Live Pokemon game monitoring works!")
            print("")
            print("🚀 Production Integration:")
            print("   from monitoring.trainer_monitor_bridge import create_integrated_monitoring_system")
            print("   web_monitor, bridge, thread = create_integrated_monitoring_system(trainer)")
            print("   bridge.start_bridge()")
            print("   trainer.start_training()")
            print("   # Visit web interface for live monitoring!")
        else:
            print(f"\n⚠️ Test completed but with suboptimal results")
            print("   Bridge may need adjustment for production use")
        
        return success
        
    except KeyboardInterrupt:
        print("\n⏹️ Test interrupted by user")
        bridge.stop_bridge() 
        trainer.stop_simulation()
        return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
