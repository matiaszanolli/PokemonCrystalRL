"""
test_web_monitor.py - Test the web monitoring dashboard

This creates a simple test that runs the web monitor with mock data
to verify everything is working correctly.
"""

import time
import threading
import numpy as np
from pokemon_crystal_rl.monitoring.web_monitor import PokemonRLWebMonitor, create_dashboard_templates


def generate_mock_screenshot():
    """Generate a mock Game Boy-style screenshot"""
    # Create a simple mock screenshot (160x144 like Game Boy)
    screenshot = np.zeros((144, 160, 3), dtype=np.uint8)
    
    # Add some mock Game Boy-style graphics
    # Top section (sky blue)
    screenshot[0:40, :] = [135, 206, 235]  # Sky blue
    
    # Middle section (grass green)
    screenshot[40:100, :] = [34, 139, 34]  # Forest green
    
    # Bottom section (darker brown for ground)
    screenshot[100:144, :] = [101, 67, 33]  # Brown
    
    # Add some random "trees" or obstacles
    for i in range(5):
        x = np.random.randint(10, 150)
        y = np.random.randint(50, 90)
        screenshot[y:y+20, x:x+15] = [0, 100, 0]  # Dark green tree
    
    # Add a simple "character" (player)
    char_x, char_y = 75, 70
    screenshot[char_y:char_y+10, char_x:char_x+8] = [255, 255, 0]  # Yellow character
    
    return screenshot


def mock_training_data():
    """Generate mock training data for testing"""
    return {
        'timestamp': time.time(),
        'player': {
            'x': np.random.randint(0, 100),
            'y': np.random.randint(0, 100),
            'map': np.random.randint(1, 10),
            'money': np.random.randint(100, 10000),
            'badges': np.random.randint(0, 8)
        },
        'party': [
            {
                'species': 25,  # Pikachu
                'level': np.random.randint(5, 50),
                'hp': np.random.randint(20, 100),
                'max_hp': 100,
                'status': 0
            },
            {
                'species': 6,   # Charizard
                'level': np.random.randint(10, 55),
                'hp': np.random.randint(30, 120),
                'max_hp': 120,
                'status': 0
            }
        ],
        'training': {
            'total_steps': np.random.randint(1000, 50000),
            'episodes': np.random.randint(10, 500),
            'decisions_made': np.random.randint(500, 5000),
            'visual_analyses': np.random.randint(200, 2000)
        }
    }


def run_mock_data_generator(monitor: PokemonRLWebMonitor):
    """Run mock data generation in a loop"""
    print("🎯 Starting mock data generation...")
    
    actions = ["UP", "DOWN", "LEFT", "RIGHT", "A", "B", "START", "SELECT"]
    decisions = [
        "Move towards tall grass",
        "Enter Pokemon Center",
        "Battle wild Pokemon",
        "Use item from bag",
        "Talk to NPC",
        "Check status menu",
        "Save game",
        "Explore new area"
    ]
    
    reasonings = [
        "Need to heal Pokemon party",
        "Looking for wild Pokemon to catch",
        "Checking current stats and progress",
        "Following the main story path",
        "Exploring for hidden items",
        "Training Pokemon for upcoming battle",
        "Saving progress before major decision",
        "Investigating interesting area"
    ]
    
    step_count = 0
    
    while True:
        try:
            # Generate and send screenshot
            screenshot = generate_mock_screenshot()
            monitor.update_screenshot(screenshot)
            
            # Update stats
            monitor.current_stats = mock_training_data()
            
            # Add random action
            action = np.random.choice(actions)
            reasoning = np.random.choice(reasonings)
            monitor.update_action(action, reasoning)
            
            # Add random decision every few steps
            if step_count % 3 == 0:
                decision_data = {
                    'decision': np.random.choice(decisions),
                    'reasoning': np.random.choice(reasonings),
                    'confidence': np.random.uniform(0.6, 0.95),
                    'visual_context': {
                        'screen_type': np.random.choice(['overworld', 'battle', 'menu', 'dialogue']),
                        'detected_text': ['POKEMON', 'HP', 'LEVEL']
                    }
                }
                monitor.update_decision(decision_data)
            
            # Add performance metrics
            monitor.add_performance_metric('reward', np.random.uniform(-1, 5))
            monitor.add_performance_metric('exploration', np.random.uniform(0, 1))
            
            step_count += 1
            time.sleep(1)  # Update every second
            
        except KeyboardInterrupt:
            print("\n🛑 Mock data generation stopped")
            break
        except Exception as e:
            print(f"❌ Mock data error: {e}")
            time.sleep(1)


def main():
    """Main test function"""
    print("🧪 Pokemon Crystal RL Web Monitor Test")
    print("=" * 40)
    print()
    
    # Create templates
    print("📄 Creating dashboard templates...")
    create_dashboard_templates()
    
    # Create monitor
    print("🌐 Initializing web monitor...")
    monitor = PokemonRLWebMonitor()
    
    # Start mock data generation in background
    print("🎲 Starting mock data generator...")
    mock_thread = threading.Thread(target=run_mock_data_generator, args=(monitor,), daemon=True)
    mock_thread.start()
    
    print("\n✅ Test setup complete!")
    print()
    print("📋 Instructions:")
    print("1. The web server will start on http://127.0.0.1:5000")
    print("2. Open your browser to that URL")
    print("3. Click 'Start Monitor' to see live mock data")
    print("4. You should see:")
    print("   - Live screenshot updates (mock Game Boy graphics)")
    print("   - Real-time game statistics")
    print("   - Action history with button presses")
    print("   - Agent decision logs")
    print("   - Training metrics")
    print()
    print("Press Ctrl+C to stop the test")
    print("=" * 40)
    print()
    
    # Run web monitor
    try:
        monitor.run(host='127.0.0.1', port=5000, debug=False)
    except KeyboardInterrupt:
        print("\n🏁 Test completed!")


if __name__ == "__main__":
    main()
