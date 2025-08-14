#!/usr/bin/env python3
"""
create_mock_state.py - Create mock state files for testing the training pipeline
"""

import json
import time
import random
import argparse
import os
from pathlib import Path

def generate_mock_state(frame_count=0):
    """Generate a realistic mock game state"""
    
    # Simulate player movement and game progression
    base_x = 10 + (frame_count // 100) % 50  # Player moves around
    base_y = 15 + (frame_count // 150) % 30
    
    state = {
        "player_x": base_x + random.randint(-2, 2),
        "player_y": base_y + random.randint(-2, 2),
        "player_map": 1,  # Starting town
        "player_hp": max(80, 100 - (frame_count // 1000)),  # HP slowly decreases
        "player_max_hp": 100,
        "player_level": 5 + (frame_count // 5000),  # Level up occasionally
        "player_exp": 150 + frame_count * 2,
        "money": 500 + frame_count,
        "badges": min(8, frame_count // 10000),  # Earn badges over time
        "party_count": min(6, 1 + (frame_count // 3000)),  # Catch more Pokémon
        
        # Additional realistic data
        "frame_count": frame_count,
        "timestamp": time.time(),
        "in_battle": (frame_count % 500) < 50,  # Occasionally in battle
        "in_menu": (frame_count % 300) < 20,   # Occasionally in menu
    }
    
    return state

def create_mock_state_file(output_dir=".", frame_count=0):
    """Create a mock state.json file"""
    state = generate_mock_state(frame_count)
    state_path = Path(output_dir) / "state.json"
    
    with open(state_path, 'w') as f:
        json.dump(state, f)
    
    return state_path

def mock_training_session(duration_seconds=30, output_dir="."):
    """Simulate a training session by creating state files periodically"""
    print(f"🎮 Starting mock training session for {duration_seconds} seconds...")
    print(f"📁 Output directory: {os.path.abspath(output_dir)}")
    
    start_time = time.time()
    frame_count = 0
    
    while time.time() - start_time < duration_seconds:
        # Create state file
        state_path = create_mock_state_file(output_dir, frame_count)
        
        # Check for action file and respond
        action_path = Path(output_dir) / "action.txt"
        if action_path.exists():
            with open(action_path, 'r') as f:
                action = f.read().strip()
            
            print(f"📥 Received action: {action}")
            
            # Remove action file to simulate processing
            action_path.unlink()
            
            print(f"🎯 Processed action {action}")
        
        # Print status periodically
        if frame_count % 60 == 0:  # Every 60 frames (~1 second)
            state = generate_mock_state(frame_count)
            print(f"Frame {frame_count}: Player at ({state['player_x']}, {state['player_y']}), HP: {state['player_hp']}")
        
        frame_count += 1
        time.sleep(0.016)  # ~60 FPS
    
    print(f"✅ Mock session completed after {frame_count} frames!")

def main():
    parser = argparse.ArgumentParser(description="Create mock Pokémon Crystal state files")
    parser.add_argument("--duration", "-d", type=int, default=30,
                       help="Duration of mock session in seconds (default: 30)")
    parser.add_argument("--output-dir", "-o", type=str, default=".",
                       help="Output directory for state files (default: current)")
    parser.add_argument("--single", "-s", action="store_true",
                       help="Create single state file and exit")
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.single:
        state_path = create_mock_state_file(args.output_dir)
        print(f"✅ Created mock state file: {state_path}")
        
        # Display the state content
        with open(state_path, 'r') as f:
            state = json.load(f)
        
        print("📊 Mock state content:")
        for key, value in state.items():
            print(f"  {key}: {value}")
    
    else:
        mock_training_session(args.duration, args.output_dir)

if __name__ == "__main__":
    main()
