#!/usr/bin/env python3
"""
Simple Pokemon Crystal RL Training Script

A minimal, stable training script that bypasses the complex trainer architecture
and directly uses PyBoy for reliable Pokemon Crystal training.
"""

import time
import numpy as np
from pyboy import PyBoy
import json
import signal
import sys
import os
from datetime import datetime

class SimpleTrainer:
    def __init__(self, rom_path, max_actions=1000):
        self.rom_path = rom_path
        self.max_actions = max_actions
        self.pyboy = None
        self.actions_taken = 0
        self.start_time = time.time()
        self.stats = {
            'actions_taken': 0,
            'frames_processed': 0,
            'training_time': 0,
            'actions_per_second': 0,
            'game_states_seen': [],
            'start_time': datetime.now().isoformat()
        }
        self.running = True
        
        # Setup graceful shutdown
        signal.signal(signal.SIGINT, self.shutdown)
        signal.signal(signal.SIGTERM, self.shutdown)
        
    def shutdown(self, signum, frame):
        """Graceful shutdown handler"""
        print(f"\n⏸️ Shutting down training...")
        self.running = False
        if self.pyboy:
            self.pyboy.stop()
        self.save_stats()
        print("✅ Training stopped cleanly")
        sys.exit(0)
    
    def initialize_pyboy(self):
        """Initialize PyBoy emulator"""
        print(f"🎮 Initializing PyBoy with {self.rom_path}")
        self.pyboy = PyBoy(self.rom_path, window="null", debug=False)
        print("✅ PyBoy initialized successfully")
        
    def skip_intro(self):
        """Skip Pokemon Crystal intro sequence"""
        print("⚡ Skipping Pokemon Crystal intro...")
        
        for i in range(200):
            if not self.running:
                break
                
            # Skip intro with START and A buttons
            if i % 15 == 0:
                self.pyboy.button_press('start')
                for _ in range(3):
                    self.pyboy.tick()
                self.pyboy.button_release('start')
            elif i % 15 == 7:
                self.pyboy.button_press('a')
                for _ in range(3):
                    self.pyboy.tick()
                self.pyboy.button_release('a')
            else:
                self.pyboy.tick()
            
            # Check if we've reached gameplay
            if i % 40 == 0:
                screen = self.pyboy.screen.ndarray
                variance = np.var(screen.astype(np.float32))
                if variance > 2000:
                    print(f"✅ Reached gameplay at frame {i} (variance: {variance:.1f})")
                    return True
        
        print("⚠️ Intro skip may not be complete, continuing anyway...")
        return True
    
    def get_rule_based_action(self):
        """Get next action using simple rule-based system"""
        # Simple action pattern that works well for Pokemon
        action_cycle = [
            'a',      # Interact/confirm
            'up',     # Move up
            'a',      # Interact
            'right',  # Move right
            'a',      # Interact
            'down',   # Move down
            'a',      # Interact
            'left',   # Move left
            'a',      # Interact
            'start'   # Open menu occasionally
        ]
        return action_cycle[self.actions_taken % len(action_cycle)]
    
    def execute_action(self, action):
        """Execute an action in the game"""
        if not self.running:
            return
            
        # Press button
        self.pyboy.button_press(action)
        
        # Hold for a few frames
        for _ in range(6):  # Hold button for 6 frames
            if not self.running:
                break
            self.pyboy.tick()
            self.stats['frames_processed'] += 1
        
        # Release button
        self.pyboy.button_release(action)
        
        # Wait a few more frames
        for _ in range(2):
            if not self.running:
                break
            self.pyboy.tick()
            self.stats['frames_processed'] += 1
        
        self.actions_taken += 1
        self.stats['actions_taken'] = self.actions_taken
    
    def analyze_screen(self):
        """Analyze current screen for variety"""
        screen = self.pyboy.screen.ndarray
        variance = np.var(screen.astype(np.float32))
        
        # Classify screen type based on variance
        if variance < 50:
            state = "blank/loading"
        elif variance < 1000:
            state = "menu/dialogue"
        elif variance < 5000:
            state = "simple_gameplay"
        else:
            state = "complex_gameplay"
            
        return {
            'variance': variance,
            'state': state,
            'unique_colors': len(np.unique(screen.reshape(-1, screen.shape[-1]), axis=0))
        }
    
    def print_progress(self):
        """Print training progress"""
        elapsed = time.time() - self.start_time
        aps = self.actions_taken / elapsed if elapsed > 0 else 0
        
        screen_analysis = self.analyze_screen()
        
        print(f"⚡ Action {self.actions_taken}/{self.max_actions} "
              f"| {aps:.1f} a/s "
              f"| State: {screen_analysis['state']} "
              f"| Variance: {screen_analysis['variance']:.1f}")
        
        self.stats['actions_per_second'] = aps
        self.stats['training_time'] = elapsed
    
    def save_stats(self):
        """Save training statistics"""
        stats_file = f"simple_training_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(stats_file, 'w') as f:
            json.dump(self.stats, f, indent=2)
        print(f"📊 Stats saved to {stats_file}")
    
    def start_training(self):
        """Start the training process"""
        print("🚀 Starting Simple Pokemon Crystal RL Training")
        print("=" * 60)
        
        try:
            # Initialize emulator
            self.initialize_pyboy()
            
            # Skip intro
            if not self.skip_intro():
                print("❌ Failed to skip intro")
                return False
            
            print(f"\n🎯 Starting training loop ({self.max_actions} actions)")
            print("🔄 Press Ctrl+C to stop training gracefully")
            print()
            
            # Main training loop
            while self.running and self.actions_taken < self.max_actions:
                # Get next action
                action = self.get_rule_based_action()
                
                # Execute action
                self.execute_action(action)
                
                # Print progress every 50 actions
                if self.actions_taken % 50 == 0:
                    self.print_progress()
            
            # Training completed
            if self.actions_taken >= self.max_actions:
                print(f"\n✅ Training completed! {self.actions_taken} actions executed")
            
            return True
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            return False
            
        finally:
            # Cleanup
            if self.pyboy:
                self.pyboy.stop()
            self.save_stats()

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Simple Pokemon Crystal RL Training")
    parser.add_argument("--rom", default="roms/pokemon_crystal.gbc", help="ROM file path")
    parser.add_argument("--actions", type=int, default=1000, help="Number of actions to execute")
    
    args = parser.parse_args()
    
    # Validate ROM file
    if not os.path.exists(args.rom):
        print(f"❌ ROM file not found: {args.rom}")
        return 1
    
    # Create and start trainer
    trainer = SimplePokemonTrainer(args.rom, args.actions)
    success = trainer.start_training()
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
