#!/usr/bin/env python3
"""
start_curriculum_training.py - Quick Start Script for Curriculum Training

This script provides an easy way to start the progressive curriculum training
system for Pokemon Crystal RL.
"""

import sys
import os
from datetime import datetime

try:
    from curriculum_training import CurriculumTrainer, TrainingStage
except ImportError:
    print("❌ curriculum_training.py not found!")
    print("Please ensure curriculum_training.py is in the same directory.")
    sys.exit(1)


def print_curriculum_overview():
    """Print overview of the training curriculum"""
    print("🎓 POKEMON CRYSTAL PROGRESSIVE TRAINING CURRICULUM")
    print("=" * 60)
    print("""
📚 TRAINING STAGES:

Stage 1: Basic Controls & Navigation (10-25 episodes)
   🎯 Master fundamental game controls and menu navigation
   
Stage 2: Dialogue & Text Interaction (15-30 episodes)  
   🎯 Learn dialogue systems and text-based interactions
   
Stage 3: Pokemon Selection & Party Management (20-35 episodes)
   🎯 Understand Pokemon selection and basic party concepts
   
Stage 4: Battle System Fundamentals (25-45 episodes)
   🎯 Master basic Pokemon battle mechanics
   
Stage 5: Exploration & World Navigation (30-55 episodes)
   🎯 Learn world exploration and route navigation
   
Stage 6: Pokemon Catching & Collection (35-65 episodes)
   🎯 Master Pokemon catching and team building
   
Stage 7: Trainer Battles & Strategy (40-85 episodes)
   🎯 Develop strategic trainer battle skills
   
Stage 8: Gym Challenge Preparation (50-105 episodes)
   🎯 Prepare for and complete first gym challenge
   
Stage 9: Advanced Strategy & Meta-Game (60-155 episodes)
   🎯 Master advanced Pokemon strategy and progression
   
Stage 10: Game Completion Mastery (100+ episodes)
   🎯 Complete Pokemon Crystal efficiently and consistently

📊 ESTIMATED TIMELINE: 4-8 weeks to full game mastery
🎯 TOTAL EPISODES: 500-1000+ episodes across all stages
""")


def check_requirements():
    """Check if all requirements are met"""
    print("🔍 Checking requirements...")
    
    # Check ROM file
    rom_path = "../roms/pokemon_crystal.gbc"
    if not os.path.exists(rom_path):
        print(f"❌ ROM file not found: {rom_path}")
        print("Please ensure Pokemon Crystal ROM is available.")
        return False
    else:
        print(f"✅ ROM file found: {rom_path}")
    
    # Check Python packages
    required_packages = ["numpy", "sqlite3", "datetime"]
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ Package available: {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ Package missing: {package}")
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("Please install missing packages before proceeding.")
        return False
    
    print("✅ All requirements met!")
    return True


def main():
    """Main function with interactive training setup"""
    print("🚀 POKEMON CRYSTAL CURRICULUM TRAINING - QUICK START")
    print("=" * 60)
    
    # Print overview
    print_curriculum_overview()
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    # Interactive setup
    print("\n🎮 TRAINING SETUP")
    print("-" * 30)
    
    # Training configuration
    print("\n1. TRAINING CONFIGURATION")
    
    max_episodes = input("📊 Maximum total episodes (default: 500): ").strip()
    max_episodes = int(max_episodes) if max_episodes else 500
    
    use_save_state = input("💾 Use save state file? (y/n, default: n): ").strip().lower()
    save_state_path = None
    if use_save_state == 'y':
        save_state_path = input("📁 Save state path (default: ../pokemon_crystal_intro.state): ").strip()
        save_state_path = save_state_path if save_state_path else "../pokemon_crystal_intro.state"
        
        if not os.path.exists(save_state_path):
            print(f"⚠️ Save state file not found: {save_state_path}")
            create_new = input("🔧 Create new save state? (y/n): ").strip().lower()
            if create_new != 'y':
                save_state_path = None
    
    # Advanced options
    print("\n2. ADVANCED OPTIONS")
    
    start_stage = input("📖 Starting stage (1-10, default: 1): ").strip()
    start_stage = int(start_stage) if start_stage and start_stage.isdigit() else 1
    start_stage = max(1, min(10, start_stage))  # Clamp to valid range
    
    # Confirm configuration
    print("\n📋 TRAINING CONFIGURATION SUMMARY")
    print("-" * 40)
    print(f"📊 Maximum Episodes: {max_episodes}")
    print(f"💾 Save State: {'Yes' if save_state_path else 'No'}")
    if save_state_path:
        print(f"   Path: {save_state_path}")
    print(f"📖 Starting Stage: {start_stage}")
    
    confirm = input("\n🚀 Start training with this configuration? (y/n): ").strip().lower()
    if confirm != 'y':
        print("❌ Training cancelled.")
        return
    
    # Initialize trainer
    print(f"\n🎓 Initializing Curriculum Trainer...")
    
    try:
        trainer = CurriculumTrainer(
            rom_path="../roms/pokemon_crystal.gbc",
            save_state_path=save_state_path,
            semantic_db_path=f"curriculum_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
        )
        
        # Set starting stage if not default
        if start_stage > 1:
            trainer.progress.current_stage = TrainingStage(start_stage)
            print(f"📖 Starting from Stage {start_stage}: {trainer.progress.current_stage.name}")
        
        print(f"\n✅ Trainer initialized successfully!")
        print(f"🎯 Ready to begin curriculum training!")
        
        # Start training
        print(f"\n" + "="*60)
        print(f"🚀 STARTING TRAINING SESSION")
        print(f"=" * 60)
        
        trainer.start_curriculum_training(max_total_episodes=max_episodes)
        
    except Exception as e:
        print(f"❌ Error initializing trainer: {e}")
        print(f"🔧 Please check your configuration and try again.")
        return
    
    print(f"\n🏁 Training session completed!")
    print(f"📊 Check the generated report above for results.")


def quick_demo():
    """Quick demonstration of basic curriculum training"""
    print("🎮 QUICK DEMO - Basic Curriculum Training")
    print("=" * 50)
    
    if not check_requirements():
        return
    
    print("\n🚀 Starting quick 20-episode demo...")
    
    try:
        trainer = CurriculumTrainer(
            rom_path="../roms/pokemon_crystal.gbc",
            save_state_path=None,
            semantic_db_path=f"demo_curriculum_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
        )
        
        # Run short demo
        trainer.start_curriculum_training(max_total_episodes=20)
        
        print("\n✅ Demo completed!")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        quick_demo()
    else:
        main()
