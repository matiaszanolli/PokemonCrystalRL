#!/usr/bin/env python3
"""
Test run script for Pokemon Crystal RL training with comprehensive monitoring.
"""

import os
import sys
import argparse
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)

from llm_trainer import (
    LLMPokemonTrainer,
    LLMAgent,
    PokemonRewardCalculator
)

def main():
    """Run training test with monitoring."""
    parser = argparse.ArgumentParser(description="Test Pokemon Crystal RL training system")
    
    # Required arguments
    parser.add_argument("rom_path", type=str, help="Path to Pokemon Crystal ROM file")
    
    # Optional configuration
    parser.add_argument("--save-state", type=str, help="Path to save state file")
    parser.add_argument("--max-actions", type=int, default=10000,
                       help="Maximum number of actions to take")
    parser.add_argument("--dqn-model", type=str, help="Path to DQN model file")
    parser.add_argument("--web-port", type=int, default=8080,
                       help="Port for web monitoring interface")
    parser.add_argument("--output-dir", type=str, default="training_output",
                       help="Directory for output files")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    # Validate ROM path
    if not os.path.exists(args.rom_path):
        print(f"Error: ROM file not found at {args.rom_path}")
        return 1
        
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Configure trainer with optimal settings
    trainer = LLMPokemonTrainer(
        rom_path=args.rom_path,
        max_actions=args.max_actions,
        save_state=args.save_state,
        llm_model="smollm2:1.7b",
        llm_interval=20,  # Query LLM every 20 actions
        llm_temperature=0.7,
        enable_web=True,
        web_port=args.web_port,
        web_host="localhost",
        enable_dqn=True,
        dqn_model_path=args.dqn_model,
        dqn_learning_rate=1e-4,
        dqn_batch_size=32,
        dqn_memory_size=50000,
        dqn_training_frequency=4,
        dqn_save_frequency=500,
        log_dir=args.output_dir,
        show_progress=True
    )
    
    print("\n=== Training Configuration ===")
    print(f"ROM: {args.rom_path}")
    print(f"Maximum Actions: {args.max_actions}")
    print(f"Web Interface: http://localhost:{args.web_port}")
    print(f"Output Directory: {args.output_dir}")
    print(f"DQN Enabled: Yes")
    print(f"LLM Model: smollm2:1.7b")
    print("===========================\n")
    
    try:
        # Start training
        success = trainer.start_training()
        if success:
            print("\n✅ Training completed successfully")
            return 0
        else:
            print("\n❌ Training failed")
            return 1
            
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        trainer.shutdown(None, None)
        return 0
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
