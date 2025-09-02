#!/usr/bin/env python3
"""
Example script demonstrating hybrid LLM-RL training for Pokemon Crystal.
"""

import json
import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from trainer.hybrid_llm_rl_trainer import create_trainer_from_config


def create_example_config():
    """Create an example training configuration."""
    config = {
        "rom_path": "pokemoncrystal.gbc",
        "headless": True,
        "observation_type": "multi_modal",
        "llm_model": "gpt-4",
        "max_context_length": 8000,
        "initial_strategy": "llm_heavy",
        "decision_db_path": "pokemon_decisions.db",
        "save_dir": "training_checkpoints",
        "log_level": "INFO"
    }
    return config


def main():
    """Main training execution."""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Create configuration
    config = create_example_config()
    config_path = "hybrid_training_config.json"
    
    # Save configuration file
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Created training configuration: {config_path}")
    logger.info("Configuration settings:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    
    try:
        # Create trainer from configuration
        logger.info("Initializing hybrid LLM-RL trainer...")
        trainer = create_trainer_from_config(config_path)
        
        # Training parameters
        training_params = {
            'total_episodes': 1000,
            'max_steps_per_episode': 10000,
            'save_interval': 100,
            'eval_interval': 50,
            'curriculum_patience': 20
        }
        
        logger.info("Starting training with parameters:")
        for key, value in training_params.items():
            logger.info(f"  {key}: {value}")
        
        # Run training
        logger.info("="*50)
        logger.info("STARTING HYBRID LLM-RL TRAINING")
        logger.info("="*50)
        
        training_summary = trainer.train(**training_params)
        
        # Display results
        logger.info("="*50)
        logger.info("TRAINING COMPLETED")
        logger.info("="*50)
        
        logger.info("Training Summary:")
        logger.info(f"  Total Episodes: {training_summary['total_episodes']}")
        logger.info(f"  Total Steps: {training_summary['total_steps']}")
        logger.info(f"  Best Reward: {training_summary['best_reward']:.2f}")
        logger.info(f"  Final Average Reward: {training_summary['final_avg_reward']:.2f}")
        logger.info(f"  Curriculum Stage Reached: {training_summary['curriculum_stage_reached']}")
        logger.info(f"  Strategy Switches: {training_summary['strategy_switches']}")
        logger.info(f"  Average Episode Length: {training_summary['avg_episode_length']:.1f}")
        logger.info(f"  Average LLM Usage: {training_summary['avg_llm_usage']:.1%}")
        
        # Final evaluation results
        final_eval = training_summary['final_evaluation']
        logger.info("Final Evaluation Results:")
        logger.info(f"  Average Reward: {final_eval['avg_reward']:.2f} ± {final_eval['std_reward']:.2f}")
        logger.info(f"  Average Episode Length: {final_eval['avg_length']:.1f}")
        logger.info(f"  Average LLM Usage: {final_eval['avg_llm_usage']:.1%}")
        
        logger.info(f"Checkpoints saved to: {config['save_dir']}")
        logger.info(f"Training summary saved to: {config['save_dir']}/training_summary.json")
        
    except FileNotFoundError as e:
        logger.error(f"ROM file not found: {e}")
        logger.error("Please ensure pokemoncrystal.gbc is available in the current directory")
        return 1
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        logger.exception("Full error details:")
        return 1
    
    logger.info("Training completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())