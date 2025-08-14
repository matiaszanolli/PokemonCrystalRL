"""
train.py - RL training script using Stable Baselines3
"""

import os
import argparse
from datetime import datetime
import numpy as np
import torch

from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    CheckpointCallback, 
    EvalCallback,
    StopTrainingOnRewardThreshold
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from env import PokemonCrystalEnv
from utils import create_custom_cnn_policy


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train RL agent on Pokémon Crystal')
    
    parser.add_argument('--algorithm', type=str, default='ppo',
                       choices=['ppo', 'dqn', 'a2c'],
                       help='RL algorithm to use')
    
    parser.add_argument('--total-timesteps', type=int, default=1000000,
                       help='Total training timesteps')
    
    parser.add_argument('--emulator-path', type=str, default='/usr/games/mgba-qt',
                       help='Path to emulator executable')
    
    parser.add_argument('--rom-path', type=str, default='../pokecrystal.gbc',
                       help='Path to Pokémon Crystal ROM')
    
    parser.add_argument('--model-save-path', type=str, default='models',
                       help='Directory to save trained models')
    
    parser.add_argument('--log-path', type=str, default='logs',
                       help='Directory for tensorboard logs')
    
    parser.add_argument('--n-envs', type=int, default=1,
                       help='Number of parallel environments')
    
    parser.add_argument('--learning-rate', type=float, default=3e-4,
                       help='Learning rate')
    
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size for training')
    
    parser.add_argument('--eval-freq', type=int, default=10000,
                       help='Evaluation frequency')
    
    parser.add_argument('--save-freq', type=int, default=50000,
                       help='Model saving frequency')
    
    parser.add_argument('--resume-from', type=str, default=None,
                       help='Path to model to resume training from')
    
    parser.add_argument('--verbose', type=int, default=1,
                       help='Verbosity level')
    
    parser.add_argument('--run-id', type=int, default=None,
                       help='Training run ID for monitoring')
    
    return parser.parse_args()


def create_env(emulator_path: str, rom_path: str, rank: int = 0):
    """
    Create a single environment instance
    
    Args:
        emulator_path: Path to emulator
        rom_path: Path to ROM file
        rank: Environment rank (for multi-env setups)
    """
    def _init():
        env = PokemonCrystalEnv(
            emulator_path=emulator_path,
            rom_path=rom_path
        )
        env = Monitor(env, filename=None)
        return env
    
    return _init


def setup_model(algorithm: str, env, learning_rate: float, **kwargs):
    """
    Setup the RL model based on chosen algorithm
    
    Args:
        algorithm: Algorithm name ('ppo', 'dqn', 'a2c')
        env: Training environment
        learning_rate: Learning rate
        **kwargs: Additional algorithm-specific parameters
    """
    common_params = {
        'env': env,
        'learning_rate': learning_rate,
        'verbose': kwargs.get('verbose', 1),
        'tensorboard_log': kwargs.get('log_path', 'logs'),
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    if algorithm == 'ppo':
        model = PPO(
            policy='MlpPolicy',
            batch_size=kwargs.get('batch_size', 64),
            n_steps=2048,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            **common_params
        )
    
    elif algorithm == 'dqn':
        model = DQN(
            policy='MlpPolicy',
            buffer_size=100000,
            learning_starts=1000,
            batch_size=kwargs.get('batch_size', 32),
            target_update_interval=1000,
            exploration_fraction=0.1,
            exploration_final_eps=0.05,
            **common_params
        )
    
    elif algorithm == 'a2c':
        model = A2C(
            policy='MlpPolicy',
            n_steps=5,
            gamma=0.99,
            gae_lambda=1.0,
            ent_coef=0.01,
            **common_params
        )
    
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    return model


def setup_callbacks(model_save_path: str, eval_env, eval_freq: int, save_freq: int):
    """
    Setup training callbacks
    
    Args:
        model_save_path: Path to save models
        eval_env: Evaluation environment
        eval_freq: Evaluation frequency
        save_freq: Save frequency
    """
    callbacks = []
    
    # Checkpoint callback - save model periodically
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=model_save_path,
        name_prefix='pokemon_crystal_rl'
    )
    callbacks.append(checkpoint_callback)
    
    # Evaluation callback - evaluate performance periodically
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=model_save_path,
        log_path=model_save_path,
        eval_freq=eval_freq,
        deterministic=True,
        render=False
    )
    callbacks.append(eval_callback)
    
    # Stop training when reward threshold is reached (optional)
    # stop_callback = StopTrainingOnRewardThreshold(
    #     reward_threshold=1000,  # Adjust based on your reward scale
    #     verbose=1
    # )
    # callbacks.append(stop_callback)
    
    return callbacks


def main():
    """Main training function"""
    args = parse_args()
    
    # Create directories
    os.makedirs(args.model_save_path, exist_ok=True)
    os.makedirs(args.log_path, exist_ok=True)
    
    print(f"Training Pokémon Crystal RL agent")
    print(f"Algorithm: {args.algorithm}")
    print(f"Total timesteps: {args.total_timesteps}")
    print(f"Number of environments: {args.n_envs}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    # Create training environment(s)
    if args.n_envs == 1:
        env = DummyVecEnv([create_env(args.emulator_path, args.rom_path)])
    else:
        env = SubprocVecEnv([
            create_env(args.emulator_path, args.rom_path, i) 
            for i in range(args.n_envs)
        ])
    
    # Create evaluation environment
    eval_env = DummyVecEnv([create_env(args.emulator_path, args.rom_path)])
    
    # Setup model
    if args.resume_from:
        print(f"Resuming training from: {args.resume_from}")
        if args.algorithm == 'ppo':
            model = PPO.load(args.resume_from, env=env)
        elif args.algorithm == 'dqn':
            model = DQN.load(args.resume_from, env=env)
        elif args.algorithm == 'a2c':
            model = A2C.load(args.resume_from, env=env)
    else:
        model = setup_model(
            algorithm=args.algorithm,
            env=env,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            verbose=args.verbose,
            log_path=args.log_path
        )
    
    # Setup callbacks
    callbacks = setup_callbacks(
        model_save_path=args.model_save_path,
        eval_env=eval_env,
        eval_freq=args.eval_freq,
        save_freq=args.save_freq
    )
    
    # Start training
    print("Starting training...")
    start_time = datetime.now()
    
    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callbacks,
            tb_log_name=f"{args.algorithm}_{start_time.strftime('%Y%m%d_%H%M%S')}"
        )
    except KeyboardInterrupt:
        print("Training interrupted by user")
    
    end_time = datetime.now()
    training_time = end_time - start_time
    print(f"Training completed in: {training_time}")
    
    # Save final model
    final_model_path = os.path.join(
        args.model_save_path, 
        f"pokemon_crystal_{args.algorithm}_final"
    )
    model.save(final_model_path)
    print(f"Final model saved to: {final_model_path}")
    
    # Clean up
    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
