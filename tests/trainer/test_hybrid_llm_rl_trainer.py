"""
Tests for the Hybrid LLM-RL Trainer.
"""

import unittest
import tempfile
import json
import numpy as np
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from trainer.hybrid_llm_rl_trainer import HybridLLMRLTrainer, create_trainer_from_config


class TestHybridLLMRLTrainer(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock environment
        self.mock_env = Mock()
        self.mock_env.reset.return_value = (np.zeros((144, 160, 3)), {'game_state': {}})
        self.mock_env.step.return_value = (
            np.zeros((144, 160, 3)),  # next_obs
            1.0,  # reward
            False,  # terminated
            False,  # truncated
            {'game_state': {}}  # info
        )
        self.mock_env.action_space = Mock()
        self.mock_env.action_space.n = 8
        
        # Mock agent
        self.mock_agent = Mock()
        self.mock_agent.get_action.return_value = (0, {'source': 'rl', 'confidence': 0.8})
        self.mock_agent.update.return_value = None
        self.mock_agent.get_state_dict.return_value = {'test': 'state'}
        self.mock_agent.load_state_dict.return_value = None
        self.mock_agent.llm_confidence_threshold = 0.7
        
        # Mock strategy system
        self.mock_strategy_system = Mock()
        self.mock_strategy_system.current_strategy = 'llm_heavy'
        self.mock_strategy_system.evaluate_performance.return_value = None
        
        # Mock decision analyzer
        self.mock_decision_analyzer = Mock()
        self.mock_decision_analyzer.add_decision.return_value = None
        
        # Mock LLM manager
        self.mock_llm_manager = Mock()
        
        # Create temporary directory for saves
        self.temp_dir = tempfile.mkdtemp()
        
        # Create trainer
        self.trainer = HybridLLMRLTrainer(
            env=self.mock_env,
            agent=self.mock_agent,
            strategy_system=self.mock_strategy_system,
            decision_analyzer=self.mock_decision_analyzer,
            llm_manager=self.mock_llm_manager,
            save_dir=self.temp_dir
        )
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_trainer_initialization(self):
        """Test trainer initialization."""
        self.assertIsNotNone(self.trainer)
        self.assertEqual(self.trainer.curriculum_stage, 0)
        self.assertEqual(len(self.trainer.episode_rewards), 0)
        self.assertEqual(self.trainer.training_stats['episodes'], 0)
    
    def test_single_episode_training(self):
        """Test single episode training loop."""
        # Configure environment to terminate after one step
        self.mock_env.step.return_value = (
            np.zeros((144, 160, 3)),
            10.0,  # reward
            True,   # terminated
            False,
            {'game_state': {}}
        )
        
        # Run single episode
        summary = self.trainer.train(total_episodes=1, max_steps_per_episode=100)
        
        # Verify training occurred
        self.assertEqual(self.trainer.training_stats['episodes'], 1)
        self.assertEqual(len(self.trainer.episode_rewards), 1)
        self.assertEqual(self.trainer.episode_rewards[0], 10.0)
        
        # Verify summary structure
        self.assertIn('total_episodes', summary)
        self.assertIn('best_reward', summary)
        self.assertIn('final_evaluation', summary)
    
    def test_curriculum_advancement(self):
        """Test curriculum learning advancement."""
        # Set up high reward episodes to trigger advancement
        self.mock_env.step.return_value = (
            np.zeros((144, 160, 3)),
            60.0,  # High reward to exceed first threshold (50.0)
            True,
            False,
            {'game_state': {}}
        )
        
        # Run enough episodes for curriculum advancement
        self.trainer.train(total_episodes=15, max_steps_per_episode=10)
        
        # Should have advanced to stage 1
        self.assertGreater(self.trainer.curriculum_stage, 0)
        self.assertGreater(self.trainer.training_stats['curriculum_advancements'], 0)
    
    def test_strategy_system_integration(self):
        """Test integration with adaptive strategy system."""
        # Run training
        self.trainer.train(total_episodes=5, max_steps_per_episode=10)
        
        # Verify strategy system was called
        self.mock_strategy_system.evaluate_performance.assert_called()
        
        # Verify metrics were passed to strategy system
        call_args = self.mock_strategy_system.evaluate_performance.call_args_list[-1][0][0]
        self.assertIn('episode_reward', call_args)
        self.assertIn('average_reward', call_args)
        self.assertIn('llm_usage_rate', call_args)
    
    def test_decision_analysis_integration(self):
        """Test integration with decision history analyzer."""
        # Run training
        self.trainer.train(total_episodes=2, max_steps_per_episode=5)
        
        # Verify decisions were recorded
        self.mock_decision_analyzer.add_decision.assert_called()
        
        # Check decision data structure
        call_args = self.mock_decision_analyzer.add_decision.call_args_list[0][0][0]
        self.assertIn('state_hash', call_args)
        self.assertIn('action', call_args)
        self.assertIn('context', call_args)
        self.assertIn('outcome', call_args)
    
    def test_llm_usage_tracking(self):
        """Test LLM usage rate tracking."""
        # Configure agent to use LLM decisions
        self.mock_agent.get_action.return_value = (1, {'source': 'llm', 'confidence': 0.9})
        
        # Run training
        self.trainer.train(total_episodes=1, max_steps_per_episode=10)
        
        # Verify LLM usage was tracked
        self.assertEqual(len(self.trainer.llm_usage_rates), 1)
        self.assertGreater(self.trainer.llm_usage_rates[0], 0)
    
    def test_checkpoint_saving_and_loading(self):
        """Test checkpoint saving and loading."""
        # Run some training
        self.trainer.train(total_episodes=2, max_steps_per_episode=5)
        
        # Save checkpoint
        self.trainer._save_checkpoint(2)
        
        # Verify checkpoint file exists
        checkpoint_files = list(Path(self.temp_dir).glob("checkpoint_*.pt"))
        self.assertTrue(len(checkpoint_files) > 0)
        
        # Create new trainer and load checkpoint
        new_trainer = HybridLLMRLTrainer(
            env=self.mock_env,
            agent=self.mock_agent,
            strategy_system=self.mock_strategy_system,
            decision_analyzer=self.mock_decision_analyzer,
            llm_manager=self.mock_llm_manager,
            save_dir=self.temp_dir
        )
        
        # Load checkpoint
        success = new_trainer.load_checkpoint(str(checkpoint_files[0]))
        self.assertTrue(success)
        
        # Verify state was loaded
        self.assertEqual(new_trainer.training_stats['episodes'], 2)
    
    def test_evaluation_mode(self):
        """Test evaluation mode with reduced exploration."""
        # Configure agent with exploration rate
        self.mock_agent.exploration_rate = 0.3
        original_rate = self.mock_agent.exploration_rate
        
        # Run evaluation
        results = self.trainer._evaluate_agent(num_eval_episodes=2)
        
        # Verify evaluation ran
        self.assertIn('avg_reward', results)
        self.assertIn('avg_length', results)
        self.assertIn('avg_llm_usage', results)
        
        # Verify exploration was restored after evaluation
        # (The evaluation method should restore the original rate)
        self.assertEqual(self.mock_agent.exploration_rate, original_rate)
    
    def test_best_model_saving(self):
        """Test best model saving when new best reward is achieved."""
        # Set up improving rewards - need more for evaluation episodes too
        rewards = [5.0, 10.0, 15.0, 8.0] * 10  # Multiply to handle evaluation
        reward_iter = iter(rewards)
        
        def mock_step(*args):
            try:
                reward = next(reward_iter)
            except StopIteration:
                reward = 1.0  # Default reward when iterator exhausted
            return (
                np.zeros((144, 160, 3)),
                reward,
                True,  # terminated
                False,
                {'game_state': {}}
            )
        
        self.mock_env.step.side_effect = mock_step
        
        # Run training without evaluation to avoid StopIteration
        self.trainer.train(total_episodes=4, max_steps_per_episode=10, eval_interval=100)
        
        # Verify best model was saved
        best_model_path = Path(self.temp_dir) / "best_model.pt"
        self.assertTrue(best_model_path.exists())
        
        # Verify best reward was tracked
        self.assertEqual(self.trainer.training_stats['best_reward'], 15.0)
    
    def test_training_summary_generation(self):
        """Test comprehensive training summary generation."""
        # Run training
        summary = self.trainer.train(total_episodes=3, max_steps_per_episode=5)
        
        # Verify summary structure
        required_keys = [
            'total_episodes', 'total_steps', 'best_reward',
            'final_avg_reward', 'curriculum_stage_reached',
            'final_evaluation', 'avg_episode_length', 'avg_llm_usage'
        ]
        
        for key in required_keys:
            self.assertIn(key, summary)
        
        # Verify summary file was created
        summary_path = Path(self.temp_dir) / "training_summary.json"
        self.assertTrue(summary_path.exists())
        
        # Verify summary file content
        with open(summary_path, 'r') as f:
            file_summary = json.load(f)
        
        for key in required_keys:
            self.assertIn(key, file_summary)
    
    def test_llm_confidence_decay(self):
        """Test gradual reduction of LLM confidence for curriculum learning."""
        initial_threshold = self.mock_agent.llm_confidence_threshold
        
        # Run training
        self.trainer.train(total_episodes=5, max_steps_per_episode=5)
        
        # Verify confidence threshold was reduced
        self.assertLess(
            self.mock_agent.llm_confidence_threshold,
            initial_threshold
        )


class TestTrainerConfiguration(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = Path(self.temp_dir) / "test_config.json"
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    @patch('trainer.hybrid_llm_rl_trainer.EnhancedPyBoyPokemonCrystalEnv')
    @patch('trainer.hybrid_llm_rl_trainer.LLMManager')
    @patch('trainer.hybrid_llm_rl_trainer.DecisionHistoryAnalyzer')
    @patch('trainer.hybrid_llm_rl_trainer.AdaptiveStrategySystem')
    @patch('trainer.hybrid_llm_rl_trainer.HybridAgent')
    def test_create_trainer_from_config(self, mock_agent, mock_strategy, mock_analyzer, mock_llm, mock_env):
        """Test trainer creation from configuration file."""
        # Create test configuration
        config = {
            'rom_path': 'test.gbc',
            'headless': True,
            'observation_type': 'multi_modal',
            'llm_model': 'gpt-4',
            'max_context_length': 8000,
            'initial_strategy': 'balanced',
            'decision_db_path': 'test.db',
            'save_dir': 'test_checkpoints',
            'log_level': 'DEBUG'
        }
        
        # Save configuration
        with open(self.config_path, 'w') as f:
            json.dump(config, f)
        
        # Mock the components
        mock_env.return_value = Mock()
        mock_env.return_value.action_space = Mock()
        mock_llm.return_value = Mock()
        mock_analyzer.return_value = Mock()
        mock_strategy.return_value = Mock()
        mock_agent.return_value = Mock()
        
        # Create trainer from config
        trainer = create_trainer_from_config(str(self.config_path))
        
        # Verify trainer was created
        self.assertIsNotNone(trainer)
        
        # Verify components were initialized with correct parameters
        mock_env.assert_called_once_with(
            rom_path='test.gbc',
            headless=True,
            observation_type='multi_modal'
        )
        
        mock_llm.assert_called_once_with(
            model_name='gpt-4',
            max_context_length=8000
        )
        
        mock_analyzer.assert_called_once_with(db_path='test.db')
        mock_strategy.assert_called_once_with(initial_strategy='balanced')
    
    def test_invalid_config_file(self):
        """Test handling of invalid configuration file."""
        # Create invalid config
        with open(self.config_path, 'w') as f:
            f.write("invalid json content")
        
        # Should raise exception
        with self.assertRaises(json.JSONDecodeError):
            create_trainer_from_config(str(self.config_path))


if __name__ == '__main__':
    unittest.main()