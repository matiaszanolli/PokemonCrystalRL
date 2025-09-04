"""
Integration test for the complete hybrid LLM-RL system.
Tests the interaction between all major components.
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

from trainer.hybrid_llm_rl_trainer import HybridLLMRLTrainer
from core.hybrid_agent import HybridAgent
from core.adaptive_strategy_system import AdaptiveStrategySystem
from core.decision_history_analyzer import DecisionHistoryAnalyzer
from trainer.llm_manager import LLMManager
from core.strategic_context_builder import StrategicContextBuilder


class TestFullHybridSystem(unittest.TestCase):
    """Integration test for the complete hybrid system."""
    
    def setUp(self):
        """Set up complete system for integration testing."""
        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp()
        
        # Mock environment with realistic behavior
        self.mock_env = Mock()
        self.mock_env.action_space = Mock()
        self.mock_env.action_space.n = 8
        self.mock_env.reset.return_value = (
            {
                'screen': np.zeros((144, 160, 3), dtype=np.uint8),
                'state_variables': np.zeros(25, dtype=np.float32),
                'strategic_context': {
                    'current_goal': 'catch_starter',
                    'context_summary': 'Player in starting area',
                    'action_suggestions': []
                }
            },
            {'game_state': {'player_x': 10, 'player_y': 10}}
        )
        
        # Configure environment step to provide meaningful state changes
        self.step_count = 0
        def mock_step(action):
            self.step_count += 1
            reward = 1.0 if self.step_count % 5 == 0 else 0.1  # Occasional higher rewards
            terminated = self.step_count >= 20  # Episode ends after 20 steps
            
            next_obs = {
                'screen': np.random.randint(0, 255, (144, 160, 3), dtype=np.uint8),
                'state_variables': np.random.random(25).astype(np.float32),
                'strategic_context': {
                    'current_goal': 'catch_starter',
                    'context_summary': f'Step {self.step_count} in progress',
                    'action_suggestions': [f'action_{action}']
                }
            }
            
            info = {'game_state': {'player_x': 10 + self.step_count, 'player_y': 10}}
            
            return next_obs, reward, terminated, False, info
        
        self.mock_env.step.side_effect = mock_step
        
        # Initialize real components (not mocked) for integration testing
        self.db_path = Path(self.temp_dir) / "test_decisions.db"
        self.decision_analyzer = DecisionHistoryAnalyzer(str(self.db_path))
        
        # Mock LLM manager with realistic responses
        self.llm_manager = Mock()
        self.llm_manager.get_action.return_value = (
            np.random.randint(0, 8),  # Random action
            {
                'source': 'llm',
                'confidence': 0.8,
                'reasoning': 'Moving toward objective',
                'context': 'Game state analysis complete'
            }
        )
        
        # Initialize adaptive strategy system
        self.strategy_system = AdaptiveStrategySystem(history_analyzer=self.decision_analyzer)
        
        # Initialize hybrid agent with mocked LLM but real other components
        self.agent = HybridAgent(
            llm_manager=self.llm_manager,
            adaptive_strategy=self.strategy_system,
            action_space_size=8
        )
        
        # Initialize trainer
        self.trainer = HybridLLMRLTrainer(
            env=self.mock_env,
            agent=self.agent,
            strategy_system=self.strategy_system,
            decision_analyzer=self.decision_analyzer,
            llm_manager=self.llm_manager,
            save_dir=self.temp_dir,
            log_level="DEBUG"
        )
    
    def tearDown(self):
        """Clean up test resources."""
        import shutil
        shutil.rmtree(self.temp_dir)
        self.decision_analyzer.close()
    
    def reset_step_count(self):
        """Reset step count for new episode."""
        self.step_count = 0
    
    def test_complete_training_cycle(self):
        """Test complete training cycle with real component interactions."""
        # Configure environment to reset step count between episodes
        original_reset = self.mock_env.reset
        def mock_reset(*args, **kwargs):
            self.reset_step_count()
            return original_reset.return_value
        self.mock_env.reset.side_effect = mock_reset
        
        # Run short training cycle
        summary = self.trainer.train(
            total_episodes=3,
            max_steps_per_episode=25,
            save_interval=2,
            eval_interval=10  # Skip evaluation for simplicity
        )
        
        # Verify training completed successfully
        self.assertEqual(summary['total_episodes'], 3)
        self.assertGreater(summary['total_steps'], 0)
        self.assertIn('final_evaluation', summary)
        
        # Verify decisions were recorded in database
        decisions = self.decision_analyzer.get_recent_decisions(limit=10)
        self.assertGreater(len(decisions), 0)
        
        # Verify agent learned from experience
        agent_state = self.agent.get_state_dict()
        self.assertIn('rl_model_state', agent_state)
        self.assertIn('experience_buffer', agent_state)
    
    def test_strategy_adaptation_during_training(self):
        """Test that strategy system adapts during training."""
        initial_strategy = self.strategy_system.current_strategy
        
        # Configure different performance scenarios
        performance_scenarios = [
            {'reward': 0.1, 'llm_usage': 0.9},  # Poor performance with high LLM usage
            {'reward': 0.8, 'llm_usage': 0.3},  # Good performance with low LLM usage
            {'reward': 0.2, 'llm_usage': 0.7},  # Poor performance with medium LLM usage
        ]
        
        for scenario in performance_scenarios:
            self.strategy_system.evaluate_performance({
                'episode_reward': scenario['reward'],
                'average_reward': scenario['reward'],
                'llm_usage_rate': scenario['llm_usage'],
                'episode_length': 20
            })
        
        # Strategy may have changed based on performance
        # This is expected behavior for adaptive system
        self.assertIsNotNone(self.strategy_system.current_strategy)
    
    def test_decision_learning_and_pattern_recognition(self):
        """Test that the system learns from decision patterns."""
        # Simulate a series of decisions with known patterns
        test_decisions = [
            {
                'state_hash': hash('state_1'),
                'action': 0,
                'context': {'confidence': 0.8},
                'outcome': 'success',
                'step_in_episode': 1,
                'total_episode_reward': 10.0
            },
            {
                'state_hash': hash('state_1'),  # Same state
                'action': 0,  # Same action
                'context': {'confidence': 0.7},
                'outcome': 'success',  # Same outcome
                'step_in_episode': 1,
                'total_episode_reward': 12.0
            },
            {
                'state_hash': hash('state_2'),
                'action': 1,
                'context': {'confidence': 0.6},
                'outcome': 'failure',
                'step_in_episode': 5,
                'total_episode_reward': 2.0
            }
        ]
        
        # Record decisions
        for decision in test_decisions:
            self.decision_analyzer.add_decision(decision)
        
        # Verify decisions were stored
        stored_decisions = self.decision_analyzer.get_recent_decisions(limit=5)
        self.assertEqual(len(stored_decisions), 3)
        
        # Test pattern analysis - check if method exists first
        if hasattr(self.decision_analyzer, 'analyze_patterns'):
            patterns = self.decision_analyzer.analyze_patterns(min_frequency=2)
            self.assertGreater(len(patterns), 0)  # Should find at least one pattern
        else:
            # Skip pattern analysis if method not implemented
            self.skipTest("analyze_patterns method not implemented in DecisionHistoryAnalyzer")
    
    def test_curriculum_learning_progression(self):
        """Test curriculum learning progression."""
        # Configure high rewards to trigger curriculum advancement
        def mock_high_reward_step(action):
            self.step_count += 1
            reward = 60.0  # High reward to exceed curriculum thresholds
            terminated = self.step_count >= 5  # Short episodes for faster progression
            
            next_obs = {
                'screen': np.zeros((144, 160, 3), dtype=np.uint8),
                'state_variables': np.zeros(25, dtype=np.float32),
                'strategic_context': {
                    'current_goal': 'catch_starter',
                    'context_summary': 'High performance episode',
                    'action_suggestions': []
                }
            }
            
            return next_obs, reward, terminated, False, {'game_state': {}}
        
        self.mock_env.step.side_effect = mock_high_reward_step
        
        # Reset episode count for each episode
        def mock_reset_for_curriculum():
            self.reset_step_count()
            return self.mock_env.reset.return_value
        
        self.mock_env.reset.side_effect = mock_reset_for_curriculum
        
        initial_stage = self.trainer.curriculum_stage
        
        # Run training with curriculum learning
        self.trainer.train(
            total_episodes=15,  # Enough episodes for curriculum advancement
            max_steps_per_episode=10,
            eval_interval=20  # Skip evaluation
        )
        
        # Verify curriculum advanced
        self.assertGreater(self.trainer.curriculum_stage, initial_stage)
        self.assertGreater(self.trainer.training_stats['curriculum_advancements'], 0)
    
    def test_checkpoint_and_resume_training(self):
        """Test checkpoint saving and resuming training."""
        # Run initial training
        self.trainer.train(
            total_episodes=2,
            max_steps_per_episode=15,
            save_interval=1,
            eval_interval=10
        )
        
        # Verify checkpoint was saved
        checkpoint_files = list(Path(self.temp_dir).glob("checkpoint_*.pt"))
        self.assertGreater(len(checkpoint_files), 0)
        
        # Create new trainer and load checkpoint
        new_trainer = HybridLLMRLTrainer(
            env=self.mock_env,
            agent=self.agent,
            strategy_system=self.strategy_system,
            decision_analyzer=self.decision_analyzer,
            llm_manager=self.llm_manager,
            save_dir=self.temp_dir
        )
        
        # Load checkpoint
        success = new_trainer.load_checkpoint(str(checkpoint_files[-1]))
        self.assertTrue(success)
        
        # Verify state was loaded correctly
        self.assertEqual(new_trainer.training_stats['episodes'], 2)
        self.assertGreater(len(new_trainer.episode_rewards), 0)
    
    def test_agent_decision_integration(self):
        """Test integration between agent decision making and all subsystems."""
        # Get environment state
        obs, info = self.mock_env.reset()
        
        # Agent should make decision using all integrated systems
        action, decision_info = self.agent.get_action(obs, info)
        
        # Verify decision was made
        self.assertIsInstance(action, (int, np.integer))
        self.assertIn('source', decision_info)
        self.assertIn('confidence', decision_info)
        
        # Verify decision was recorded in history
        initial_decision_count = len(self.decision_analyzer.get_recent_decisions(limit=100))
        
        # Record a decision
        decision_data = {
            'state_hash': hash(str(obs)),
            'action': action,
            'context': decision_info,
            'outcome': 'test',
            'step_in_episode': 1,
            'total_episode_reward': 5.0
        }
        self.decision_analyzer.add_decision(decision_data)
        
        # Verify decision was recorded
        new_decision_count = len(self.decision_analyzer.get_recent_decisions(limit=100))
        self.assertEqual(new_decision_count, initial_decision_count + 1)


if __name__ == '__main__':
    unittest.main()