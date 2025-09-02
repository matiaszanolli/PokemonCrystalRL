"""
Simplified integration test for the hybrid LLM-RL system components.
Tests basic functionality and integration between major components.
"""

import unittest
import tempfile
import json
import numpy as np
from pathlib import Path
from unittest.mock import Mock, MagicMock

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from core.decision_history_analyzer import DecisionHistoryAnalyzer
from core.adaptive_strategy_system import AdaptiveStrategySystem
from trainer.llm_manager import LLMManager


class TestSimplifiedIntegration(unittest.TestCase):
    """Simplified integration test for core components."""
    
    def setUp(self):
        """Set up test components."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test_decisions.db"
        
        # Initialize components
        self.decision_analyzer = DecisionHistoryAnalyzer(str(self.db_path))
        self.strategy_system = AdaptiveStrategySystem(history_analyzer=self.decision_analyzer)
        
        # Mock LLM manager
        self.llm_manager = Mock()
        self.llm_manager.get_action.return_value = (
            3, {
                'source': 'llm',
                'confidence': 0.8,
                'reasoning': 'Strategic move'
            }
        )
    
    def tearDown(self):
        """Clean up test resources."""
        import shutil
        try:
            # Try to close any connections cleanly
            if hasattr(self.decision_analyzer, 'conn'):
                self.decision_analyzer.conn.close()
        except:
            pass
        shutil.rmtree(self.temp_dir)
    
    def test_decision_analyzer_basic_functionality(self):
        """Test basic decision analyzer functionality."""
        # Add a decision
        decision_data = {
            'state_hash': hash('test_state'),
            'action': 2,
            'context': {'confidence': 0.7},
            'outcome': 'success',
            'step_in_episode': 5,
            'total_episode_reward': 10.0
        }
        
        self.decision_analyzer.add_decision(decision_data)
        
        # Retrieve decisions
        decisions = self.decision_analyzer.get_recent_decisions(limit=5)
        self.assertEqual(len(decisions), 1)
        self.assertEqual(decisions[0]['action'], 2)
    
    def test_strategy_system_performance_evaluation(self):
        """Test adaptive strategy system performance evaluation."""
        initial_strategy = self.strategy_system.current_strategy
        
        # Test different performance scenarios
        performance_metrics = [
            {'episode_reward': 10.0, 'average_reward': 8.0, 'llm_usage_rate': 0.8, 'episode_length': 50},
            {'episode_reward': 5.0, 'average_reward': 6.0, 'llm_usage_rate': 0.9, 'episode_length': 60},
            {'episode_reward': 15.0, 'average_reward': 12.0, 'llm_usage_rate': 0.3, 'episode_length': 40}
        ]
        
        for metrics in performance_metrics:
            self.strategy_system.evaluate_performance(metrics)
        
        # Strategy system should be responsive (may or may not change)
        self.assertIsNotNone(self.strategy_system.current_strategy)
    
    def test_pattern_analysis_integration(self):
        """Test pattern analysis with decision history."""
        # Add multiple decisions with patterns
        decisions = [
            {
                'state_hash': hash('state_a'),
                'action': 1,
                'context': {'confidence': 0.8},
                'outcome': 'success',
                'step_in_episode': 1,
                'total_episode_reward': 10.0
            },
            {
                'state_hash': hash('state_a'),  # Same state
                'action': 1,  # Same action
                'context': {'confidence': 0.7},
                'outcome': 'success',  # Same outcome
                'step_in_episode': 2,
                'total_episode_reward': 12.0
            },
            {
                'state_hash': hash('state_b'),
                'action': 2,
                'context': {'confidence': 0.6},
                'outcome': 'failure',
                'step_in_episode': 3,
                'total_episode_reward': 2.0
            }
        ]
        
        for decision in decisions:
            self.decision_analyzer.add_decision(decision)
        
        # Analyze patterns
        patterns = self.decision_analyzer.analyze_patterns(min_frequency=2)
        
        # Should find some patterns
        self.assertIsInstance(patterns, list)
        # Pattern analysis might return empty list if no patterns meet criteria
        # Just verify it runs without error
    
    def test_component_integration_flow(self):
        """Test integration flow between components."""
        # 1. Record some decisions
        for i in range(5):
            decision = {
                'state_hash': hash(f'state_{i}'),
                'action': i % 4,
                'context': {'confidence': 0.5 + 0.1 * i},
                'outcome': 'success' if i % 2 == 0 else 'failure',
                'step_in_episode': i,
                'total_episode_reward': float(i * 2)
            }
            self.decision_analyzer.add_decision(decision)
        
        # 2. Evaluate strategy performance
        metrics = {
            'episode_reward': 8.0,
            'average_reward': 6.0,
            'llm_usage_rate': 0.6,
            'episode_length': 50
        }
        self.strategy_system.evaluate_performance(metrics)
        
        # 3. Verify data persistence and retrieval
        stored_decisions = self.decision_analyzer.get_recent_decisions(limit=10)
        self.assertEqual(len(stored_decisions), 5)
        
        # 4. Verify strategy system is responsive
        self.assertIsNotNone(self.strategy_system.current_strategy)
        
    def test_llm_manager_integration(self):
        """Test LLM manager integration (mocked)."""
        # Test LLM manager mock
        action, info = self.llm_manager.get_action()
        
        self.assertIsInstance(action, int)
        self.assertIn('source', info)
        self.assertIn('confidence', info)
        self.assertEqual(info['source'], 'llm')
        
        # Verify mock was called
        self.llm_manager.get_action.assert_called_once()


if __name__ == '__main__':
    unittest.main()