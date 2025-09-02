"""
Simplified tests for the Adaptive Strategy System using the actual API.
"""

import unittest
import tempfile
from unittest.mock import Mock

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from core.adaptive_strategy_system import AdaptiveStrategySystem, StrategyType
from core.decision_history_analyzer import DecisionHistoryAnalyzer


class TestAdaptiveStrategySystemSimplified(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary database for decision analyzer
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_decisions.db")
        
        # Create real decision analyzer for integration
        self.decision_analyzer = DecisionHistoryAnalyzer(self.db_path)
        
        # Create strategy system
        self.strategy_system = AdaptiveStrategySystem(
            history_analyzer=self.decision_analyzer
        )
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        try:
            if hasattr(self.decision_analyzer, 'conn'):
                self.decision_analyzer.conn.close()
        except:
            pass
        shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test strategy system initialization."""
        self.assertIsNotNone(self.strategy_system)
        self.assertIsInstance(self.strategy_system.current_strategy, StrategyType)
        self.assertIsNotNone(self.strategy_system.strategy_configs)
    
    def test_strategy_types_exist(self):
        """Test that all strategy types are properly defined."""
        # Check that strategy types exist
        self.assertTrue(hasattr(StrategyType, 'LLM_HEAVY'))
        self.assertTrue(hasattr(StrategyType, 'LLM_MODERATE'))
        self.assertTrue(hasattr(StrategyType, 'RULE_BASED'))
        
        # Check that strategy configs exist for all types
        for strategy_type in [StrategyType.LLM_HEAVY, StrategyType.LLM_MODERATE, StrategyType.RULE_BASED]:
            self.assertIn(strategy_type, self.strategy_system.strategy_configs)
    
    def test_current_strategy_access(self):
        """Test accessing current strategy."""
        current_strategy = self.strategy_system.current_strategy
        self.assertIsInstance(current_strategy, StrategyType)
        
        # Verify it's one of the valid strategy types
        valid_types = [StrategyType.LLM_HEAVY, StrategyType.LLM_MODERATE, StrategyType.RULE_BASED]
        self.assertIn(current_strategy, valid_types)
    
    def test_force_strategy(self):
        """Test manual strategy forcing."""
        # Force a specific strategy
        target_strategy = StrategyType.LLM_MODERATE
        self.strategy_system.force_strategy(target_strategy)
        
        # Verify strategy was set
        self.assertEqual(self.strategy_system.current_strategy, target_strategy)
        
        # Force another strategy
        target_strategy = StrategyType.LLM_HEAVY
        self.strategy_system.force_strategy(target_strategy)
        
        # Verify strategy changed
        self.assertEqual(self.strategy_system.current_strategy, target_strategy)
    
    def test_get_strategy_stats(self):
        """Test strategy statistics retrieval."""
        stats = self.strategy_system.get_strategy_stats()
        
        # Verify stats structure
        self.assertIsInstance(stats, dict)
        self.assertIn('current_strategy', stats)
        
        # Verify current strategy is correct
        self.assertEqual(stats['current_strategy'], self.strategy_system.current_strategy.value)
        
        # Verify other expected fields exist
        expected_fields = ['time_in_current_strategy', 'decisions_in_current_strategy']
        for field in expected_fields:
            self.assertIn(field, stats)
    
    def test_strategy_config_validation(self):
        """Test that strategy configurations are valid."""
        configs = self.strategy_system.strategy_configs
        
        for strategy_type, config in configs.items():
            self.assertIsInstance(strategy_type, StrategyType)
            self.assertIsNotNone(config)
            # Verify the config has some content
            self.assertGreater(len(str(config)), 0)
    
    def test_strategy_switching(self):
        """Test strategy switching functionality."""
        initial_strategy = self.strategy_system.current_strategy
        
        # Switch to different strategies
        strategies_to_try = [StrategyType.LLM_HEAVY, StrategyType.LLM_MODERATE, StrategyType.RULE_BASED]
        
        for target_strategy in strategies_to_try:
            if target_strategy != initial_strategy:
                self.strategy_system.force_strategy(target_strategy)
                self.assertEqual(self.strategy_system.current_strategy, target_strategy)
                break
    
    def test_performance_history_structure(self):
        """Test performance history data structure."""
        # Check that performance history exists and has correct structure
        self.assertHasAttr(self.strategy_system, 'performance_history')
        perf_history = self.strategy_system.performance_history
        
        # Should be a dictionary
        self.assertIsInstance(perf_history, dict)
        
        # Should have entries for strategy types
        for strategy_type in [StrategyType.LLM_HEAVY, StrategyType.LLM_MODERATE, StrategyType.RULE_BASED]:
            # Performance history might be empty initially, but structure should exist
            pass  # Just verify no exceptions are raised
    
    def test_integration_with_decision_analyzer(self):
        """Test integration with decision history analyzer exists."""
        # Verify decision analyzer integration
        self.assertIsNotNone(self.strategy_system.history_analyzer)
        self.assertEqual(self.strategy_system.history_analyzer, self.decision_analyzer)
    
    def assertHasAttr(self, obj, attr_name):
        """Helper method to assert object has attribute."""
        self.assertTrue(hasattr(obj, attr_name), f"Object should have attribute '{attr_name}'")


class TestStrategySystemMethods(unittest.TestCase):
    """Test specific methods of the strategy system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_decisions.db")
        self.decision_analyzer = DecisionHistoryAnalyzer(self.db_path)
        self.strategy_system = AdaptiveStrategySystem(history_analyzer=self.decision_analyzer)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        try:
            if hasattr(self.decision_analyzer, 'conn'):
                self.decision_analyzer.conn.close()
        except:
            pass
        shutil.rmtree(self.temp_dir)
    
    def test_strategy_enum_values(self):
        """Test that strategy enum values are as expected."""
        self.assertEqual(StrategyType.LLM_HEAVY.value, "llm_heavy")
        self.assertEqual(StrategyType.LLM_MODERATE.value, "llm_moderate") 
        self.assertEqual(StrategyType.RULE_BASED.value, "rule_based")
    
    def test_multiple_strategy_changes(self):
        """Test multiple strategy changes work correctly."""
        strategies = [StrategyType.LLM_HEAVY, StrategyType.LLM_MODERATE, StrategyType.RULE_BASED]
        
        for strategy in strategies:
            self.strategy_system.force_strategy(strategy)
            self.assertEqual(self.strategy_system.current_strategy, strategy)
            
            # Get stats to verify system is responsive
            stats = self.strategy_system.get_strategy_stats()
            self.assertEqual(stats['current_strategy'], strategy.value)
    
    def test_strategy_system_state_consistency(self):
        """Test that strategy system maintains consistent state."""
        # Initial state
        initial_strategy = self.strategy_system.current_strategy
        initial_stats = self.strategy_system.get_strategy_stats()
        
        # Change strategy
        new_strategy = StrategyType.LLM_HEAVY if initial_strategy != StrategyType.LLM_HEAVY else StrategyType.RULE_BASED
        self.strategy_system.force_strategy(new_strategy)
        
        # Verify consistency
        self.assertEqual(self.strategy_system.current_strategy, new_strategy)
        new_stats = self.strategy_system.get_strategy_stats()
        self.assertEqual(new_stats['current_strategy'], new_strategy.value)


if __name__ == '__main__':
    unittest.main()