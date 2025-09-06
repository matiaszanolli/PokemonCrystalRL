#!/usr/bin/env python3
"""
Unit tests for Goal-Oriented Planner
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from core.goal_oriented_planner import (
    GoalOrientedPlanner, Goal, GoalPriority, GoalStatus
)
from core.state.analyzer import (
    GameStateAnalysis, GamePhase, SituationCriticality
)


class TestGoalOrientedPlanner:
    """Test cases for Goal-Oriented Planner"""

    def setup_method(self):
        """Set up test fixtures"""
        self.planner = GoalOrientedPlanner()

    def test_initialization(self):
        """Test planner initialization"""
        assert len(self.planner.goals) > 0
        assert len(self.planner.active_goals) == 0
        
        # Check that emergency goals exist
        emergency_goals = [g for g in self.planner.goals.values() 
                         if g.priority == GoalPriority.EMERGENCY]
        assert len(emergency_goals) > 0

    def test_goal_registration(self):
        """Test goal registration"""
        initial_count = len(self.planner.goals)
        
        test_goal = Goal(
            id="test_goal",
            name="Test Goal",
            description="A test goal",
            priority=GoalPriority.LOW
        )
        
        self.planner.register_goal(test_goal)
        assert len(self.planner.goals) == initial_count + 1
        assert "test_goal" in self.planner.goals

    def test_emergency_goal_evaluation(self):
        """Test that emergency goals are prioritized"""
        # Create mock analysis with critical health
        mock_analysis = self._create_mock_analysis(health_percentage=5.0)
        
        active_goals = self.planner.evaluate_goals(mock_analysis)
        
        # Should prioritize emergency goals
        assert len(active_goals) > 0
        emergency_found = any(goal.priority == GoalPriority.EMERGENCY 
                            for goal in active_goals)
        assert emergency_found

    def test_early_game_goals(self):
        """Test early game goal selection"""
        mock_analysis = self._create_mock_analysis(
            phase=GamePhase.EARLY_GAME,
            health_percentage=80.0,
            party_size=0
        )
        
        active_goals = self.planner.evaluate_goals(mock_analysis)
        
        # Should include starter Pokemon goal
        goal_names = [goal.name for goal in active_goals]
        starter_related = any("starter" in name.lower() or "pokemon" in name.lower() 
                            for name in goal_names)
        assert starter_related

    def test_action_recommendations(self):
        """Test action recommendations"""
        mock_analysis = self._create_mock_analysis()
        
        recommendations = self.planner.get_recommended_actions(mock_analysis)
        
        assert isinstance(recommendations, list)
        if recommendations:
            # Each recommendation should be (action, reason, weight)
            for rec in recommendations:
                assert len(rec) == 3
                action, reason, weight = rec
                assert isinstance(action, str)
                assert isinstance(reason, str)
                assert isinstance(weight, (int, float))

    def test_goal_completion_detection(self):
        """Test goal completion detection and progress calculation"""
        # Test with starter Pokemon goal - start with no Pokemon
        starter_goal = self.planner.goals.get("obtain_starter_pokemon")
        if starter_goal:
            # First test: No Pokemon yet (goal should be applicable and active)
            mock_analysis_no_pokemon = self._create_mock_analysis(party_size=0)
            active_goals = self.planner.evaluate_goals(mock_analysis_no_pokemon)
            
            # Goal should be active when no Pokemon
            active_goal_ids = [g.id for g in active_goals]
            assert "obtain_starter_pokemon" in active_goal_ids
            
            # Second test: Now with Pokemon (goal should show completion)
            mock_analysis_with_pokemon = self._create_mock_analysis(party_size=1)
            
            # Directly test progress calculation
            progress = self.planner._calculate_goal_progress(starter_goal, mock_analysis_with_pokemon)
            assert progress == 100.0  # Should be 100% when we have Pokemon

    def test_strategy_summary(self):
        """Test strategy summary generation"""
        mock_analysis = self._create_mock_analysis()
        self.planner.evaluate_goals(mock_analysis)
        
        summary = self.planner.get_current_strategy_summary()
        
        assert isinstance(summary, str)
        assert len(summary) > 0

    def test_goal_statistics(self):
        """Test goal statistics"""
        stats = self.planner.get_goal_stats()
        
        required_keys = ["total_goals", "completed_goals", "active_goals", "completion_rate"]
        for key in required_keys:
            assert key in stats
        
        assert isinstance(stats["total_goals"], int)
        assert stats["total_goals"] > 0

    def _create_mock_analysis(self, **kwargs):
        """Create mock GameStateAnalysis for testing"""
        defaults = {
            'phase': GamePhase.EARLY_GAME,
            'criticality': SituationCriticality.MODERATE,
            'health_percentage': 80.0,
            'progression_score': 10.0,
            'exploration_score': 15.0,
            'immediate_threats': [],
            'opportunities': [],
            'recommended_priorities': [],
            'situation_summary': "Test situation",
            'strategic_context': "Test context",
            'risk_assessment': "Low risk",
            'state_variables': {}
        }
        
        # Add party_size to state_variables if provided
        if 'party_size' in kwargs:
            from core.state.analyzer import AnalysisStateVariable as StateVariable
            defaults['state_variables']['party_size'] = StateVariable(
                name='party_size',
                type='int',
                current_value=kwargs['party_size'],
                normal_range=(0, 6),
                critical_thresholds={},
                impact_on_rewards=[],
                impact_on_survival=0.0,
                description="Number of Pokemon in party"
            )
            # Remove party_size from kwargs so it doesn't get passed to GameStateAnalysis
            kwargs = {k: v for k, v in kwargs.items() if k != 'party_size'}
        
        # Override with any provided kwargs (except party_size which is handled above)
        defaults.update(kwargs)
        
        return GameStateAnalysis(**defaults)


if __name__ == "__main__":
    # Run tests if called directly
    pytest.main([__file__, "-v"])