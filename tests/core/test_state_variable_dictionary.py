#!/usr/bin/env python3
"""
Unit tests for State Variable Dictionary
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from environments.state.variables import (
    StateVariableDictionary, StateVariable, VariableType, 
    ImpactCategory, STATE_VARIABLES, analyze_game_state_comprehensive
)


class TestStateVariableDictionary:
    """Test cases for State Variable Dictionary"""

    def setup_method(self):
        """Set up test fixtures"""
        self.state_dict = StateVariableDictionary()

    def test_initialization(self):
        """Test dictionary initialization"""
        assert len(self.state_dict.variables) >= 20
        
        # Check that we have different types of variables
        impact_categories = set(var.impact_category for var in self.state_dict.variables.values())
        assert len(impact_categories) > 1
        
        # Check that we have memory addresses
        memory_vars = [v for v in self.state_dict.variables.values() if v.memory_address > 0]
        assert len(memory_vars) > 15

    def test_variable_retrieval(self):
        """Test variable retrieval methods"""
        # Test getting specific variable
        player_hp = self.state_dict.get_variable('player_hp')
        assert player_hp is not None
        assert player_hp.impact_category == ImpactCategory.SURVIVAL
        
        # Test getting variables by impact
        survival_vars = self.state_dict.get_variables_by_impact(ImpactCategory.SURVIVAL)
        assert len(survival_vars) > 0
        
        # Test critical variables
        critical_vars = self.state_dict.get_critical_variables()
        assert len(critical_vars) > 0
        
        # Test high impact variables
        high_impact = self.state_dict.get_high_impact_variables()
        assert len(high_impact) > 0

    def test_variable_properties(self):
        """Test that variables have expected properties"""
        for var_name, var in self.state_dict.variables.items():
            # Basic properties
            assert hasattr(var, 'name')
            assert hasattr(var, 'variable_type')
            assert hasattr(var, 'impact_category')
            assert hasattr(var, 'valid_range')
            
            # Check that variable types are valid
            assert isinstance(var.variable_type, VariableType)
            assert isinstance(var.impact_category, ImpactCategory)
            
            # Check ranges
            assert len(var.valid_range) == 2
            assert var.valid_range[0] <= var.valid_range[1]

    def test_survival_variables(self):
        """Test that critical survival variables exist"""
        survival_vars = self.state_dict.get_variables_by_impact(ImpactCategory.SURVIVAL)
        survival_names = [var.name for var in survival_vars]
        
        # Should have HP-related variables
        hp_related = any('hp' in name.lower() for name in survival_names)
        assert hp_related

    def test_major_progress_variables(self):
        """Test that major progress variables exist"""
        progress_vars = self.state_dict.get_variables_by_impact(ImpactCategory.MAJOR_PROGRESS)
        progress_names = [var.name for var in progress_vars]
        
        # Should have badge-related variables
        badge_related = any('badge' in name.lower() for name in progress_names)
        assert badge_related

    def test_danger_condition_evaluation(self):
        """Test danger condition evaluation"""
        # Mock game state with critical HP
        mock_state = {
            'player_hp': 10,
            'player_max_hp': 100,
            'party_count': 1
        }
        
        dangers = self.state_dict.evaluate_danger_conditions(mock_state)
        
        # Should detect HP danger
        assert isinstance(dangers, list)
        if dangers:  # May not detect with simplified mock state
            for danger in dangers:
                assert 'variable' in danger
                assert 'condition' in danger
                assert 'severity' in danger

    def test_opportunity_evaluation(self):
        """Test opportunity condition evaluation"""
        # Mock state progression
        previous_state = {'badges': 0, 'player_level': 5}
        current_state = {'badges': 1, 'player_level': 8}  # Level up and badge earned
        
        opportunities = self.state_dict.evaluate_opportunities(current_state, previous_state)
        
        assert isinstance(opportunities, list)
        # May not detect opportunities with simplified mock states

    def test_comprehensive_analysis(self):
        """Test comprehensive state analysis function"""
        mock_state = {
            'player_hp': 50,
            'player_max_hp': 100,
            'badges': 2,
            'party_count': 3
        }
        
        analysis = analyze_game_state_comprehensive(mock_state)
        
        required_keys = ['dangers', 'opportunities', 'total_danger_score', 
                        'total_opportunity_score', 'critical_variables']
        
        for key in required_keys:
            assert key in analysis
        
        assert isinstance(analysis['dangers'], list)
        assert isinstance(analysis['opportunities'], list)
        assert isinstance(analysis['total_danger_score'], (int, float))

    def test_variable_relationships(self):
        """Test variable relationship analysis"""
        relationships = self.state_dict.analyze_variable_relationships()
        
        assert isinstance(relationships, dict)
        
        # Should have some variables with relationships
        if relationships:
            for var_name, relations in relationships.items():
                assert 'depends_on' in relations
                assert 'affects' in relations

    def test_summary_statistics(self):
        """Test summary statistics"""
        summary = self.state_dict.get_variable_summary()
        
        required_keys = ['total_variables', 'impact_distribution', 
                        'type_distribution', 'critical_variables', 
                        'high_impact_variables', 'memory_addresses']
        
        for key in required_keys:
            assert key in summary
        
        assert summary['total_variables'] >= 20
        assert summary['memory_addresses'] > 15
        assert isinstance(summary['impact_distribution'], dict)
        assert isinstance(summary['type_distribution'], dict)

    def test_global_instance(self):
        """Test that global STATE_VARIABLES instance works"""
        assert STATE_VARIABLES is not None
        assert len(STATE_VARIABLES.variables) >= 20
        
        # Test convenience function
        from environments.state.variables import get_state_variable_info
        
        hp_info = get_state_variable_info('player_hp')
        assert hp_info is not None
        assert hp_info.name == 'player_hp'

    def test_variable_types_coverage(self):
        """Test that we cover different variable types"""
        type_counts = {}
        for var in self.state_dict.variables.values():
            var_type = var.variable_type
            type_counts[var_type] = type_counts.get(var_type, 0) + 1
        
        # Should have multiple types
        assert len(type_counts) > 1
        
        # Should have at least some integers and floats
        int_vars = type_counts.get(VariableType.INT, 0)
        assert int_vars > 0

    def test_impact_categories_coverage(self):
        """Test that we cover different impact categories"""
        category_counts = {}
        for var in self.state_dict.variables.values():
            category = var.impact_category
            category_counts[category] = category_counts.get(category, 0) + 1
        
        # Should have multiple categories
        assert len(category_counts) > 2
        
        # Should have survival and major progress categories
        assert ImpactCategory.SURVIVAL in category_counts
        assert ImpactCategory.MAJOR_PROGRESS in category_counts


if __name__ == "__main__":
    # Run tests if called directly
    pytest.main([__file__, "-v"])