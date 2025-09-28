#!/usr/bin/env python3
"""
Test Suite for Game State Analyzer

Tests the game state analysis system that provides strategic insights
and LLM context for Pokemon Crystal RL decision making.
"""

import pytest
from unittest.mock import Mock, patch

from environments.state.analyzer import (
    GamePhase, SituationCriticality, AnalysisStateVariable,
    GameStateAnalysis, GameStateAnalyzer
)


class TestGamePhase:
    """Test GamePhase enum functionality."""

    def test_game_phase_values(self):
        """Test that all game phase enums have correct values."""
        assert GamePhase.TUTORIAL.value == "tutorial"
        assert GamePhase.INTRO.value == "intro"
        assert GamePhase.EARLY_GAME.value == "early_game"
        assert GamePhase.STARTER_PHASE.value == "starter_phase"
        assert GamePhase.EXPLORATION.value == "exploration"
        assert GamePhase.GYM_BATTLES.value == "gym_battles"
        assert GamePhase.LATE_GAME.value == "late_game"
        assert GamePhase.POST_GAME.value == "post_game"

    def test_game_phase_completeness(self):
        """Test that all expected game phases are defined."""
        expected_phases = [
            "tutorial", "intro", "early_game", "starter_phase",
            "exploration", "gym_battles", "late_game", "post_game"
        ]

        actual_phases = [phase.value for phase in GamePhase]
        assert set(actual_phases) == set(expected_phases)


class TestSituationCriticality:
    """Test SituationCriticality enum functionality."""

    def test_criticality_values(self):
        """Test that all criticality enums have correct values."""
        assert SituationCriticality.EMERGENCY.value == "emergency"
        assert SituationCriticality.URGENT.value == "urgent"
        assert SituationCriticality.MODERATE.value == "moderate"
        assert SituationCriticality.OPTIMAL.value == "optimal"

    def test_criticality_ordering(self):
        """Test that criticality levels are properly ordered."""
        # This could be implemented if there's a need for comparison
        criticalities = [
            SituationCriticality.EMERGENCY,
            SituationCriticality.URGENT,
            SituationCriticality.MODERATE,
            SituationCriticality.OPTIMAL
        ]

        # Test that they're all different
        assert len(set(criticalities)) == 4


class TestAnalysisStateVariable:
    """Test AnalysisStateVariable data structure."""

    def test_state_variable_creation(self):
        """Test creating AnalysisStateVariable."""
        var = AnalysisStateVariable(
            name="test_var",
            type="int",
            current_value=50,
            normal_range=(0, 100),
            critical_thresholds={"low": 20, "high": 80},
            impact_on_rewards=["survival", "progression"],
            impact_on_survival=0.8,
            description="Test variable"
        )

        assert var.name == "test_var"
        assert var.type == "int"
        assert var.current_value == 50
        assert var.normal_range == (0, 100)
        assert var.critical_thresholds == {"low": 20, "high": 80}
        assert var.impact_on_rewards == ["survival", "progression"]
        assert var.impact_on_survival == 0.8
        assert var.description == "Test variable"

    def test_state_variable_types(self):
        """Test different variable types."""
        types_to_test = ['int', 'float', 'bool', 'tuple', 'bitfield']

        for var_type in types_to_test:
            var = AnalysisStateVariable(
                name=f"test_{var_type}",
                type=var_type,
                current_value=None,
                normal_range=(0, 1),
                critical_thresholds={},
                impact_on_rewards=[],
                impact_on_survival=0.0,
                description=f"Test {var_type} variable"
            )
            assert var.type == var_type


class TestGameStateAnalysis:
    """Test GameStateAnalysis data structure."""

    def test_analysis_creation(self):
        """Test creating GameStateAnalysis."""
        analysis = GameStateAnalysis(
            phase=GamePhase.EXPLORATION,
            criticality=SituationCriticality.MODERATE,
            health_percentage=0.8,
            progression_score=45.0,
            exploration_score=60.0,
            immediate_threats=["low_hp"],
            opportunities=["new_area"],
            recommended_priorities=["heal", "explore"],
            situation_summary="Player exploring with good health",
            strategic_context="Safe to continue exploration",
            risk_assessment="Low risk situation",
            state_variables={}
        )

        assert analysis.phase == GamePhase.EXPLORATION
        assert analysis.criticality == SituationCriticality.MODERATE
        assert analysis.health_percentage == 0.8
        assert analysis.progression_score == 45.0
        assert analysis.exploration_score == 60.0
        assert analysis.immediate_threats == ["low_hp"]
        assert analysis.opportunities == ["new_area"]
        assert analysis.recommended_priorities == ["heal", "explore"]
        assert "exploring" in analysis.situation_summary
        assert "exploration" in analysis.strategic_context
        assert "Low risk" in analysis.risk_assessment
        assert analysis.state_variables == {}

    def test_analysis_with_state_variables(self):
        """Test GameStateAnalysis with state variables."""
        state_var = AnalysisStateVariable(
            name="hp", type="int", current_value=50, normal_range=(0, 100),
            critical_thresholds={}, impact_on_rewards=[], impact_on_survival=1.0,
            description="Health points"
        )

        analysis = GameStateAnalysis(
            phase=GamePhase.GYM_BATTLES,
            criticality=SituationCriticality.URGENT,
            health_percentage=0.5,
            progression_score=30.0,
            exploration_score=40.0,
            immediate_threats=["battle"],
            opportunities=["experience"],
            recommended_priorities=["attack"],
            situation_summary="In battle",
            strategic_context="Battle in progress",
            risk_assessment="Medium risk",
            state_variables={"hp": state_var}
        )

        assert "hp" in analysis.state_variables
        assert analysis.state_variables["hp"] == state_var


class TestGameStateAnalyzer:
    """Test GameStateAnalyzer functionality."""

    @pytest.fixture
    def analyzer(self):
        """Create a GameStateAnalyzer instance."""
        return GameStateAnalyzer()

    def test_initialization(self, analyzer):
        """Test analyzer initializes correctly."""
        assert hasattr(analyzer, 'state_variable_definitions')
        assert hasattr(analyzer, 'location_knowledge')
        assert hasattr(analyzer, 'pokemon_knowledge')
        assert isinstance(analyzer.state_variable_definitions, dict)

    def test_state_variable_definitions_structure(self, analyzer):
        """Test state variable definitions are properly structured."""
        definitions = analyzer.state_variable_definitions

        # Check that essential variables are defined
        essential_vars = [
            'player_hp', 'player_max_hp', 'player_level',
            'party_count', 'badges', 'money',
            'player_x', 'player_y', 'player_map'
        ]

        for var in essential_vars:
            assert var in definitions
            assert 'type' in definitions[var]
            assert 'normal_range' in definitions[var]
            assert 'critical_thresholds' in definitions[var]
            assert 'impact_on_rewards' in definitions[var]
            assert 'impact_on_survival' in definitions[var]
            assert 'description' in definitions[var]

    def test_hp_variable_definition(self, analyzer):
        """Test HP variable definition is correct."""
        hp_def = analyzer.state_variable_definitions['player_hp']

        assert hp_def['type'] == 'int'
        assert hp_def['normal_range'] == (1, 999)
        assert 'emergency' in hp_def['critical_thresholds']
        assert 'low' in hp_def['critical_thresholds']
        assert 'survival' in hp_def['impact_on_rewards']
        assert hp_def['impact_on_survival'] == 1.0
        assert 'HP' in hp_def['description']

    def test_badges_variable_definition(self, analyzer):
        """Test badges variable definition is correct."""
        badges_def = analyzer.state_variable_definitions['badges']

        assert badges_def['type'] == 'bitfield'
        assert badges_def['normal_range'] == (0, 16)
        assert 'early' in badges_def['critical_thresholds']
        assert 'mid' in badges_def['critical_thresholds']
        assert 'complete' in badges_def['critical_thresholds']
        assert 'major_progression' in badges_def['impact_on_rewards']

    def test_position_variables_definition(self, analyzer):
        """Test position variables definition."""
        for var in ['player_x', 'player_y']:
            pos_def = analyzer.state_variable_definitions[var]
            assert pos_def['type'] == 'int'
            assert pos_def['normal_range'] == (0, 255)
            assert 'exploration' in pos_def['impact_on_rewards']
            assert pos_def['impact_on_survival'] == 0.0

    @patch.object(GameStateAnalyzer, '_parse_state_variables')
    @patch.object(GameStateAnalyzer, '_determine_game_phase')
    @patch.object(GameStateAnalyzer, '_assess_criticality')
    @patch.object(GameStateAnalyzer, '_calculate_health_percentage')
    @patch.object(GameStateAnalyzer, '_calculate_progression_score')
    @patch.object(GameStateAnalyzer, '_calculate_exploration_score')
    @patch.object(GameStateAnalyzer, '_identify_threats')
    @patch.object(GameStateAnalyzer, '_identify_opportunities')
    @patch.object(GameStateAnalyzer, '_recommend_priorities')
    @patch.object(GameStateAnalyzer, '_generate_situation_summary')
    @patch.object(GameStateAnalyzer, '_generate_strategic_context')
    @patch.object(GameStateAnalyzer, '_generate_risk_assessment')
    def test_analyze_method_integration(
        self, mock_risk, mock_strategic, mock_situation, mock_priorities,
        mock_opportunities, mock_threats, mock_exploration, mock_progression,
        mock_health, mock_criticality, mock_phase, mock_parse, analyzer
    ):
        """Test analyze method calls all sub-methods."""
        # Setup mocks
        mock_parse.return_value = {}
        mock_phase.return_value = GamePhase.EXPLORATION
        mock_criticality.return_value = SituationCriticality.MODERATE
        mock_health.return_value = 0.8
        mock_progression.return_value = 50.0
        mock_exploration.return_value = 60.0
        mock_threats.return_value = []
        mock_opportunities.return_value = ["explore"]
        mock_priorities.return_value = ["move_forward"]
        mock_situation.return_value = "Good situation"
        mock_strategic.return_value = "Continue exploring"
        mock_risk.return_value = "Low risk"

        # Test input
        raw_state = {
            'player_hp': 80,
            'player_max_hp': 100,
            'player_level': 15,
            'badges': 2
        }

        # Call analyze
        result = analyzer.analyze(raw_state)

        # Verify all methods were called
        mock_parse.assert_called_once_with(raw_state)
        mock_phase.assert_called_once()
        mock_criticality.assert_called_once()
        mock_health.assert_called_once()
        mock_progression.assert_called_once()
        mock_exploration.assert_called_once()
        mock_threats.assert_called_once()
        mock_opportunities.assert_called_once()
        mock_priorities.assert_called_once()
        mock_situation.assert_called_once()
        mock_strategic.assert_called_once()
        mock_risk.assert_called_once()

        # Verify result structure
        assert isinstance(result, GameStateAnalysis)
        assert result.phase == GamePhase.EXPLORATION
        assert result.criticality == SituationCriticality.MODERATE
        assert result.health_percentage == 0.8
        assert result.progression_score == 50.0
        assert result.exploration_score == 60.0

    def test_analyze_method_return_type(self, analyzer):
        """Test analyze method returns correct type."""
        # Mock the methods that might not exist yet
        analyzer._parse_state_variables = Mock(return_value={})
        analyzer._determine_game_phase = Mock(return_value=GamePhase.EARLY_GAME)
        analyzer._assess_criticality = Mock(return_value=SituationCriticality.MODERATE)
        analyzer._calculate_health_percentage = Mock(return_value=1.0)
        analyzer._calculate_progression_score = Mock(return_value=0.0)
        analyzer._calculate_exploration_score = Mock(return_value=0.0)
        analyzer._identify_threats = Mock(return_value=[])
        analyzer._identify_opportunities = Mock(return_value=[])
        analyzer._recommend_priorities = Mock(return_value=[])
        analyzer._generate_situation_summary = Mock(return_value="Test summary")
        analyzer._generate_strategic_context = Mock(return_value="Test context")
        analyzer._generate_risk_assessment = Mock(return_value="Test assessment")

        raw_state = {'player_hp': 100}
        result = analyzer.analyze(raw_state)

        assert isinstance(result, GameStateAnalysis)

    def test_variable_types_completeness(self, analyzer):
        """Test that all expected variable types are covered."""
        definitions = analyzer.state_variable_definitions
        types_found = set()

        for var_def in definitions.values():
            types_found.add(var_def['type'])

        expected_types = {'int', 'float', 'bool', 'tuple', 'bitfield'}
        # At least some of these types should be present
        assert len(types_found.intersection(expected_types)) > 0

    def test_impact_categories_completeness(self, analyzer):
        """Test that all impact categories are covered."""
        definitions = analyzer.state_variable_definitions
        impact_categories = set()

        for var_def in definitions.values():
            impact_categories.update(var_def['impact_on_rewards'])

        # Should have various impact categories
        expected_categories = {
            'survival', 'progression', 'exploration', 'battle_performance',
            'major_progression', 'resources', 'movement'
        }

        # At least some categories should be present
        assert len(impact_categories.intersection(expected_categories)) > 0

    def test_survival_impact_ranges(self, analyzer):
        """Test that survival impact values are in valid range."""
        definitions = analyzer.state_variable_definitions

        for var_name, var_def in definitions.items():
            survival_impact = var_def['impact_on_survival']
            assert 0.0 <= survival_impact <= 1.0, f"{var_name} has invalid survival impact: {survival_impact}"

    def test_normal_ranges_validity(self, analyzer):
        """Test that normal ranges are valid."""
        definitions = analyzer.state_variable_definitions

        for var_name, var_def in definitions.items():
            normal_range = var_def['normal_range']
            assert isinstance(normal_range, tuple), f"{var_name} normal_range is not tuple"
            assert len(normal_range) == 2, f"{var_name} normal_range doesn't have 2 elements"

            min_val, max_val = normal_range
            if isinstance(min_val, (int, float)) and isinstance(max_val, (int, float)):
                assert min_val <= max_val, f"{var_name} has invalid range: {normal_range}"


class TestAnalyzerKnowledgeBases:
    """Test analyzer knowledge base initialization."""

    @pytest.fixture
    def analyzer(self):
        """Create a GameStateAnalyzer instance."""
        return GameStateAnalyzer()

    def test_location_knowledge_exists(self, analyzer):
        """Test that location knowledge is initialized."""
        assert hasattr(analyzer, 'location_knowledge')
        # Even if empty, should be a dictionary-like structure
        knowledge = analyzer.location_knowledge
        assert knowledge is not None

    def test_pokemon_knowledge_exists(self, analyzer):
        """Test that pokemon knowledge is initialized."""
        assert hasattr(analyzer, 'pokemon_knowledge')
        # Even if empty, should be a dictionary-like structure
        knowledge = analyzer.pokemon_knowledge
        assert knowledge is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])