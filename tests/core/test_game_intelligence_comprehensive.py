#!/usr/bin/env python3
"""
Comprehensive Test Suite for Game Intelligence Module

This module provides comprehensive testing for the game intelligence system,
including location analysis, battle strategy, inventory management, and
multi-step action planning.

Test Coverage Areas:
- LocationAnalyzer: Location context analysis and strategy recommendations
- ProgressTracker: Game progression phase detection and goal setting
- BattleStrategy: Type effectiveness, battle analysis, and move recommendations
- InventoryManager: Item analysis, usage recommendations, and strategy
- GameIntelligence: Orchestrated intelligence with contextual advice
- Edge cases and integration scenarios
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from dataclasses import asdict
import tempfile
import os
from typing import Dict, List, Any

from core.game_intelligence import (
    LocationType,
    IntelligenceGameContext,
    GameContext,  # Backward compatibility alias
    ActionPlan,
    LocationAnalyzer,
    ProgressTracker,
    BattleStrategy,
    InventoryManager,
    GameIntelligence
)
from environments.state.analyzer import GamePhase, SituationCriticality


class TestLocationAnalyzer:
    """Test location analysis and strategic recommendations"""

    @pytest.fixture
    def location_analyzer(self):
        """Create LocationAnalyzer instance"""
        return LocationAnalyzer()

    def test_location_analyzer_initialization(self, location_analyzer):
        """Test LocationAnalyzer initialization"""
        assert location_analyzer.logger is not None
        assert isinstance(location_analyzer.location_types, dict)
        assert isinstance(location_analyzer.pokemon_center_maps, set)
        assert isinstance(location_analyzer.gym_locations, dict)

        # Verify some key location mappings
        assert location_analyzer.location_types[1] == (LocationType.TOWN, "New Bark Town")
        assert location_analyzer.location_types[6] == (LocationType.TOWN, "Violet City")
        assert 6 in location_analyzer.gym_locations

    def test_analyze_location_known_locations(self, location_analyzer):
        """Test location analysis for known locations"""
        # Test town location
        game_state = {'player_map': 1}
        location_type, name = location_analyzer.analyze_location(game_state)
        assert location_type == LocationType.TOWN
        assert name == "New Bark Town"

        # Test route location
        game_state = {'player_map': 2}
        location_type, name = location_analyzer.analyze_location(game_state)
        assert location_type == LocationType.ROUTE
        assert name == "Route 29"

        # Test gym location
        game_state = {'player_map': 7}
        location_type, name = location_analyzer.analyze_location(game_state)
        assert location_type == LocationType.GYM
        assert name == "Sprout Tower"

    def test_analyze_location_unknown_location(self, location_analyzer):
        """Test location analysis for unknown locations"""
        game_state = {'player_map': 999}
        location_type, name = location_analyzer.analyze_location(game_state)
        assert location_type == LocationType.UNKNOWN
        assert name == "Unknown Location 999"

    def test_analyze_location_missing_map_data(self, location_analyzer):
        """Test location analysis with missing map data"""
        game_state = {}
        location_type, name = location_analyzer.analyze_location(game_state)
        assert location_type == LocationType.UNKNOWN
        assert name == "Unknown Location 0"

    def test_get_location_strategy_pokemon_center(self, location_analyzer):
        """Test strategy recommendations for Pokemon Center"""
        game_state = {
            'player_hp': 50,
            'player_max_hp': 100
        }
        strategies = location_analyzer.get_location_strategy(LocationType.POKEMON_CENTER, game_state)
        assert "Heal Pokemon at counter" in strategies
        assert "Check PC for stored Pokemon" in strategies

    def test_get_location_strategy_town(self, location_analyzer):
        """Test strategy recommendations for towns"""
        game_state = {
            'player_hp': 25,
            'player_max_hp': 100
        }
        strategies = location_analyzer.get_location_strategy(LocationType.TOWN, game_state)
        assert any("Pokemon Center" in strategy for strategy in strategies)
        assert any("gym" in strategy.lower() for strategy in strategies)

    def test_get_location_strategy_route(self, location_analyzer):
        """Test strategy recommendations for routes"""
        game_state = {'player_hp': 100, 'player_max_hp': 100}
        strategies = location_analyzer.get_location_strategy(LocationType.ROUTE, game_state)
        assert any("wild Pokemon" in strategy for strategy in strategies)
        assert any("trainers" in strategy for strategy in strategies)
        assert any("grass" in strategy for strategy in strategies)

    def test_get_location_strategy_gym_healthy(self, location_analyzer):
        """Test gym strategy when Pokemon is healthy"""
        game_state = {
            'player_hp': 85,
            'player_max_hp': 100
        }
        strategies = location_analyzer.get_location_strategy(LocationType.GYM, game_state)
        assert any("Challenge gym leader" in strategy for strategy in strategies)

    def test_get_location_strategy_gym_low_health(self, location_analyzer):
        """Test gym strategy when Pokemon has low health"""
        game_state = {
            'player_hp': 40,
            'player_max_hp': 100
        }
        strategies = location_analyzer.get_location_strategy(LocationType.GYM, game_state)
        assert any("Heal before" in strategy for strategy in strategies)

    def test_get_location_strategy_cave(self, location_analyzer):
        """Test strategy recommendations for caves"""
        game_state = {'player_hp': 100, 'player_max_hp': 100}
        strategies = location_analyzer.get_location_strategy(LocationType.CAVE, game_state)
        assert any("rare Pokemon" in strategy for strategy in strategies)
        assert any("items" in strategy for strategy in strategies)


class TestProgressTracker:
    """Test game progression tracking and goal setting"""

    @pytest.fixture
    def progress_tracker(self):
        """Create ProgressTracker instance"""
        return ProgressTracker()

    def test_progress_tracker_initialization(self, progress_tracker):
        """Test ProgressTracker initialization"""
        assert progress_tracker.logger is not None

    def test_get_game_phase_tutorial(self, progress_tracker):
        """Test tutorial phase detection"""
        game_state = {'party_count': 0, 'badges_total': 0}
        phase = progress_tracker.get_game_phase(game_state)
        assert phase == GamePhase.TUTORIAL

    def test_get_game_phase_early_game(self, progress_tracker):
        """Test early game phase detection"""
        game_state = {'party_count': 1, 'badges_total': 0}
        phase = progress_tracker.get_game_phase(game_state)
        assert phase == GamePhase.EARLY_GAME

    def test_get_game_phase_gym_battles(self, progress_tracker):
        """Test gym battles phase detection"""
        game_state = {'party_count': 2, 'badges_total': 2}
        phase = progress_tracker.get_game_phase(game_state)
        assert phase == GamePhase.GYM_BATTLES

    def test_get_game_phase_late_game_mid_badges(self, progress_tracker):
        """Test late game phase detection with mid-range badges"""
        game_state = {'party_count': 4, 'badges_total': 5}
        phase = progress_tracker.get_game_phase(game_state)
        assert phase == GamePhase.LATE_GAME

    def test_get_game_phase_late_game(self, progress_tracker):
        """Test late game phase detection"""
        game_state = {'party_count': 6, 'badges_total': 10}
        phase = progress_tracker.get_game_phase(game_state)
        assert phase == GamePhase.POST_GAME

    def test_get_game_phase_post_game(self, progress_tracker):
        """Test post game phase detection"""
        game_state = {'party_count': 6, 'badges_total': 16}
        phase = progress_tracker.get_game_phase(game_state)
        assert phase == GamePhase.POST_GAME

    def test_get_immediate_goals_tutorial(self, progress_tracker):
        """Test immediate goals during tutorial phase"""
        game_state = {'party_count': 0, 'badges_total': 0}
        goals = progress_tracker.get_immediate_goals(game_state, LocationType.TOWN)
        assert any("first Pokemon" in goal for goal in goals)

    def test_get_immediate_goals_with_urgent_healing(self, progress_tracker):
        """Test immediate goals when healing is urgent"""
        game_state = {
            'party_count': 1,
            'player_hp': 15,
            'player_max_hp': 100,
            'badges_total': 1
        }
        goals = progress_tracker.get_immediate_goals(game_state, LocationType.ROUTE)
        assert any("URGENT" in goal for goal in goals)
        assert any("Heal" in goal for goal in goals)

    def test_get_immediate_goals_early_game(self, progress_tracker):
        """Test immediate goals during early game"""
        game_state = {
            'party_count': 1,
            'player_hp': 80,
            'player_max_hp': 100,
            'badges_total': 0
        }
        goals = progress_tracker.get_immediate_goals(game_state, LocationType.ROUTE)
        assert any("Level up" in goal for goal in goals)
        assert any("Violet City" in goal for goal in goals)

    def test_get_immediate_goals_gym_battles(self, progress_tracker):
        """Test immediate goals during gym battles phase"""
        game_state = {
            'party_count': 2,
            'player_hp': 100,
            'player_max_hp': 100,
            'badges_total': 1
        }
        goals = progress_tracker.get_immediate_goals(game_state, LocationType.TOWN)
        assert any("gym #2" in goal for goal in goals)

    def test_get_immediate_goals_no_pokemon(self, progress_tracker):
        """Test immediate goals when player has no Pokemon"""
        game_state = {'party_count': 0, 'badges_total': 0}
        goals = progress_tracker.get_immediate_goals(game_state, LocationType.TOWN)
        # Should not include health-related goals
        assert not any("Heal" in goal for goal in goals)
        assert any("Pokemon" in goal for goal in goals)

    def test_get_strategic_goals_tutorial(self, progress_tracker):
        """Test strategic goals during tutorial"""
        game_state = {'party_count': 0, 'badges_total': 0}
        goals = progress_tracker.get_strategic_goals(game_state)
        assert any("Professor Elm" in goal for goal in goals)
        assert any("mechanics" in goal for goal in goals)

    def test_get_strategic_goals_early_game(self, progress_tracker):
        """Test strategic goals during early game"""
        game_state = {'party_count': 1, 'badges_total': 0}
        goals = progress_tracker.get_strategic_goals(game_state)
        assert any("balanced party" in goal for goal in goals)
        assert any("first gym badge" in goal for goal in goals)

    def test_get_strategic_goals_gym_battles(self, progress_tracker):
        """Test strategic goals during gym battles phase"""
        game_state = {'party_count': 3, 'badges_total': 2}
        goals = progress_tracker.get_strategic_goals(game_state)
        assert any("8 Johto badges" in goal for goal in goals)
        assert any("Elite Four" in goal for goal in goals)


class TestBattleStrategy:
    """Test battle strategy and type effectiveness system"""

    @pytest.fixture
    def battle_strategy(self):
        """Create BattleStrategy instance"""
        return BattleStrategy()

    def test_battle_strategy_initialization(self, battle_strategy):
        """Test BattleStrategy initialization"""
        assert battle_strategy.logger is not None
        assert isinstance(battle_strategy.type_effectiveness, dict)
        assert isinstance(battle_strategy.status_conditions, dict)
        assert isinstance(battle_strategy.move_categories, dict)

    def test_get_type_effectiveness_super_effective(self, battle_strategy):
        """Test super effective type matchups"""
        # Water beats Fire
        effectiveness = battle_strategy.get_type_effectiveness("WATER", "FIRE")
        assert effectiveness == 2.0

        # Electric beats Water
        effectiveness = battle_strategy.get_type_effectiveness("ELECTRIC", "WATER")
        assert effectiveness == 2.0

        # Fighting beats Normal
        effectiveness = battle_strategy.get_type_effectiveness("FIGHTING", "NORMAL")
        assert effectiveness == 2.0

    def test_get_type_effectiveness_not_very_effective(self, battle_strategy):
        """Test not very effective type matchups"""
        # Water resists Water
        effectiveness = battle_strategy.get_type_effectiveness("WATER", "WATER")
        assert effectiveness == 0.5

        # Fire resists Fire
        effectiveness = battle_strategy.get_type_effectiveness("FIRE", "FIRE")
        assert effectiveness == 0.5

    def test_get_type_effectiveness_no_effect(self, battle_strategy):
        """Test no effect type matchups"""
        # Electric has no effect on Ground
        effectiveness = battle_strategy.get_type_effectiveness("ELECTRIC", "GROUND")
        assert effectiveness == 0.0

        # Fighting has no effect on Ghost
        effectiveness = battle_strategy.get_type_effectiveness("FIGHTING", "GHOST")
        assert effectiveness == 0.0

    def test_get_type_effectiveness_neutral(self, battle_strategy):
        """Test neutral type matchups"""
        # Test unlisted combination defaults to 1.0
        effectiveness = battle_strategy.get_type_effectiveness("NORMAL", "NORMAL")
        assert effectiveness == 1.0

    def test_get_type_effectiveness_case_insensitive(self, battle_strategy):
        """Test type effectiveness with different cases"""
        effectiveness1 = battle_strategy.get_type_effectiveness("water", "fire")
        effectiveness2 = battle_strategy.get_type_effectiveness("WATER", "FIRE")
        effectiveness3 = battle_strategy.get_type_effectiveness("Water", "Fire")

        assert effectiveness1 == effectiveness2 == effectiveness3 == 2.0

    def test_analyze_battle_situation_not_in_battle(self, battle_strategy):
        """Test battle analysis when not in battle"""
        game_state = {'in_battle': 0}
        analysis = battle_strategy.analyze_battle_situation(game_state)
        assert analysis['in_battle'] is False

    def test_analyze_battle_situation_aggressive_phase(self, battle_strategy):
        """Test battle analysis in aggressive phase"""
        game_state = {
            'in_battle': 1,
            'player_hp': 85,
            'player_max_hp': 100,
            'enemy_level': 10,
            'player_level': 12,
            'player_species': 1,
            'enemy_species': 2
        }
        analysis = battle_strategy.analyze_battle_situation(game_state)

        assert analysis['in_battle'] is True
        assert analysis['player_hp_ratio'] == 0.85
        assert analysis['level_difference'] == -2
        assert analysis['level_advantage'] == "even"
        assert analysis['battle_phase'] == "aggressive"

    def test_analyze_battle_situation_cautious_phase(self, battle_strategy):
        """Test battle analysis in cautious phase"""
        game_state = {
            'in_battle': 1,
            'player_hp': 45,
            'player_max_hp': 100,
            'enemy_level': 15,
            'player_level': 13
        }
        analysis = battle_strategy.analyze_battle_situation(game_state)

        assert analysis['battle_phase'] == "cautious"
        assert analysis['level_difference'] == 2
        assert analysis['level_advantage'] == "even"

    def test_analyze_battle_situation_defensive_phase(self, battle_strategy):
        """Test battle analysis in defensive phase"""
        game_state = {
            'in_battle': 1,
            'player_hp': 20,
            'player_max_hp': 100,
            'enemy_level': 20,
            'player_level': 10
        }
        analysis = battle_strategy.analyze_battle_situation(game_state)

        assert analysis['battle_phase'] == "defensive"
        assert analysis['level_difference'] == 10
        assert analysis['level_advantage'] == "enemy"

    def test_get_action_priority_emergency_heal(self, battle_strategy):
        """Test emergency heal action priority"""
        priority = battle_strategy._get_action_priority(0.1, 0)
        assert priority == "emergency_heal"

    def test_get_action_priority_switch_or_heal(self, battle_strategy):
        """Test switch or heal action priority"""
        priority = battle_strategy._get_action_priority(0.25, 6)
        assert priority == "switch_or_heal"

    def test_get_action_priority_consider_flee(self, battle_strategy):
        """Test consider flee action priority"""
        priority = battle_strategy._get_action_priority(0.5, 10)
        assert priority == "consider_flee"

    def test_get_action_priority_aggressive_attack(self, battle_strategy):
        """Test aggressive attack action priority"""
        priority = battle_strategy._get_action_priority(0.9, -3)
        assert priority == "aggressive_attack"

    def test_get_action_priority_standard_attack(self, battle_strategy):
        """Test standard attack action priority"""
        priority = battle_strategy._get_action_priority(0.6, 0)
        assert priority == "standard_attack"

    def test_get_battle_strategy_not_in_battle(self, battle_strategy):
        """Test battle strategy when not in battle"""
        game_state = {'in_battle': 0}
        strategy = battle_strategy.get_battle_strategy(game_state)
        assert strategy == "Not in battle"

    def test_get_battle_strategy_emergency(self, battle_strategy):
        """Test battle strategy for emergency situations"""
        game_state = {
            'in_battle': 1,
            'player_hp': 5,
            'player_max_hp': 100,
            'enemy_level': 10,
            'player_level': 10
        }
        strategy = battle_strategy.get_battle_strategy(game_state)
        assert "EMERGENCY" in strategy
        assert "heal" in strategy.lower()

    def test_get_battle_strategy_retreat(self, battle_strategy):
        """Test battle strategy for retreat situations"""
        game_state = {
            'in_battle': 1,
            'player_hp': 60,
            'player_max_hp': 100,
            'enemy_level': 25,
            'player_level': 8
        }
        strategy = battle_strategy.get_battle_strategy(game_state)
        assert "RETREAT" in strategy

    def test_get_battle_strategy_aggressive(self, battle_strategy):
        """Test aggressive battle strategy"""
        game_state = {
            'in_battle': 1,
            'player_hp': 90,
            'player_max_hp': 100,
            'enemy_level': 8,
            'player_level': 15,
            'party_count': 2
        }
        strategy = battle_strategy.get_battle_strategy(game_state)
        assert "aggressive" in strategy.lower()

    def test_recommend_move_selection_not_in_battle(self, battle_strategy):
        """Test move selection when not in battle"""
        game_state = {'in_battle': 0}
        recommendations = battle_strategy.recommend_move_selection(game_state)
        assert recommendations['recommendation'] == 'Not in battle'

    def test_recommend_move_selection_aggressive(self, battle_strategy):
        """Test move selection for aggressive battle phase"""
        game_state = {
            'in_battle': 1,
            'player_hp': 90,
            'player_max_hp': 100,
            'enemy_level': 10,
            'player_level': 15
        }
        recommendations = battle_strategy.recommend_move_selection(game_state)

        assert recommendations['battle_phase'] == 'aggressive'
        assert 'attack' in recommendations['suggested_move_types']

    def test_recommend_move_selection_defensive(self, battle_strategy):
        """Test move selection for defensive battle phase"""
        game_state = {
            'in_battle': 1,
            'player_hp': 15,
            'player_max_hp': 100,
            'enemy_level': 15,
            'player_level': 10
        }
        recommendations = battle_strategy.recommend_move_selection(game_state)

        assert recommendations['battle_phase'] == 'defensive'
        assert 'healing' in recommendations['suggested_move_types']
        assert 'risky_attack' in recommendations['avoid_moves']


class TestInventoryManager:
    """Test inventory and item management system"""

    @pytest.fixture
    def inventory_manager(self):
        """Create InventoryManager instance"""
        return InventoryManager()

    def test_inventory_manager_initialization(self, inventory_manager):
        """Test InventoryManager initialization"""
        assert inventory_manager.logger is not None
        assert isinstance(inventory_manager.item_categories, dict)
        assert isinstance(inventory_manager.usage_strategies, dict)

        # Verify item categories structure
        assert 'healing' in inventory_manager.item_categories
        assert 'pokeballs' in inventory_manager.item_categories
        assert 'battle_items' in inventory_manager.item_categories
        assert 'key_items' in inventory_manager.item_categories

    def test_analyze_inventory_needs_emergency_healing(self, inventory_manager):
        """Test inventory analysis when emergency healing is needed"""
        game_state = {
            'in_battle': False,
            'party_count': 1,
            'player_hp': 10,
            'player_max_hp': 100
        }
        context = {'detected_state': 'overworld', 'badges_total': 1}

        analysis = inventory_manager.analyze_inventory_needs(game_state, context)

        assert 'emergency_healing' in analysis['immediate_needs']
        assert any('strongest healing item' in advice for advice in analysis['item_usage_advice'])

    def test_analyze_inventory_needs_low_health(self, inventory_manager):
        """Test inventory analysis for low health situations"""
        game_state = {
            'in_battle': False,
            'party_count': 1,
            'player_hp': 40,
            'player_max_hp': 100
        }
        context = {'detected_state': 'overworld'}

        analysis = inventory_manager.analyze_inventory_needs(game_state, context)

        assert 'healing_item' in analysis['recommended_items']

    def test_analyze_inventory_needs_tough_battle(self, inventory_manager):
        """Test inventory analysis during tough battles"""
        game_state = {
            'in_battle': True,
            'party_count': 1,
            'player_hp': 80,
            'player_max_hp': 100,
            'enemy_level': 20,
            'player_level': 12
        }
        context = {'detected_state': 'battle'}

        analysis = inventory_manager.analyze_inventory_needs(game_state, context)

        assert 'battle_enhancement' in analysis['recommended_items']

    def test_analyze_inventory_needs_early_game(self, inventory_manager):
        """Test inventory priorities for early game"""
        game_state = {
            'in_battle': False,
            'party_count': 1,
            'player_hp': 80,
            'player_max_hp': 100,
            'badges_total': 1
        }
        context = {'detected_state': 'overworld', 'badges_total': 1}

        analysis = inventory_manager.analyze_inventory_needs(game_state, context)

        priorities = analysis['inventory_priorities']
        assert any('pokeballs' in priority.lower() for priority in priorities)
        assert any('healing' in priority.lower() for priority in priorities)

    def test_analyze_inventory_needs_no_pokemon(self, inventory_manager):
        """Test inventory analysis when player has no Pokemon"""
        game_state = {
            'in_battle': False,
            'party_count': 0
        }
        context = {'detected_state': 'overworld'}

        analysis = inventory_manager.analyze_inventory_needs(game_state, context)

        # Should not recommend healing items when no Pokemon
        assert 'emergency_healing' not in analysis['immediate_needs']

    def test_recommend_item_usage_no_pokemon(self, inventory_manager):
        """Test item usage recommendation when no Pokemon"""
        game_state = {'party_count': 0}

        recommendations = inventory_manager.recommend_item_usage(game_state)

        assert recommendations['should_use_item'] is False

    def test_recommend_item_usage_critical_health(self, inventory_manager):
        """Test item usage for critical health"""
        game_state = {
            'party_count': 1,
            'player_hp': 10,
            'player_max_hp': 100,
            'in_battle': False
        }

        recommendations = inventory_manager.recommend_item_usage(game_state)

        assert recommendations['should_use_item'] is True
        assert recommendations['item_type'] == 'healing'
        assert recommendations['urgency'] == 'critical'

    def test_recommend_item_usage_battle_low_health(self, inventory_manager):
        """Test item usage for low health in battle"""
        game_state = {
            'party_count': 1,
            'player_hp': 25,
            'player_max_hp': 100,
            'in_battle': True
        }

        recommendations = inventory_manager.recommend_item_usage(game_state)

        assert recommendations['should_use_item'] is True
        assert recommendations['item_type'] == 'healing'

    def test_recommend_item_usage_healthy(self, inventory_manager):
        """Test item usage when Pokemon is healthy"""
        game_state = {
            'party_count': 1,
            'player_hp': 95,
            'player_max_hp': 100,
            'in_battle': False
        }

        recommendations = inventory_manager.recommend_item_usage(game_state)

        assert recommendations['should_use_item'] is False

    def test_get_optimal_pokeball(self, inventory_manager):
        """Test optimal pokeball recommendation"""
        game_state = {'party_count': 2}
        wild_pokemon_info = {'level': 15, 'hp_ratio': 0.3}

        # This method isn't fully implemented in the source,
        # but we test it exists and returns expected format
        result = inventory_manager.get_optimal_pokeball(game_state, wild_pokemon_info)
        assert isinstance(result, str)

    def test_evaluate_held_item_strategy(self, inventory_manager):
        """Test held item strategy evaluation"""
        game_state = {'party_count': 1, 'player_level': 15}

        # This method isn't fully implemented in the source,
        # but we test it exists and returns expected format
        result = inventory_manager.evaluate_held_item_strategy(game_state)
        assert isinstance(result, dict)


class TestGameIntelligence:
    """Test main GameIntelligence orchestrator class"""

    @pytest.fixture
    def game_intelligence(self):
        """Create GameIntelligence instance"""
        return GameIntelligence()

    def test_game_intelligence_initialization(self, game_intelligence):
        """Test GameIntelligence initialization"""
        assert game_intelligence.logger is not None
        assert isinstance(game_intelligence.location_analyzer, LocationAnalyzer)
        assert isinstance(game_intelligence.progress_tracker, ProgressTracker)
        assert isinstance(game_intelligence.battle_strategy, BattleStrategy)
        assert isinstance(game_intelligence.inventory_manager, InventoryManager)

    def test_analyze_game_context_tutorial_phase(self, game_intelligence):
        """Test game context analysis for tutorial phase"""
        game_state = {
            'party_count': 0,
            'badges_total': 0,
            'player_map': 1,
            'player_hp': 0,
            'player_max_hp': 0
        }
        screen_analysis = {'detected_state': 'overworld'}

        context = game_intelligence.analyze_game_context(game_state, screen_analysis)

        assert context.phase == GamePhase.TUTORIAL
        assert context.location_type == LocationType.TOWN
        assert context.urgency_level == 4  # High priority to get first Pokemon
        assert any("first Pokemon" in goal for goal in context.immediate_goals)

    def test_analyze_game_context_early_game(self, game_intelligence):
        """Test game context analysis for early game phase"""
        game_state = {
            'party_count': 1,
            'badges_total': 0,
            'player_map': 2,
            'player_hp': 80,
            'player_max_hp': 100
        }
        screen_analysis = {'detected_state': 'overworld'}

        context = game_intelligence.analyze_game_context(game_state, screen_analysis)

        assert context.phase == GamePhase.EARLY_GAME
        assert context.location_type == LocationType.ROUTE
        assert context.urgency_level == 1  # Normal priority
        assert "Good" in context.health_status

    def test_analyze_game_context_critical_health(self, game_intelligence):
        """Test game context analysis with critical health"""
        game_state = {
            'party_count': 1,
            'badges_total': 1,
            'player_map': 6,
            'player_hp': 15,
            'player_max_hp': 100
        }
        screen_analysis = {'detected_state': 'overworld'}

        context = game_intelligence.analyze_game_context(game_state, screen_analysis)

        assert context.urgency_level == 5  # Critical
        assert "Critical" in context.health_status
        assert any("Use" in action or "item" in action.lower() for action in context.recommended_actions)

    def test_analyze_game_context_in_battle(self, game_intelligence):
        """Test game context analysis during battle"""
        game_state = {
            'party_count': 1,
            'badges_total': 1,
            'player_map': 2,
            'player_hp': 70,
            'player_max_hp': 100,
            'in_battle': 1,
            'enemy_level': 10,
            'player_level': 12
        }
        screen_analysis = {'detected_state': 'battle'}

        context = game_intelligence.analyze_game_context(game_state, screen_analysis)

        assert any("Battle:" in action for action in context.recommended_actions)

    def test_get_action_plan_emergency_healing(self, game_intelligence):
        """Test action plan generation for emergency healing"""
        game_state = {
            'party_count': 1,
            'player_hp': 10,
            'player_max_hp': 100
        }
        screen_analysis = {'detected_state': 'overworld'}
        game_context = game_intelligence.analyze_game_context(game_state, screen_analysis)

        plans = game_intelligence.get_action_plan(game_context, game_state)

        assert len(plans) > 0
        emergency_plan = plans[0]  # Should be highest priority
        assert emergency_plan.priority >= 9
        assert "heal" in emergency_plan.goal.lower()

    def test_get_action_plan_tutorial(self, game_intelligence):
        """Test action plan generation for tutorial phase"""
        game_state = {
            'party_count': 0,
            'badges_total': 0,
            'player_map': 1
        }
        screen_analysis = {'detected_state': 'overworld'}
        game_context = game_intelligence.analyze_game_context(game_state, screen_analysis)

        plans = game_intelligence.get_action_plan(game_context, game_state)

        tutorial_plans = [p for p in plans if "starter" in p.goal.lower() or "elm" in p.goal.lower()]
        assert len(tutorial_plans) > 0

    def test_get_action_plan_gym_challenge(self, game_intelligence):
        """Test action plan generation for gym challenges"""
        game_state = {
            'party_count': 2,
            'badges_total': 1,
            'player_map': 6,  # Violet City
            'player_hp': 90,
            'player_max_hp': 100
        }
        screen_analysis = {'detected_state': 'overworld'}

        # Mock the location as gym for this test
        with patch.object(game_intelligence.location_analyzer, 'analyze_location') as mock_analyze:
            mock_analyze.return_value = (LocationType.GYM, "Test Gym")
            game_context = game_intelligence.analyze_game_context(game_state, screen_analysis)
            plans = game_intelligence.get_action_plan(game_context, game_state)

        gym_plans = [p for p in plans if "gym" in p.goal.lower()]
        assert len(gym_plans) > 0

    def test_get_contextual_advice_urgent_situation(self, game_intelligence):
        """Test contextual advice for urgent situations"""
        game_context = IntelligenceGameContext(
            phase=GamePhase.EARLY_GAME,
            location_type=LocationType.ROUTE,
            location_name="Route 29",
            immediate_goals=["URGENT: Heal Pokemon immediately"],
            strategic_goals=["Build balanced party"],
            health_status="Critical",
            party_status="1 Pokemon",
            recommended_actions=["Use healing item"],
            urgency_level=5
        )
        recent_actions = ["up", "up", "a"]

        advice = game_intelligence.get_contextual_advice(game_context, recent_actions)

        assert "⚠️ URGENT" in advice
        assert "(5/5)" in advice

    def test_get_contextual_advice_stuck_detection(self, game_intelligence):
        """Test contextual advice with stuck action detection"""
        game_context = IntelligenceGameContext(
            phase=GamePhase.EARLY_GAME,
            location_type=LocationType.ROUTE,
            location_name="Route 29",
            immediate_goals=["Explore area"],
            strategic_goals=["Build party"],
            health_status="Healthy",
            party_status="1 Pokemon",
            recommended_actions=["Battle wild Pokemon"],
            urgency_level=1
        )
        recent_actions = ["up", "up", "up"]

        advice = game_intelligence.get_contextual_advice(game_context, recent_actions)

        assert "⚠️ Repeated action detected" in advice

    def test_get_contextual_advice_menu_detection(self, game_intelligence):
        """Test contextual advice with menu state detection"""
        game_context = IntelligenceGameContext(
            phase=GamePhase.EARLY_GAME,
            location_type=LocationType.ROUTE,
            location_name="Route 29",
            immediate_goals=["Explore area"],
            strategic_goals=["Build party"],
            health_status="Healthy",
            party_status="1 Pokemon",
            recommended_actions=["Battle wild Pokemon"],
            urgency_level=1
        )
        recent_actions = ["a", "START", "up"]

        advice = game_intelligence.get_contextual_advice(game_context, recent_actions)

        assert "⚠️ Recently opened menu" in advice


class TestGameIntelligenceDataClasses:
    """Test dataclasses and enums in game intelligence module"""

    def test_location_type_enum(self):
        """Test LocationType enum values"""
        assert LocationType.TOWN is not None
        assert LocationType.ROUTE is not None
        assert LocationType.GYM is not None
        assert LocationType.POKEMON_CENTER is not None
        assert LocationType.POKEMON_LAB is not None
        assert LocationType.CAVE is not None
        assert LocationType.FOREST is not None
        assert LocationType.UNKNOWN is not None

    def test_intelligence_game_context_creation(self):
        """Test IntelligenceGameContext dataclass creation"""
        context = IntelligenceGameContext(
            phase=GamePhase.EARLY_GAME,
            location_type=LocationType.ROUTE,
            location_name="Route 29",
            immediate_goals=["Level up Pokemon"],
            strategic_goals=["Earn badges"],
            health_status="Healthy",
            party_status="1 Pokemon",
            recommended_actions=["Battle wild Pokemon"],
            urgency_level=2
        )

        assert context.phase == GamePhase.EARLY_GAME
        assert context.location_type == LocationType.ROUTE
        assert context.location_name == "Route 29"
        assert context.urgency_level == 2

    def test_game_context_alias(self):
        """Test GameContext backward compatibility alias"""
        # GameContext should be the same as IntelligenceGameContext
        assert GameContext == IntelligenceGameContext

    def test_action_plan_creation(self):
        """Test ActionPlan dataclass creation"""
        plan = ActionPlan(
            goal="Challenge gym leader",
            steps=["Navigate through gym", "Battle trainers", "Challenge leader"],
            priority=7,
            estimated_actions=30
        )

        assert plan.goal == "Challenge gym leader"
        assert len(plan.steps) == 3
        assert plan.priority == 7
        assert plan.estimated_actions == 30

    def test_intelligence_game_context_serialization(self):
        """Test IntelligenceGameContext serialization"""
        context = IntelligenceGameContext(
            phase=GamePhase.TUTORIAL,
            location_type=LocationType.TOWN,
            location_name="New Bark Town",
            immediate_goals=["Get first Pokemon"],
            strategic_goals=["Learn game mechanics"],
            health_status="No Pokemon",
            party_status="Empty",
            recommended_actions=["Visit Professor Elm"],
            urgency_level=4
        )

        # Test serialization
        context_dict = asdict(context)
        assert isinstance(context_dict, dict)
        assert context_dict['location_name'] == "New Bark Town"
        assert context_dict['urgency_level'] == 4

    def test_action_plan_serialization(self):
        """Test ActionPlan serialization"""
        plan = ActionPlan(
            goal="Heal Pokemon",
            steps=["Go to Pokemon Center", "Talk to nurse"],
            priority=8,
            estimated_actions=5
        )

        # Test serialization
        plan_dict = asdict(plan)
        assert isinstance(plan_dict, dict)
        assert plan_dict['goal'] == "Heal Pokemon"
        assert len(plan_dict['steps']) == 2


class TestGameIntelligenceEdgeCases:
    """Test edge cases and error handling"""

    @pytest.fixture
    def game_intelligence(self):
        """Create GameIntelligence instance"""
        return GameIntelligence()

    def test_analyze_game_context_empty_state(self, game_intelligence):
        """Test game context analysis with empty game state"""
        game_state = {}
        screen_analysis = {}

        context = game_intelligence.analyze_game_context(game_state, screen_analysis)

        # Should handle missing values gracefully
        assert context.phase is not None
        assert context.location_type is not None
        assert isinstance(context.immediate_goals, list)

    def test_analyze_game_context_invalid_values(self, game_intelligence):
        """Test game context analysis with invalid values"""
        game_state = {
            'party_count': -1,  # Invalid
            'badges_total': 999,  # Extreme
            'player_map': -5,  # Invalid
            'player_hp': -10,  # Invalid
            'player_max_hp': 0  # Edge case
        }
        screen_analysis = {'detected_state': 'unknown'}

        context = game_intelligence.analyze_game_context(game_state, screen_analysis)

        # Should handle invalid values without crashing
        assert context is not None
        assert isinstance(context.recommended_actions, list)

    def test_battle_strategy_missing_battle_data(self):
        """Test battle strategy with missing battle data"""
        battle_strategy = BattleStrategy()
        game_state = {
            'in_battle': 1,
            # Missing hp, level, and species data
        }

        analysis = battle_strategy.analyze_battle_situation(game_state)

        # Should handle missing data gracefully
        assert analysis['in_battle'] is True
        assert 'battle_phase' in analysis

    def test_inventory_manager_extreme_health_values(self):
        """Test inventory manager with extreme health values"""
        inventory_manager = InventoryManager()
        game_state = {
            'party_count': 1,
            'player_hp': 1000,  # Extreme value
            'player_max_hp': 1,  # Invalid ratio
            'in_battle': False
        }

        recommendations = inventory_manager.recommend_item_usage(game_state)

        # Should handle extreme values without crashing
        assert isinstance(recommendations, dict)
        assert 'should_use_item' in recommendations

    def test_location_analyzer_boundary_map_ids(self):
        """Test location analyzer with boundary map ID values"""
        location_analyzer = LocationAnalyzer()

        # Test with zero
        location_type, name = location_analyzer.analyze_location({'player_map': 0})
        assert location_type == LocationType.UNKNOWN

        # Test with negative value
        location_type, name = location_analyzer.analyze_location({'player_map': -1})
        assert location_type == LocationType.UNKNOWN

    def test_progress_tracker_edge_case_values(self):
        """Test progress tracker with edge case values"""
        progress_tracker = ProgressTracker()

        # Test with extreme badge count
        game_state = {'party_count': 1, 'badges_total': 100}
        phase = progress_tracker.get_game_phase(game_state)
        assert phase == GamePhase.POST_GAME

        # Test with missing data
        game_state = {}
        phase = progress_tracker.get_game_phase(game_state)
        assert phase == GamePhase.TUTORIAL  # Default fallback


if __name__ == "__main__":
    pytest.main([__file__])