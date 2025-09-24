"""
Tests for Battle Strategy Plugins

This module contains comprehensive tests for the official battle strategy plugins
including aggressive, defensive, and balanced strategies.
"""

import pytest
from unittest.mock import Mock, patch
from typing import Dict, Any

from plugins.battle_strategies import (
    AggressiveBattleStrategy, DefensiveBattleStrategy, BalancedBattleStrategy
)
from core.plugin_system import PluginMetadata, PluginType


class TestAggressiveBattleStrategy:
    """Test AggressiveBattleStrategy plugin"""

    def setup_method(self):
        """Setup for each test"""
        self.strategy = AggressiveBattleStrategy()
        self.strategy.initialize()

    def test_metadata(self):
        """Test plugin metadata"""
        metadata = self.strategy.get_metadata()

        assert isinstance(metadata, PluginMetadata)
        assert metadata.name == "aggressive_battle"
        assert metadata.plugin_type == PluginType.BATTLE_STRATEGY
        assert metadata.hot_swappable is True
        assert metadata.priority == 7
        assert "aggressive" in metadata.tags

    def test_initialization(self):
        """Test plugin initialization"""
        config = {'aggression_factor': 0.9}
        strategy = AggressiveBattleStrategy(config)
        result = strategy.initialize()

        assert result is True
        assert strategy.aggression_factor == 0.9
        assert 'super_effective' in strategy.damage_multipliers

    def test_shutdown(self):
        """Test plugin shutdown"""
        result = self.strategy.shutdown()
        assert result is True

    def test_analyze_battle_situation_winning(self):
        """Test battle analysis when winning"""
        game_state = {
            'player_hp': 80,
            'player_max_hp': 100
        }
        battle_context = {
            'enemy_hp': 30,
            'enemy_max_hp': 100
        }

        analysis = self.strategy.analyze_battle_situation(game_state, battle_context)

        assert analysis['strategy_type'] == 'aggressive'
        assert analysis['player_hp_ratio'] == 0.8
        assert analysis['enemy_hp_ratio'] == 0.3
        assert analysis['damage_focus'] is True
        assert analysis['risk_tolerance'] == 'high'
        assert analysis['recommended_approach'] == 'finish_off'
        assert analysis['urgency_level'] == 4  # High urgency to finish enemy

    def test_analyze_battle_situation_losing(self):
        """Test battle analysis when losing"""
        game_state = {
            'player_hp': 20,
            'player_max_hp': 100
        }
        battle_context = {
            'enemy_hp': 80,
            'enemy_max_hp': 100
        }

        analysis = self.strategy.analyze_battle_situation(game_state, battle_context)

        assert analysis['player_hp_ratio'] == 0.2
        assert analysis['enemy_hp_ratio'] == 0.8
        assert analysis['recommended_approach'] == 'desperate_attack'
        assert analysis['urgency_level'] == 5  # Critical

    def test_recommend_move_with_moves(self):
        """Test move recommendation with available moves"""
        game_state = {'player_hp': 100, 'player_max_hp': 100}
        available_moves = [
            {
                'name': 'tackle',
                'action': 5,
                'power': 40,
                'accuracy': 100,
                'type_effectiveness': 'normal'
            },
            {
                'name': 'thunderbolt',
                'action': 6,
                'power': 90,
                'accuracy': 100,
                'type_effectiveness': 'super_effective'
            },
            {
                'name': 'focus_energy',
                'action': 7,
                'power': 0,
                'accuracy': 100,
                'crit_chance': 50
            }
        ]

        recommendation = self.strategy.recommend_move(game_state, available_moves)

        # Should recommend the super effective move (thunderbolt)
        assert recommendation['action'] == 6
        assert recommendation['move_type'] == 'aggressive_attack'
        assert 'thunderbolt' in recommendation['reasoning']
        assert recommendation['confidence'] > 0.8
        assert 'move_details' in recommendation

    def test_recommend_move_no_moves(self):
        """Test move recommendation with no available moves"""
        recommendation = self.strategy.recommend_move({}, [])

        assert recommendation['action'] == 5  # Default A button
        assert recommendation['confidence'] == 0.3
        assert 'No moves available' in recommendation['reasoning']
        assert recommendation['move_type'] == 'default'

    def test_recommend_move_scoring(self):
        """Test move scoring algorithm"""
        game_state = {}

        # Test move with high power
        high_power_move = {
            'power': 100,
            'accuracy': 100,
            'type_effectiveness': 'normal'
        }
        score1 = self.strategy._score_aggressive_move(high_power_move, game_state)

        # Test move with super effective but lower power
        super_effective_move = {
            'power': 60,
            'accuracy': 100,
            'type_effectiveness': 'super_effective'
        }
        score2 = self.strategy._score_aggressive_move(super_effective_move, game_state)

        # Super effective should score higher due to 2x multiplier
        assert score2 > score1

    def test_assess_switch_opportunity_healthy(self):
        """Test switch assessment when Pokemon is healthy"""
        game_state = {
            'player_hp': 80,
            'player_max_hp': 100
        }
        party = [
            {'name': 'pikachu', 'hp': 50, 'can_battle': True},
            {'name': 'charizard', 'hp': 100, 'can_battle': True}
        ]

        assessment = self.strategy.assess_switch_opportunity(game_state, party)

        # Aggressive strategy should rarely switch when healthy
        assert assessment['should_switch'] is False
        assert assessment['confidence'] == 0.9
        assert 'stay and fight' in assessment['reasoning']

    def test_assess_switch_opportunity_critical(self):
        """Test switch assessment when Pokemon has critical HP"""
        game_state = {
            'player_hp': 10,
            'player_max_hp': 100
        }
        party = [
            {'name': 'pikachu', 'hp': 50, 'can_battle': True},
            {'name': 'charizard', 'hp': 100, 'can_battle': True}
        ]

        assessment = self.strategy.assess_switch_opportunity(game_state, party)

        # Should switch when HP is critical
        assert assessment['should_switch'] is True
        assert assessment['target_pokemon'] == 0  # First available Pokemon
        assert assessment['switch_type'] == 'emergency'
        assert 'Critical HP switch' in assessment['reasoning']

    def test_assess_switch_opportunity_no_alternatives(self):
        """Test switch assessment with no viable alternatives"""
        game_state = {
            'player_hp': 10,
            'player_max_hp': 100
        }
        party = [
            {'name': 'pikachu', 'hp': 0, 'can_battle': False},
            {'name': 'charizard', 'hp': 0, 'can_battle': False}
        ]

        assessment = self.strategy.assess_switch_opportunity(game_state, party)

        # Should not switch if no alternatives
        assert assessment['should_switch'] is False

    def test_urgency_calculation(self):
        """Test urgency level calculation"""
        # Critical HP
        urgency1 = self.strategy._calculate_urgency(0.1, 0.8)
        assert urgency1 == 5

        # Enemy almost defeated
        urgency2 = self.strategy._calculate_urgency(0.8, 0.2)
        assert urgency2 == 4

        # Moderate HP loss
        urgency3 = self.strategy._calculate_urgency(0.4, 0.7)
        assert urgency3 == 3

        # Healthy state
        urgency4 = self.strategy._calculate_urgency(0.9, 0.8)
        assert urgency4 == 2


class TestDefensiveBattleStrategy:
    """Test DefensiveBattleStrategy plugin"""

    def setup_method(self):
        """Setup for each test"""
        self.strategy = DefensiveBattleStrategy()
        self.strategy.initialize()

    def test_metadata(self):
        """Test plugin metadata"""
        metadata = self.strategy.get_metadata()

        assert metadata.name == "defensive_battle"
        assert metadata.plugin_type == PluginType.BATTLE_STRATEGY
        assert metadata.priority == 5
        assert "defensive" in metadata.tags

    def test_initialization_with_config(self):
        """Test initialization with custom config"""
        config = {
            'defense_threshold': 0.3,
            'healing_priority': False
        }
        strategy = DefensiveBattleStrategy(config)
        strategy.initialize()

        assert strategy.defense_threshold == 0.3
        assert strategy.healing_priority is False

    def test_analyze_battle_situation(self):
        """Test defensive battle analysis"""
        game_state = {
            'player_hp': 30,
            'player_max_hp': 100
        }
        battle_context = {
            'enemy_hp': 80,
            'enemy_max_hp': 100
        }

        analysis = self.strategy.analyze_battle_situation(game_state, battle_context)

        assert analysis['strategy_type'] == 'defensive'
        assert analysis['player_hp_ratio'] == 0.3
        assert analysis['damage_focus'] is False
        assert analysis['risk_tolerance'] == 'low'
        assert analysis['healing_needed'] is True  # Below default threshold of 0.5
        assert analysis['recommended_approach'] == 'cautious_defense'

    def test_recommend_move_healing_priority(self):
        """Test move recommendation prioritizing healing"""
        game_state = {
            'player_hp': 30,
            'player_max_hp': 100
        }
        available_moves = [
            {
                'name': 'tackle',
                'action': 5,
                'power': 40,
                'category': 'physical'
            },
            {
                'name': 'recover',
                'action': 6,
                'category': 'healing'
            }
        ]

        recommendation = self.strategy.recommend_move(game_state, available_moves)

        # Should prioritize healing move
        assert recommendation['action'] == 6
        assert recommendation['move_type'] == 'healing'
        assert 'healing' in recommendation['reasoning']
        assert recommendation['confidence'] == 0.9

    def test_recommend_move_status_when_safe(self):
        """Test move recommendation using status moves when safe"""
        game_state = {
            'player_hp': 80,
            'player_max_hp': 100
        }
        available_moves = [
            {
                'name': 'tackle',
                'action': 5,
                'power': 40,
                'category': 'physical'
            },
            {
                'name': 'sleep_powder',
                'action': 6,
                'category': 'status',
                'status_effect': 'sleep'
            }
        ]

        recommendation = self.strategy.recommend_move(game_state, available_moves)

        # Should use status move when HP is good
        assert recommendation['action'] == 6
        assert recommendation['move_type'] == 'status'
        assert 'status' in recommendation['reasoning']

    def test_recommend_move_safe_attack_fallback(self):
        """Test fallback to safest attack when no healing/status available"""
        game_state = {
            'player_hp': 80,
            'player_max_hp': 100
        }
        available_moves = [
            {
                'name': 'tackle',
                'action': 5,
                'power': 40,
                'accuracy': 100,
                'category': 'physical'
            },
            {
                'name': 'double_edge',
                'action': 6,
                'power': 120,
                'accuracy': 85,
                'category': 'physical'
            }
        ]

        recommendation = self.strategy.recommend_move(game_state, available_moves)

        # Should choose safer move (higher accuracy)
        assert recommendation['action'] == 5
        assert recommendation['move_type'] == 'safe_attack'

    def test_assess_switch_opportunity_early_switch(self):
        """Test that defensive strategy switches earlier than aggressive"""
        game_state = {
            'player_hp': 35,
            'player_max_hp': 100  # 35% HP
        }
        party = [
            {'name': 'pikachu', 'hp': 80, 'can_battle': True}
        ]

        assessment = self.strategy.assess_switch_opportunity(game_state, party)

        # Defensive strategy should switch at 40% HP
        assert assessment['should_switch'] is True
        assert assessment['switch_type'] == 'preservation'
        assert 'preserve' in assessment['reasoning']

    def test_find_healing_move(self):
        """Test finding healing moves"""
        moves_with_heal = [
            {'name': 'tackle', 'category': 'physical'},
            {'name': 'recover', 'category': 'healing'}
        ]
        moves_without_heal = [
            {'name': 'tackle', 'category': 'physical'},
            {'name': 'thunderbolt', 'category': 'special'}
        ]

        heal_move = self.strategy._find_healing_move(moves_with_heal)
        no_heal_move = self.strategy._find_healing_move(moves_without_heal)

        assert heal_move is not None
        assert heal_move['name'] == 'recover'
        assert no_heal_move is None

    def test_find_status_move(self):
        """Test finding status moves"""
        moves_with_status = [
            {'name': 'tackle', 'category': 'physical'},
            {'name': 'sleep_powder', 'category': 'status', 'status_effect': 'sleep'}
        ]

        status_move = self.strategy._find_status_move(moves_with_status)
        assert status_move is not None
        assert status_move['name'] == 'sleep_powder'

    def test_find_safest_attack(self):
        """Test finding safest attack move"""
        moves = [
            {
                'name': 'risky_move',
                'category': 'physical',
                'power': 120,
                'accuracy': 70
            },
            {
                'name': 'safe_move',
                'category': 'physical',
                'power': 60,
                'accuracy': 100
            }
        ]

        safest = self.strategy._find_safest_attack(moves)
        assert safest is not None
        assert safest['name'] == 'safe_move'


class TestBalancedBattleStrategy:
    """Test BalancedBattleStrategy plugin"""

    def setup_method(self):
        """Setup for each test"""
        self.strategy = BalancedBattleStrategy()
        self.strategy.initialize()

    def test_metadata(self):
        """Test plugin metadata"""
        metadata = self.strategy.get_metadata()

        assert metadata.name == "balanced_battle"
        assert metadata.plugin_type == PluginType.BATTLE_STRATEGY
        assert metadata.priority == 6
        assert "balanced" in metadata.tags

    def test_initialization_with_config(self):
        """Test initialization with custom thresholds"""
        config = {
            'aggressive_threshold': 0.8,
            'defensive_threshold': 0.2
        }
        strategy = BalancedBattleStrategy(config)
        strategy.initialize()

        assert strategy.aggressive_threshold == 0.8
        assert strategy.defensive_threshold == 0.2

    def test_analyze_battle_situation_aggressive_mode(self):
        """Test analysis in aggressive mode"""
        game_state = {
            'player_hp': 80,
            'player_max_hp': 100
        }
        battle_context = {
            'enemy_hp': 60,
            'enemy_max_hp': 100
        }

        analysis = self.strategy.analyze_battle_situation(game_state, battle_context)

        assert analysis['strategy_type'] == 'balanced'
        assert analysis['current_mode'] == 'aggressive'
        assert analysis['risk_tolerance'] == 'medium-high'
        assert analysis['recommended_approach'] == 'full_assault'

    def test_analyze_battle_situation_defensive_mode(self):
        """Test analysis in defensive mode"""
        game_state = {
            'player_hp': 20,
            'player_max_hp': 100
        }
        battle_context = {
            'enemy_hp': 80,
            'enemy_max_hp': 100
        }

        analysis = self.strategy.analyze_battle_situation(game_state, battle_context)

        assert analysis['current_mode'] == 'defensive'
        assert analysis['risk_tolerance'] == 'low'
        assert analysis['recommended_approach'] == 'damage_control'

    def test_analyze_battle_situation_balanced_mode(self):
        """Test analysis in balanced mode"""
        game_state = {
            'player_hp': 50,
            'player_max_hp': 100
        }
        battle_context = {
            'enemy_hp': 50,
            'enemy_max_hp': 100
        }

        analysis = self.strategy.analyze_battle_situation(game_state, battle_context)

        assert analysis['current_mode'] == 'balanced'
        assert analysis['risk_tolerance'] == 'medium'
        assert analysis['situation_assessment'] == 'even'

    def test_recommend_move_aggressive_mode(self):
        """Test move recommendation in aggressive mode"""
        game_state = {
            'player_hp': 80,
            'player_max_hp': 100
        }
        available_moves = [
            {
                'name': 'tackle',
                'action': 5,
                'power': 40,
                'accuracy': 100
            },
            {
                'name': 'hyper_beam',
                'action': 6,
                'power': 150,
                'accuracy': 90
            }
        ]

        with patch.object(self.strategy, 'analyze_battle_situation') as mock_analyze:
            mock_analyze.return_value = {'current_mode': 'aggressive'}

            recommendation = self.strategy.recommend_move(game_state, available_moves)

            assert recommendation['move_type'] == 'aggressive'
            assert recommendation['action'] == 6  # Should pick higher power move
            assert 'high damage' in recommendation['reasoning']

    def test_recommend_move_defensive_mode(self):
        """Test move recommendation in defensive mode"""
        game_state = {
            'player_hp': 20,
            'player_max_hp': 100
        }
        available_moves = [
            {
                'name': 'tackle',
                'action': 5,
                'power': 40,
                'accuracy': 100
            },
            {
                'name': 'recover',
                'action': 6,
                'power': 0,
                'accuracy': 100
            }
        ]

        with patch.object(self.strategy, 'analyze_battle_situation') as mock_analyze:
            mock_analyze.return_value = {'current_mode': 'defensive'}

            with patch.object(self.strategy, '_recommend_defensive_move') as mock_defensive:
                mock_defensive.return_value = {
                    'action': 6,
                    'confidence': 0.9,
                    'reasoning': 'Balanced-defensive: healing move',
                    'move_type': 'healing'
                }

                recommendation = self.strategy.recommend_move(game_state, available_moves)

                assert recommendation['move_type'] == 'healing'
                assert 'defensive' in recommendation['reasoning']

    def test_recommend_move_balanced_mode(self):
        """Test move recommendation in balanced mode"""
        game_state = {
            'player_hp': 60,
            'player_max_hp': 100
        }
        available_moves = [
            {
                'name': 'tackle',
                'action': 5,
                'power': 60,
                'accuracy': 100
            },
            {
                'name': 'thunderbolt',
                'action': 6,
                'power': 90,
                'accuracy': 85
            }
        ]

        with patch.object(self.strategy, 'analyze_battle_situation') as mock_analyze:
            mock_analyze.return_value = {'current_mode': 'balanced'}

            recommendation = self.strategy.recommend_move(game_state, available_moves)

            assert recommendation['move_type'] == 'balanced'
            assert 'optimal damage/accuracy ratio' in recommendation['reasoning']

    def test_assess_switch_opportunity_balanced(self):
        """Test balanced switching assessment"""
        game_state = {
            'player_hp': 20,
            'player_max_hp': 100  # 20% HP
        }
        party = [
            {'name': 'pikachu', 'hp': 80, 'can_battle': True}
        ]

        assessment = self.strategy.assess_switch_opportunity(game_state, party)

        # Balanced strategy switches at 25% HP
        assert assessment['should_switch'] is True
        assert assessment['switch_type'] == 'tactical'
        assert 'quarter health' in assessment['reasoning']

    def test_situation_assessment(self):
        """Test battle situation assessment"""
        # Winning situation
        winning = self.strategy._assess_battle_state(0.8, 0.4)
        assert winning == 'winning'

        # Losing situation
        losing = self.strategy._assess_battle_state(0.3, 0.8)
        assert losing == 'losing'

        # Even situation
        even = self.strategy._assess_battle_state(0.6, 0.7)
        assert even == 'even'

    def test_recommended_approach_logic(self):
        """Test recommended approach logic"""
        # Enemy almost defeated
        approach1 = self.strategy._get_balanced_approach(0.8, 0.2)
        assert approach1 == 'finish_enemy'

        # Player in trouble
        approach2 = self.strategy._get_balanced_approach(0.2, 0.8)
        assert approach2 == 'damage_control'

        # Both healthy
        approach3 = self.strategy._get_balanced_approach(0.8, 0.8)
        assert approach3 == 'establish_advantage'

        # Mixed situation
        approach4 = self.strategy._get_balanced_approach(0.5, 0.6)
        assert approach4 == 'maintain_pressure'


@pytest.mark.integration
class TestBattleStrategyIntegration:
    """Integration tests for battle strategy plugins"""

    def test_all_strategies_consistency(self):
        """Test that all strategies implement required interface consistently"""
        strategies = [
            AggressiveBattleStrategy(),
            DefensiveBattleStrategy(),
            BalancedBattleStrategy()
        ]

        game_state = {
            'player_hp': 50,
            'player_max_hp': 100
        }
        battle_context = {
            'enemy_hp': 60,
            'enemy_max_hp': 100
        }
        available_moves = [
            {'name': 'tackle', 'action': 5, 'power': 40, 'accuracy': 100}
        ]
        party = [
            {'name': 'pikachu', 'hp': 80, 'can_battle': True}
        ]

        for strategy in strategies:
            strategy.initialize()

            # Test all required methods
            analysis = strategy.analyze_battle_situation(game_state, battle_context)
            assert 'strategy_type' in analysis
            assert 'confidence' in analysis or 'player_hp_ratio' in analysis

            recommendation = strategy.recommend_move(game_state, available_moves)
            assert 'action' in recommendation
            assert 'confidence' in recommendation
            assert 'reasoning' in recommendation

            switch_assessment = strategy.assess_switch_opportunity(game_state, party)
            assert 'should_switch' in switch_assessment
            assert 'confidence' in switch_assessment
            assert 'reasoning' in switch_assessment

            strategy.shutdown()

    def test_strategy_performance_tracking(self):
        """Test that strategies track performance correctly"""
        strategy = AggressiveBattleStrategy()
        strategy.initialize()

        initial_calls = strategy.performance_stats['calls']

        # Make several calls
        for _ in range(5):
            strategy.analyze_battle_situation({}, {})
            strategy.recommend_move({}, [])
            strategy.assess_switch_opportunity({}, [])

        # Performance should be tracked
        assert strategy.performance_stats['calls'] > initial_calls
        assert strategy.performance_stats['total_time'] > 0

    def test_strategy_config_variations(self):
        """Test strategies with different configurations"""
        # Test aggressive with different aggression levels
        aggressive_mild = AggressiveBattleStrategy({'aggression_factor': 0.5})
        aggressive_extreme = AggressiveBattleStrategy({'aggression_factor': 1.0})

        aggressive_mild.initialize()
        aggressive_extreme.initialize()

        # Both should work but might score moves differently
        moves = [{'power': 100, 'accuracy': 80, 'type_effectiveness': 'normal'}]
        score_mild = aggressive_mild._score_aggressive_move(moves[0], {})
        score_extreme = aggressive_extreme._score_aggressive_move(moves[0], {})

        assert score_extreme > score_mild  # More aggressive should score higher

        # Test defensive with different thresholds
        defensive_cautious = DefensiveBattleStrategy({'defense_threshold': 0.8})
        defensive_normal = DefensiveBattleStrategy({'defense_threshold': 0.5})

        defensive_cautious.initialize()
        defensive_normal.initialize()

        # Test that different thresholds affect behavior
        game_state = {'player_hp': 60, 'player_max_hp': 100}  # 60% HP

        analysis_cautious = defensive_cautious.analyze_battle_situation(game_state, {})
        analysis_normal = defensive_normal.analyze_battle_situation(game_state, {})

        # Cautious should need healing, normal should not
        assert analysis_cautious['healing_needed'] is True
        assert analysis_normal['healing_needed'] is False

    @patch('time.time')
    def test_strategy_error_handling(self, mock_time):
        """Test strategy error handling and recovery"""
        mock_time.return_value = 1000.0

        strategy = AggressiveBattleStrategy()
        strategy.initialize()

        # Test with malformed game state
        malformed_state = {'invalid_key': 'invalid_value'}

        # Should not crash and return reasonable defaults
        analysis = strategy.analyze_battle_situation(malformed_state, {})
        assert isinstance(analysis, dict)
        assert 'strategy_type' in analysis

        recommendation = strategy.recommend_move(malformed_state, [])
        assert isinstance(recommendation, dict)
        assert 'action' in recommendation

        assessment = strategy.assess_switch_opportunity(malformed_state, [])
        assert isinstance(assessment, dict)
        assert 'should_switch' in assessment