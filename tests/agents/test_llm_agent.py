"""
Comprehensive tests for LLMAgent class.

Tests the LLM-based decision making agent including initialization, decision making,
prompt building, response parsing, fallback mechanisms, and integration with
game intelligence systems.
"""

import pytest
import time
import requests
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Tuple, Any

from agents.llm_agent import LLMAgent


class TestLLMAgentInitialization:
    """Test LLMAgent initialization with different configurations."""

    def test_init_old_style_minimal(self):
        """Test initialization with old-style minimal arguments."""
        agent = LLMAgent('test-model')

        assert agent.model_name == 'test-model'
        assert agent.base_url == 'http://localhost:11434'
        assert isinstance(agent.decision_history, list)
        assert len(agent.decision_history) == 0
        assert agent.last_decision_time == 0

    def test_init_old_style_full(self):
        """Test initialization with old-style full arguments."""
        agent = LLMAgent('custom-model', 'http://custom:8080')

        assert agent.model_name == 'custom-model'
        assert agent.base_url == 'http://custom:8080'

    def test_init_new_style_config(self):
        """Test initialization with new-style config dictionary."""
        config = {
            'model_name': 'new-model',
            'base_url': 'http://new:9090',
            'use_game_intelligence': False,
            'use_experience_memory': False,
            'use_strategic_context': False
        }
        agent = LLMAgent(config=config)

        assert agent.model_name == 'new-model'
        assert agent.base_url == 'http://new:9090'
        assert agent.use_game_intelligence is False
        assert agent.use_experience_memory is False
        assert agent.use_strategic_context is False

    def test_init_empty_config(self):
        """Test initialization with empty config."""
        agent = LLMAgent(config={})

        assert agent.model_name == 'smollm2:1.7b'  # Default
        assert agent.base_url == 'http://localhost:11434'  # Default

    def test_init_none_config(self):
        """Test initialization with None config."""
        agent = LLMAgent(config=None)

        assert agent.model_name == 'smollm2:1.7b'  # Default
        assert agent.base_url == 'http://localhost:11434'  # Default

    @patch('agents.llm_agent.LLMAgent._test_llm_connection')
    def test_init_systems_creation(self, mock_test_connection):
        """Test that all required systems are created during initialization."""
        mock_test_connection.return_value = True

        agent = LLMAgent('test-model')

        assert hasattr(agent, 'game_intelligence')
        assert hasattr(agent, 'experience_memory')
        assert hasattr(agent, 'context_builder')
        assert agent.available is True

    @patch('agents.llm_agent.LLMAgent._test_llm_connection')
    def test_init_llm_unavailable(self, mock_test_connection):
        """Test initialization when LLM is unavailable."""
        mock_test_connection.return_value = False

        agent = LLMAgent('test-model')

        assert agent.available is False


class TestLLMConnectionTesting:
    """Test LLM connection testing functionality."""

    @patch('requests.get')
    def test_connection_success(self, mock_get):
        """Test successful LLM connection."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        agent = LLMAgent('test-model')
        result = agent._test_llm_connection()

        assert result is True
        # Should be called twice: once during init, once during explicit call
        assert mock_get.call_count == 2
        mock_get.assert_called_with('http://localhost:11434/api/tags', timeout=5)

    @patch('requests.get')
    def test_connection_failure_status(self, mock_get):
        """Test LLM connection failure due to status code."""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response

        agent = LLMAgent('test-model')
        result = agent._test_llm_connection()

        assert result is False

    @patch('requests.get')
    def test_connection_failure_exception(self, mock_get):
        """Test LLM connection failure due to exception."""
        mock_get.side_effect = requests.RequestException("Connection failed")

        agent = LLMAgent('test-model')
        result = agent._test_llm_connection()

        assert result is False

    @patch('requests.get')
    def test_connection_timeout(self, mock_get):
        """Test LLM connection timeout."""
        mock_get.side_effect = requests.Timeout("Request timed out")

        agent = LLMAgent('test-model')
        result = agent._test_llm_connection()

        assert result is False


class TestLLMAgentGetAction:
    """Test LLMAgent get_action method (AgentInterface compatibility)."""

    @patch('agents.llm_agent.LLMAgent.get_decision')
    def test_get_action_basic(self, mock_get_decision):
        """Test basic get_action functionality."""
        mock_get_decision.return_value = ('up', 'Moving up for exploration')

        agent = LLMAgent('test-model')
        observation = {'player_x': 10, 'player_y': 20}
        info = {'state': 'overworld'}

        action = agent.get_action(observation, info)

        assert action == 'up'
        mock_get_decision.assert_called_once_with(observation, info, [])

    @patch('agents.llm_agent.LLMAgent.get_decision')
    def test_get_action_no_info(self, mock_get_decision):
        """Test get_action with no info parameter."""
        mock_get_decision.return_value = ('a', 'Interacting with object')

        agent = LLMAgent('test-model')
        observation = {'player_x': 10, 'player_y': 20}

        action = agent.get_action(observation)

        assert action == 'a'
        mock_get_decision.assert_called_once_with(observation, {}, [])


class TestLLMAgentDecisionMaking:
    """Test LLM decision making functionality."""

    def test_get_decision_llm_unavailable(self):
        """Test decision making when LLM is unavailable."""
        with patch('agents.llm_agent.LLMAgent._test_llm_connection', return_value=False):
            agent = LLMAgent('test-model')

        game_state = {'player_x': 10}
        screen_analysis = {'state': 'overworld'}
        recent_actions = ['up', 'right']

        with patch.object(agent, '_fallback_decision', return_value='down') as mock_fallback:
            action, reasoning = agent.get_decision(game_state, screen_analysis, recent_actions)

        assert action == 'down'
        assert reasoning == "LLM unavailable - using fallback"
        mock_fallback.assert_called_once_with(game_state)

    @patch('requests.post')
    @patch('agents.llm_agent.LLMAgent._test_llm_connection')
    def test_get_decision_success(self, mock_test_connection, mock_post):
        """Test successful LLM decision making."""
        mock_test_connection.return_value = True

        # Mock successful LLM response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'response': 'ACTION: up\nReasoning: Need to explore north'
        }
        mock_post.return_value = mock_response

        agent = LLMAgent('test-model')

        with patch.object(agent, '_build_prompt', return_value="test prompt"):
            with patch.object(agent, '_parse_llm_response', return_value='up'):
                action, reasoning = agent.get_decision({}, {}, [])

        assert action == 'up'
        assert 'Need to explore north' in reasoning

    @patch('requests.post')
    @patch('agents.llm_agent.LLMAgent._test_llm_connection')
    def test_get_decision_request_failure(self, mock_test_connection, mock_post):
        """Test LLM decision making with request failure."""
        mock_test_connection.return_value = True

        # Mock failed LLM response
        mock_response = Mock()
        mock_response.status_code = 500
        mock_post.return_value = mock_response

        agent = LLMAgent('test-model')

        with patch.object(agent, '_fallback_decision', return_value='b') as mock_fallback:
            action, reasoning = agent.get_decision({}, {}, [])

        assert action == 'b'
        assert reasoning == "LLM request failed"

    @patch('requests.post')
    @patch('agents.llm_agent.LLMAgent._test_llm_connection')
    def test_get_decision_exception(self, mock_test_connection, mock_post):
        """Test LLM decision making with exception."""
        mock_test_connection.return_value = True
        mock_post.side_effect = Exception("Network error")

        agent = LLMAgent('test-model')

        with patch.object(agent, '_fallback_decision', return_value='a') as mock_fallback:
            action, reasoning = agent.get_decision({}, {}, [])

        assert action == 'a'
        assert "LLM error: Network error" in reasoning

    @patch('agents.llm_agent.LLMAgent._test_llm_connection')
    def test_decision_history_tracking(self, mock_test_connection):
        """Test that decisions are tracked in history."""
        mock_test_connection.return_value = True

        agent = LLMAgent('test-model')

        with patch('requests.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {'response': 'ACTION: left\nGoing left'}
            mock_post.return_value = mock_response

            with patch.object(agent, '_build_prompt', return_value="prompt"):
                with patch.object(agent, '_parse_llm_response', return_value='left'):
                    game_state = {'test': 'state'}
                    action, reasoning = agent.get_decision(game_state, {}, [])

        assert len(agent.decision_history) == 1
        decision = agent.decision_history[0]
        assert decision['action'] == 'left'
        assert decision['reasoning'] == 'Going left'
        assert decision['game_state'] == game_state
        assert 'timestamp' in decision


class TestLLMResponseParsing:
    """Test LLM response parsing functionality."""

    def test_parse_action_format(self):
        """Test parsing ACTION: format."""
        agent = LLMAgent('test-model')

        response = "ACTION: up\nReasoning: Need to go north"
        action = agent._parse_llm_response(response)
        assert action == 'up'

    def test_parse_direct_action_words(self):
        """Test parsing direct action words."""
        agent = LLMAgent('test-model')

        test_cases = [
            ('Move up to explore', 'up'),
            ('Press the interact button', 'a'),
            ('Cancel this menu', 'b'),
            ('Go left towards the door', 'left'),
            ('Run away from battle', 'b'),  # 'run' maps to 'b'
            ('Use start menu', 'start'),
        ]

        for response, expected in test_cases:
            action = agent._parse_llm_response(response)
            assert action == expected, f"Failed for '{response}': expected '{expected}', got '{action}'"

    def test_parse_synonyms(self):
        """Test parsing action synonyms."""
        agent = LLMAgent('test-model')

        synonym_tests = [
            ('Move north', 'up'),
            ('Go south', 'down'),
            ('Head west', 'left'),
            ('Move east', 'right'),
            ('Confirm selection', 'a'),
            ('Escape menu', 'b'),
            ('Open pause menu', 'start'),
        ]

        for response, expected in synonym_tests:
            action = agent._parse_llm_response(response)
            assert action == expected

    def test_parse_no_match_fallback(self):
        """Test fallback when no action is found."""
        agent = LLMAgent('test-model')

        response = "I'm not sure what to do here"
        action = agent._parse_llm_response(response)
        assert action == 'a'  # Default fallback

    def test_parse_case_insensitive(self):
        """Test case-insensitive parsing."""
        agent = LLMAgent('test-model')

        test_cases = [
            ('ACTION: UP', 'up'),
            ('Move DOWN', 'down'),
            ('Press A Button', 'a'),
            ('INTERACT with object', 'a'),
        ]

        for response, expected in test_cases:
            action = agent._parse_llm_response(response)
            assert action == expected


class TestLLMFallbackDecision:
    """Test LLM fallback decision mechanisms."""

    def test_fallback_battle_state(self):
        """Test fallback decision in battle."""
        agent = LLMAgent('test-model')

        game_state = {'in_battle': 1}
        action = agent._fallback_decision(game_state)
        assert action == 'a'  # Attack in battle

    def test_fallback_menu_state(self):
        """Test fallback decision in menu."""
        agent = LLMAgent('test-model')

        game_state = {'menu_state': 1}
        action = agent._fallback_decision(game_state)
        assert action == 'a'  # Navigate menus

    def test_fallback_stuck_pattern_detection(self):
        """Test fallback with stuck pattern detection."""
        agent = LLMAgent('test-model')
        agent.recent_actions = ['up', 'up', 'up', 'up', 'up']

        action = agent._fallback_decision({})
        # Should return an action not in recent actions
        assert action not in ['up']

    def test_fallback_phase_aware(self):
        """Test fallback decision is phase-aware."""
        agent = LLMAgent('test-model')

        # Early game phase
        game_state = {'phase': 'early_game'}
        action = agent._fallback_decision(game_state)
        assert action in ['a', 'up', 'right', 'down']

        # Starter phase
        game_state = {'phase': 'starter_phase'}
        action = agent._fallback_decision(game_state)
        assert action in ['a', 'up', 'down', 'left', 'right']

    def test_fallback_time_based(self):
        """Test fallback uses time-based selection."""
        agent = LLMAgent('test-model')

        # Should be deterministic based on time
        with patch('time.time', return_value=123456):
            action1 = agent._fallback_decision({})

        with patch('time.time', return_value=123456):
            action2 = agent._fallback_decision({})

        assert action1 == action2  # Same time should give same result


class TestLLMPromptBuilding:
    """Test LLM prompt building functionality."""

    @patch('agents.llm_agent.LLMAgent._test_llm_connection')
    def test_build_prompt_basic_structure(self, mock_test_connection):
        """Test basic prompt structure."""
        mock_test_connection.return_value = True

        agent = LLMAgent('test-model')

        # Mock the context builder and game intelligence
        mock_context = Mock()
        mock_context.current_analysis = Mock()
        mock_context.current_analysis.phase = Mock()
        mock_context.current_analysis.phase.name = 'early_game'
        mock_context.current_analysis.criticality = Mock()
        mock_context.current_analysis.criticality.value = 2

        with patch.object(agent.context_builder, 'build_context', return_value=mock_context):
            with patch.object(agent.game_intelligence, 'analyze_game_context') as mock_analyze:
                with patch.object(agent.game_intelligence, 'get_action_plan', return_value=[]):
                    with patch.object(agent.game_intelligence, 'get_contextual_advice', return_value="Test advice"):
                        game_state = {'player_x': 10, 'party_count': 0}
                        screen_analysis = {'state': 'overworld'}
                        recent_actions = ['up', 'right']

                        prompt = agent._build_prompt(game_state, screen_analysis, recent_actions, mock_context)

        assert isinstance(prompt, str)
        assert len(prompt) > 100  # Should be substantial prompt
        assert 'Pokemon Crystal' in prompt
        assert 'CURRENT STATUS:' in prompt
        assert 'AVAILABLE ACTIONS:' in prompt

    @patch('agents.llm_agent.LLMAgent._test_llm_connection')
    def test_build_prompt_failsafe_active(self, mock_test_connection):
        """Test prompt building with failsafe intervention active."""
        mock_test_connection.return_value = True

        agent = LLMAgent('test-model')
        agent.failsafe_context = {
            'stuck_detected': True,
            'stuck_location': (24, 0, 0),
            'actions_without_reward': 50
        }

        mock_context = Mock()
        mock_context.current_analysis = None

        with patch.object(agent.context_builder, 'build_context', return_value=mock_context):
            with patch.object(agent.game_intelligence, 'analyze_game_context') as mock_analyze:
                with patch.object(agent.game_intelligence, 'get_action_plan', return_value=[]):
                    with patch.object(agent.game_intelligence, 'get_contextual_advice', return_value="Test advice"):
                        game_state = {'party_count': 0}
                        prompt = agent._build_prompt(game_state, {}, [], mock_context)

        assert 'FAILSAFE INTERVENTION ACTIVE' in prompt
        assert 'STUCK at Map 24' in prompt
        assert 'Actions without progress: 50' in prompt

    @patch('agents.llm_agent.LLMAgent._test_llm_connection')
    def test_build_prompt_battle_context(self, mock_test_connection):
        """Test prompt building with battle context."""
        mock_test_connection.return_value = True

        agent = LLMAgent('test-model')

        mock_context = Mock()
        mock_context.current_analysis = None

        with patch.object(agent.context_builder, 'build_context', return_value=mock_context):
            with patch.object(agent.game_intelligence, 'analyze_game_context') as mock_analyze:
                with patch.object(agent.game_intelligence, 'get_action_plan', return_value=[]):
                    with patch.object(agent.game_intelligence, 'get_contextual_advice', return_value="Test advice"):
                        game_state = {
                            'in_battle': 1,
                            'enemy_level': 5,
                            'enemy_species': 12
                        }
                        prompt = agent._build_prompt(game_state, {}, [], mock_context)

        assert 'IN BATTLE' in prompt
        assert 'Enemy Level 5' in prompt


class TestLLMAgentConfiguration:
    """Test LLMAgent configuration options."""

    @patch('agents.llm_agent.LLMAgent._test_llm_connection')
    def test_disable_game_intelligence(self, mock_test_connection):
        """Test disabling game intelligence."""
        mock_test_connection.return_value = True

        config = {'use_game_intelligence': False}
        agent = LLMAgent(config=config)

        assert agent.use_game_intelligence is False

    @patch('agents.llm_agent.LLMAgent._test_llm_connection')
    def test_disable_experience_memory(self, mock_test_connection):
        """Test disabling experience memory."""
        mock_test_connection.return_value = True

        config = {'use_experience_memory': False}
        agent = LLMAgent(config=config)

        assert agent.use_experience_memory is False

    @patch('agents.llm_agent.LLMAgent._test_llm_connection')
    def test_disable_strategic_context(self, mock_test_connection):
        """Test disabling strategic context."""
        mock_test_connection.return_value = True

        config = {'use_strategic_context': False}
        agent = LLMAgent(config=config)

        assert agent.use_strategic_context is False


@pytest.mark.integration
class TestLLMAgentIntegration:
    """Integration tests for LLMAgent."""

    @patch('requests.get')
    @patch('requests.post')
    def test_full_decision_cycle(self, mock_post, mock_get):
        """Test complete decision making cycle."""
        # Mock connection test
        mock_get.return_value = Mock(status_code=200)

        # Mock LLM response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'response': 'ACTION: up\nReasoning: Exploring to find Pokemon'
        }
        mock_post.return_value = mock_response

        agent = LLMAgent('test-model')

        game_state = {
            'player_x': 10,
            'player_y': 20,
            'party_count': 0,
            'money': 3000
        }
        screen_analysis = {'state': 'overworld', 'variance': 1000}
        recent_actions = ['right', 'down']

        action, reasoning = agent.get_decision(game_state, screen_analysis, recent_actions)

        assert action == 'up'
        assert 'Exploring to find Pokemon' in reasoning
        assert len(agent.decision_history) == 1

    def test_multiple_decisions_tracking(self):
        """Test tracking multiple decisions."""
        with patch('agents.llm_agent.LLMAgent._test_llm_connection', return_value=False):
            agent = LLMAgent('test-model')

        # Make multiple decisions using fallback
        for i in range(5):
            action, reasoning = agent.get_decision({}, {}, [])

        # Since LLM is unavailable, should use fallback but no history
        assert len(agent.decision_history) == 0

        # Now simulate LLM becoming available
        with patch('requests.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {'response': 'ACTION: a\nTest'}
            mock_post.return_value = mock_response

            agent.available = True

            with patch.object(agent, '_build_prompt', return_value="prompt"):
                action, reasoning = agent.get_decision({}, {}, [])

        assert len(agent.decision_history) == 1

    @patch('agents.llm_agent.LLMAgent._test_llm_connection')
    def test_agent_interface_compatibility(self, mock_test_connection):
        """Test compatibility with AgentInterface."""
        mock_test_connection.return_value = False

        agent = LLMAgent('test-model')

        # Test AgentInterface methods
        observation = {'test': 'data'}
        info = {'additional': 'info'}

        # get_action should work
        action = agent.get_action(observation, info)
        assert action in ['up', 'down', 'left', 'right', 'a', 'b', 'start', 'select']

        # BaseAgent methods should work
        agent.update(1.0)
        assert agent.total_reward == 1.0

        agent.reset()
        assert agent.episode_reward == 0.0

        stats = agent.get_stats()
        assert 'total_steps' in stats


if __name__ == "__main__":
    pytest.main([__file__, "-v"])