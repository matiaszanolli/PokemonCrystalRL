"""
Comprehensive tests for agents package __init__.py module.

Tests the agents package interface including imports, exports, and module structure.
"""

import pytest
from unittest.mock import patch


class TestAgentsPackageImports:
    """Test agents package import functionality."""

    def test_import_agents_package(self):
        """Test importing the agents package."""
        import agents

        # Should be able to import without error
        assert agents is not None

    def test_import_base_agent(self):
        """Test importing BaseAgent from agents package."""
        from agents import BaseAgent

        assert BaseAgent is not None
        assert hasattr(BaseAgent, '__init__')
        assert hasattr(BaseAgent, 'update')
        assert hasattr(BaseAgent, 'reset')
        assert hasattr(BaseAgent, 'get_stats')

    def test_import_llm_agent(self):
        """Test importing LLMAgent from agents package."""
        from agents import LLMAgent

        assert LLMAgent is not None
        assert hasattr(LLMAgent, '__init__')
        assert hasattr(LLMAgent, 'get_action')
        assert hasattr(LLMAgent, 'get_decision')

    def test_import_dqn_agent(self):
        """Test importing DQNAgent from agents package."""
        from agents import DQNAgent

        assert DQNAgent is not None
        assert hasattr(DQNAgent, '__init__')
        assert hasattr(DQNAgent, 'get_action')
        assert hasattr(DQNAgent, 'train_step')

    def test_import_hybrid_agent(self):
        """Test importing HybridAgent from agents package."""
        from agents import HybridAgent

        assert HybridAgent is not None
        assert hasattr(HybridAgent, '__init__')
        assert hasattr(HybridAgent, 'get_action')
        assert hasattr(HybridAgent, 'set_mode')

    def test_import_all_agents_at_once(self):
        """Test importing all agents in one statement."""
        from agents import BaseAgent, LLMAgent, DQNAgent, HybridAgent

        assert BaseAgent is not None
        assert LLMAgent is not None
        assert DQNAgent is not None
        assert HybridAgent is not None

    def test_star_import(self):
        """Test star import from agents package."""
        # Import everything exposed in __all__
        import agents

        # Test that star import would work at module level
        # by checking the __all__ attribute
        expected_exports = ['BaseAgent', 'LLMAgent', 'DQNAgent', 'HybridAgent']
        for export in expected_exports:
            assert export in agents.__all__
            assert hasattr(agents, export)


class TestAgentsPackageStructure:
    """Test agents package structure and interface."""

    def test_package_has_all_attribute(self):
        """Test that agents package has __all__ defined."""
        import agents

        assert hasattr(agents, '__all__')
        assert isinstance(agents.__all__, list)

    def test_all_contains_expected_exports(self):
        """Test that __all__ contains expected exports."""
        import agents

        expected_exports = ['BaseAgent', 'LLMAgent', 'DQNAgent', 'HybridAgent']

        for export in expected_exports:
            assert export in agents.__all__, f"{export} not in __all__"

    def test_all_exports_are_importable(self):
        """Test that all items in __all__ can be imported."""
        import agents

        for item_name in agents.__all__:
            assert hasattr(agents, item_name), f"{item_name} not available in package"
            item = getattr(agents, item_name)
            assert item is not None, f"{item_name} is None"

    def test_no_deprecated_exports(self):
        """Test that deprecated items are not exported."""
        import agents

        # BasicHybridTrainer was removed and should not be exported
        assert 'BasicHybridTrainer' not in agents.__all__

        # Should not be available as an attribute either
        assert not hasattr(agents, 'BasicHybridTrainer')

    def test_package_docstring(self):
        """Test that package has proper docstring."""
        import agents

        assert agents.__doc__ is not None
        assert len(agents.__doc__.strip()) > 0
        assert 'Pokemon Crystal' in agents.__doc__ or 'agents' in agents.__doc__.lower()


class TestAgentsClassHierarchy:
    """Test class relationships and inheritance."""

    def test_base_agent_is_base_class(self):
        """Test that BaseAgent is the base class."""
        from agents import BaseAgent
        from interfaces.trainers import AgentInterface

        # BaseAgent should inherit from AgentInterface
        assert issubclass(BaseAgent, AgentInterface)

    def test_llm_agent_inheritance(self):
        """Test LLMAgent inheritance."""
        from agents import LLMAgent, BaseAgent

        # LLMAgent should inherit from BaseAgent
        assert issubclass(LLMAgent, BaseAgent)

    def test_agents_have_common_interface(self):
        """Test that all agents implement common interface methods."""
        from agents import BaseAgent, LLMAgent

        common_methods = ['update', 'reset', 'get_stats', 'train', 'eval', 'get_action']

        for method in common_methods:
            assert hasattr(BaseAgent, method), f"BaseAgent missing {method}"
            assert hasattr(LLMAgent, method), f"LLMAgent missing {method}"

    def test_dqn_agent_standalone(self):
        """Test that DQNAgent can be used standalone."""
        from agents import DQNAgent

        # DQNAgent should have its own methods
        assert hasattr(DQNAgent, 'get_action')
        assert hasattr(DQNAgent, 'train_step')
        assert hasattr(DQNAgent, 'save_model')
        assert hasattr(DQNAgent, 'load_model')

    def test_hybrid_agent_composition(self):
        """Test that HybridAgent composes other agents."""
        from agents import HybridAgent

        # HybridAgent should have hybrid-specific methods
        assert hasattr(HybridAgent, 'get_action')
        assert hasattr(HybridAgent, 'set_mode')
        assert hasattr(HybridAgent, 'get_stats')
        assert hasattr(HybridAgent, 'save_state')
        assert hasattr(HybridAgent, 'load_state')


class TestAgentsInstantiation:
    """Test that agents can be instantiated correctly."""

    def test_base_agent_instantiation(self):
        """Test BaseAgent interface exists and concrete agents can inherit from it."""
        from agents import BaseAgent
        from interfaces.trainers import AgentInterface

        # BaseAgent is abstract, so test inheritance structure
        assert issubclass(BaseAgent, AgentInterface)
        assert hasattr(BaseAgent, '__init__')
        assert hasattr(BaseAgent, 'update')
        assert hasattr(BaseAgent, 'reset')
        assert hasattr(BaseAgent, 'get_stats')

    @patch('agents.llm_agent.GameIntelligence')
    @patch('agents.llm_agent.ExperienceMemory')
    @patch('agents.llm_agent.StrategicContextBuilder')
    @patch('requests.get')
    def test_llm_agent_instantiation(self, mock_get, mock_context, mock_memory, mock_intelligence):
        """Test LLMAgent can be instantiated."""
        from agents import LLMAgent

        # Mock the LLM connection test
        mock_response = mock_get.return_value
        mock_response.status_code = 200

        agent = LLMAgent('test-model')
        assert agent is not None
        assert agent.model_name == 'test-model'

    def test_dqn_agent_instantiation(self):
        """Test DQNAgent can be instantiated."""
        from agents import DQNAgent

        agent = DQNAgent()
        assert agent is not None
        assert agent.state_size == 32
        assert agent.action_size == 8

    @patch('agents.hybrid_agent.LLMAgent')
    @patch('agents.hybrid_agent.RLAgent')
    def test_hybrid_agent_instantiation(self, mock_rl_agent, mock_llm_agent):
        """Test HybridAgent can be instantiated."""
        from agents import HybridAgent
        from unittest.mock import Mock

        mock_llm_manager = Mock()
        mock_adaptive_strategy = Mock()

        agent = HybridAgent(mock_llm_manager, mock_adaptive_strategy)
        assert agent is not None
        assert hasattr(agent, 'mode')


class TestAgentsCompatibility:
    """Test agents compatibility and interface consistency."""

    def test_agent_interface_compatibility(self):
        """Test that agents are compatible with AgentInterface."""
        from agents import BaseAgent, LLMAgent
        from interfaces.trainers import AgentInterface

        # BaseAgent should be compatible
        assert issubclass(BaseAgent, AgentInterface)

        # LLMAgent should be compatible through inheritance
        assert issubclass(LLMAgent, AgentInterface)

    @patch('agents.llm_agent.GameIntelligence')
    @patch('agents.llm_agent.ExperienceMemory')
    @patch('agents.llm_agent.StrategicContextBuilder')
    @patch('requests.get')
    def test_agents_common_methods(self, mock_get, mock_context, mock_memory, mock_intelligence):
        """Test that agents have common methods that work."""
        from agents import BaseAgent, LLMAgent

        # Mock LLM connection
        mock_response = mock_get.return_value
        mock_response.status_code = 200

        # Create a concrete test agent that implements get_action
        class ConcreteTestAgent(BaseAgent):
            def get_action(self, observation, info=None):
                return "up"

        base_agent = ConcreteTestAgent()
        llm_agent = LLMAgent('test-model')

        # Test common methods exist and work
        for agent in [base_agent, llm_agent]:
            # Update method
            agent.update(1.0)
            assert agent.total_reward == 1.0

            # Reset method
            agent.reset()
            assert agent.episode_reward == 0.0

            # Stats method
            stats = agent.get_stats()
            assert isinstance(stats, dict)
            assert 'total_steps' in stats

            # Training mode methods
            agent.train()
            assert agent.is_training is True

            agent.eval()
            assert agent.is_training is False

    def test_action_method_consistency(self):
        """Test that agents have consistent action methods."""
        from agents import BaseAgent, LLMAgent, DQNAgent

        # All should have get_action method
        assert hasattr(BaseAgent, 'get_action')
        assert hasattr(LLMAgent, 'get_action')
        assert hasattr(DQNAgent, 'get_action')


class TestAgentsModuleIntegration:
    """Integration tests for the agents module."""

    def test_full_module_import_cycle(self):
        """Test complete import cycle of the module."""
        # Should be able to import and use all components
        from agents import BaseAgent, LLMAgent, DQNAgent, HybridAgent

        # Should be able to create instances (with mocking for complex ones)
        # BaseAgent is abstract, so create a concrete implementation
        class ConcreteTestAgent(BaseAgent):
            def get_action(self, observation, info=None):
                return "up"

        base_agent = ConcreteTestAgent()
        assert base_agent is not None

        # Should be able to use common functionality
        base_agent.update(0.5)
        stats = base_agent.get_stats()
        assert stats['total_reward'] == 0.5

    def test_module_documentation_accessibility(self):
        """Test that module documentation is accessible."""
        import agents

        # Package should have docstring
        assert agents.__doc__ is not None

        # Classes should have docstrings
        from agents import BaseAgent, LLMAgent, DQNAgent, HybridAgent

        for cls in [BaseAgent, LLMAgent, DQNAgent, HybridAgent]:
            assert cls.__doc__ is not None
            assert len(cls.__doc__.strip()) > 0

    def test_no_circular_imports(self):
        """Test that there are no circular import issues."""
        # This test passes if the import completes without error
        try:
            from agents import BaseAgent, LLMAgent, DQNAgent, HybridAgent
            # Force import of all submodules
            import agents.base_agent
            import agents.llm_agent
            import agents.dqn_agent
            import agents.hybrid_agent

            success = True
        except ImportError as e:
            success = False
            pytest.fail(f"Circular import detected: {e}")

        assert success

    def test_module_compatibility_with_main(self):
        """Test that agents module is compatible with main.py usage."""
        # Test the import pattern used in main.py
        try:
            from agents import LLMAgent, DQNAgent, HybridAgent

            # These should be the classes used in main.py
            assert LLMAgent is not None
            assert DQNAgent is not None
            assert HybridAgent is not None

        except ImportError as e:
            pytest.fail(f"Agents module not compatible with main.py: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])