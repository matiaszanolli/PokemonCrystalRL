"""
Configuration Comparator - Plugin and agent configuration testing

This module provides functionality to compare different configurations of
plugins, agents, and training parameters in A/B testing scenarios.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import copy

from .experiment_models import ExperimentConfig, ExperimentType
from core.plugin_system import PluginType, get_plugin_registry
from agents.multi_agent_coordinator import AgentRole


class ConfigurationType(Enum):
    """Types of configurations that can be compared"""
    PLUGIN_CONFIG = "plugin_config"
    AGENT_CONFIG = "agent_config"
    TRAINING_PARAMS = "training_params"
    HYBRID_CONFIG = "hybrid_config"


@dataclass
class ConfigurationVariant:
    """Represents a configuration variant for A/B testing"""
    name: str
    description: str
    configuration: Dict[str, Any]
    configuration_type: ConfigurationType
    expected_impact: str = ""  # Description of expected performance impact
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> Tuple[bool, str]:
        """
        Validate the configuration variant.

        Returns:
            (is_valid, error_message)
        """
        if not self.name or not self.configuration:
            return False, "Name and configuration are required"

        if self.configuration_type == ConfigurationType.PLUGIN_CONFIG:
            return self._validate_plugin_config()
        elif self.configuration_type == ConfigurationType.AGENT_CONFIG:
            return self._validate_agent_config()
        elif self.configuration_type == ConfigurationType.TRAINING_PARAMS:
            return self._validate_training_params()

        return True, ""

    def _validate_plugin_config(self) -> Tuple[bool, str]:
        """Validate plugin configuration"""
        if 'plugins' not in self.configuration:
            return False, "Plugin configuration must contain 'plugins' key"

        for plugin_name, plugin_config in self.configuration['plugins'].items():
            if not isinstance(plugin_config, dict):
                return False, f"Plugin config for '{plugin_name}' must be a dictionary"

        return True, ""

    def _validate_agent_config(self) -> Tuple[bool, str]:
        """Validate agent configuration"""
        if 'agents' not in self.configuration:
            return False, "Agent configuration must contain 'agents' key"

        for agent_name, agent_config in self.configuration['agents'].items():
            if not isinstance(agent_config, dict):
                return False, f"Agent config for '{agent_name}' must be a dictionary"

        return True, ""

    def _validate_training_params(self) -> Tuple[bool, str]:
        """Validate training parameters"""
        required_params = ['max_actions', 'llm_interval']
        for param in required_params:
            if param not in self.configuration:
                return False, f"Training parameters must contain '{param}'"

        return True, ""


class ConfigurationComparator:
    """
    Handles comparison of different plugin, agent, and training configurations.

    This class provides functionality to:
    - Create configuration variants for A/B testing
    - Compare plugin configurations
    - Compare agent configurations
    - Generate experiment configurations
    """

    def __init__(self):
        self.logger = logging.getLogger("ConfigurationComparator")
        self.plugin_registry = get_plugin_registry()

    def create_plugin_comparison(
        self,
        base_config: Dict[str, Any],
        plugin_variants: Dict[str, Dict[str, Any]],
        experiment_name: str = "Plugin Comparison"
    ) -> ExperimentConfig:
        """
        Create an experiment configuration for plugin comparison.

        Args:
            base_config: Base configuration shared by all variants
            plugin_variants: Dictionary of variant_name -> plugin_config
            experiment_name: Name for the experiment

        Returns:
            ExperimentConfig for the plugin comparison
        """
        experiment_config = ExperimentConfig(
            name=experiment_name,
            experiment_type=ExperimentType.PLUGIN_COMPARISON,
            description=f"Comparing {len(plugin_variants)} plugin configurations"
        )

        # Create variants
        for variant_name, plugin_config in plugin_variants.items():
            variant_config = copy.deepcopy(base_config)
            variant_config['plugins'] = plugin_config

            experiment_config.add_variant(variant_name, variant_config)

        self.logger.info(f"Created plugin comparison experiment with {len(plugin_variants)} variants")
        return experiment_config

    def create_agent_comparison(
        self,
        base_config: Dict[str, Any],
        agent_variants: Dict[str, Dict[str, Any]],
        experiment_name: str = "Agent Comparison"
    ) -> ExperimentConfig:
        """
        Create an experiment configuration for agent comparison.

        Args:
            base_config: Base configuration shared by all variants
            agent_variants: Dictionary of variant_name -> agent_config
            experiment_name: Name for the experiment

        Returns:
            ExperimentConfig for the agent comparison
        """
        experiment_config = ExperimentConfig(
            name=experiment_name,
            experiment_type=ExperimentType.AGENT_COMPARISON,
            description=f"Comparing {len(agent_variants)} agent configurations"
        )

        # Create variants
        for variant_name, agent_config in agent_variants.items():
            variant_config = copy.deepcopy(base_config)
            variant_config['agents'] = agent_config

            experiment_config.add_variant(variant_name, variant_config)

        self.logger.info(f"Created agent comparison experiment with {len(agent_variants)} variants")
        return experiment_config

    def create_battle_strategy_comparison(
        self,
        base_config: Dict[str, Any] = None,
        experiment_name: str = "Battle Strategy Comparison"
    ) -> ExperimentConfig:
        """
        Create a pre-configured experiment comparing different battle strategies.

        Args:
            base_config: Base configuration (optional)
            experiment_name: Name for the experiment

        Returns:
            ExperimentConfig for battle strategy comparison
        """
        if base_config is None:
            base_config = self._get_default_base_config()

        # Define battle strategy variants
        battle_variants = {
            "aggressive": {
                "aggressive_battle_strategy": {
                    "aggression_level": 0.9,
                    "risk_tolerance": 0.8,
                    "priority_attack": True
                }
            },
            "defensive": {
                "defensive_battle_strategy": {
                    "aggression_level": 0.3,
                    "risk_tolerance": 0.2,
                    "priority_healing": True
                }
            },
            "balanced": {
                "balanced_battle_strategy": {
                    "aggression_level": 0.6,
                    "risk_tolerance": 0.5,
                    "adaptive_strategy": True
                }
            }
        }

        return self.create_plugin_comparison(base_config, battle_variants, experiment_name)

    def create_exploration_pattern_comparison(
        self,
        base_config: Dict[str, Any] = None,
        experiment_name: str = "Exploration Pattern Comparison"
    ) -> ExperimentConfig:
        """
        Create a pre-configured experiment comparing exploration patterns.

        Args:
            base_config: Base configuration (optional)
            experiment_name: Name for the experiment

        Returns:
            ExperimentConfig for exploration pattern comparison
        """
        if base_config is None:
            base_config = self._get_default_base_config()

        # Define exploration pattern variants
        exploration_variants = {
            "systematic": {
                "systematic_exploration": {
                    "pattern": "grid_sweep",
                    "coverage_priority": 0.8
                }
            },
            "spiral": {
                "spiral_exploration": {
                    "pattern": "spiral_outward",
                    "center_bias": 0.6
                }
            },
            "random": {
                "random_exploration": {
                    "pattern": "random_walk",
                    "exploration_rate": 0.7
                }
            },
            "wall_following": {
                "wall_following_exploration": {
                    "pattern": "wall_follow",
                    "direction_preference": "right"
                }
            }
        }

        return self.create_plugin_comparison(base_config, exploration_variants, experiment_name)

    def create_multi_agent_comparison(
        self,
        base_config: Dict[str, Any] = None,
        experiment_name: str = "Multi-Agent Strategy Comparison"
    ) -> ExperimentConfig:
        """
        Create experiment comparing different multi-agent configurations.

        Args:
            base_config: Base configuration (optional)
            experiment_name: Name for the experiment

        Returns:
            ExperimentConfig for multi-agent comparison
        """
        if base_config is None:
            base_config = self._get_default_base_config()

        # Define agent coordination variants
        agent_variants = {
            "battle_focused": {
                "coordination_strategy": "priority_based",
                "agent_weights": {
                    "battle": 0.7,
                    "explorer": 0.2,
                    "progression": 0.1
                }
            },
            "exploration_focused": {
                "coordination_strategy": "priority_based",
                "agent_weights": {
                    "battle": 0.2,
                    "explorer": 0.7,
                    "progression": 0.1
                }
            },
            "balanced": {
                "coordination_strategy": "adaptive",
                "agent_weights": {
                    "battle": 0.4,
                    "explorer": 0.4,
                    "progression": 0.2
                }
            },
            "progression_focused": {
                "coordination_strategy": "priority_based",
                "agent_weights": {
                    "battle": 0.2,
                    "explorer": 0.2,
                    "progression": 0.6
                }
            }
        }

        return self.create_agent_comparison(base_config, agent_variants, experiment_name)

    def create_hybrid_comparison(
        self,
        base_config: Dict[str, Any] = None,
        experiment_name: str = "Hybrid Configuration Comparison"
    ) -> ExperimentConfig:
        """
        Create experiment comparing hybrid plugin + agent configurations.

        Args:
            base_config: Base configuration (optional)
            experiment_name: Name for the experiment

        Returns:
            ExperimentConfig for hybrid comparison
        """
        if base_config is None:
            base_config = self._get_default_base_config()

        # Define hybrid variants
        hybrid_variants = {
            "aggressive_explorer": {
                "plugins": {
                    "aggressive_battle_strategy": {"aggression_level": 0.8},
                    "systematic_exploration": {"coverage_priority": 0.9}
                },
                "agents": {
                    "coordination_strategy": "priority_based",
                    "agent_weights": {"battle": 0.6, "explorer": 0.4}
                }
            },
            "defensive_progression": {
                "plugins": {
                    "defensive_battle_strategy": {"aggression_level": 0.3},
                    "progression_focused_rewards": {"story_weight": 0.8}
                },
                "agents": {
                    "coordination_strategy": "adaptive",
                    "agent_weights": {"battle": 0.3, "progression": 0.7}
                }
            },
            "balanced_adaptive": {
                "plugins": {
                    "balanced_battle_strategy": {"aggression_level": 0.5},
                    "spiral_exploration": {"coverage_priority": 0.6}
                },
                "agents": {
                    "coordination_strategy": "adaptive",
                    "agent_weights": {"battle": 0.4, "explorer": 0.3, "progression": 0.3}
                }
            }
        }

        experiment_config = ExperimentConfig(
            name=experiment_name,
            experiment_type=ExperimentType.MULTI_FACTOR,
            description=f"Comparing {len(hybrid_variants)} hybrid configurations"
        )

        for variant_name, hybrid_config in hybrid_variants.items():
            variant_config = copy.deepcopy(base_config)
            variant_config.update(hybrid_config)
            experiment_config.add_variant(variant_name, variant_config)

        return experiment_config

    def validate_variant(self, variant: ConfigurationVariant) -> Tuple[bool, str]:
        """
        Validate a configuration variant.

        Args:
            variant: Configuration variant to validate

        Returns:
            (is_valid, error_message)
        """
        return variant.validate()

    def generate_configuration_suggestions(
        self,
        configuration_type: ConfigurationType,
        base_performance: Dict[str, float] = None
    ) -> List[ConfigurationVariant]:
        """
        Generate suggested configuration variants for testing.

        Args:
            configuration_type: Type of configuration to generate suggestions for
            base_performance: Base performance metrics to improve upon

        Returns:
            List of suggested configuration variants
        """
        suggestions = []

        if configuration_type == ConfigurationType.PLUGIN_CONFIG:
            suggestions.extend(self._generate_plugin_suggestions())
        elif configuration_type == ConfigurationType.AGENT_CONFIG:
            suggestions.extend(self._generate_agent_suggestions())

        return suggestions

    def _generate_plugin_suggestions(self) -> List[ConfigurationVariant]:
        """Generate plugin configuration suggestions"""
        suggestions = []

        # Battle strategy suggestions
        suggestions.append(ConfigurationVariant(
            name="High Aggression Battle",
            description="High aggression battle strategy for faster progression",
            configuration={
                "plugins": {
                    "aggressive_battle_strategy": {
                        "aggression_level": 0.9,
                        "risk_tolerance": 0.8
                    }
                }
            },
            configuration_type=ConfigurationType.PLUGIN_CONFIG,
            expected_impact="Faster battles but higher risk"
        ))

        suggestions.append(ConfigurationVariant(
            name="Conservative Battle",
            description="Conservative battle strategy for stability",
            configuration={
                "plugins": {
                    "defensive_battle_strategy": {
                        "aggression_level": 0.2,
                        "risk_tolerance": 0.3
                    }
                }
            },
            configuration_type=ConfigurationType.PLUGIN_CONFIG,
            expected_impact="Safer battles but slower progression"
        ))

        return suggestions

    def _generate_agent_suggestions(self) -> List[ConfigurationVariant]:
        """Generate agent configuration suggestions"""
        suggestions = []

        suggestions.append(ConfigurationVariant(
            name="Battle-Heavy Coordination",
            description="Prioritize battle agent for combat scenarios",
            configuration={
                "agents": {
                    "coordination_strategy": "priority_based",
                    "agent_weights": {
                        "battle": 0.8,
                        "explorer": 0.1,
                        "progression": 0.1
                    }
                }
            },
            configuration_type=ConfigurationType.AGENT_CONFIG,
            expected_impact="Better battle performance"
        ))

        return suggestions

    def _get_default_base_config(self) -> Dict[str, Any]:
        """Get default base configuration for experiments"""
        return {
            "max_actions": 1000,
            "llm_interval": 10,
            "enable_web": False,
            "headless": True,
            "save_state_required": True
        }

    def compare_configurations(
        self,
        config_a: Dict[str, Any],
        config_b: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compare two configurations and identify differences.

        Args:
            config_a: First configuration
            config_b: Second configuration

        Returns:
            Dictionary describing the differences
        """
        differences = {
            'added_keys': [],
            'removed_keys': [],
            'modified_values': {},
            'summary': ''
        }

        # Find added and removed keys
        keys_a = set(config_a.keys())
        keys_b = set(config_b.keys())

        differences['added_keys'] = list(keys_b - keys_a)
        differences['removed_keys'] = list(keys_a - keys_b)

        # Find modified values
        common_keys = keys_a & keys_b
        for key in common_keys:
            if config_a[key] != config_b[key]:
                differences['modified_values'][key] = {
                    'from': config_a[key],
                    'to': config_b[key]
                }

        # Generate summary
        changes = len(differences['added_keys']) + len(differences['removed_keys']) + len(differences['modified_values'])
        differences['summary'] = f"{changes} total differences found"

        return differences