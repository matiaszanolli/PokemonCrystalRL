"""
Plugin System - Modular architecture for extensible Pokemon Crystal RL

This system provides a comprehensive plugin architecture that allows:
- Hot-swappable components during training
- Community plugin development
- Modular battle strategies, exploration patterns, and reward systems
- Dynamic plugin discovery and lifecycle management
"""

import logging
import importlib
import inspect
import threading
from typing import Dict, Any, List, Type, Optional, Callable, Set
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import json
import time

from ..core.event_system import EventType, Event, EventSubscriber, get_event_bus


class PluginType(Enum):
    """Different types of plugins in the system"""
    BATTLE_STRATEGY = "battle_strategy"
    EXPLORATION_PATTERN = "exploration_pattern"
    REWARD_CALCULATOR = "reward_calculator"
    SCREEN_ANALYZER = "screen_analyzer"
    MEMORY_INTERPRETER = "memory_interpreter"
    DECISION_ENHANCER = "decision_enhancer"
    TRAINING_MONITOR = "training_monitor"
    STATE_PROCESSOR = "state_processor"


class PluginStatus(Enum):
    """Plugin lifecycle states"""
    UNLOADED = "unloaded"
    LOADING = "loading"
    LOADED = "loaded"
    ACTIVE = "active"
    PAUSED = "paused"
    ERROR = "error"
    UNLOADING = "unloading"


@dataclass
class PluginMetadata:
    """Metadata for a plugin"""
    name: str
    version: str
    description: str
    author: str
    plugin_type: PluginType
    dependencies: List[str] = field(default_factory=list)
    compatibility: Dict[str, str] = field(default_factory=dict)  # version requirements
    config_schema: Dict[str, Any] = field(default_factory=dict)
    hot_swappable: bool = True
    priority: int = 5  # 1-10, higher = more important
    tags: List[str] = field(default_factory=list)


@dataclass
class PluginInfo:
    """Runtime information about a plugin"""
    metadata: PluginMetadata
    module_path: str
    class_name: str
    status: PluginStatus
    instance: Optional['BasePlugin'] = None
    load_time: Optional[float] = None
    error_message: Optional[str] = None
    config: Dict[str, Any] = field(default_factory=dict)
    performance_stats: Dict[str, Any] = field(default_factory=dict)


class BasePlugin(ABC):
    """Abstract base class for all plugins"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(f"Plugin.{self.__class__.__name__}")
        self.event_bus = get_event_bus()
        self.is_active = False
        self.performance_stats = {
            'calls': 0,
            'total_time': 0.0,
            'errors': 0,
            'last_call': None
        }

    @abstractmethod
    def get_metadata(self) -> PluginMetadata:
        """Return plugin metadata"""
        pass

    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the plugin. Return True if successful."""
        pass

    @abstractmethod
    def shutdown(self) -> bool:
        """Shutdown the plugin. Return True if successful."""
        pass

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate plugin configuration. Override if needed."""
        return True

    def update_config(self, config: Dict[str, Any]) -> bool:
        """Update plugin configuration at runtime. Override if needed."""
        if self.validate_config(config):
            self.config.update(config)
            return True
        return False

    def get_status(self) -> Dict[str, Any]:
        """Get plugin status information"""
        return {
            'active': self.is_active,
            'config': self.config,
            'performance': self.performance_stats
        }

    def _track_performance(self, operation_name: str, start_time: float) -> None:
        """Track performance metrics"""
        duration = time.time() - start_time
        self.performance_stats['calls'] += 1
        self.performance_stats['total_time'] += duration
        self.performance_stats['last_call'] = {
            'operation': operation_name,
            'duration': duration,
            'timestamp': time.time()
        }


class BattleStrategyPlugin(BasePlugin):
    """Base class for battle strategy plugins"""

    @abstractmethod
    def analyze_battle_situation(self, game_state: Dict[str, Any], battle_context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze current battle situation and provide strategic insights"""
        pass

    @abstractmethod
    def recommend_move(self, game_state: Dict[str, Any], available_moves: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Recommend best move given current situation"""
        pass

    @abstractmethod
    def assess_switch_opportunity(self, game_state: Dict[str, Any], party: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess whether to switch Pokemon and which one"""
        pass


class ExplorationPatternPlugin(BasePlugin):
    """Base class for exploration pattern plugins"""

    @abstractmethod
    def get_exploration_direction(self, game_state: Dict[str, Any], exploration_context: Dict[str, Any]) -> Dict[str, Any]:
        """Get next exploration direction based on pattern"""
        pass

    @abstractmethod
    def update_exploration_state(self, game_state: Dict[str, Any], last_action: int) -> None:
        """Update internal exploration state based on last action"""
        pass

    @abstractmethod
    def reset_exploration_pattern(self) -> None:
        """Reset exploration pattern to initial state"""
        pass


class RewardCalculatorPlugin(BasePlugin):
    """Base class for reward calculation plugins"""

    @abstractmethod
    def calculate_reward(self, old_state: Dict[str, Any], new_state: Dict[str, Any], action: int) -> float:
        """Calculate reward for state transition"""
        pass

    @abstractmethod
    def get_reward_breakdown(self, old_state: Dict[str, Any], new_state: Dict[str, Any]) -> Dict[str, float]:
        """Get detailed breakdown of reward components"""
        pass


class PluginRegistry:
    """Central registry for managing plugins"""

    def __init__(self):
        self.logger = logging.getLogger("PluginRegistry")
        self.plugins: Dict[str, PluginInfo] = {}
        self.active_plugins: Dict[PluginType, List[str]] = {ptype: [] for ptype in PluginType}
        self.plugin_hooks: Dict[str, List[Callable]] = {}
        self.lock = threading.RLock()
        self.event_bus = get_event_bus()

        # Plugin discovery paths
        self.plugin_paths = [
            Path("plugins"),
            Path("plugins/community"),
            Path("plugins/official"),
            Path(__file__).parent.parent / "plugins"
        ]

        self.logger.info("PluginRegistry initialized")

    def discover_plugins(self) -> List[str]:
        """Discover all available plugins"""
        discovered = []

        for plugin_path in self.plugin_paths:
            if not plugin_path.exists():
                continue

            for plugin_file in plugin_path.glob("**/*.py"):
                if plugin_file.name.startswith("__"):
                    continue

                try:
                    plugin_id = self._analyze_plugin_file(plugin_file)
                    if plugin_id:
                        discovered.append(plugin_id)
                        self.logger.debug(f"Discovered plugin: {plugin_id}")
                except Exception as e:
                    self.logger.warning(f"Error analyzing plugin {plugin_file}: {e}")

        self.logger.info(f"Discovered {len(discovered)} plugins")
        return discovered

    def _analyze_plugin_file(self, plugin_file: Path) -> Optional[str]:
        """Analyze a plugin file to extract metadata"""
        try:
            # Import the module
            spec = importlib.util.spec_from_file_location(
                plugin_file.stem, plugin_file
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Find plugin classes
            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and
                    issubclass(obj, BasePlugin) and
                    obj != BasePlugin and
                    not obj.__name__.startswith('Base')):

                    # Try to get metadata
                    try:
                        instance = obj()
                        metadata = instance.get_metadata()

                        plugin_id = f"{metadata.plugin_type.value}.{metadata.name}"

                        plugin_info = PluginInfo(
                            metadata=metadata,
                            module_path=str(plugin_file),
                            class_name=name,
                            status=PluginStatus.UNLOADED
                        )

                        with self.lock:
                            self.plugins[plugin_id] = plugin_info

                        return plugin_id

                    except Exception as e:
                        self.logger.warning(f"Error getting metadata from {name}: {e}")

        except Exception as e:
            self.logger.warning(f"Error importing plugin file {plugin_file}: {e}")

        return None

    def load_plugin(self, plugin_id: str, config: Dict[str, Any] = None) -> bool:
        """Load a specific plugin"""
        with self.lock:
            if plugin_id not in self.plugins:
                self.logger.error(f"Plugin {plugin_id} not found")
                return False

            plugin_info = self.plugins[plugin_id]

            if plugin_info.status != PluginStatus.UNLOADED:
                self.logger.warning(f"Plugin {plugin_id} already loaded")
                return True

            try:
                plugin_info.status = PluginStatus.LOADING

                # Import the module
                spec = importlib.util.spec_from_file_location(
                    "dynamic_plugin", plugin_info.module_path
                )
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                # Get the plugin class
                plugin_class = getattr(module, plugin_info.class_name)

                # Create instance with config
                plugin_config = config or {}
                plugin_info.config = plugin_config

                instance = plugin_class(plugin_config)

                # Validate configuration
                if not instance.validate_config(plugin_config):
                    raise ValueError("Invalid plugin configuration")

                # Initialize the plugin
                if not instance.initialize():
                    raise RuntimeError("Plugin initialization failed")

                plugin_info.instance = instance
                plugin_info.status = PluginStatus.LOADED
                plugin_info.load_time = time.time()

                # Publish plugin loaded event
                self._publish_plugin_event("plugin_loaded", plugin_id, plugin_info.metadata)

                self.logger.info(f"Successfully loaded plugin: {plugin_id}")
                return True

            except Exception as e:
                plugin_info.status = PluginStatus.ERROR
                plugin_info.error_message = str(e)
                self.logger.error(f"Failed to load plugin {plugin_id}: {e}")
                return False

    def unload_plugin(self, plugin_id: str) -> bool:
        """Unload a specific plugin"""
        with self.lock:
            if plugin_id not in self.plugins:
                self.logger.error(f"Plugin {plugin_id} not found")
                return False

            plugin_info = self.plugins[plugin_id]

            if plugin_info.status == PluginStatus.UNLOADED:
                return True

            try:
                plugin_info.status = PluginStatus.UNLOADING

                # Deactivate if active
                if plugin_info.status == PluginStatus.ACTIVE:
                    self.deactivate_plugin(plugin_id)

                # Shutdown the plugin
                if plugin_info.instance:
                    plugin_info.instance.shutdown()
                    plugin_info.instance = None

                plugin_info.status = PluginStatus.UNLOADED
                plugin_info.error_message = None

                # Publish plugin unloaded event
                self._publish_plugin_event("plugin_unloaded", plugin_id, plugin_info.metadata)

                self.logger.info(f"Successfully unloaded plugin: {plugin_id}")
                return True

            except Exception as e:
                plugin_info.status = PluginStatus.ERROR
                plugin_info.error_message = str(e)
                self.logger.error(f"Failed to unload plugin {plugin_id}: {e}")
                return False

    def activate_plugin(self, plugin_id: str) -> bool:
        """Activate a loaded plugin"""
        with self.lock:
            if plugin_id not in self.plugins:
                return False

            plugin_info = self.plugins[plugin_id]

            if plugin_info.status != PluginStatus.LOADED:
                if plugin_info.status == PluginStatus.UNLOADED:
                    if not self.load_plugin(plugin_id):
                        return False
                else:
                    return False

            try:
                plugin_info.status = PluginStatus.ACTIVE
                plugin_info.instance.is_active = True

                # Add to active plugins list
                plugin_type = plugin_info.metadata.plugin_type
                if plugin_id not in self.active_plugins[plugin_type]:
                    self.active_plugins[plugin_type].append(plugin_id)
                    # Sort by priority
                    self.active_plugins[plugin_type].sort(
                        key=lambda pid: self.plugins[pid].metadata.priority,
                        reverse=True
                    )

                # Publish plugin activated event
                self._publish_plugin_event("plugin_activated", plugin_id, plugin_info.metadata)

                self.logger.info(f"Activated plugin: {plugin_id}")
                return True

            except Exception as e:
                self.logger.error(f"Failed to activate plugin {plugin_id}: {e}")
                return False

    def deactivate_plugin(self, plugin_id: str) -> bool:
        """Deactivate an active plugin"""
        with self.lock:
            if plugin_id not in self.plugins:
                return False

            plugin_info = self.plugins[plugin_id]

            if plugin_info.status != PluginStatus.ACTIVE:
                return True

            try:
                plugin_info.status = PluginStatus.LOADED
                if plugin_info.instance:
                    plugin_info.instance.is_active = False

                # Remove from active plugins list
                plugin_type = plugin_info.metadata.plugin_type
                if plugin_id in self.active_plugins[plugin_type]:
                    self.active_plugins[plugin_type].remove(plugin_id)

                # Publish plugin deactivated event
                self._publish_plugin_event("plugin_deactivated", plugin_id, plugin_info.metadata)

                self.logger.info(f"Deactivated plugin: {plugin_id}")
                return True

            except Exception as e:
                self.logger.error(f"Failed to deactivate plugin {plugin_id}: {e}")
                return False

    def get_active_plugins(self, plugin_type: PluginType) -> List[BasePlugin]:
        """Get all active plugins of a specific type"""
        with self.lock:
            active_instances = []
            for plugin_id in self.active_plugins[plugin_type]:
                plugin_info = self.plugins.get(plugin_id)
                if plugin_info and plugin_info.instance and plugin_info.status == PluginStatus.ACTIVE:
                    active_instances.append(plugin_info.instance)
            return active_instances

    def get_plugin_info(self, plugin_id: str) -> Optional[PluginInfo]:
        """Get information about a specific plugin"""
        return self.plugins.get(plugin_id)

    def list_plugins(self, plugin_type: Optional[PluginType] = None, status: Optional[PluginStatus] = None) -> List[str]:
        """List plugins matching criteria"""
        with self.lock:
            result = []
            for plugin_id, plugin_info in self.plugins.items():
                if plugin_type and plugin_info.metadata.plugin_type != plugin_type:
                    continue
                if status and plugin_info.status != status:
                    continue
                result.append(plugin_id)
            return result

    def reload_plugin(self, plugin_id: str) -> bool:
        """Hot-reload a plugin"""
        with self.lock:
            if plugin_id not in self.plugins:
                return False

            plugin_info = self.plugins[plugin_id]

            if not plugin_info.metadata.hot_swappable:
                self.logger.warning(f"Plugin {plugin_id} is not hot-swappable")
                return False

            # Save current state
            was_active = plugin_info.status == PluginStatus.ACTIVE
            current_config = plugin_info.config.copy()

            # Unload and reload
            if not self.unload_plugin(plugin_id):
                return False

            if not self.load_plugin(plugin_id, current_config):
                return False

            # Reactivate if it was active
            if was_active:
                return self.activate_plugin(plugin_id)

            return True

    def _publish_plugin_event(self, event_type: str, plugin_id: str, metadata: PluginMetadata) -> None:
        """Publish plugin lifecycle events"""
        event = Event(
            event_type=EventType.COMPONENT_INITIALIZED if "loaded" in event_type or "activated" in event_type else EventType.COMPONENT_SHUTDOWN,
            timestamp=time.time(),
            source="plugin_registry",
            data={
                'event_type': event_type,
                'plugin_id': plugin_id,
                'plugin_name': metadata.name,
                'plugin_type': metadata.plugin_type.value,
                'plugin_version': metadata.version
            },
            priority=6
        )

        self.event_bus.publish(event)

    def get_registry_stats(self) -> Dict[str, Any]:
        """Get comprehensive plugin registry statistics"""
        with self.lock:
            stats = {
                'total_plugins': len(self.plugins),
                'status_distribution': {},
                'type_distribution': {},
                'active_by_type': {},
                'performance_summary': {}
            }

            # Status distribution
            for plugin_info in self.plugins.values():
                status = plugin_info.status.value
                stats['status_distribution'][status] = stats['status_distribution'].get(status, 0) + 1

            # Type distribution
            for plugin_info in self.plugins.values():
                ptype = plugin_info.metadata.plugin_type.value
                stats['type_distribution'][ptype] = stats['type_distribution'].get(ptype, 0) + 1

            # Active plugins by type
            for ptype, plugin_ids in self.active_plugins.items():
                stats['active_by_type'][ptype.value] = len(plugin_ids)

            # Performance summary
            total_calls = 0
            total_time = 0.0
            total_errors = 0

            for plugin_info in self.plugins.values():
                if plugin_info.instance:
                    perf = plugin_info.instance.performance_stats
                    total_calls += perf['calls']
                    total_time += perf['total_time']
                    total_errors += perf['errors']

            stats['performance_summary'] = {
                'total_calls': total_calls,
                'total_time': total_time,
                'total_errors': total_errors,
                'average_call_time': total_time / max(total_calls, 1)
            }

            return stats


# Global plugin registry instance
_plugin_registry: Optional[PluginRegistry] = None

def get_plugin_registry() -> PluginRegistry:
    """Get the global plugin registry instance"""
    global _plugin_registry
    if _plugin_registry is None:
        _plugin_registry = PluginRegistry()
    return _plugin_registry

def initialize_plugin_system() -> PluginRegistry:
    """Initialize the global plugin system"""
    global _plugin_registry
    _plugin_registry = PluginRegistry()

    # Discover and load default plugins
    discovered = _plugin_registry.discover_plugins()

    return _plugin_registry