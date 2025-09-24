"""
REST API Data Models for Pokemon Crystal RL

This module defines data models specifically for the REST API,
providing comprehensive control and configuration capabilities.
"""

from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from enum import Enum
import time


class TrainingStatus(Enum):
    """Training session status"""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    PAUSING = "pausing"
    PAUSED = "paused"
    STOPPING = "stopping"
    ERROR = "error"


class AgentType(Enum):
    """Available agent types"""
    BATTLE = "battle"
    EXPLORER = "explorer"
    PROGRESSION = "progression"
    HYBRID = "hybrid"
    LLM = "llm"
    DQN = "dqn"


class PluginAction(Enum):
    """Plugin management actions"""
    LOAD = "load"
    UNLOAD = "unload"
    ACTIVATE = "activate"
    DEACTIVATE = "deactivate"
    CONFIGURE = "configure"
    RELOAD = "reload"


@dataclass
class TrainingConfigModel:
    """Training session configuration"""
    # Basic settings
    rom_path: str
    save_state_path: Optional[str] = None
    max_actions: int = 10000
    headless: bool = True

    # LLM settings
    enable_llm: bool = True
    llm_model: str = "smollm2:1.7b"
    llm_interval: int = 20

    # Agent configuration
    primary_agent: str = "hybrid"  # AgentType
    enabled_agents: List[str] = None  # List of AgentType

    # Plugin configuration
    active_plugins: Dict[str, Dict[str, Any]] = None  # {plugin_id: config}

    # Monitoring
    enable_web: bool = True
    web_port: int = 8080
    enable_logging: bool = True
    log_level: str = "INFO"

    # Performance settings
    target_fps: int = 60
    enable_save_states: bool = True
    save_interval: int = 1000  # actions

    timestamp: float = None

    def __post_init__(self):
        if self.enabled_agents is None:
            self.enabled_agents = ["battle", "explorer", "progression"]
        if self.active_plugins is None:
            self.active_plugins = {}
        if self.timestamp is None:
            self.timestamp = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TrainingSessionModel:
    """Active training session information"""
    session_id: str
    status: str  # TrainingStatus
    config: TrainingConfigModel
    start_time: float
    end_time: Optional[float] = None
    current_action: int = 0
    total_reward: float = 0.0
    active_agents: List[str] = None
    errors: List[str] = None

    def __post_init__(self):
        if self.active_agents is None:
            self.active_agents = []
        if self.errors is None:
            self.errors = []

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['config'] = self.config.to_dict()
        return data


@dataclass
class AgentStatusModel:
    """Individual agent status"""
    agent_id: str
    agent_type: str  # AgentType
    status: str  # "active", "inactive", "error"
    performance_metrics: Dict[str, float] = None
    configuration: Dict[str, Any] = None
    last_action_time: float = None
    error_message: Optional[str] = None

    def __post_init__(self):
        if self.performance_metrics is None:
            self.performance_metrics = {}
        if self.configuration is None:
            self.configuration = {}
        if self.last_action_time is None:
            self.last_action_time = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MultiAgentCoordinationModel:
    """Multi-agent coordination status"""
    coordinator_active: bool = False
    coordination_strategy: str = "priority_based"
    agent_priorities: Dict[str, float] = None  # {agent_id: priority}
    recent_decisions: List[Dict[str, Any]] = None
    conflict_resolutions: int = 0
    performance_score: float = 0.0

    def __post_init__(self):
        if self.agent_priorities is None:
            self.agent_priorities = {}
        if self.recent_decisions is None:
            self.recent_decisions = []

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PluginStatusModel:
    """Plugin status information"""
    plugin_id: str
    plugin_type: str
    status: str  # PluginStatus
    version: str = "1.0.0"
    configuration: Dict[str, Any] = None
    performance_metrics: Dict[str, float] = None
    dependencies: List[str] = None
    error_message: Optional[str] = None
    last_update: float = None

    def __post_init__(self):
        if self.configuration is None:
            self.configuration = {}
        if self.performance_metrics is None:
            self.performance_metrics = {}
        if self.dependencies is None:
            self.dependencies = []
        if self.last_update is None:
            self.last_update = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PluginActionRequest:
    """Plugin management action request"""
    plugin_id: str
    action: str  # PluginAction
    configuration: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AgentActionRequest:
    """Agent management action request"""
    agent_id: str
    action: str  # "start", "stop", "pause", "resume", "configure"
    configuration: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TrainingActionRequest:
    """Training session control request"""
    action: str  # "start", "stop", "pause", "resume", "restart"
    config: Optional[TrainingConfigModel] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        if self.config:
            data['config'] = self.config.to_dict()
        return data


@dataclass
class RestApiResponse:
    """Standard REST API response format"""
    success: bool
    data: Optional[Any] = None
    message: Optional[str] = None
    error: Optional[str] = None
    timestamp: float = None
    request_id: Optional[str] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ApiEndpointInfo:
    """API endpoint documentation"""
    path: str
    method: str
    description: str
    parameters: List[Dict[str, str]] = None
    response_model: str = "RestApiResponse"
    example_request: Optional[Dict[str, Any]] = None
    example_response: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.parameters is None:
            self.parameters = []

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ApiDocumentationModel:
    """Complete API documentation"""
    api_version: str = "1.0.0"
    title: str = "Pokemon Crystal RL REST API"
    description: str = "Comprehensive API for Pokemon Crystal reinforcement learning platform"
    endpoints: List[ApiEndpointInfo] = None
    models: List[str] = None

    def __post_init__(self):
        if self.endpoints is None:
            self.endpoints = []
        if self.models is None:
            self.models = [
                "TrainingConfigModel", "TrainingSessionModel", "AgentStatusModel",
                "MultiAgentCoordinationModel", "PluginStatusModel", "RestApiResponse"
            ]

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['endpoints'] = [ep.to_dict() for ep in self.endpoints]
        return data