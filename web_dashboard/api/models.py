"""
Unified API data models for Pokemon Crystal RL Web Dashboard.

This module defines the data structures for all API responses,
ensuring consistency across the entire web monitoring system.
"""

from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import time


@dataclass
class GameStateModel:
    """Game state information from memory reading."""
    current_map: int = 0
    player_position: Dict[str, int] = None
    money: int = 0
    badges_earned: int = 0
    party_count: int = 0
    player_level: int = 0
    hp_current: int = 0
    hp_max: int = 0
    in_battle: bool = False
    facing_direction: int = 0
    timestamp: float = None

    def __post_init__(self):
        if self.player_position is None:
            self.player_position = {"x": 0, "y": 0}
        if self.timestamp is None:
            self.timestamp = time.time()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class TrainingStatsModel:
    """Training statistics and metrics."""
    total_actions: int = 0
    actions_per_second: float = 0.0
    llm_decisions: int = 0
    total_reward: float = 0.0
    session_duration: float = 0.0
    success_rate: float = 0.0
    exploration_rate: float = 0.0
    recent_rewards: List[float] = None
    timestamp: float = None

    def __post_init__(self):
        if self.recent_rewards is None:
            self.recent_rewards = []
        if self.timestamp is None:
            self.timestamp = time.time()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class MemoryDebugModel:
    """Memory debug information with raw memory values."""
    memory_addresses: Dict[str, Any] = None
    memory_read_success: bool = False
    pyboy_available: bool = False
    cache_info: Dict[str, Any] = None
    timestamp: float = None

    def __post_init__(self):
        if self.memory_addresses is None:
            self.memory_addresses = {}
        if self.cache_info is None:
            self.cache_info = {}
        if self.timestamp is None:
            self.timestamp = time.time()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class LLMDecisionModel:
    """Individual LLM decision record."""
    action: Union[int, str]
    action_name: str
    reasoning: str = ""
    confidence: float = 0.0
    response_time_ms: float = 0.0
    game_state_snapshot: Dict[str, Any] = None
    timestamp: float = None

    def __post_init__(self):
        if self.game_state_snapshot is None:
            self.game_state_snapshot = {}
        if self.timestamp is None:
            self.timestamp = time.time()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class SystemStatusModel:
    """System status and health information."""
    training_active: bool = False
    web_server_status: str = "stopped"
    websocket_connections: int = 0
    last_update: float = None
    errors: List[str] = None
    uptime_seconds: float = 0.0

    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.last_update is None:
            self.last_update = time.time()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class VisualizationDataModel:
    """Data model specifically for training visualizations."""
    # Reward progression data
    reward_history: List[Dict[str, float]] = None  # [{"timestamp": t, "reward": r, "cumulative": c}]

    # Action frequency and success data
    action_performance: Dict[str, Dict[str, Any]] = None  # {"up": {"count": 10, "success_rate": 0.8}}

    # LLM decision patterns
    decision_patterns: List[Dict[str, Any]] = None  # Decision flow data

    # Performance metrics over time
    performance_metrics: List[Dict[str, float]] = None  # Time series data

    # Exploration heatmap data
    exploration_data: Dict[str, Any] = None  # Position frequency data

    timestamp: float = None

    def __post_init__(self):
        if self.reward_history is None:
            self.reward_history = []
        if self.action_performance is None:
            self.action_performance = {}
        if self.decision_patterns is None:
            self.decision_patterns = []
        if self.performance_metrics is None:
            self.performance_metrics = []
        if self.exploration_data is None:
            self.exploration_data = {}
        if self.timestamp is None:
            self.timestamp = time.time()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class UnifiedDashboardModel:
    """Complete dashboard data model combining all components."""
    game_state: GameStateModel
    training_stats: TrainingStatsModel
    memory_debug: MemoryDebugModel
    recent_llm_decisions: List[LLMDecisionModel]
    system_status: SystemStatusModel
    timestamp: float = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "game_state": self.game_state.to_dict(),
            "training_stats": self.training_stats.to_dict(),
            "memory_debug": self.memory_debug.to_dict(),
            "recent_llm_decisions": [d.to_dict() for d in self.recent_llm_decisions],
            "system_status": self.system_status.to_dict(),
            "timestamp": self.timestamp
        }


@dataclass
class ApiResponseModel:
    """Standard API response wrapper."""
    success: bool
    data: Any = None
    error: str = None
    timestamp: float = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        response = {
            "success": self.success,
            "timestamp": self.timestamp
        }

        if self.data is not None:
            if hasattr(self.data, 'to_dict'):
                response["data"] = self.data.to_dict()
            elif isinstance(self.data, list):
                # Handle list of objects with to_dict method
                response["data"] = []
                for item in self.data:
                    if hasattr(item, 'to_dict'):
                        response["data"].append(item.to_dict())
                    else:
                        response["data"].append(item)
            else:
                response["data"] = self.data

        if self.error:
            response["error"] = self.error

        return response