"""
Unified API endpoint definitions for Pokemon Crystal RL Web Dashboard.

This module defines all available API endpoints with their specifications,
consolidating functionality from multiple previous implementations.
"""

from typing import Dict, Any, Optional, List
import json
import logging
from .models import (
    GameStateModel, TrainingStatsModel, MemoryDebugModel,
    LLMDecisionModel, SystemStatusModel, UnifiedDashboardModel,
    ApiResponseModel
)

logger = logging.getLogger(__name__)


class UnifiedApiEndpoints:
    """
    Unified API endpoints that consolidate all web dashboard functionality.

    This class provides a clean interface to access all training data,
    game state, memory debugging, and system status information.
    """

    def __init__(self, trainer=None):
        """Initialize API endpoints with trainer reference."""
        self.trainer = trainer
        self.logger = logger

    def get_dashboard_data(self) -> Dict[str, Any]:
        """
        Get complete dashboard data in one call.

        Returns:
            Unified dashboard data including all components

        Endpoint: GET /api/dashboard
        """
        try:
            game_state = self._get_game_state()
            training_stats = self._get_training_stats()
            memory_debug = self._get_memory_debug()
            llm_decisions = self._get_recent_llm_decisions()
            system_status = self._get_system_status()

            dashboard_data = UnifiedDashboardModel(
                game_state=game_state,
                training_stats=training_stats,
                memory_debug=memory_debug,
                recent_llm_decisions=llm_decisions,
                system_status=system_status
            )

            return ApiResponseModel(success=True, data=dashboard_data).to_dict()

        except Exception as e:
            self.logger.error(f"Dashboard data error: {e}")
            return ApiResponseModel(
                success=False,
                error=f"Failed to get dashboard data: {str(e)}"
            ).to_dict()

    def get_game_state(self) -> Dict[str, Any]:
        """
        Get current game state information.

        Returns:
            Game state data (map, position, money, badges, etc.)

        Endpoint: GET /api/game_state
        """
        try:
            game_state = self._get_game_state()
            return ApiResponseModel(success=True, data=game_state).to_dict()
        except Exception as e:
            self.logger.error(f"Game state error: {e}")
            return ApiResponseModel(
                success=False,
                error=f"Failed to get game state: {str(e)}"
            ).to_dict()

    def get_training_stats(self) -> Dict[str, Any]:
        """
        Get training statistics and metrics.

        Returns:
            Training statistics (actions, rewards, performance, etc.)

        Endpoint: GET /api/training_stats
        """
        try:
            training_stats = self._get_training_stats()
            return ApiResponseModel(success=True, data=training_stats).to_dict()
        except Exception as e:
            self.logger.error(f"Training stats error: {e}")
            return ApiResponseModel(
                success=False,
                error=f"Failed to get training stats: {str(e)}"
            ).to_dict()

    def get_memory_debug(self) -> Dict[str, Any]:
        """
        Get memory debug information.

        Returns:
            Memory debug data with raw memory addresses and values

        Endpoint: GET /api/memory_debug
        """
        try:
            memory_debug = self._get_memory_debug()
            return ApiResponseModel(success=True, data=memory_debug).to_dict()
        except Exception as e:
            self.logger.error(f"Memory debug error: {e}")
            return ApiResponseModel(
                success=False,
                error=f"Failed to get memory debug data: {str(e)}"
            ).to_dict()

    def get_llm_decisions(self) -> Dict[str, Any]:
        """
        Get recent LLM decisions and reasoning.

        Returns:
            List of recent LLM decisions with reasoning and context

        Endpoint: GET /api/llm_decisions
        """
        try:
            decisions = self._get_recent_llm_decisions()
            return ApiResponseModel(success=True, data=decisions).to_dict()
        except Exception as e:
            self.logger.error(f"LLM decisions error: {e}")
            return ApiResponseModel(
                success=False,
                error=f"Failed to get LLM decisions: {str(e)}"
            ).to_dict()

    def get_system_status(self) -> Dict[str, Any]:
        """
        Get system status and health information.

        Returns:
            System status (training active, connections, errors, etc.)

        Endpoint: GET /api/system_status
        """
        try:
            system_status = self._get_system_status()
            return ApiResponseModel(success=True, data=system_status).to_dict()
        except Exception as e:
            self.logger.error(f"System status error: {e}")
            return ApiResponseModel(
                success=False,
                error=f"Failed to get system status: {str(e)}"
            ).to_dict()

    # Private helper methods

    def _get_game_state(self) -> GameStateModel:
        """Extract game state from trainer."""
        if not self.trainer:
            return GameStateModel()

        try:
            # Get game state from statistics tracker
            tracker = None
            if hasattr(self.trainer, 'stats_tracker') and self.trainer.stats_tracker:
                tracker = self.trainer.stats_tracker
            elif hasattr(self.trainer, 'statistics_tracker') and self.trainer.statistics_tracker:
                tracker = self.trainer.statistics_tracker

            if tracker:
                stats = tracker.get_current_stats()
                # Extract position properly
                pos = stats.get('player_position', {})
                if isinstance(pos, dict):
                    position = {"x": pos.get('x', 0), "y": pos.get('y', 0)}
                else:
                    position = {"x": 0, "y": 0}

                # Try to get map name/ID - check different possible keys
                current_map = (
                    stats.get('current_map') or
                    stats.get('map_id') or
                    stats.get('map') or
                    stats.get('location') or
                    0
                )

                return GameStateModel(
                    current_map=current_map,
                    player_position=position,
                    money=stats.get('money', 0),
                    badges_earned=stats.get('badges', 0),
                    party_count=stats.get('party_count', 0),
                    player_level=stats.get('level', 0),
                    hp_current=stats.get('hp_current', 0),
                    hp_max=stats.get('hp_max', 0),
                    in_battle=stats.get('in_battle', False),
                    facing_direction=stats.get('facing_direction', 0)
                )
        except Exception as e:
            self.logger.warning(f"Could not get game state from statistics: {e}")

        return GameStateModel()

    def _get_training_stats(self) -> TrainingStatsModel:
        """Extract training statistics from trainer."""
        if not self.trainer:
            return TrainingStatsModel()

        try:
            # Try both possible attribute names for compatibility
            tracker = None
            if hasattr(self.trainer, 'stats_tracker') and self.trainer.stats_tracker:
                tracker = self.trainer.stats_tracker
            elif hasattr(self.trainer, 'statistics_tracker') and self.trainer.statistics_tracker:
                tracker = self.trainer.statistics_tracker

            if tracker:
                stats = tracker.get_current_stats()
                return TrainingStatsModel(
                    total_actions=stats.get('total_actions', 0),
                    actions_per_second=stats.get('actions_per_second', 0.0),
                    llm_decisions=stats.get('llm_calls', 0),
                    total_reward=stats.get('total_reward', 0.0),
                    session_duration=stats.get('session_duration', 0.0),
                    success_rate=stats.get('success_rate', 0.0),
                    exploration_rate=stats.get('exploration_rate', 0.0),
                    recent_rewards=stats.get('recent_rewards', [])
                )
        except Exception as e:
            self.logger.warning(f"Could not get training stats: {e}")

        return TrainingStatsModel()

    def _get_memory_debug(self) -> MemoryDebugModel:
        """Extract memory debug information."""
        if not self.trainer:
            return MemoryDebugModel()

        try:
            # Try to get PyBoy instance - check multiple possible attribute paths
            pyboy_instance = None
            memory_reader = None

            # Check different trainer implementations
            if hasattr(self.trainer, 'emulation_manager') and self.trainer.emulation_manager:
                if hasattr(self.trainer.emulation_manager, 'get_instance'):
                    pyboy_instance = self.trainer.emulation_manager.get_instance()
                elif hasattr(self.trainer.emulation_manager, 'pyboy'):
                    pyboy_instance = self.trainer.emulation_manager.pyboy
            elif hasattr(self.trainer, 'pyboy') and self.trainer.pyboy:
                pyboy_instance = self.trainer.pyboy

            # Check for memory reader
            if hasattr(self.trainer, 'memory_reader') and self.trainer.memory_reader:
                memory_reader = self.trainer.memory_reader
            elif hasattr(self.trainer, 'stats_tracker') and self.trainer.stats_tracker:
                # Try to get memory state from stats tracker
                stats = self.trainer.stats_tracker.get_current_stats()
                if stats and len(stats) > 0:
                    return MemoryDebugModel(
                        memory_addresses=stats,
                        memory_read_success=True,
                        pyboy_available=bool(pyboy_instance),
                        cache_info={"source": "statistics_tracker"}
                    )

            if pyboy_instance and memory_reader:
                memory_state = memory_reader.read_game_state()
                debug_info = memory_reader.get_debug_info() if hasattr(memory_reader, 'get_debug_info') else {}

                return MemoryDebugModel(
                    memory_addresses=memory_state,
                    memory_read_success=True,
                    pyboy_available=True,
                    cache_info=debug_info
                )
            elif pyboy_instance:
                # At least we have PyBoy, even if no memory reader
                return MemoryDebugModel(
                    memory_addresses={"pyboy_status": "available"},
                    memory_read_success=True,
                    pyboy_available=True,
                    cache_info={"note": "PyBoy available but no memory reader"}
                )

        except Exception as e:
            self.logger.warning(f"Could not get memory debug data: {e}")

        return MemoryDebugModel(
            memory_read_success=False,
            pyboy_available=False,
            cache_info={"error": "No PyBoy instance or memory reader found"}
        )

    def _get_recent_llm_decisions(self) -> List[LLMDecisionModel]:
        """Extract recent LLM decisions."""
        if not self.trainer:
            return []

        try:
            if hasattr(self.trainer, 'llm_decisions') and self.trainer.llm_decisions:
                decisions = []
                for decision in list(self.trainer.llm_decisions)[-10:]:  # Last 10 decisions
                    decisions.append(LLMDecisionModel(
                        action=decision.get('action', 0),
                        action_name=decision.get('action_name', 'Unknown'),
                        reasoning=decision.get('reasoning', ''),
                        confidence=decision.get('confidence', 0.0),
                        response_time_ms=decision.get('response_time_ms', 0.0),
                        game_state_snapshot=decision.get('game_state', {}),
                        timestamp=decision.get('timestamp', 0.0)
                    ))
                return decisions
        except Exception as e:
            self.logger.warning(f"Could not get LLM decisions: {e}")

        return []

    def _get_system_status(self) -> SystemStatusModel:
        """Get system status information."""
        training_active = False
        websocket_connections = 0

        try:
            if self.trainer:
                # Check multiple ways to determine if training is active
                training_active = (
                    (hasattr(self.trainer, 'training_active') and self.trainer.training_active) or
                    # Infer from statistics tracker activity
                    (hasattr(self.trainer, 'stats_tracker') and self.trainer.stats_tracker and
                     self.trainer.stats_tracker.get_current_stats().get('total_actions', 0) > 0) or
                    (hasattr(self.trainer, 'statistics_tracker') and self.trainer.statistics_tracker and
                     self.trainer.statistics_tracker.get_current_stats().get('total_actions', 0) > 0)
                )

            # Check for web monitor - try multiple attribute names
            if hasattr(self.trainer, 'web_monitor') and self.trainer.web_monitor:
                websocket_connections = getattr(self.trainer.web_monitor, 'active_connections', 0)
            elif hasattr(self.trainer, 'websocket_handler') and self.trainer.websocket_handler:
                websocket_connections = getattr(self.trainer.websocket_handler, 'connection_count', 0)

        except Exception as e:
            self.logger.warning(f"Could not get system status: {e}")

        return SystemStatusModel(
            training_active=training_active,
            web_server_status="active" if training_active else "stopped",
            websocket_connections=websocket_connections
        )