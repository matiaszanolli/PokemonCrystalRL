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
            if hasattr(self.trainer, 'statistics_tracker') and self.trainer.statistics_tracker:
                stats = self.trainer.statistics_tracker.get_current_stats()
                return GameStateModel(
                    current_map=stats.get('current_map', 0),
                    player_position={
                        "x": stats.get('player_position', {}).get('x', 0),
                        "y": stats.get('player_position', {}).get('y', 0)
                    },
                    money=stats.get('money', 0),
                    badges_earned=stats.get('badges', 0),
                    party_count=stats.get('party_count', 0),
                    player_level=stats.get('level', 0)
                )
        except Exception as e:
            self.logger.warning(f"Could not get game state from statistics: {e}")

        return GameStateModel()

    def _get_training_stats(self) -> TrainingStatsModel:
        """Extract training statistics from trainer."""
        if not self.trainer:
            return TrainingStatsModel()

        try:
            if hasattr(self.trainer, 'statistics_tracker') and self.trainer.statistics_tracker:
                stats = self.trainer.statistics_tracker.get_current_stats()
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
            # Try to get PyBoy instance
            pyboy_instance = None
            if hasattr(self.trainer, 'emulation_manager') and self.trainer.emulation_manager:
                pyboy_instance = self.trainer.emulation_manager.get_instance()
            elif hasattr(self.trainer, 'pyboy') and self.trainer.pyboy:
                pyboy_instance = self.trainer.pyboy

            if pyboy_instance and hasattr(self.trainer, 'memory_reader') and self.trainer.memory_reader:
                memory_state = self.trainer.memory_reader.read_game_state()
                debug_info = self.trainer.memory_reader.get_debug_info()

                return MemoryDebugModel(
                    memory_addresses=memory_state,
                    memory_read_success=True,
                    pyboy_available=True,
                    cache_info=debug_info
                )
        except Exception as e:
            self.logger.warning(f"Could not get memory debug data: {e}")

        return MemoryDebugModel(
            memory_read_success=False,
            pyboy_available=False
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
                training_active = hasattr(self.trainer, 'training_active') and self.trainer.training_active

            # Check for web monitor
            if hasattr(self.trainer, 'web_monitor') and self.trainer.web_monitor:
                websocket_connections = getattr(self.trainer.web_monitor, 'active_connections', 0)

        except Exception as e:
            self.logger.warning(f"Could not get system status: {e}")

        return SystemStatusModel(
            training_active=training_active,
            web_server_status="active" if training_active else "stopped",
            websocket_connections=websocket_connections
        )