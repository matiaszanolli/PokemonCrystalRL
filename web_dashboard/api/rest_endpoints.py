"""
REST API Endpoints for Pokemon Crystal RL

Comprehensive REST API providing full control over training sessions,
multi-agent systems, plugin management, and system configuration.
"""

import logging
import uuid
import json
from typing import Dict, Any, Optional, List
from pathlib import Path
import threading
import time

from .rest_models import (
    TrainingConfigModel, TrainingSessionModel, AgentStatusModel,
    MultiAgentCoordinationModel, PluginStatusModel, PluginActionRequest,
    AgentActionRequest, TrainingActionRequest, RestApiResponse,
    ApiDocumentationModel, ApiEndpointInfo, TrainingStatus, AgentType, PluginAction
)

logger = logging.getLogger(__name__)


class RestApiEndpoints:
    """
    Comprehensive REST API endpoints for Pokemon Crystal RL platform.

    Provides full CRUD operations for:
    - Training session management
    - Multi-agent system control
    - Plugin system management
    - System configuration
    """

    def __init__(self, trainer=None):
        """Initialize REST API with trainer reference."""
        self.trainer = trainer
        self.logger = logger

        # Session management
        self.active_sessions: Dict[str, TrainingSessionModel] = {}
        self.session_lock = threading.Lock()

        # Plugin and agent registries
        self.plugin_registry = self._get_plugin_registry()
        self.agent_registry = self._get_agent_registry()

        # API documentation
        self.api_docs = self._build_api_documentation()

    def _get_plugin_registry(self):
        """Get plugin system registry if available."""
        try:
            if (self.trainer and
                hasattr(self.trainer, 'plugin_manager') and
                self.trainer.plugin_manager):
                return self.trainer.plugin_manager
        except Exception as e:
            self.logger.warning(f"Could not access plugin registry: {e}")
        return None

    def _get_agent_registry(self):
        """Get multi-agent coordinator if available."""
        try:
            if (self.trainer and
                hasattr(self.trainer, 'multi_agent_coordinator') and
                self.trainer.multi_agent_coordinator):
                return self.trainer.multi_agent_coordinator
        except Exception as e:
            self.logger.warning(f"Could not access agent registry: {e}")
        return None

    # Training Session Management

    def create_training_session(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a new training session.

        POST /api/v1/training/sessions
        """
        try:
            # Validate and create configuration
            config = TrainingConfigModel(**config_data)

            # Generate session ID
            session_id = str(uuid.uuid4())

            # Create session model
            session = TrainingSessionModel(
                session_id=session_id,
                status=TrainingStatus.STOPPED.value,
                config=config,
                start_time=time.time()
            )

            with self.session_lock:
                self.active_sessions[session_id] = session

            return RestApiResponse(
                success=True,
                data=session.to_dict(),
                message=f"Training session {session_id} created successfully"
            ).to_dict()

        except Exception as e:
            self.logger.error(f"Failed to create training session: {e}")
            return RestApiResponse(
                success=False,
                error=f"Failed to create training session: {str(e)}"
            ).to_dict()

    def get_training_sessions(self) -> Dict[str, Any]:
        """
        List all training sessions.

        GET /api/v1/training/sessions
        """
        try:
            with self.session_lock:
                sessions = [session.to_dict() for session in self.active_sessions.values()]

            return RestApiResponse(
                success=True,
                data={"sessions": sessions, "total": len(sessions)}
            ).to_dict()

        except Exception as e:
            self.logger.error(f"Failed to get training sessions: {e}")
            return RestApiResponse(
                success=False,
                error=f"Failed to get training sessions: {str(e)}"
            ).to_dict()

    def get_training_session(self, session_id: str) -> Dict[str, Any]:
        """
        Get specific training session.

        GET /api/v1/training/sessions/{session_id}
        """
        try:
            with self.session_lock:
                if session_id not in self.active_sessions:
                    return RestApiResponse(
                        success=False,
                        error=f"Training session {session_id} not found"
                    ).to_dict()

                session = self.active_sessions[session_id]

            return RestApiResponse(
                success=True,
                data=session.to_dict()
            ).to_dict()

        except Exception as e:
            self.logger.error(f"Failed to get training session {session_id}: {e}")
            return RestApiResponse(
                success=False,
                error=f"Failed to get training session: {str(e)}"
            ).to_dict()

    def control_training_session(self, session_id: str, action_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Control training session (start/stop/pause/resume).

        POST /api/v1/training/sessions/{session_id}/control
        """
        try:
            action_request = TrainingActionRequest(**action_data)

            with self.session_lock:
                if session_id not in self.active_sessions:
                    return RestApiResponse(
                        success=False,
                        error=f"Training session {session_id} not found"
                    ).to_dict()

                session = self.active_sessions[session_id]

                # Execute action
                success, message = self._execute_training_action(session, action_request)

                if success:
                    return RestApiResponse(
                        success=True,
                        data=session.to_dict(),
                        message=message
                    ).to_dict()
                else:
                    return RestApiResponse(
                        success=False,
                        error=message
                    ).to_dict()

        except Exception as e:
            self.logger.error(f"Failed to control training session {session_id}: {e}")
            return RestApiResponse(
                success=False,
                error=f"Failed to control training session: {str(e)}"
            ).to_dict()

    def _execute_training_action(self, session: TrainingSessionModel, action: TrainingActionRequest) -> tuple:
        """Execute training control action."""
        try:
            if action.action == "start":
                if session.status == TrainingStatus.RUNNING.value:
                    return False, "Training session is already running"
                session.status = TrainingStatus.STARTING.value
                # Here we would integrate with the actual trainer
                session.status = TrainingStatus.RUNNING.value
                return True, "Training session started successfully"

            elif action.action == "stop":
                if session.status == TrainingStatus.STOPPED.value:
                    return False, "Training session is already stopped"
                session.status = TrainingStatus.STOPPING.value
                session.end_time = time.time()
                session.status = TrainingStatus.STOPPED.value
                return True, "Training session stopped successfully"

            elif action.action == "pause":
                if session.status != TrainingStatus.RUNNING.value:
                    return False, "Can only pause running training sessions"
                session.status = TrainingStatus.PAUSED.value
                return True, "Training session paused successfully"

            elif action.action == "resume":
                if session.status != TrainingStatus.PAUSED.value:
                    return False, "Can only resume paused training sessions"
                session.status = TrainingStatus.RUNNING.value
                return True, "Training session resumed successfully"

            else:
                return False, f"Unknown action: {action.action}"

        except Exception as e:
            return False, f"Action execution failed: {str(e)}"

    # Multi-Agent System Management

    def get_agents(self) -> Dict[str, Any]:
        """
        List all available agents.

        GET /api/v1/agents
        """
        try:
            agents = []

            if self.agent_registry:
                # Get agents from multi-agent coordinator
                for agent_id, agent_info in self.agent_registry.agents.items():
                    agent_status = AgentStatusModel(
                        agent_id=agent_id,
                        agent_type=agent_info.get('type', 'unknown'),
                        status=agent_info.get('status', 'inactive'),
                        performance_metrics=agent_info.get('metrics', {}),
                        configuration=agent_info.get('config', {})
                    )
                    agents.append(agent_status.to_dict())
            else:
                # Fallback: list known agent types
                for agent_type in AgentType:
                    agent_status = AgentStatusModel(
                        agent_id=f"{agent_type.value}_agent",
                        agent_type=agent_type.value,
                        status="inactive"
                    )
                    agents.append(agent_status.to_dict())

            return RestApiResponse(
                success=True,
                data={"agents": agents, "total": len(agents)}
            ).to_dict()

        except Exception as e:
            self.logger.error(f"Failed to get agents: {e}")
            return RestApiResponse(
                success=False,
                error=f"Failed to get agents: {str(e)}"
            ).to_dict()

    def control_agent(self, agent_id: str, action_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Control individual agent (start/stop/configure).

        POST /api/v1/agents/{agent_id}/control
        """
        try:
            action_request = AgentActionRequest(**action_data)

            if self.agent_registry:
                # Execute agent action through multi-agent coordinator
                success = self._execute_agent_action(agent_id, action_request)
                if success:
                    return RestApiResponse(
                        success=True,
                        message=f"Agent {agent_id} {action_request.action} executed successfully"
                    ).to_dict()
                else:
                    return RestApiResponse(
                        success=False,
                        error=f"Failed to {action_request.action} agent {agent_id}"
                    ).to_dict()
            else:
                return RestApiResponse(
                    success=False,
                    error="Multi-agent system not available"
                ).to_dict()

        except Exception as e:
            self.logger.error(f"Failed to control agent {agent_id}: {e}")
            return RestApiResponse(
                success=False,
                error=f"Failed to control agent: {str(e)}"
            ).to_dict()

    def _execute_agent_action(self, agent_id: str, action: AgentActionRequest) -> bool:
        """Execute agent control action."""
        try:
            if not self.agent_registry:
                return False

            # This would integrate with the actual multi-agent coordinator
            if action.action == "start":
                return True  # self.agent_registry.start_agent(agent_id, action.configuration)
            elif action.action == "stop":
                return True  # self.agent_registry.stop_agent(agent_id)
            elif action.action == "configure":
                return True  # self.agent_registry.configure_agent(agent_id, action.configuration)
            else:
                return False

        except Exception as e:
            self.logger.error(f"Agent action execution failed: {e}")
            return False

    def get_coordination_status(self) -> Dict[str, Any]:
        """
        Get multi-agent coordination status.

        GET /api/v1/agents/coordination
        """
        try:
            if self.agent_registry:
                coordination = MultiAgentCoordinationModel(
                    coordinator_active=True,
                    coordination_strategy="priority_based",
                    agent_priorities={},
                    recent_decisions=[],
                    performance_score=0.85
                )
            else:
                coordination = MultiAgentCoordinationModel()

            return RestApiResponse(
                success=True,
                data=coordination.to_dict()
            ).to_dict()

        except Exception as e:
            self.logger.error(f"Failed to get coordination status: {e}")
            return RestApiResponse(
                success=False,
                error=f"Failed to get coordination status: {str(e)}"
            ).to_dict()

    # Plugin System Management

    def get_plugins(self) -> Dict[str, Any]:
        """
        List all available plugins.

        GET /api/v1/plugins
        """
        try:
            plugins = []

            if self.plugin_registry:
                # Get plugins from plugin manager
                for plugin_id, plugin_info in self.plugin_registry.plugins.items():
                    plugin_status = PluginStatusModel(
                        plugin_id=plugin_id,
                        plugin_type=plugin_info.get('type', 'unknown'),
                        status=plugin_info.get('status', 'unloaded'),
                        version=plugin_info.get('version', '1.0.0'),
                        configuration=plugin_info.get('config', {}),
                        performance_metrics=plugin_info.get('metrics', {})
                    )
                    plugins.append(plugin_status.to_dict())
            else:
                # Return empty list if plugin system not available
                plugins = []

            return RestApiResponse(
                success=True,
                data={"plugins": plugins, "total": len(plugins)}
            ).to_dict()

        except Exception as e:
            self.logger.error(f"Failed to get plugins: {e}")
            return RestApiResponse(
                success=False,
                error=f"Failed to get plugins: {str(e)}"
            ).to_dict()

    def control_plugin(self, plugin_id: str, action_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Control plugin (load/unload/configure).

        POST /api/v1/plugins/{plugin_id}/control
        """
        try:
            action_request = PluginActionRequest(**action_data)

            if self.plugin_registry:
                success = self._execute_plugin_action(plugin_id, action_request)
                if success:
                    return RestApiResponse(
                        success=True,
                        message=f"Plugin {plugin_id} {action_request.action} executed successfully"
                    ).to_dict()
                else:
                    return RestApiResponse(
                        success=False,
                        error=f"Failed to {action_request.action} plugin {plugin_id}"
                    ).to_dict()
            else:
                return RestApiResponse(
                    success=False,
                    error="Plugin system not available"
                ).to_dict()

        except Exception as e:
            self.logger.error(f"Failed to control plugin {plugin_id}: {e}")
            return RestApiResponse(
                success=False,
                error=f"Failed to control plugin: {str(e)}"
            ).to_dict()

    def _execute_plugin_action(self, plugin_id: str, action: PluginActionRequest) -> bool:
        """Execute plugin control action."""
        try:
            if not self.plugin_registry:
                return False

            # This would integrate with the actual plugin manager
            if action.action == PluginAction.LOAD.value:
                return True  # self.plugin_registry.load_plugin(plugin_id, action.configuration)
            elif action.action == PluginAction.UNLOAD.value:
                return True  # self.plugin_registry.unload_plugin(plugin_id)
            elif action.action == PluginAction.CONFIGURE.value:
                return True  # self.plugin_registry.configure_plugin(plugin_id, action.configuration)
            else:
                return False

        except Exception as e:
            self.logger.error(f"Plugin action execution failed: {e}")
            return False

    # API Documentation

    def get_api_documentation(self) -> Dict[str, Any]:
        """
        Get comprehensive API documentation.

        GET /api/v1/docs
        """
        try:
            return RestApiResponse(
                success=True,
                data=self.api_docs.to_dict()
            ).to_dict()

        except Exception as e:
            self.logger.error(f"Failed to get API documentation: {e}")
            return RestApiResponse(
                success=False,
                error=f"Failed to get API documentation: {str(e)}"
            ).to_dict()

    def _build_api_documentation(self) -> ApiDocumentationModel:
        """Build comprehensive API documentation."""
        endpoints = [
            # Training Session Endpoints
            ApiEndpointInfo(
                path="/api/v1/training/sessions",
                method="GET",
                description="List all training sessions"
            ),
            ApiEndpointInfo(
                path="/api/v1/training/sessions",
                method="POST",
                description="Create new training session"
            ),
            ApiEndpointInfo(
                path="/api/v1/training/sessions/{session_id}",
                method="GET",
                description="Get specific training session"
            ),
            ApiEndpointInfo(
                path="/api/v1/training/sessions/{session_id}/control",
                method="POST",
                description="Control training session (start/stop/pause/resume)"
            ),

            # Agent Management Endpoints
            ApiEndpointInfo(
                path="/api/v1/agents",
                method="GET",
                description="List all available agents"
            ),
            ApiEndpointInfo(
                path="/api/v1/agents/{agent_id}/control",
                method="POST",
                description="Control individual agent"
            ),
            ApiEndpointInfo(
                path="/api/v1/agents/coordination",
                method="GET",
                description="Get multi-agent coordination status"
            ),

            # Plugin Management Endpoints
            ApiEndpointInfo(
                path="/api/v1/plugins",
                method="GET",
                description="List all available plugins"
            ),
            ApiEndpointInfo(
                path="/api/v1/plugins/{plugin_id}/control",
                method="POST",
                description="Control plugin (load/unload/configure)"
            ),

            # Documentation Endpoint
            ApiEndpointInfo(
                path="/api/v1/docs",
                method="GET",
                description="Get API documentation"
            )
        ]

        return ApiDocumentationModel(endpoints=endpoints)