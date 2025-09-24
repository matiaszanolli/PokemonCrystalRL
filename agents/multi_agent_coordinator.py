"""
MultiAgentCoordinator - Orchestrates specialist agents for optimal decision making

This system coordinates multiple specialist agents (Battle, Explorer, Progression)
to make intelligent decisions about which agent should handle each situation.
"""

import logging
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

from .base_agent import BaseAgent
from .battle_agent import BattleAgent
from .explorer_agent import ExplorerAgent
from .progression_agent import ProgressionAgent


class AgentRole(Enum):
    """Different agent roles for specialist decision making"""
    BATTLE = "battle"
    EXPLORER = "explorer"
    PROGRESSION = "progression"
    HYBRID = "hybrid"  # When multiple agents need to collaborate


@dataclass
class AgentRecommendation:
    """Represents a recommendation from a specialist agent"""
    agent_role: AgentRole
    action: int
    confidence: float
    reasoning: str
    context_match: float  # How well this agent matches the current context
    expected_outcome: str
    priority_score: float


@dataclass
class CoordinationDecision:
    """Final coordinated decision from multiple agents"""
    chosen_agent: AgentRole
    action: int
    confidence: float
    reasoning: str
    agent_consensus: float  # Agreement level between agents
    fallback_agents: List[AgentRole]  # Backup agents if primary fails


class MultiAgentCoordinator(BaseAgent):
    """Coordinates multiple specialist agents for optimal Pokemon Crystal gameplay"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.logger = logging.getLogger("MultiAgentCoordinator")

        # Coordination configuration
        self.coordination_config = config.get('coordination_config', {}) if config else {}
        self.consensus_threshold = self.coordination_config.get('consensus_threshold', 0.7)
        self.conflict_resolution = self.coordination_config.get('conflict_resolution', 'highest_confidence')
        self.agent_weights = self.coordination_config.get('agent_weights', {
            'battle': 1.0,
            'explorer': 1.0,
            'progression': 1.0
        })

        # Initialize specialist agents
        self.battle_agent = BattleAgent(config.get('battle_config', {}))
        self.explorer_agent = ExplorerAgent(config.get('exploration_config', {}))
        self.progression_agent = ProgressionAgent(config.get('progression_config', {}))

        # Coordination state
        self.context_analyzer = ContextAnalyzer()
        self.decision_history = []
        self.agent_performance = {
            AgentRole.BATTLE: {'success_rate': 0.5, 'total_decisions': 0, 'successful_outcomes': 0},
            AgentRole.EXPLORER: {'success_rate': 0.5, 'total_decisions': 0, 'successful_outcomes': 0},
            AgentRole.PROGRESSION: {'success_rate': 0.5, 'total_decisions': 0, 'successful_outcomes': 0}
        }

        # Dynamic agent weighting based on performance
        self.adaptive_weights = True
        self.learning_rate = 0.1

        self.logger.info("MultiAgentCoordinator initialized with 3 specialist agents")

    def get_action(self, game_state: Dict[str, Any], info: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
        """Coordinate specialist agents to get optimal action"""

        # Analyze context to determine agent relevance
        context_analysis = self.context_analyzer.analyze_context(game_state, info)

        # Get recommendations from all relevant agents
        agent_recommendations = self._get_agent_recommendations(game_state, info, context_analysis)

        # Coordinate and make final decision
        coordination_decision = self._coordinate_agents(agent_recommendations, context_analysis)

        # Track decision for learning
        self._track_coordination_decision(coordination_decision, game_state, context_analysis)

        # Update the chosen agent with the action
        self._update_chosen_agent(coordination_decision, game_state, info)

        return coordination_decision.action, {
            'source': 'multi_agent_coordinator',
            'chosen_agent': coordination_decision.chosen_agent.value,
            'confidence': coordination_decision.confidence,
            'reasoning': coordination_decision.reasoning,
            'agent_consensus': coordination_decision.agent_consensus,
            'context_type': context_analysis['primary_context'],
            'recommendations_count': len(agent_recommendations),
            'fallback_agents': [agent.value for agent in coordination_decision.fallback_agents]
        }

    def _get_agent_recommendations(self,
                                 game_state: Dict[str, Any],
                                 info: Dict[str, Any],
                                 context_analysis: Dict[str, Any]) -> List[AgentRecommendation]:
        """Get recommendations from all relevant specialist agents"""
        recommendations = []

        # Get recommendation from each agent
        agents_to_consult = self._determine_relevant_agents(context_analysis)

        for agent_role in agents_to_consult:
            try:
                recommendation = self._get_agent_recommendation(agent_role, game_state, info, context_analysis)
                if recommendation:
                    recommendations.append(recommendation)
            except Exception as e:
                self.logger.warning(f"Agent {agent_role.value} failed to provide recommendation: {e}")

        return recommendations

    def _get_agent_recommendation(self,
                                agent_role: AgentRole,
                                game_state: Dict[str, Any],
                                info: Dict[str, Any],
                                context_analysis: Dict[str, Any]) -> Optional[AgentRecommendation]:
        """Get recommendation from specific agent"""

        agent = self._get_agent_by_role(agent_role)
        if not agent:
            return None

        # Get action from agent
        action, agent_info = agent.get_action(game_state, info)

        # Calculate context match score
        context_match = self._calculate_context_match(agent_role, context_analysis)

        # Calculate priority score (confidence * context_match * performance * weight)
        performance_score = self.agent_performance[agent_role]['success_rate']
        weight = self.agent_weights.get(agent_role.value, 1.0)
        priority_score = (
            agent_info.get('confidence', 0.5) *
            context_match *
            performance_score *
            weight
        )

        return AgentRecommendation(
            agent_role=agent_role,
            action=action,
            confidence=agent_info.get('confidence', 0.5),
            reasoning=agent_info.get('reasoning', f'{agent_role.value} recommendation'),
            context_match=context_match,
            expected_outcome=agent_info.get('expected_outcome', 'unknown'),
            priority_score=priority_score
        )

    def _coordinate_agents(self,
                         recommendations: List[AgentRecommendation],
                         context_analysis: Dict[str, Any]) -> CoordinationDecision:
        """Coordinate agent recommendations into final decision"""

        if not recommendations:
            return self._get_fallback_decision(context_analysis)

        # Sort recommendations by priority score
        sorted_recommendations = sorted(recommendations, key=lambda r: r.priority_score, reverse=True)

        # Get the top recommendation
        primary_recommendation = sorted_recommendations[0]

        # Calculate consensus level
        consensus = self._calculate_agent_consensus(recommendations)

        # Check if we need conflict resolution
        if len(recommendations) > 1 and consensus < self.consensus_threshold:
            resolved_recommendation = self._resolve_agent_conflict(sorted_recommendations, context_analysis)
            if resolved_recommendation:
                primary_recommendation = resolved_recommendation

        # Create fallback agent list
        fallback_agents = [rec.agent_role for rec in sorted_recommendations[1:3]]

        return CoordinationDecision(
            chosen_agent=primary_recommendation.agent_role,
            action=primary_recommendation.action,
            confidence=primary_recommendation.confidence,
            reasoning=f"[{primary_recommendation.agent_role.value.upper()}] {primary_recommendation.reasoning}",
            agent_consensus=consensus,
            fallback_agents=fallback_agents
        )

    def _resolve_agent_conflict(self,
                              sorted_recommendations: List[AgentRecommendation],
                              context_analysis: Dict[str, Any]) -> Optional[AgentRecommendation]:
        """Resolve conflicts between agent recommendations"""

        if self.conflict_resolution == 'highest_confidence':
            return max(sorted_recommendations, key=lambda r: r.confidence)

        elif self.conflict_resolution == 'context_match':
            return max(sorted_recommendations, key=lambda r: r.context_match)

        elif self.conflict_resolution == 'performance_weighted':
            return max(sorted_recommendations, key=lambda r: r.priority_score)

        elif self.conflict_resolution == 'situation_specific':
            return self._situation_specific_resolution(sorted_recommendations, context_analysis)

        else:
            return sorted_recommendations[0]  # Default to highest priority

    def _situation_specific_resolution(self,
                                     recommendations: List[AgentRecommendation],
                                     context_analysis: Dict[str, Any]) -> AgentRecommendation:
        """Resolve conflicts based on specific situation analysis"""
        primary_context = context_analysis.get('primary_context', 'unknown')

        # Battle situations: strongly favor battle agent
        if primary_context == 'battle':
            battle_recs = [r for r in recommendations if r.agent_role == AgentRole.BATTLE]
            if battle_recs:
                return battle_recs[0]

        # New area discovery: favor explorer agent
        elif primary_context == 'exploration' or context_analysis.get('new_area_detected', False):
            explorer_recs = [r for r in recommendations if r.agent_role == AgentRole.EXPLORER]
            if explorer_recs:
                return explorer_recs[0]

        # Story/quest contexts: favor progression agent
        elif primary_context == 'story' or context_analysis.get('quest_available', False):
            progression_recs = [r for r in recommendations if r.agent_role == AgentRole.PROGRESSION]
            if progression_recs:
                return progression_recs[0]

        # Default to highest priority
        return recommendations[0]

    def _determine_relevant_agents(self, context_analysis: Dict[str, Any]) -> List[AgentRole]:
        """Determine which agents are relevant for current context"""
        relevant_agents = []
        primary_context = context_analysis.get('primary_context', 'unknown')

        # Always include progression agent for general guidance
        relevant_agents.append(AgentRole.PROGRESSION)

        # Context-specific agents
        if primary_context == 'battle' or context_analysis.get('in_battle', False):
            relevant_agents.append(AgentRole.BATTLE)

        if primary_context == 'exploration' or context_analysis.get('exploration_opportunity', False):
            relevant_agents.append(AgentRole.EXPLORER)

        # If context is unclear, consult all agents
        if primary_context == 'unknown':
            relevant_agents = [AgentRole.BATTLE, AgentRole.EXPLORER, AgentRole.PROGRESSION]

        return relevant_agents

    def _calculate_context_match(self, agent_role: AgentRole, context_analysis: Dict[str, Any]) -> float:
        """Calculate how well an agent matches the current context"""
        primary_context = context_analysis.get('primary_context', 'unknown')

        match_scores = {
            AgentRole.BATTLE: {
                'battle': 0.9,
                'combat': 0.9,
                'gym': 0.8,
                'trainer': 0.8,
                'story': 0.3,
                'exploration': 0.2,
                'unknown': 0.4
            },
            AgentRole.EXPLORER: {
                'exploration': 0.9,
                'discovery': 0.9,
                'new_area': 0.9,
                'navigation': 0.8,
                'battle': 0.2,
                'story': 0.4,
                'unknown': 0.6
            },
            AgentRole.PROGRESSION: {
                'story': 0.9,
                'quest': 0.9,
                'objective': 0.9,
                'progression': 0.9,
                'battle': 0.5,
                'exploration': 0.5,
                'unknown': 0.7
            }
        }

        return match_scores.get(agent_role, {}).get(primary_context, 0.5)

    def _calculate_agent_consensus(self, recommendations: List[AgentRecommendation]) -> float:
        """Calculate level of consensus between agent recommendations"""
        if len(recommendations) <= 1:
            return 1.0

        # Check action agreement
        actions = [rec.action for rec in recommendations]
        most_common_action = max(set(actions), key=actions.count)
        action_agreement = actions.count(most_common_action) / len(actions)

        # Check confidence agreement (how close are the confidence levels)
        confidences = [rec.confidence for rec in recommendations]
        confidence_variance = sum((c - sum(confidences)/len(confidences))**2 for c in confidences) / len(confidences)
        confidence_agreement = max(0, 1 - confidence_variance)

        # Combined consensus score
        return (action_agreement * 0.7) + (confidence_agreement * 0.3)

    def _get_agent_by_role(self, agent_role: AgentRole) -> Optional[BaseAgent]:
        """Get agent instance by role"""
        agent_map = {
            AgentRole.BATTLE: self.battle_agent,
            AgentRole.EXPLORER: self.explorer_agent,
            AgentRole.PROGRESSION: self.progression_agent
        }
        return agent_map.get(agent_role)

    def _get_fallback_decision(self, context_analysis: Dict[str, Any]) -> CoordinationDecision:
        """Get fallback decision when no agent recommendations available"""
        return CoordinationDecision(
            chosen_agent=AgentRole.PROGRESSION,  # Default to progression
            action=5,  # A button - safe default
            confidence=0.3,
            reasoning="Fallback decision - no agent recommendations available",
            agent_consensus=0.0,
            fallback_agents=[]
        )

    def _track_coordination_decision(self,
                                   decision: CoordinationDecision,
                                   game_state: Dict[str, Any],
                                   context_analysis: Dict[str, Any]):
        """Track coordination decision for learning and improvement"""
        decision_record = {
            'chosen_agent': decision.chosen_agent.value,
            'action': decision.action,
            'confidence': decision.confidence,
            'consensus': decision.agent_consensus,
            'context': context_analysis.get('primary_context', 'unknown'),
            'timestamp': self.total_steps
        }

        self.decision_history.append(decision_record)

        # Keep only recent history
        if len(self.decision_history) > 200:
            self.decision_history.pop(0)

        # Update agent usage statistics
        self.agent_performance[decision.chosen_agent]['total_decisions'] += 1

    def _update_chosen_agent(self,
                           decision: CoordinationDecision,
                           game_state: Dict[str, Any],
                           info: Dict[str, Any]):
        """Update the chosen agent with the action outcome"""
        agent = self._get_agent_by_role(decision.chosen_agent)
        if agent:
            # The agent's update method will be called later with reward
            pass

    def update(self, reward: float) -> None:
        """Update coordinator and all agents with reward feedback"""
        super().update(reward)

        # Update all specialist agents
        self.battle_agent.update(reward)
        self.explorer_agent.update(reward)
        self.progression_agent.update(reward)

        # Update coordination learning
        self._update_coordination_learning(reward)

    def _update_coordination_learning(self, reward: float):
        """Update coordination learning based on reward feedback"""
        if self.decision_history:
            last_decision = self.decision_history[-1]
            chosen_agent = AgentRole(last_decision['chosen_agent'])

            # Update agent performance tracking
            if reward > 0.1:  # Consider positive reward as success
                self.agent_performance[chosen_agent]['successful_outcomes'] += 1

            # Recalculate success rate
            agent_stats = self.agent_performance[chosen_agent]
            if agent_stats['total_decisions'] > 0:
                agent_stats['success_rate'] = agent_stats['successful_outcomes'] / agent_stats['total_decisions']

            # Adaptive weight adjustment
            if self.adaptive_weights:
                self._adjust_agent_weights(reward, chosen_agent)

    def _adjust_agent_weights(self, reward: float, chosen_agent: AgentRole):
        """Adjust agent weights based on performance"""
        if reward > 0.1:  # Good outcome
            self.agent_weights[chosen_agent.value] = min(2.0,
                self.agent_weights[chosen_agent.value] + self.learning_rate)
        elif reward < -0.1:  # Poor outcome
            self.agent_weights[chosen_agent.value] = max(0.5,
                self.agent_weights[chosen_agent.value] - self.learning_rate)

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive multi-agent coordinator statistics"""
        base_stats = super().get_stats()

        # Get individual agent stats
        agent_stats = {
            'battle_agent': self.battle_agent.get_stats(),
            'explorer_agent': self.explorer_agent.get_stats(),
            'progression_agent': self.progression_agent.get_stats()
        }

        # Coordination stats
        coordination_stats = {
            'consensus_threshold': self.consensus_threshold,
            'agent_weights': self.agent_weights,
            'agent_performance': self.agent_performance,
            'decision_history_size': len(self.decision_history),
            'adaptive_weights': self.adaptive_weights,
            'conflict_resolution': self.conflict_resolution
        }

        return {
            **base_stats,
            'agent_stats': agent_stats,
            'coordination_stats': coordination_stats
        }


class ContextAnalyzer:
    """Analyzes game context to determine agent relevance"""

    def __init__(self):
        self.logger = logging.getLogger("ContextAnalyzer")

    def analyze_context(self, game_state: Dict[str, Any], info: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze current game context for agent coordination"""

        context = {
            'primary_context': self._determine_primary_context(game_state, info),
            'secondary_contexts': self._identify_secondary_contexts(game_state, info),
            'urgency_level': self._calculate_urgency_level(game_state),
            'complexity_level': self._assess_situation_complexity(game_state, info),
            'agent_relevance': self._assess_agent_relevance(game_state, info)
        }

        # Add specific context flags
        context.update({
            'in_battle': game_state.get('in_battle', False),
            'new_area_detected': self._detect_new_area(game_state),
            'quest_available': self._detect_quest_opportunity(game_state, info),
            'exploration_opportunity': self._detect_exploration_opportunity(game_state, info),
            'story_progression_needed': self._detect_story_progression_need(game_state)
        })

        return context

    def _determine_primary_context(self, game_state: Dict[str, Any], info: Dict[str, Any]) -> str:
        """Determine the primary context for the current situation"""

        # Battle takes highest priority
        if game_state.get('in_battle', False):
            return 'battle'

        # Story progression needs
        party_count = game_state.get('party_count', 0)
        if party_count == 0:
            return 'story'  # Need starter Pokemon

        # Exploration contexts
        current_map = game_state.get('player_map', 0)
        if self._is_new_or_unexplored_area(current_map):
            return 'exploration'

        # Default to progression
        return 'progression'

    def _identify_secondary_contexts(self, game_state: Dict[str, Any], info: Dict[str, Any]) -> List[str]:
        """Identify secondary contexts that might be relevant"""
        contexts = []

        # Always consider progression as secondary if not primary
        if self._determine_primary_context(game_state, info) != 'progression':
            contexts.append('progression')

        # Exploration opportunities
        if self._detect_exploration_opportunity(game_state, info):
            contexts.append('exploration')

        return contexts

    def _calculate_urgency_level(self, game_state: Dict[str, Any]) -> int:
        """Calculate urgency level 1-5"""
        urgency = 1

        # Battle urgency
        if game_state.get('in_battle', False):
            hp_ratio = game_state.get('player_hp', 0) / max(game_state.get('player_max_hp', 1), 1)
            if hp_ratio < 0.2:
                urgency = 5
            elif hp_ratio < 0.5:
                urgency = 4
            else:
                urgency = 3

        # Progression urgency
        party_count = game_state.get('party_count', 0)
        if party_count == 0:
            urgency = max(urgency, 4)

        return urgency

    def _assess_situation_complexity(self, game_state: Dict[str, Any], info: Dict[str, Any]) -> int:
        """Assess situation complexity 1-5"""
        complexity = 1

        # Multiple factors increase complexity
        factors = 0
        if game_state.get('in_battle', False):
            factors += 1
        if self._detect_quest_opportunity(game_state, info):
            factors += 1
        if self._detect_exploration_opportunity(game_state, info):
            factors += 1

        return min(5, factors + 1)

    def _assess_agent_relevance(self, game_state: Dict[str, Any], info: Dict[str, Any]) -> Dict[str, float]:
        """Assess relevance of each agent type for current situation"""
        relevance = {
            'battle': 0.3,
            'explorer': 0.3,
            'progression': 0.5  # Always somewhat relevant
        }

        # Adjust based on context
        if game_state.get('in_battle', False):
            relevance['battle'] = 0.9
            relevance['explorer'] = 0.1

        if self._detect_exploration_opportunity(game_state, info):
            relevance['explorer'] = 0.8

        if self._detect_story_progression_need(game_state):
            relevance['progression'] = 0.9

        return relevance

    # Helper methods (simplified implementations)
    def _detect_new_area(self, game_state: Dict[str, Any]) -> bool:
        return False  # Would implement area detection logic

    def _detect_quest_opportunity(self, game_state: Dict[str, Any], info: Dict[str, Any]) -> bool:
        return False  # Would implement quest detection logic

    def _detect_exploration_opportunity(self, game_state: Dict[str, Any], info: Dict[str, Any]) -> bool:
        return True  # Always some exploration opportunity

    def _detect_story_progression_need(self, game_state: Dict[str, Any]) -> bool:
        return game_state.get('party_count', 0) == 0  # Need starter Pokemon

    def _is_new_or_unexplored_area(self, map_id: int) -> bool:
        return False  # Would track explored areas