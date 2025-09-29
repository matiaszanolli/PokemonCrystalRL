#!/usr/bin/env python3
"""
Hybrid LLM-RL Decision Engine

This module implements a sophisticated decision engine that combines the strategic
reasoning capabilities of Large Language Models with the tactical optimization
of Reinforcement Learning. The hybrid approach leverages:

- LLM for high-level strategic decisions and novel situation handling
- RL for optimized tactical execution and pattern recognition
- Temporal memory for bridging reasoning across time scales
- Curriculum learning integration for progressive skill development
"""

import logging
import numpy as np
import random
import time
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum

from core.temporal_memory import TemporalMemoryBuffer, TemporalState, TemporalExperience, DecisionSource
from agents.llm_agent import LLMAgent
from agents.dqn_agent import DQNAgent
from core.experience_memory import ExperienceMemory
from training.curriculum_learning import CurriculumManager
from core.game_intelligence import GameIntelligence

logger = logging.getLogger(__name__)


class DecisionMode(Enum):
    """Decision making modes for the hybrid agent."""
    LLM_STRATEGIC = "llm_strategic"      # LLM for strategic decisions
    RL_TACTICAL = "rl_tactical"          # RL for tactical optimization
    HYBRID_BALANCED = "hybrid_balanced"   # Weighted combination
    EXPLORATION = "exploration"          # Exploration mode
    CURRICULUM_GUIDED = "curriculum_guided"  # Curriculum learning guided


@dataclass
class DecisionMetrics:
    """Metrics for decision performance tracking."""
    llm_decisions: int = 0
    rl_decisions: int = 0
    hybrid_decisions: int = 0
    exploration_decisions: int = 0

    llm_success_rate: float = 0.0
    rl_success_rate: float = 0.0
    hybrid_success_rate: float = 0.0

    avg_llm_confidence: float = 0.0
    avg_rl_q_value: float = 0.0

    mode_switches: int = 0
    total_decisions: int = 0


@dataclass
class DecisionContext:
    """Context information for making decision mode choices."""
    game_state: Dict[str, Any]
    temporal_state: TemporalState
    recent_performance: Dict[str, float]
    curriculum_stage: Optional[str] = None
    time_since_last_reward: float = 0.0
    exploration_factor: float = 0.1
    novelty_score: float = 0.0


class HybridLLMRLAgent:
    """
    Hybrid agent that intelligently combines LLM strategic reasoning
    with RL tactical optimization using temporal memory.
    """

    def __init__(
        self,
        llm_agent: LLMAgent,
        rl_agent: Optional[DQNAgent] = None,
        temporal_memory: Optional[TemporalMemoryBuffer] = None,
        curriculum_manager: Optional[CurriculumManager] = None,
        game_intelligence: Optional[GameIntelligence] = None,
        config: Optional[Dict] = None
    ):
        self.llm_agent = llm_agent
        self.rl_agent = rl_agent
        self.temporal_memory = temporal_memory
        self.curriculum_manager = curriculum_manager
        self.game_intelligence = game_intelligence

        # Configuration with defaults
        self.config = config or {}
        self.llm_weight = self.config.get('llm_weight', 0.7)
        self.rl_weight = self.config.get('rl_weight', 0.3)
        self.exploration_rate = self.config.get('exploration_rate', 0.1)
        self.mode_switch_threshold = self.config.get('mode_switch_threshold', 0.2)
        self.novelty_threshold = self.config.get('novelty_threshold', 0.5)

        # Performance tracking
        self.metrics = DecisionMetrics()
        self.decision_history: List[Tuple[DecisionMode, int, float]] = []
        self.last_mode = DecisionMode.LLM_STRATEGIC
        self.mode_confidence_history: Dict[DecisionMode, List[float]] = {
            mode: [] for mode in DecisionMode
        }

        # Adaptive parameters
        self.llm_success_window = 10
        self.rl_success_window = 20
        self.performance_window = 50

        logger.info("🤖 Hybrid LLM-RL Agent initialized")
        logger.info(f"   LLM Weight: {self.llm_weight}, RL Weight: {self.rl_weight}")
        logger.info(f"   Exploration Rate: {self.exploration_rate}")

    def decide_action(
        self,
        game_state: Dict[str, Any],
        action_space: List[int],
        context: Optional[Dict] = None
    ) -> Tuple[int, Dict[str, Any]]:
        """
        Main decision function that chooses between LLM and RL based on context.

        Returns:
            Tuple of (action, decision_info)
        """
        start_time = time.time()

        # Create temporal state representation
        temporal_state = self._create_temporal_state(game_state, context)

        # Create decision context
        decision_context = self._create_decision_context(
            game_state, temporal_state, context
        )

        # Determine decision mode
        decision_mode = self._select_decision_mode(decision_context)

        # Execute decision based on mode
        action, confidence = self._execute_decision(
            decision_mode, decision_context, action_space
        )

        decision_time = time.time() - start_time

        # Update metrics and history
        self._update_metrics(decision_mode, confidence, decision_time)

        # Store decision in temporal memory
        if self.temporal_memory:
            # Map decision mode to DecisionSource
            decision_source_map = {
                DecisionMode.LLM_STRATEGIC: DecisionSource.LLM,
                DecisionMode.RL_TACTICAL: DecisionSource.RL,
                DecisionMode.HYBRID_BALANCED: DecisionSource.HYBRID,
                DecisionMode.EXPLORATION: DecisionSource.RULE_BASED
            }

            experience = TemporalExperience(
                state=temporal_state,
                action=action,
                reward=0.0,  # Will be updated later with actual reward
                next_state=None,  # Will be filled later
                done=False,  # Will be determined later
                decision_source=decision_source_map.get(decision_mode, DecisionSource.HYBRID),
                llm_reasoning=context.get('llm_reasoning') if context else None,
                llm_confidence=confidence if decision_mode == DecisionMode.LLM_STRATEGIC else None
            )
            # Record experience in temporal memory using the correct method
            self.temporal_memory.record_temporal_experience(
                game_state=decision_context.game_state,
                screen_analysis={'state': temporal_state.screen_state},
                action=action,
                reward=0.0,  # Will be updated later with actual reward
                decision_source=decision_source_map.get(decision_mode, DecisionSource.HYBRID),
                context=context,
                llm_reasoning=context.get('llm_reasoning') if context else None,
                llm_confidence=confidence if decision_mode == DecisionMode.LLM_STRATEGIC else None
            )

        decision_info = {
            'mode': decision_mode.value,
            'confidence': confidence,
            'decision_time': decision_time,
            'temporal_state': temporal_state,
            'llm_weight': self.llm_weight,
            'rl_weight': self.rl_weight,
            'metrics': self.metrics
        }

        self.last_mode = decision_mode
        return action, decision_info

    def _create_temporal_state(
        self,
        game_state: Dict[str, Any],
        context: Optional[Dict] = None
    ) -> TemporalState:
        """Create temporal state representation for current game state."""
        # Extract position
        position = (game_state.get('player_x', 0), game_state.get('player_y', 0))

        # Calculate HP ratio
        current_hp = game_state.get('player_hp', 100)
        max_hp = game_state.get('player_max_hp', 100)
        hp_ratio = current_hp / max_hp if max_hp > 0 else 1.0

        # Determine screen state flags
        screen_state = game_state.get('screen_state', 'unknown')
        in_battle = screen_state == 'battle'
        in_menu = screen_state in ['menu', 'settings']

        # Determine location type based on map_id (simplified)
        map_id = game_state.get('map_id', 0)
        if map_id < 10:
            location_type = 'town'
        elif map_id < 50:
            location_type = 'route'
        else:
            location_type = 'building'

        return TemporalState(
            position=position,
            map_id=map_id,
            hp_ratio=hp_ratio,
            level=game_state.get('player_level', 1),
            badges=game_state.get('badges', 0),
            party_size=len(game_state.get('party', [])),
            money=game_state.get('money', 0),
            screen_state=screen_state,
            game_phase=self._determine_game_phase(game_state),
            location_type=location_type,
            in_battle=in_battle,
            in_menu=in_menu,
            timestep=int(time.time() * 1000) % 1000000,  # Convert to timestep
            episode_time=time.time() % 3600  # Episode time in seconds
        )

    def _extract_feature_vector(self, game_state: Dict[str, Any]) -> np.ndarray:
        """Extract numerical feature vector from game state for RL."""
        features = []

        # Basic stats
        features.extend([
            game_state.get('player_hp', 0) / 100.0,  # Normalized HP
            game_state.get('player_level', 1) / 100.0,  # Normalized level
            game_state.get('badges', 0) / 16.0,  # Normalized badges
            game_state.get('money', 0) / 10000.0,  # Normalized money
            len(game_state.get('party', [])) / 6.0,  # Party size
        ])

        # Location features
        features.extend([
            game_state.get('player_x', 0) / 255.0,  # Normalized coordinates
            game_state.get('player_y', 0) / 255.0,
            game_state.get('map_id', 0) / 100.0,
        ])

        # Screen state features (one-hot encoded)
        screen_state = game_state.get('screen_state', 'unknown')
        screen_states = ['overworld', 'battle', 'dialogue', 'menu', 'settings']
        screen_features = [1.0 if screen_state == state else 0.0 for state in screen_states]
        features.extend(screen_features)

        # Ensure consistent size
        while len(features) < 32:
            features.append(0.0)

        return np.array(features[:32], dtype=np.float32)

    def _determine_game_phase(self, game_state: Dict[str, Any]) -> str:
        """Determine current game phase for curriculum integration."""
        badges = game_state.get('badges', 0)
        party_size = len(game_state.get('party', []))

        if party_size == 0:
            return 'tutorial'
        elif badges == 0:
            return 'early_game'
        elif badges < 4:
            return 'mid_game'
        elif badges < 8:
            return 'late_game'
        else:
            return 'post_game'

    def _create_decision_context(
        self,
        game_state: Dict[str, Any],
        temporal_state: TemporalState,
        context: Optional[Dict] = None
    ) -> DecisionContext:
        """Create decision context for mode selection."""
        # Calculate recent performance
        recent_performance = self._calculate_recent_performance()

        # Get curriculum stage if available
        curriculum_stage = None
        if self.curriculum_manager:
            curriculum_stage = self.curriculum_manager.get_current_stage().value

        # Calculate novelty score
        novelty_score = self._calculate_novelty_score(temporal_state)

        # Time since last reward
        time_since_reward = self._time_since_last_reward()

        return DecisionContext(
            game_state=game_state,
            temporal_state=temporal_state,
            recent_performance=recent_performance,
            curriculum_stage=curriculum_stage,
            time_since_last_reward=time_since_reward,
            exploration_factor=self.exploration_rate,
            novelty_score=novelty_score
        )

    def _select_decision_mode(self, context: DecisionContext) -> DecisionMode:
        """Select appropriate decision mode based on context."""
        # Curriculum learning guidance
        if context.curriculum_stage:
            if context.curriculum_stage == 'tutorial':
                return DecisionMode.LLM_STRATEGIC  # LLM better for learning
            elif context.curriculum_stage in ['advanced', 'expert']:
                return DecisionMode.RL_TACTICAL  # RL better for optimization

        # Novelty-based selection
        if context.novelty_score > self.novelty_threshold:
            return DecisionMode.LLM_STRATEGIC  # LLM better for novel situations

        # Performance-based selection
        llm_performance = context.recent_performance.get('llm', 0.5)
        rl_performance = context.recent_performance.get('rl', 0.5)

        if abs(llm_performance - rl_performance) > self.mode_switch_threshold:
            if llm_performance > rl_performance:
                return DecisionMode.LLM_STRATEGIC
            else:
                return DecisionMode.RL_TACTICAL

        # Exploration check
        if random.random() < context.exploration_factor:
            return DecisionMode.EXPLORATION

        # Default to hybrid balanced
        return DecisionMode.HYBRID_BALANCED

    def _execute_decision(
        self,
        mode: DecisionMode,
        context: DecisionContext,
        action_space: List[int]
    ) -> Tuple[int, float]:
        """Execute decision based on selected mode."""
        if mode == DecisionMode.LLM_STRATEGIC:
            return self._llm_decision(context, action_space)

        elif mode == DecisionMode.RL_TACTICAL:
            return self._rl_decision(context, action_space)

        elif mode == DecisionMode.HYBRID_BALANCED:
            return self._hybrid_decision(context, action_space)

        elif mode == DecisionMode.EXPLORATION:
            return self._exploration_decision(action_space)

        else:
            # Fallback to LLM
            return self._llm_decision(context, action_space)

    def _llm_decision(
        self,
        context: DecisionContext,
        action_space: List[int]
    ) -> Tuple[int, float]:
        """Get decision from LLM agent."""
        try:
            # Use game intelligence for enhanced context
            if self.game_intelligence:
                enhanced_context = self.game_intelligence.analyze_game_context(
                    context.game_state,
                    {'state': context.temporal_state.screen_state}
                )
            else:
                enhanced_context = context.game_state

            # Convert enhanced context to dict for LLM agent
            if hasattr(enhanced_context, '__dict__'):
                context_dict = enhanced_context.__dict__
            else:
                context_dict = enhanced_context

            action = self.llm_agent.get_action(context_dict)

            # Ensure action is in action space
            if action not in action_space:
                action = random.choice(action_space)

            # LLM confidence based on context richness
            confidence = min(0.9, 0.5 + 0.1 * len(str(enhanced_context)))

            return action, confidence

        except Exception as e:
            logger.warning(f"LLM decision failed: {e}")
            return random.choice(action_space), 0.1

    def _rl_decision(
        self,
        context: DecisionContext,
        action_space: List[int]
    ) -> Tuple[int, float]:
        """Get decision from RL agent."""
        if not self.rl_agent:
            # Fallback to random if no RL agent
            return random.choice(action_space), 0.1

        try:
            # Convert temporal state to game state format for DQN agent
            game_state = {
                'player_x': context.temporal_state.position[0],
                'player_y': context.temporal_state.position[1],
                'player_hp': int(context.temporal_state.hp_ratio * 100),
                'player_level': context.temporal_state.level,
                'badges': context.temporal_state.badges,
                'money': context.temporal_state.money,
                'map_id': context.temporal_state.map_id,
                'party': [{'level': context.temporal_state.level}] * context.temporal_state.party_size
            }
            screen_analysis = {'state': context.temporal_state.screen_state}

            action_str, confidence_score = self.rl_agent.get_action(
                game_state, screen_analysis, training=True
            )

            # Convert action string to integer
            action_map = {
                'none': 0, 'up': 1, 'down': 2, 'left': 3, 'right': 4,
                'a': 5, 'b': 6, 'start': 7, 'select': 8
            }
            action = action_map.get(action_str.lower(), 0)

            # Ensure action is in action space
            if action not in action_space:
                action = random.choice(action_space)

            # RL confidence from the DQN agent confidence score
            confidence = min(0.9, max(0.1, confidence_score))

            return action, confidence

        except Exception as e:
            logger.warning(f"RL decision failed: {e}")
            return random.choice(action_space), 0.1

    def _hybrid_decision(
        self,
        context: DecisionContext,
        action_space: List[int]
    ) -> Tuple[int, float]:
        """Combine LLM and RL decisions."""
        llm_action, llm_conf = self._llm_decision(context, action_space)
        rl_action, rl_conf = self._rl_decision(context, action_space)

        # Weighted selection based on confidence and weights
        llm_score = llm_conf * self.llm_weight
        rl_score = rl_conf * self.rl_weight

        if llm_score > rl_score:
            action = llm_action
            confidence = llm_conf * 0.8  # Slight penalty for hybrid
        else:
            action = rl_action
            confidence = rl_conf * 0.8

        return action, confidence

    def _exploration_decision(self, action_space: List[int]) -> Tuple[int, float]:
        """Random exploration decision."""
        action = random.choice(action_space)
        confidence = 0.3  # Low confidence for exploration
        return action, confidence

    def _calculate_recent_performance(self) -> Dict[str, float]:
        """Calculate recent performance for each decision mode."""
        if not self.decision_history:
            return {'llm': 0.5, 'rl': 0.5, 'hybrid': 0.5}

        # Get recent decisions (last 20)
        recent = self.decision_history[-20:]

        performance = {}
        for mode in ['llm_strategic', 'rl_tactical', 'hybrid_balanced']:
            mode_decisions = []
            for entry in recent:
                # Handle variable-length tuples (with or without reward)
                if len(entry) >= 4:  # Has reward
                    decision_mode, action, reward, confidence = entry[:4]
                    if decision_mode.value == mode:
                        mode_decisions.append((reward, confidence))
                elif len(entry) == 3:  # No reward yet
                    decision_mode, action, confidence = entry
                    if decision_mode.value == mode:
                        mode_decisions.append((0.0, confidence))  # Default reward

            if mode_decisions:
                avg_reward = np.mean([reward for reward, _ in mode_decisions])
                performance[mode.split('_')[0]] = max(0.0, min(1.0, (avg_reward + 1) / 2))
            else:
                performance[mode.split('_')[0]] = 0.5

        return performance

    def _calculate_novelty_score(self, temporal_state: TemporalState) -> float:
        """Calculate novelty score for current state."""
        if not self.temporal_memory:
            return 0.5

        try:
            # Use simple heuristic based on position and state
            # More sophisticated similarity could be added later
            position_str = f"{temporal_state.position[0]}_{temporal_state.position[1]}"
            state_str = f"{temporal_state.map_id}_{temporal_state.screen_state}"

            # Check against recent experiences for novelty
            if len(self.temporal_memory.experience_buffer) > 10:
                recent_positions = set()
                for exp in list(self.temporal_memory.experience_buffer)[-10:]:
                    exp_pos = f"{exp.state.position[0]}_{exp.state.position[1]}"
                    recent_positions.add(exp_pos)

                # Novel if current position not in recent positions
                if position_str not in recent_positions:
                    return 0.8  # High novelty
                else:
                    return 0.2  # Low novelty

            return 0.5  # Medium novelty for insufficient data

        except Exception as e:
            logger.warning(f"Novelty calculation failed: {e}")
            return 0.5

    def _time_since_last_reward(self) -> float:
        """Calculate time since last positive reward."""
        if not self.decision_history:
            return 0.0

        current_time = time.time()
        for i in range(len(self.decision_history) - 1, -1, -1):
            mode, action, reward, confidence = self.decision_history[i]
            if reward > 0:
                # Use index as proxy for time (could be enhanced)
                return len(self.decision_history) - i

        return len(self.decision_history)

    def _update_metrics(
        self,
        mode: DecisionMode,
        confidence: float,
        decision_time: float
    ):
        """Update performance metrics."""
        self.metrics.total_decisions += 1

        if mode == DecisionMode.LLM_STRATEGIC:
            self.metrics.llm_decisions += 1
        elif mode == DecisionMode.RL_TACTICAL:
            self.metrics.rl_decisions += 1
        elif mode == DecisionMode.HYBRID_BALANCED:
            self.metrics.hybrid_decisions += 1
        elif mode == DecisionMode.EXPLORATION:
            self.metrics.exploration_decisions += 1

        # Track mode switches
        if mode != self.last_mode:
            self.metrics.mode_switches += 1

        # Update confidence history
        self.mode_confidence_history[mode].append(confidence)

        # Keep only recent history
        for mode_hist in self.mode_confidence_history.values():
            if len(mode_hist) > 50:
                mode_hist.pop(0)

    def update_with_reward(self, action: int, reward: float, next_state: Dict[str, Any]):
        """Update agent with reward feedback."""
        # Update decision history with reward
        if self.decision_history:
            last_entry = self.decision_history[-1]
            # Add reward to last decision tuple
            updated_entry = last_entry + (reward,)
            self.decision_history[-1] = updated_entry

        # Update RL agent if available
        if self.rl_agent and self.temporal_memory:
            try:
                # Get most recent experience from buffer
                if len(self.temporal_memory.experience_buffer) > 0:
                    experience = self.temporal_memory.experience_buffer[-1]
                    current_state = experience.state.feature_vector
                    next_state_vector = self._extract_feature_vector(next_state)

                    # Store experience in RL agent
                    game_state = {
                        'player_x': experience.state.position[0],
                        'player_y': experience.state.position[1],
                        'player_hp': int(experience.state.hp_ratio * 100),
                        'player_level': experience.state.level,
                        'badges': experience.state.badges,
                        'money': experience.state.money,
                        'map_id': experience.state.map_id
                    }
                    screen_analysis = {'state': experience.state.screen_state}

                    # Convert action int to string for DQN agent
                    action_map = {
                        0: 'none', 1: 'up', 2: 'down', 3: 'left', 4: 'right',
                        5: 'a', 6: 'b', 7: 'start', 8: 'select'
                    }
                    action_str = action_map.get(action, 'none')

                    # Create next screen analysis from next state
                    next_screen_analysis = {'state': next_state.get('screen_state', 'unknown')}

                    self.rl_agent.store_experience(
                        game_state, screen_analysis, action_str, reward, next_state, next_screen_analysis, done=(reward < -10)
                    )

                    # Trigger training if enough experiences
                    if hasattr(self.rl_agent, 'train_step'):
                        self.rl_agent.train_step()

            except Exception as e:
                logger.warning(f"RL update failed: {e}")

        # Update temporal memory with reward
        if self.temporal_memory and len(self.temporal_memory.experience_buffer) > 0:
            # Update the last experience with the reward
            last_experience = self.temporal_memory.experience_buffer[-1]
            last_experience.reward = reward

        # Update success rates
        self._update_success_rates(reward)

    def _update_success_rates(self, reward: float):
        """Update success rates for different modes."""
        success = reward > 0

        # Update based on last decision mode
        if self.last_mode == DecisionMode.LLM_STRATEGIC:
            self._update_mode_success_rate('llm', success)
        elif self.last_mode == DecisionMode.RL_TACTICAL:
            self._update_mode_success_rate('rl', success)
        elif self.last_mode == DecisionMode.HYBRID_BALANCED:
            self._update_mode_success_rate('hybrid', success)

    def _update_mode_success_rate(self, mode: str, success: bool):
        """Update success rate for specific mode."""
        if mode == 'llm':
            # Simple moving average
            current = self.metrics.llm_success_rate
            self.metrics.llm_success_rate = current * 0.9 + (1.0 if success else 0.0) * 0.1
        elif mode == 'rl':
            current = self.metrics.rl_success_rate
            self.metrics.rl_success_rate = current * 0.9 + (1.0 if success else 0.0) * 0.1
        elif mode == 'hybrid':
            current = self.metrics.hybrid_success_rate
            self.metrics.hybrid_success_rate = current * 0.9 + (1.0 if success else 0.0) * 0.1

    def get_decision_summary(self) -> Dict[str, Any]:
        """Get comprehensive decision summary for monitoring."""
        total = max(1, self.metrics.total_decisions)

        return {
            'total_decisions': total,
            'mode_distribution': {
                'llm': self.metrics.llm_decisions / total,
                'rl': self.metrics.rl_decisions / total,
                'hybrid': self.metrics.hybrid_decisions / total,
                'exploration': self.metrics.exploration_decisions / total
            },
            'success_rates': {
                'llm': self.metrics.llm_success_rate,
                'rl': self.metrics.rl_success_rate,
                'hybrid': self.metrics.hybrid_success_rate
            },
            'current_weights': {
                'llm_weight': self.llm_weight,
                'rl_weight': self.rl_weight
            },
            'mode_switches': self.metrics.mode_switches,
            'last_mode': self.last_mode.value,
            'avg_confidences': {
                mode.value: np.mean(hist) if hist else 0.0
                for mode, hist in self.mode_confidence_history.items()
            }
        }

    def adapt_weights(self):
        """Adapt LLM/RL weights based on recent performance."""
        performance = self._calculate_recent_performance()

        llm_perf = performance.get('llm', 0.5)
        rl_perf = performance.get('rl', 0.5)

        # Adaptive weight adjustment
        total_perf = llm_perf + rl_perf
        if total_perf > 0:
            self.llm_weight = 0.3 + 0.4 * (llm_perf / total_perf)
            self.rl_weight = 0.3 + 0.4 * (rl_perf / total_perf)

        # Ensure weights sum close to 1
        total_weight = self.llm_weight + self.rl_weight
        if total_weight > 0:
            self.llm_weight /= total_weight
            self.rl_weight /= total_weight

        logger.debug(f"Adapted weights: LLM={self.llm_weight:.3f}, RL={self.rl_weight:.3f}")