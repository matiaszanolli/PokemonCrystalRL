"""
BattleAgent - Specialist agent for combat optimization

This agent specializes in battle decisions, move selection, and combat strategy.
It leverages the enhanced BattleStrategy system and focuses purely on optimal combat performance.
"""

import logging
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass

from .base_agent import BaseAgent
from ..core.event_system import EventType, Event, EventSubscriber, get_event_bus

try:
    from core.game_intelligence import BattleStrategy
except ImportError:
    print("⚠️  BattleStrategy not available")
    BattleStrategy = None


@dataclass
class BattleDecision:
    """Represents a battle decision with reasoning"""
    action: int
    move_type: str  # 'attack', 'defend', 'item', 'switch'
    confidence: float
    reasoning: str
    risk_level: str  # 'low', 'medium', 'high'
    expected_outcome: str


class BattleAgent(BaseAgent, EventSubscriber):
    """Specialist agent optimized for Pokemon battle decisions"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.logger = logging.getLogger("BattleAgent")

        # Battle-specific configuration
        self.battle_config = config.get('battle_config', {}) if config else {}
        self.aggression_level = self.battle_config.get('aggression', 0.7)  # 0.0-1.0
        self.risk_tolerance = self.battle_config.get('risk_tolerance', 0.5)  # 0.0-1.0
        self.switch_threshold = self.battle_config.get('switch_threshold', 0.2)  # HP ratio

        # Initialize battle strategy system
        self.battle_strategy = BattleStrategy() if BattleStrategy else None

        # Battle tracking
        self.battle_history = []
        self.move_effectiveness_stats = {}
        self.type_matchup_memory = {}
        self.battle_win_rate = 0.0
        self.battles_fought = 0
        self.battles_won = 0

        # Decision patterns
        self.successful_strategies = {}
        self.failed_strategies = {}

        # Event system integration
        self.event_bus = get_event_bus()
        self.event_bus.subscribe(self)

        self.logger.info(f"BattleAgent initialized with aggression={self.aggression_level}, risk_tolerance={self.risk_tolerance}")

    def get_action(self, game_state: Dict[str, Any], info: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
        """Get battle-optimized action"""
        # Verify we're in battle
        if not game_state.get('in_battle', False):
            return self._non_battle_action(game_state, info)

        # Analyze battle situation
        battle_analysis = self._analyze_battle_context(game_state, info)

        # Make battle decision
        decision = self._make_battle_decision(game_state, battle_analysis)

        # Update decision tracking
        self._track_decision(decision, game_state)

        # Return action with battle intelligence
        return decision.action, {
            'source': 'battle_agent',
            'decision_type': decision.move_type,
            'confidence': decision.confidence,
            'reasoning': decision.reasoning,
            'risk_level': decision.risk_level,
            'expected_outcome': decision.expected_outcome,
            'battle_phase': battle_analysis.get('battle_phase', 'unknown')
        }

    def _analyze_battle_context(self, game_state: Dict[str, Any], info: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive battle situation analysis"""
        if not self.battle_strategy:
            return {'analysis_available': False}

        # Get base battle analysis
        situation = self.battle_strategy.analyze_battle_situation(game_state)

        # Add specialist battle agent insights
        player_hp_ratio = situation.get('player_hp_ratio', 1.0)
        level_difference = situation.get('level_difference', 0)

        # Battle momentum analysis
        momentum = self._calculate_battle_momentum(game_state)

        # Type advantage analysis (if type info available)
        type_advantage = self._analyze_type_matchup(game_state)

        # Strategic position assessment
        strategic_position = self._assess_strategic_position(game_state, situation)

        enhanced_analysis = {
            **situation,
            'momentum': momentum,
            'type_advantage': type_advantage,
            'strategic_position': strategic_position,
            'recommended_strategy': self._determine_optimal_strategy(situation, momentum, type_advantage),
            'urgency_level': self._calculate_urgency(player_hp_ratio, level_difference)
        }

        return enhanced_analysis

    def _make_battle_decision(self, game_state: Dict[str, Any], analysis: Dict[str, Any]) -> BattleDecision:
        """Make intelligent battle decision based on analysis"""

        # Handle emergency situations first
        if analysis.get('urgency_level', 0) >= 4:
            return self._emergency_decision(game_state, analysis)

        # Determine decision approach based on battle phase and strategy
        recommended_strategy = analysis.get('recommended_strategy', 'balanced')
        battle_phase = analysis.get('battle_phase', 'unknown')

        if recommended_strategy == 'aggressive':
            return self._aggressive_decision(game_state, analysis)
        elif recommended_strategy == 'defensive':
            return self._defensive_decision(game_state, analysis)
        elif recommended_strategy == 'tactical':
            return self._tactical_decision(game_state, analysis)
        else:
            return self._balanced_decision(game_state, analysis)

    def _aggressive_decision(self, game_state: Dict[str, Any], analysis: Dict[str, Any]) -> BattleDecision:
        """Make aggressive battle decision prioritizing offense"""
        hp_ratio = analysis.get('player_hp_ratio', 1.0)
        momentum = analysis.get('momentum', 0.0)

        if momentum > 0.3 and hp_ratio > 0.4:
            return BattleDecision(
                action=5,  # A button - attack
                move_type='attack',
                confidence=0.8,
                reasoning='High momentum and good HP - press advantage with aggressive attack',
                risk_level='medium',
                expected_outcome='damage_enemy'
            )
        else:
            return BattleDecision(
                action=5,  # A button - attack
                move_type='attack',
                confidence=0.6,
                reasoning='Aggressive strategy - attack despite moderate position',
                risk_level='high',
                expected_outcome='trade_damage'
            )

    def _defensive_decision(self, game_state: Dict[str, Any], analysis: Dict[str, Any]) -> BattleDecision:
        """Make defensive battle decision prioritizing survival"""
        hp_ratio = analysis.get('player_hp_ratio', 1.0)

        if hp_ratio < 0.3:
            # Consider healing or switching
            party_count = game_state.get('party_count', 1)
            if party_count > 1 and hp_ratio < self.switch_threshold:
                return BattleDecision(
                    action=6,  # B button - might access switch menu
                    move_type='switch',
                    confidence=0.7,
                    reasoning='Low HP - attempt to switch to healthier Pokemon',
                    risk_level='low',
                    expected_outcome='preserve_pokemon'
                )
            else:
                return BattleDecision(
                    action=2,  # Down - might select healing move/item
                    move_type='defend',
                    confidence=0.6,
                    reasoning='Low HP and no switch option - defensive move selection',
                    risk_level='medium',
                    expected_outcome='minimize_damage'
                )
        else:
            return BattleDecision(
                action=5,  # A button - cautious attack
                move_type='attack',
                confidence=0.5,
                reasoning='Defensive approach - cautious attack',
                risk_level='low',
                expected_outcome='steady_damage'
            )

    def _tactical_decision(self, game_state: Dict[str, Any], analysis: Dict[str, Any]) -> BattleDecision:
        """Make tactical decision based on type advantages and positioning"""
        type_advantage = analysis.get('type_advantage', 'neutral')
        level_difference = analysis.get('level_difference', 0)

        if type_advantage == 'super_effective':
            return BattleDecision(
                action=5,  # A button - exploit type advantage
                move_type='attack',
                confidence=0.9,
                reasoning='Super effective type advantage - exploit with strong attack',
                risk_level='low',
                expected_outcome='major_damage'
            )
        elif type_advantage == 'not_very_effective':
            if level_difference > 5:
                # We're overleveled, power through
                return BattleDecision(
                    action=5,  # A button
                    move_type='attack',
                    confidence=0.6,
                    reasoning='Type disadvantage but level advantage - continue attacking',
                    risk_level='medium',
                    expected_outcome='gradual_progress'
                )
            else:
                # Consider switching or different strategy
                return BattleDecision(
                    action=2,  # Down - try different move
                    move_type='tactical',
                    confidence=0.7,
                    reasoning='Type disadvantage - seek better move option',
                    risk_level='low',
                    expected_outcome='better_positioning'
                )
        else:
            return self._balanced_decision(game_state, analysis)

    def _balanced_decision(self, game_state: Dict[str, Any], analysis: Dict[str, Any]) -> BattleDecision:
        """Make balanced decision considering all factors"""
        hp_ratio = analysis.get('player_hp_ratio', 1.0)
        momentum = analysis.get('momentum', 0.0)

        # Standard balanced approach
        if hp_ratio > 0.6 and momentum >= 0:
            return BattleDecision(
                action=5,  # A button - standard attack
                move_type='attack',
                confidence=0.7,
                reasoning='Good HP and position - standard attack approach',
                risk_level='medium',
                expected_outcome='steady_progress'
            )
        elif hp_ratio > 0.3:
            return BattleDecision(
                action=5,  # A button - cautious attack
                move_type='attack',
                confidence=0.6,
                reasoning='Moderate HP - cautious but offensive approach',
                risk_level='medium',
                expected_outcome='careful_progress'
            )
        else:
            return self._defensive_decision(game_state, analysis)

    def _emergency_decision(self, game_state: Dict[str, Any], analysis: Dict[str, Any]) -> BattleDecision:
        """Handle critical battle situations"""
        hp_ratio = analysis.get('player_hp_ratio', 1.0)
        party_count = game_state.get('party_count', 1)

        if hp_ratio < 0.15:
            if party_count > 1:
                return BattleDecision(
                    action=6,  # B button - try to escape/switch
                    move_type='emergency_switch',
                    confidence=0.9,
                    reasoning='CRITICAL HP - emergency switch attempt',
                    risk_level='high',
                    expected_outcome='save_pokemon'
                )
            else:
                return BattleDecision(
                    action=1,  # Up - try to access items/healing
                    move_type='emergency_heal',
                    confidence=0.8,
                    reasoning='CRITICAL HP - emergency healing attempt',
                    risk_level='high',
                    expected_outcome='survive_turn'
                )
        else:
            return BattleDecision(
                action=5,  # A button - try to finish quickly
                move_type='urgent_attack',
                confidence=0.7,
                reasoning='Urgent situation - finish battle quickly',
                risk_level='high',
                expected_outcome='end_battle'
            )

    def _calculate_battle_momentum(self, game_state: Dict[str, Any]) -> float:
        """Calculate battle momentum (-1.0 to 1.0)"""
        # This would analyze recent damage dealt vs received
        # For now, use HP ratio as proxy
        player_hp_ratio = game_state.get('player_hp', 0) / max(game_state.get('player_max_hp', 1), 1)

        # Momentum based on HP advantage
        if player_hp_ratio > 0.7:
            return 0.5  # Good momentum
        elif player_hp_ratio > 0.4:
            return 0.0  # Neutral momentum
        else:
            return -0.3  # Poor momentum

    def _analyze_type_matchup(self, game_state: Dict[str, Any]) -> str:
        """Analyze type effectiveness (requires Pokemon type data)"""
        # This would analyze player vs enemy types
        # For now, return neutral since we need type mapping
        return 'neutral'

    def _assess_strategic_position(self, game_state: Dict[str, Any], situation: Dict[str, Any]) -> str:
        """Assess overall strategic battle position"""
        hp_ratio = situation.get('player_hp_ratio', 1.0)
        level_advantage = situation.get('level_advantage', 'even')

        if hp_ratio > 0.7 and level_advantage == 'player':
            return 'dominant'
        elif hp_ratio > 0.5 and level_advantage != 'enemy':
            return 'favorable'
        elif hp_ratio > 0.3:
            return 'contested'
        else:
            return 'disadvantaged'

    def _determine_optimal_strategy(self, situation: Dict[str, Any], momentum: float, type_advantage: str) -> str:
        """Determine optimal battle strategy"""
        strategic_position = self._assess_strategic_position({}, situation)

        if strategic_position == 'dominant':
            return 'aggressive'
        elif strategic_position == 'disadvantaged':
            return 'defensive'
        elif type_advantage in ['super_effective', 'not_very_effective']:
            return 'tactical'
        else:
            return 'balanced'

    def _calculate_urgency(self, hp_ratio: float, level_difference: int) -> int:
        """Calculate urgency level 1-5"""
        if hp_ratio < 0.15:
            return 5  # Critical
        elif hp_ratio < 0.3 and level_difference > 5:
            return 4  # High
        elif hp_ratio < 0.5:
            return 3  # Medium
        else:
            return 2  # Low

    def _non_battle_action(self, game_state: Dict[str, Any], info: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
        """Handle non-battle situations (fallback to exploration)"""
        return 5, {  # A button - interact/move forward
            'source': 'battle_agent_fallback',
            'reasoning': 'Not in battle - basic interaction',
            'confidence': 0.3
        }

    def _track_decision(self, decision: BattleDecision, game_state: Dict[str, Any]):
        """Track decision for learning and statistics"""
        battle_record = {
            'action': decision.action,
            'move_type': decision.move_type,
            'confidence': decision.confidence,
            'hp_ratio': game_state.get('player_hp', 0) / max(game_state.get('player_max_hp', 1), 1),
            'level_diff': game_state.get('enemy_level', 0) - game_state.get('player_level', 0)
        }

        self.battle_history.append(battle_record)

        # Keep only recent history
        if len(self.battle_history) > 100:
            self.battle_history.pop(0)

    def update(self, reward: float) -> None:
        """Update agent with battle-specific learning"""
        super().update(reward)

        # Track battle outcomes
        if len(self.battle_history) > 0:
            last_decision = self.battle_history[-1]

            # Learn from battle results
            if reward > 0.5:  # Good outcome
                self._reinforce_successful_pattern(last_decision)
            elif reward < -0.5:  # Poor outcome
                self._learn_from_failure(last_decision)

    def _reinforce_successful_pattern(self, decision: Dict[str, Any]):
        """Reinforce successful battle patterns"""
        pattern_key = f"{decision['move_type']}_{decision['hp_ratio']:.1f}"
        self.successful_strategies[pattern_key] = self.successful_strategies.get(pattern_key, 0) + 1

    def _learn_from_failure(self, decision: Dict[str, Any]):
        """Learn from failed battle decisions"""
        pattern_key = f"{decision['move_type']}_{decision['hp_ratio']:.1f}"
        self.failed_strategies[pattern_key] = self.failed_strategies.get(pattern_key, 0) + 1

    def get_stats(self) -> Dict[str, Any]:
        """Get battle agent statistics"""
        base_stats = super().get_stats()

        battle_stats = {
            'battles_fought': self.battles_fought,
            'battles_won': self.battles_won,
            'win_rate': self.battle_win_rate,
            'aggression_level': self.aggression_level,
            'risk_tolerance': self.risk_tolerance,
            'decision_history_size': len(self.battle_history),
            'successful_patterns': len(self.successful_strategies),
            'failed_patterns': len(self.failed_strategies)
        }

        return {**base_stats, **battle_stats}

    def get_subscribed_events(self) -> set:
        """Return set of event types this subscriber is interested in"""
        return {
            EventType.BATTLE_STARTED,
            EventType.BATTLE_ENDED,
            EventType.HP_CRITICAL,
            EventType.GAME_STATE_CHANGED
        }

    def handle_event(self, event: Event) -> None:
        """Handle incoming events from the event system"""
        try:
            if event.event_type == EventType.BATTLE_STARTED:
                self._handle_battle_started_event(event)
            elif event.event_type == EventType.BATTLE_ENDED:
                self._handle_battle_ended_event(event)
            elif event.event_type == EventType.HP_CRITICAL:
                self._handle_hp_critical_event(event)
            elif event.event_type == EventType.GAME_STATE_CHANGED:
                self._handle_game_state_change_event(event)
        except Exception as e:
            self.logger.error(f"Error handling event {event.event_type.value}: {e}")

    def _handle_battle_started_event(self, event: Event) -> None:
        """Handle battle started event"""
        enemy_level = event.data.get('enemy_level', 0)
        self.logger.info(f"Battle started against level {enemy_level} enemy")

        # Update battle tracking
        self.battles_fought += 1

        # Reset battle-specific state
        self.current_battle_start_time = event.timestamp
        self.current_battle_context = event.data
        self.current_defensive_mode = False

    def _handle_battle_ended_event(self, event: Event) -> None:
        """Handle battle ended event"""
        player_won = event.data.get('player_won', False)

        if player_won:
            self.battles_won += 1
            self.logger.info("Battle won! Updating battle strategy.")
        else:
            self.logger.info("Battle lost. Analyzing for improvements.")

        # Update win rate
        if self.battles_fought > 0:
            self.battle_win_rate = self.battles_won / self.battles_fought

        # Publish battle performance update
        self._publish_battle_performance_event(player_won)

    def _handle_hp_critical_event(self, event: Event) -> None:
        """Handle critical HP event"""
        hp_ratio = event.data.get('hp_ratio', 0.0)
        self.logger.warning(f"Critical HP detected: {hp_ratio:.2%}")

        # Adjust battle strategy to be more defensive
        self.current_defensive_mode = True

    def _handle_game_state_change_event(self, event: Event) -> None:
        """Handle general game state changes"""
        changes = event.data.get('changes', {})

        # Track if battle state changed
        if 'battle_started' in changes:
            self.current_battle_context = changes['battle_started']
        elif 'battle_ended' in changes:
            self.current_battle_context = None
            self.current_defensive_mode = False

    def _publish_battle_performance_event(self, won: bool) -> None:
        """Publish battle performance update event"""
        import time

        event = Event(
            event_type=EventType.AGENT_PERFORMANCE_UPDATE,
            timestamp=time.time(),
            source="battle_agent",
            data={
                'agent_type': 'battle',
                'battle_result': 'won' if won else 'lost',
                'battles_fought': self.battles_fought,
                'battles_won': self.battles_won,
                'win_rate': self.battle_win_rate,
                'performance_change': 'improved' if won else 'declined'
            },
            priority=5
        )

        self.event_bus.publish(event)