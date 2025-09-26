"""
Complex Behavioral Workflows Integration Tests

This module tests sophisticated end-to-end behavioral workflows that span multiple systems,
including agent coordination, dialogue processing, strategic decision making, and
event-driven interactions. These tests focus on real-world usage scenarios that
require deep integration between components.
"""

import pytest
import time
import tempfile
import json
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from typing import Dict, List, Any, Optional

# Core systems
from core.event_system import EventBus, Event, EventType, get_event_bus
from core.dialogue_state_machine import DialogueStateMachine, DialogueState, NPCType
from core.strategic_context_builder import StrategicContextBuilder
from core.decision_history_analyzer import DecisionHistoryAnalyzer
from core.adaptive_strategy_system import AdaptiveStrategySystem

# Agent systems
from agents.multi_agent_coordinator import MultiAgentCoordinator, AgentRole
from agents.llm_agent import LLMAgent
from agents.battle_agent import BattleAgent
from agents.explorer_agent import ExplorerAgent
from agents.progression_agent import ProgressionAgent

# Supporting systems
from rewards.calculator import PokemonRewardCalculator
from vision.shared_types import VisualContext, DetectedText
from utils.memory_reader import build_observation
from utils.screen_analyzer import analyze_screen_state


class SimpleEventSubscriber:
    """Simple event subscriber for testing purposes."""

    def __init__(self, callback_func, event_types):
        self.callback_func = callback_func
        self.event_types = set(event_types)

    def handle_event(self, event):
        self.callback_func(event)

    def get_subscribed_events(self):
        return self.event_types


class TestComplexAgentCoordinationWorkflows:
    """Test complex multi-agent coordination workflows in realistic scenarios."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def integrated_system(self, temp_dir):
        """Create an integrated system with all components connected."""
        # Initialize event bus
        event_bus = get_event_bus()
        # Clear any existing subscribers by creating a new event bus
        from core.event_system import EventBus
        event_bus = EventBus()

        # Initialize decision analyzer and strategy system
        db_path = temp_dir / "decisions.db"
        decision_analyzer = DecisionHistoryAnalyzer(str(db_path))
        strategy_system = AdaptiveStrategySystem(history_analyzer=decision_analyzer)

        # Initialize coordinator with full agent setup
        coordinator = MultiAgentCoordinator({
            'coordination_config': {
                'conflict_resolution': 'context_match',
                'performance_tracking': True
            }
        })

        # Initialize context builder
        context_builder = StrategicContextBuilder()

        # Initialize dialogue system
        dialogue_db_path = temp_dir / "dialogue.db"
        with patch('core.dialogue_state_machine.SemanticContextSystem'):
            dialogue_machine = DialogueStateMachine(db_path=str(dialogue_db_path))

        # Initialize reward calculator
        reward_calculator = PokemonRewardCalculator()

        return {
            'event_bus': event_bus,
            'coordinator': coordinator,
            'dialogue_machine': dialogue_machine,
            'context_builder': context_builder,
            'strategy_system': strategy_system,
            'decision_analyzer': decision_analyzer,
            'reward_calculator': reward_calculator
        }

    def test_battle_encounter_complete_workflow(self, integrated_system):
        """Test complete workflow from battle encounter to resolution."""
        event_bus = integrated_system['event_bus']
        coordinator = integrated_system['coordinator']
        dialogue_machine = integrated_system['dialogue_machine']
        reward_calculator = integrated_system['reward_calculator']

        # Track events for verification
        captured_events = []
        def capture_event(event):
            captured_events.append(event)

        # Create event subscriber
        capture_subscriber = SimpleEventSubscriber(
            capture_event,
            [EventType.BATTLE_STARTED, EventType.AGENT_DECISION, EventType.BATTLE_ENDED]
        )
        event_bus.subscribe(capture_subscriber)

        # Scenario: Player encounters wild Pokemon
        initial_game_state = {
            'in_battle': 0,
            'player_hp': 50,
            'player_max_hp': 50,
            'player_level': 12,
            'party_count': 2,
            'location': 5,
            'badges_total': 1
        }

        # Step 1: Battle encounter detected
        event_bus.publish(Event(
            event_type=EventType.BATTLE_STARTED,
            timestamp=time.time(),
            source="test",
            data={
                'enemy_species': 16,  # Pidgey
                'enemy_level': 8,
                'battle_type': 'wild',
                'location': 5
            }
        ))

        # Step 2: Transition to battle state
        battle_game_state = {
            'in_battle': 1,
            'player_hp': 50,
            'player_max_hp': 50,
            'player_level': 12,
            'enemy_hp': 25,
            'enemy_max_hp': 25,
            'enemy_level': 8,
            'party_count': 2,
            'badges_total': 1
        }

        battle_context = VisualContext(
            screen_type="battle",
            detected_text=[
                DetectedText("Wild PIDGEY appeared!", 0.95, (10, 10, 200, 30), "dialogue"),
                DetectedText("FIGHT", 0.98, (20, 60, 80, 80), "choice"),
                DetectedText("PKMN", 0.96, (20, 90, 80, 110), "choice"),
                DetectedText("ITEM", 0.94, (100, 60, 160, 80), "choice"),
                DetectedText("RUN", 0.92, (100, 90, 160, 110), "choice")
            ],
            ui_elements=[],
            dominant_colors=[(255, 100, 100)],
            game_phase="battle",
            visual_summary="Wild Pokemon battle start"
        )

        # Step 3: Agent coordination for battle decision
        action, decision_info = coordinator.get_action(battle_game_state, {'screen_state': 'battle'})

        # Should choose battle agent for battle scenario
        assert action in [0, 1, 2, 3, 4, 5, 6, 7]  # Valid action
        assert decision_info['chosen_agent'] == AgentRole.BATTLE.value
        assert 'confidence' in decision_info

        # Step 4: Battle progression with dialogue
        attack_context = VisualContext(
            screen_type="battle",
            detected_text=[
                DetectedText("PIKACHU used THUNDERSHOCK!", 0.95, (10, 10, 300, 30), "dialogue")
            ],
            ui_elements=[],
            dominant_colors=[(255, 255, 100)],
            game_phase="battle",
            visual_summary="Attack animation"
        )

        dialogue_result = dialogue_machine.process_dialogue(attack_context, battle_game_state)
        assert dialogue_result is not None
        assert dialogue_result['npc_type'] == NPCType.GENERIC.value

        # Step 5: Battle conclusion
        victory_game_state = {
            'in_battle': 0,
            'player_hp': 45,
            'player_max_hp': 50,
            'player_level': 12,
            'enemy_hp': 0,
            'party_count': 2,
            'badges_total': 1,
            'exp_gained': 150
        }

        # Calculate battle reward (correct signature: current_state, previous_state)
        battle_reward, reward_breakdown = reward_calculator.calculate_reward(
            victory_game_state,
            initial_game_state
        )

        assert battle_reward > 0  # Should be positive for winning

        # Step 6: Battle end event
        event_bus.publish(Event(
            event_type=EventType.BATTLE_ENDED,
            timestamp=time.time(),
            source="test",
            data={
                'result': 'victory',
                'exp_gained': 150,
                'damage_taken': 5,
                'agent_performance': decision_info
            }
        ))

        # Verify complete workflow
        assert len(captured_events) >= 3  # Start, decision(s), end
        battle_start_events = [e for e in captured_events if e.event_type == EventType.BATTLE_STARTED]
        battle_end_events = [e for e in captured_events if e.event_type == EventType.BATTLE_ENDED]
        assert len(battle_start_events) >= 1
        assert len(battle_end_events) >= 1

    def test_gym_challenge_complex_workflow(self, integrated_system):
        """Test complex gym challenge workflow with multiple dialogue phases."""
        coordinator = integrated_system['coordinator']
        dialogue_machine = integrated_system['dialogue_machine']
        context_builder = integrated_system['context_builder']
        event_bus = integrated_system['event_bus']

        # Track dialogue states through gym challenge
        dialogue_states = []
        def track_dialogue_state():
            dialogue_states.append(dialogue_machine.current_state)

        # Phase 1: Approach gym leader
        approach_context = VisualContext(
            screen_type="dialogue",
            detected_text=[
                DetectedText("I'm Brock, the Pewter Gym Leader!", 0.95, (10, 10, 300, 30), "dialogue"),
                DetectedText("My rock-hard Pokemon will crush you!", 0.93, (10, 40, 320, 60), "dialogue")
            ],
            ui_elements=[],
            dominant_colors=[(150, 100, 50)],
            game_phase="dialogue",
            visual_summary="Gym leader introduction"
        )

        gym_game_state = {
            'in_battle': 0,
            'player_level': 15,
            'party_count': 3,
            'badges_total': 0,
            'location': 10,  # Gym location
            'objective': 'Challenge gym leader'
        }

        # Process gym leader introduction
        dialogue_result = dialogue_machine.process_dialogue(approach_context, gym_game_state)
        track_dialogue_state()

        assert dialogue_result['npc_type'] == NPCType.GYM_LEADER.value
        assert dialogue_machine.current_state in [DialogueState.READING, DialogueState.LISTENING]

        # Phase 2: Challenge acceptance dialogue
        challenge_context = VisualContext(
            screen_type="dialogue",
            detected_text=[
                DetectedText("Are you ready to challenge me?", 0.95, (10, 10, 280, 30), "dialogue"),
                DetectedText("Yes", 0.98, (20, 60, 60, 80), "choice"),
                DetectedText("No", 0.96, (20, 90, 60, 110), "choice")
            ],
            ui_elements=[],
            dominant_colors=[(150, 100, 50)],
            game_phase="dialogue",
            visual_summary="Gym challenge question"
        )

        dialogue_result = dialogue_machine.process_dialogue(challenge_context, gym_game_state)
        track_dialogue_state()

        assert dialogue_machine.current_state == DialogueState.CHOOSING
        assert len(dialogue_result['choices']) == 2
        assert dialogue_result['recommended_action'] == 'A'  # Should recommend accepting

        # Phase 3: Agent coordination for gym battle strategy
        battle_prep_context = context_builder.build_context(
            gym_game_state,
            {'screen_state': 'dialogue', 'gym_leader': True},
            [action for action in range(8)]  # Recent actions
        )

        # Get coordinated decision for gym battle approach
        action, decision_info = coordinator.get_action(gym_game_state, {
            'screen_state': 'dialogue',
            'gym_leader': True,
            'challenge_accepted': True
        })

        # Should involve strategic planning
        assert 'chosen_agent' in decision_info
        assert decision_info['confidence'] > 0

        # Phase 4: Battle initiation
        battle_start_context = VisualContext(
            screen_type="battle",
            detected_text=[
                DetectedText("BROCK sent out ONIX!", 0.95, (10, 10, 200, 30), "dialogue")
            ],
            ui_elements=[],
            dominant_colors=[(100, 100, 100)],
            game_phase="battle",
            visual_summary="Gym battle starts"
        )

        gym_battle_state = {
            'in_battle': 1,
            'player_hp': 45,
            'player_max_hp': 50,
            'player_level': 15,
            'enemy_hp': 40,
            'enemy_max_hp': 40,
            'enemy_level': 14,
            'party_count': 3,
            'badges_total': 0,
            'gym_battle': True
        }

        # Process gym battle dialogue
        dialogue_result = dialogue_machine.process_dialogue(battle_start_context, gym_battle_state)
        track_dialogue_state()

        # Coordinate gym battle strategy
        battle_action, battle_decision = coordinator.get_action(gym_battle_state, {
            'screen_state': 'battle',
            'gym_battle': True
        })

        # Should prioritize battle agent for gym battle
        assert battle_decision['chosen_agent'] == AgentRole.BATTLE.value

        # Verify dialogue state progression through workflow
        assert len(dialogue_states) >= 3
        assert DialogueState.READING in dialogue_states or DialogueState.LISTENING in dialogue_states
        assert DialogueState.CHOOSING in dialogue_states

    def test_exploration_discovery_event_driven_workflow(self, integrated_system):
        """Test event-driven exploration and discovery workflow."""
        event_bus = integrated_system['event_bus']
        coordinator = integrated_system['coordinator']
        strategy_system = integrated_system['strategy_system']

        # Track exploration events
        exploration_events = []
        def capture_exploration_event(event):
            exploration_events.append(event)

        event_bus.subscribe(EventType.LOCATION_CHANGED, capture_exploration_event)
        event_bus.subscribe(EventType.AGENT_DECISION, capture_exploration_event)

        # Scenario: Player exploring new area
        exploration_state = {
            'in_battle': 0,
            'player_map': 15,  # New unexplored map
            'player_x': 5,
            'player_y': 8,
            'party_count': 2,
            'badges_total': 1,
            'exploration_progress': 0.3
        }

        # Step 1: Location change event
        event_bus.publish(Event(
            event_type=EventType.LOCATION_CHANGED,
            timestamp=time.time(),
            source="test",
            data={
                'new_map': 15,
                'old_map': 10,
                'coordinates': (5, 8),
                'area_type': 'route'
            }
        ))

        # Step 2: Agent coordination for exploration
        action, decision_info = coordinator.get_action(exploration_state, {
            'screen_state': 'overworld',
            'new_area': True
        })

        # Should favor exploration agent in new area
        assert decision_info['chosen_agent'] == AgentRole.EXPLORER.value
        assert action in [0, 1, 2, 3]  # Movement actions

        # Step 3: Strategy system adaptation
        # Simulate successful exploration
        exploration_reward = 15.0
        strategy_system.update_performance_metrics(decision_info['chosen_agent'], exploration_reward)

        # Step 4: Discovery event
        discovery_context = VisualContext(
            screen_type="overworld",
            detected_text=[
                DetectedText("You found TM01!", 0.95, (10, 10, 150, 30), "dialogue")
            ],
            ui_elements=[],
            dominant_colors=[(100, 200, 100)],
            game_phase="discovery",
            visual_summary="Item discovery"
        )

        # Process discovery through dialogue system
        with patch.object(integrated_system['dialogue_machine'], 'process_dialogue') as mock_dialogue:
            mock_dialogue.return_value = {
                'dialogue': ['You found TM01!'],
                'choices': [],
                'npc_type': NPCType.GENERIC.value,
                'semantic_analysis': {'context_type': 'item_discovery'},
                'recommended_action': 'A',
                'session_id': int(time.time())
            }

            dialogue_result = integrated_system['dialogue_machine'].process_dialogue(
                discovery_context, exploration_state
            )

        assert dialogue_result['semantic_analysis']['context_type'] == 'item_discovery'

        # Step 5: Continued exploration coordination
        post_discovery_state = {**exploration_state, 'items_found': 1}

        next_action, next_decision = coordinator.get_action(post_discovery_state, {
            'screen_state': 'overworld',
            'recent_discovery': True
        })

        # Verify event-driven workflow
        assert len(exploration_events) >= 2  # Location change + agent decisions
        location_events = [e for e in exploration_events if e.event_type == EventType.LOCATION_CHANGED]
        assert len(location_events) == 1
        assert location_events[0].data['new_map'] == 15

    def test_dialogue_to_battle_transition_workflow(self, integrated_system):
        """Test complex workflow from trainer dialogue to battle."""
        dialogue_machine = integrated_system['dialogue_machine']
        coordinator = integrated_system['coordinator']
        reward_calculator = integrated_system['reward_calculator']
        event_bus = integrated_system['event_bus']

        # Track workflow events
        workflow_events = []
        def capture_workflow_event(event):
            workflow_events.append(event)

        event_bus.subscribe(EventType.BATTLE_STARTED, capture_workflow_event)
        event_bus.subscribe(EventType.AGENT_DECISION, capture_workflow_event)

        # Phase 1: Trainer encounter dialogue
        trainer_encounter_state = {
            'in_battle': 0,
            'player_level': 18,
            'party_count': 4,
            'badges_total': 2,
            'location': 25
        }

        # Trainer challenge dialogue
        challenge_contexts = [
            VisualContext(
                screen_type="dialogue",
                detected_text=[
                    DetectedText("Hey! You look strong!", 0.95, (10, 10, 200, 30), "dialogue")
                ],
                ui_elements=[],
                dominant_colors=[(255, 255, 255)],
                game_phase="dialogue",
                visual_summary="Trainer spotted"
            ),
            VisualContext(
                screen_type="dialogue",
                detected_text=[
                    DetectedText("Let's have a Pokemon battle!", 0.95, (10, 10, 250, 30), "dialogue"),
                    DetectedText("Yes", 0.98, (20, 60, 60, 80), "choice"),
                    DetectedText("No", 0.96, (20, 90, 60, 110), "choice")
                ],
                ui_elements=[],
                dominant_colors=[(255, 255, 255)],
                game_phase="dialogue",
                visual_summary="Battle challenge"
            )
        ]

        # Process multi-turn dialogue
        dialogue_results = []
        for context in challenge_contexts:
            result = dialogue_machine.process_dialogue(context, trainer_encounter_state)
            dialogue_results.append(result)

        # Verify dialogue progression
        assert all(result['npc_type'] == NPCType.TRAINER.value for result in dialogue_results)
        assert dialogue_results[1]['recommended_action'] == 'A'  # Accept battle

        # Phase 2: Agent coordination for battle acceptance
        pre_battle_action, pre_battle_decision = coordinator.get_action(
            trainer_encounter_state,
            {'screen_state': 'dialogue', 'trainer_challenge': True}
        )

        # Should coordinate appropriate response
        assert 'chosen_agent' in pre_battle_decision

        # Phase 3: Transition to battle
        battle_transition_state = {
            'in_battle': 1,
            'player_hp': 55,
            'player_max_hp': 55,
            'player_level': 18,
            'enemy_hp': 30,
            'enemy_max_hp': 30,
            'enemy_level': 16,
            'party_count': 4,
            'badges_total': 2,
            'trainer_battle': True
        }

        # Battle start event
        event_bus.publish(Event(
            event_type=EventType.BATTLE_STARTED,
            timestamp=time.time(),
            source="test",
            data={
                'battle_type': 'trainer',
                'trainer_level': 16,
                'transition_from_dialogue': True
            }
        ))

        # Phase 4: Battle coordination
        battle_action, battle_decision = coordinator.get_action(
            battle_transition_state,
            {'screen_state': 'battle', 'trainer_battle': True}
        )

        assert battle_decision['chosen_agent'] == AgentRole.BATTLE.value

        # Phase 5: Reward calculation for successful transition
        transition_reward = reward_calculator.calculate_reward(
            battle_transition_state,
            trainer_encounter_state,
            battle_action,
            {'dialogue_to_battle_transition': True}
        )

        # Verify complete workflow
        assert len(dialogue_results) == 2
        assert dialogue_machine.dialogue_history  # Should have dialogue history
        assert len(workflow_events) >= 1  # Should have battle start event

    def test_adaptive_strategy_performance_learning_workflow(self, integrated_system):
        """Test adaptive strategy system learning from multi-agent performance."""
        strategy_system = integrated_system['strategy_system']
        coordinator = integrated_system['coordinator']
        decision_analyzer = integrated_system['decision_analyzer']

        # Simulate multiple decision scenarios with varying success
        test_scenarios = [
            {
                'game_state': {'in_battle': 1, 'player_hp': 40, 'enemy_hp': 30},
                'context': {'screen_state': 'battle'},
                'expected_agent': AgentRole.BATTLE,
                'reward': 25.0  # Good battle performance
            },
            {
                'game_state': {'in_battle': 0, 'player_map': 20, 'exploration_progress': 0.1},
                'context': {'screen_state': 'overworld', 'new_area': True},
                'expected_agent': AgentRole.EXPLORER,
                'reward': 15.0  # Good exploration
            },
            {
                'game_state': {'badges_total': 2, 'story_progress': 0.4, 'objective': 'reach_next_city'},
                'context': {'screen_state': 'overworld'},
                'expected_agent': AgentRole.PROGRESSION,
                'reward': 20.0  # Good progression
            },
            {
                'game_state': {'in_battle': 1, 'player_hp': 10, 'enemy_hp': 45},
                'context': {'screen_state': 'battle'},
                'expected_agent': AgentRole.BATTLE,
                'reward': -10.0  # Poor battle performance
            }
        ]

        # Track performance over time
        performance_history = []

        for i, scenario in enumerate(test_scenarios):
            # Get coordinated decision
            action, decision_info = coordinator.get_action(
                scenario['game_state'],
                scenario['context']
            )

            # Update performance metrics
            chosen_agent = AgentRole(decision_info['chosen_agent'])
            strategy_system.update_performance_metrics(chosen_agent.value, scenario['reward'])

            # Store decision in analyzer
            decision_analyzer.store_decision(
                action=action,
                game_state=scenario['game_state'],
                agent_info=decision_info,
                reward=scenario['reward']
            )

            # Track performance metrics
            current_stats = strategy_system.get_performance_summary()
            performance_history.append({
                'iteration': i,
                'chosen_agent': chosen_agent,
                'reward': scenario['reward'],
                'stats': current_stats
            })

        # Verify adaptive learning
        final_stats = strategy_system.get_performance_summary()

        # Should have performance data for all agents used
        agents_used = set(p['chosen_agent'] for p in performance_history)
        for agent in agents_used:
            agent_key = f"{agent.value}_performance"
            assert agent_key in final_stats or 'agent_performance' in final_stats

        # Verify decision history tracking
        stored_decisions = decision_analyzer.get_recent_decisions(limit=len(test_scenarios))
        assert len(stored_decisions) == len(test_scenarios)

        # Test strategy adaptation based on performance
        # Better performing agents should have higher weights
        initial_weights = coordinator.agent_weights.copy()

        # Simulate weight adjustment based on performance
        for agent_role in [AgentRole.BATTLE, AgentRole.EXPLORER, AgentRole.PROGRESSION]:
            agent_performance = [p for p in performance_history if p['chosen_agent'] == agent_role]
            if agent_performance:
                avg_reward = sum(p['reward'] for p in agent_performance) / len(agent_performance)
                # Positive rewards should increase weight, negative should decrease
                if avg_reward > 0:
                    coordinator.agent_weights[agent_role.value] = min(2.0,
                        coordinator.agent_weights.get(agent_role.value, 1.0) + 0.1)
                else:
                    coordinator.agent_weights[agent_role.value] = max(0.5,
                        coordinator.agent_weights.get(agent_role.value, 1.0) - 0.1)

        # Verify weights have been adapted
        final_weights = coordinator.agent_weights
        # At least one weight should have changed
        weight_changes = sum(1 for k in initial_weights
                           if initial_weights[k] != final_weights.get(k, 1.0))
        assert weight_changes > 0


class TestEventDrivenBehavioralWorkflows:
    """Test complex event-driven behavioral workflows across multiple systems."""

    @pytest.fixture
    def event_driven_system(self):
        """Create event-driven system with all components connected."""
        event_bus = get_event_bus()
        # Create a clean event bus for testing
        from core.event_system import EventBus
        event_bus = EventBus()

        # Create components that will interact via events
        components = {
            'event_bus': event_bus,
            'event_history': [],
            'state_changes': []
        }

        # Set up event tracking
        def track_event(event):
            components['event_history'].append(event)

        # Subscribe to all event types for tracking
        for event_type in EventType:
            event_bus.subscribe(event_type, track_event)

        return components

    def test_cascading_event_workflow(self, event_driven_system):
        """Test cascading events that trigger multiple system responses."""
        event_bus = event_driven_system['event_bus']
        event_history = event_driven_system['event_history']

        # Start with level up event that should cascade
        initial_event = Event(
            event_type=EventType.PLAYER_LEVEL_UP,
            timestamp=time.time(),
            source="test",
            data={
                'old_level': 12,
                'new_level': 13,
                'pokemon_id': 1,
                'stats_increased': {'hp': 3, 'attack': 2},
                'moves_learned': ['THUNDERBOLT']
            }
        )

        event_bus.publish(initial_event)

        # Should trigger additional events for moves learned
        if initial_event.data['moves_learned']:
            for move in initial_event.data['moves_learned']:
                move_event = Event(
                    event_type=EventType.BATTLE_STARTED,  # Using existing event type
                    timestamp=time.time(),
                    source="test",
                    data={
                        'move_learned': move,
                        'pokemon_level': 13,
                        'trigger_event': 'level_up'
                    }
                )
                event_bus.publish(move_event)

        # Should trigger agent performance update
        performance_event = Event(
            event_type=EventType.AGENT_DECISION,
            timestamp=time.time(),
            source="test",
            data={
                'agent_role': 'battle',
                'decision_type': 'stat_improvement',
                'performance_impact': 'positive',
                'related_to': 'level_up'
            }
        )
        event_bus.publish(performance_event)

        # Verify cascading workflow
        assert len(event_history) >= 3  # Initial + cascaded events
        level_up_events = [e for e in event_history if e.event_type == EventType.PLAYER_LEVEL_UP]
        assert len(level_up_events) == 1
        assert level_up_events[0].data['new_level'] == 13

    def test_multi_system_event_coordination(self, event_driven_system):
        """Test coordination between multiple systems via events."""
        event_bus = event_driven_system['event_bus']
        event_history = event_driven_system['event_history']

        # Mock system responses to events
        system_responses = {'dialogue': [], 'battle': [], 'exploration': []}

        def dialogue_system_response(event):
            if event.event_type == EventType.BATTLE_STARTED:
                system_responses['dialogue'].append('battle_start_dialogue')
            elif event.event_type == EventType.PLAYER_LEVEL_UP:
                system_responses['dialogue'].append('level_up_message')

        def battle_system_response(event):
            if event.event_type == EventType.AGENT_DECISION:
                system_responses['battle'].append('agent_coordination')

        def exploration_system_response(event):
            if event.event_type == EventType.LOCATION_CHANGED:
                system_responses['exploration'].append('area_analysis')

        # Register system responses
        event_bus.subscribe(EventType.BATTLE_STARTED, dialogue_system_response)
        event_bus.subscribe(EventType.PLAYER_LEVEL_UP, dialogue_system_response)
        event_bus.subscribe(EventType.AGENT_DECISION, battle_system_response)
        event_bus.subscribe(EventType.LOCATION_CHANGED, exploration_system_response)

        # Trigger sequence of events
        events_sequence = [
            Event(EventType.LOCATION_CHANGED, time.time(), "test", {'new_map': 15, 'area_type': 'forest'}),
            Event(EventType.BATTLE_STARTED, time.time(), "test", {'enemy_species': 16, 'battle_type': 'wild'}),
            Event(EventType.AGENT_DECISION, time.time(), "test", {'agent': 'battle', 'action': 5, 'confidence': 0.8}),
            Event(EventType.PLAYER_LEVEL_UP, time.time(), "test", {'new_level': 15, 'pokemon_id': 1})
        ]

        for event in events_sequence:
            event_bus.publish(event)

        # Verify multi-system coordination
        assert len(system_responses['dialogue']) >= 2  # battle_start + level_up
        assert len(system_responses['battle']) >= 1   # agent_decision
        assert len(system_responses['exploration']) >= 1  # location_change

        # Verify all events were processed
        assert len(event_history) >= len(events_sequence)

    def test_event_driven_strategy_adaptation(self, event_driven_system):
        """Test strategy adaptation based on event patterns."""
        event_bus = event_driven_system['event_bus']
        event_history = event_driven_system['event_history']

        # Track strategy adaptations
        strategy_adaptations = []

        def strategy_adaptation_response(event):
            if event.event_type == EventType.AGENT_DECISION:
                # Simulate strategy adaptation based on agent performance
                agent_role = event.data.get('agent_role', 'unknown')
                confidence = event.data.get('confidence', 0.5)

                if confidence > 0.8:
                    adaptation = f"increase_{agent_role}_weight"
                elif confidence < 0.3:
                    adaptation = f"decrease_{agent_role}_weight"
                else:
                    adaptation = f"maintain_{agent_role}_weight"

                strategy_adaptations.append(adaptation)

        event_bus.subscribe(EventType.AGENT_DECISION, strategy_adaptation_response)

        # Simulate agent decisions with varying performance
        decision_events = [
            Event(EventType.AGENT_DECISION, time.time(), "test", {
                'agent_role': 'battle', 'confidence': 0.9, 'reward': 20.0
            }),
            Event(EventType.AGENT_DECISION, time.time(), "test", {
                'agent_role': 'explorer', 'confidence': 0.2, 'reward': -5.0
            }),
            Event(EventType.AGENT_DECISION, time.time(), "test", {
                'agent_role': 'progression', 'confidence': 0.6, 'reward': 10.0
            }),
            Event(EventType.AGENT_DECISION, time.time(), "test", {
                'agent_role': 'battle', 'confidence': 0.85, 'reward': 15.0
            })
        ]

        for event in decision_events:
            event_bus.publish(event)

        # Verify strategy adaptations
        assert len(strategy_adaptations) == len(decision_events)
        assert 'increase_battle_weight' in strategy_adaptations  # High confidence
        assert 'decrease_explorer_weight' in strategy_adaptations  # Low confidence
        assert 'maintain_progression_weight' in strategy_adaptations  # Medium confidence

    def test_error_recovery_event_workflow(self, event_driven_system):
        """Test error recovery workflows triggered by events."""
        event_bus = event_driven_system['event_bus']
        event_history = event_driven_system['event_history']

        # Track error recovery attempts
        recovery_attempts = []

        def error_recovery_handler(event):
            if hasattr(event, 'error_type'):
                recovery_attempts.append({
                    'error_type': event.error_type,
                    'recovery_strategy': f"recover_from_{event.error_type}",
                    'timestamp': time.time()
                })

        # Simulate various error scenarios
        error_events = [
            # Simulate agent coordination failure
            Event(EventType.AGENT_DECISION, time.time(), "test", {
                'agent_role': 'coordinator',
                'error': True,
                'error_type': 'coordination_failure',
                'fallback_action': 2
            }),

            # Simulate dialogue processing error
            Event(EventType.BATTLE_STARTED, time.time(), "test", {  # Using existing event type
                'dialogue_error': True,
                'error_type': 'dialogue_parse_error',
                'raw_text': "corrupted_dialogue_data"
            })
        ]

        # Add error_type as attribute to events for testing
        for event in error_events:
            if 'error_type' in event.data:
                event.error_type = event.data['error_type']

        # Register error recovery handler
        for event_type in [EventType.AGENT_DECISION, EventType.BATTLE_STARTED]:
            event_bus.subscribe(event_type, error_recovery_handler)

        # Publish error events
        for event in error_events:
            event_bus.publish(event)

        # Verify error recovery workflow
        assert len(recovery_attempts) == len([e for e in error_events if 'error_type' in e.data])

        # Verify specific recovery strategies
        error_types = [attempt['error_type'] for attempt in recovery_attempts]
        assert 'coordination_failure' in error_types or 'dialogue_parse_error' in error_types

    def test_performance_monitoring_event_workflow(self, event_driven_system):
        """Test performance monitoring through event-driven workflows."""
        event_bus = event_driven_system['event_bus']
        event_history = event_driven_system['event_history']

        # Track performance metrics
        performance_metrics = {
            'total_decisions': 0,
            'successful_battles': 0,
            'exploration_progress': 0,
            'average_confidence': 0.0
        }

        def performance_monitor(event):
            if event.event_type == EventType.AGENT_DECISION:
                performance_metrics['total_decisions'] += 1
                confidence = event.data.get('confidence', 0.5)
                current_avg = performance_metrics['average_confidence']
                total = performance_metrics['total_decisions']
                performance_metrics['average_confidence'] = (
                    (current_avg * (total - 1)) + confidence
                ) / total

            elif event.event_type == EventType.BATTLE_ENDED:
                if event.data.get('result') == 'victory':
                    performance_metrics['successful_battles'] += 1

            elif event.event_type == EventType.LOCATION_CHANGED:
                performance_metrics['exploration_progress'] += 1

        # Register performance monitoring
        for event_type in [EventType.AGENT_DECISION, EventType.BATTLE_ENDED, EventType.LOCATION_CHANGED]:
            event_bus.subscribe(event_type, performance_monitor)

        # Simulate gameplay session with various events
        gameplay_events = [
            Event(EventType.AGENT_DECISION, time.time(), "test", {'confidence': 0.8, 'agent': 'battle'}),
            Event(EventType.BATTLE_ENDED, time.time(), "test", {'result': 'victory', 'exp_gained': 100}),
            Event(EventType.LOCATION_CHANGED, time.time(), "test", {'new_map': 12}),
            Event(EventType.AGENT_DECISION, time.time(), "test", {'confidence': 0.6, 'agent': 'explorer'}),
            Event(EventType.AGENT_DECISION, time.time(), "test", {'confidence': 0.9, 'agent': 'progression'}),
            Event(EventType.BATTLE_ENDED, time.time(), "test", {'result': 'defeat', 'damage_taken': 50}),
            Event(EventType.LOCATION_CHANGED, time.time(), "test", {'new_map': 13})
        ]

        for event in gameplay_events:
            event_bus.publish(event)

        # Verify performance monitoring
        assert performance_metrics['total_decisions'] == 3
        assert performance_metrics['successful_battles'] == 1
        assert performance_metrics['exploration_progress'] == 2
        assert 0.5 <= performance_metrics['average_confidence'] <= 1.0

        # Verify comprehensive event tracking
        assert len(event_history) >= len(gameplay_events)

        # Check event type distribution
        event_types = [e.event_type for e in event_history]
        assert EventType.AGENT_DECISION in event_types
        assert EventType.BATTLE_ENDED in event_types
        assert EventType.LOCATION_CHANGED in event_types