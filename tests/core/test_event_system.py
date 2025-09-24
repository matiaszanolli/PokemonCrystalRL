"""
Tests for the Event System - Reactive Architecture

This module contains comprehensive tests for the event system including
event bus functionality, event subscribers, analytics, and state detection.
"""

import pytest
import time
from unittest.mock import Mock, patch
from typing import Dict, Any, Set

from core.event_system import (
    EventType, Event, EventBus, EventSubscriber, EventFilter,
    GameStateEventDetector, EventDrivenAnalytics,
    get_event_bus, initialize_event_system
)


class TestEventSubscriber(EventSubscriber):
    """Test event subscriber for testing purposes"""

    def __init__(self, subscribed_events: Set[EventType] = None):
        self.subscribed_events = subscribed_events or {EventType.GAME_STATE_CHANGED}
        self.received_events = []
        self.handle_event_called = False

    def get_subscribed_events(self) -> Set[EventType]:
        return self.subscribed_events

    def handle_event(self, event: Event) -> None:
        self.handle_event_called = True
        self.received_events.append(event)


class TestEvent:
    """Test Event class functionality"""

    def test_event_creation(self):
        """Test basic event creation"""
        event = Event(
            event_type=EventType.PLAYER_LEVEL_UP,
            timestamp=time.time(),
            source="test",
            data={'old_level': 5, 'new_level': 6}
        )

        assert event.event_type == EventType.PLAYER_LEVEL_UP
        assert event.source == "test"
        assert event.data['old_level'] == 5
        assert event.priority == 5  # Default priority
        assert event.event_id.startswith("player_level_up_")

    def test_event_auto_id_generation(self):
        """Test automatic event ID generation"""
        timestamp = time.time()
        event = Event(
            event_type=EventType.BADGE_EARNED,
            timestamp=timestamp,
            source="test"
        )

        expected_id = f"badge_earned_{int(timestamp * 1000)}"
        assert event.event_id == expected_id

    def test_event_custom_id(self):
        """Test custom event ID"""
        custom_id = "custom_test_id"
        event = Event(
            event_type=EventType.BATTLE_STARTED,
            timestamp=time.time(),
            source="test",
            event_id=custom_id
        )

        assert event.event_id == custom_id


class TestEventFilter:
    """Test EventFilter functionality"""

    def test_event_type_filter(self):
        """Test filtering by event type"""
        event_filter = EventFilter(event_types={EventType.BATTLE_STARTED, EventType.BATTLE_ENDED})

        battle_event = Event(EventType.BATTLE_STARTED, time.time(), "test")
        level_event = Event(EventType.PLAYER_LEVEL_UP, time.time(), "test")

        assert event_filter.matches(battle_event) is True
        assert event_filter.matches(level_event) is False

    def test_source_filter(self):
        """Test filtering by source"""
        event_filter = EventFilter(sources={"battle_agent", "trainer"})

        battle_event = Event(EventType.BATTLE_STARTED, time.time(), "battle_agent")
        other_event = Event(EventType.BATTLE_STARTED, time.time(), "explorer_agent")

        assert event_filter.matches(battle_event) is True
        assert event_filter.matches(other_event) is False

    def test_priority_filter(self):
        """Test filtering by minimum priority"""
        event_filter = EventFilter(min_priority=7)

        high_priority_event = Event(EventType.BADGE_EARNED, time.time(), "test", priority=8)
        low_priority_event = Event(EventType.AGENT_DECISION, time.time(), "test", priority=5)

        assert event_filter.matches(high_priority_event) is True
        assert event_filter.matches(low_priority_event) is False

    def test_custom_filter(self):
        """Test custom filter function"""
        def custom_filter(event: Event) -> bool:
            return event.data.get('level', 0) > 10

        event_filter = EventFilter(custom_filter=custom_filter)

        high_level_event = Event(EventType.PLAYER_LEVEL_UP, time.time(), "test", data={'level': 15})
        low_level_event = Event(EventType.PLAYER_LEVEL_UP, time.time(), "test", data={'level': 5})

        assert event_filter.matches(high_level_event) is True
        assert event_filter.matches(low_level_event) is False


class TestEventBus:
    """Test EventBus functionality"""

    def setup_method(self):
        """Setup for each test"""
        self.event_bus = EventBus()
        self.test_subscriber = TestEventSubscriber()

    def test_event_bus_initialization(self):
        """Test event bus initialization"""
        assert self.event_bus is not None
        assert len(self.event_bus.subscribers) == 0
        assert len(self.event_bus.event_history) == 0
        assert self.event_bus.max_history == 1000

    def test_subscribe_unsubscribe(self):
        """Test subscriber management"""
        # Subscribe
        self.event_bus.subscribe(self.test_subscriber)
        assert EventType.GAME_STATE_CHANGED in self.event_bus.subscribers
        assert self.test_subscriber in self.event_bus.subscribers[EventType.GAME_STATE_CHANGED]

        # Unsubscribe
        self.event_bus.unsubscribe(self.test_subscriber)
        assert self.test_subscriber not in self.event_bus.subscribers.get(EventType.GAME_STATE_CHANGED, [])

    def test_filtered_subscription(self):
        """Test subscription with filter"""
        event_filter = EventFilter(event_types={EventType.BATTLE_STARTED})
        self.event_bus.subscribe(self.test_subscriber, event_filter)

        assert len(self.event_bus.filtered_subscribers) == 1
        assert self.event_bus.filtered_subscribers[0][1] == self.test_subscriber

    def test_event_publishing(self):
        """Test basic event publishing"""
        self.event_bus.subscribe(self.test_subscriber)

        event = Event(EventType.GAME_STATE_CHANGED, time.time(), "test")
        self.event_bus.publish(event)

        assert self.test_subscriber.handle_event_called is True
        assert len(self.test_subscriber.received_events) == 1
        assert self.test_subscriber.received_events[0] == event

    def test_event_history_tracking(self):
        """Test event history functionality"""
        event = Event(EventType.PLAYER_LEVEL_UP, time.time(), "test")
        self.event_bus.publish(event)

        assert len(self.event_bus.event_history) == 1
        assert self.event_bus.event_history[0] == event

    def test_event_stats_tracking(self):
        """Test event statistics tracking"""
        event1 = Event(EventType.BATTLE_STARTED, time.time(), "test")
        event2 = Event(EventType.BATTLE_STARTED, time.time(), "test")
        event3 = Event(EventType.PLAYER_LEVEL_UP, time.time(), "test")

        self.event_bus.publish(event1)
        self.event_bus.publish(event2)
        self.event_bus.publish(event3)

        assert self.event_bus.event_stats[EventType.BATTLE_STARTED] == 2
        assert self.event_bus.event_stats[EventType.PLAYER_LEVEL_UP] == 1

    def test_correlation_tracking(self):
        """Test event correlation tracking"""
        correlation_id = "test_correlation_123"
        event1 = Event(EventType.BATTLE_STARTED, time.time(), "test", correlation_id=correlation_id)
        event2 = Event(EventType.BATTLE_ENDED, time.time(), "test", correlation_id=correlation_id)

        self.event_bus.publish(event1)
        self.event_bus.publish(event2)

        assert correlation_id in self.event_bus.correlation_tracker
        assert len(self.event_bus.correlation_tracker[correlation_id]) == 2

    def test_error_handling(self):
        """Test error handling in event processing"""
        # Create a subscriber that raises an exception
        class FailingSubscriber(EventSubscriber):
            def get_subscribed_events(self) -> Set[EventType]:
                return {EventType.GAME_STATE_CHANGED}

            def handle_event(self, event: Event) -> None:
                raise ValueError("Test error")

        failing_subscriber = FailingSubscriber()
        self.event_bus.subscribe(failing_subscriber)

        # Add error handler to track errors
        errors = []
        def error_handler(event: Event, error: Exception):
            errors.append((event, error))

        self.event_bus.add_error_handler(error_handler)

        # Publish event that will cause error
        event = Event(EventType.GAME_STATE_CHANGED, time.time(), "test")
        self.event_bus.publish(event)

        # Check that error was handled
        assert len(errors) == 1
        assert isinstance(errors[0][1], ValueError)

    def test_game_state_change_publishing(self):
        """Test game state change event publishing"""
        old_state = {'player_level': 5, 'badges_total': 0}
        new_state = {'player_level': 6, 'badges_total': 0}

        # Subscribe to level up events
        level_subscriber = TestEventSubscriber({EventType.PLAYER_LEVEL_UP})
        self.event_bus.subscribe(level_subscriber)

        self.event_bus.publish_game_state_change(old_state, new_state, "test_source")

        # Should receive both general state change and specific level up event
        assert len(level_subscriber.received_events) == 1
        level_event = level_subscriber.received_events[0]
        assert level_event.event_type == EventType.PLAYER_LEVEL_UP

    def test_get_recent_events(self):
        """Test getting recent events"""
        events = []
        for i in range(5):
            event = Event(EventType.AGENT_DECISION, time.time(), f"source_{i}")
            events.append(event)
            self.event_bus.publish(event)

        recent = self.event_bus.get_recent_events(3)
        assert len(recent) == 3
        assert recent == events[-3:]

    def test_get_recent_events_with_filter(self):
        """Test getting recent events with filter"""
        # Publish mixed events
        battle_event = Event(EventType.BATTLE_STARTED, time.time(), "test")
        level_event = Event(EventType.PLAYER_LEVEL_UP, time.time(), "test")
        badge_event = Event(EventType.BADGE_EARNED, time.time(), "test")

        self.event_bus.publish(battle_event)
        self.event_bus.publish(level_event)
        self.event_bus.publish(badge_event)

        # Filter for only battle events
        battle_filter = EventFilter(event_types={EventType.BATTLE_STARTED})
        recent_battles = self.event_bus.get_recent_events(10, battle_filter)

        assert len(recent_battles) == 1
        assert recent_battles[0].event_type == EventType.BATTLE_STARTED

    def test_get_event_stats(self):
        """Test getting event bus statistics"""
        # Publish some events
        self.event_bus.publish(Event(EventType.BATTLE_STARTED, time.time(), "test"))
        self.event_bus.publish(Event(EventType.PLAYER_LEVEL_UP, time.time(), "test"))

        # Subscribe a test subscriber
        self.event_bus.subscribe(self.test_subscriber)

        stats = self.event_bus.get_event_stats()

        assert stats['total_events'] == 2
        assert stats['event_type_counts'][EventType.BATTLE_STARTED] == 1
        assert stats['event_type_counts'][EventType.PLAYER_LEVEL_UP] == 1
        assert stats['subscribers_count'] >= 1


class TestGameStateEventDetector:
    """Test GameStateEventDetector functionality"""

    def setup_method(self):
        """Setup for each test"""
        self.event_bus = EventBus()
        self.detector = GameStateEventDetector(self.event_bus)

    def test_initial_state_update(self):
        """Test first state update doesn't publish events"""
        state = {'player_level': 5, 'badges_total': 0}

        # First update should not publish events (no previous state)
        self.detector.update(state)
        assert len(self.event_bus.event_history) == 0

    def test_level_up_detection(self):
        """Test level up event detection"""
        initial_state = {'player_level': 5, 'badges_total': 0}
        level_up_state = {'player_level': 6, 'badges_total': 0}

        self.detector.update(initial_state)
        self.detector.update(level_up_state)

        # Should have published state change and level up events
        assert len(self.event_bus.event_history) >= 1

        # Check for level up event
        level_events = [e for e in self.event_bus.event_history if e.event_type == EventType.PLAYER_LEVEL_UP]
        assert len(level_events) == 1
        assert level_events[0].data['old_level'] == 5
        assert level_events[0].data['new_level'] == 6

    def test_badge_earned_detection(self):
        """Test badge earned event detection"""
        initial_state = {'player_level': 10, 'badges_total': 0}
        badge_state = {'player_level': 10, 'badges_total': 1}

        self.detector.update(initial_state)
        self.detector.update(badge_state)

        # Check for badge earned event
        badge_events = [e for e in self.event_bus.event_history if e.event_type == EventType.BADGE_EARNED]
        assert len(badge_events) == 1
        assert badge_events[0].data['old_badges'] == 0
        assert badge_events[0].data['new_badges'] == 1

    def test_location_change_detection(self):
        """Test location change event detection"""
        initial_state = {'player_map': 1}
        new_location_state = {'player_map': 2}

        self.detector.update(initial_state)
        self.detector.update(new_location_state)

        # Check for location change event
        location_events = [e for e in self.event_bus.event_history if e.event_type == EventType.LOCATION_CHANGED]
        assert len(location_events) == 1
        assert location_events[0].data['old_map'] == 1
        assert location_events[0].data['new_map'] == 2

    def test_battle_state_detection(self):
        """Test battle state change detection"""
        initial_state = {'in_battle': False}
        battle_start_state = {'in_battle': True, 'enemy_level': 10}
        battle_end_state = {'in_battle': False}

        self.detector.update(initial_state)
        self.detector.update(battle_start_state)

        # Check for battle started event
        battle_start_events = [e for e in self.event_bus.event_history if e.event_type == EventType.BATTLE_STARTED]
        assert len(battle_start_events) == 1
        assert battle_start_events[0].data['enemy_level'] == 10

        self.detector.update(battle_end_state)

        # Check for battle ended event
        battle_end_events = [e for e in self.event_bus.event_history if e.event_type == EventType.BATTLE_ENDED]
        assert len(battle_end_events) == 1

    def test_hp_critical_detection(self):
        """Test critical HP event detection"""
        critical_state = {
            'party_count': 1,
            'player_hp': 10,
            'player_max_hp': 100
        }

        self.detector.update(critical_state)
        self.detector.update(critical_state)  # Need second update to trigger detection

        # Check for HP critical event
        hp_events = [e for e in self.event_bus.event_history if e.event_type == EventType.HP_CRITICAL]
        assert len(hp_events) == 1
        assert hp_events[0].data['hp_ratio'] == 0.1


class TestEventDrivenAnalytics:
    """Test EventDrivenAnalytics functionality"""

    def setup_method(self):
        """Setup for each test"""
        self.analytics = EventDrivenAnalytics()

    def test_analytics_initialization(self):
        """Test analytics initialization"""
        assert self.analytics.metrics['battles_fought'] == 0
        assert self.analytics.metrics['levels_gained'] == 0
        assert len(self.analytics.event_timeline) == 0

    def test_battle_event_tracking(self):
        """Test battle event analytics"""
        battle_start = Event(EventType.BATTLE_STARTED, time.time(), "test")
        battle_end_win = Event(EventType.BATTLE_ENDED, time.time(), "test", data={'player_won': True})

        self.analytics.handle_event(battle_start)
        self.analytics.handle_event(battle_end_win)

        assert self.analytics.metrics['battles_fought'] == 1
        assert self.analytics.metrics['battles_won'] == 1

    def test_progression_event_tracking(self):
        """Test progression event analytics"""
        level_event = Event(EventType.PLAYER_LEVEL_UP, time.time(), "test")
        badge_event = Event(EventType.BADGE_EARNED, time.time(), "test")

        self.analytics.handle_event(level_event)
        self.analytics.handle_event(badge_event)

        assert self.analytics.metrics['levels_gained'] == 1
        assert self.analytics.metrics['badges_earned'] == 1

    def test_agent_performance_tracking(self):
        """Test agent performance analytics"""
        agent_decision = Event(
            EventType.AGENT_DECISION,
            time.time(),
            "multi_agent_coordinator",
            data={'chosen_agent': 'battle'}
        )

        agent_performance = Event(
            EventType.AGENT_PERFORMANCE_UPDATE,
            time.time(),
            "battle_agent",
            data={
                'agent_type': 'battle',
                'battle_result': 'won'
            }
        )

        self.analytics.handle_event(agent_decision)
        self.analytics.handle_event(agent_performance)

        assert self.analytics.metrics['agent_decisions'] == 1
        assert self.analytics.agent_performance_tracker['battle']['decisions'] == 1
        assert self.analytics.agent_performance_tracker['battle']['successes'] == 1

    def test_analytics_summary(self):
        """Test getting analytics summary"""
        # Add some events
        self.analytics.handle_event(Event(EventType.BATTLE_STARTED, time.time(), "test"))
        self.analytics.handle_event(Event(EventType.BATTLE_ENDED, time.time(), "test", data={'player_won': True}))
        self.analytics.handle_event(Event(EventType.PLAYER_LEVEL_UP, time.time(), "test"))

        summary = self.analytics.get_analytics_summary()

        assert 'metrics' in summary
        assert 'derived_metrics' in summary
        assert 'agent_performance' in summary
        assert 'real_time_stats' in summary
        assert summary['metrics']['battles_fought'] == 1
        assert summary['derived_metrics']['battle_win_rate'] == 1.0

    def test_performance_insights(self):
        """Test performance insights generation"""
        # Simulate successful battles
        for i in range(10):
            self.analytics.handle_event(Event(EventType.BATTLE_STARTED, time.time(), "test"))
            self.analytics.handle_event(Event(EventType.BATTLE_ENDED, time.time(), "test", data={'player_won': True}))

        insights = self.analytics.get_performance_insights()

        assert 'highlights' in insights
        assert 'concerns' in insights
        assert 'recommendations' in insights
        assert len(insights['highlights']) > 0  # Should highlight good battle performance

    def test_event_distribution(self):
        """Test event distribution tracking"""
        # Add various events
        events = [
            Event(EventType.BATTLE_STARTED, time.time(), "test"),
            Event(EventType.BATTLE_STARTED, time.time(), "test"),
            Event(EventType.PLAYER_LEVEL_UP, time.time(), "test"),
        ]

        for event in events:
            self.analytics.handle_event(event)

        summary = self.analytics.get_analytics_summary()
        distribution = summary['event_distribution']

        assert distribution['battle_started'] == 2
        assert distribution['player_level_up'] == 1


class TestGlobalEventSystem:
    """Test global event system functions"""

    def test_get_event_bus_singleton(self):
        """Test global event bus singleton"""
        bus1 = get_event_bus()
        bus2 = get_event_bus()

        assert bus1 is bus2  # Should be the same instance

    def test_initialize_event_system(self):
        """Test event system initialization"""
        event_bus = initialize_event_system()

        assert event_bus is not None
        assert isinstance(event_bus, EventBus)

        # Should have analytics subscriber
        stats = event_bus.get_event_stats()
        assert stats['subscribers_count'] > 0


@pytest.mark.integration
class TestEventSystemIntegration:
    """Integration tests for the event system"""

    def setup_method(self):
        """Setup for integration tests"""
        self.event_bus = initialize_event_system()

    def test_full_game_state_workflow(self):
        """Test complete game state change workflow"""
        detector = GameStateEventDetector(self.event_bus)

        # Initial state
        initial_state = {
            'player_level': 5,
            'badges_total': 0,
            'player_map': 1,
            'in_battle': False
        }

        # State after level up and badge
        final_state = {
            'player_level': 6,
            'badges_total': 1,
            'player_map': 2,
            'in_battle': False
        }

        detector.update(initial_state)
        detector.update(final_state)

        # Check that appropriate events were generated
        event_types = [e.event_type for e in self.event_bus.event_history]

        assert EventType.GAME_STATE_CHANGED in event_types
        assert EventType.PLAYER_LEVEL_UP in event_types
        assert EventType.BADGE_EARNED in event_types
        assert EventType.LOCATION_CHANGED in event_types

    def test_analytics_integration(self):
        """Test analytics integration with event detection"""
        # Get the analytics instance (should be auto-subscribed)
        stats_before = self.event_bus.get_event_stats()
        subscribers_before = stats_before['subscribers_count']

        # Publish some events
        self.event_bus.publish(Event(EventType.BATTLE_STARTED, time.time(), "test"))
        self.event_bus.publish(Event(EventType.PLAYER_LEVEL_UP, time.time(), "test"))

        # Analytics should have processed these events
        stats_after = self.event_bus.get_event_stats()
        assert stats_after['total_events'] == 2

    @patch('time.time')
    def test_performance_under_load(self, mock_time):
        """Test event system performance under load"""
        mock_time.return_value = 1000.0

        # Create multiple subscribers
        subscribers = [TestEventSubscriber() for _ in range(10)]
        for subscriber in subscribers:
            self.event_bus.subscribe(subscriber)

        # Publish many events
        events_count = 100
        for i in range(events_count):
            event = Event(EventType.GAME_STATE_CHANGED, 1000.0 + i, f"source_{i}")
            self.event_bus.publish(event)

        # Verify all subscribers received all events
        for subscriber in subscribers:
            assert len(subscriber.received_events) == events_count

        # Verify event history is maintained properly
        assert len(self.event_bus.event_history) == events_count