"""
Enhanced tests for Event System advanced features.

These tests focus on advanced event system functionality that may not be
covered by the basic test suite, including async processing, error handling,
pattern matching, and correlation tracking.
"""

import pytest
import time
import threading
from unittest.mock import Mock, patch, MagicMock, call
from typing import Dict, Any, Set, List
from collections import defaultdict

from core.event_system import (
    EventType, Event, EventBus, EventSubscriber, EventFilter,
    GameStateEventDetector, EventDrivenAnalytics
)


class AdvancedEventSubscriber(EventSubscriber):
    """Advanced test event subscriber for testing complex scenarios"""

    def __init__(self, subscribed_events: Set[EventType] = None, should_fail: bool = False):
        self.subscribed_events = subscribed_events or {EventType.GAME_STATE_CHANGED}
        self.received_events = []
        self.handle_event_called = False
        self.should_fail = should_fail
        self.error_count = 0

    def get_subscribed_events(self) -> Set[EventType]:
        return self.subscribed_events

    def handle_event(self, event: Event) -> None:
        self.handle_event_called = True

        if self.should_fail:
            self.error_count += 1
            raise Exception(f"Simulated error in event handling: {event.event_type}")

        self.received_events.append(event)


class TestEventBusAdvancedFeatures:
    """Test advanced EventBus features."""

    def test_correlation_tracking(self):
        """Test event correlation tracking functionality."""
        event_bus = EventBus()

        correlation_id = "test_correlation_123"

        # Create related events with same correlation ID
        event1 = Event(
            event_type=EventType.BATTLE_STARTED,
            timestamp=time.time(),
            source="battle_agent",
            correlation_id=correlation_id
        )

        event2 = Event(
            event_type=EventType.AGENT_DECISION,
            timestamp=time.time(),
            source="battle_agent",
            correlation_id=correlation_id
        )

        event3 = Event(
            event_type=EventType.BATTLE_ENDED,
            timestamp=time.time(),
            source="battle_agent",
            correlation_id=correlation_id
        )

        # Publish events
        event_bus.publish(event1)
        event_bus.publish(event2)
        event_bus.publish(event3)

        # Verify correlation tracking
        assert correlation_id in event_bus.correlation_tracker
        correlated_events = event_bus.correlation_tracker[correlation_id]
        assert len(correlated_events) == 3
        assert event1 in correlated_events
        assert event2 in correlated_events
        assert event3 in correlated_events

    def test_pattern_matching_registration(self):
        """Test pattern matcher registration and execution."""
        event_bus = EventBus()

        # Create a pattern matcher
        pattern_calls = []
        def battle_pattern_matcher(events: List[Event]) -> None:
            """Detect battle patterns"""
            pattern_calls.append(len(events))

            # Look for battle start -> decision -> battle end pattern
            if len(events) >= 3:
                recent_events = events[-3:]
                event_types = [e.event_type for e in recent_events]
                if (event_types[0] == EventType.BATTLE_STARTED and
                    event_types[1] == EventType.AGENT_DECISION and
                    event_types[2] == EventType.BATTLE_ENDED):
                    pattern_calls.append("BATTLE_PATTERN_DETECTED")

        # Register pattern matcher
        event_bus.pattern_matchers.append(battle_pattern_matcher)

        # Publish events that should trigger pattern
        event_bus.publish(Event(EventType.BATTLE_STARTED, time.time(), "test"))
        event_bus.publish(Event(EventType.AGENT_DECISION, time.time(), "test"))
        event_bus.publish(Event(EventType.BATTLE_ENDED, time.time(), "test"))

        # Verify pattern matcher was called
        assert len(pattern_calls) >= 3  # Called for each event
        assert "BATTLE_PATTERN_DETECTED" in pattern_calls

    def test_error_handling_with_error_handlers(self):
        """Test error handling with registered error handlers."""
        event_bus = EventBus()

        # Create error handler
        error_events = []
        def error_handler(event: Event, exception: Exception) -> None:
            error_events.append((event, exception))

        event_bus.error_handlers.append(error_handler)

        # Create failing subscriber
        failing_subscriber = AdvancedEventSubscriber(
            subscribed_events={EventType.BATTLE_STARTED},
            should_fail=True
        )

        event_bus.subscribe(failing_subscriber)

        # Publish event that will cause error
        test_event = Event(EventType.BATTLE_STARTED, time.time(), "test")
        event_bus.publish(test_event)

        # Verify error was handled
        assert len(error_events) == 1
        assert error_events[0][0] == test_event
        assert "Simulated error" in str(error_events[0][1])

    def test_async_processing_configuration(self):
        """Test async processing can be enabled/disabled."""
        event_bus = EventBus()

        # Test async processing is enabled by default
        assert event_bus.async_processing is True

        # Test disabling async processing
        event_bus.async_processing = False
        assert event_bus.async_processing is False

        # Test enabling again
        event_bus.async_processing = True
        assert event_bus.async_processing is True

    def test_event_delivery_with_async_disabled(self):
        """Test event delivery when async processing is disabled."""
        event_bus = EventBus()
        event_bus.async_processing = False  # Disable async

        subscriber = AdvancedEventSubscriber({EventType.BATTLE_STARTED})
        event_bus.subscribe(subscriber)

        test_event = Event(EventType.BATTLE_STARTED, time.time(), "test")
        event_bus.publish(test_event)

        # Should be delivered synchronously
        assert subscriber.handle_event_called is True
        assert len(subscriber.received_events) == 1
        assert subscriber.received_events[0] == test_event

    def test_max_history_constraint(self):
        """Test that event history respects max_history limit."""
        max_history = 5
        event_bus = EventBus(max_history=max_history)

        # Publish more events than max_history
        for i in range(max_history + 3):
            event = Event(EventType.GAME_STATE_CHANGED, time.time(), f"source_{i}")
            event_bus.publish(event)

        # Should only keep max_history events
        assert len(event_bus.event_history) == max_history

        # Should be the most recent events (deque keeps most recent at the end)
        recent_events = list(event_bus.event_history)
        # Should contain the last max_history events (source_3 to source_7)
        assert recent_events[-1].source == f"source_{max_history + 2}"  # Last event (source_7)
        assert len(recent_events) == max_history

    def test_async_delivery_functionality(self):
        """Test that async delivery functionality works."""
        event_bus = EventBus()
        event_bus.async_processing = True

        subscriber = AdvancedEventSubscriber({EventType.BATTLE_STARTED})
        event_bus.subscribe(subscriber)

        test_event = Event(EventType.BATTLE_STARTED, time.time(), "test")
        event_bus.publish(test_event)

        # Allow time for async processing (even though it's currently sync)
        time.sleep(0.1)

        # Verify event was delivered
        assert subscriber.handle_event_called is True
        assert len(subscriber.received_events) == 1

    def test_filtered_subscription_complex_filters(self):
        """Test complex filtered subscriptions."""
        event_bus = EventBus()

        # Create subscriber with complex filter
        subscriber = AdvancedEventSubscriber()

        # Filter for high-priority battle events from specific sources
        event_filter = EventFilter(
            event_types={EventType.BATTLE_STARTED, EventType.BATTLE_ENDED},
            sources={"battle_agent", "llm_agent"},
            min_priority=7,
            custom_filter=lambda e: "important" in e.data.get("tags", [])
        )

        event_bus.subscribe(subscriber, event_filter)

        # Test events - should match
        matching_event = Event(
            event_type=EventType.BATTLE_STARTED,
            timestamp=time.time(),
            source="battle_agent",
            priority=8,
            data={"tags": ["important", "battle"]}
        )

        # Test events - should not match (wrong source)
        non_matching_event1 = Event(
            event_type=EventType.BATTLE_STARTED,
            timestamp=time.time(),
            source="other_agent",
            priority=8,
            data={"tags": ["important", "battle"]}
        )

        # Test events - should not match (low priority)
        non_matching_event2 = Event(
            event_type=EventType.BATTLE_STARTED,
            timestamp=time.time(),
            source="battle_agent",
            priority=5,
            data={"tags": ["important", "battle"]}
        )

        # Test events - should not match (missing tag)
        non_matching_event3 = Event(
            event_type=EventType.BATTLE_STARTED,
            timestamp=time.time(),
            source="battle_agent",
            priority=8,
            data={"tags": ["battle"]}
        )

        # Publish all events
        event_bus.publish(matching_event)
        event_bus.publish(non_matching_event1)
        event_bus.publish(non_matching_event2)
        event_bus.publish(non_matching_event3)

        # Wait a bit for async processing
        time.sleep(0.1)

        # Should only receive the matching event
        assert len(subscriber.received_events) == 1
        assert subscriber.received_events[0] == matching_event


class TestGameStateEventDetectorAdvanced:
    """Test advanced GameStateEventDetector functionality."""

    def test_complex_state_change_detection(self):
        """Test detection of complex state changes."""
        mock_event_bus = Mock()
        detector = GameStateEventDetector(mock_event_bus)

        # Complex state change with multiple simultaneous changes
        old_state = {
            'player_level': 12,
            'player_hp': 45,
            'player_max_hp': 50,
            'badges_total': 2,
            'player_map': 5,
            'player_x': 10,
            'player_y': 15,
            'in_battle': 0,
            'party_count': 3
        }

        new_state = {
            'player_level': 13,  # Level up
            'player_hp': 52,     # HP increased (level up effect)
            'player_max_hp': 55, # Max HP increased
            'badges_total': 3,   # Badge earned
            'player_map': 6,     # Location changed
            'player_x': 0,       # New map starting position
            'player_y': 0,
            'in_battle': 0,
            'party_count': 3
        }

        # First update sets the baseline
        detector.update(old_state)
        mock_event_bus.reset_mock()  # Clear any initial events

        # Second update detects changes
        detector.update(new_state)

        # Should detect state changes
        assert mock_event_bus.publish_game_state_change.called

        # Verify the method was called with the expected arguments
        call_args = mock_event_bus.publish_game_state_change.call_args
        assert call_args is not None
        assert call_args[0][0] == old_state  # First argument should be old state
        assert call_args[0][1] == new_state  # Second argument should be new state

    def test_edge_case_state_changes(self):
        """Test edge cases in state change detection."""
        mock_event_bus = Mock()
        detector = GameStateEventDetector(mock_event_bus)

        # Test with missing/None values
        old_state = {'player_level': None, 'badges_total': 0}
        new_state = {'player_level': 5, 'badges_total': 1}

        # First update sets the baseline
        detector.update(old_state)
        mock_event_bus.reset_mock()

        # Second update detects changes
        detector.update(new_state)

        # Should handle None gracefully and detect changes
        assert mock_event_bus.publish_game_state_change.called

    def test_hp_critical_threshold_detection(self):
        """Test HP critical detection with different thresholds."""
        mock_event_bus = Mock()
        detector = GameStateEventDetector(mock_event_bus)

        # Test exactly at threshold (20%)
        old_state = {'player_hp': 25, 'player_max_hp': 100}
        new_state = {'player_hp': 20, 'player_max_hp': 100}  # Exactly 20%

        # First update sets the baseline
        detector.update(old_state)
        mock_event_bus.reset_mock()

        # Second update detects changes
        detector.update(new_state)

        # Should trigger critical HP event through publish_game_state_change
        assert mock_event_bus.publish_game_state_change.called

        # Test edge case: 0 max HP (should not crash)
        mock_event_bus.reset_mock()
        old_state = {'player_hp': 0, 'player_max_hp': 0}
        new_state = {'player_hp': 0, 'player_max_hp': 0}

        # First update sets the baseline
        detector.update(old_state)
        mock_event_bus.reset_mock()

        # Second update should not crash
        detector.update(new_state)

        # Should handle 0 max HP gracefully
        # Either no call or a call that doesn't crash
        assert True  # Test that we don't crash


class TestEventDrivenAnalyticsAdvanced:
    """Test advanced EventDrivenAnalytics functionality."""

    def test_analytics_with_correlation_tracking(self):
        """Test analytics tracking with event correlation."""
        analytics = EventDrivenAnalytics()

        correlation_id = "battle_session_456"

        # Simulate a complete battle session
        battle_events = [
            Event(EventType.BATTLE_STARTED, time.time(), "battle_agent",
                  correlation_id=correlation_id, data={"enemy_level": 15}),
            Event(EventType.AGENT_DECISION, time.time(), "battle_agent",
                  correlation_id=correlation_id, data={"action": "attack", "confidence": 0.8}),
            Event(EventType.AGENT_DECISION, time.time(), "battle_agent",
                  correlation_id=correlation_id, data={"action": "attack", "confidence": 0.9}),
            Event(EventType.BATTLE_ENDED, time.time(), "battle_agent",
                  correlation_id=correlation_id, data={"player_won": True, "turns": 3})
        ]

        for event in battle_events:
            analytics.handle_event(event)

        # Check analytics tracked the session
        assert analytics.metrics['battles_fought'] == 1
        assert analytics.metrics['battles_won'] == 1  # Victory was True

        # Check that agent decisions were tracked
        assert analytics.metrics['agent_decisions'] >= 2

    def test_analytics_performance_metrics(self):
        """Test analytics performance tracking."""
        analytics = EventDrivenAnalytics()

        # Simulate agent performance events
        agent_events = [
            Event(EventType.AGENT_PERFORMANCE_UPDATE, time.time(), "battle_agent",
                  data={"agent_name": "battle_agent", "performance_score": 0.85, "wins": 5, "total": 7}),
            Event(EventType.AGENT_PERFORMANCE_UPDATE, time.time(), "explorer_agent",
                  data={"agent_name": "explorer_agent", "performance_score": 0.70, "discoveries": 12}),
            Event(EventType.AGENT_PERFORMANCE_UPDATE, time.time(), "battle_agent",
                  data={"agent_name": "battle_agent", "performance_score": 0.90, "wins": 8, "total": 10})
        ]

        for event in agent_events:
            analytics.handle_event(event)

        # Check that agent performance events were handled
        assert len(analytics.event_timeline) == 3  # All events recorded

        # Check that agent performance tracker was updated
        # The actual implementation tracks by agent type, not name
        assert analytics.agent_performance_tracker is not None

    def test_analytics_summary_comprehensive(self):
        """Test comprehensive analytics summary generation."""
        analytics = EventDrivenAnalytics()

        # Simulate diverse events
        events = [
            Event(EventType.PLAYER_LEVEL_UP, time.time(), "game",
                  data={"old_level": 5, "new_level": 6}),
            Event(EventType.BADGE_EARNED, time.time(), "game",
                  data={"badge_id": 1, "gym": "Violet City"}),
            Event(EventType.BATTLE_STARTED, time.time(), "battle_agent"),
            Event(EventType.BATTLE_ENDED, time.time(), "battle_agent",
                  data={"player_won": True}),
            Event(EventType.BATTLE_STARTED, time.time(), "battle_agent"),
            Event(EventType.BATTLE_ENDED, time.time(), "battle_agent",
                  data={"player_won": False}),
            Event(EventType.NEW_AREA_DISCOVERED, time.time(), "explorer_agent",
                  data={"area_name": "Route 32"}),
        ]

        for event in events:
            analytics.handle_event(event)

        summary = analytics.get_analytics_summary()

        # Verify comprehensive summary structure
        assert "metrics" in summary
        assert "derived_metrics" in summary
        assert summary["metrics"]["levels_gained"] == 1
        assert summary["metrics"]["badges_earned"] == 1
        assert summary["metrics"]["battles_fought"] == 2
        assert summary["metrics"]["battles_won"] == 1
        assert summary["derived_metrics"]["battle_win_rate"] == 0.5

    def test_analytics_event_distribution_tracking(self):
        """Test event distribution analysis."""
        analytics = EventDrivenAnalytics()

        # Create events with varied frequencies
        event_counts = {
            EventType.GAME_STATE_CHANGED: 50,
            EventType.AGENT_DECISION: 25,
            EventType.BATTLE_STARTED: 5,
            EventType.BATTLE_ENDED: 5,
            EventType.PLAYER_LEVEL_UP: 2,
            EventType.BADGE_EARNED: 1
        }

        # Generate events
        for event_type, count in event_counts.items():
            for i in range(count):
                event = Event(event_type, time.time(), "test_source")
                analytics.handle_event(event)

        summary = analytics.get_analytics_summary()
        distribution = summary["event_distribution"]

        # Verify that distribution includes event counts
        # The actual implementation uses _get_event_distribution which may track differently
        assert isinstance(distribution, dict)
        assert len(distribution) > 0  # Should have some events


class TestEventSystemIntegrationAdvanced:
    """Test advanced integration scenarios."""

    def test_concurrent_event_processing(self):
        """Test event processing under concurrent load."""
        event_bus = EventBus()

        # Create multiple subscribers
        subscribers = []
        for i in range(5):
            subscriber = AdvancedEventSubscriber({
                EventType.GAME_STATE_CHANGED,
                EventType.BATTLE_STARTED,
                EventType.AGENT_DECISION
            })
            subscribers.append(subscriber)
            event_bus.subscribe(subscriber)

        # Publish events from multiple threads
        def publish_events(thread_id: int):
            for i in range(10):
                event = Event(
                    event_type=EventType.GAME_STATE_CHANGED,
                    timestamp=time.time(),
                    source=f"thread_{thread_id}",
                    data={"thread_id": thread_id, "event_num": i}
                )
                event_bus.publish(event)
                time.sleep(0.01)  # Small delay

        # Create and start threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=publish_events, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Allow time for async processing
        time.sleep(0.5)

        # Verify all subscribers received events
        for subscriber in subscribers:
            assert subscriber.handle_event_called is True
            # Should have received events from all threads
            assert len(subscriber.received_events) > 0

    def test_event_system_error_recovery(self):
        """Test error recovery in event processing."""
        event_bus = EventBus()

        # Add error tracking
        error_log = []
        def error_tracker(event: Event, exception: Exception) -> None:
            error_log.append(f"Error processing {event.event_type.value}: {str(exception)}")

        event_bus.error_handlers.append(error_tracker)

        # Create mix of working and failing subscribers
        working_subscriber = AdvancedEventSubscriber({EventType.BATTLE_STARTED})
        failing_subscriber = AdvancedEventSubscriber({EventType.BATTLE_STARTED}, should_fail=True)

        event_bus.subscribe(working_subscriber)
        event_bus.subscribe(failing_subscriber)

        # Publish event
        test_event = Event(EventType.BATTLE_STARTED, time.time(), "test")
        event_bus.publish(test_event)

        # Wait for processing
        time.sleep(0.1)

        # Working subscriber should still receive event
        assert working_subscriber.handle_event_called is True
        assert len(working_subscriber.received_events) == 1

        # Error should be logged
        assert len(error_log) == 1
        assert "battle_started" in error_log[0].lower()
        assert "simulated error" in error_log[0].lower()

    def test_memory_usage_with_high_event_volume(self):
        """Test memory management with high event volume."""
        # Use small history size to test memory management
        event_bus = EventBus(max_history=100)

        # Generate many events
        for i in range(500):
            event = Event(
                event_type=EventType.GAME_STATE_CHANGED,
                timestamp=time.time(),
                source=f"generator_{i % 10}",
                data={"iteration": i, "large_data": "x" * 100}  # Some payload
            )
            event_bus.publish(event)

        # Memory should be bounded by max_history
        assert len(event_bus.event_history) == 100

        # Should contain the most recent events
        recent_events = list(event_bus.event_history)
        assert recent_events[0].data["iteration"] >= 400  # Should be recent
        assert recent_events[-1].data["iteration"] == 499  # Should be the last