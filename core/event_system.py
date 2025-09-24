"""
Event System - Reactive architecture for game state changes

This system provides a comprehensive event-driven architecture that allows
components to react to game state changes, training events, and system events
without tight coupling.
"""

import logging
import time
import threading
from typing import Dict, Any, List, Callable, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
from abc import ABC, abstractmethod


class EventType(Enum):
    """Different types of events in the system"""
    # Game state events
    GAME_STATE_CHANGED = "game_state_changed"
    PLAYER_LEVEL_UP = "player_level_up"
    BADGE_EARNED = "badge_earned"
    POKEMON_CAUGHT = "pokemon_caught"
    BATTLE_STARTED = "battle_started"
    BATTLE_ENDED = "battle_ended"
    LOCATION_CHANGED = "location_changed"
    HP_CRITICAL = "hp_critical"

    # Training events
    TRAINING_STARTED = "training_started"
    TRAINING_STOPPED = "training_stopped"
    EPISODE_STARTED = "episode_started"
    EPISODE_ENDED = "episode_ended"
    REWARD_RECEIVED = "reward_received"

    # Agent events
    AGENT_DECISION = "agent_decision"
    AGENT_ACTION_TAKEN = "agent_action_taken"
    AGENT_SWITCHED = "agent_switched"
    AGENT_PERFORMANCE_UPDATE = "agent_performance_update"

    # System events
    SYSTEM_ERROR = "system_error"
    SYSTEM_WARNING = "system_warning"
    COMPONENT_INITIALIZED = "component_initialized"
    COMPONENT_SHUTDOWN = "component_shutdown"

    # Discovery events
    NEW_AREA_DISCOVERED = "new_area_discovered"
    ITEM_DISCOVERED = "item_discovered"
    NPC_ENCOUNTERED = "npc_encountered"

    # Quest events
    QUEST_STARTED = "quest_started"
    QUEST_COMPLETED = "quest_completed"
    QUEST_FAILED = "quest_failed"
    OBJECTIVE_UPDATED = "objective_updated"


@dataclass
class Event:
    """Represents an event in the system"""
    event_type: EventType
    timestamp: float
    source: str  # Component that generated the event
    data: Dict[str, Any] = field(default_factory=dict)
    priority: int = 5  # 1-10, higher = more important
    event_id: str = field(default="")
    correlation_id: Optional[str] = None  # For tracking related events

    def __post_init__(self):
        if not self.event_id:
            self.event_id = f"{self.event_type.value}_{int(self.timestamp * 1000)}"


class EventSubscriber(ABC):
    """Abstract base class for event subscribers"""

    @abstractmethod
    def handle_event(self, event: Event) -> None:
        """Handle an incoming event"""
        pass

    @abstractmethod
    def get_subscribed_events(self) -> Set[EventType]:
        """Return set of event types this subscriber is interested in"""
        pass


class EventFilter:
    """Filters events based on criteria"""

    def __init__(self,
                 event_types: Optional[Set[EventType]] = None,
                 sources: Optional[Set[str]] = None,
                 min_priority: int = 1,
                 custom_filter: Optional[Callable[[Event], bool]] = None):
        self.event_types = event_types
        self.sources = sources
        self.min_priority = min_priority
        self.custom_filter = custom_filter

    def matches(self, event: Event) -> bool:
        """Check if event matches filter criteria"""
        if self.event_types and event.event_type not in self.event_types:
            return False

        if self.sources and event.source not in self.sources:
            return False

        if event.priority < self.min_priority:
            return False

        if self.custom_filter and not self.custom_filter(event):
            return False

        return True


class EventBus:
    """Central event bus for publishing and subscribing to events"""

    def __init__(self, max_history: int = 1000):
        self.logger = logging.getLogger("EventBus")

        # Subscriber management
        self.subscribers: Dict[EventType, List[EventSubscriber]] = defaultdict(list)
        self.filtered_subscribers: List[tuple[EventFilter, EventSubscriber]] = []

        # Event history and analytics
        self.event_history: deque = deque(maxlen=max_history)
        self.event_stats: Dict[EventType, int] = defaultdict(int)

        # Performance and reliability
        self.async_processing = True
        self.thread_pool = None
        self.error_handlers: List[Callable[[Event, Exception], None]] = []

        # Event correlation and patterns
        self.correlation_tracker: Dict[str, List[Event]] = defaultdict(list)
        self.pattern_matchers: List[Callable[[List[Event]], None]] = []

        self.logger.info("EventBus initialized")

    def subscribe(self,
                  subscriber: EventSubscriber,
                  event_filter: Optional[EventFilter] = None) -> None:
        """Subscribe to events"""
        if event_filter:
            self.filtered_subscribers.append((event_filter, subscriber))
        else:
            # Subscribe to all events the subscriber is interested in
            for event_type in subscriber.get_subscribed_events():
                self.subscribers[event_type].append(subscriber)

        self.logger.debug(f"Subscribed {subscriber.__class__.__name__} to events")

    def unsubscribe(self, subscriber: EventSubscriber) -> None:
        """Unsubscribe from all events"""
        # Remove from direct subscribers
        for event_type, subscriber_list in self.subscribers.items():
            if subscriber in subscriber_list:
                subscriber_list.remove(subscriber)

        # Remove from filtered subscribers
        self.filtered_subscribers = [
            (f, s) for f, s in self.filtered_subscribers if s != subscriber
        ]

        self.logger.debug(f"Unsubscribed {subscriber.__class__.__name__}")

    def publish(self, event: Event) -> None:
        """Publish an event to all interested subscribers"""
        try:
            # Record event
            self._record_event(event)

            # Get all interested subscribers
            interested_subscribers = self._get_interested_subscribers(event)

            # Deliver event to subscribers
            if self.async_processing:
                self._deliver_async(event, interested_subscribers)
            else:
                self._deliver_sync(event, interested_subscribers)

            # Check for patterns
            self._check_patterns(event)

        except Exception as e:
            self.logger.error(f"Error publishing event {event.event_type.value}: {e}")
            self._handle_error(event, e)

    def publish_game_state_change(self,
                                old_state: Dict[str, Any],
                                new_state: Dict[str, Any],
                                source: str = "game") -> None:
        """Convenience method for publishing game state changes"""
        changes = self._detect_state_changes(old_state, new_state)

        # Publish general state change event
        event = Event(
            event_type=EventType.GAME_STATE_CHANGED,
            timestamp=time.time(),
            source=source,
            data={
                'old_state': old_state,
                'new_state': new_state,
                'changes': changes
            },
            priority=6
        )
        self.publish(event)

        # Publish specific change events
        for change_type, change_data in changes.items():
            self._publish_specific_change_event(change_type, change_data, source)

    def _detect_state_changes(self,
                            old_state: Dict[str, Any],
                            new_state: Dict[str, Any]) -> Dict[str, Any]:
        """Detect specific changes between game states"""
        changes = {}

        # Level changes
        old_level = old_state.get('player_level', 0)
        new_level = new_state.get('player_level', 0)
        if new_level > old_level:
            changes['level_up'] = {'old_level': old_level, 'new_level': new_level}

        # Badge changes
        old_badges = old_state.get('badges_total', 0)
        new_badges = new_state.get('badges_total', 0)
        if new_badges > old_badges:
            changes['badge_earned'] = {'old_badges': old_badges, 'new_badges': new_badges}

        # Location changes
        old_map = old_state.get('player_map', 0)
        new_map = new_state.get('player_map', 0)
        if old_map != new_map:
            changes['location_changed'] = {'old_map': old_map, 'new_map': new_map}

        # Battle state changes
        old_battle = old_state.get('in_battle', False)
        new_battle = new_state.get('in_battle', False)
        if old_battle != new_battle:
            if new_battle:
                changes['battle_started'] = {'enemy_level': new_state.get('enemy_level', 0)}
            else:
                changes['battle_ended'] = {'player_won': True}  # Simplified

        # HP critical state
        if new_state.get('party_count', 0) > 0:
            hp_ratio = new_state.get('player_hp', 0) / max(new_state.get('player_max_hp', 1), 1)
            if hp_ratio < 0.2:
                changes['hp_critical'] = {'hp_ratio': hp_ratio}

        return changes

    def _publish_specific_change_event(self,
                                     change_type: str,
                                     change_data: Dict[str, Any],
                                     source: str) -> None:
        """Publish specific change events"""
        event_type_map = {
            'level_up': EventType.PLAYER_LEVEL_UP,
            'badge_earned': EventType.BADGE_EARNED,
            'location_changed': EventType.LOCATION_CHANGED,
            'battle_started': EventType.BATTLE_STARTED,
            'battle_ended': EventType.BATTLE_ENDED,
            'hp_critical': EventType.HP_CRITICAL
        }

        event_type = event_type_map.get(change_type)
        if event_type:
            event = Event(
                event_type=event_type,
                timestamp=time.time(),
                source=source,
                data=change_data,
                priority=7 if change_type in ['badge_earned', 'level_up'] else 6
            )
            self.publish(event)

    def _get_interested_subscribers(self, event: Event) -> List[EventSubscriber]:
        """Get all subscribers interested in this event"""
        interested = []

        # Direct subscribers
        interested.extend(self.subscribers.get(event.event_type, []))

        # Filtered subscribers
        for event_filter, subscriber in self.filtered_subscribers:
            if event_filter.matches(event):
                interested.append(subscriber)

        return interested

    def _deliver_sync(self, event: Event, subscribers: List[EventSubscriber]) -> None:
        """Deliver event synchronously"""
        for subscriber in subscribers:
            try:
                subscriber.handle_event(event)
            except Exception as e:
                self.logger.error(f"Error delivering event to {subscriber.__class__.__name__}: {e}")
                self._handle_error(event, e)

    def _deliver_async(self, event: Event, subscribers: List[EventSubscriber]) -> None:
        """Deliver event asynchronously (simplified implementation)"""
        # For now, deliver synchronously but could be enhanced with thread pool
        self._deliver_sync(event, subscribers)

    def _record_event(self, event: Event) -> None:
        """Record event for history and analytics"""
        self.event_history.append(event)
        self.event_stats[event.event_type] += 1

        # Track correlation if present
        if event.correlation_id:
            self.correlation_tracker[event.correlation_id].append(event)

            # Clean up old correlations
            if len(self.correlation_tracker[event.correlation_id]) > 100:
                self.correlation_tracker[event.correlation_id].pop(0)

    def _check_patterns(self, event: Event) -> None:
        """Check for event patterns"""
        for pattern_matcher in self.pattern_matchers:
            try:
                # Get recent events for pattern analysis
                recent_events = list(self.event_history)[-10:]
                pattern_matcher(recent_events)
            except Exception as e:
                self.logger.error(f"Error in pattern matcher: {e}")

    def _handle_error(self, event: Event, error: Exception) -> None:
        """Handle errors in event processing"""
        for error_handler in self.error_handlers:
            try:
                error_handler(event, error)
            except Exception as handler_error:
                self.logger.error(f"Error in error handler: {handler_error}")

    def add_error_handler(self, handler: Callable[[Event, Exception], None]) -> None:
        """Add error handler"""
        self.error_handlers.append(handler)

    def add_pattern_matcher(self, matcher: Callable[[List[Event]], None]) -> None:
        """Add pattern matcher"""
        self.pattern_matchers.append(matcher)

    def get_event_stats(self) -> Dict[str, Any]:
        """Get event system statistics"""
        return {
            'total_events': len(self.event_history),
            'event_type_counts': dict(self.event_stats),
            'subscribers_count': sum(len(subs) for subs in self.subscribers.values()),
            'filtered_subscribers_count': len(self.filtered_subscribers),
            'active_correlations': len(self.correlation_tracker)
        }

    def get_recent_events(self,
                         count: int = 10,
                         event_filter: Optional[EventFilter] = None) -> List[Event]:
        """Get recent events, optionally filtered"""
        events = list(self.event_history)[-count:]

        if event_filter:
            events = [e for e in events if event_filter.matches(e)]

        return events


class GameStateEventDetector:
    """Detects and publishes game state change events"""

    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.logger = logging.getLogger("GameStateEventDetector")
        self.last_state: Optional[Dict[str, Any]] = None

    def update(self, current_state: Dict[str, Any]) -> None:
        """Update with current game state and detect changes"""
        if self.last_state is not None:
            self.event_bus.publish_game_state_change(
                self.last_state,
                current_state,
                "game_state_detector"
            )

        self.last_state = current_state.copy()


class EventDrivenAnalytics(EventSubscriber):
    """Enhanced analytics system that responds to events"""

    def __init__(self):
        self.logger = logging.getLogger("EventDrivenAnalytics")
        self.metrics = {
            'battles_fought': 0,
            'battles_won': 0,
            'levels_gained': 0,
            'badges_earned': 0,
            'areas_discovered': 0,
            'critical_hp_events': 0,
            'agent_decisions': 0,
            'coordination_events': 0,
            'exploration_progress': 0,
            'quest_completions': 0
        }
        self.event_timeline = []
        self.agent_performance_tracker = {
            'battle': {'decisions': 0, 'successes': 0, 'failures': 0},
            'explorer': {'decisions': 0, 'areas_found': 0, 'items_found': 0},
            'progression': {'decisions': 0, 'milestones': 0, 'phase_advances': 0}
        }
        self.real_time_stats = {
            'events_per_minute': 0,
            'last_activity_timestamp': time.time(),
            'activity_periods': []
        }

    def get_subscribed_events(self) -> Set[EventType]:
        return {
            EventType.BATTLE_STARTED,
            EventType.BATTLE_ENDED,
            EventType.PLAYER_LEVEL_UP,
            EventType.BADGE_EARNED,
            EventType.LOCATION_CHANGED,
            EventType.HP_CRITICAL,
            EventType.NEW_AREA_DISCOVERED,
            EventType.ITEM_DISCOVERED,
            EventType.NPC_ENCOUNTERED,
            EventType.AGENT_DECISION,
            EventType.AGENT_ACTION_TAKEN,
            EventType.AGENT_PERFORMANCE_UPDATE,
            EventType.QUEST_COMPLETED,
            EventType.QUEST_STARTED,
            EventType.QUEST_FAILED
        }

    def handle_event(self, event: Event) -> None:
        """Handle analytics events"""
        # Update real-time activity tracking
        self._update_activity_tracking(event)

        # Update core metrics
        if event.event_type == EventType.BATTLE_STARTED:
            self.metrics['battles_fought'] += 1
        elif event.event_type == EventType.BATTLE_ENDED:
            if event.data.get('player_won', False):
                self.metrics['battles_won'] += 1
        elif event.event_type == EventType.PLAYER_LEVEL_UP:
            self.metrics['levels_gained'] += 1
        elif event.event_type == EventType.BADGE_EARNED:
            self.metrics['badges_earned'] += 1
        elif event.event_type == EventType.LOCATION_CHANGED:
            self.metrics['areas_discovered'] += 1
        elif event.event_type == EventType.HP_CRITICAL:
            self.metrics['critical_hp_events'] += 1
        elif event.event_type == EventType.NEW_AREA_DISCOVERED:
            self.metrics['exploration_progress'] += 1
        elif event.event_type == EventType.QUEST_COMPLETED:
            self.metrics['quest_completions'] += 1

        # Update agent-specific metrics
        self._update_agent_metrics(event)

        # Record detailed timeline
        self.event_timeline.append({
            'timestamp': event.timestamp,
            'event_type': event.event_type.value,
            'source': event.source,
            'priority': event.priority,
            'data': event.data
        })

        # Keep timeline manageable
        if len(self.event_timeline) > 500:
            self.event_timeline.pop(0)

    def _update_activity_tracking(self, event: Event) -> None:
        """Update real-time activity tracking"""
        current_time = time.time()

        # Update activity periods
        self.real_time_stats['activity_periods'].append(current_time)

        # Keep only last minute of activity
        one_minute_ago = current_time - 60
        self.real_time_stats['activity_periods'] = [
            t for t in self.real_time_stats['activity_periods']
            if t > one_minute_ago
        ]

        # Calculate events per minute
        self.real_time_stats['events_per_minute'] = len(self.real_time_stats['activity_periods'])
        self.real_time_stats['last_activity_timestamp'] = current_time

    def _update_agent_metrics(self, event: Event) -> None:
        """Update agent-specific performance metrics"""
        if event.event_type == EventType.AGENT_DECISION:
            self.metrics['agent_decisions'] += 1
            chosen_agent = event.data.get('chosen_agent', 'unknown')
            if chosen_agent in self.agent_performance_tracker:
                self.agent_performance_tracker[chosen_agent]['decisions'] += 1

        elif event.event_type == EventType.AGENT_PERFORMANCE_UPDATE:
            agent_type = event.data.get('agent_type', 'unknown')
            if agent_type in self.agent_performance_tracker:
                tracker = self.agent_performance_tracker[agent_type]

                # Battle agent specific tracking
                if agent_type == 'battle':
                    battle_result = event.data.get('battle_result')
                    if battle_result == 'won':
                        tracker['successes'] += 1
                    elif battle_result == 'lost':
                        tracker['failures'] += 1

                # Explorer agent specific tracking
                elif agent_type == 'explorer':
                    action_type = event.data.get('action_type')
                    if action_type == 'location_change':
                        tracker['areas_found'] += 1

                # Progression agent specific tracking
                elif agent_type == 'progression':
                    milestone_type = event.data.get('milestone_type')
                    if milestone_type in ['badge_earned', 'quest_completed']:
                        tracker['milestones'] += 1
                    elif milestone_type == 'phase_advancement':
                        tracker['phase_advances'] += 1

    def get_analytics_summary(self) -> Dict[str, Any]:
        """Get comprehensive analytics summary"""
        # Calculate derived metrics
        battle_win_rate = 0.0
        if self.metrics['battles_fought'] > 0:
            battle_win_rate = self.metrics['battles_won'] / self.metrics['battles_fought']

        agent_efficiency = {}
        for agent_type, tracker in self.agent_performance_tracker.items():
            if tracker['decisions'] > 0:
                if agent_type == 'battle' and (tracker['successes'] + tracker['failures']) > 0:
                    agent_efficiency[agent_type] = tracker['successes'] / (tracker['successes'] + tracker['failures'])
                else:
                    agent_efficiency[agent_type] = tracker.get('successes', 0) / tracker['decisions']
            else:
                agent_efficiency[agent_type] = 0.0

        return {
            'metrics': self.metrics,
            'derived_metrics': {
                'battle_win_rate': battle_win_rate,
                'exploration_rate': self.metrics['exploration_progress'] / max(self.metrics['areas_discovered'], 1),
                'quest_success_rate': self.metrics['quest_completions'] / max(self.metrics.get('quest_starts', 1), 1)
            },
            'agent_performance': self.agent_performance_tracker,
            'agent_efficiency': agent_efficiency,
            'real_time_stats': self.real_time_stats,
            'timeline_events': len(self.event_timeline),
            'recent_events': self.event_timeline[-5:] if self.event_timeline else [],
            'event_distribution': self._get_event_distribution()
        }

    def _get_event_distribution(self) -> Dict[str, int]:
        """Get distribution of recent event types"""
        distribution = {}
        for event_record in self.event_timeline[-50:]:  # Last 50 events
            event_type = event_record['event_type']
            distribution[event_type] = distribution.get(event_type, 0) + 1
        return distribution

    def get_performance_insights(self) -> Dict[str, Any]:
        """Get performance insights and recommendations"""
        insights = {
            'highlights': [],
            'concerns': [],
            'recommendations': []
        }

        # Battle performance insights
        if self.metrics['battles_fought'] > 5:
            win_rate = self.metrics['battles_won'] / self.metrics['battles_fought']
            if win_rate > 0.7:
                insights['highlights'].append(f"Excellent battle performance: {win_rate:.1%} win rate")
            elif win_rate < 0.3:
                insights['concerns'].append(f"Low battle win rate: {win_rate:.1%}")
                insights['recommendations'].append("Consider adjusting battle strategy or training approach")

        # Exploration insights
        if self.metrics['areas_discovered'] > 10:
            insights['highlights'].append(f"Good exploration progress: {self.metrics['areas_discovered']} areas discovered")

        # Agent coordination insights
        if self.metrics['agent_decisions'] > 20:
            coordination_rate = self.metrics['coordination_events'] / self.metrics['agent_decisions']
            if coordination_rate > 0.8:
                insights['highlights'].append("Effective multi-agent coordination")

        # Activity level insights
        if self.real_time_stats['events_per_minute'] > 30:
            insights['concerns'].append("High event volume - potential performance impact")
            insights['recommendations'].append("Consider event filtering or throttling")

        return insights


# Global event bus instance
_event_bus: Optional[EventBus] = None

def get_event_bus() -> EventBus:
    """Get the global event bus instance"""
    global _event_bus
    if _event_bus is None:
        _event_bus = EventBus()
    return _event_bus

def initialize_event_system() -> EventBus:
    """Initialize the global event system"""
    global _event_bus
    _event_bus = EventBus()

    # Add default analytics
    analytics = EventDrivenAnalytics()
    _event_bus.subscribe(analytics)

    return _event_bus