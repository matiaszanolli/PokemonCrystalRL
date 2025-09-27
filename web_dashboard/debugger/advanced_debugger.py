"""
Advanced Debugging Interface for Pokemon Crystal RL

Provides comprehensive debugging capabilities including:
- Real-time system monitoring
- Memory inspection and analysis
- AI decision tracing
- Performance profiling
- Interactive debugging tools
"""

import json
import time
import traceback
import threading
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
import logging
import inspect
import sys
import gc

logger = logging.getLogger(__name__)


@dataclass
class DebugEvent:
    """Debug event data structure"""
    event_id: str
    timestamp: float
    event_type: str  # 'error', 'warning', 'info', 'trace', 'performance'
    component: str
    message: str
    stack_trace: Optional[str] = None
    metadata: Dict[str, Any] = None
    resolved: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PerformanceTrace:
    """Performance trace for function execution"""
    function_name: str
    module: str
    start_time: float
    end_time: float
    duration: float
    memory_before: int
    memory_after: int
    call_count: int
    metadata: Dict[str, Any] = None


@dataclass
class SystemState:
    """Current system state snapshot"""
    timestamp: float
    cpu_usage: float
    memory_usage: int
    thread_count: int
    active_connections: int
    queue_sizes: Dict[str, int]
    component_status: Dict[str, str]
    error_count: int
    warning_count: int


class FunctionProfiler:
    """Function-level performance profiler"""

    def __init__(self):
        self.traces: Dict[str, List[PerformanceTrace]] = defaultdict(list)
        self.active_calls: Dict[str, float] = {}
        self.call_counts: Dict[str, int] = defaultdict(int)
        self.enabled = False

    def start_profiling(self):
        """Start function profiling"""
        self.enabled = True
        logger.info("Function profiling started")

    def stop_profiling(self):
        """Stop function profiling"""
        self.enabled = False
        logger.info("Function profiling stopped")

    def profile_function(self, func: Callable) -> Callable:
        """Decorator to profile function execution"""
        def wrapper(*args, **kwargs):
            if not self.enabled:
                return func(*args, **kwargs)

            func_name = f"{func.__module__}.{func.__name__}"
            start_time = time.time()
            memory_before = self._get_memory_usage()

            try:
                result = func(*args, **kwargs)
                return result
            finally:
                end_time = time.time()
                memory_after = self._get_memory_usage()
                duration = end_time - start_time

                self.call_counts[func_name] += 1

                trace = PerformanceTrace(
                    function_name=func_name,
                    module=func.__module__,
                    start_time=start_time,
                    end_time=end_time,
                    duration=duration,
                    memory_before=memory_before,
                    memory_after=memory_after,
                    call_count=self.call_counts[func_name]
                )

                self.traces[func_name].append(trace)

                # Keep only recent traces
                if len(self.traces[func_name]) > 1000:
                    self.traces[func_name] = self.traces[func_name][-500:]

        return wrapper

    def _get_memory_usage(self) -> int:
        """Get current memory usage in bytes"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss
        except ImportError:
            # Fallback to garbage collector stats
            return sum(sys.getsizeof(obj) for obj in gc.get_objects())

    def get_profile_summary(self, function_name: Optional[str] = None) -> Dict[str, Any]:
        """Get profiling summary"""
        if function_name:
            traces = self.traces.get(function_name, [])
            return self._analyze_function_traces(function_name, traces)
        else:
            summary = {}
            for func_name, traces in self.traces.items():
                summary[func_name] = self._analyze_function_traces(func_name, traces)
            return summary

    def _analyze_function_traces(self, func_name: str, traces: List[PerformanceTrace]) -> Dict[str, Any]:
        """Analyze traces for a specific function"""
        if not traces:
            return {}

        durations = [trace.duration for trace in traces]
        memory_deltas = [trace.memory_after - trace.memory_before for trace in traces]

        return {
            'function_name': func_name,
            'call_count': len(traces),
            'total_time': sum(durations),
            'average_time': sum(durations) / len(durations),
            'min_time': min(durations),
            'max_time': max(durations),
            'average_memory_delta': sum(memory_deltas) / len(memory_deltas) if memory_deltas else 0,
            'last_called': max(trace.end_time for trace in traces),
            'performance_trend': self._calculate_performance_trend(durations)
        }

    def _calculate_performance_trend(self, durations: List[float]) -> str:
        """Calculate performance trend (improving/degrading/stable)"""
        if len(durations) < 5:
            return 'insufficient_data'

        recent = durations[-10:]  # Last 10 calls
        earlier = durations[-20:-10] if len(durations) >= 20 else durations[:-10]

        if not earlier:
            return 'insufficient_data'

        recent_avg = sum(recent) / len(recent)
        earlier_avg = sum(earlier) / len(earlier)

        improvement = (earlier_avg - recent_avg) / earlier_avg

        if improvement > 0.1:
            return 'improving'
        elif improvement < -0.1:
            return 'degrading'
        else:
            return 'stable'


class AdvancedDebugger:
    """
    Advanced debugging interface providing comprehensive system monitoring,
    error tracking, performance analysis, and interactive debugging tools.
    """

    def __init__(self, max_events: int = 10000):
        """Initialize advanced debugger"""
        self.max_events = max_events

        # Event storage
        self.debug_events: deque = deque(maxlen=max_events)
        self.system_states: deque = deque(maxlen=1000)

        # Component tracking
        self.component_health: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.error_patterns: Dict[str, int] = defaultdict(int)

        # Performance monitoring
        self.profiler = FunctionProfiler()
        self.performance_monitors: Dict[str, Callable] = {}

        # Real-time monitoring
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None

        # Debugging tools
        self.breakpoints: Set[str] = set()
        self.watch_variables: Dict[str, Any] = {}
        self.debug_flags: Dict[str, bool] = defaultdict(bool)

        # Thread safety
        self.lock = threading.RLock()

        self.logger = logger

    def start_monitoring(self):
        """Start real-time system monitoring"""
        with self.lock:
            if self.monitoring_active:
                return

            self.monitoring_active = True
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitoring_thread.start()
            self.profiler.start_profiling()

            self.log_event('info', 'debugger', 'Advanced debugging monitoring started')

    def stop_monitoring(self):
        """Stop real-time system monitoring"""
        with self.lock:
            self.monitoring_active = False
            self.profiler.stop_profiling()

            if self.monitoring_thread:
                self.monitoring_thread.join(timeout=1.0)

            self.log_event('info', 'debugger', 'Advanced debugging monitoring stopped')

    def log_event(self, event_type: str, component: str, message: str,
                  metadata: Dict[str, Any] = None, include_stack: bool = False):
        """Log a debug event"""
        with self.lock:
            event_id = f"{component}_{event_type}_{time.time()}"

            stack_trace = None
            if include_stack or event_type == 'error':
                stack_trace = traceback.format_stack()

            event = DebugEvent(
                event_id=event_id,
                timestamp=time.time(),
                event_type=event_type,
                component=component,
                message=message,
                stack_trace=stack_trace,
                metadata=metadata or {},
                resolved=False
            )

            self.debug_events.append(event)

            # Track error patterns
            if event_type == 'error':
                error_key = f"{component}:{message[:50]}"
                self.error_patterns[error_key] += 1

            # Update component health
            self._update_component_health(component, event_type)

    def get_debug_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive debug dashboard data"""
        with self.lock:
            current_time = time.time()

            dashboard = {
                'timestamp': current_time,
                'monitoring_active': self.monitoring_active,
                'system_health': self._get_system_health(),
                'recent_events': self._get_recent_events(50),
                'error_summary': self._get_error_summary(),
                'performance_summary': self.profiler.get_profile_summary(),
                'component_status': dict(self.component_health),
                'memory_analysis': self._get_memory_analysis(),
                'thread_analysis': self._get_thread_analysis(),
                'alerts': self._get_debug_alerts()
            }

            return dashboard

    def get_detailed_analysis(self, component: str = None) -> Dict[str, Any]:
        """Get detailed analysis for specific component or overall system"""
        with self.lock:
            analysis = {
                'error_analysis': self._analyze_errors(component),
                'performance_analysis': self._analyze_performance(component),
                'memory_analysis': self._analyze_memory_usage(component),
                'timing_analysis': self._analyze_timing_patterns(component),
                'recommendations': self._generate_debug_recommendations(component)
            }

            return analysis

    def get_interactive_debug_session(self) -> Dict[str, Any]:
        """Get interactive debugging session data"""
        with self.lock:
            session = {
                'breakpoints': list(self.breakpoints),
                'watch_variables': dict(self.watch_variables),
                'debug_flags': dict(self.debug_flags),
                'system_state': self._capture_system_state(),
                'call_stack': self._get_current_call_stack(),
                'variable_inspector': self._get_variable_inspector(),
                'performance_hotspots': self._identify_performance_hotspots()
            }

            return session

    def set_breakpoint(self, location: str):
        """Set a debugging breakpoint"""
        with self.lock:
            self.breakpoints.add(location)
            self.log_event('info', 'debugger', f'Breakpoint set at {location}')

    def remove_breakpoint(self, location: str):
        """Remove a debugging breakpoint"""
        with self.lock:
            self.breakpoints.discard(location)
            self.log_event('info', 'debugger', f'Breakpoint removed from {location}')

    def watch_variable(self, name: str, value: Any):
        """Add variable to watch list"""
        with self.lock:
            self.watch_variables[name] = {
                'value': value,
                'type': type(value).__name__,
                'timestamp': time.time(),
                'size': sys.getsizeof(value)
            }

    def set_debug_flag(self, flag: str, enabled: bool):
        """Set a debug flag for conditional debugging"""
        with self.lock:
            self.debug_flags[flag] = enabled
            self.log_event('info', 'debugger', f'Debug flag {flag} set to {enabled}')

    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                # Capture system state
                state = self._capture_system_state()
                self.system_states.append(state)

                # Check for anomalies
                self._check_system_anomalies(state)

                # Update component health
                self._refresh_component_health()

                time.sleep(1.0)  # Monitor every second

            except Exception as e:
                self.log_event('error', 'debugger', f'Monitoring loop error: {e}', include_stack=True)
                time.sleep(5.0)  # Back off on error

    def _capture_system_state(self) -> SystemState:
        """Capture current system state"""
        try:
            import psutil
            process = psutil.Process()

            return SystemState(
                timestamp=time.time(),
                cpu_usage=process.cpu_percent(),
                memory_usage=process.memory_info().rss,
                thread_count=threading.active_count(),
                active_connections=0,  # Would need connection tracking
                queue_sizes={},  # Would need queue tracking
                component_status=self._get_component_status(),
                error_count=len([e for e in self.debug_events if e.event_type == 'error']),
                warning_count=len([e for e in self.debug_events if e.event_type == 'warning'])
            )
        except ImportError:
            # Fallback without psutil
            return SystemState(
                timestamp=time.time(),
                cpu_usage=0.0,
                memory_usage=0,
                thread_count=threading.active_count(),
                active_connections=0,
                queue_sizes={},
                component_status=self._get_component_status(),
                error_count=len([e for e in self.debug_events if e.event_type == 'error']),
                warning_count=len([e for e in self.debug_events if e.event_type == 'warning'])
            )

    def _get_component_status(self) -> Dict[str, str]:
        """Get status of all components"""
        status = {}
        for component, health in self.component_health.items():
            error_count = health.get('error_count', 0)
            last_activity = health.get('last_activity', 0)
            current_time = time.time()

            if error_count > 5:
                status[component] = 'critical'
            elif error_count > 0:
                status[component] = 'warning'
            elif current_time - last_activity > 300:  # 5 minutes
                status[component] = 'inactive'
            else:
                status[component] = 'healthy'

        return status

    def _update_component_health(self, component: str, event_type: str):
        """Update component health tracking"""
        if component not in self.component_health:
            self.component_health[component] = {
                'error_count': 0,
                'warning_count': 0,
                'info_count': 0,
                'last_activity': time.time(),
                'first_seen': time.time()
            }

        health = self.component_health[component]
        health['last_activity'] = time.time()

        if event_type == 'error':
            health['error_count'] += 1
        elif event_type == 'warning':
            health['warning_count'] += 1
        elif event_type == 'info':
            health['info_count'] += 1

    def _get_system_health(self) -> Dict[str, Any]:
        """Calculate overall system health"""
        if not self.system_states:
            return {'status': 'unknown', 'score': 0.0}

        recent_states = list(self.system_states)[-10:]  # Last 10 states
        if not recent_states:
            return {'status': 'unknown', 'score': 0.0}

        # Calculate health metrics
        avg_memory = sum(s.memory_usage for s in recent_states) / len(recent_states)
        error_rate = sum(s.error_count for s in recent_states) / len(recent_states)
        thread_count = recent_states[-1].thread_count

        # Health score (0-100)
        health_score = 100.0

        # Penalize high memory usage (> 2GB)
        if avg_memory > 2 * 1024 * 1024 * 1024:
            health_score -= 30

        # Penalize errors
        health_score -= min(error_rate * 10, 40)

        # Penalize too many threads (> 50)
        if thread_count > 50:
            health_score -= 20

        health_score = max(0.0, health_score)

        if health_score >= 80:
            status = 'healthy'
        elif health_score >= 60:
            status = 'warning'
        elif health_score >= 30:
            status = 'degraded'
        else:
            status = 'critical'

        return {
            'status': status,
            'score': health_score,
            'memory_usage_mb': avg_memory / (1024 * 1024),
            'error_rate': error_rate,
            'thread_count': thread_count
        }

    def _get_recent_events(self, count: int) -> List[Dict[str, Any]]:
        """Get recent debug events"""
        recent_events = list(self.debug_events)[-count:]
        return [event.to_dict() for event in recent_events]

    def _get_error_summary(self) -> Dict[str, Any]:
        """Get error summary and patterns"""
        errors = [e for e in self.debug_events if e.event_type == 'error']
        recent_errors = [e for e in errors if time.time() - e.timestamp < 3600]  # Last hour

        return {
            'total_errors': len(errors),
            'recent_errors': len(recent_errors),
            'error_patterns': dict(self.error_patterns),
            'most_common_errors': sorted(self.error_patterns.items(),
                                       key=lambda x: x[1], reverse=True)[:5]
        }

    def _get_memory_analysis(self) -> Dict[str, Any]:
        """Analyze memory usage patterns"""
        if not self.system_states:
            return {}

        memory_values = [s.memory_usage for s in self.system_states]
        return {
            'current_mb': memory_values[-1] / (1024 * 1024) if memory_values else 0,
            'average_mb': sum(memory_values) / len(memory_values) / (1024 * 1024) if memory_values else 0,
            'peak_mb': max(memory_values) / (1024 * 1024) if memory_values else 0,
            'trend': self._calculate_memory_trend(memory_values)
        }

    def _calculate_memory_trend(self, memory_values: List[int]) -> str:
        """Calculate memory usage trend"""
        if len(memory_values) < 5:
            return 'insufficient_data'

        recent = memory_values[-5:]
        earlier = memory_values[-10:-5] if len(memory_values) >= 10 else memory_values[:-5]

        if not earlier:
            return 'insufficient_data'

        recent_avg = sum(recent) / len(recent)
        earlier_avg = sum(earlier) / len(earlier)

        change = (recent_avg - earlier_avg) / earlier_avg

        if change > 0.1:
            return 'increasing'
        elif change < -0.1:
            return 'decreasing'
        else:
            return 'stable'

    def _get_thread_analysis(self) -> Dict[str, Any]:
        """Analyze thread usage"""
        active_threads = threading.enumerate()
        return {
            'total_threads': len(active_threads),
            'daemon_threads': len([t for t in active_threads if t.daemon]),
            'alive_threads': len([t for t in active_threads if t.is_alive()]),
            'thread_names': [t.name for t in active_threads]
        }

    def _get_debug_alerts(self) -> List[Dict[str, Any]]:
        """Get active debug alerts"""
        alerts = []
        current_time = time.time()

        # Check for high error rates
        recent_errors = [e for e in self.debug_events
                        if e.event_type == 'error' and current_time - e.timestamp < 300]
        if len(recent_errors) > 10:  # More than 10 errors in 5 minutes
            alerts.append({
                'type': 'high_error_rate',
                'severity': 'high',
                'message': f'High error rate: {len(recent_errors)} errors in last 5 minutes'
            })

        # Check for memory leaks
        if len(self.system_states) > 10:
            memory_trend = self._calculate_memory_trend([s.memory_usage for s in self.system_states])
            if memory_trend == 'increasing':
                alerts.append({
                    'type': 'memory_leak',
                    'severity': 'medium',
                    'message': 'Potential memory leak detected (increasing memory usage)'
                })

        return alerts

    def _analyze_errors(self, component: str = None) -> Dict[str, Any]:
        """Analyze error patterns"""
        errors = [e for e in self.debug_events if e.event_type == 'error']

        if component:
            errors = [e for e in errors if e.component == component]

        if not errors:
            return {}

        # Group errors by message pattern
        error_groups = defaultdict(list)
        for error in errors:
            # Group by first 50 characters of message
            group_key = error.message[:50]
            error_groups[group_key].append(error)

        return {
            'total_errors': len(errors),
            'unique_error_types': len(error_groups),
            'error_frequency': {key: len(group) for key, group in error_groups.items()},
            'recent_errors': len([e for e in errors if time.time() - e.timestamp < 3600])
        }

    def _analyze_performance(self, component: str = None) -> Dict[str, Any]:
        """Analyze performance patterns"""
        profile_data = self.profiler.get_profile_summary()

        if component:
            # Filter by component
            profile_data = {k: v for k, v in profile_data.items() if component in k}

        if not profile_data:
            return {}

        # Find slowest functions
        slowest_functions = sorted(profile_data.items(),
                                 key=lambda x: x[1].get('average_time', 0), reverse=True)

        return {
            'monitored_functions': len(profile_data),
            'slowest_functions': slowest_functions[:5],
            'performance_trends': {k: v.get('performance_trend', 'unknown')
                                 for k, v in profile_data.items()}
        }

    def _analyze_memory_usage(self, component: str = None) -> Dict[str, Any]:
        """Analyze memory usage patterns"""
        # This would analyze memory usage by component
        return self._get_memory_analysis()

    def _analyze_timing_patterns(self, component: str = None) -> Dict[str, Any]:
        """Analyze timing patterns in events"""
        events = list(self.debug_events)
        if component:
            events = [e for e in events if e.component == component]

        if len(events) < 2:
            return {}

        # Calculate intervals between events
        intervals = []
        for i in range(1, len(events)):
            interval = events[i].timestamp - events[i-1].timestamp
            intervals.append(interval)

        return {
            'average_interval': sum(intervals) / len(intervals) if intervals else 0,
            'min_interval': min(intervals) if intervals else 0,
            'max_interval': max(intervals) if intervals else 0,
            'event_rate_per_minute': len(events) / ((events[-1].timestamp - events[0].timestamp) / 60) if len(events) > 1 else 0
        }

    def _generate_debug_recommendations(self, component: str = None) -> List[Dict[str, Any]]:
        """Generate debugging recommendations"""
        recommendations = []

        # Analyze error patterns
        errors = [e for e in self.debug_events if e.event_type == 'error']
        if component:
            errors = [e for e in errors if e.component == component]

        if len(errors) > 10:
            recommendations.append({
                'type': 'error_reduction',
                'priority': 'high',
                'title': 'High Error Rate',
                'description': f'Component has {len(errors)} errors. Review error patterns and implement better error handling.'
            })

        # Check performance
        profile_data = self.profiler.get_profile_summary()
        slow_functions = [k for k, v in profile_data.items()
                         if v.get('average_time', 0) > 1.0]  # > 1 second

        if slow_functions:
            recommendations.append({
                'type': 'performance',
                'priority': 'medium',
                'title': 'Slow Functions Detected',
                'description': f'Functions {slow_functions[:3]} are running slowly. Consider optimization.'
            })

        return recommendations

    def _check_system_anomalies(self, state: SystemState):
        """Check for system anomalies and alert"""
        # High memory usage
        if state.memory_usage > 4 * 1024 * 1024 * 1024:  # > 4GB
            self.log_event('warning', 'system', f'High memory usage: {state.memory_usage / (1024**3):.1f}GB')

        # Too many threads
        if state.thread_count > 100:
            self.log_event('warning', 'system', f'High thread count: {state.thread_count}')

        # High error rate
        if len(self.system_states) > 1:
            prev_state = self.system_states[-2]
            error_increase = state.error_count - prev_state.error_count
            if error_increase > 5:  # More than 5 new errors
                self.log_event('warning', 'system', f'Error rate spike: {error_increase} new errors')

    def _refresh_component_health(self):
        """Refresh component health status"""
        current_time = time.time()
        for component, health in self.component_health.items():
            # Reset error counts if no recent errors
            if current_time - health.get('last_activity', 0) > 3600:  # 1 hour
                health['error_count'] = 0
                health['warning_count'] = 0

    def _get_current_call_stack(self) -> List[Dict[str, Any]]:
        """Get current call stack information"""
        stack = traceback.extract_stack()
        return [
            {
                'filename': frame.filename,
                'function': frame.name,
                'line_number': frame.lineno,
                'code': frame.line
            }
            for frame in stack[-10:]  # Last 10 frames
        ]

    def _get_variable_inspector(self) -> Dict[str, Any]:
        """Get variable inspector data"""
        inspector = {
            'watched_variables': dict(self.watch_variables),
            'global_variables': {},
            'memory_objects': {}
        }

        # Sample some global variables (be careful not to expose sensitive data)
        frame = inspect.currentframe()
        if frame and frame.f_back:
            globals_dict = frame.f_back.f_globals
            safe_globals = {k: str(v)[:100] for k, v in globals_dict.items()
                           if not k.startswith('_') and k in ['__name__', '__file__']}
            inspector['global_variables'] = safe_globals

        return inspector

    def _identify_performance_hotspots(self) -> List[Dict[str, Any]]:
        """Identify performance hotspots"""
        profile_data = self.profiler.get_profile_summary()

        hotspots = []
        for func_name, data in profile_data.items():
            total_time = data.get('total_time', 0)
            call_count = data.get('call_count', 0)
            avg_time = data.get('average_time', 0)

            if total_time > 10.0 or avg_time > 1.0 or call_count > 1000:
                hotspots.append({
                    'function': func_name,
                    'total_time': total_time,
                    'average_time': avg_time,
                    'call_count': call_count,
                    'severity': 'high' if avg_time > 2.0 else 'medium'
                })

        return sorted(hotspots, key=lambda x: x['total_time'], reverse=True)[:10]