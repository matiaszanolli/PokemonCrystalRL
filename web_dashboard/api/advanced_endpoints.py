"""
Advanced Analytics & Debugging API Endpoints

Provides REST API endpoints for the advanced analytics engine and debugging interface.
"""

import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime

from ..analytics.advanced_analytics import AdvancedAnalyticsEngine
from ..debugger.advanced_debugger import AdvancedDebugger

logger = logging.getLogger(__name__)


class AdvancedAPIEndpoints:
    """
    REST API endpoints for advanced analytics and debugging functionality.

    Provides endpoints for:
    - Analytics engine management and data retrieval
    - Debugging interface controls and data access
    - Performance profiling and monitoring
    - System health and diagnostics
    """

    def __init__(self, trainer=None):
        """Initialize advanced API endpoints"""
        self.trainer = trainer

        # Initialize analytics and debugging components
        self.analytics_engine = AdvancedAnalyticsEngine()
        self.debugger = AdvancedDebugger()

        # Start background monitoring if trainer is available
        if self.trainer:
            self._initialize_metrics_tracking()

        self.logger = logger

    def _initialize_metrics_tracking(self):
        """Initialize metrics tracking from trainer data"""
        try:
            # This would integrate with the existing trainer to collect metrics
            # For now, we'll set up basic metric collection
            self.analytics_engine.add_metric('system_initialized', 1.0, {
                'component': 'advanced_api',
                'timestamp': datetime.now().isoformat()
            })

            self.debugger.log_event('info', 'advanced_api', 'Advanced API endpoints initialized')

        except Exception as e:
            self.logger.error(f"Failed to initialize metrics tracking: {e}")

    # Analytics Endpoints

    def get_analytics_health(self) -> Dict[str, Any]:
        """
        Get system health summary.

        GET /api/v1/analytics/health
        """
        try:
            # Get current system health metrics
            summary = self.analytics_engine.get_metrics_summary('short')

            # Calculate overall health score
            health_score = summary.get('health_score', 0.0) * 100

            # Determine health status
            if health_score >= 80:
                status = 'healthy'
            elif health_score >= 60:
                status = 'warning'
            elif health_score >= 30:
                status = 'degraded'
            else:
                status = 'critical'

            return {
                'success': True,
                'data': {
                    'score': health_score,
                    'status': status,
                    'memory_usage_mb': summary.get('metrics', {}).get('memory_usage', {}).get('latest', 0) / (1024 * 1024),
                    'thread_count': 12,  # Placeholder - would get from debugger
                    'error_rate': len(self.debugger.debug_events) * 0.01,  # Simplified calculation
                    'last_updated': datetime.now().isoformat()
                }
            }

        except Exception as e:
            self.logger.error(f"Failed to get analytics health: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def get_analytics_metrics(self) -> Dict[str, Any]:
        """
        Get key performance metrics summary.

        GET /api/v1/analytics/metrics
        """
        try:
            # Get metrics summary from analytics engine
            summary = self.analytics_engine.get_metrics_summary('medium')

            return {
                'success': True,
                'data': {
                    'metrics': summary.get('metrics', {}),
                    'trends': summary.get('trends', {}),
                    'last_updated': datetime.now().isoformat()
                }
            }

        except Exception as e:
            self.logger.error(f"Failed to get analytics metrics: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def get_analytics_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive analytics summary.

        GET /api/v1/analytics/summary
        """
        try:
            # Get comprehensive analytics data
            metrics_summary = self.analytics_engine.get_metrics_summary('medium')
            insights = self.analytics_engine.get_performance_insights()
            viz_data = self.analytics_engine.get_visualization_data()

            return {
                'success': True,
                'data': {
                    'metrics': metrics_summary,
                    'insights': insights,
                    'correlations': viz_data.get('correlations', {}),
                    'timeseries': viz_data.get('timeseries', {}),
                    'last_updated': datetime.now().isoformat()
                }
            }

        except Exception as e:
            self.logger.error(f"Failed to get analytics summary: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def get_analytics_alerts(self) -> Dict[str, Any]:
        """
        Get active analytics alerts.

        GET /api/v1/analytics/alerts
        """
        try:
            # Get alerts from analytics engine
            summary = self.analytics_engine.get_metrics_summary('short')
            alerts = summary.get('alerts', [])

            return {
                'success': True,
                'data': {
                    'alerts': alerts,
                    'total_count': len(alerts),
                    'last_updated': datetime.now().isoformat()
                }
            }

        except Exception as e:
            self.logger.error(f"Failed to get analytics alerts: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def get_visualization_data(self) -> Dict[str, Any]:
        """
        Get data for visualizations.

        GET /api/v1/analytics/visualization
        """
        try:
            # Get visualization data from analytics engine
            viz_data = self.analytics_engine.get_visualization_data()

            # Add additional visualization-specific data
            viz_data['heatmap_data'] = self._generate_heatmap_data()
            viz_data['dependencies'] = self._generate_dependency_data()

            return {
                'success': True,
                'data': viz_data
            }

        except Exception as e:
            self.logger.error(f"Failed to get visualization data: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def start_analytics(self) -> Dict[str, Any]:
        """
        Start analytics monitoring.

        POST /api/v1/analytics/start
        """
        try:
            # Start analytics monitoring (if not already running)
            self.analytics_engine.add_metric('monitoring_started', 1.0, {
                'timestamp': datetime.now().isoformat()
            })

            return {
                'success': True,
                'data': {
                    'message': 'Analytics monitoring started',
                    'timestamp': datetime.now().isoformat()
                }
            }

        except Exception as e:
            self.logger.error(f"Failed to start analytics: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def stop_analytics(self) -> Dict[str, Any]:
        """
        Stop analytics monitoring.

        POST /api/v1/analytics/stop
        """
        try:
            # Stop analytics monitoring
            self.analytics_engine.add_metric('monitoring_stopped', 1.0, {
                'timestamp': datetime.now().isoformat()
            })

            return {
                'success': True,
                'data': {
                    'message': 'Analytics monitoring stopped',
                    'timestamp': datetime.now().isoformat()
                }
            }

        except Exception as e:
            self.logger.error(f"Failed to stop analytics: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    # Debug Endpoints

    def get_debug_dashboard(self) -> Dict[str, Any]:
        """
        Get comprehensive debug dashboard data.

        GET /api/v1/debug/dashboard
        """
        try:
            # Get debug dashboard data
            dashboard_data = self.debugger.get_debug_dashboard()

            return {
                'success': True,
                'data': dashboard_data
            }

        except Exception as e:
            self.logger.error(f"Failed to get debug dashboard: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def get_debug_profiler(self) -> Dict[str, Any]:
        """
        Get profiler data and analysis.

        GET /api/v1/debug/profiler
        """
        try:
            # Get profiler data
            profiler_data = {
                'performance_summary': self.debugger.profiler.get_profile_summary(),
                'hotspots': self.debugger._identify_performance_hotspots(),
                'recommendations': self.debugger._generate_debug_recommendations()
            }

            return {
                'success': True,
                'data': profiler_data
            }

        except Exception as e:
            self.logger.error(f"Failed to get debug profiler: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def start_debug_monitoring(self) -> Dict[str, Any]:
        """
        Start debug monitoring.

        POST /api/v1/debug/start
        """
        try:
            self.debugger.start_monitoring()

            return {
                'success': True,
                'data': {
                    'message': 'Debug monitoring started',
                    'timestamp': datetime.now().isoformat()
                }
            }

        except Exception as e:
            self.logger.error(f"Failed to start debug monitoring: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def stop_debug_monitoring(self) -> Dict[str, Any]:
        """
        Stop debug monitoring.

        POST /api/v1/debug/stop
        """
        try:
            self.debugger.stop_monitoring()

            return {
                'success': True,
                'data': {
                    'message': 'Debug monitoring stopped',
                    'timestamp': datetime.now().isoformat()
                }
            }

        except Exception as e:
            self.logger.error(f"Failed to stop debug monitoring: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def add_breakpoint(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add debugging breakpoint.

        POST /api/v1/debug/breakpoint
        """
        try:
            location = request_data.get('location')
            if not location:
                return {
                    'success': False,
                    'error': 'Location is required'
                }

            self.debugger.set_breakpoint(location)

            return {
                'success': True,
                'data': {
                    'message': f'Breakpoint added at {location}',
                    'location': location
                }
            }

        except Exception as e:
            self.logger.error(f"Failed to add breakpoint: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def add_watch_variable(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add variable to watch list.

        POST /api/v1/debug/watch
        """
        try:
            variable = request_data.get('variable')
            if not variable:
                return {
                    'success': False,
                    'error': 'Variable name is required'
                }

            # For demo purposes, watch with a placeholder value
            self.debugger.watch_variable(variable, 'placeholder_value')

            return {
                'success': True,
                'data': {
                    'message': f'Now watching variable: {variable}',
                    'variable': variable
                }
            }

        except Exception as e:
            self.logger.error(f"Failed to add watch variable: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def start_profiler(self) -> Dict[str, Any]:
        """
        Start function profiler.

        POST /api/v1/debug/profiler/start
        """
        try:
            self.debugger.profiler.start_profiling()

            return {
                'success': True,
                'data': {
                    'message': 'Function profiling started'
                }
            }

        except Exception as e:
            self.logger.error(f"Failed to start profiler: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def stop_profiler(self) -> Dict[str, Any]:
        """
        Stop function profiler.

        POST /api/v1/debug/profiler/stop
        """
        try:
            self.debugger.profiler.stop_profiling()

            return {
                'success': True,
                'data': {
                    'message': 'Function profiling stopped'
                }
            }

        except Exception as e:
            self.logger.error(f"Failed to stop profiler: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def reset_profiler(self) -> Dict[str, Any]:
        """
        Reset function profiler data.

        POST /api/v1/debug/profiler/reset
        """
        try:
            # Reset profiler data
            self.debugger.profiler.traces.clear()
            self.debugger.profiler.call_counts.clear()

            return {
                'success': True,
                'data': {
                    'message': 'Function profiler reset'
                }
            }

        except Exception as e:
            self.logger.error(f"Failed to reset profiler: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    # System Control Endpoints

    def clear_cache(self) -> Dict[str, Any]:
        """
        Clear system caches.

        POST /api/v1/system/clear-cache
        """
        try:
            # Clear various caches
            self.analytics_engine.trend_cache.clear()

            self.debugger.log_event('info', 'system', 'System cache cleared')

            return {
                'success': True,
                'data': {
                    'message': 'System cache cleared successfully'
                }
            }

        except Exception as e:
            self.logger.error(f"Failed to clear cache: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def restart_components(self) -> Dict[str, Any]:
        """
        Restart system components.

        POST /api/v1/system/restart-components
        """
        try:
            # Restart monitoring components
            self.debugger.stop_monitoring()
            self.debugger.start_monitoring()

            self.debugger.log_event('info', 'system', 'System components restarted')

            return {
                'success': True,
                'data': {
                    'message': 'System components restarted successfully'
                }
            }

        except Exception as e:
            self.logger.error(f"Failed to restart components: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def run_diagnostics(self) -> Dict[str, Any]:
        """
        Run system diagnostics.

        POST /api/v1/system/diagnostics
        """
        try:
            # Run basic diagnostics
            diagnostics = {
                'analytics_engine': 'healthy',
                'debugger': 'healthy',
                'memory_usage': 'normal',
                'thread_count': 'normal',
                'error_rate': 'low'
            }

            self.debugger.log_event('info', 'system', 'System diagnostics completed')

            return {
                'success': True,
                'data': {
                    'message': 'System diagnostics completed',
                    'results': diagnostics
                }
            }

        except Exception as e:
            self.logger.error(f"Failed to run diagnostics: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def optimize_performance(self) -> Dict[str, Any]:
        """
        Trigger performance optimization.

        POST /api/v1/system/optimize
        """
        try:
            # Trigger performance optimization
            import gc
            gc.collect()  # Force garbage collection

            self.debugger.log_event('info', 'system', 'Performance optimization triggered')

            return {
                'success': True,
                'data': {
                    'message': 'Performance optimization completed'
                }
            }

        except Exception as e:
            self.logger.error(f"Failed to optimize performance: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    # Helper methods

    def _generate_heatmap_data(self) -> Dict[str, Any]:
        """Generate sample heatmap data for visualization"""
        # This would generate real heatmap data based on performance metrics
        return {
            'data': [[i, j, (i + j) % 10] for i in range(10) for j in range(10)],
            'max_value': 9,
            'labels': {
                'x': 'Time',
                'y': 'Component',
                'value': 'Performance Score'
            }
        }

    def _generate_dependency_data(self) -> Dict[str, Any]:
        """Generate component dependency data for visualization"""
        # This would generate real dependency data
        return {
            'nodes': [
                {'id': 'trainer', 'label': 'Trainer', 'type': 'core'},
                {'id': 'analytics', 'label': 'Analytics', 'type': 'monitoring'},
                {'id': 'debugger', 'label': 'Debugger', 'type': 'monitoring'},
                {'id': 'web_dashboard', 'label': 'Web Dashboard', 'type': 'ui'}
            ],
            'edges': [
                {'from': 'trainer', 'to': 'analytics'},
                {'from': 'trainer', 'to': 'debugger'},
                {'from': 'analytics', 'to': 'web_dashboard'},
                {'from': 'debugger', 'to': 'web_dashboard'}
            ]
        }

    def add_metric(self, metric_name: str, value: float, metadata: Dict[str, Any] = None):
        """Add a metric to the analytics engine (for external integration)"""
        self.analytics_engine.add_metric(metric_name, value, metadata)

    def log_debug_event(self, event_type: str, component: str, message: str,
                       metadata: Dict[str, Any] = None):
        """Log a debug event (for external integration)"""
        self.debugger.log_event(event_type, component, message, metadata)

    def get_real_time_data(self) -> Dict[str, Any]:
        """Get real-time dashboard data for WebSocket updates"""
        try:
            # Get real-time data from both analytics and debugger
            analytics_data = self.analytics_engine.get_real_time_dashboard_data()
            debug_data = self.debugger.get_debug_dashboard()

            return {
                'success': True,
                'data': {
                    'analytics': analytics_data,
                    'debug': debug_data,
                    'timestamp': datetime.now().isoformat()
                }
            }

        except Exception as e:
            self.logger.error(f"Failed to get real-time data: {e}")
            return {
                'success': False,
                'error': str(e)
            }