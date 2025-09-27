"""
Advanced Analytics Engine for Pokemon Crystal RL

Provides deep analytics, performance profiling, and visualization data
for enhanced monitoring and debugging capabilities.
"""

import numpy as np
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
import logging
import threading
import time

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetric:
    """Individual performance metric data point"""
    timestamp: float
    value: float
    metric_type: str
    metadata: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TrendAnalysis:
    """Trend analysis result"""
    metric_name: str
    trend_direction: str  # 'increasing', 'decreasing', 'stable'
    trend_strength: float  # 0.0 to 1.0
    slope: float
    r_squared: float
    prediction_next: float
    confidence_interval: Tuple[float, float]


@dataclass
class PerformanceAlert:
    """Performance alert data"""
    alert_id: str
    metric_name: str
    alert_type: str  # 'threshold', 'anomaly', 'trend'
    severity: str  # 'low', 'medium', 'high', 'critical'
    message: str
    timestamp: float
    resolved: bool = False
    metadata: Dict[str, Any] = None


class AdvancedAnalyticsEngine:
    """
    Advanced analytics engine providing deep insights into training performance,
    AI decision patterns, and system health.
    """

    def __init__(self, max_history_size: int = 10000):
        """Initialize analytics engine"""
        self.max_history_size = max_history_size

        # Data storage
        self.metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history_size))
        self.alerts: List[PerformanceAlert] = []
        self.trend_cache: Dict[str, TrendAnalysis] = {}

        # Configuration
        self.thresholds = {
            'actions_per_second': {'min': 0.1, 'max': 1000.0},
            'reward_rate': {'min': -10.0, 'max': 10.0},
            'llm_response_time': {'min': 0.0, 'max': 5.0},
            'memory_usage': {'min': 0.0, 'max': 4096.0},  # MB
            'error_rate': {'min': 0.0, 'max': 0.1}  # 10% max error rate
        }

        # Analysis windows
        self.analysis_windows = {
            'short': 300,    # 5 minutes
            'medium': 1800,  # 30 minutes
            'long': 3600     # 1 hour
        }

        # Thread safety
        self.lock = threading.RLock()

        self.logger = logger

    def add_metric(self, metric_name: str, value: float, metadata: Dict[str, Any] = None):
        """Add a new metric data point"""
        with self.lock:
            metric = PerformanceMetric(
                timestamp=time.time(),
                value=value,
                metric_type=metric_name,
                metadata=metadata or {}
            )

            self.metrics_history[metric_name].append(metric)

            # Check for alerts
            self._check_threshold_alerts(metric_name, value)
            self._check_anomaly_alerts(metric_name)

    def get_metrics_summary(self, time_window: str = 'medium') -> Dict[str, Any]:
        """Get comprehensive metrics summary for specified time window"""
        with self.lock:
            window_seconds = self.analysis_windows.get(time_window, 1800)
            cutoff_time = time.time() - window_seconds

            summary = {
                'time_window': time_window,
                'window_seconds': window_seconds,
                'metrics': {},
                'alerts': self._get_active_alerts(),
                'trends': {},
                'health_score': 0.0
            }

            total_metrics = 0
            healthy_metrics = 0

            for metric_name, history in self.metrics_history.items():
                if not history:
                    continue

                # Filter by time window
                recent_metrics = [m for m in history if m.timestamp >= cutoff_time]
                if not recent_metrics:
                    continue

                values = [m.value for m in recent_metrics]
                timestamps = [m.timestamp for m in recent_metrics]

                # Basic statistics
                metric_stats = {
                    'count': len(values),
                    'mean': np.mean(values),
                    'median': np.median(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'latest': values[-1] if values else None,
                    'change_rate': self._calculate_change_rate(values),
                    'percentiles': {
                        '25': np.percentile(values, 25),
                        '75': np.percentile(values, 75),
                        '95': np.percentile(values, 95),
                        '99': np.percentile(values, 99)
                    }
                }

                # Health assessment
                is_healthy = self._assess_metric_health(metric_name, metric_stats)
                if is_healthy:
                    healthy_metrics += 1
                total_metrics += 1

                metric_stats['healthy'] = is_healthy
                summary['metrics'][metric_name] = metric_stats

                # Trend analysis
                if len(values) >= 5:  # Minimum points for trend analysis
                    trend = self._analyze_trend(metric_name, timestamps, values)
                    summary['trends'][metric_name] = asdict(trend)

            # Overall health score
            if total_metrics > 0:
                summary['health_score'] = healthy_metrics / total_metrics

            return summary

    def get_performance_insights(self) -> Dict[str, Any]:
        """Get advanced performance insights and recommendations"""
        with self.lock:
            insights = {
                'optimization_opportunities': [],
                'performance_bottlenecks': [],
                'efficiency_metrics': {},
                'recommendations': [],
                'anomaly_detection': {}
            }

            # Analyze training efficiency
            insights['efficiency_metrics'] = self._analyze_training_efficiency()

            # Detect performance bottlenecks
            insights['performance_bottlenecks'] = self._detect_bottlenecks()

            # Generate optimization recommendations
            insights['recommendations'] = self._generate_recommendations()

            # Anomaly detection
            insights['anomaly_detection'] = self._detect_anomalies()

            return insights

    def get_visualization_data(self, metric_names: List[str] = None,
                             time_window: str = 'medium') -> Dict[str, Any]:
        """Get data formatted for visualization charts"""
        with self.lock:
            window_seconds = self.analysis_windows.get(time_window, 1800)
            cutoff_time = time.time() - window_seconds

            if metric_names is None:
                metric_names = list(self.metrics_history.keys())

            viz_data = {
                'timeseries': {},
                'histograms': {},
                'heatmaps': {},
                'correlations': {},
                'metadata': {
                    'time_window': time_window,
                    'start_time': cutoff_time,
                    'end_time': time.time()
                }
            }

            for metric_name in metric_names:
                if metric_name not in self.metrics_history:
                    continue

                history = self.metrics_history[metric_name]
                recent_metrics = [m for m in history if m.timestamp >= cutoff_time]

                if not recent_metrics:
                    continue

                # Time series data
                viz_data['timeseries'][metric_name] = {
                    'timestamps': [m.timestamp for m in recent_metrics],
                    'values': [m.value for m in recent_metrics],
                    'metadata': [m.metadata for m in recent_metrics]
                }

                # Histogram data
                values = [m.value for m in recent_metrics]
                hist, bin_edges = np.histogram(values, bins=20)
                viz_data['histograms'][metric_name] = {
                    'bins': bin_edges.tolist(),
                    'counts': hist.tolist()
                }

            # Cross-metric correlations
            viz_data['correlations'] = self._calculate_correlations(metric_names, cutoff_time)

            return viz_data

    def get_decision_analysis(self) -> Dict[str, Any]:
        """Analyze AI decision patterns and effectiveness"""
        with self.lock:
            analysis = {
                'decision_patterns': {},
                'effectiveness_metrics': {},
                'decision_timing': {},
                'context_analysis': {}
            }

            # Analyze LLM decision patterns
            if 'llm_decisions' in self.metrics_history:
                decisions = self.metrics_history['llm_decisions']
                analysis['decision_patterns'] = self._analyze_decision_patterns(decisions)

            # Analyze decision effectiveness
            analysis['effectiveness_metrics'] = self._analyze_decision_effectiveness()

            # Decision timing analysis
            analysis['decision_timing'] = self._analyze_decision_timing()

            return analysis

    def _calculate_change_rate(self, values: List[float]) -> float:
        """Calculate rate of change for a metric"""
        if len(values) < 2:
            return 0.0

        # Calculate percentage change from first to last value
        first, last = values[0], values[-1]
        if first == 0:
            return float('inf') if last > 0 else 0.0

        return ((last - first) / abs(first)) * 100

    def _analyze_trend(self, metric_name: str, timestamps: List[float],
                      values: List[float]) -> TrendAnalysis:
        """Analyze trend for a metric using linear regression"""
        if len(values) < 2:
            return TrendAnalysis(
                metric_name=metric_name,
                trend_direction='stable',
                trend_strength=0.0,
                slope=0.0,
                r_squared=0.0,
                prediction_next=values[-1] if values else 0.0,
                confidence_interval=(0.0, 0.0)
            )

        # Normalize timestamps to start from 0
        t_norm = np.array(timestamps) - timestamps[0]
        v_array = np.array(values)

        # Linear regression
        coeffs = np.polyfit(t_norm, v_array, 1)
        slope, intercept = coeffs[0], coeffs[1]

        # Calculate R-squared
        y_pred = np.polyval(coeffs, t_norm)
        ss_res = np.sum((v_array - y_pred) ** 2)
        ss_tot = np.sum((v_array - np.mean(v_array)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

        # Determine trend direction and strength
        if abs(slope) < 0.001:  # Threshold for "stable"
            trend_direction = 'stable'
            trend_strength = 0.0
        else:
            trend_direction = 'increasing' if slope > 0 else 'decreasing'
            trend_strength = min(abs(slope) / (np.std(values) + 1e-6), 1.0)

        # Predict next value
        next_time = t_norm[-1] + (t_norm[-1] - t_norm[-2]) if len(t_norm) > 1 else t_norm[-1] + 1
        prediction_next = slope * next_time + intercept

        # Simple confidence interval (±1 std)
        std_error = np.std(v_array - y_pred)
        confidence_interval = (prediction_next - std_error, prediction_next + std_error)

        return TrendAnalysis(
            metric_name=metric_name,
            trend_direction=trend_direction,
            trend_strength=trend_strength,
            slope=slope,
            r_squared=r_squared,
            prediction_next=prediction_next,
            confidence_interval=confidence_interval
        )

    def _assess_metric_health(self, metric_name: str, stats: Dict[str, Any]) -> bool:
        """Assess if a metric is within healthy bounds"""
        if metric_name not in self.thresholds:
            return True  # No thresholds defined, assume healthy

        thresholds = self.thresholds[metric_name]
        latest_value = stats.get('latest', 0)

        # Check thresholds
        if 'min' in thresholds and latest_value < thresholds['min']:
            return False
        if 'max' in thresholds and latest_value > thresholds['max']:
            return False

        # Check for excessive volatility
        if stats.get('std', 0) > stats.get('mean', 0) * 2:  # CV > 200%
            return False

        return True

    def _check_threshold_alerts(self, metric_name: str, value: float):
        """Check for threshold-based alerts"""
        if metric_name not in self.thresholds:
            return

        thresholds = self.thresholds[metric_name]
        alert_type = None
        severity = 'low'

        if 'min' in thresholds and value < thresholds['min']:
            alert_type = 'below_threshold'
            severity = 'medium' if value < thresholds['min'] * 0.5 else 'low'
        elif 'max' in thresholds and value > thresholds['max']:
            alert_type = 'above_threshold'
            severity = 'high' if value > thresholds['max'] * 1.5 else 'medium'

        if alert_type:
            alert = PerformanceAlert(
                alert_id=f"{metric_name}_{alert_type}_{time.time()}",
                metric_name=metric_name,
                alert_type='threshold',
                severity=severity,
                message=f"{metric_name} {alert_type}: {value:.2f}",
                timestamp=time.time(),
                metadata={'threshold': thresholds, 'value': value}
            )
            self.alerts.append(alert)

            # Keep only recent alerts
            self._cleanup_alerts()

    def _check_anomaly_alerts(self, metric_name: str):
        """Check for statistical anomalies"""
        history = self.metrics_history[metric_name]
        if len(history) < 10:  # Need minimum history for anomaly detection
            return

        values = [m.value for m in list(history)[-50:]]  # Last 50 values
        mean_val = np.mean(values)
        std_val = np.std(values)

        latest_value = values[-1]
        z_score = abs((latest_value - mean_val) / (std_val + 1e-6))

        if z_score > 3.0:  # 3-sigma rule
            alert = PerformanceAlert(
                alert_id=f"{metric_name}_anomaly_{time.time()}",
                metric_name=metric_name,
                alert_type='anomaly',
                severity='high' if z_score > 4.0 else 'medium',
                message=f"Anomalous value detected in {metric_name}: {latest_value:.2f} (z-score: {z_score:.2f})",
                timestamp=time.time(),
                metadata={'z_score': z_score, 'value': latest_value, 'mean': mean_val, 'std': std_val}
            )
            self.alerts.append(alert)

    def _get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get list of active (unresolved) alerts"""
        # Auto-resolve old alerts (older than 1 hour)
        current_time = time.time()
        for alert in self.alerts:
            if current_time - alert.timestamp > 3600:  # 1 hour
                alert.resolved = True

        active_alerts = [alert for alert in self.alerts if not alert.resolved]
        return [asdict(alert) for alert in active_alerts[-50:]]  # Last 50 alerts

    def _cleanup_alerts(self):
        """Clean up old alerts"""
        current_time = time.time()
        # Keep alerts from last 24 hours
        self.alerts = [alert for alert in self.alerts
                      if current_time - alert.timestamp < 86400]

    def _analyze_training_efficiency(self) -> Dict[str, Any]:
        """Analyze training efficiency metrics"""
        efficiency = {
            'actions_per_reward': 0.0,
            'llm_utilization': 0.0,
            'time_efficiency': 0.0,
            'resource_utilization': {}
        }

        # Calculate actions per reward
        if 'total_actions' in self.metrics_history and 'total_reward' in self.metrics_history:
            actions_hist = list(self.metrics_history['total_actions'])
            reward_hist = list(self.metrics_history['total_reward'])

            if actions_hist and reward_hist:
                latest_actions = actions_hist[-1].value
                latest_reward = reward_hist[-1].value

                if latest_reward != 0:
                    efficiency['actions_per_reward'] = latest_actions / latest_reward

        return efficiency

    def _detect_bottlenecks(self) -> List[Dict[str, Any]]:
        """Detect performance bottlenecks"""
        bottlenecks = []

        # Check for slow LLM responses
        if 'llm_response_time' in self.metrics_history:
            llm_times = [m.value for m in list(self.metrics_history['llm_response_time'])[-20:]]
            if llm_times and np.mean(llm_times) > 2.0:  # > 2 seconds average
                bottlenecks.append({
                    'type': 'llm_performance',
                    'severity': 'medium',
                    'description': f'LLM response time averaging {np.mean(llm_times):.2f}s',
                    'metric': 'llm_response_time',
                    'value': np.mean(llm_times)
                })

        # Check for low action rate
        if 'actions_per_second' in self.metrics_history:
            action_rates = [m.value for m in list(self.metrics_history['actions_per_second'])[-20:]]
            if action_rates and np.mean(action_rates) < 1.0:  # < 1 action/second
                bottlenecks.append({
                    'type': 'low_action_rate',
                    'severity': 'low',
                    'description': f'Low action rate: {np.mean(action_rates):.2f} actions/second',
                    'metric': 'actions_per_second',
                    'value': np.mean(action_rates)
                })

        return bottlenecks

    def _generate_recommendations(self) -> List[Dict[str, Any]]:
        """Generate optimization recommendations"""
        recommendations = []

        # Analyze trends for recommendations
        for metric_name, trend in self.trend_cache.items():
            if trend.trend_direction == 'decreasing' and trend.trend_strength > 0.5:
                if metric_name == 'actions_per_second':
                    recommendations.append({
                        'type': 'performance',
                        'priority': 'high',
                        'title': 'Improve Action Rate',
                        'description': 'Action rate is declining. Consider optimizing the training loop or reducing LLM interval.',
                        'metric': metric_name
                    })
                elif metric_name == 'total_reward':
                    recommendations.append({
                        'type': 'training',
                        'priority': 'medium',
                        'title': 'Optimize Reward Strategy',
                        'description': 'Reward is declining. Consider adjusting agent strategy or reward function.',
                        'metric': metric_name
                    })

        return recommendations

    def _detect_anomalies(self) -> Dict[str, Any]:
        """Advanced anomaly detection"""
        anomalies = {
            'statistical_anomalies': [],
            'pattern_breaks': [],
            'correlation_anomalies': []
        }

        # Statistical anomalies (already handled in alerts)
        # Pattern breaks (sudden changes in behavior)
        # Correlation anomalies (metrics that should correlate but don't)

        return anomalies

    def _calculate_correlations(self, metric_names: List[str], cutoff_time: float) -> Dict[str, Any]:
        """Calculate cross-metric correlations"""
        correlations = {}

        # Get recent data for all metrics
        metric_data = {}
        for metric_name in metric_names:
            if metric_name in self.metrics_history:
                recent_metrics = [m for m in self.metrics_history[metric_name]
                               if m.timestamp >= cutoff_time]
                if len(recent_metrics) > 5:  # Minimum data for correlation
                    metric_data[metric_name] = [m.value for m in recent_metrics]

        # Calculate pairwise correlations
        for i, metric1 in enumerate(metric_data.keys()):
            for metric2 in list(metric_data.keys())[i+1:]:
                values1 = metric_data[metric1]
                values2 = metric_data[metric2]

                # Align lengths
                min_len = min(len(values1), len(values2))
                if min_len > 5:
                    corr_coef = np.corrcoef(values1[-min_len:], values2[-min_len:])[0, 1]
                    if not np.isnan(corr_coef):
                        correlations[f"{metric1}_vs_{metric2}"] = {
                            'correlation': float(corr_coef),
                            'strength': 'strong' if abs(corr_coef) > 0.7 else 'moderate' if abs(corr_coef) > 0.3 else 'weak'
                        }

        return correlations

    def _analyze_decision_patterns(self, decisions: deque) -> Dict[str, Any]:
        """Analyze patterns in AI decision making"""
        if not decisions:
            return {}

        recent_decisions = list(decisions)[-100:]  # Last 100 decisions

        patterns = {
            'decision_frequency': len(recent_decisions),
            'average_confidence': 0.0,
            'decision_types': defaultdict(int),
            'temporal_patterns': {}
        }

        # Analyze decision metadata if available
        for decision in recent_decisions:
            if decision.metadata:
                decision_type = decision.metadata.get('action_type', 'unknown')
                patterns['decision_types'][decision_type] += 1

                confidence = decision.metadata.get('confidence', 0.0)
                patterns['average_confidence'] += confidence

        if recent_decisions:
            patterns['average_confidence'] /= len(recent_decisions)

        return patterns

    def _analyze_decision_effectiveness(self) -> Dict[str, Any]:
        """Analyze effectiveness of AI decisions"""
        effectiveness = {
            'reward_correlation': 0.0,
            'success_rate': 0.0,
            'improvement_trend': 'stable'
        }

        # Correlate decisions with reward changes
        if 'llm_decisions' in self.metrics_history and 'total_reward' in self.metrics_history:
            # Implementation would analyze reward changes following LLM decisions
            pass

        return effectiveness

    def _analyze_decision_timing(self) -> Dict[str, Any]:
        """Analyze timing patterns of AI decisions"""
        timing = {
            'average_interval': 0.0,
            'interval_variance': 0.0,
            'optimal_interval_suggestion': 0.0
        }

        if 'llm_decisions' in self.metrics_history:
            decisions = list(self.metrics_history['llm_decisions'])
            if len(decisions) > 1:
                intervals = []
                for i in range(1, len(decisions)):
                    interval = decisions[i].timestamp - decisions[i-1].timestamp
                    intervals.append(interval)

                if intervals:
                    timing['average_interval'] = np.mean(intervals)
                    timing['interval_variance'] = np.var(intervals)

        return timing

    def get_real_time_dashboard_data(self) -> Dict[str, Any]:
        """Get real-time data optimized for dashboard display"""
        with self.lock:
            current_time = time.time()
            dashboard_data = {
                'timestamp': current_time,
                'summary': self.get_metrics_summary('short'),
                'alerts': self._get_active_alerts()[-5:],  # Last 5 alerts
                'key_metrics': {},
                'health_indicators': {}
            }

            # Key metrics for dashboard display
            key_metric_names = ['actions_per_second', 'total_reward', 'llm_response_time', 'memory_usage']
            for metric_name in key_metric_names:
                if metric_name in self.metrics_history and self.metrics_history[metric_name]:
                    latest = self.metrics_history[metric_name][-1]
                    dashboard_data['key_metrics'][metric_name] = {
                        'value': latest.value,
                        'timestamp': latest.timestamp,
                        'healthy': self._assess_metric_health(metric_name, {'latest': latest.value})
                    }

            return dashboard_data