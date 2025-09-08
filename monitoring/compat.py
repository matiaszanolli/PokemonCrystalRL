"""
Monitoring System Compatibility Layer

This module provides backward compatibility with the old monitoring system
while the transition to the new architecture is completed.
"""

from typing import Optional, Dict, Any

from .data.bus import DataBus, get_data_bus
from .components.web import WebServer
from .components.metrics import MetricsCollector
from .base import MonitoringConfig

# Re-export old names with new implementations
DatabaseManager = None  # To be implemented in Phase 4.2
try:
    from .unified_monitor import UnifiedMonitor
except ImportError:
    UnifiedMonitor = None  # Fallback if not available
GameStreamer = None  # Now part of ScreenCapture
MonitoringClient = None  # Now part of MetricsCollector
StatsCollector = None  # Now part of MetricsCollector
TextLogger = None  # To be implemented in Phase 4.2
try:
    from .trainer_monitor_bridge import TrainerMonitorBridge
except ImportError:
    TrainerMonitorBridge = None  # Fallback if not available
WebInterface = WebServer  # Now implemented as WebServer
try:
    from .error_handler import ErrorHandler, ErrorSeverity, ErrorCategory, RecoveryStrategy, ErrorEvent
except ImportError:
    ErrorHandler = None
    ErrorSeverity = None
    ErrorCategory = None
    RecoveryStrategy = None
    ErrorEvent = None

# Compatibility warning
import warnings
warnings.warn(
    "Using compatibility layer for monitoring system. "
    "Please migrate to new monitoring.components.* imports.",
    DeprecationWarning,
    stacklevel=2
)
