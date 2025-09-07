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
UnifiedMonitor = None  # To be implemented in Phase 4.2
GameStreamer = None  # Now part of ScreenCapture
MonitoringClient = None  # Now part of MetricsCollector
StatsCollector = None  # Now part of MetricsCollector
TextLogger = None  # To be implemented in Phase 4.2
TrainerMonitorBridge = None  # To be implemented in Phase 4.2
WebInterface = WebServer  # Now implemented as WebServer
ErrorHandler = None  # To be implemented in Phase 4.2
ErrorSeverity = None  # To be implemented in Phase 4.2
ErrorCategory = None  # To be implemented in Phase 4.2
RecoveryStrategy = None  # To be implemented in Phase 4.2
ErrorEvent = None  # To be implemented in Phase 4.2

# Compatibility warning
import warnings
warnings.warn(
    "Using compatibility layer for monitoring system. "
    "Please migrate to new monitoring.components.* imports.",
    DeprecationWarning,
    stacklevel=2
)
