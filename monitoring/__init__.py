"""
Pokemon Crystal RL Monitoring System

This package provides a comprehensive monitoring system with both new and legacy APIs:

New Architecture (Recommended):
- Component-based monitoring system
- Clear interfaces and protocols
- Efficient resource usage
- Type safety and validation

Legacy Support (Deprecated):
- Original monitoring tools
- Web interface compatibility
- Training data management
- System resource monitoring
"""

# New architecture imports
from .base import (
    MonitorComponent,
    DataPublisher,
    DataSubscriber,
    MetricDefinition,
    ScreenCaptureConfig,
    MonitoringConfig,
    BaseMonitor,
    DataBusProtocol,
    MonitoringError,
    ComponentError,
    DataBusError
)

from .components.capture import ScreenCapture
from .components.metrics import MetricsCollector
from .components.web import WebServer

from .data.bus import DataBus, get_data_bus, DataValidator

# Legacy support (via compatibility layer)
from .compat import (
    DatabaseManager,
    UnifiedMonitor,
    GameStreamer,
    MonitoringClient,
    StatsCollector,
    TextLogger,
    TrainerMonitorBridge,
    WebInterface,
    ErrorHandler,
    ErrorSeverity,
    ErrorCategory,
    RecoveryStrategy,
    ErrorEvent
)

__all__ = [
    # New architecture
    'MonitorComponent',
    'DataPublisher',
    'DataSubscriber',
    'DataBusProtocol',
    'BaseMonitor',
    'MetricDefinition',
    'ScreenCaptureConfig',
    'MonitoringConfig',
    'ScreenCapture',
    'MetricsCollector',
    'WebServer',
    'DataBus',
    'get_data_bus',
    'DataValidator',
    'DataType',
    'MonitoringError',
    'ComponentError',
    'DataBusError',
    
    # Legacy support
    'DatabaseManager',
    'UnifiedMonitor',
    'GameStreamer',
    'MonitoringClient',
    'StatsCollector',
    'TextLogger',
    'TrainerMonitorBridge',
    'WebInterface',
    'ErrorHandler',
    'ErrorSeverity',
    'ErrorCategory',
    'RecoveryStrategy',
    'ErrorEvent'
]
