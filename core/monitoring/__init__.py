"""
Core monitoring module that provides access to monitoring components.

This module acts as a bridge to the main monitoring package,
allowing imports from core.monitoring to work correctly.
"""

# Import the required modules from the root monitoring package
try:
    from monitoring.web_server import WebServer, TrainingWebServer, TrainingHandler
    from monitoring.data_bus import DataBus, DataType, get_data_bus
    from monitoring.bridge import TrainerMonitorBridge
    from monitoring.unified_monitor import UnifiedMonitor
    from core.web_monitor import WebMonitor
    from monitoring.error_handler import ErrorHandler
except ImportError as e:
    # Fallback for when monitoring modules are not available
    import warnings
    warnings.warn(f"Could not import monitoring modules: {e}", ImportWarning)
    
    # Create mock classes to prevent import errors
    class MockWebServer:
        def __init__(self, *args, **kwargs):
            pass
        def start(self):
            pass
        def shutdown(self):
            pass
    
    class MockDataBus:
        def __init__(self, *args, **kwargs):
            pass
        def register_component(self, *args, **kwargs):
            pass
        def publish(self, *args, **kwargs):
            pass
    
    def mock_get_data_bus():
        return MockDataBus()
    
    # Assign mock classes
    WebServer = MockWebServer
    TrainingWebServer = MockWebServer
    TrainingHandler = MockWebServer
    DataBus = MockDataBus
    DataType = type('DataType', (), {})
    get_data_bus = mock_get_data_bus
    TrainerMonitorBridge = MockWebServer
    UnifiedMonitor = MockWebServer
    WebMonitor = MockWebServer
    ErrorHandler = MockWebServer

# Make the modules available as attributes of this package
web_server = type('WebServerModule', (), {
    'WebServer': WebServer,
    'TrainingWebServer': TrainingWebServer,
    'TrainingHandler': TrainingHandler
})()

data_bus = type('DataBusModule', (), {
    'DataBus': DataBus,
    'DataType': DataType,
    'get_data_bus': get_data_bus
})()

__all__ = [
    'web_server',
    'data_bus',
    'WebServer',
    'TrainingWebServer', 
    'TrainingHandler',
    'DataBus',
    'DataType',
    'get_data_bus',
    'TrainerMonitorBridge',
    'UnifiedMonitor',
    'WebMonitor',
    'ErrorHandler'
]