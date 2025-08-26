"""
Data bus module for core.monitoring package.

This module provides access to data bus components from the main monitoring package.
"""

try:
    from monitoring.data_bus import DataBus, DataType, get_data_bus
except ImportError:
    # Fallback mock classes
    class MockDataBus:
        def __init__(self, *args, **kwargs):
            pass
        def register_component(self, *args, **kwargs):
            pass
        def publish(self, *args, **kwargs):
            pass
    
    def mock_get_data_bus():
        return MockDataBus()
    
    DataBus = MockDataBus
    DataType = type('DataType', (), {})
    get_data_bus = mock_get_data_bus

__all__ = ['DataBus', 'DataType', 'get_data_bus']