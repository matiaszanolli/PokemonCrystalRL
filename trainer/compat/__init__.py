"""Compatibility layer for trainer components.

This package provides backward compatibility for old trainer APIs
while using the new modular architecture internally.
"""

from .web_monitor import WebMonitorCompat as WebMonitor
from monitoring.web.api import WebAPICompat as WebAPI

__all__ = ['WebMonitor', 'WebAPI']
