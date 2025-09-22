"""Compatibility layer for trainer components.

This package provides backward compatibility for old trainer APIs
while using the new modular architecture internally.

⚠️  DEPRECATED: Use the unified web dashboard instead:
    from web_dashboard import create_web_server
"""

import warnings

warnings.warn(
    "trainer.compat is deprecated. Use web_dashboard instead. "
    "See web_dashboard/README.md for migration instructions.",
    DeprecationWarning,
    stacklevel=2
)

# Compatibility stubs
class WebMonitor:
    """Deprecated WebMonitor compatibility stub."""

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "WebMonitor is deprecated. Use web_dashboard.create_web_server instead.",
            DeprecationWarning,
            stacklevel=2
        )
        self.started = False

    def start(self):
        self.started = True
        return True

    def stop(self):
        self.started = False

    def update_pyboy(self, pyboy):
        pass

class WebAPI:
    """Deprecated WebAPI compatibility stub."""

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "WebAPI is deprecated. Use web_dashboard API endpoints instead.",
            DeprecationWarning,
            stacklevel=2
        )

__all__ = ['WebMonitor', 'WebAPI']
