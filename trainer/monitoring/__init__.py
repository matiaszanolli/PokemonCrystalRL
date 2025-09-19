"""
Monitoring and web interface components.

⚠️  DEPRECATED: This module is deprecated. Use the unified web dashboard instead:
    from web_dashboard import create_web_server

For migration instructions, see: web_dashboard/README.md and MIGRATION_PLAN.md
"""

import warnings

warnings.warn(
    "trainer.monitoring is deprecated. Use web_dashboard instead. "
    "See web_dashboard/README.md for migration instructions.",
    DeprecationWarning,
    stacklevel=2
)

# Provide stub for backward compatibility
class WebMonitor:
    """Deprecated WebMonitor stub. Use web_dashboard.create_web_server instead."""

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

__all__ = ['WebMonitor']
