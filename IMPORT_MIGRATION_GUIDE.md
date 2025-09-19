# Import Migration Guide

## Old imports to replace:

```python
# REMOVE these imports:
from core.web_monitor import WebMonitor
from core.web_monitor.monitor import WebMonitor
from monitoring.web import HttpHandler
from monitoring.web.server import run_server

# REPLACE with:
from web_dashboard import create_web_server
```

## Usage migration:

```python
# Old usage:
web_monitor = WebMonitor(trainer)
web_monitor.start()

# New usage:
web_server = create_web_server(trainer)
web_server.start()
```

See web_dashboard/README.md for complete migration instructions.
