#!/usr/bin/env python3
"""
Simple entry point to run the monitoring web server.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from monitoring.web.server import MonitoringServer, WebServerConfig

def main():
    """Run the monitoring web server."""
    config = WebServerConfig(
        host="localhost",
        port=8080,
        debug=True,
        enable_api=True,
        enable_websocket=True,
        enable_metrics=True
    )
    
    server = MonitoringServer(config)
    server.start()

if __name__ == "__main__":
    main()
