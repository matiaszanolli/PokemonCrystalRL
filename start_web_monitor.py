#!/usr/bin/env python3

import asyncio
from dataclasses import dataclass
from pathlib import Path
from monitoring.web_monitor import WebMonitor, MonitorConfig
from trainer.web_server import ServerConfig, WebServer

def main():
    print("Starting web monitor...")
    
    # Create config
    config = MonitorConfig(
        host="localhost",
        port=8080,
        static_dir="monitoring/static",
        data_dir="monitor_data"
    )
    
    # Create and start monitor
    monitor = WebMonitor(config=config)
    
    loop = asyncio.get_event_loop()
    
    try:
        loop.run_until_complete(monitor.start())
        print(f"Web monitor started at http://{config.host}:{config.port}")
        loop.run_forever()
    except KeyboardInterrupt:
        print("\nShutting down web monitor...")
        loop.run_until_complete(monitor.shutdown())
    finally:
        loop.close()

if __name__ == "__main__":
    main()
