"""Mock web server for tests."""

import dataclasses
import threading
import logging

from trainer.web_server import ServerConfig
from monitoring.data_bus import get_data_bus


class MockWebServer:
    """Mock web server for tests with proper port handling"""
    ServerConfig = ServerConfig
    
    _used_ports = set()  # Track ports in use across instances
    _port_lock = threading.Lock()  # Lock for thread-safe port management
    _port_cleanup_enabled = True  # Enable automatic port cleanup
    
    @classmethod
    def cleanup_ports(cls):
        """Clean up any remaining ports."""
        with cls._port_lock:
            cls._used_ports.clear()
    
    def __init__(self, config=None, trainer=None):
        self.config = config or ServerConfig()
        self.trainer = trainer
        self._trainer = trainer  # For backward compatibility
        self.port = self.config.port
        self._running = False
        self._ready = threading.Event()
        self._server_thread = None
        self.logger = logging.getLogger("mock_web_server")
        self.logger.setLevel(logging.DEBUG)  # Ensure debug is enabled
        self.logger.debug(f"MockWebServer initializing with port {self.port}")

        # Initialize DataBus connection
        self.data_bus = get_data_bus()
        
        # Initialize shutdown state
        self._shutdown_event = threading.Event()
        # This used to be in start() but we move it here for clarity
        # If port is in use, will try next available
        if self.port in self._used_ports:
            for new_port in range(self.port + 1, self.port + 10):
                if new_port not in self._used_ports:
                    self.port = new_port
                    break
            else:
                self.logger.error("No available ports")
                raise RuntimeError("No available ports")
        
        # Claim the port
        with self._port_lock:
            self._used_ports.add(self.port)
            self.logger.debug(f"Claimed port {self.port}")
        
    def start(self):
        """Start mock server"""
        try:
            self.logger.debug(f"Starting mock server on port {self.port}...")
            # Register with data bus
            if self.data_bus:
                self.data_bus.register_component(
                    "web_server",
                    {
                        "type": "monitoring",
                        "port": self.port,
                        "host": getattr(self.config, 'web_host', 'localhost')
                    }
                )
                self.logger.debug("Registered with data bus")
            
            # Start the server thread
            self._running = True
            self._server_thread = threading.Thread(target=self.run_in_thread)
            self._server_thread.daemon = True
            self._server_thread.start()
            
            # Wait for server to be ready
            if not self._ready.wait(timeout=1.0):
                self.logger.warning("Server did not become ready within timeout")
            
            self.logger.debug("Server started successfully")
            return self
            
        except Exception as e:
            self.logger.error(f"Error starting server: {e}")
            return None
        
    def stop(self):
        """Stop mock server"""
        self.logger.debug("Stopping mock server...")
        
        try:
            # First unregister from data bus to prevent new messages
            if self.data_bus:
                try:
                    self.logger.debug("Unregistering from data bus...")
                    self.data_bus.unregister_component("web_server")
                    self.logger.debug("Unregistered from data bus")
                except Exception as e:
                    self.logger.error(f"Error unregistering from data bus: {e}")

            # Signal shutdown to server thread
            self.logger.debug("Signaling server thread shutdown...")
            self._running = False
            self._shutdown_event.set()
            
            # Wait for server thread to stop with timeout
            if self._server_thread and self._server_thread.is_alive():
                self.logger.debug("Waiting for server thread to stop...")
                start = time.time()
                while self._server_thread.is_alive() and time.time() - start < 2.0:
                    time.sleep(0.1)
                if self._server_thread.is_alive():
                    self.logger.warning("Server thread did not stop within timeout")
                else:
                    self.logger.debug("Server thread stopped successfully")

            # Clean up port allocation
            with self._port_lock:
                try:
                    self.logger.debug(f"Releasing port {self.port}")
                    if self.port in self._used_ports:
                        self._used_ports.discard(self.port)
                        self.logger.debug("Port released successfully")
                except Exception as e:
                    self.logger.error(f"Error releasing port: {e}")

            self.logger.debug("Server cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during server stop: {e}")
            # Force cleanup in case of error
            self._running = False
            self._shutdown_event.set()
            self.cleanup_ports()
        
    def run_in_thread(self):
        """Run mock server thread"""
        self.logger.debug("Server thread starting...")
        
        # Notify that we're ready
        self._ready.set()
        self._running = True
        
        # Wait for shutdown signal with timeout
        self.logger.debug("Waiting for shutdown signal...")
        while not self._shutdown_event.is_set():
            # Poll shutdown event with timeout to stay responsive
            if self._shutdown_event.wait(timeout=0.1):  # 100ms polling
                break
            
        self.logger.debug("Server thread received shutdown signal")
        self._running = False
        self.logger.debug("Server thread exited")
        
    def _cleanup_threads(self):
        """Clean up any running threads."""
        # Make sure server thread is stopped
        if self._server_thread and self._server_thread.is_alive():
            self.logger.debug("Forcing server thread cleanup...")
            self._running = False
            self._shutdown_event.set()
            start = time.time()
            while self._server_thread.is_alive() and time.time() - start < 1.0:
                time.sleep(0.1)
            if self._server_thread.is_alive():
                self.logger.warning("Server thread could not be cleaned up")
        
    def shutdown(self):
        """Stop mock server and cleanup"""
        self.stop()
