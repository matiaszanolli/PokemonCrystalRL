"""Port management utilities for tests."""
import socket
from contextlib import contextmanager
import time

class TestPortManager:
    """Manages port allocation for tests to prevent conflicts."""
    
    _used_ports = set()
    _base_port = 8000
    
    @classmethod
    def get_free_port(cls):
        """Get next available port."""
        while cls._base_port < 65535:
            if cls._base_port not in cls._used_ports:
                try:
                    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                        s.bind(('', cls._base_port))
                        cls._used_ports.add(cls._base_port)
                        return cls._base_port
                except OSError:
                    pass
            cls._base_port += 1
        raise RuntimeError("No available ports")

    @classmethod
    def release_port(cls, port):
        """Release a port back to the pool."""
        if port in cls._used_ports:
            cls._used_ports.remove(port)

    @classmethod
    @contextmanager
    def get_port(cls):
        """Context manager for port allocation."""
        port = cls.get_free_port()
        try:
            yield port
        finally:
            time.sleep(0.1)  # Give socket time to close
            cls.release_port(port)
