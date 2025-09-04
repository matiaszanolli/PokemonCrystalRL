"""
Test helper functions used across test files
"""
import socket


def get_available_port(start_port=8000, max_tries=100):
    """Find an available port starting from start_port

    Args:
        start_port (int): The port number to start searching from
        max_tries (int): Maximum number of ports to try before giving up

    Returns:
        int: An available port number

    Raises:
        RuntimeError: If no available ports were found in max_tries attempts
    """
    for port in range(start_port, start_port + max_tries):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind(('localhost', port))
            sock.listen(1)
            sock.close()
            return port
        except OSError:
            continue

    raise RuntimeError(f"Could not find available port after trying {max_tries} ports")
