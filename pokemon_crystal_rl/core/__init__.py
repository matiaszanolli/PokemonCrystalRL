"""
Core module for Pokemon Crystal RL

Contains core game environments, memory mapping, base classes, and video streaming.
"""

# Import memory_map first as it has no external dependencies
from .memory_map import MEMORY_ADDRESSES

# Import PyBoy environment and video streaming as optional dependencies
try:
    from .pyboy_env import PyBoyPokemonCrystalEnv
    PYBOY_AVAILABLE = True
except ImportError:
    PYBOY_AVAILABLE = False
    PyBoyPokemonCrystalEnv = None

# Import video streaming (optional, requires PIL)
try:
    from .video_streaming import PyBoyVideoStreamer, create_video_streamer, StreamQuality
    VIDEO_STREAMING_AVAILABLE = True
except ImportError:
    VIDEO_STREAMING_AVAILABLE = False
    PyBoyVideoStreamer = None
    create_video_streamer = None
    StreamQuality = None

__all__ = [
    'MEMORY_ADDRESSES', 'PyBoyPokemonCrystalEnv', 
    'PyBoyVideoStreamer', 'create_video_streamer', 'StreamQuality', 'VIDEO_STREAMING_AVAILABLE'
]

# Try to import optional modules
try:
    from .env import *
except ImportError:
    pass
