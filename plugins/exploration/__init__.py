"""
Exploration Pattern Plugins Package

This package contains modular exploration pattern implementations for the Pokemon Crystal RL agent.
Each pattern is in its own file for better maintainability.

Available Patterns:
- SystematicSweepPattern: Thorough map coverage with horizontal sweeping
- SpiralSearchPattern: Expanding outward search from center point
- WallFollowingPattern: Boundary exploration following walls
- RandomWalkPattern: Biased random walk toward unexplored areas
"""

from .systematic_sweep import SystematicSweepPattern
from .spiral_search import SpiralSearchPattern
from .wall_following import WallFollowingPattern
from .random_walk import RandomWalkPattern

__all__ = [
    'SystematicSweepPattern',
    'SpiralSearchPattern',
    'WallFollowingPattern',
    'RandomWalkPattern',
]
