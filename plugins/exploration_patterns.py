"""
Official Exploration Pattern Plugins

This module has been refactored into the plugins/exploration/ package for better
maintainability and organization. All pattern classes are re-exported here for
backward compatibility.

Original file: 745 lines → Now modular:
- plugins/exploration/systematic_sweep.py (206 lines)
- plugins/exploration/spiral_search.py (157 lines)
- plugins/exploration/wall_following.py (198 lines)
- plugins/exploration/random_walk.py (171 lines)

All existing imports will continue to work:
    from plugins.exploration_patterns import SystematicSweepPattern
    from plugins.exploration_patterns import SpiralSearchPattern
    from plugins.exploration_patterns import WallFollowingPattern
    from plugins.exploration_patterns import RandomWalkPattern

New preferred import style:
    from plugins.exploration import SystematicSweepPattern
    from plugins.exploration import SpiralSearchPattern
    from plugins.exploration import WallFollowingPattern
    from plugins.exploration import RandomWalkPattern
"""

from .exploration import (
    SystematicSweepPattern,
    SpiralSearchPattern,
    WallFollowingPattern,
    RandomWalkPattern,
)

__all__ = [
    'SystematicSweepPattern',
    'SpiralSearchPattern',
    'WallFollowingPattern',
    'RandomWalkPattern',
]
