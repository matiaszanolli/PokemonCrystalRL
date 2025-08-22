#!/usr/bin/env python3
"""Core vision processing module for Pokemon Crystal RL.

This module provides the core vision processing functionality expected by
the integration tests, including DetectedText and VisualContext classes.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import time
import os
from dataclasses import dataclass
from collections import defaultdict, OrderedDict
import logging
import json
import hashlib

from shared_types import (
    DetectedText,
    GameUIElement,
    VisualContext,
    PyBoyGameState
)

# Import the main vision processor from the vision module
from vision.vision_processor import UnifiedVisionProcessor

# Re-export the main classes for backwards compatibility
__all__ = ['DetectedText', 'VisualContext', 'UnifiedVisionProcessor']

# Create a default processor instance for easy access
_default_processor = None

def get_vision_processor(template_path: str = None, rom_path: str = None, debug_mode: bool = False):
    """Get or create the default vision processor instance."""
    global _default_processor
    if _default_processor is None:
        _default_processor = UnifiedVisionProcessor(template_path, rom_path, debug_mode)
    return _default_processor

def process_screenshot(screen: np.ndarray) -> VisualContext:
    """Process a screenshot using the default vision processor."""
    processor = get_vision_processor()
    return processor.process_screenshot(screen)