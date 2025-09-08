#!/usr/bin/env python3
"""
Image processing utilities for Pokemon Crystal vision system.

This module provides pure, stateless image processing functions that can be
used by the vision processor and other components. All functions are designed
to be side-effect free and easily testable.
"""

import cv2
import numpy as np
import hashlib
import base64
import io
import logging
from typing import List, Tuple
from PIL import Image


def upscale_screenshot(screenshot: np.ndarray, scale_factor: int = 2) -> np.ndarray:
    """Upscale screenshot for better processing.
    
    Args:
        screenshot: Input screenshot to upscale
        scale_factor: Factor by which to upscale the image
        
    Returns:
        Upscaled screenshot
        
    Raises:
        ValueError: If screenshot is None or invalid
    """
    if screenshot is None:
        raise ValueError("Screenshot cannot be None")
    
    if not isinstance(screenshot, np.ndarray):
        raise ValueError("Screenshot must be a numpy array")
    
    if screenshot.size == 0:
        raise ValueError("Screenshot cannot be empty")
    
    if len(screenshot.shape) != 3 or screenshot.shape[2] != 3:
        raise ValueError("Screenshot must be a 3-channel RGB image")
    
    height, width = screenshot.shape[:2]
    if height == 0 or width == 0:
        raise ValueError("Screenshot dimensions cannot be zero")
    
    try:
        # Use nearest neighbor interpolation to maintain pixel art style
        upscaled = cv2.resize(
            screenshot, 
            (width * scale_factor, height * scale_factor), 
            interpolation=cv2.INTER_NEAREST
        )
        return upscaled
    except cv2.error as e:
        raise ValueError(f"Failed to upscale screenshot: {e}")


def hash_image(image: np.ndarray) -> str:
    """Create a hash for image caching.
    
    Args:
        image: Input image to hash
        
    Returns:
        MD5 hash string of the image
        
    Raises:
        ValueError: If image is invalid
    """
    if image is None:
        raise ValueError("Image cannot be None")
    
    if not isinstance(image, np.ndarray):
        raise ValueError("Image must be a numpy array")
    
    if image.size == 0:
        raise ValueError("Image cannot be empty")
    
    try:
        # Downsample for faster hashing
        small = cv2.resize(image, (32, 32))
        return hashlib.md5(small.tobytes()).hexdigest()
    except cv2.error as e:
        raise ValueError(f"Failed to hash image: {e}")


def get_dominant_colors(image: np.ndarray, k: int = 3) -> List[Tuple[int, int, int]]:
    """Get dominant colors from image using k-means clustering.
    
    Args:
        image: Input image to analyze
        k: Number of dominant colors to extract
        
    Returns:
        List of dominant colors as RGB tuples
    """
    # Default fallback color
    default_color = [(128, 128, 128)]
    
    if image is None or image.size == 0:
        return default_color
    
    # Check if image has valid dimensions for color analysis
    if len(image.shape) != 3 or image.shape[2] != 3:
        return default_color
    
    try:
        pixels = image.reshape(-1, 3).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        return [tuple(map(int, color)) for color in centers]
    except Exception as e:
        # Use logging without requiring a logger instance
        logging.getLogger(__name__).error(f"⚠️ Color analysis error: {e}")
        return default_color


def encode_screenshot_for_llm(screenshot: np.ndarray) -> str:
    """Encode screenshot for LLM processing as base64 PNG.
    
    Args:
        screenshot: Screenshot to encode
        
    Returns:
        Base64 encoded screenshot string
    """
    if screenshot is None or screenshot.size == 0:
        return ""
    
    try:
        # Convert numpy array to PIL Image
        if len(screenshot.shape) == 3:
            image = Image.fromarray(screenshot)
        else:
            image = Image.fromarray(screenshot)
        
        # Save to bytes buffer
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        
        # Encode to base64
        encoded = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return encoded
    except Exception as e:
        # Use logging without requiring a logger instance
        logging.getLogger(__name__).error(f"Failed to encode screenshot: {e}")
        return ""


def validate_image(image: np.ndarray, require_color: bool = True) -> bool:
    """Validate that an image array is suitable for processing.
    
    Args:
        image: Image array to validate
        require_color: Whether to require 3-channel color image
        
    Returns:
        True if image is valid, False otherwise
    """
    if image is None:
        return False
    
    if not isinstance(image, np.ndarray):
        return False
    
    if image.size == 0:
        return False
    
    if len(image.shape) < 2:
        return False
    
    if require_color and (len(image.shape) != 3 or image.shape[2] != 3):
        return False
    
    # Check for reasonable dimensions
    height, width = image.shape[:2]
    if height == 0 or width == 0:
        return False
    
    return True


def resize_image(image: np.ndarray, target_size: Tuple[int, int], 
                interpolation: int = cv2.INTER_NEAREST) -> np.ndarray:
    """Resize image to target dimensions.
    
    Args:
        image: Input image to resize
        target_size: Target (width, height) dimensions
        interpolation: OpenCV interpolation method
        
    Returns:
        Resized image
        
    Raises:
        ValueError: If inputs are invalid
    """
    if not validate_image(image, require_color=False):
        raise ValueError("Invalid input image")
    
    if not isinstance(target_size, (tuple, list)) or len(target_size) != 2:
        raise ValueError("Target size must be a (width, height) tuple")
    
    width, height = target_size
    if width <= 0 or height <= 0:
        raise ValueError("Target dimensions must be positive")
    
    try:
        return cv2.resize(image, (width, height), interpolation=interpolation)
    except cv2.error as e:
        raise ValueError(f"Failed to resize image: {e}")