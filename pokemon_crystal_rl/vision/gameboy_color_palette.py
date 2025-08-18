"""Game Boy Color palette detection and handling."""

from enum import Enum
import numpy as np
from typing import Dict, List, Optional, Tuple


class GBCPalette(Enum):
    """Game Boy Color palette types."""
    DEFAULT = "default"
    TEXT_WHITE = "text_white"
    TEXT_BLACK = "text_black"
    DIALOGUE_NORMAL = "dialogue_normal"
    DIALOGUE_DARK = "dialogue_dark"
    MENU_LIGHT = "menu_light"
    MENU_DARK = "menu_dark"
    BATTLE_LIGHT = "battle_light"
    BATTLE_DARK = "battle_dark"


class GameBoyColorPalette:
    """Game Boy Color palette detection and analysis."""
    
    def __init__(self):
        # RGB values for each palette (main colors)
        self.palettes = {
            GBCPalette.DEFAULT: np.array([
                [255, 255, 255],  # White
                [192, 192, 192],  # Light gray
                [96, 96, 96],     # Dark gray
                [0, 0, 0]         # Black
            ], dtype=np.uint8),
            
            GBCPalette.TEXT_WHITE: np.array([
                [255, 255, 255],  # White text
                [224, 224, 224],  # Light gray text
                [32, 32, 32],     # Dark background
                [0, 0, 0]         # Black border
            ], dtype=np.uint8),
            
            GBCPalette.TEXT_BLACK: np.array([
                [32, 32, 32],     # Dark text
                [64, 64, 64],     # Gray text
                [224, 224, 224],  # Light background
                [255, 255, 255]   # White border
            ], dtype=np.uint8),
            
            GBCPalette.DIALOGUE_NORMAL: np.array([
                [255, 255, 255],  # White background
                [192, 192, 255],  # Light blue tint
                [32, 32, 64],     # Dark blue text
                [0, 0, 32]        # Navy border
            ], dtype=np.uint8),
            
            GBCPalette.DIALOGUE_DARK: np.array([
                [224, 224, 255],  # Light blue background
                [160, 160, 224],  # Medium blue
                [64, 64, 160],    # Dark blue text
                [32, 32, 128]     # Navy border
            ], dtype=np.uint8)
        }
        
        # Color distance thresholds
        self.distance_threshold = 48.0  # RGB Euclidean distance threshold
        
    def detect_palette_from_image(self, image: np.ndarray) -> GBCPalette:
        """Detect the most likely palette used in the image."""
        # Convert to RGB if needed
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[-1] == 4:
            image = image[..., :3]
        
        # Get unique colors and their frequencies
        unique_colors, counts = np.unique(
            image.reshape(-1, 3), axis=0, return_counts=True)
        
        # Sort by frequency (most common first)
        sort_idx = np.argsort(-counts)
        unique_colors = unique_colors[sort_idx]
        
        # Take top 4 most common colors
        main_colors = unique_colors[:4]
        
        # Compare against each palette
        best_match = GBCPalette.DEFAULT
        best_score = float('inf')
        
        for palette_type, palette_colors in self.palettes.items():
            score = self._compute_palette_score(main_colors, palette_colors)
            if score < best_score:
                best_score = score
                best_match = palette_type
        
        return best_match
    
    def _compute_palette_score(self, colors: np.ndarray, 
                             palette: np.ndarray) -> float:
        """Compute matching score between colors and palette."""
        # Compute all pairwise distances
        distances = np.zeros((len(colors), len(palette)))
        for i, color in enumerate(colors):
            for j, palette_color in enumerate(palette):
                distances[i, j] = np.sqrt(np.sum((color - palette_color) ** 2))
        
        # Find best matching using Hungarian algorithm
        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(distances)
        
        return distances[row_ind, col_ind].sum()
    
    def analyze_text_region_colors(self, region: np.ndarray) -> Dict:
        """Analyze color patterns in a potential text region."""
        # Convert to RGB if needed
        if len(region.shape) == 2:
            region = np.stack([region] * 3, axis=-1)
        elif region.shape[-1] == 4:
            region = region[..., :3]
        
        # Get unique colors and frequencies
        colors, counts = np.unique(
            region.reshape(-1, 3), axis=0, return_counts=True)
        
        # Calculate statistics
        total_pixels = region.shape[0] * region.shape[1]
        mean_color = np.mean(region, axis=(0, 1))
        mean_brightness = np.mean(mean_color)
        
        # Detect if high contrast (text likely present)
        color_range = np.max(region, axis=(0, 1)) - np.min(region, axis=(0, 1))
        is_high_contrast = np.max(color_range) > 128
        
        # Determine text style based on brightness patterns
        if mean_brightness > 192:
            text_style = "dark_on_light"
        elif mean_brightness < 64:
            text_style = "light_on_dark"
        else:
            text_style = "unknown"
        
        # Get most likely palette
        detected_palette = self.detect_palette_from_image(region)
        
        return {
            "mean_brightness": mean_brightness,
            "unique_colors": len(colors),
            "is_high_contrast": is_high_contrast,
            "text_style": text_style,
            "detected_palette": detected_palette.value
        }


def test_gameboy_color_palette():
    """Test GameBoyColorPalette functionality."""
    # Create test image
    palette = GameBoyColorPalette()
    test_image = np.zeros((32, 32, 3), dtype=np.uint8)
    
    # Add text-like patterns
    test_image[8:24, 8:24] = [255, 255, 255]  # White background
    test_image[12:20, 12:20] = [32, 32, 64]   # Dark text
    
    # Test palette detection
    detected = palette.detect_palette_from_image(test_image)
    assert detected in GBCPalette
    
    # Test region analysis
    analysis = palette.analyze_text_region_colors(test_image)
    assert "mean_brightness" in analysis
    assert "text_style" in analysis
    assert "is_high_contrast" in analysis
    
    print("✅ GameBoyColorPalette tests passed")
