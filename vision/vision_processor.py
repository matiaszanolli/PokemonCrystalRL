"""
vision_processor.py - Computer Vision for Pokemon Crystal Agent

This module processes PyBoy screenshots to extract visual context
for the LLM agent, including text recognition, UI elements detection,
and game state analysis.
"""

import cv2
import numpy as np
from PIL import Image
from typing import Dict, List, Tuple, Optional, Any
import base64
import io
import logging
import os
import hashlib
import time
from dataclasses import dataclass
from collections import OrderedDict
from core.game_states import PyBoyGameState
from .enhanced_font_decoder import EnhancedFontDecoder
from .rom_font_extractor import ROMFontExtractor


# Visual context classes
@dataclass
class DetectedText:
    """Represents detected text in the game"""
    text: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    location: str  # 'menu', 'dialogue', 'ui', 'world'


@dataclass
class GameUIElement:
    """Represents a UI element detected in the game"""
    element_type: str  # 'healthbar', 'menu', 'dialogue_box', 'battle_ui'
    bbox: Tuple[int, int, int, int]
    confidence: float


@dataclass
class VisualContext:
    """Complete visual analysis of a game screenshot"""
    screen_type: str  # 'overworld', 'battle', 'menu', 'dialogue', 'intro'
    detected_text: List[DetectedText]
    ui_elements: List[GameUIElement]
    dominant_colors: List[Tuple[int, int, int]]
    game_phase: str  # 'intro', 'gameplay', 'battle', 'menu_navigation'
    visual_summary: str


class PokemonVisionProcessor:
    """Processes Pokemon Crystal screenshots for state detection"""
    def __init__(self, template_path: str = None, rom_path: str = None, debug_mode: bool = False):
        self.debug_mode = debug_mode
        self.logger = logging.getLogger(__name__)
        
        # Initialize enhanced font detection
        self.font_decoder = EnhancedFontDecoder(template_path)
        
        # Recognition statistics
        self.recognition_stats = {
            'total_attempts': 0,
            'successful_matches': 0,
            'failed_matches': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        # Cache for optimization
        self.cache_size = 1000
        self.context_cache = OrderedDict()  # LRU cache for visual contexts
        
        self.logger.info("🎮 Vision processor initialized")

    def process_screenshot(self, screen: np.ndarray) -> VisualContext:
        """Process a game screenshot"""
        if screen is None or screen.size == 0:
            return self._create_empty_context("empty_screenshot")

        if len(screen.shape) != 3 or screen.shape[2] != 3:
            return self._create_empty_context("invalid_dimensions")

        # Check if image has reasonable dimensions
        height, width = screen.shape[:2]
        if height < 10 or width < 10:
            return self._create_empty_context("too_small")

        # Detect text
        detected_text = self._detect_text(screen)

        # Analyze UI elements
        ui_elements = self._detect_ui_elements(screen)

        # Determine screen type
        screen_type = self._classify_screen_type(screen, detected_text, ui_elements)

        # Extract dominant colors
        dominant_colors = self._get_dominant_colors(screen)

        # Convert screen type enum to string for visual context
        screen_type_str = screen_type.name.lower() if screen_type else "unknown"

        # Determine game phase
        game_phase = self._determine_game_phase(screen_type_str, detected_text)

        # Generate visual summary
        visual_summary = self._generate_visual_summary(
            screen_type_str, detected_text, ui_elements, dominant_colors
        )

        return VisualContext(
            screen_type=screen_type_str,
            detected_text=detected_text,
            ui_elements=ui_elements,
            dominant_colors=dominant_colors,
            game_phase=game_phase,
            visual_summary=visual_summary
        )

    def _create_empty_context(self, reason: str = "unknown") -> VisualContext:
        """Create an empty visual context"""
        return VisualContext(
            screen_type="unknown",
            detected_text=[],
            ui_elements=[],
            dominant_colors=[(128, 128, 128)],  # Gray fallback
            game_phase="unknown",
            visual_summary=f"Processing failed: {reason}"
        )

    def _detect_text(self, image: np.ndarray) -> List[DetectedText]:
        """Detect text in the image"""
        if image is None or image.size == 0:
            return []

        height, width = image.shape[:2]
        detected_texts = []

        try:
            # Convert to grayscale for text detection
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

            # Define text regions to scan
            text_regions = {
                'dialogue': gray[int(height * 0.7):, :],  # Bottom 30%
                'ui': gray[:int(height * 0.3), int(width * 0.6):],  # Top-right
                'menu': gray[:, int(width * 0.7):],  # Right side
                'world': gray[int(height * 0.3):int(height * 0.7), :int(width * 0.6)]  # Center-left
            }

            for region_name, region_img in text_regions.items():
                if region_img.size == 0:
                    continue

                # Simple text detection for region
                contours = self._get_text_contours(region_img)
                if len(contours) > 0:
                    # Calculate bbox for this region in full image coordinates
                    if region_name == 'dialogue':
                        bbox = (0, int(height * 0.7), width, int(height * 0.3))
                    elif region_name == 'ui':
                        bbox = (int(width * 0.6), 0, int(width * 0.4), int(height * 0.3))
                    elif region_name == 'menu':
                        bbox = (int(width * 0.7), 0, int(width * 0.3), height)
                    else:  # world
                        bbox = (0, int(height * 0.3), int(width * 0.6), int(height * 0.4))

                    detected_texts.append(DetectedText(
                        text="TEXT",  # Simple placeholder
                        confidence=0.6,
                        bbox=bbox,
                        location=region_name
                    ))

            return detected_texts

        except Exception as e:
            print(f"⚠️ Text detection error: {e}")
            return []

    def _get_text_contours(self, image: np.ndarray) -> List:
        """Get text-like contours from image"""
        if image is None or image.size == 0:
            return []

        # Multiple threshold attempts to catch text
        thresholds = [60, 120, 180]
        all_contours = []

        for threshold in thresholds:
            # Try both regular and inverted thresholds
            _, binary = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            all_contours.extend(contours)

            _, binary_inv = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY_INV)
            contours_inv, _ = cv2.findContours(binary_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            all_contours.extend(contours_inv)

        return all_contours

    def _detect_ui_elements(self, image: np.ndarray) -> List[GameUIElement]:
        """Detect UI elements in the image"""
        ui_elements = []

        if image is None or image.size == 0:
            return ui_elements

        try:
            # First check if the image is valid and has reasonable dimensions
            height, width = image.shape[:2]
            if height < 144 or width < 160:  # Minimum expected GB screen dimensions
                return ui_elements

            # Check for health bars in typical locations
            try:
                # Check upper health bar region
                health_region1 = image[20:25, 30:90]  # Upper health bar area
                health_region2 = image[80:85, 70:130]  # Lower health bar area
                
                # Print regions for debugging
                if self.debug_mode:
                    print("Health Region 1 shape:", health_region1.shape)
                    print("Health Region 1 content:", health_region1)
                    print("Health Region 2 shape:", health_region2.shape)
                    print("Health Region 2 content:", health_region2)
                
                # Check for bright green with low red/blue
                # Typical health bar is (0, 255, 0) in test data
                green_mask1 = (health_region1[:, :, 1] > 200) & (health_region1[:, :, 0] < 50) & (health_region1[:, :, 2] < 50)
                green_mask2 = (health_region2[:, :, 1] > 200) & (health_region2[:, :, 0] < 50) & (health_region2[:, :, 2] < 50)
                
                if self.debug_mode:
                    print("Green Mask 1:", green_mask1)
                    print("Green Mask 2:", green_mask2)
                
                # Count green pixels in each region
                pixel_count1 = np.sum(green_mask1)
                pixel_count2 = np.sum(green_mask2)
                region_size = health_region1.shape[0] * health_region1.shape[1]
                
                if self.debug_mode:
                    print("Pixel count 1:", pixel_count1)
                    print("Pixel count 2:", pixel_count2)
                    print("Region size:", region_size)
                    print("Threshold:", region_size * 0.4)
                
                # Add health bar if either region has sufficient green pixels
                if pixel_count1 > region_size * 0.4 or pixel_count2 > region_size * 0.4:
                    ui_elements.append(GameUIElement(
                        element_type="healthbar",
                        bbox=(30, 20, 130, 85),  # Encompass both potential health bar areas
                        confidence=0.95
                    ))
            except (IndexError, ValueError):
                pass

            # Check for dialogue box (white box at bottom)
            try:
                dialogue_region = image[100:140, :]  # Bottom area
                light_mask = np.mean(dialogue_region, axis=2) > 200
                dark_mask = np.mean(dialogue_region, axis=2) < 50
                if np.sum(light_mask) > dialogue_region.shape[0] * dialogue_region.shape[1] * 0.3:
                    ui_elements.append(GameUIElement(
                        element_type="dialogue_box",
                        bbox=(0, 100, width, 140),
                        confidence=0.9
                    ))
            except (IndexError, ValueError):
                pass

            # Check for menu box (contrast-based detection)
            try:
                menu_region = image[:, 100:]  # Assume menus are on the right
                avg_brightness = np.mean(menu_region, axis=2)
                
                # Check both contrast and edge strength
                contrast = np.std(avg_brightness)
                edges = np.abs(np.diff(avg_brightness, axis=1))
                edge_strength = np.sum(edges > 50)
                edge_uniformity = np.std(edges)
                
                # Menu needs clear edges, moderate contrast, and consistent edge strength
                has_good_contrast = 20 < contrast < 80
                has_clear_edges = edge_strength > 30
                has_uniform_edges = edge_uniformity < 20
                
                if has_good_contrast and has_clear_edges and has_uniform_edges:
                    ui_elements.append(GameUIElement(
                        element_type="menu_box",
                        bbox=(100, 0, width, height),
                        confidence=0.85
                    ))
            except (IndexError, ValueError):
                pass

        except Exception as e:
            print(f"⚠️ UI element detection error: {e}")

        return ui_elements

    def _classify_screen_type(self, image: np.ndarray, detected_text: List[DetectedText],
                         ui_elements: List[GameUIElement]) -> PyBoyGameState:
        """Classify screen type"""
        # First check for invalid screen
        if (image is None or image.size == 0 or len(image.shape) != 3 or
            image.shape[0] != 144 or image.shape[1] != 160):  # Check exact dimensions
            return PyBoyGameState.UNKNOWN

        # Simple criteria based on UI elements - check these first
        dialogue_boxes = [e for e in ui_elements if e.element_type == "dialogue_box"]
        health_bars = [e for e in ui_elements if e.element_type == "healthbar"]
        menu_boxes = [e for e in ui_elements if e.element_type == "menu_box"]

        # First check presence of key UI elements
        if health_bars:  # Battle UI is most distinctive
            return PyBoyGameState.BATTLE
        elif menu_boxes:  # Then check for menu
            return PyBoyGameState.MENU
        elif dialogue_boxes:  # Then dialogue
            return PyBoyGameState.DIALOGUE

        # If no UI elements, check screen characteristics
        mean_brightness = np.mean(image)
        std_dev = np.std(image)

        # Check for empty screens
        if mean_brightness < 30 and std_dev < 15:  # Uniform and dark
            return PyBoyGameState.UNKNOWN

        # Check for random noise or artifacts
        color_variance = np.mean(np.var(image, axis=2))
        if color_variance > 2000:  # Very high color variance indicates noise
            return PyBoyGameState.UNKNOWN

        # Check for overly bright screens
        if mean_brightness > 220 and std_dev > 20:  # Bright and varied
            return PyBoyGameState.UNKNOWN

        # Default to overworld only if we have a clear, stable image
        # and brightness/contrast are in expected ranges for a potential game screen
        is_stable = 15 < std_dev < 70  # Some variation but not too noisy
        is_normal_brightness = 40 < mean_brightness < 200
        low_color_variance = color_variance < 1000
        
        # Only return OVERWORLD if all aspects are reasonably normal
        if is_stable and is_normal_brightness and low_color_variance:
            return PyBoyGameState.OVERWORLD
        
        return PyBoyGameState.UNKNOWN

    def _get_dominant_colors(self, image: np.ndarray, k: int = 3) -> List[Tuple[int, int, int]]:
        """Get dominant colors from image"""
        if image is None or image.size == 0:
            return [(128, 128, 128)]  # Gray fallback

        try:
            # Reshape and convert to float32 for k-means
            pixels = image.reshape(-1, 3).astype(np.float32)

            # k-means clustering
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

            # Convert back to uint8
            centers = np.uint8(centers)
            return [tuple(color) for color in centers]

        except Exception as e:
            print(f"⚠️ Color analysis error: {e}")
            return [(128, 128, 128)]  # Gray fallback

    def _determine_game_phase(self, screen_type: str, detected_text: List[DetectedText]) -> str:
        """Determine game phase"""
        if screen_type == "battle":
            return "battle"
        elif screen_type == "menu":
            return "menu_navigation"
        elif screen_type == "dialogue":
            return "dialogue_interaction"
        else:
            return "overworld_exploration"

    def _generate_visual_summary(self, screen_type: str, detected_text: List[DetectedText],
                            ui_elements: List[GameUIElement],
                            dominant_colors: List[Tuple[int, int, int]]) -> str:
        """Generate visual summary"""
        summary_parts = []

        # Screen type
        summary_parts.append(f"Screen: {screen_type}")

        # UI elements
        if ui_elements:
            elements = list(set(e.element_type for e in ui_elements))
            summary_parts.append(f"UI: {', '.join(elements)}")

        # Text
        if detected_text:
            text_locations = list(set(t.location for t in detected_text))
            summary_parts.append(f"Text found in: {', '.join(text_locations)}")

        # Colors
        color_descriptions = []
        for r, g, b in dominant_colors[:2]:  # Top 2 colors
            if r > 200 and g > 200 and b > 200:
                color_descriptions.append("bright")
            elif r < 50 and g < 50 and b < 50:
                color_descriptions.append("dark")
            elif g > r and g > b:
                color_descriptions.append("green")
            elif r > g and r > b:
                color_descriptions.append("red")
            elif b > r and b > g:
                color_descriptions.append("blue")

        if color_descriptions:
            summary_parts.append(f"Colors: {', '.join(color_descriptions)}")

        return " | ".join(summary_parts)
