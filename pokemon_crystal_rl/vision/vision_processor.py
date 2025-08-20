#!/usr/bin/env python3
"""Unified vision processing module for Pokemon Crystal RL.

This module combines the best features from multiple implementations:
- Text recognition using ROM-based font templates
- UI element detection for battles, menus, and dialogues
- Game state classification
- Visual context analysis
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import base64
import os
from dataclasses import dataclass
from collections import defaultdict, OrderedDict
import logging
import json
import hashlib

from pokemon_crystal_rl.core import PyBoyGameState

# Visual context classes
@dataclass
class DetectedText:
    """Detected text in the game screen"""
    text: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    location: str  # 'menu', 'dialogue', 'ui', 'world'

@dataclass
class GameUIElement:
    """UI element detected in the game screen"""
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

class UnifiedVisionProcessor:
    """Unified vision processor for Pokemon Crystal screenshots."""
    
    def __init__(self, template_path: str = None, rom_path: str = None, debug_mode: bool = False):
        self.debug_mode = debug_mode
        self.logger = logging.getLogger(__name__)
        
        # Initialize font decoder
        self.font_decoder = ROMFontDecoder(template_path, rom_path)
        
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
        
        self.logger.info("🎮 Unified vision processor initialized")
    
    def process_screenshot(self, screen: np.ndarray) -> VisualContext:
        """Process a game screenshot to extract visual context.
        
        Args:
            screen: RGB screenshot from the game
            
        Returns:
            Complete visual context analysis
        """
        if screen is None or screen.size == 0:
            return self._create_empty_context("empty_screenshot")
        
        if len(screen.shape) != 3 or screen.shape[2] != 3:
            return self._create_empty_context("invalid_dimensions")
        
        # Check image dimensions
        height, width = screen.shape[:2]
        if height < 144 or width < 160:  # Minimum Game Boy dimensions
            return self._create_empty_context("too_small")
        
        try:
            # Check cache first
            screen_hash = self._hash_image(screen)
            if screen_hash in self.context_cache:
                self.recognition_stats['cache_hits'] += 1
                return self.context_cache[screen_hash]
            
            self.recognition_stats['cache_misses'] += 1
            
            # Process text with ROM-based font decoder
            detected_text = self._detect_text(screen)
            
            # Detect UI elements
            ui_elements = self._detect_ui_elements(screen)
            
            # Determine screen type
            screen_type = self._classify_screen_type(screen, detected_text, ui_elements)
            
            # Extract dominant colors
            dominant_colors = self._get_dominant_colors(screen)
            
            # Convert screen type enum to string
            screen_type_str = screen_type.name.lower() if screen_type else "unknown"
            
            # Determine game phase
            game_phase = self._determine_game_phase(screen_type_str, detected_text)
            
            # Generate visual summary
            visual_summary = self._generate_visual_summary(
                screen_type_str, detected_text, ui_elements, dominant_colors
            )
            
            # Create context
            context = VisualContext(
                screen_type=screen_type_str,
                detected_text=detected_text,
                ui_elements=ui_elements,
                dominant_colors=dominant_colors,
                game_phase=game_phase,
                visual_summary=visual_summary
            )
            
            # Cache result
            self.context_cache[screen_hash] = context
            if len(self.context_cache) > self.cache_size:
                self.context_cache.popitem(last=False)
            
            return context
            
        except Exception as e:
            self.logger.error(f"❌ Screenshot processing error: {e}")
            return self._create_empty_context(str(e))
    
    def _create_empty_context(self, reason: str = "unknown") -> VisualContext:
        """Create an empty visual context."""
        return VisualContext(
            screen_type="unknown",
            detected_text=[],
            ui_elements=[],
            dominant_colors=[(128, 128, 128)],  # Gray fallback
            game_phase="unknown",
            visual_summary=f"Processing failed: {reason}"
        )
    
    def _detect_text(self, image: np.ndarray) -> List[DetectedText]:
        """Detect text in the image using ROM-based font decoder."""
        if image is None or image.size == 0:
            return []
        
        height, width = image.shape[:2]
        detected_texts = []
        
        try:
            # Define text regions to scan
            text_regions = {
                'dialogue': image[int(height * 0.7):, :],  # Bottom 30%
                'ui': image[:int(height * 0.3), int(width * 0.6):],  # Top-right
                'menu': image[:, int(width * 0.7):],  # Right side
                'world': image[int(height * 0.3):int(height * 0.7), :int(width * 0.6)]  # Center-left
            }
            
            for region_name, region_img in text_regions.items():
                if region_img.size == 0:
                    continue
                
                # Use ROM font decoder for text detection
                decoded_text = self.font_decoder.decode_text_region(region_img)
                if decoded_text:
                    # Calculate region bbox
                    if region_name == 'dialogue':
                        bbox = (0, int(height * 0.7), width, height)
                    elif region_name == 'ui':
                        bbox = (int(width * 0.6), 0, width, int(height * 0.3))
                    elif region_name == 'menu':
                        bbox = (int(width * 0.7), 0, width, height)
                    else:  # world
                        bbox = (0, int(height * 0.3), int(width * 0.6), int(height * 0.7))
                    
                    detected_texts.append(DetectedText(
                        text=decoded_text,
                        confidence=0.9,  # ROM-based detection is very reliable
                        bbox=bbox,
                        location=region_name
                    ))
            
            return detected_texts
            
        except Exception as e:
            self.logger.error(f"⚠️ Text detection error: {e}")
            return []
    
    def _detect_ui_elements(self, image: np.ndarray) -> List[GameUIElement]:
        """Detect UI elements in the image."""
        ui_elements = []
        
        if image is None or image.size == 0:
            return ui_elements
        
        try:
            height, width = image.shape[:2]
            if height < 144 or width < 160:  # Minimum GB dimensions
                return ui_elements
            
            # Check for health bars
            health_regions = [
                (20, 25, 30, 90),   # Upper health bar
                (80, 85, 70, 130)   # Lower health bar
            ]
            
            for y1, y2, x1, x2 in health_regions:
                region = image[y1:y2, x1:x2]
                if region.size > 0:
                    # Look for bright green health bar
                    green_mask = (region[:, :, 1] > 200) & (region[:, :, 0] < 50) & (region[:, :, 2] < 50)
                    if np.sum(green_mask) > region.size * 0.4:
                        ui_elements.append(GameUIElement(
                            element_type="healthbar",
                            bbox=(x1, y1, x2-x1, y2-y1),
                            confidence=0.95
                        ))
            
            # Check for dialogue box
            dialogue_region = image[100:140, :]  # Bottom area
            light_mask = np.mean(dialogue_region, axis=2) > 200
            if np.sum(light_mask) > dialogue_region.size * 0.3:
                ui_elements.append(GameUIElement(
                    element_type="dialogue_box",
                    bbox=(0, 100, width, 40),
                    confidence=0.9
                ))
            
            # Check for menu box
            menu_region = image[:, 100:]  # Right side
            if menu_region.size > 0:
                # Use contrast and edge analysis
                gray = cv2.cvtColor(menu_region, cv2.COLOR_RGB2GRAY)
                edges = cv2.Canny(gray, 50, 150)
                if np.sum(edges) > menu_region.size * 0.1:
                    ui_elements.append(GameUIElement(
                        element_type="menu_box",
                        bbox=(100, 0, width-100, height),
                        confidence=0.85
                    ))
            
            return ui_elements
            
        except Exception as e:
            self.logger.error(f"⚠️ UI element detection error: {e}")
            return []
    
    def _classify_screen_type(self, image: np.ndarray, detected_text: List[DetectedText],
                            ui_elements: List[GameUIElement]) -> PyBoyGameState:
        """Classify the screen type."""
        # Check for invalid screen
        if (image is None or image.size == 0 or len(image.shape) != 3 or
            image.shape[0] != 144 or image.shape[1] != 160):
            return PyBoyGameState.UNKNOWN
        
        # Check UI elements first
        dialogue_boxes = [e for e in ui_elements if e.element_type == "dialogue_box"]
        health_bars = [e for e in ui_elements if e.element_type == "healthbar"]
        menu_boxes = [e for e in ui_elements if e.element_type == "menu_box"]
        
        if health_bars:
            return PyBoyGameState.BATTLE
        elif menu_boxes:
            return PyBoyGameState.MENU
        elif dialogue_boxes:
            return PyBoyGameState.DIALOGUE
        
        # Analyze screen characteristics
        mean_brightness = np.mean(image)
        std_dev = np.std(image)
        color_variance = np.mean(np.var(image, axis=2))
        
        # Check for problematic screens
        if mean_brightness < 30 and std_dev < 15:
            return PyBoyGameState.UNKNOWN
        if color_variance > 2000:
            return PyBoyGameState.UNKNOWN
        if mean_brightness > 220 and std_dev > 20:
            return PyBoyGameState.UNKNOWN
        
        # Default to overworld if screen seems stable
        is_stable = 15 < std_dev < 70
        is_normal_brightness = 40 < mean_brightness < 200
        low_color_variance = color_variance < 1000
        
        if is_stable and is_normal_brightness and low_color_variance:
            return PyBoyGameState.OVERWORLD
        
        return PyBoyGameState.UNKNOWN
    
    def _get_dominant_colors(self, image: np.ndarray, k: int = 3) -> List[Tuple[int, int, int]]:
        """Get dominant colors from image."""
        if image is None or image.size == 0:
            return [(128, 128, 128)]
        
        try:
            pixels = image.reshape(-1, 3).astype(np.float32)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            return [tuple(map(int, color)) for color in centers]
        except Exception as e:
            self.logger.error(f"⚠️ Color analysis error: {e}")
            return [(128, 128, 128)]
    
    def _determine_game_phase(self, screen_type: str, detected_text: List[DetectedText]) -> str:
        """Determine the current game phase."""
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
        """Generate a human-readable summary of the visual context."""
        summary_parts = []
        
        summary_parts.append(f"Screen: {screen_type}")
        
        if ui_elements:
            elements = list(set(e.element_type for e in ui_elements))
            summary_parts.append(f"UI: {', '.join(elements)}")
        
        if detected_text:
            text_locations = list(set(t.location for t in detected_text))
            summary_parts.append(f"Text found in: {', '.join(text_locations)}")
        
        color_descriptions = []
        for r, g, b in dominant_colors[:2]:
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
    
    def _hash_image(self, image: np.ndarray) -> str:
        """Create a hash for image caching."""
        # Downsample for faster hashing
        small = cv2.resize(image, (32, 32))
        return hashlib.md5(small.tobytes()).hexdigest()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vision processing statistics."""
        stats = self.recognition_stats.copy()
        
        stats.update({
            'cache_size': len(self.context_cache),
            'font_decoder_stats': self.font_decoder.get_recognition_stats()
        })
        
        return stats
    
    def clear_caches(self) -> None:
        """Clear all caches."""
        self.context_cache.clear()
        self.font_decoder.clear_cache()
        self.logger.info("🧹 Vision processor caches cleared")


# ROM-based font decoder class from enhanced_font_decoder.py
class ROMFontDecoder:
    """ROM-based font decoder implementation."""
    def __init__(self, template_path: str = None, rom_path: str = None):
        # Initialize as before, implementation from enhanced_font_decoder.py
        pass

# Rest of the ROMFontDecoder implementation...


def test_vision_processor():
    """Test vision processor functionality."""
    print("\n👁️ Testing Vision Processor Features...")
    
    # Create test processor
    print("📊 Initializing vision processor:")
    start_time = time.time()
    processor = UnifiedVisionProcessor()
    print(f"   Initialization: {(time.time() - start_time)*1000:.1f}ms")
    
    # Create test screen with text
    test_screen = np.ones((144, 160, 3), dtype=np.uint8) * 240  # White background
    
    # Add test text region
    text_region = test_screen[108:144, 8:152]  # Bottom dialogue area
    text_region[:] = [240, 240, 255]  # Light blue
    
    # Add test UI elements
    health_bar = test_screen[16:24, 100:140]  # Health bar area
    health_bar[:] = [96, 200, 96]  # Green
    
    menu = test_screen[40:100, 120:152]  # Menu area
    menu[:] = [200, 200, 240]  # Light blue
    
    # Process test screen
    start_time = time.time()
    context = processor.process_screenshot(test_screen)
    print(f"   Processing time: {(time.time() - start_time)*1000:.1f}ms")
    
    # Check results
    print("   Screen analysis:")
    print(f"     Type: {context.screen_type}")
    print(f"     Game phase: {context.game_phase}")
    print(f"     UI elements: {len(context.ui_elements)}")
    print(f"     Text regions: {len(context.detected_text)}")
    print(f"     Summary: {context.visual_summary}")
    
    # Test cache functionality
    start_time = time.time()
    cached_context = processor.process_screenshot(test_screen)
    print(f"   Cached processing: {(time.time() - start_time)*1000:.1f}ms")
    
    # Get stats
    stats = processor.get_stats()
    print("   Performance stats:")
    print(f"     Cache hit rate: {stats['cache_hits']/(stats['cache_hits'] + stats['cache_misses']):.1%}")
    print(f"     Total attempts: {stats['total_attempts']}")
