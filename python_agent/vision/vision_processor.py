"""
vision_processor.py - Computer Vision for Pokemon Crystal Agent

This module processes PyBoy screenshots to extract visual context
for the LLM agent, including text recognition, UI elements detection,
and game state analysis.
"""

import cv2
import numpy as np
# import easyocr  # Replaced with custom Pokemon font decoder
from PIL import Image, ImageEnhance, ImageFilter
from typing import Dict, List, Tuple, Optional, Any
import base64
import io
from dataclasses import dataclass
import re

# Import our custom Pokemon font decoder
try:
    from enhanced_font_decoder import ROMFontDecoder
    print("📚 Using ROM-based font decoder")
except ImportError:
    try:
        from pokemon_font_decoder import PokemonFontDecoder as ROMFontDecoder
        print("📚 Using fallback font decoder")
    except ImportError:
        print("⚠️ No font decoder available, text recognition disabled")
        ROMFontDecoder = None


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
    """
    Computer vision processor for Pokemon Crystal screenshots
    """
    
    def __init__(self):
        """Initialize the vision processor"""
        # Initialize custom Pokemon Crystal font decoder
        print("🔍 Initializing vision processor...")
        
        if ROMFontDecoder is not None:
            self.font_decoder = ROMFontDecoder()
            print("📚 Using ROM-based font decoder for improved accuracy")
        else:
            # Fallback to basic text recognition
            self.font_decoder = None
            print("⚠️ No font decoder available, using basic text detection")
        
        # Game-specific color definitions (Game Boy Color palette)
        self.pokemon_colors = {
            'health_green': (0, 255, 0),
            'health_yellow': (255, 255, 0), 
            'health_red': (255, 0, 0),
            'menu_blue': (0, 0, 255),
            'dialogue_white': (255, 255, 255),
            'text_black': (0, 0, 0)
        }
        
        # UI element templates (simplified detection)
        self.ui_templates = self._load_ui_templates()
        
        print("✅ Vision processor initialized")
    
    def _create_empty_context(self, reason: str = "unknown") -> VisualContext:
        """Create an empty visual context when processing fails"""
        return VisualContext(
            screen_type='unknown',
            detected_text=[],
            ui_elements=[],
            dominant_colors=[(128, 128, 128)],  # Gray fallback
            game_phase='unknown',
            visual_summary=f"Processing failed: {reason}"
        )
    
    def _load_ui_templates(self) -> Dict[str, Any]:
        """Load UI element templates for detection"""
        # In a full implementation, these would be loaded from files
        # For now, we define basic patterns
        return {
            'dialogue_box': {
                'color_range': [(200, 200, 200), (255, 255, 255)],
                'aspect_ratio': (2.0, 4.0),  # width/height range
                'relative_position': (0.0, 0.6, 1.0, 1.0)  # bottom portion of screen
            },
            'health_bar': {
                'color_range': [(0, 200, 0), (255, 255, 0)],
                'aspect_ratio': (3.0, 8.0),
                'relative_position': (0.0, 0.0, 1.0, 0.3)  # top portion
            },
            'menu': {
                'color_range': [(100, 100, 200), (200, 200, 255)],
                'aspect_ratio': (0.5, 2.0),
                'relative_position': (0.6, 0.0, 1.0, 1.0)  # right side
            }
        }
    
    def process_screenshot(self, screenshot: np.ndarray) -> VisualContext:
        """
        Process a PyBoy screenshot and extract visual context
        
        Args:
            screenshot: RGB numpy array from PyBoy (160x144 pixels)
            
        Returns:
            VisualContext with extracted information
        """
        # Validate input screenshot
        if screenshot is None or screenshot.size == 0:
            return self._create_empty_context("empty_screenshot")
        
        # Check dimensions
        if len(screenshot.shape) != 3 or screenshot.shape[2] != 3:
            return self._create_empty_context("invalid_dimensions")
        
        # Check if image has reasonable dimensions
        height, width = screenshot.shape[:2]
        if height < 10 or width < 10:
            return self._create_empty_context("too_small")
        
        try:
            # Upscale screenshot for better OCR
            upscaled = self._upscale_screenshot(screenshot)
        except Exception as e:
            print(f"⚠️ Screenshot upscaling failed: {e}")
            return self._create_empty_context("upscaling_failed")
        
        # Detect text
        detected_text = self._detect_text(upscaled)
        
        # Analyze UI elements
        ui_elements = self._detect_ui_elements(upscaled)
        
        # Determine screen type
        screen_type = self._classify_screen_type(upscaled, detected_text, ui_elements)
        
        # Extract dominant colors
        dominant_colors = self._get_dominant_colors(screenshot)
        
        # Determine game phase
        game_phase = self._determine_game_phase(screen_type, detected_text)
        
        # Generate visual summary
        visual_summary = self._generate_visual_summary(
            screen_type, detected_text, ui_elements, dominant_colors
        )
        
        return VisualContext(
            screen_type=screen_type,
            detected_text=detected_text,
            ui_elements=ui_elements,
            dominant_colors=dominant_colors,
            game_phase=game_phase,
            visual_summary=visual_summary
        )
    
    def _upscale_screenshot(self, screenshot: np.ndarray, scale_factor: int = 4) -> np.ndarray:
        """Upscale screenshot for better OCR accuracy"""
        # Additional validation
        if screenshot is None or screenshot.size == 0:
            raise ValueError("Empty screenshot provided for upscaling")
        
        height, width = screenshot.shape[:2]
        if height <= 0 or width <= 0:
            raise ValueError(f"Invalid dimensions: {height}x{width}")
        
        new_width = width * scale_factor
        new_height = height * scale_factor
        
        # Use nearest neighbor to preserve pixel art
        upscaled = cv2.resize(screenshot, (new_width, new_height), 
                             interpolation=cv2.INTER_NEAREST)
        
        # Apply slight blur to smooth pixel edges for OCR
        upscaled = cv2.GaussianBlur(upscaled, (3, 3), 0)
        
        return upscaled
    
    def _detect_text(self, image: np.ndarray) -> List[DetectedText]:
        """Detect and extract text using enhanced ROM font decoder"""
        detected_texts = []
        
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image.copy()
            
            height, width = gray.shape
            
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
                
                # Use enhanced font decoder if available
                if self.font_decoder is not None:
                    text_content = self._decode_text_with_rom_font(region_img, region_name)
                else:
                    # Fallback to simple detection
                    text_content = self._simple_text_detection(region_img)
                
                if text_content and len(text_content.strip()) > 0:
                    # Calculate bbox for this region in full image coordinates
                    if region_name == 'dialogue':
                        bbox = (0, int(height * 0.7), width, int(height * 0.3))
                    elif region_name == 'ui':
                        bbox = (int(width * 0.6), 0, int(width * 0.4), int(height * 0.3))
                    elif region_name == 'menu':
                        bbox = (int(width * 0.7), 0, int(width * 0.3), height)
                    else:  # world
                        bbox = (0, int(height * 0.3), int(width * 0.6), int(height * 0.4))
                    
                    # Determine confidence based on method used
                    confidence = 0.85 if self.font_decoder is not None else 0.6
                    
                    detected_texts.append(DetectedText(
                        text=text_content,
                        confidence=confidence,
                        bbox=bbox,
                        location=region_name
                    ))
            
            return detected_texts
            
        except Exception as e:
            print(f"⚠️ Text detection failed: {e}")
            return []
    
    def _simple_text_detection(self, region: np.ndarray) -> str:
        """Aggressive text detection - designed to catch more text"""
        
        if region.size == 0:
            return ""
        
        try:
            height, width = region.shape
            
            # Try multiple aggressive approaches to find text
            text_found = False
            best_method_result = ""
            max_contour_count = 0
            
            # Method 1: Multiple thresholds with very low requirements
            thresholds = [60, 120, 180, 220]  # Multiple threshold levels
            
            for threshold in thresholds:
                # Dark text on light background
                _, binary = cv2.threshold(region, threshold, 255, cv2.THRESH_BINARY_INV)
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                valid_contours = 0
                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    area = w * h
                    
                    # Very relaxed criteria - any reasonable character-sized shape
                    if (3 <= w <= width and 
                        3 <= h <= height and 
                        area >= 9):  # Minimum 3x3 pixels
                        valid_contours += 1
                
                if valid_contours > max_contour_count:
                    max_contour_count = valid_contours
                    if valid_contours >= 1:  # Even 1 contour is enough
                        text_found = True
                        best_method_result = self._aggressive_text_guess(region, valid_contours, "threshold")
            
            # Method 2: Adaptive threshold
            try:
                binary_adaptive = cv2.adaptiveThreshold(region, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 7, 2)
                contours, _ = cv2.findContours(binary_adaptive, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                valid_contours = 0
                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    if (2 <= w <= width and 2 <= h <= height):
                        valid_contours += 1
                
                if valid_contours > max_contour_count:
                    max_contour_count = valid_contours
                    text_found = True
                    best_method_result = self._aggressive_text_guess(region, valid_contours, "adaptive")
                    
            except cv2.error:
                pass  # Skip adaptive if it fails
            
            # Method 3: Edge detection (for outlined text)
            edges = cv2.Canny(region, 30, 100)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            valid_contours = 0
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if (2 <= w <= width//2 and 2 <= h <= height//2):
                    valid_contours += 1
            
            if valid_contours > max_contour_count:
                max_contour_count = valid_contours
                text_found = True
                best_method_result = self._aggressive_text_guess(region, valid_contours, "edges")
            
            # Method 4: Look for any significant variation in brightness (text areas)
            brightness_std = np.std(region.astype(np.float32))
            avg_brightness = np.mean(region)
            
            # If there's significant brightness variation, assume text
            if brightness_std > 15 and max_contour_count == 0:  # Fallback when no contours found
                text_found = True
                best_method_result = self._aggressive_text_guess(region, 1, "brightness")
            
            return best_method_result if text_found else ""
            
        except Exception as e:
            print(f"⚠️ Aggressive text detection error: {e}")
            return "TEXT"  # Default fallback - assume there's text
    
    def _aggressive_text_guess(self, region: np.ndarray, contour_count: int, method: str) -> str:
        """Aggressive text content guessing - assumes text is present"""
        
        height, width = region.shape
        avg_brightness = np.mean(region)
        brightness_std = np.std(region.astype(np.float32))
        
        # Determine region characteristics
        is_wide = width > height * 1.3
        is_tall = height > width * 1.3
        is_bright = avg_brightness > 150
        is_dark = avg_brightness < 100
        has_variation = brightness_std > 20
        
        # Context-based guessing with lower thresholds
        
        # Wide, bright regions are likely dialogue
        if is_wide and is_bright:
            return "DIALOGUE"
        
        # Wide regions with text variation (dialogue/menu)
        elif is_wide and has_variation:
            return "MENU TEXT"
        
        # Tall regions are likely menus
        elif is_tall:
            return "MENU"
        
        # Dark regions with text might be title/intro
        elif is_dark and contour_count >= 1:
            return "GAME TITLE"
        
        # Small bright regions might be UI elements
        elif is_bright and width < 100 and height < 50:
            return "UI"
        
        # Medium regions with some text
        elif contour_count >= 2:
            return "TEXT"
        
        # Single text element
        elif contour_count >= 1:
            return "WORD"
        
        # Fallback - if we detected any variation, assume text
        elif has_variation:
            return "TEXT"
        
        # Last resort - method-based fallback
        elif method == "brightness":
            return "CONTENT"
        
        return "TEXT"  # Always return something if we got this far
    
    def _decode_text_with_rom_font(self, region: np.ndarray, region_name: str) -> str:
        """Decode text using the enhanced ROM font decoder"""
        
        if region is None or region.size == 0:
            return ""
        
        try:
            # Use the enhanced font decoder's decode_text_region method
            decoded_text = self.font_decoder.decode_text_region(
                region, 
                char_width=8, 
                char_height=8, 
                min_confidence=0.6
            )
            
            # If ROM decoder returns empty, try with palette awareness (if available)
            if not decoded_text.strip() and hasattr(self.font_decoder, 'decode_text_region_with_palette'):
                screen_type_map = {
                    'dialogue': 'dialogue',
                    'menu': 'menu', 
                    'ui': 'battle',
                    'world': 'overworld'
                }
                screen_type = screen_type_map.get(region_name, 'overworld')
                
                decoded_text = self.font_decoder.decode_text_region_with_palette(
                    region,
                    screen_type=screen_type,
                    char_width=8,
                    char_height=8,
                    min_confidence=0.6
                )
            
            # If still empty, fallback to simple detection 
            if not decoded_text.strip():
                decoded_text = self._simple_text_detection(region)
            
            return decoded_text.strip()
            
        except Exception as e:
            print(f"⚠️ ROM font decoding error for {region_name}: {e}")
            # Fallback to simple detection
            return self._simple_text_detection(region)
    
    def _guess_dense_text_content(self, region: np.ndarray, contour_count: int) -> str:
        """Legacy method - now redirects to aggressive guess"""
        return self._aggressive_text_guess(region, contour_count, "legacy")
    
    def _guess_sparse_text_content(self, region: np.ndarray, contour_count: int) -> str:
        """Balanced guess for sparse text - more flexible"""
        
        height, width = region.shape
        
        # More flexible patterns for better detection
        
        # Very wide regions might be status text
        if (width > height * 2.5 and 
            contour_count <= 3 and 
            width >= 25):  # More flexible
            return "HP"  # Status text like HP/level
        
        # Small compact regions might be level indicators
        elif (contour_count <= 3 and 
              width <= 30 and height <= 20 and 
              width >= height * 0.8):  # More flexible
            return "Lv"  # Level indicator
        
        # Medium-sized regions might be UI text
        elif (contour_count >= 1 and contour_count <= 4 and
              20 <= width <= 80 and 10 <= height <= 40):
            return "UI"  # Generic UI text
        
        # Large single elements might be buttons or labels
        elif (contour_count == 1 and 
              width >= 50 and height >= 20):
            return "BUTTON"  # Button or large label
        
        # Default for any detected text-like patterns
        elif contour_count >= 1:
            return "TEXT"  # Generic text indicator
        
        return ""
    
    def _classify_text_location(self, bbox: Tuple[int, int, int, int], 
                               image_shape: Tuple[int, int]) -> str:
        """Classify where text appears on screen"""
        x1, y1, x2, y2 = bbox
        height, width = image_shape[:2]
        
        # Relative positions
        rel_y = (y1 + y2) / (2 * height)
        rel_x = (x1 + x2) / (2 * width)
        
        # Classify based on position
        if rel_y > 0.7:
            return 'dialogue'  # Bottom of screen
        elif rel_y < 0.3 and rel_x > 0.6:
            return 'ui'  # Top right (stats, menus)
        elif rel_x > 0.7:
            return 'menu'  # Right side
        else:
            return 'world'  # In-game world text
    
    def _detect_ui_elements(self, image: np.ndarray) -> List[GameUIElement]:
        """Detect common Pokemon UI elements"""
        ui_elements = []
        
        # Validate image before processing
        if image is None or image.size == 0:
            return ui_elements
        
        if len(image.shape) != 3 or image.shape[2] != 3:
            return ui_elements
        
        try:
            # Convert to HSV for better color detection
            hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            
            # Detect dialogue boxes (typically white/light colored rectangles at bottom)
            dialogue_box = self._detect_dialogue_box(image, hsv_image)
            if dialogue_box:
                ui_elements.append(dialogue_box)
            
            # Detect health bars (green/yellow/red horizontal bars)
            health_bars = self._detect_health_bars(image, hsv_image)
            ui_elements.extend(health_bars)
            
            # Detect menu boxes
            menus = self._detect_menus(image, hsv_image)
            ui_elements.extend(menus)
            
        except Exception as e:
            print(f"⚠️ UI element detection failed: {e}")
        
        return ui_elements
    
    def _detect_dialogue_box(self, image: np.ndarray, hsv_image: np.ndarray) -> Optional[GameUIElement]:
        """Detect dialogue/text boxes with improved detection"""
        # Validate input
        if image is None or image.size == 0:
            return None
        
        height, width = image.shape[:2]
        if height < 10 or width < 10:
            return None
        
        # Look for dialogue boxes in bottom portion of screen (Pokemon Crystal dialogue)
        # Dialogue boxes are typically in bottom 50% of screen
        lower_region = image[int(height * 0.5):, :]
        
        # Check if lower region is valid
        if lower_region.size == 0:
            return None
        
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(lower_region, cv2.COLOR_RGB2GRAY)
        except cv2.error as e:
            print(f"⚠️ Color conversion error in dialogue detection: {e}")
            return None
        
        # Try multiple detection methods for dialogue boxes
        dialogue_found = False
        best_contour = None
        best_area = 0
        
        # Method 1: White/light areas (traditional dialogue boxes)
        _, binary_light = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
        contours_light, _ = cv2.findContours(binary_light, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours_light:
            area = cv2.contourArea(contour)
            if area > (width * height * 0.05):  # Lowered threshold
                x, y, w, h = cv2.boundingRect(contour)
                # Check if it looks like a dialogue box (wide rectangle)
                if w > h and w >= width * 0.6 and h >= 20:  # Wide dialogue box
                    if area > best_area:
                        best_area = area
                        best_contour = (x, y, w, h)
                        dialogue_found = True
        
        # Method 2: Dark borders around dialogue (dialogue box borders)
        _, binary_borders = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
        contours_borders, _ = cv2.findContours(binary_borders, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours_borders:
            area = cv2.contourArea(contour)
            if area > (width * height * 0.08):  # Larger area for border detection
                x, y, w, h = cv2.boundingRect(contour)
                # Check if it's a dialogue-sized rectangle
                if w >= width * 0.7 and h >= 25 and w > h * 2:  # Wide dialogue box
                    if area > best_area:
                        best_area = area
                        best_contour = (x, y, w, h)
                        dialogue_found = True
        
        # Method 3: Look for text regions that might indicate dialogue
        # Use edge detection to find text-heavy areas
        edges = cv2.Canny(gray, 50, 150)
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=2)
        contours_text, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours_text:
            area = cv2.contourArea(contour)
            if area > (width * height * 0.06):
                x, y, w, h = cv2.boundingRect(contour)
                # Check for dialogue-like text regions
                if w >= width * 0.5 and h >= 15 and w > h * 1.5:
                    if area > best_area:
                        best_area = area
                        best_contour = (x, y, w, h)
                        dialogue_found = True
        
        if dialogue_found and best_contour:
            x, y, w, h = best_contour
            # Adjust y coordinate for full image
            y += int(height * 0.5)
            
            return GameUIElement(
                element_type='dialogue_box',
                bbox=(x, y, x + w, y + h),
                confidence=0.8
            )
        
        return None
    
    def _detect_health_bars(self, image: np.ndarray, hsv_image: np.ndarray) -> List[GameUIElement]:
        """Detect Pokemon health bars with strict battle-specific criteria"""
        health_bars = []
        
        try:
            # Very restrictive color ranges for actual health bars only
            health_colors = [
                ((45, 150, 150), (75, 255, 255)),  # Green (very restrictive)
                ((25, 150, 150), (35, 255, 255)),  # Yellow (very restrictive)
                ((0, 150, 150), (15, 255, 255))    # Red (very restrictive)
            ]
            
            height, width = image.shape[:2]
            
            for i, (lower, upper) in enumerate(health_colors):
                mask = cv2.inRange(hsv_image, lower, upper)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Very strict criteria for health bars
                    aspect_ratio = w / h if h > 0 else 0
                    area = w * h
                    rel_y = y / height
                    rel_x = x / width
                    
                    # Health bars must be:
                    # 1. Horizontal rectangles with aspect ratio 4:1 to 10:1 (more strict)
                    # 2. Substantial size (not decorative elements)
                    # 3. In battle UI areas (top 35% and not center)
                    # 4. Not in home/building interior areas (center region)
                    if (4 <= aspect_ratio <= 10 and 
                        w >= 40 and h >= 4 and h <= 12 and
                        area >= 160 and area <= 1000 and
                        rel_y <= 0.35 and
                        not (0.2 <= rel_x <= 0.8 and 0.3 <= rel_y <= 0.7)):  # Exclude center area (home interior)
                        
                        health_bars.append(GameUIElement(
                            element_type='healthbar',
                            bbox=(x, y, x + w, y + h),
                            confidence=0.9
                        ))
        
        except Exception as e:
            print(f"⚠️ Health bar detection error: {e}")
        
        return health_bars
    
    def _detect_menus(self, image: np.ndarray, hsv_image: np.ndarray) -> List[GameUIElement]:
        """Detect menu boxes"""
        menus = []
        
        # Validate input
        if image is None or image.size == 0:
            return menus
        
        try:
            # Look for blue/dark colored rectangular regions (typical menu colors)
            # This is a simplified detection - in practice would be more sophisticated
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Find dark regions that might be menus
            _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            height, width = image.shape[:2]
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > (width * height * 0.05):  # Reasonable menu size
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Check aspect ratio (menus are typically taller than wide or square)
                    if h >= w * 0.8:
                        menus.append(GameUIElement(
                            element_type='menu',
                            bbox=(x, y, x + w, y + h),
                            confidence=0.6
                        ))
        
        except Exception as e:
            print(f"⚠️ Menu detection error: {e}")
        
        return menus
    
    def _classify_screen_type(self, image: np.ndarray, detected_text: List[DetectedText], 
                             ui_elements: List[GameUIElement]) -> str:
        """Classify what type of screen is being shown with improved accuracy"""
        
        all_text = ' '.join([t.text.lower() for t in detected_text])
        
        # Check for indoor/home environment FIRST to prevent false classifications
        if self._is_indoor_environment(image, detected_text, all_text):
            return 'overworld'  # Indoor areas are part of overworld
        
        # Check for actual dialogue (must be substantial dialogue content)
        dialogue_texts = [t for t in detected_text if t.location == 'dialogue']
        dialogue_boxes = [e for e in ui_elements if e.element_type == 'dialogue_box']
        
        # Only classify as dialogue if there's substantial dialogue content
        if self._is_actual_dialogue(dialogue_texts, dialogue_boxes, all_text):
            return 'dialogue'
        
        # Check for actual battle context (requires multiple strong indicators)
        health_bars = [e for e in ui_elements if e.element_type == 'healthbar']
        
        # Battle detection requires health bars AND battle-specific context
        if health_bars and self._is_actual_battle(image, detected_text, health_bars):
            return 'battle'
        
        # Check for actual menus (not just any UI elements)
        menu_texts = [t for t in detected_text if t.location == 'menu']
        menu_boxes = [e for e in ui_elements if e.element_type == 'menu']
        
        if self._is_actual_menu(menu_texts, menu_boxes, all_text):
            return 'menu'
        
        # Look for intro/title screens (prioritize specific intro indicators)
        intro_keywords = ['press start', 'new game', 'continue', 'pokemon crystal', 'game freak']
        title_keywords = ['pokemon', 'crystal', 'version']
        
        # Strong intro indicators (definitive intro screens)
        if any(phrase in all_text for phrase in intro_keywords):
            return 'intro'
        
        # Weaker title indicators (could be intro if no other strong signals)
        if any(word in all_text for word in title_keywords) and not all_text.strip():
            # Only classify as intro if minimal other content (title screen)
            return 'intro'
        
        # Default to overworld
        return 'overworld'
    
    def _is_indoor_environment(self, image: np.ndarray, detected_text: List[DetectedText], all_text: str) -> bool:
        """Detect if we're in an indoor environment (home, building, etc.)"""
        
        # Don't classify intro/menu screens as indoor (even if they mention 'new game')
        intro_exclusions = ['new game', 'continue', 'press start', 'pokemon crystal']
        is_intro_screen = any(phrase in all_text for phrase in intro_exclusions)
        
        if is_intro_screen:
            return False  # Never classify intro screens as indoor
        
        # Strong indoor text indicators (player's home specifically)
        strong_indoor_keywords = ['mom', 'bed', 'pc', 'home', 'house', 'room', 'your room']
        weak_indoor_keywords = ['inside', 'stairs', 'upstairs', 'downstairs', 'door', 'floor']
        
        strong_indoor_text = any(keyword in all_text for keyword in strong_indoor_keywords)
        weak_indoor_text = any(keyword in all_text for keyword in weak_indoor_keywords)
        
        # Check visual characteristics of indoor environments
        height, width = image.shape[:2]
        
        try:
            # Convert to HSV for color analysis
            hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            
            # Check for brown/wooden colors (furniture)
            brown_lower = np.array([8, 30, 30])
            brown_upper = np.array([25, 255, 200])
            brown_mask = cv2.inRange(hsv_image, brown_lower, brown_upper)
            brown_ratio = np.sum(brown_mask > 0) / (height * width)
            
            # Check for typical indoor lighting patterns
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            avg_brightness = np.mean(gray)
            brightness_std = np.std(gray)
            
            # Check for indoor color palette (less vibrant than outdoor/battle)
            saturation = hsv_image[:, :, 1]
            avg_saturation = np.mean(saturation)
            
            # Indoor visual indicators:
            # 1. Some brown/wooden elements (furniture)
            # 2. Moderate brightness (not too bright like battles)
            # 3. Lower saturation (indoor lighting is more muted)
            # 4. Consistent lighting (lower brightness variation)
            visual_indoor_score = 0
            
            if brown_ratio > 0.05:  # Some furniture
                visual_indoor_score += 2
            if 70 <= avg_brightness <= 170:  # Indoor lighting range
                visual_indoor_score += 1
            if avg_saturation < 80:  # Muted colors
                visual_indoor_score += 1
            if brightness_std < 40:  # Consistent lighting
                visual_indoor_score += 1
            
            visual_indoor_indicators = visual_indoor_score >= 3
            
        except Exception as e:
            print(f"⚠️ Indoor detection visual analysis error: {e}")
            visual_indoor_indicators = False
        
        # Definitive indoor if strong text indicators
        if strong_indoor_text:
            return True
        
        # Likely indoor if weak text + visual indicators
        if weak_indoor_text and visual_indoor_indicators:
            return True
        
        # Indoor if strong visual indicators alone
        return visual_indoor_indicators
    
    def _is_actual_battle(self, image: np.ndarray, detected_text: List[DetectedText], 
                         health_bars: List[GameUIElement]) -> bool:
        """Determine if health bars indicate an actual battle vs UI elements - very strict"""
        
        # Battle indicators
        all_text = ' '.join([t.text.lower() for t in detected_text])
        
        # Check for explicit battle-specific text (more specific keywords)
        battle_keywords = ['fight', 'run', 'pkmn', 'attack', 'move', 'wild', 'appeared', 'fainted']
        strong_battle_keywords = ['fight', 'run', 'wild', 'appeared', 'fainted']
        
        battle_text_found = any(keyword in all_text for keyword in battle_keywords)
        strong_battle_text = any(keyword in all_text for keyword in strong_battle_keywords)
        
        # Exclude indoor/home environment
        indoor_keywords = ['home', 'house', 'room', 'mom', 'bed', 'pc']
        is_indoor = any(keyword in all_text for keyword in indoor_keywords)
        
        if is_indoor:
            return False  # Never classify indoor areas as battles
        
        # Check health bar positioning (battle health bars are in very specific locations)
        height, width = image.shape[:2]
        valid_battle_bars = 0
        
        for bar in health_bars:
            x1, y1, x2, y2 = bar.bbox
            rel_y = y1 / height
            rel_x = x1 / width
            
            # Battle health bars are typically:
            # - In top-left or top-right corners
            # - Very specific positioning
            # - Not in center areas where furniture/decorations might be
            if ((rel_y <= 0.25 and rel_x <= 0.6) or  # Top-left area
                (rel_y <= 0.25 and rel_x >= 0.4)):   # Top-right area
                valid_battle_bars += 1
        
        # Very strict requirements:
        # 1. Must have strong battle text OR multiple properly positioned health bars
        # 2. Cannot be in indoor environment
        # 3. Must have at least one valid battle-positioned health bar
        return (strong_battle_text or valid_battle_bars >= 2) and valid_battle_bars > 0
    
    def _is_actual_dialogue(self, dialogue_texts: List[DetectedText], dialogue_boxes: List[GameUIElement], all_text: str) -> bool:
        """Determine if detected dialogue elements indicate actual dialogue vs UI noise - improved"""
        
        # Check for dialogue content
        dialogue_content = ' '.join([t.text for t in dialogue_texts])
        dialogue_length = len(dialogue_content.strip())
        
        # Strong dialogue indicators
        dialogue_keywords = ['said', 'says', 'asked', 'replied', 'told', 'explained', 'professor', 'mom', 'dialogue']
        has_dialogue_keywords = any(keyword in all_text for keyword in dialogue_keywords)
        
        # Check if we have actual dialogue text content (including generic TEXT in dialogue region)
        dialogue_text_indicators = ['dialogue', 'text']  # Our dialogue detection
        has_dialogue_text = any(indicator in all_text for indicator in dialogue_text_indicators)
        
        # Question indicators
        has_questions = '?' in dialogue_content
        
        # Dialogue box visual indicators (prioritize this!)
        has_dialogue_box = len(dialogue_boxes) > 0
        
        # Check if text is specifically detected in dialogue location
        has_dialogue_location_text = any(t.location == 'dialogue' for t in dialogue_texts)
        
        # Multiple text lines often indicate dialogue
        has_multiple_text = len(dialogue_texts) > 1 or dialogue_length > 15
        
        # Don't exclude home environment completely - dialogues can happen at home
        home_keywords = ['mom', 'home', 'house', 'bed', 'pc']
        is_home_context = any(keyword in all_text for keyword in home_keywords)
        
        # Priority-based dialogue detection:
        # 1. Visual dialogue box is the strongest indicator
        # 2. Text detected in dialogue location
        # 3. Traditional dialogue keywords
        # 4. Question marks or substantial text
        
        # Strong visual evidence
        visual_dialogue_evidence = has_dialogue_box and (has_dialogue_location_text or has_dialogue_text)
        
        # Traditional text evidence
        text_dialogue_evidence = (
            has_dialogue_keywords or 
            (has_dialogue_text and dialogue_length > 3) or
            has_multiple_text or
            has_questions
        )
        
        # Accept dialogue if we have strong visual evidence OR traditional evidence
        substantial_dialogue = visual_dialogue_evidence or text_dialogue_evidence
        
        # Only exclude very minimal text in home context
        return substantial_dialogue and not (is_home_context and dialogue_length < 3 and not has_dialogue_box)
    
    def _is_actual_menu(self, menu_texts: List[DetectedText], menu_boxes: List[GameUIElement], all_text: str) -> bool:
        """Determine if detected menu elements indicate actual menu vs UI noise"""
        
        # Start menu specific indicators (title screen menus)
        start_menu_indicators = ['new game', 'continue', 'option', 'mystery gift']
        is_start_menu = any(indicator in all_text for indicator in start_menu_indicators)
        
        # In-game menu indicators
        game_menu_keywords = ['pokemon', 'bag', 'trainer', 'save', 'settings', 'items', 'status']
        has_game_menu_keywords = any(keyword in all_text for keyword in game_menu_keywords)
        
        # Menu navigation indicators
        navigation_indicators = ['▶', '▼', '▲', '◀', 'select', 'cancel', 'back']
        has_navigation = any(indicator in all_text for indicator in navigation_indicators)
        
        # Exclude home environment from menu classification
        home_keywords = ['mom', 'home', 'house', 'bed', 'pc']
        is_home_context = any(keyword in all_text for keyword in home_keywords)
        
        # Must have clear menu indicators and not be in home
        # Start menu gets priority for title screen detection
        return (is_start_menu or has_game_menu_keywords or has_navigation or len(menu_boxes) > 0) and not is_home_context
    
    def _get_dominant_colors(self, image: np.ndarray, k: int = 3) -> List[Tuple[int, int, int]]:
        """Extract dominant colors from the screenshot"""
        # Validate input
        if image is None or image.size == 0:
            return [(128, 128, 128)]  # Gray fallback
        
        if len(image.shape) != 3 or image.shape[2] != 3:
            return [(128, 128, 128)]  # Gray fallback
        
        try:
            # Reshape image for k-means
            data = image.reshape((-1, 3))
            data = np.float32(data)
            
            # Apply k-means clustering
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
            _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            
            # Convert back to integers and return
            centers = np.uint8(centers)
            return [tuple(map(int, center)) for center in centers]
            
        except Exception as e:
            print(f"⚠️ Color analysis error: {e}")
            return [(128, 128, 128)]  # Gray fallback
    
    def _determine_game_phase(self, screen_type: str, detected_text: List[DetectedText]) -> str:
        """Determine the current phase of gameplay"""
        all_text = ' '.join([t.text.lower() for t in detected_text])
        
        if screen_type == 'intro':
            return 'intro'
        elif screen_type == 'battle':
            return 'battle'
        elif screen_type == 'menu':
            if any(word in all_text for word in ['pokemon', 'bag', 'trainer']):
                return 'menu_navigation'
            else:
                return 'menu_navigation'
        elif screen_type == 'dialogue':
            return 'dialogue_interaction'
        else:
            return 'overworld_exploration'
    
    def _generate_visual_summary(self, screen_type: str, detected_text: List[DetectedText],
                                ui_elements: List[GameUIElement], 
                                dominant_colors: List[Tuple[int, int, int]]) -> str:
        """Generate a text summary of the visual context"""
        
        summary_parts = []
        
        # Screen type
        summary_parts.append(f"Screen: {screen_type}")
        
        # Text content
        if detected_text:
            texts = [t.text for t in detected_text[:3]]  # First 3 texts
            summary_parts.append(f"Text: {', '.join(texts)}")
        
        # UI elements
        if ui_elements:
            elements = list(set([e.element_type for e in ui_elements]))
            summary_parts.append(f"UI: {', '.join(elements)}")
        
        # Color context
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
    
    def encode_screenshot_for_llm(self, screenshot: np.ndarray) -> str:
        """Encode screenshot as base64 for LLM vision models"""
        # Convert to PIL Image
        pil_image = Image.fromarray(screenshot)
        
        # Upscale for better visibility
        upscaled = pil_image.resize((320, 288), Image.NEAREST)  # 2x scale
        
        # Convert to base64
        buffer = io.BytesIO()
        upscaled.save(buffer, format='PNG')
        buffer.seek(0)
        encoded = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return encoded


def test_vision_processor():
    """Test the vision processor with a sample screenshot"""
    print("🧪 Testing Vision Processor...")
    
    # Create a simple test image (simulating Game Boy screen)
    test_image = np.zeros((144, 160, 3), dtype=np.uint8)
    test_image.fill(255)  # White background
    
    # Add some colored areas to simulate game elements
    test_image[100:140, 10:150] = [200, 200, 255]  # Light blue dialogue area
    test_image[10:30, 120:150] = [0, 255, 0]      # Green health bar
    
    # Initialize processor
    processor = PokemonVisionProcessor()
    
    # Process the test image
    context = processor.process_screenshot(test_image)
    
    print(f"✅ Screen Type: {context.screen_type}")
    print(f"✅ Game Phase: {context.game_phase}")
    print(f"✅ UI Elements: {len(context.ui_elements)}")
    print(f"✅ Detected Text: {len(context.detected_text)}")
    print(f"✅ Visual Summary: {context.visual_summary}")
    print(f"✅ Dominant Colors: {context.dominant_colors}")
    
    print("\n🎉 Vision processor test completed!")


if __name__ == "__main__":
    test_vision_processor()
