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
        """Detect and extract text using Pokemon Crystal font decoder"""
        detected_texts = []
        
        if self.font_decoder is None:
            # No font decoder available, return empty list
            return detected_texts
        
        try:
            # Extract text regions using ROM font decoder
            height, width = image.shape[:2]
            
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
                
                # Use ROM font decoder to extract text from this region
                try:
                    decoded_text = self.font_decoder.decode_text_region(region_img)
                    
                    if decoded_text and len(decoded_text.strip()) > 0:
                        # Calculate bbox for this region in full image coordinates
                        if region_name == 'dialogue':
                            bbox = (0, int(height * 0.7), width, int(height * 0.3))
                        elif region_name == 'ui':
                            bbox = (int(width * 0.6), 0, int(width * 0.4), int(height * 0.3))
                        elif region_name == 'menu':
                            bbox = (int(width * 0.7), 0, int(width * 0.3), height)
                        else:  # world
                            bbox = (0, int(height * 0.3), int(width * 0.6), int(height * 0.4))
                        
                        # Split into lines and add each line
                        lines = decoded_text.split('\n')
                        for i, line in enumerate(lines):
                            line = line.strip()
                            if len(line) > 0:
                                # Adjust bbox for individual lines
                                line_height = bbox[3] // max(1, len(lines))
                                line_bbox = (
                                    bbox[0], 
                                    bbox[1] + i * line_height,
                                    bbox[2],
                                    min(line_height, bbox[3] - i * line_height)
                                )
                                
                                detected_texts.append(DetectedText(
                                    text=line,
                                    confidence=0.8,  # High confidence for ROM decoder
                                    bbox=line_bbox,
                                    location=region_name
                                ))
                        
                except Exception as region_e:
                    # If ROM decoder fails, try basic text detection
                    print(f"⚠️ ROM decoder failed for {region_name}: {region_e}")
                    continue
            
            return detected_texts
            
        except Exception as e:
            print(f"⚠️ Text detection failed: {e}")
            return []
    
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
        """Detect dialogue/text boxes"""
        # Validate input
        if image is None or image.size == 0:
            return None
        
        height, width = image.shape[:2]
        if height < 10 or width < 10:
            return None
        
        # Look for white/light areas in bottom 40% of screen
        lower_region = image[int(height * 0.6):, :]
        
        # Check if lower region is valid
        if lower_region.size == 0:
            return None
        
        try:
            # Simple white detection
            gray = cv2.cvtColor(lower_region, cv2.COLOR_RGB2GRAY)
        except cv2.error as e:
            print(f"⚠️ Color conversion error in dialogue detection: {e}")
            return None
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > (width * height * 0.1):  # Large enough area
                x, y, w, h = cv2.boundingRect(contour)
                
                # Adjust y coordinate for full image
                y += int(height * 0.6)
                
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
        """Determine if detected dialogue elements indicate actual dialogue vs UI noise"""
        
        # Check for substantial dialogue content
        dialogue_content = ' '.join([t.text for t in dialogue_texts])
        dialogue_length = len(dialogue_content.strip())
        
        # Dialogue indicators
        dialogue_keywords = ['said', 'says', 'asked', 'replied', 'told', 'explained', 'professor', 'mom']
        has_dialogue_keywords = any(keyword in all_text for keyword in dialogue_keywords)
        
        # Question indicators
        has_questions = '?' in dialogue_content
        
        # Exclude home environment from dialogue classification
        home_keywords = ['mom', 'home', 'house', 'bed', 'pc']
        is_home_context = any(keyword in all_text for keyword in home_keywords)
        
        # Must have substantial content OR clear dialogue indicators
        # But not if we're clearly in a home environment with minimal dialogue
        substantial_dialogue = dialogue_length > 10 or has_dialogue_keywords or has_questions
        
        return substantial_dialogue and not (is_home_context and dialogue_length < 20)
    
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
