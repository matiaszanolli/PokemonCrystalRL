"""
vision_processor.py - Computer Vision for Pokemon Crystal Agent

This module processes PyBoy screenshots to extract visual context
for the LLM agent, including text recognition, UI elements detection,
and game state analysis.
"""

import cv2
import numpy as np
import easyocr
from PIL import Image, ImageEnhance, ImageFilter
from typing import Dict, List, Tuple, Optional, Any
import base64
import io
from dataclasses import dataclass
import re


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
        # Initialize EasyOCR with English language
        print("🔍 Initializing vision processor...")
        self.ocr_reader = easyocr.Reader(['en'], gpu=True)
        
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
        # Upscale screenshot for better OCR
        upscaled = self._upscale_screenshot(screenshot)
        
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
        height, width = screenshot.shape[:2]
        new_width = width * scale_factor
        new_height = height * scale_factor
        
        # Use nearest neighbor to preserve pixel art
        upscaled = cv2.resize(screenshot, (new_width, new_height), 
                             interpolation=cv2.INTER_NEAREST)
        
        # Apply slight blur to smooth pixel edges for OCR
        upscaled = cv2.GaussianBlur(upscaled, (3, 3), 0)
        
        return upscaled
    
    def _detect_text(self, image: np.ndarray) -> List[DetectedText]:
        """Detect and extract text from the screenshot"""
        try:
            # OCR detection
            results = self.ocr_reader.readtext(image, paragraph=False, width_ths=0.9)
            
            detected_texts = []
            for (bbox, text, confidence) in results:
                if confidence > 0.5 and len(text.strip()) > 1:  # Filter low confidence
                    # Convert bbox format
                    x1, y1 = int(bbox[0][0]), int(bbox[0][1])
                    x2, y2 = int(bbox[2][0]), int(bbox[2][1])
                    
                    # Classify text location
                    location = self._classify_text_location((x1, y1, x2, y2), image.shape)
                    
                    detected_texts.append(DetectedText(
                        text=text.strip(),
                        confidence=confidence,
                        bbox=(x1, y1, x2, y2),
                        location=location
                    ))
            
            return detected_texts
            
        except Exception as e:
            print(f"⚠️ OCR detection failed: {e}")
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
        height, width = image.shape[:2]
        
        # Look for white/light areas in bottom 40% of screen
        lower_region = image[int(height * 0.6):, :]
        
        # Simple white detection
        gray = cv2.cvtColor(lower_region, cv2.COLOR_RGB2GRAY)
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
        """Detect Pokemon health bars"""
        health_bars = []
        
        try:
            # Define color ranges for health (green, yellow, red)
            health_colors = [
                ((40, 100, 100), (80, 255, 255)),  # Green
                ((20, 100, 100), (40, 255, 255)),  # Yellow  
                ((0, 100, 100), (20, 255, 255))    # Red
            ]
            
            for i, (lower, upper) in enumerate(health_colors):
                mask = cv2.inRange(hsv_image, lower, upper)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Check if it looks like a health bar (horizontal rectangle)
                    if w > h * 2 and w > 20 and h > 5:
                        health_bars.append(GameUIElement(
                            element_type='healthbar',
                            bbox=(x, y, x + w, y + h),
                            confidence=0.7
                        ))
        
        except Exception as e:
            print(f"⚠️ Health bar detection error: {e}")
        
        return health_bars
    
    def _detect_menus(self, image: np.ndarray, hsv_image: np.ndarray) -> List[GameUIElement]:
        """Detect menu boxes"""
        menus = []
        
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
        """Classify what type of screen is being shown"""
        
        # Check for dialogue
        dialogue_texts = [t for t in detected_text if t.location == 'dialogue']
        dialogue_boxes = [e for e in ui_elements if e.element_type == 'dialogue_box']
        
        if dialogue_texts or dialogue_boxes:
            return 'dialogue'
        
        # Check for battle (health bars present)
        health_bars = [e for e in ui_elements if e.element_type == 'healthbar']
        if health_bars:
            return 'battle'
        
        # Check for menus
        menu_texts = [t for t in detected_text if t.location == 'menu']
        menu_boxes = [e for e in ui_elements if e.element_type == 'menu']
        
        if menu_texts or menu_boxes:
            return 'menu'
        
        # Look for common intro/title text
        all_text = ' '.join([t.text.lower() for t in detected_text])
        if any(word in all_text for word in ['pokemon', 'press', 'start', 'continue', 'new game']):
            return 'intro'
        
        # Default to overworld
        return 'overworld'
    
    def _get_dominant_colors(self, image: np.ndarray, k: int = 3) -> List[Tuple[int, int, int]]:
        """Extract dominant colors from the screenshot"""
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
