from enum import Enum
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import cv2


@dataclass
class GBCPalette:
    """Game Boy Color Palette data structure"""
    name: str
    colors: List[Tuple[int, int, int]]
    description: str = ""
    
    # Predefined palette instances
    DEFAULT = None
    TEXT_WHITE = None
    TEXT_BLACK = None
    DIALOGUE_NORMAL = None
    DIALOGUE_DARK = None

# Initialize predefined palettes
GBCPalette.DEFAULT = GBCPalette(
    name='default',
    colors=[(255, 255, 255), (192, 192, 192), (96, 96, 96), (0, 0, 0)],
    description='Default Game Boy Color palette'
)

GBCPalette.TEXT_WHITE = GBCPalette(
    name='text_white',
    colors=[(255, 255, 255), (224, 224, 224), (32, 32, 32), (0, 0, 0)],
    description='White text on dark background'
)

GBCPalette.TEXT_BLACK = GBCPalette(
    name='text_black',
    colors=[(32, 32, 32), (64, 64, 64), (224, 224, 224), (255, 255, 255)],
    description='Black text on light background'
)

GBCPalette.DIALOGUE_NORMAL = GBCPalette(
    name='dialogue_normal',
    colors=[(255, 255, 255), (192, 192, 255), (32, 32, 64), (0, 0, 32)],
    description='Normal dialogue palette'
)

GBCPalette.DIALOGUE_DARK = GBCPalette(
    name='dialogue_dark',
    colors=[(224, 224, 255), (160, 160, 224), (64, 64, 160), (32, 32, 128)],
    description='Dark dialogue palette'
)


class GameBoyColorPalette:
    """Game Boy Color Palette Detection and Analysis"""
    
    def __init__(self):
        # Initialize palettes dictionary with both enum keys and string keys
        self.palettes = {
            'default': GBCPalette.DEFAULT,
            'text_white': GBCPalette.TEXT_WHITE,
            'text_black': GBCPalette.TEXT_BLACK,
            'dialogue_normal': GBCPalette.DIALOGUE_NORMAL,
            'dialogue_dark': GBCPalette.DIALOGUE_DARK,
            'menu_standard': GBCPalette(
                name='menu_standard',
                colors=[(248, 248, 248), (184, 184, 184), (88, 88, 88), (8, 8, 8)],
                description='Standard menu palette'
            ),
            'menu_battle': GBCPalette(
                name='menu_battle',
                colors=[(255, 248, 240), (192, 160, 128), (96, 64, 32), (32, 16, 8)],
                description='Battle menu palette'
            ),
            'pokemon_crystal': GBCPalette(
                name='pokemon_crystal',
                colors=[(240, 248, 255), (160, 192, 224), (64, 96, 160), (16, 32, 96)],
                description='Pokemon Crystal intro palette'
            ),
            'health_green': GBCPalette(
                name='health_green',
                colors=[(144, 248, 144), (88, 200, 88), (32, 120, 32), (8, 64, 8)],
                description='Green health bar palette'
            ),
            'health_yellow': GBCPalette(
                name='health_yellow',
                colors=[(248, 248, 144), (200, 200, 88), (120, 120, 32), (64, 64, 8)],
                description='Yellow health bar palette'
            ),
            'health_red': GBCPalette(
                name='health_red',
                colors=[(248, 144, 144), (200, 88, 88), (120, 32, 32), (64, 8, 8)],
                description='Red health bar palette'
            )
        }
        
        # Screen type to palette mapping
        self.screen_type_mapping = {
            'dialogue': 'dialogue_normal',
            'battle': 'menu_battle',
            'menu': 'menu_standard',
            'overworld': 'text_white',
            'intro': 'pokemon_crystal'
        }
        
        # Color distance threshold
        self.distance_threshold = 48.0
    
    def get_palette(self, palette_name: str) -> Optional[GBCPalette]:
        """Get a palette by name"""
        return self.palettes.get(palette_name)
    
    def list_palettes(self) -> List[str]:
        """List all available palette names"""
        return list(self.palettes.keys())
    
    def convert_to_gbc_palette(self, image: np.ndarray, palette_name: str) -> np.ndarray:
        """Convert image to use specific GBC palette colors"""
        palette = self.get_palette(palette_name)
        if palette is None:
            return image
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Create RGB output
        if len(image.shape) == 2:
            output = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
        else:
            output = image.copy()
        
        # Map grayscale levels to palette colors
        palette_colors = np.array(palette.colors, dtype=np.uint8)
        
        # Divide grayscale range into 4 levels
        for i in range(4):
            level_min = i * 64
            level_max = (i + 1) * 64 - 1 if i < 3 else 255
            mask = (gray >= level_min) & (gray <= level_max)
            output[mask] = palette_colors[i]
        
        return output
    
    def detect_palette_from_image(self, image: np.ndarray) -> str:
        """Detect the most likely palette used in the image"""
        # Convert to RGB if needed
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[-1] == 4:
            image = image[..., :3]
        
        # Get dominant colors
        dominant_colors = self._get_dominant_colors(image, k=4)
        
        # Compare against each palette
        best_match = 'default'
        best_score = float('inf')
        
        for palette_name, palette in self.palettes.items():
            palette_colors = [(int(c[0]), int(c[1]), int(c[2])) for c in palette.colors]
            score = self._calculate_palette_similarity(dominant_colors, palette_colors)
            if score < best_score:
                best_score = score
                best_match = palette_name
        
        return best_match
    
    def _get_dominant_colors(self, image: np.ndarray, k: int = 4) -> List[Tuple[int, int, int]]:
        """Extract dominant colors from image using k-means clustering"""
        try:
            # Reshape image to be a list of pixels
            pixels = image.reshape(-1, 3).astype(np.float32)
            
            # Handle single pixel case
            if pixels.shape[0] == 1:
                # Return the single pixel color repeated k times
                single_color = (int(pixels[0, 0]), int(pixels[0, 1]), int(pixels[0, 2]))
                return [single_color] * k
            
            # Ensure k doesn't exceed number of pixels
            actual_k = min(k, pixels.shape[0])
            
            # Apply k-means clustering
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
            _, labels, centers = cv2.kmeans(pixels, actual_k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            
            # Convert centers to integers and return as tuples
            centers = centers.astype(np.uint8)
            result = [(int(c[0]), int(c[1]), int(c[2])) for c in centers]
            
            # Pad with the last color if we got fewer than k colors
            while len(result) < k:
                result.append(result[-1] if result else (128, 128, 128))
            
            return result
        
        except cv2.error:
            # Fallback: return gray colors
            return [(128, 128, 128)] * k
    
    def _calculate_palette_similarity(self, colors1: List[Tuple[int, int, int]], 
                                    colors2: List[Tuple[int, int, int]]) -> float:
        """Calculate similarity between two color palettes"""
        if not colors1 or not colors2:
            return 0.0
        
        total_distance = 0.0
        comparisons = 0
        
        for c1 in colors1:
            min_distance = float('inf')
            for c2 in colors2:
                # Calculate Euclidean distance in RGB space
                distance = np.sqrt(sum((a - b) ** 2 for a, b in zip(c1, c2)))
                min_distance = min(min_distance, distance)
            total_distance += min_distance
            comparisons += 1
        
        # Normalize the distance to [0, 1] range
        # Maximum possible distance in RGB space is sqrt(3 * 255^2) ≈ 441.67
        max_distance = np.sqrt(3 * 255 ** 2)
        normalized_distance = (total_distance / comparisons) / max_distance if comparisons > 0 else 0.0
        
        return min(normalized_distance, 1.0)  # Ensure it doesn't exceed 1.0
    
    def create_color_aware_template(self, template: np.ndarray, 
                                  source_palette: str, target_palette: str) -> np.ndarray:
        """Create color-aware template by converting between palettes"""
        if source_palette == target_palette:
            return template
        
        # Convert source to target palette
        converted = self.convert_to_gbc_palette(template, target_palette)
        
        # Convert back to grayscale for template matching
        if len(converted.shape) == 3:
            return cv2.cvtColor(converted, cv2.COLOR_BGR2GRAY)
        
        return converted
    
    def enhance_template_for_lighting(self, template: np.ndarray, 
                                    lighting_condition: str) -> np.ndarray:
        """Enhance template for different lighting conditions"""
        if lighting_condition == 'normal':
            return template
        
        enhanced = template.copy().astype(np.float32)
        
        if lighting_condition == 'bright':
            # Increase contrast and brightness
            enhanced = enhanced * 1.2 + 20
        elif lighting_condition == 'dark':
            # Decrease brightness, increase contrast
            enhanced = enhanced * 1.1 - 15
        elif lighting_condition == 'night':
            # Significant brightness reduction
            enhanced = enhanced * 0.8 - 30
        else:
            return template
        
        # Clamp values to valid range
        enhanced = np.clip(enhanced, 0, 255)
        return enhanced.astype(np.uint8)
    
    def get_palette_for_screen_type(self, screen_type: str) -> str:
        """Get the appropriate palette for a screen type"""
        if not screen_type:
            return 'text_white'
        
        return self.screen_type_mapping.get(screen_type, 'text_white')
    
    def analyze_text_region_colors(self, region: np.ndarray) -> Dict:
        """Analyze color patterns in a potential text region"""
        # Convert to RGB if needed
        if len(region.shape) == 2:
            region_rgb = np.stack([region] * 3, axis=-1)
        elif region.shape[-1] == 4:
            region_rgb = region[..., :3]
        else:
            region_rgb = region
        
        # Convert to grayscale for analysis
        if len(region.shape) == 3:
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        else:
            gray = region
        
        # Get unique colors and frequencies
        colors, counts = np.unique(region_rgb.reshape(-1, 3), axis=0, return_counts=True)
        
        # Calculate statistics
        mean_brightness = float(np.mean(gray))
        std_brightness = float(np.std(gray))
        min_brightness = int(np.min(gray))
        max_brightness = int(np.max(gray))
        contrast = max_brightness - min_brightness
        
        # Detect if high contrast (text likely present)
        is_high_contrast = contrast > 128
        
        # Check if uniform color
        is_uniform = std_brightness < 5.0
        
        # Determine text style based on brightness patterns
        if mean_brightness > 128:
            text_style = "dark_on_light"
        else:
            text_style = "light_on_dark"
        
        # Get most likely palette
        detected_palette = self.detect_palette_from_image(region_rgb)
        
        return {
            "mean_brightness": mean_brightness,
            "std_brightness": std_brightness,
            "min_brightness": min_brightness,
            "max_brightness": max_brightness,
            "contrast": contrast,
            "text_style": text_style,
            "detected_palette": detected_palette,
            "is_high_contrast": is_high_contrast,
            "is_uniform": is_uniform,
            "unique_colors": len(colors)
        }
    
    def preview_palette(self, palette_name: str) -> None:
        """Preview a palette by printing its colors"""
        palette = self.get_palette(palette_name)
        if palette is None:
            print(f"Palette '{palette_name}' not found")
            return
        
        print(f"Palette: {palette.name}")
        print(f"Description: {palette.description}")
        print("Colors:")
        for i, color in enumerate(palette.colors):
            print(f"  {i}: RGB{color}")

def test_gameboy_color_palette():
    """Test function for GameBoyColorPalette"""
    gbc = GameBoyColorPalette()
    
    # Test palette listing
    print("Available palettes:", gbc.list_palettes())
    
    # Test palette retrieval
    palette = gbc.get_palette('text_white')
    if palette:
        print(f"Retrieved palette: {palette.name}")
    
    # Test image analysis
    test_image = np.random.randint(0, 256, (16, 16, 3), dtype=np.uint8)
    analysis = gbc.analyze_text_region_colors(test_image)
    print("Analysis result:", analysis)
    
    # Test palette detection
    detected = gbc.detect_palette_from_image(test_image)
    print("Detected palette:", detected)

if __name__ == "__main__":
    test_gameboy_color_palette()
