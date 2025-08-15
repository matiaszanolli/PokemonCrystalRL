"""
gameboy_color_palette.py - Game Boy Color Palette Utilities

Provides accurate Game Boy Color palette conversion and color-aware template matching
for improved Pokemon Crystal text recognition under different lighting conditions.
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class GBCPalette:
    """Represents a Game Boy Color palette"""
    name: str
    colors: List[Tuple[int, int, int]]  # RGB tuples
    description: str

class GameBoyColorPalette:
    """
    Game Boy Color palette utilities for Pokemon Crystal
    """
    
    def __init__(self):
        """Initialize with Pokemon Crystal palettes"""
        
        # Standard Game Boy Color palettes used in Pokemon Crystal
        self.palettes = {
            # Text palettes
            'text_white': GBCPalette(
                name='text_white',
                colors=[(248, 248, 248), (184, 184, 184), (104, 104, 104), (0, 0, 0)],
                description='White text on dark background'
            ),
            'text_black': GBCPalette(
                name='text_black', 
                colors=[(0, 0, 0), (104, 104, 104), (184, 184, 184), (248, 248, 248)],
                description='Black text on light background'
            ),
            'text_blue': GBCPalette(
                name='text_blue',
                colors=[(224, 248, 248), (152, 192, 216), (80, 136, 184), (16, 56, 104)],
                description='Blue text palette'
            ),
            
            # Menu palettes
            'menu_standard': GBCPalette(
                name='menu_standard',
                colors=[(248, 248, 248), (192, 192, 192), (128, 128, 128), (64, 64, 64)],
                description='Standard menu colors'
            ),
            'menu_battle': GBCPalette(
                name='menu_battle',
                colors=[(248, 224, 192), (216, 176, 144), (152, 112, 88), (88, 64, 48)],
                description='Battle menu colors'
            ),
            
            # Dialogue palettes
            'dialogue_normal': GBCPalette(
                name='dialogue_normal',
                colors=[(248, 248, 240), (208, 208, 200), (144, 144, 136), (64, 64, 56)],
                description='Normal dialogue colors'
            ),
            'dialogue_night': GBCPalette(
                name='dialogue_night',
                colors=[(192, 192, 224), (144, 144, 176), (96, 96, 128), (48, 48, 80)],
                description='Night time dialogue colors'
            ),
            
            # Health bar palettes
            'health_green': GBCPalette(
                name='health_green',
                colors=[(224, 248, 208), (160, 208, 144), (96, 168, 80), (32, 128, 16)],
                description='Green health bar'
            ),
            'health_yellow': GBCPalette(
                name='health_yellow',
                colors=[(248, 248, 160), (224, 208, 112), (200, 168, 64), (176, 128, 16)],
                description='Yellow health bar'
            ),
            'health_red': GBCPalette(
                name='health_red',
                colors=[(248, 192, 192), (216, 144, 144), (184, 96, 96), (152, 48, 48)],
                description='Red health bar'
            ),
            
            # Special palettes
            'pokemon_crystal': GBCPalette(
                name='pokemon_crystal',
                colors=[(240, 248, 255), (192, 216, 232), (128, 168, 192), (64, 104, 136)],
                description='Pokemon Crystal theme colors'
            )
        }
        
        print("🎨 Game Boy Color palette system initialized")
    
    def get_palette(self, palette_name: str) -> Optional[GBCPalette]:
        """
        Get a specific palette by name
        
        Args:
            palette_name: Name of the palette
            
        Returns:
            GBCPalette object or None if not found
        """
        return self.palettes.get(palette_name)
    
    def list_palettes(self) -> List[str]:
        """
        Get list of available palette names
        
        Returns:
            List of palette names
        """
        return list(self.palettes.keys())
    
    def convert_to_gbc_palette(self, image: np.ndarray, palette_name: str) -> np.ndarray:
        """
        Convert an image to a specific Game Boy Color palette
        
        Args:
            image: Input image (grayscale or RGB)
            palette_name: Name of the palette to use
            
        Returns:
            Image converted to the specified palette
        """
        palette = self.get_palette(palette_name)
        if palette is None:
            print(f"⚠️ Unknown palette: {palette_name}")
            return image
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # Map grayscale values to palette colors
        palette_image = np.zeros((gray.shape[0], gray.shape[1], 3), dtype=np.uint8)
        
        # Divide grayscale range into 4 levels (0-3)
        levels = np.digitize(gray, bins=[64, 128, 192, 255]) - 1
        levels = np.clip(levels, 0, 3)
        
        # Apply palette colors
        for level in range(4):
            mask = levels == level
            color = palette.colors[level]
            palette_image[mask] = color
        
        return palette_image
    
    def detect_palette_from_image(self, image: np.ndarray) -> str:
        """
        Detect which Game Boy Color palette best matches an image
        
        Args:
            image: Input image to analyze
            
        Returns:
            Name of the best matching palette
        """
        if len(image.shape) == 3:
            # Get dominant colors
            dominant_colors = self._get_dominant_colors(image, k=4)
        else:
            # For grayscale, create synthetic color analysis
            gray_values = np.unique(image)
            dominant_colors = [(v, v, v) for v in gray_values[:4]]
        
        best_palette = 'text_white'  # Default
        best_score = 0.0
        
        # Compare against each palette
        for palette_name, palette in self.palettes.items():
            score = self._calculate_palette_similarity(dominant_colors, palette.colors)
            if score > best_score:
                best_score = score
                best_palette = palette_name
        
        return best_palette
    
    def _get_dominant_colors(self, image: np.ndarray, k: int = 4) -> List[Tuple[int, int, int]]:
        """
        Extract dominant colors from an image using k-means clustering
        
        Args:
            image: Input image
            k: Number of colors to extract
            
        Returns:
            List of dominant colors as RGB tuples
        """
        try:
            # Reshape image for k-means
            data = image.reshape((-1, 3))
            data = np.float32(data)
            
            # Apply k-means clustering
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
            _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            
            # Convert back to integers
            centers = np.uint8(centers)
            return [tuple(map(int, center)) for center in centers]
            
        except Exception as e:
            print(f"⚠️ Color extraction error: {e}")
            return [(128, 128, 128)] * k  # Gray fallback
    
    def _calculate_palette_similarity(self, colors1: List[Tuple[int, int, int]], 
                                    colors2: List[Tuple[int, int, int]]) -> float:
        """
        Calculate similarity between two color palettes
        
        Args:
            colors1: First palette colors
            colors2: Second palette colors
            
        Returns:
            Similarity score (0.0 to 1.0)
        """
        if not colors1 or not colors2:
            return 0.0
        
        total_distance = 0.0
        comparisons = 0
        
        # Compare each color in first palette with closest in second
        for c1 in colors1:
            min_distance = float('inf')
            for c2 in colors2:
                # Euclidean distance in RGB space
                distance = np.sqrt(sum((a - b) ** 2 for a, b in zip(c1, c2)))
                min_distance = min(min_distance, distance)
            
            total_distance += min_distance
            comparisons += 1
        
        if comparisons == 0:
            return 0.0
        
        # Convert distance to similarity (0-1 scale)
        avg_distance = total_distance / comparisons
        max_distance = np.sqrt(3 * 255 ** 2)  # Maximum possible RGB distance
        similarity = 1.0 - (avg_distance / max_distance)
        
        return similarity
    
    def create_color_aware_template(self, template: np.ndarray, 
                                  source_palette: str = 'text_white',
                                  target_palette: str = 'text_black') -> np.ndarray:
        """
        Create a color-aware template by converting between palettes
        
        Args:
            template: Original template (8x8 grayscale)
            source_palette: Source palette name
            target_palette: Target palette name
            
        Returns:
            Color-converted template
        """
        # Convert template to target palette
        colored_template = self.convert_to_gbc_palette(template, target_palette)
        
        # Convert back to grayscale for template matching
        if len(colored_template.shape) == 3:
            gray_template = cv2.cvtColor(colored_template, cv2.COLOR_RGB2GRAY)
        else:
            gray_template = colored_template
        
        return gray_template
    
    def enhance_template_for_lighting(self, template: np.ndarray, 
                                    lighting_condition: str = 'normal') -> np.ndarray:
        """
        Enhance a template for specific lighting conditions
        
        Args:
            template: Original template
            lighting_condition: 'normal', 'bright', 'dark', 'night'
            
        Returns:
            Enhanced template
        """
        enhanced = template.copy()
        
        if lighting_condition == 'bright':
            # Increase contrast for bright conditions
            enhanced = cv2.convertScaleAbs(enhanced, alpha=1.2, beta=10)
        elif lighting_condition == 'dark':
            # Decrease contrast and brighten for dark conditions
            enhanced = cv2.convertScaleAbs(enhanced, alpha=0.8, beta=20)
        elif lighting_condition == 'night':
            # Apply blue tint simulation for night mode
            enhanced = cv2.convertScaleAbs(enhanced, alpha=0.9, beta=15)
        
        return enhanced
    
    def get_palette_for_screen_type(self, screen_type: str) -> str:
        """
        Get the most appropriate palette for a screen type
        
        Args:
            screen_type: Type of screen ('dialogue', 'battle', 'menu', 'overworld')
            
        Returns:
            Recommended palette name
        """
        palette_mapping = {
            'dialogue': 'dialogue_normal',
            'battle': 'menu_battle', 
            'menu': 'menu_standard',
            'overworld': 'text_white',
            'intro': 'pokemon_crystal'
        }
        
        return palette_mapping.get(screen_type, 'text_white')
    
    def analyze_text_region_colors(self, region: np.ndarray) -> Dict:
        """
        Analyze the color characteristics of a text region
        
        Args:
            region: Text region to analyze
            
        Returns:
            Dictionary with color analysis results
        """
        if len(region.shape) == 3:
            gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
        else:
            gray = region
        
        # Basic statistics
        mean_brightness = np.mean(gray)
        std_brightness = np.std(gray)
        min_brightness = np.min(gray)
        max_brightness = np.max(gray)
        
        # Detect if text is light-on-dark or dark-on-light
        if mean_brightness < 128:
            text_style = 'light_on_dark'
        else:
            text_style = 'dark_on_light'
        
        # Estimate contrast
        contrast = max_brightness - min_brightness
        
        # Detect potential palette
        detected_palette = self.detect_palette_from_image(region)
        
        return {
            'mean_brightness': mean_brightness,
            'std_brightness': std_brightness,
            'min_brightness': min_brightness,
            'max_brightness': max_brightness,
            'contrast': contrast,
            'text_style': text_style,
            'detected_palette': detected_palette,
            'is_high_contrast': contrast > 100,
            'is_uniform': std_brightness < 30
        }
    
    def preview_palette(self, palette_name: str) -> None:
        """
        Print a visual preview of a palette
        
        Args:
            palette_name: Name of palette to preview
        """
        palette = self.get_palette(palette_name)
        if palette is None:
            print(f"❌ Palette '{palette_name}' not found")
            return
        
        print(f"\n🎨 Palette: {palette.name}")
        print(f"📝 {palette.description}")
        print("Colors:")
        
        for i, (r, g, b) in enumerate(palette.colors):
            print(f"  {i}: RGB({r:3d}, {g:3d}, {b:3d}) {'█' * 8}")


def test_gameboy_color_palette():
    """Test the Game Boy Color palette system"""
    print("🧪 Testing Game Boy Color Palette System...")
    
    # Initialize palette system
    gbc_palette = GameBoyColorPalette()
    
    # List available palettes
    palettes = gbc_palette.list_palettes()
    print(f"📋 Available palettes: {len(palettes)}")
    
    # Preview a few palettes
    for palette_name in ['text_white', 'health_green', 'pokemon_crystal']:
        gbc_palette.preview_palette(palette_name)
    
    # Test palette conversion
    test_image = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
    converted = gbc_palette.convert_to_gbc_palette(test_image, 'text_white')
    print(f"\n🖼️ Converted image shape: {converted.shape}")
    
    # Test palette detection
    detected = gbc_palette.detect_palette_from_image(converted)
    print(f"🔍 Detected palette: {detected}")
    
    # Test color analysis
    analysis = gbc_palette.analyze_text_region_colors(test_image)
    print(f"\n📊 Color analysis:")
    print(f"   Mean brightness: {analysis['mean_brightness']:.1f}")
    print(f"   Text style: {analysis['text_style']}")
    print(f"   Detected palette: {analysis['detected_palette']}")
    print(f"   High contrast: {analysis['is_high_contrast']}")
    
    print("\n✅ Game Boy Color palette test completed!")


if __name__ == "__main__":
    test_gameboy_color_palette()
