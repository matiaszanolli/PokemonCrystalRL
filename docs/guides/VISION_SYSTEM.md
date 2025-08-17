# 👁️ Vision System Documentation

The Pokemon Crystal RL vision system provides advanced screen analysis and text recognition capabilities for the training platform.

## 🌟 Key Features

- ROM-based font decoding for accurate text recognition
- UI element detection (menus, battles, dialogues)
- Game state classification
- Visual context analysis
- Performance-optimized processing

## 📋 Components

### 1. UnifiedVisionProcessor

The main vision processing class that coordinates all visual analysis:

```python
from pokemon_crystal_rl.vision import UnifiedVisionProcessor

processor = UnifiedVisionProcessor()
context = processor.process_screenshot(screen)
```

Features:
- Text recognition using ROM fonts
- UI element detection
- Game state classification
- Memory-efficient processing
- Performance optimization

### 2. ROM Font Decoder

Uses actual Pokemon Crystal font data for accurate text recognition:

```python
from pokemon_crystal_rl.vision import ROMFontDecoder

decoder = ROMFontDecoder(rom_path="pokemon_crystal.gbc")
text = decoder.decode_text_region(screen_region)
```

Features:
- ROM-based font templates
- High-accuracy matching
- Fast template matching
- Memory caching
- Multiple font variations

### 3. Game State Detection

Classifies the current game screen state:

```python
state = processor.classify_screen_type(screen)
# Returns: BATTLE, MENU, DIALOGUE, OVERWORLD, etc.
```

Detection capabilities:
- Battle screens
- Menu interfaces
- Dialogue boxes
- Overworld exploration
- Loading screens

### 4. UI Element Detection

Identifies UI elements in the game screen:

```python
elements = processor.detect_ui_elements(screen)
# Returns: health bars, menu boxes, dialogue boxes, etc.
```

Detectable elements:
- Health bars
- Menu boxes
- Dialogue boxes
- Battle UI elements
- Interface components

## 🔧 Usage

### Basic Usage

```python
from pokemon_crystal_rl.vision import UnifiedVisionProcessor

# Initialize
processor = UnifiedVisionProcessor(
    template_path="font_templates.npz",
    rom_path="pokemon_crystal.gbc"
)

# Process screenshot
context = processor.process_screenshot(screen)

# Access results
print(f"Screen type: {context.screen_type}")
print(f"Detected text: {context.detected_text}")
print(f"UI elements: {context.ui_elements}")
print(f"Visual summary: {context.visual_summary}")
```

### Advanced Features

```python
# Font decoder with custom templates
decoder = ROMFontDecoder()
decoder.add_custom_template('☺', custom_template)

# Get detailed statistics
stats = processor.get_stats()
print(f"Processing success rate: {stats['success_rate']:.2%}")
print(f"Average confidence: {stats['average_confidence']:.2f}")

# Cache management
processor.clear_caches()  # Clear processing caches
```

## 📊 Performance

### Processing Time
- Basic screen analysis: ~5-10ms
- Text recognition: ~20-30ms per region
- UI detection: ~10-15ms
- Total processing: ~50ms per frame

### Memory Usage
- Font templates: ~500KB
- Processing cache: ~10MB
- Total memory: ~20MB average

### Optimization Tips
- Use caching for repeated regions
- Process only changed screen areas
- Adjust confidence thresholds
- Manage memory with cleanup()

## 🎯 Best Practices

1. **Memory Management**
   ```python
   # Clear caches periodically
   if frames % 1000 == 0:
       processor.clear_caches()
   ```

2. **Error Handling**
   ```python
   try:
       context = processor.process_screenshot(screen)
   except Exception as e:
       logger.error(f"Vision processing error: {e}")
       context = processor.create_empty_context()
   ```

3. **Performance Optimization**
   ```python
   # Process only when needed
   if screen_changed:
       context = processor.process_screenshot(screen)
   ```

4. **Quality Control**
   ```python
   # Check confidence scores
   if context.confidence < 0.8:
       logger.warning("Low confidence detection")
   ```

## 🔍 Troubleshooting

Common issues and solutions:

1. **Low Text Recognition Accuracy**
   - Check ROM font templates
   - Verify screen resolution
   - Adjust confidence threshold

2. **Slow Processing**
   - Enable caching
   - Reduce processing frequency
   - Check screen update detection

3. **Memory Usage**
   - Monitor cache size
   - Clear caches regularly
   - Adjust cache limits

4. **False Detections**
   - Tune detection thresholds
   - Verify screen preprocessing
   - Check for screen artifacts

## 🚀 Future Improvements

Planned enhancements:

1. **Enhanced Recognition**
   - Multi-font support
   - Dynamic template adaptation
   - Context-aware processing

2. **Performance**
   - GPU acceleration
   - Parallel processing
   - Smarter caching

3. **Features**
   - Animated sprite detection
   - Battle scene analysis
   - Item recognition

4. **Integration**
   - LLM vision processing
   - Real-time strategy feedback
   - Advanced scene understanding
