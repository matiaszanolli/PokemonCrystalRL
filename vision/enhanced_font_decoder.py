#!/usr/bin/env python3
"""
Enhanced Font Decoder - Real-time text detection using neural network models
"""

import numpy as np
import torch
import torch.nn as nn
import cv2
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from collections import deque
import logging
import json
import os

logger = logging.getLogger(__name__)

# Font Detection Model
class FontDetectionNet(nn.Module):
    """Neural network for font detection"""
    def __init__(self):
        super().__init__()
        # Convolutional layers for feature extraction
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        # Linear layers for classification
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),  # Binary classification
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(-1, 64 * 4 * 4)
        x = self.fc_layers(x)
        return x

@dataclass
class DetectedText:
    """Represents detected text in the game screen"""
    text: str
    confidence: float
    bbox: tuple  # (x1, y1, x2, y2)
    location: str  # menu, dialogue, ui, world

class EnhancedFontDecoder:
    """Enhanced font detection using deep learning and traditional methods"""
    
    def __init__(self, model_path: Optional[str] = None):
        """Initialize the font decoder"""
        self.model = FontDetectionNet()
        
        # Load pre-trained model if available
        if model_path and os.path.exists(model_path):
            try:
                self.model.load_state_dict(torch.load(model_path))
                self.model.eval()
                logger.info(f"Loaded font detection model from {model_path}")
            except Exception as e:
                logger.error(f"Error loading model: {e}")
        
        # Initialize text cache
        self.text_cache = {}
        self.recent_detections = deque(maxlen=100)
        
        # Configure text processing
        self.min_text_height = 8
        self.max_text_height = 32
        self.text_confidence_threshold = 0.7
        
        # Load character templates
        self._load_character_templates()
    
    def _load_character_templates(self):
        """Load Pokemon font character templates"""
        try:
            template_path = os.path.join(os.path.dirname(__file__), "font_templates.json")
            if os.path.exists(template_path):
                with open(template_path, 'r') as f:
                    self.char_templates = json.load(f)
                logger.info(f"Loaded {len(self.char_templates)} character templates")
            else:
                logger.warning("No font templates found")
                self.char_templates = {}
        except Exception as e:
            logger.error(f"Error loading templates: {e}")
            self.char_templates = {}
    
    def detect_text(self, image: np.ndarray) -> List[DetectedText]:
        """Detect text in a game screenshot"""
        if image is None or image.size == 0:
            return []
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        detected_texts = []
        
        # Try cache first
        image_hash = self._hash_image(gray)
        if image_hash in self.text_cache:
            cached = self.text_cache[image_hash]
            if cached['timestamp'] > time.time() - 300:  # 5 minute cache
                return cached['texts']
        
        # Process different regions of the screen
        regions = [
            ("dialogue", (0, int(0.7*gray.shape[0]), gray.shape[1], gray.shape[0])),
            ("menu", (int(0.7*gray.shape[1]), 0, gray.shape[1], gray.shape[0])),
            ("ui", (0, 0, int(0.7*gray.shape[1]), int(0.3*gray.shape[0]))),
        ]
        
        for region_name, (x1, y1, x2, y2) in regions:
            region = gray[y1:y2, x1:x2]
            if region.size == 0:
                continue
            
            # Find potential text regions
            text_regions = self._find_text_regions(region)
            
            for text_bbox in text_regions:
                # Extract text region
                rx1, ry1, rx2, ry2 = text_bbox
                text_region = region[ry1:ry2, rx1:rx2]
                
                # Process through neural network
                confidence = self._process_text_region(text_region)
                
                if confidence > self.text_confidence_threshold:
                    # Extract characters using template matching
                    text = self._extract_text_from_region(text_region)
                    
                    if text:
                        # Convert region coordinates to full image coordinates
                        bbox = (x1 + rx1, y1 + ry1, x1 + rx2, y1 + ry2)
                        
                        detected = DetectedText(
                            text=text,
                            confidence=confidence,
                            bbox=bbox,
                            location=region_name
                        )
                        detected_texts.append(detected)
        
        # Update cache
        self.text_cache[image_hash] = {
            'texts': detected_texts,
            'timestamp': time.time()
        }
        
        # Cleanup old cache entries
        self._cleanup_cache()
        
        return detected_texts
    
    def _find_text_regions(self, image: np.ndarray) -> List[tuple]:
        """Find potential text regions using image processing"""
        regions = []
        
        try:
            # Apply adaptive thresholding
            binary = cv2.adaptiveThreshold(
                image, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 11, 2
            )
            
            # Find contours
            contours, _ = cv2.findContours(
                binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            
            # Filter contours by size and aspect ratio
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Apply size constraints
                if h >= self.min_text_height and h <= self.max_text_height:
                    aspect_ratio = w / float(h)
                    if 0.5 <= aspect_ratio <= 20:  # Allow for text of varying lengths
                        regions.append((x, y, x + w, y + h))
            
        except Exception as e:
            logger.error(f"Error finding text regions: {e}")
        
        return regions
    
    def _process_text_region(self, region: np.ndarray) -> float:
        """Process text region through neural network"""
        try:
            # Preprocess region
            processed = cv2.resize(region, (16, 16))
            processed = processed.astype(np.float32) / 255.0
            processed = torch.from_numpy(processed).unsqueeze(0).unsqueeze(0)
            
            # Get prediction
            with torch.no_grad():
                confidence = self.model(processed).item()
            
            return confidence
            
        except Exception as e:
            logger.error(f"Error processing text region: {e}")
            return 0.0
    
    def _extract_text_from_region(self, region: np.ndarray) -> str:
        """Extract text from region using template matching"""
        if not self.char_templates:
            return ""
        
        try:
            # Normalize region
            region = cv2.normalize(region, None, 0, 255, cv2.NORM_MINMAX)
            
            extracted_text = ""
            width = region.shape[1]
            x_pos = 0
            
            while x_pos < width:
                best_match = (None, -1, 0)  # (char, score, width)
                
                for char, template in self.char_templates.items():
                    # Convert template to numpy array
                    tmpl = np.array(template, dtype=np.uint8)
                    
                    # Skip if remaining width is too small
                    if x_pos + tmpl.shape[1] > width:
                        continue
                    
                    # Match template
                    result = cv2.matchTemplate(
                        region[:, x_pos:x_pos + tmpl.shape[1]],
                        tmpl,
                        cv2.TM_CCOEFF_NORMED
                    )
                    
                    score = np.max(result)
                    if score > best_match[1]:
                        best_match = (char, score, tmpl.shape[1])
                
                if best_match[1] > 0.8:  # Confidence threshold
                    extracted_text += best_match[0]
                    x_pos += best_match[2]
                else:
                    x_pos += 1  # Skip one pixel if no match
            
            return extracted_text.strip()
            
        except Exception as e:
            logger.error(f"Error extracting text: {e}")
            return ""
    
    def _hash_image(self, image: np.ndarray) -> int:
        """Create a hash for image caching"""
        try:
            # Downsample for faster hashing
            small = cv2.resize(image, (32, 32))
            return hash(small.tobytes())
        except Exception:
            return 0
    
    def _cleanup_cache(self):
        """Remove old cache entries"""
        current_time = time.time()
        to_remove = []
        
        for key, value in self.text_cache.items():
            if current_time - value['timestamp'] > 300:  # 5 minutes
                to_remove.append(key)
        
        for key in to_remove:
            del self.text_cache[key]
