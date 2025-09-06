"""Base interfaces for vision components."""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import numpy as np


@dataclass
class BoundingBox:
    """Bounding box for detected UI elements."""
    x: int
    y: int
    width: int
    height: int
    confidence: float = 1.0

    @property
    def area(self) -> int:
        """Get area of bounding box."""
        return self.width * self.height

    @property
    def center(self) -> Tuple[int, int]:
        """Get center point of bounding box."""
        return (self.x + self.width // 2, self.y + self.height // 2)


@dataclass
class DetectedText:
    """Detected text with metadata."""
    text: str
    confidence: float
    bounding_box: Optional[BoundingBox] = None
    

@dataclass
class GameUIElement:
    """Detected UI element in game screen."""
    element_type: str
    bounding_box: BoundingBox
    confidence: float
    text: Optional[str] = None
    properties: Dict[str, Any] = None

    def __post_init__(self):
        if self.properties is None:
            self.properties = {}


@dataclass
class VisualContext:
    """Complete visual context from current screen."""
    screen: np.ndarray
    ui_elements: List[GameUIElement]
    detected_text: List[DetectedText]
    screen_state: str = "unknown"
    dialogue_visible: bool = False
    battle_visible: bool = False
    menu_visible: bool = False


class FontDecoderInterface(ABC):
    """Interface for Pokemon font decoding."""
    
    @abstractmethod
    def decode_text(self, image: np.ndarray, min_confidence: float = 0.0) -> List[DetectedText]:
        """Decode text from image.
        
        Args:
            image: Image to decode text from
            min_confidence: Minimum confidence threshold
            
        Returns:
            List of detected text regions
        """
        pass
        
    @abstractmethod
    def load_font_templates(self, template_path: str) -> bool:
        """Load font templates from file.
        
        Args:
            template_path: Path to font templates
            
        Returns:
            True if templates loaded successfully
        """
        pass


class ScreenProcessorInterface(ABC):
    """Interface for game screen processing."""
    
    @abstractmethod
    def process_screen(self, screen: np.ndarray) -> VisualContext:
        """Process raw screen image.
        
        Args:
            screen: Raw screen image
            
        Returns:
            Visual context from screen
        """
        pass
        
    @abstractmethod
    def detect_ui_elements(self, screen: np.ndarray) -> List[GameUIElement]:
        """Detect UI elements in screen.
        
        Args:
            screen: Screen image to process
            
        Returns:
            List of detected UI elements
        """
        pass
        
    @abstractmethod
    def detect_text(self, screen: np.ndarray) -> List[DetectedText]:
        """Detect text in screen.
        
        Args:
            screen: Screen image to process
            
        Returns:
            List of detected text regions
        """
        pass
        
    @abstractmethod
    def classify_screen_state(self, screen: np.ndarray) -> str:
        """Classify type of screen being shown.
        
        Args:
            screen: Screen image to classify
            
        Returns:
            Screen state classification
        """
        pass


class VisionProcessorInterface(ABC):
    """Main interface for vision processing system."""
    
    @abstractmethod
    def process_frame(self, frame: np.ndarray) -> VisualContext:
        """Process a single game frame.
        
        Args:
            frame: Raw game frame to process
            
        Returns:
            Complete visual context from frame
        """
        pass
        
    @abstractmethod
    def get_current_context(self) -> VisualContext:
        """Get current visual context.
        
        Returns:
            Current visual context
        """
        pass
        
    @abstractmethod
    def reset(self) -> None:
        """Reset processor state."""
        pass
