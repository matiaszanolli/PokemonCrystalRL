"""Tests for the interface refactoring to verify dependency structure."""
import pytest

def test_import_interfaces():
    """Test that interfaces can be imported without circular dependencies."""
    from interfaces import (
        MonitoringComponent,
        MonitoringStats,
        WebMonitorInterface,
        TrainerInterface,
        VisionProcessorInterface
    )
    assert MonitoringComponent is not None
    assert MonitoringStats is not None
    assert WebMonitorInterface is not None
    assert TrainerInterface is not None
    assert VisionProcessorInterface is not None

def test_monitoring_imports():
    """Test that monitoring components can import interfaces."""
    from core.web_monitor import ScreenCapture
    from interfaces.monitoring import ScreenCaptureComponent
    assert issubclass(ScreenCapture, ScreenCaptureComponent)

def test_trainer_imports():
    """Test that trainer components can import interfaces."""
    from interfaces.trainers import (
        TrainerInterface,
        GameState, 
        TrainerConfig
    )
    # Create a trainer config to verify the class works
    config = TrainerConfig(
        rom_path="/test/rom.gb",
        save_path="/test/save.state"
    )
    assert config.rom_path == "/test/rom.gb"
    assert config.save_path == "/test/save.state"

    # Create a game state to verify the class works 
    state = GameState(
        in_battle=True,
        map_id=1,
        player_x=10,
        player_y=10
    )
    assert state.in_battle is True
    assert state.map_id == 1
    assert state.player_x == 10
    assert state.player_y == 10

def test_vision_imports():
    """Test that vision components can import interfaces."""
    from interfaces.vision import (
        BoundingBox,
        DetectedText,
        GameUIElement,
        VisualContext
    )
    # Create a bounding box to verify the class works
    bbox = BoundingBox(x=0, y=0, width=100, height=100)
    assert bbox.area == 10000
    assert bbox.center == (50, 50)

    # Create a detected text object
    text = DetectedText(text="test", confidence=0.9, bounding_box=bbox)
    assert text.text == "test"
    assert text.confidence == 0.9
    assert text.bounding_box.area == 10000

    # Create a UI element
    ui_element = GameUIElement(
        element_type="button",
        bounding_box=bbox,
        confidence=0.9,
        text="Click me"
    )
    assert ui_element.element_type == "button"
    assert ui_element.text == "Click me"
    assert ui_element.confidence == 0.9

    # Create a visual context
    import numpy as np
    context = VisualContext(
        screen=np.zeros((100, 100, 3)),
        ui_elements=[ui_element],
        detected_text=[text],
        screen_state="menu",
        menu_visible=True
    )
    assert context.screen_state == "menu"
    assert context.menu_visible is True
    assert len(context.ui_elements) == 1
    assert len(context.detected_text) == 1
